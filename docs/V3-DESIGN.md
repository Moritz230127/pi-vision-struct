# pi-vision-struct V3.0 设计文档（PLAN）

> 状态: DRAFT — 待用户确认后实施
> 日期: 2026-09-01
> 版本: v3.0.0
> 约束: 主模型仅 DeepSeek-V4-Flash-0731（纯文本）；零第三方 API；轻量 DL 仅作"视觉传感器"（≤25M 参数，本地 CPU/8GB VRAM）；全部输出确定性可复算。

---

## 0. 设计原则（系统工程视角）

1. **整体最优 > 局部最优**：每个传感器服务于融合引擎的"证据质量"，而非独立输出。
2. **证据链完整**：每个结论携带 `evidence[]`（来源传感器 + 原始数值 + 置信度），DeepSeek 可审计、可追问。
3. **确定性优先**：融合层全部纯算法（D-S 证据理论 / 匈牙利匹配 / 不确定性量化），DL 只做感知（mask/bbox/深度），不进数值主链。
4. **性能即特性**：常驻守护进程 + numpy 向量化 + 惰性加载，目标单图全流水线 < 5s（CPU）。
5. **可商业化**：schema v3 契约化、传感器注册表、任务编排声明式、错误分级、健康检查。

---

## 1. 架构总览

```text
┌─────────────────────────────────────────────────────────────────┐
│                    L0 源层 (Source Layer)                        │
│  dom(Playwright) · pptx · pdf · a11y · capture · 文件输入        │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              L1 确定性测量层 (Deterministic Layer)               │
│  ocr(双后端+后处理) · pix(超像素量化) · scene_stats(真Otsu)      │
│  edge(亚像素) · geometry(轮廓/SVG) · ascii(栅格) · diff           │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              L2 轻量感知层 (Lightweight DL, ≤25M)                 │
│  saliency(U²-Net) · segment(MobileSAM) · depth(MiDaS-small)     │
│  detect(OWLv2 保留) · omniparser(保留, UI 专用)                  │
│  → 全部输出 mask/bbox/深度图 → 转成数值证据                       │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              F1 融合引擎 (Fusion Engine, 纯算法)                  │
│  align(匈牙利/OT 匹配) → evidence(D-S 组合) →                      │
│  uncertainty(置信度传播) → findings(带 belief/plausibility)       │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              F2 推理接口 (Reasoning Interface)                    │
│  schema v3 报告 · 多轮反馈协议(粗看→选区→细看) · notation 原语    │
└─────────────────────────────────────────────────────────────────┘
```

**关键变化 vs v2.x**：
- 单端口 22 动作 → **分层流水线 + 传感器注册表**（动作仍保留，但内部走统一管线）
- 硬阈值融合 → **D-S 证据融合 + 不确定性输出**
- 朴素最大 IoU 匹配 → **匈牙利/最优传输全局匹配**
- 新增 4 个传感器：saliency / segment / depth / ascii / edge
- 移除 qwen3-vl:8b 依赖（L2 语义层删除，改由融合引擎 + DeepSeek 推理承担）
- schema v2 → **v3**（证据链 + 不确定性 + 传感器元数据）

---

## 2. 融合引擎设计（核心创新）

### 2.1 证据模型（Evidence Model）

每个传感器输出转成**证据质量函数 (mass function)**，定义在辨识框架 Θ = {一致, 冲突, 不确定}：

```json
{
  "sensor": "ocr",
  "claim": "text_present",
  "bbox": [x1,y1,x2,y2],
  "mass": {"consistent": 0.72, "conflict": 0.08, "uncertain": 0.20},
  "raw": {"conf": 0.9, "text": "..."}
}
```

mass 转换规则（确定性映射，无学习）：
- OCR 文本存在性：`mass[consistent] = conf × 0.8`，`mass[uncertain] = 1 - mass[consistent]`
- DOM 声明色 vs 像素实测：ΔE → `mass[consistent] = exp(-ΔE/τ)`（τ=5.0 可配）
- 检测器：`mass[consistent] = conf`（OWLv2/MobileSAM 自带置信度）

### 2.2 D-S 组合规则（Dempster's Rule）

多源证据用正交和组合：

```text
m₁₂(A) = Σ_{B∩C=A} m₁(B)·m₂(C) / (1 - K)
K = Σ_{B∩C=∅} m₁(B)·m₂(C)   (冲突系数)
```

输出 `belief(A) = Σ_{B⊆A} m(B)` 和 `plausibility(A) = 1 - belief(¬A)`。

**决策规则**：`belief(consistent) > 0.6` → 确认；`plausibility(consistent) < 0.4` → 冲突（anomaly）；中间 → 不确定（标记 `needs_review`，触发 zoom-in 协议）。

### 2.3 全局匹配（匈牙利 / 最优传输）

替代朴素 `max(IoU)`：
- 构建代价矩阵 `C[i][j] = 1 - IoU(box_i, box_j)`（+ 类别/文本相似度项）
- **匈牙利算法**（scipy.optimize.linear_sum_assignment）全局最优一对一匹配
- 可选 **UOT**（不平衡最优传输）处理"多对一"（一个 DOM 元素 = 多个 OCR 片段）

### 2.4 不确定性传播

每个 finding 输出 `uncertainty` 字段：
```json
{
  "finding": "text_missing",
  "belief": 0.72,
  "plausibility": 0.85,
  "uncertainty": 0.13,
  "evidence": ["ocr:conf=0.9", "dom:text='提交'", "iou_match=0.0"]
}
```
DeepSeek 据此决定：直接采信 / 请求 zoom-in 复查 / 标记存疑。

---

## 3. 传感器清单（新增/优化/保留）

### 3.1 新增传感器

| 传感器 | 模型/算法 | 参数 | 输出 | 破解痛点 |
|---|---|---|---|---|
| `saliency` | U²-Net（轻量版） | ~4M | 显著性图 → top-N 候选区域 bbox | **zoom-in 死锁**（复杂背景） |
| `segment` | MobileSAM（Tiny-ViT 5M） | 5M, CPU 3s | 前景/背景 mask → 前景 bbox + 面积比 | **复杂背景分离** |
| `depth` | MiDaS-small（21M, ONNX 80MB） | 256×256 | 逆深度图 → 相对深度统计 | 精密零件空间关系 |
| `edge` | Devernay 亚像素边缘 | 纯算法 | 亚像素边缘点 → 线段/轮廓 | 精密零件边缘测量 |
| `ascii` | 多分辨率 ASCII 栅格 | 纯算法 | 64×36 粗 + 128×72 细文本栅格 | 复杂构建粗定位 |
| `geometry` | VTracer/轮廓 → 文本原语 | 纯算法 | 形状原语（圆/矩形/线段/多边形） | 工程图纸/示意图 |

### 3.2 优化传感器

| 传感器 | v2 现状 | v3 优化 | 依据 |
|---|---|---|---|
| `pix` | MEDIANCUT 量化 + 纯 Python BFS | **SLIC 超像素 + K-means**；numpy 向量化连通域 | PAMI 2011 SLIC；Numba 10-100× |
| `scene_stats` | median 当 Otsu 阈值 | **真 Otsu 多阈值** + 超像素前景分离 | 经典 Otsu |
| `ocr` | 双后端无后处理 | **置信度加权 + 词典纠错 + 多假设** | ConfBERT 思路（轻量版） |
| `crosscheck` | 硬阈值 + 朴素匹配 | **D-S 融合 + 匈牙利匹配**（见 §2） | D-S 证据理论 |
| `audit3d` | OBB-SAT + cKDTree（精度已最优） | **BVH 粗筛 + 并行化**（性能） | BVH vs KDTree |
| `detect` | OWLv2（保留） | 保留 + 与 segment 联动（检测→分割） | — |

### 3.3 移除

| 组件 | 原因 |
|---|---|
| `vs_semantic.py`（qwen3-vl:8b L2 语义） | 违反"零其他模型"约束；语义推理由 DeepSeek 承担 |
| `vs_critic.py`（VLM 复核） | 同上；改由 D-S 融合的 belief/plausibility 承担"复核"角色 |
| `vs_vlm.py` 网关 | 同上 |

---

## 4. 性能设计

### 4.1 常驻守护进程（统一 daemon 管理器）

现有 ocrserver/omniserver 模式 → 统一 `vsd`（vision daemon）：
- 单进程管理所有 DL 传感器（saliency/segment/depth/detect），模型驻留内存
- unix socket 行分隔 JSON（现有模式），冷载一次，后续 <100ms
- 健康检查 + 自动拉起 + 优雅降级（daemon 不可用 → 纯算法路径）

### 4.2 计算优化

| 热点 | 优化 | 预期 |
|---|---|---|
| 连通域（vs_pix diff） | numpy 向量化 / Numba JIT | 10-100× |
| 颜色量化 | SLIC（skimage 已向量化） | 5-10× |
| 匹配（crosscheck） | 匈牙利（scipy）替代 O(n²) 朴素 | 匹配质量↑ |
| 3D 审计 | BVH 粗筛 + multiprocessing | 2-5× |
| 全流水线 | 惰性加载 + 结果缓存（LRU by image hash） | 重复分析 <1s |

### 4.3 资源预算（8GB VRAM / 31GB RAM）

| 组件 | VRAM | RAM | 说明 |
|---|---|---|---|
| MobileSAM | ~1GB（或纯 CPU） | 2GB | 默认 CPU（3s），GPU 可选 |
| U²-Net | ~0.5GB | 1GB | CPU 可跑 |
| MiDaS-small | ~0.3GB | 1GB | ONNX CPU |
| OWLv2 | ~2GB | 3GB | 保留，omniparser env |
| OmniParser | ~2GB | 3GB | 保留，UI 专用 |
| **合计** | **≤5.8GB** | **≤10GB** | 8GB VRAM 富余 |

---

## 5. 数据流与协议

### 5.1 schema v3 报告契约

```json
{
  "schema": "vision-report/v3",
  "task": "diagnose",
  "sensors": ["pix", "ocr", "saliency", "segment", "fuse"],
  "coordsys": "image_px",
  "source": {"type": "image", "path": "...", "size_px": [w, h]},
  "elements": [...],           // 带 conf + source[]
  "findings": [                // 融合结论
    {
      "type": "text_missing",
      "bbox": [...],
      "belief": 0.72, "plausibility": 0.85, "uncertainty": 0.13,
      "evidence": [{"sensor": "ocr", "raw": {...}}],
      "verdict": "confirmed|conflict|needs_review"
    }
  ],
  "candidates": [...],         // saliency 候选区域（zoom-in 入口）
  "foreground": {...},          // segment 前景信息
  "metrics": {...},
  "notation": "..."            // 原语标注指南
}
```

### 5.2 多轮反馈协议（SeeingEye 模式）

```text
DeepSeek ──vs analyze──→ 粗报告（saliency 候选 + ascii 粗栅格 + 全局统计）
DeepSeek 选候选区 ──vs zoom [region]──→ 细报告（ocr 高倍 + segment 前景 + edge 亚像素）
DeepSeek 追问 ──vs probe [bbox, sensor]──→ 单传感器定向证据
DeepSeek 综合推理 → 最终答案
```

协议由 DeepSeek 的 agentic 工具调用驱动（DeepSeek-V4-Flash-0731 官方强化 agentic 能力），扩展提供 `analyze/zoom/probe` 三个动作即可。

---

## 6. 迁移路径（v2.2.2 → v3.0.0）

| 阶段 | 内容 | 验证 |
|---|---|---|
| M1 | 融合引擎重构（D-S + 匈牙利 + 不确定性） | 64 断言回归 + 误报率对比 |
| M2 | 新增 saliency/segment/ascii/edge 传感器 | 各传感器基准卡 |
| M3 | 性能优化（向量化 + daemon 统一） | 全流水线 <5s 基准 |
| M4 | schema v3 + 多轮反馈协议 | 端到端真实图片测试 |
| M5 | 移除 L2 语义层 + 文档/知识库同步 | 回归全绿 + kb-audit |

**兼容策略**：v3 保留 v2 的 action 枚举（`vs` 单端口不变），内部走新管线；schema v2 报告自动升级 v3（加 evidence/uncertainty 字段）。旧 task 配置兼容。

---

## 7. 验证标准

1. **回归**：现有 64 断言测试全绿（升级后）
2. **误报率**：crosscheck 误报率 vs v2 下降 ≥30%（D-S 融合）
3. **匹配准确率**：匈牙利匹配 vs 朴素 IoU，F1 提升 ≥10%
4. **性能**：全流水线（pix+ocr+saliency+segment+fuse）CPU <5s
5. **复杂背景**：真实复杂背景图，saliency 候选区命中有效内容 ≥80%
6. **精密零件**：edge 亚像素测量误差 <0.5px；segment 前景 IoU ≥0.8
7. **合规**：零第三方 API 调用；零 qwen3-vl 依赖；全部输出可复算

---

## 8. 风险与回滚

| 风险 | 缓解 | 回滚 |
|---|---|---|
| MobileSAM 分割质量不足 | 保留 OWLv2 检测联动；阈值可调 | 关闭 segment 传感器 |
| D-S 组合对冲突证据敏感 | K 值上限保护；冲突→needs_review 而非误报 | 回退硬阈值模式（配置开关） |
| 性能不达标 | daemon 驻留 + 缓存 + 可选 GPU | 降级纯算法路径 |
| 新依赖（skimage/onnxruntime） | 独立 conda env（vsensor），不污染主 env | 删除 env |

**回滚总策略**：v3 与 v2 并行部署（`vs` 动作兼容），git tag v2.2.2 保留，一键回退。

---

## 9. 里程碑

- **M1**（融合引擎）: 2 天
- **M2**（新传感器）: 3 天
- **M3**（性能）: 2 天
- **M4**（schema v3 + 协议）: 2 天
- **M5**（清理 + 文档）: 1 天
- **合计**: ~10 天（含测试）

---

## 10. 待确认决策

1. **MobileSAM 默认 CPU 还是 GPU**？（CPU 3s 稳定，GPU 更快但占 VRAM）
2. **U²-Net 显著性 vs 纯算法显著性**（边缘密度+颜色聚类）？DL 版更准，纯算法零依赖
3. **schema v3 是否完全替换 v2**，还是双 schema 并存一个版本？
4. **daemon 统一管理器**是否值得（vs 现有 ocrserver/omniserver 已可用）？
5. **VTracer（SVG 化）**是否引入（新增 Rust 二进制依赖）？
