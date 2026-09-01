# pi-vision-struct V3.0 分步执行计划（FINAL · Gate-Based）

> 状态: APPROVED — 用户已确认全部决策（D1-D7）
> 日期: 2026-09-01
> 版本: v3.0.0
> 约束: 主模型仅 DeepSeek-V4-Flash-0731（纯文本）；零第三方 API；轻量 DL 仅作视觉传感器（≤25M）；全部输出确定性可复算；GPU 全面授权 + 功耗感知。

---

## 0. 执行总则（Gate-Based Workflow）

```
每步 = 设计 → 实现 → 验证 → 门禁(Gate)
门禁通过（全部印证项绿）→ 进入下一步
门禁失败 → 修复 → 重验 → 通过后才继续
全部步骤完成 → 系统整合 + 全量回归 → 发布 v3.0.0
```

**每步固定交付物**：
1. 设计方案（本 PLAN 已含，实施时细化）
2. 代码实现
3. 印证方案执行结果（测试/基准/压测）
4. 门禁结论（PASS/FAIL）

**门禁规则**：任何一步的印证项有 FAIL → 该步不通过，禁止进入下一步。修复后重跑全部印证项。

---

## 1. 阶段总览

| 阶段 | 内容 | 门禁 |
|---|---|---|
| S0 | 环境基线（vsensor env + GPU + 权重） | GPU 可用 + 三 env 隔离 |
| S1 | 融合引擎（D-S + 匈牙利 + 不确定性） | 单元测试 + 误报率对比 |
| S2 | 新传感器（saliency/segment/depth/edge/ascii/geometry） | 各传感器基准卡 |
| S3 | 统一 daemon + 调度（vsd + vsched + 功耗感知） | 并发压测 + 功耗切换 |
| S4 | 优化重构（pix/scene_stats/ocr/audit3d + GPU 加速） | 性能基准 + 精度对比 |
| S5 | schema v3 + 多轮协议（analyze/zoom/probe） | 端到端真实图测试 |
| S6 | 系统整合 + 全量回归 + 文档 | 64 断言全绿 + kb-audit |

---

## 2. S0 环境基线

### 2.1 设计方案
新建 `vsensor` conda env（GPU torch + skimage + onnxruntime），与 pi-vision（确定性）、omniparser（CPU DL）三 env 隔离。下载 U²-Net / MobileSAM / MiDaS-small 权重到本地缓存。

### 2.2 架构逻辑
```
vsensor env (GPU torch cu124)  ← L2 轻量 DL 传感器
pi-vision env (确定性算法)      ← L1 测量 + F1 融合
omniparser env (CPU torch)     ← detect/omniparser（保留）
三 env 互不 import，通过子进程 + JSON 通信
```

### 2.3 技术实现
- [ ] `conda create -n vsensor python=3.12`
- [ ] `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`
- [ ] `pip install scikit-image onnxruntime pillow numpy`
- [ ] 下载权重：U²-Net（u2netp.pth ~4.7MB）、MobileSAM（mobile_sam.pt ~40MB）、MiDaS-small（midas_v21_small_256.onnx ~80MB）→ `~/.cache/vsensor/`
- [ ] 安装 vtracer：`pip install vtracer`（或 cargo 版）
- [ ] 验证：`torch.cuda.is_available() == True`，`torch.cuda.get_device_name()`

### 2.4 工程收敛边际
- vsensor env 创建成功，GPU torch 可用
- 三 env 互不污染（vsensor 不装 paddle/playwright；pi-vision 不装 torch）
- 权重全部下载到本地缓存（离线可用）
- vtracer 可执行

### 2.5 印证方案
| # | 印证项 | 通过标准 |
|---|---|---|
| S0-1 | `torch.cuda.is_available()` | True |
| S0-2 | GPU 名称 | "NVIDIA GeForce RTX 4060 Laptop GPU" |
| S0-3 | 三 env 隔离 | vsensor 无 paddle；pi-vision 无 torch |
| S0-4 | 权重文件存在 | 3 个文件 + 大小正确 |
| S0-5 | vtracer 可运行 | `vtracer --help` 退出码 0 |

**门禁 S0**：S0-1~S0-5 全 PASS → 进入 S1。

---

## 3. S1 融合引擎（核心）

### 3.1 设计方案
`vs_fusion.py`：多传感器证据融合。输入 schema v3 元素（带 conf），输出 findings（belief/plausibility/uncertainty/verdict）。三阶段：align（匈牙利匹配）→ evidence（mass 映射）→ combine（D-S 组合）。

### 3.2 架构逻辑
```
输入: [{sensor, bbox, conf, text/color/...}, ...]
  ↓ align: 代价矩阵 C[i][j] = 1 - IoU - α·text_sim
  ↓        scipy.optimize.linear_sum_assignment → 全局匹配对
  ↓ evidence: 每对 → mass{consistent, conflict, uncertain}
  ↓ combine: D-S 正交和 m₁₂(A) = Σ_{B∩C=A} m₁(B)m₂(C)/(1-K)
  ↓ decide: belief>0.6→confirmed; plausibility<0.4→conflict; 中间→needs_review
输出: findings[] + evidence[] + uncertainty
```

### 3.3 技术实现
- [ ] `vs_fusion.py`：mass 映射表（§3.1.1 设计文档）+ D-S 组合 + 匈牙利匹配
- [ ] K 冲突系数上限保护（K>0.95 → needs_review）
- [ ] 不确定性传播：`uncertainty = 1 - (belief - plausibility)`
- [ ] 单元测试 `tests/test_fusion.py`：合成数据验证 D-S 组合数学正确性
- [ ] 与旧 vs_crosscheck 对比基准（同图误报率）

### 3.4 工程收敛边际
- vs_fusion.py 独立可运行（CLI + 库接口）
- 不依赖任何 DL 模型（纯算法）
- 旧 vs_crosscheck.py 暂保留（S4 删除），用于对比
- 输出 schema v3 findings 结构

### 3.5 印证方案
| # | 印证项 | 通过标准 |
|---|---|---|
| S1-1 | D-S 组合数学 | 已知 mass 输入 → 手算期望输出一致 |
| S1-2 | 匈牙利匹配 | 合成 10 对 bbox → 全局最优匹配（无重复） |
| S1-3 | 冲突处理 | K>0.95 → needs_review 而非误报 |
| S1-4 | 误报率对比 | 同 20 张图，v3 误报 < v2 的 70% |
| S1-5 | 确定性 | 同输入两次运行输出 diff=0 |

**门禁 S1**：S1-1~S1-5 全 PASS → 进入 S2。

---

## 4. S2 新传感器（6 个）

### 4.1 设计方案
6 个新传感器，全部输出 schema v3 元素/证据。DL 传感器（saliency/segment/depth）跑 vsensor env GPU；纯算法（edge/ascii/geometry）跑 pi-vision env。

### 4.2 架构逻辑
```
vs_saliency.py  U²-Net → 显著性图 → top-N 候选区域 bbox + score
vs_segment.py   MobileSAM → 前景 mask → bbox + 面积比 + 实例数
vs_depth.py     MiDaS-small → 逆深度图 → 相对深度统计（近/中/远分布）
vs_edge.py      Devernay 亚像素 → 边缘点 → 线段/轮廓（Hough 拟合）
vs_ascii.py     多分辨率栅格 → 64×36 粗 + 128×72 细文本
vs_geometry.py  VTracer → SVG → 形状原语（圆/矩形/线段/多边形）
```

### 4.3 技术实现
- [ ] `vs_saliency.py`：U²-Net 推理（GPU），`--top-n 5` 候选区
- [ ] `vs_segment.py`：MobileSAM 推理（GPU），`--point-prompt` 可选
- [ ] `vs_depth.py`：MiDaS ONNX 推理，输出深度分位数
- [ ] `vs_edge.py`：Devernay 亚像素 + Hough 线段拟合
- [ ] `vs_ascii.py`：亮度/颜色 → 字符映射（调色板 + 边框）
- [ ] `vs_geometry.py`：VTracer 光栅→SVG → 原语解析
- [ ] 各传感器基准卡：精度/延迟/VRAM

### 4.4 工程收敛边际
- 每传感器独立 CLI + 库接口，可单独调用
- DL 传感器默认走 vsd daemon（S3 前先直连）
- 输出全部 schema v3 元素结构
- 不引入 >25M 参数模型

### 4.5 印证方案
| # | 印证项 | 通过标准 |
|---|---|---|
| S2-1 | saliency 候选区 | 复杂背景图 top-5 命中有效内容 ≥80%（人工标注） |
| S2-2 | segment 前景 | 前景 mask IoU ≥0.8（标注对比） |
| S2-3 | depth 输出 | 合成深度图误差 <10%（相对） |
| S2-4 | edge 亚像素 | 合成图已知边缘误差 <0.5px |
| S2-5 | ascii 栅格 | 粗栅格可辨结构（人工目检） |
| S2-6 | geometry 原语 | 合成 SVG 图原语还原率 ≥90% |
| S2-7 | 延迟 | 每传感器 GPU <2s，CPU <5s |

**门禁 S2**：S2-1~S2-7 全 PASS → 进入 S3。

---

## 5. S3 统一 daemon + 调度（vsd + vsched）

### 5.1 设计方案
`vsd.py`：单进程管理 L2 全部 DL 模型（GPU 驻留），unix socket 行分隔 JSON。`vsched.py`：显存预算 + 并发信号量 + LRU 淘汰 + 优先级队列 + CPU 降级 + 功耗感知。

### 5.2 架构逻辑
```
请求 → vsched 队列（优先级 analyze>zoom>probe>bg）
     → 显存预算检查（总预算 6.5GB）
     → 模型加载/复用（LRU，冷载 1-3s / 热 <100ms）
     → 并发信号量（GPU 槽位=1 串行）
     → 功耗模式检查（battery→降频/CPU 降级；plugged→全速）
     → 执行 → 释放 → 下一任务
```

### 5.3 技术实现
- [ ] `vsd.py`：模型注册表 + unix socket 服务 + 健康检查 + 自动拉起
- [ ] `vsched.py`：显存预算表 + 信号量 + LRU + 优先级队列 + 超时保护
- [ ] 功耗感知：`/sys/class/power_supply/BAT1/status` 轮询（30s）+ 模式切换
- [ ] 功耗遥测：`nvidia-smi power.draw` → metrics
- [ ] 与 ocrserver/omniserver 统一协议（兼容旧 socket）
- [ ] 并发压测脚本 `tests/stress_sched.py`

### 5.4 工程收敛边际
- vsd/vsched 同进程（避免双进程争显存）
- 所有 L2 请求必须经 vsched（唯一入口）
- 显存峰值 ≤6.5GB（8GB 留 1.5GB 安全水位）
- 任务永不因 OOM 失败（CPU 降级兜底）

### 5.5 印证方案
| # | 印证项 | 通过标准 |
|---|---|---|
| S3-1 | 并发压测 | 5 任务并发，显存峰值 ≤6.5GB，零 OOM |
| S3-2 | 优先级 | analyze 任务不被 bg 饿死（延迟 <2× 串行） |
| S3-3 | LRU 淘汰 | 冷模型重载 <3s，热模型 <100ms |
| S3-4 | 功耗切换 | battery→plugged 自动切换无需重启 |
| S3-5 | CPU 降级 | 显存紧张时 MobileSAM 自动 CPU，任务完成 |
| S3-6 | 超时保护 | 卡死任务强制终止，daemon 存活 |

**门禁 S3**：S3-1~S3-6 全 PASS → 进入 S4。

---

## 6. S4 优化重构 + GPU 加速

### 6.1 设计方案
5 个现有工具优化 + GPU 加速清单（§3.9）。全部保持输出兼容（schema v3 元素结构不变，仅内部算法升级）。

### 6.2 架构逻辑
```
vs_pix.py        MEDIANCUT→SLIC+K-means（GPU torch）；BFS→torch 向量化
vs_scene_stats   median→真 Otsu 多阈值（GPU）
vs_ocr.py        置信度加权 + 词典纠错 + 多假设（CPU，后处理）
vs_audit3d.py    BVH 粗筛 + multiprocessing（CPU 并行）
vs_crosscheck.py → 删除（并入 vs_fusion.py）
GPU 加速: 量化/连通域/边缘/直方图/OCR 预处理/点云 → torch
```

### 6.3 技术实现
- [ ] `vs_pix.py`：skimage SLIC + torch K-means + torch 连通域
- [ ] `vs_scene_stats.py`：skimage threshold_otsu 多阈值
- [ ] `vs_ocr.py`：conf 加权投票 + 词典纠错（编辑距离）
- [ ] `vs_audit3d.py`：cKDTree BVH 粗筛 + multiprocessing.Pool
- [ ] 删除 `vs_crosscheck.py`（功能并入 vs_fusion）
- [ ] GPU 加速模块 `vs_gpu.py`（torch 工具函数库）

### 6.4 工程收敛边际
- 输出 schema 与 S1/S2 完全兼容
- 旧测试（64 断言）中涉及这些工具的用例仍通过（或更新后通过）
- GPU 加速仅用于大矩阵计算（小矩阵保持 CPU）
- 删除 vs_crosscheck 前确认 vs_fusion 覆盖其全部功能

### 6.5 印证方案
| # | 印证项 | 通过标准 |
|---|---|---|
| S4-1 | pix 精度 | SLIC 量化 vs MEDIANCUT，主色还原率 ≥95% |
| S4-2 | pix 性能 | 连通域提速 ≥10×（基准图计时） |
| S4-3 | scene_stats | Otsu 阈值 vs 旧 median，分割准确率 ↑ |
| S4-4 | ocr 后处理 | 词典纠错准确率 ↑（合成错字测试） |
| S4-5 | audit3d | 涡扇基准：4 真干涉不变，耗时 ↓50% |
| S4-6 | 回归 | 64 断言全绿（更新后） |

**门禁 S4**：S4-1~S4-6 全 PASS → 进入 S5。

---

## 7. S5 schema v3 + 多轮协议

### 7.1 设计方案
schema v3 全量替换 v2（vs_schema.py 重写）。新增 analyze/zoom/probe 三动作（extensions/pi-vision-core.ts），实现多轮反馈协议。

### 7.2 架构逻辑
```
vs analyze [image]  → 粗报告（saliency 候选 + ascii 粗栅格 + 全局统计 + 融合 findings）
vs zoom [region]    → 细报告（region 内 ocr 高倍 + segment 前景 + edge 亚像素 + depth）
vs probe [bbox, sensor] → 单传感器定向证据
DeepSeek 驱动多轮：粗看 → 选区 → 细看 → 追问 → 综合推理
```

### 7.3 技术实现
- [ ] `vs_schema.py`：v3 结构（elements/findings/candidates/foreground/metrics/notation）
- [ ] `extensions/pi-vision-core.ts`：动作表更新（analyze/zoom/probe + 保留旧动作）
- [ ] `python/tasks/`：新任务配置（diagnose-v3 / zoom / probe）
- [ ] 端到端测试：真实复杂背景图走完整多轮流程

### 7.4 工程收敛边际
- schema v2 输出全部升级 v3（无 v2 残留）
- 旧动作（capture/pix/ocr/dom 等）仍可用（内部走新管线）
- analyze/zoom/probe 三动作可用
- DeepSeek 可通过工具调用驱动多轮

### 7.5 印证方案
| # | 印证项 | 通过标准 |
|---|---|---|
| S5-1 | schema 合规 | 全部输出 schema==vision-report/v3 |
| S5-2 | 多轮流程 | 真实复杂图：analyze→zoom→probe 全链路成功 |
| S5-3 | 旧动作兼容 | capture/pix/ocr/dom 等旧动作仍工作 |
| S5-4 | 端到端 | DeepSeek 基于 v3 报告给出正确结论（人工评估） |

**门禁 S5**：S5-1~S5-4 全 PASS → 进入 S6。

---

## 8. S6 系统整合 + 全量回归

### 8.1 设计方案
全部模块整合为完整系统：移除 L2 语义层（vs_semantic/vs_critic/vs_vlm），统一配置，全量回归，文档同步。

### 8.2 架构逻辑
```
最终系统:
  vs 单端口 → pi-vision-core.ts → 子进程调度
    ├─ L1 确定性（pi-vision env）
    ├─ L2 轻量 DL（vsensor env，经 vsd/vsched）
    └─ F1 融合（pi-vision env，vs_fusion）
  输出 schema v3 → DeepSeek 推理
```

### 8.3 技术实现
- [ ] 删除 vs_semantic.py / vs_critic.py / vs_vlm.py
- [ ] 统一配置 `~/.config/pi-vision-struct.json`（sensors.enabled / power_mode / budgets）
- [ ] 全量回归：64 断言 + 各阶段印证项重跑
- [ ] 知识库同步：01-06 + CHANGELOG + README
- [ ] `kb-audit.sh` 通过
- [ ] git commit + tag v3.0.0

### 8.4 工程收敛边际
- 零 qwen3-vl 依赖（代码审计确认）
- 零第三方 API 调用（网络监控确认）
- 全部输出 schema v3 + 可复算
- 文档与实现一致

### 8.5 印证方案
| # | 印证项 | 通过标准 |
|---|---|---|
| S6-1 | 全量回归 | 64 断言 + 全部阶段印证项 PASS |
| S6-2 | 零依赖审计 | grep 无 qwen3-vl/ollama/API 调用 |
| S6-3 | 端到端 | 3 类真实图（UI/复杂背景/精密零件）全链路成功 |
| S6-4 | 性能 | 全流水线 CPU <5s，GPU <2s |
| S6-5 | 文档 | kb-audit.sh 退出码 0 |
| S6-6 | 发布 | git tag v3.0.0 + CHANGELOG 更新 |

**门禁 S6**：S6-1~S6-6 全 PASS → **v3.0.0 发布完成**。

---

## 9. 系统优化与整合（发布后）

发布后进入**系统级优化**阶段（非门禁，持续迭代）：
1. 真实使用数据收集 → 传感器权重调优（阈值/预算）
2. 性能剖析（cProfile/nvidia-smi）→ 热点优化
3. 多轮协议体验优化（DeepSeek 调用模式分析）
4. 新场景扩展（按需新增传感器/任务配置）

---

## 10. 风险与回滚（不变）

- git tag v2.2.2 保留，一键回滚
- 三 env 分离，删除 vsensor 即回退
- 每传感器独立开关
- 每阶段门禁失败 → 修复重验，不跨阶段

---

## 11. 时间线

| 阶段 | 工期 | 门禁 |
|---|---|---|
| S0 | 0.5 天 | GPU + 三 env |
| S1 | 2 天 | 融合正确性 |
| S2 | 3 天 | 传感器基准 |
| S3 | 3 天 | 并发 + 功耗 |
| S4 | 2 天 | 性能 + 回归 |
| S5 | 2 天 | 端到端 |
| S6 | 1 天 | 全量回归 + 发布 |
| **合计** | **~13.5 天** | 7 道门禁 |
