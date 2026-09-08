# pi-vision-struct V4 · 全面 Rust 重写计划（FINAL）

> 状态: APPROVED — 用户已确认核心约束（全面 Rust / 不用其他语言 / 最高效率调度 / 最低延迟）
> 日期: 2026-09-06
> 版本: v4.0.0（目标）
> 约束: **先不发布**（不 push GitHub / 不 Release）；实现语言仅 Rust；schema v3 输出契约与 V3 差分兼
  容；零第三方 API；全部输出确定性可复算。

---

## 0. 目标与硬约束

| # | 约束 | 含义 |
| --- | --- | --- |
| C1 | **全面 Rust** | 实现语言仅 Rust。Python/TS 不作为实现语言存在 |
| C2 | **最高效率调度** | GPU/CPU 统一调度器，显存预算 + 优先级 + 功耗感知 + 并发控制，任务永不 OOM |
| C3 | **最低延迟** | 单二进制零解释器启动；常驻 daemon 热模型；零拷贝图像管线；rayon 并行 |
| C4 | 先不发布 | 全部工作本地，不 push / 不 Release |
| C5 | 输出契约 | schema v3 与 Python 版逐字段兼容（差分验证） |
| C6 | 零第三方 API | 全部本地推理，无网络外发 |

---

## 1. 技术选型（crate 清单）

### 1.1 核心依赖

| 用途 | crate | 说明 |
| --- | --- | --- |
| CLI/分发 | `clap` | 子命令 = action 枚举（25 动作） |
| 序列化 | `serde` + `serde_json` | schema v3 输出 |
| 异步运行时 | `tokio` | daemon + 调度器 |
| 并行计算 | `rayon` | CPU 算法并行 |
| 图像 | `image` + `ndarray` | 解码/像素操作 |
| 数值 | `nalgebra` 或 `ndarray` | 矩阵/线性代数（替代 numpy/scipy） |
| DL 推理 | `ort` | onnxruntime 绑定（CUDA EP） |
| 矢量追踪 | `vtracer` | 本身就是 Rust，内联（替代 vs_geometry 子进程） |
| GPU 监控 | `nvml-wrapper` | 显存/功耗/温度（替代 nvidia-smi 轮询） |
| 沙箱 | `landlock` + `rustix` | Landlock LSM + seccomp + namespaces（纯 Rust 沙箱） |
| 截图 | `xcap` | Wayland screencopy（替代 grim） |
| 浏览器 | `chromiumoxide` | CDP 驱动 headless Chrome（替代 Playwright） |
| 无障碍 | `atspi` | 官方 AT-SPI2 Rust 绑定（替代 python-gobject） |
| PDF | `pdfium-render` | PDFium 绑定（替代 PyMuPDF） |
| PPTX | `zip` + `quick-xml` | 直接解析 OOXML（python-pptx 无 Rust 等价物，自研） |
| .blend | `blend` (blend-rs) | 纯 Rust 解析 Blender 文件（替代 bpy） |
| OCR | `rapidocr-core` | RapidOCR ONNX 流水线（与 Python 同模型） |
| MCP | `rmcp` | Claude Code 宿主（纯 Rust MCP server） |

### 1.2 模型 ONNX 化清单（全 Rust 推理的前提）

| 传感器 | 模型 | ONNX 状态 | 难度 |
| --- | --- | --- | --- |
| ocr | ch_PP-OCRv4 det/rec/cls | **已是 ONNX**（RapidOCR 即 ONNX 版 PaddleOCR） | ✅ 直接可用 |
| saliency | U²-Net | **已是 ONNX**（V3 已用 ONNX GPU） | ✅ 直接可用 |
| depth | MiDaS | ONNX 官方导出可用 | ✅ 低 |
| segment | MobileSAM | 官方导出脚本存在 | ◐ 中 |
| omniparser | YOLOv9 + Florence-2 | YOLOv9 标准导出；Florence-2 手动拆分导出（encoder/decoder/embed_tokens，opset 17，已验证可行） | ◐ 高 |
| detect | OWLv2 | transformers 导出复杂（开放词表 + 双编码器） | ✗ 边界候选 |
| layout | PP-DocLayoutV3 | paddle2onnx 导出 | ◐ 中 |

---

## 2. 架构设计

```text
┌─ 宿主层 ─────────────────────────────────────────────┐
│ pi-coding-agent: TS 薄壳（平台约束，~50 行 exec 胶水） │  ← 唯一非 Rust 代码
│ Claude Code: Rust MCP server（rmcp crate）            │  ← 纯 Rust
└──────────────────────┬───────────────────────────────┘
                       ▼
┌─ vs 单二进制（Rust）──────────────────────────────────┐
│ clap 子命令 = action（25 动作）                        │
│ serde schema v3 输出（与 Python 逐字段兼容）           │
│ 分发: 纯算法内联 / DL 走 ort / 外部走子进程            │
│ 基础设施: preflight · 失败日志 · 超时 · 沙箱(landlock) │
└──────┬──────────────────────┬────────────────────────┘
       ▼                      ▼
┌─ Rust 算法核心 ────────┐  ┌─ vsd daemon（tokio）──────┐
│ vs-core crate:         │  │ unix socket 行 JSON 协议   │
│  schema·color·bbox      │  │ 调度器: 显存预算+信号量    │
│  pixels·edge·ascii      │  │ LRU 模型缓存 · 优先级队列  │
│  scene_stats·fusion     │  │ 功耗感知(NVML) · CPU 降级  │
│  rules·audit·geometry   │  │ 热模型常驻（零冷载）       │
│  ocr(rapidocr-core)     │  └───────────────────────────┘
│  saliency·depth·segment │  ← ort (onnxruntime CUDA)
│  omniparser(YOLOv9+Florence2 ONNX) │
└─────────────────────────┘
```

### 2.1 单二进制 + 常驻 daemon 双模式

- **CLI 模式**：`vs <action> [args]` — 单次调用，进程即起即灭（低开销动作：capture/pixels/ascii）
- **Daemon 模式**：`vsd` 常驻 — 热模型 + 调度器（高开销动作：ocr/omniparser/saliency/segment/depth）
- 自动路由：动作按开销分级，轻量动作直连，重量动作走 daemon socket

### 2.2 调度器设计（最高效率）

```text
请求 → 优先级队列（analyze > zoom > probe > bg）
     → 显存预算检查（NVML 实时查询，总预算 6.5GB / 8GB VRAM）
     → GPU 信号量（串行推理，杜绝同刻双 GPU 占用）
     → LRU 模型缓存（热 <1ms，冷载 1-3s，显存紧张时淘汰最久未用）
     → 功耗感知（battery → 大模型 CPU 降级；plugged → 全速）
     → CPU 降级（显存不足时自动 CPU 推理，任务永不失败）
```

**与 V3 的关键差异**：V3 用子进程隔离（每次推理 spawn 子进程，峰值 816MB/5 并发）；
V4 用**进程内推理 + 显式模型卸载**（ort session drop）——省掉进程 spawn 开销
（~50-100ms/次），延迟更低，显存由调度器精确控制。

### 2.3 延迟优化策略

| 手段 | 收益 |
| --- | --- |
| 单二进制，零解释器启动 | 省 Python 启动 100-300ms/次 |
| daemon 热模型常驻 | 冷载 1-3s → 热 <1ms |
| 进程内推理（无子进程 spawn） | 省 50-100ms/次 |
| 零拷贝图像管线（image→ndarray→ort 直传） | 省编码/解码往返 |
| rayon 并行（SLIC/K-means/连通域/边缘） | CPU 算法 2-10× |
| 惰性加载 + LRU | 显存与延迟最优平衡 |
| ort 线程池调优（intra-op/inter-op） | 推理吞吐最大化 |

---

## 3. 分阶段实现路径（Gate-Based）

> 每阶段 = 设计 → 实现 → 差分验证（Python vs Rust 同 fixture）→ 门禁 PASS/FAIL。
> 门禁标准：**Rust 输出与 Python 输出 JSON 逐字段一致**（数值 ≤1e-6，bbox 整数一致）。

### S0 · 工程基座

- cargo workspace（`rust/`）：`vs-core`（算法库）+ `vs-bin`（CLI）+ `vsd`（daemon）
- 移植 `vs_schema.py` → `vs-core::schema`（color math: srgb→linear, Lab, ΔE76, WCAG；bbox 原语；schema v3 envelope）
- 差分测试基座：`rust/tests/diff_*.rs` + fixture 加载器
- **门禁**: schema 单元测试 + 与 Python vs_schema 输出逐字段一致

### S1 · 纯算法传感器（无 DL）

- pixels（SLIC + K-means + diff 连通域 + WCAG）
- scene_stats（真 Otsu + 直方图 + 对比度）
- edge（Devernay 亚像素）
- ascii（多分辨率栅格）
- **门禁**: 5 传感器 × bench_fixtures 差分全绿

### S2 · 融合与规则引擎

- fusion（D-S 证据理论 + 匈牙利匹配）
- rules（设计元素规则引擎）
- audit（重叠/对齐审计）
- geometry（vtracer 内联）
- **门禁**: 61 断言回归（Rust 版）+ 误报率对比

### S3 · 基础设施 + 调度器

- vsd daemon（tokio + unix socket）
- 调度器（显存预算 + 信号量 + LRU + 优先级 + 功耗感知 + CPU 降级）
- 纯 Rust 沙箱（landlock + seccomp + namespaces，替代 bwrap）
- preflight / 失败日志 / 超时
- **门禁**: 并发压测（5 任务零 OOM）+ 功耗切换 + 沙箱逃逸测试

### S4 · ONNX DL 传感器（ort）

- saliency（U²-Net ONNX）→ ort
- depth（MiDaS ONNX）→ ort
- segment（MobileSAM ONNX 导出）→ ort
- **门禁**: 与 Python 版输出差分（候选区 IoU ≥ 0.95）+ 显存峰值对比

### S5 · OCR 全 Rust 化

- rapidocr-core 集成（det→cls→rec，与 Python 同模型）
- 词典纠错 + 4 点 bbox + conf 输出（schema v3 兼容）
- **门禁**: 与 Python OCR 差分（CER=0，bbox IoU ≥ 0.95）

### S6 · OmniParser + detect + layout（最难）

- YOLOv9 ONNX 导出 → ort
- Florence-2 手动拆分 ONNX 导出（encoder/decoder/embed_tokens）→ ort 自研生成循环
- layout（PP-DocLayoutV3 → paddle2onnx）→ ort
- detect（OWLv2）→ **边界决策点**（见 §4.3）
- **门禁**: 与 Python 版输出差分（元素 IoU ≥ 0.9）

### S7 · 外部集成（全 Rust 化）

- capture（xcap，Wayland screencopy）
- dom（chromiumoxide 驱动 headless Chrome）
- a11y（atspi crate）
- pdf（pdfium-render）
- pptx（OOXML 自研解析）
- 3D（blend-rs 解析 .blend + 自研深度栅格化）→ **边界决策点**（见 §4.4）
- **门禁**: 端到端真实图测试

### S8 · 宿主接入 + 系统整合

- TS 薄壳改造（index.ts → exec vs 二进制，~50 行）
- Rust MCP server（rmcp）替代 server.mcp.ts
- 全量回归（Rust 版 61 断言 + 31 全功能 + 基准卡）
- 文档同步（README / SKILL.md / 知识库）
- **门禁**: 全绿 + kb-audit 通过

### S9 · 发布（用户确认后）

- GitHub push + tag v4.0.0 + Release（**当前不执行**）

---

## 4. 工程边界（明确"全面 Rust"的精确含义）

### 4.1 宿主层：pi 必须保留 TS 薄壳（平台约束）

- pi 扩展 API 是 TS 原生（ExtensionAPI），**无内置 MCP 支持**（官方文档明确）
- 边界：`extensions/index.ts` 缩至 ~50 行 exec 胶水（spawn vs 二进制 + 参数透传 + 输出回传），**零业务逻辑**
- Claude Code 侧：Rust MCP server（rmcp crate）——纯 Rust，无 TS

### 4.2 外部应用程序（非实现语言，作为子进程调用）

| 工具 | 用途 | 边界说明 |
| --- | --- | --- |
| headless Chrome | dom 传感器 | chromiumoxide 驱动，Chrome 是浏览器应用非语言 |
| Blender | 3D 模块 | **边界决策点**（见 4.4） |
| onnxruntime | DL 推理 | 通过 ort crate 绑定，C++ 库非语言 |

### 4.3 detect（OWLv2）边界决策点

- OWLv2 是 transformers 开放词表检测，ONNX 导出复杂（双编码器 + 动态文本）
- 选项：A. 手动 ONNX 导出（高成本） B. 替换为 Rust 友好的开放词表检测器 C. 降级为固定类别检测（YOLO 系）
- **决策：S6 先行验证，失败则选 B/C**

### 4.4 3D 模块（blender_dump/depth/audit3d）边界决策点

- 选项：A. blend-rs 纯 Rust 解析 .blend + 自研深度栅格化（全 Rust，工作量大） B. 保留 Blender 子进程
  （Blender 是独立应用，但需其 Python 脚本驱动——违反"不用其他语言"精神） C. 3D 模块延后到 V4.1
- **决策：A 优先；若栅格化精度不达标，C 延后**

### 4.5 明确排除（不进入 V4 范围）

- Docker 镜像（V3 产物，Rust 单二进制无需容器分发）
- Windows/macOS 安装脚本（V4 先 Linux-only，与 V3 一致）

---

## 5. 验证策略（差分测试）

```text
fixture (bench_fixtures/*.json + bench/samples/*.png)
   ├── Python 实现 → output_py.json   （基线，已锁定）
   └── Rust 实现   → output_rs.json
        └── 逐字段 diff（数值 ≤1e-6，bbox 整数一致，schema 字段名一致）
```

- 每阶段新增 `rust/tests/diff_*.rs` 差分测试
- 性能基准：Rust 版必须 ≥ Python 版（目标 2-10×）
- 延迟基准：daemon 热调用 vs Python 冷调用（目标 <50ms 热路径）

---

## 6. 风险与缓解

| 风险 | 等级 | 缓解 |
| --- | --- | --- |
| Florence-2 ONNX 导出（OmniParser 核心） | 高 | 手动拆分导出已验证可行（xenova gist）；S6 先行验证 |
| OWLv2 无法 Rust 化 | 中 | 边界决策点 4.3：替换检测器或降级 |
| 3D 深度栅格化精度 | 中 | 边界决策点 4.4：blend-rs + 自研栅格化，不达标则延后 |
| SLIC/K-means 数值一致性 | 中 | 容差策略 + 逐字段 diff |
| 匈牙利算法实现细节 | 低 | 自研 O(n³) 或 crate，输出匹配即可 |
| pi TS 薄壳（唯一非 Rust 代码） | 低 | 平台约束，~50 行 exec 胶水，零逻辑 |
| 工作量（~8500 行 Python → Rust） | 高 | 分阶段 Gate-Based，每阶段独立交付 |

---

## 7. 里程碑

| 里程碑 | 内容 | 预计 |
| --- | --- | --- |
| M1 | S0-S2 完成（基座 + 纯算法 + 融合规则） | 第一批可交付 |
| M2 | S3-S5 完成（调度器 + DL + OCR） | 核心能力 Rust 化 |
| M3 | S6-S8 完成（OmniParser + 外部集成 + 宿主） | 全面 Rust 达成 |
| M4 | S9 发布（用户确认后） | 待定 |

---

## 8. 下一步（S0 启动清单）

1. 建 `rust/` cargo workspace（vs-core / vs-bin / vsd 三 crate）
2. 移植 vs_schema → vs-core::schema（含全部 color math + bbox 原语）
3. 差分测试基座（fixture 加载 + 逐字段比较器）
4. 门禁验证：schema 输出与 Python 逐字段一致
