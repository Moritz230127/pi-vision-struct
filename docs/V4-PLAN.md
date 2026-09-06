# pi-vision-struct V4 · 全栈 Rust 重写计划（草案 v0.1）

> 状态: DRAFT — 待用户逐项确认
> 日期: 2026-09-06
> 约束: **先不发布**（不 push GitHub / 不 Release）；一点一点来（Gate-Based）；保持 schema v3 输出契约与 V3 完全兼容（差分验证）；零第三方 API 约束不变。

---

## 0. 现状分析（V3.0.0 源码盘点）

### 0.1 代码规模

| 层 | 文件 | 行数 | 语言 |
| --- | --- | --- | --- |
| 宿主适配 | extensions/index.ts + server.mcp.ts | 185 | TS |
| 框架无关核心 | extensions/pi-vision-core.ts | 802 | TS |
| 传感器/引擎 | python/*.py（40 文件） | 8491 | Python |
| 安装/沙箱 | python/setup/* | ~600 | sh/ps1/py |
| 测试/基准 | tests/ + bench/ | ~1500 | Python |
| 文档 | README + docs/ + skills/ | ~2000 | MD |

### 0.2 组件依赖图（Rust 重写可行性标注）

```
┌─ 宿主层 ─────────────────────────────────────────────┐
│ pi-coding-agent (index.ts) ─┐                        │
│ Claude Code MCP (server.mcp.ts) ─┤ → pi-vision-core  │  ← 薄适配器，保留 TS
└──────────────────────────────┼───────────────────────┘
                               ▼
┌─ 核心分发 (pi-vision-core.ts) ───────────────────────┐
│ Act 分发表(22动作) · python 解析 · preflight ·        │  ← Rust 重写（clap+serde）
│ bwrap 沙箱 · 失败日志 · TypeBox→MCP schema            │
└──────────────────────────────┼───────────────────────┘
                               ▼
┌─ Python 传感器层（40 脚本）──────────────────────────┐
│ 纯算法（Rust 直接重写）:                              │
│   vs_schema(190) vs_pix(166) vs_edge(199)            │  ★ 高价值：类型安全+去 scipy
│   vs_ascii vs_scene_stats(171) vs_fusion(370)        │
│   vs_rules(334) vs_audit(138) vs_geometry(167)       │  ★ vtracer 本身就是 Rust！
│   vs_wall(197) vs_cluster(156)                       │
│ DL 推理（onnxruntime → Rust ort crate 可行）:         │
│   vs_saliency(U²-Net) vs_depth(MiDaS) vs_segment     │  ◐ 需验证 ort + ONNX 导出
│ DL 推理（transformers 生态，Python 绑定）:            │
│   vs_ocr(Paddle/Rapid) vs_omniparser(YOLOv9+Florence)│  ✗ 保留 Python 子进程
│   vs_detect(OWLv2) vs_layout(PP-DocLayoutV3)         │
│ 外部进程:                                            │
│   vs_dom(Playwright) vs_pptx(python-pptx)            │  ◐ Rust crate 或子进程
│   vs_pdf(PyMuPDF) vs_a11y(AT-SPI)                    │  ◐ atspi crate 可行
│   vs_capture(grim) vs_blender_dump(bpy)              │  ◐ 子进程保留
│ 基础设施:                                            │
│   vsd.py(daemon) vsched.py(调度) ocrserver.py        │  ★ Rust tokio 重写
│   omniserver.py                                      │
└──────────────────────────────────────────────────────┘
```

### 0.3 关键发现（分析结论）

1. **纯算法传感器占 ~40% 代码量**（schema/pix/edge/ascii/scene_stats/fusion/rules/audit/geometry/wall/cluster ≈ 2200 行），是 Rust 重写的最高价值目标——它们依赖 scipy/numpy，Rust 侧用 ndarray + 自研算法可完全替代，且消除 conda env 依赖。
2. **vs_geometry 的 VTracer 本身就是 Rust 库**（crate `vtracer`），重写后直接内联，省掉一个子进程 + vsensor env。
3. **DL 传感器分两类**：
   - ONNX 系（U²-Net/MiDaS/MobileSAM）→ Rust `ort` crate 直接加载 ONNX，可脱离 Python；
   - transformers 系（OCR/OmniParser/OWLv2/layout）→ 无成熟 Rust 等价物，保留 Python 子进程（这是"全栈 Rust"的现实边界）。
4. **宿主层必须保留 TS 薄壳**：pi 扩展 API 是 TS 原生（ExtensionAPI），Claude Code 走 MCP。Rust 二进制作为后端，TS 壳只做 exec + 参数透传（index.ts 可缩到 ~50 行）。
5. **既有健壮性缺口**（pi-lens 扫描 138 blocking，全部为源仓库既有）：`vs_audit3d.py L272 os 未定义`、tuple 尺寸不匹配、`json.loads` 无保护、`os.remove` 竞态——Rust 重写天然消除（编译期类型 + Result 错误链）。
6. **验证资产可复用**：`tests/bench_fixtures/*.json`（golden 输出）+ `bench/samples/*.gt.json`（真值）+ 61 断言回归 = Rust 差分测试的黄金标准。

---

## 1. 目标架构（V4）

```
┌─ 宿主薄壳（TS，保留）───────────────────────────────┐
│ index.ts (pi) · server.mcp.ts (Claude Code)          │
│ 只做: exec vs 二进制 + 参数透传 + 输出回传            │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─ vs 二进制（Rust，单二进制）─────────────────────────┐
│ clap 子命令 = action 枚举（25 动作）                  │
│ serde schema v3 输出（与 Python 逐字节兼容）          │
│ 分发: 纯算法内联 / DL 走 ort / transformers 走子进程  │
│ 基础设施: preflight · 失败日志 · bwrap 沙箱 · 超时    │
│ daemon: vsd (tokio + unix socket) · 调度 (信号量)     │
└──────┬──────────────────────┬───────────────────────┘
       ▼                      ▼
┌─ Rust 算法核心 ────────┐  ┌─ Python 子进程（保留）───┐
│ vs-core crate:         │  │ ocr / omniparser /      │
│  schema·color·bbox      │  │ detect / layout         │
│  pixels·edge·ascii      │  │ dom / pptx / pdf / a11y │
│  scene_stats·fusion     │  │ blender_dump            │
│  rules·audit·geometry   │  └─────────────────────────┘
│  (vtracer 内联)         │
│  saliency·depth·segment │  ← ort (onnxruntime)
└─────────────────────────┘
```

**关键决策（待确认）**：

| # | 决策 | 选项 | 建议 |
| --- | --- | --- | --- |
| D1 | 二进制形态 | A. 单二进制 `vs`（clap 子命令） B. 每传感器一二进制 | A（延续单端口设计） |
| D2 | DL 策略 | A. ort 全内联 B. 保留 Python 子进程 C. 混合（ONNX 内联 + transformers 子进程） | C |
| D3 | 宿主壳 | A. TS 薄壳 exec 二进制 B. Rust MCP server 直连 | A（pi 扩展 API 是 TS 原生） |
| D4 | 沙箱 | A. 保留 bwrap B. Rust 侧 landlock/seccomp | A（bwrap 是内核工具，与语言无关） |
| D5 | 输出契约 | A. 与 schema v3 逐字节兼容 B. 允许 v4 schema 演进 | A（先兼容，后演进） |
| D6 | 工作区 | A. 在 pi-vision-struct-v4 内建 cargo workspace B. 独立仓库 | A（同仓库不同目录） |

---

## 2. 分阶段计划（Gate-Based，一点一点来）

> 每阶段 = 设计 → 实现 → 差分验证（Python vs Rust 同 fixture 输出对比）→ 门禁 PASS/FAIL。
> 门禁标准：**Rust 输出与 Python 输出 JSON 逐字段一致**（数值容差 ≤1e-6，bbox 整数一致）。

### S0 · 工程基座

- 建 cargo workspace（`rust/` 目录）：`vs-core`（算法库）+ `vs-bin`（CLI）+ `vsd`（daemon）
- 依赖选型：clap / serde / serde_json / ndarray / image / rayon / tokio / ort / vtracer
- 移植 `vs_schema.py` → `vs-core::schema`（color math: srgb→linear, Lab, ΔE76, WCAG；bbox 原语；schema v3 envelope）
- **门禁**: schema 单元测试 + 与 Python vs_schema 输出逐字段一致

### S1 · 纯算法传感器（第一批）

- pixels（SLIC 超像素 + K-means + diff 连通域 + WCAG）
- scene_stats（真 Otsu + 直方图 + 对比度）
- edge（Devernay 亚像素）
- ascii（多分辨率栅格）
- **门禁**: 5 传感器 × bench_fixtures 差分全绿

### S2 · 融合与规则引擎

- fusion（D-S 证据理论 + 匈牙利匹配，替代 scipy.optimize.linear_sum_assignment）
- rules（设计元素规则引擎）
- audit（重叠/对齐审计）
- geometry（vtracer crate 内联）
- **门禁**: 61 断言回归（Rust 版）+ 误报率对比

### S3 · 基础设施

- vsd daemon（tokio + unix socket 行 JSON 协议，替代 vsd.py）
- 调度器（显存预算 + 信号量 + LRU + 优先级 + 功耗感知，替代 vsched.py）
- preflight / 失败日志 / 超时 / bwrap 集成
- **门禁**: 并发压测（5 任务零 OOM）+ 功耗切换测试

### S4 · ONNX DL 传感器（ort）

- saliency（U²-Net ONNX）→ ort
- depth（MiDaS ONNX）→ ort
- segment（MobileSAM ONNX 导出）→ ort
- **门禁**: 与 Python 版输出差分（候选区 IoU ≥ 0.95）+ 显存峰值对比

### S5 · 外部集成（子进程保留）

- capture（grim 子进程 or xcap crate）
- dom（Playwright 子进程）/ pptx / pdf / a11y（atspi crate 试点）
- blender_dump / depth_geom（bash 包装保留）
- **门禁**: 端到端真实图测试

### S6 · 宿主接入 + 系统整合

- TS 薄壳改造（index.ts exec vs 二进制）
- server.mcp.ts 指向 Rust MCP 输出（或保持 TS 壳）
- 全量回归（Rust 版 61 断言 + 31 全功能 + 基准卡）
- 文档同步（README / SKILL.md / 知识库）
- **门禁**: 全绿 + kb-audit 通过

### S7 · 发布（用户确认后）

- GitHub push + tag v4.0.0 + Release（**当前不执行**）

---

## 3. 验证策略（差分测试）

```
fixture (bench_fixtures/*.json + bench/samples/*.png)
   ├── Python 实现 → output_py.json   （基线，已锁定）
   └── Rust 实现   → output_rs.json
        └── 逐字段 diff（数值 ≤1e-6，bbox 整数一致，schema 字段名一致）
```

- 每阶段新增 `rust/tests/diff_*.rs` 差分测试
- 数值容差策略：颜色/统计 ≤1e-6；bbox/坐标整数一致；置信度 ≤1e-4
- 性能基准：Rust 版必须 ≥ Python 版（目标 2-10×，消除 numpy/scipy 启动开销）

---

## 4. 风险与边界

| 风险 | 等级 | 缓解 |
| --- | --- | --- |
| transformers 系（OCR/OmniParser/OWLv2/layout）无法 Rust 化 | 高 | 明确保留 Python 子进程，这是"全栈 Rust"的现实边界 |
| ort + ONNX 模型导出兼容性（MobileSAM） | 中 | S4 先行验证，失败则回退子进程 |
| SLIC/K-means 数值一致性（numpy 浮点 vs Rust） | 中 | 容差策略 + 逐字段 diff |
| 匈牙利算法（scipy linear_sum_assignment 实现细节） | 低 | 自研 O(n³) 或 crate，输出匹配即可 |
| pi 扩展加载机制对二进制后端的限制 | 低 | S6 验证 TS 壳 exec 方案 |
| 工作量（~8500 行 Python → Rust） | 高 | 分阶段 Gate-Based，每阶段可独立交付 |

---

## 5. 待用户确认

1. D1-D6 决策（见 §1 表格）
2. 阶段顺序是否调整（建议 S0→S1→S2→S3→S4→S5→S6）
3. "全栈 Rust"的边界确认：transformers 系传感器保留 Python 子进程是否可接受
4. 是否先做 S0 基座（cargo workspace + schema 移植）作为第一步
