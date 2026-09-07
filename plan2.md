# plan2.md — V4.1 全面超越计划（Rust 版逐行超越 Python 版）

> 目标：**不是功能覆盖，而是每一行代码的超越**——消除全部 Python 回退、全部差分容差、
> 全部架构差距。Rust 版在功能、数值一致性、性能、安全、可部署性五个维度全面超过 Python 版。
>
> 前置：V4（S0-S8）已完成，159 测试全绿，生产环境已替换 pi 扩展。

---

## 1. 差距全景（实测盘点，2026-09-07）

### A. 功能差距：Python 回退点（8 处，1,308 行 Python 仍被调用）

| # | 回退点 | Python 脚本 | 行数 | 根因 |
| --- | --- | --- | --- | --- |
| A1 | `layout` | vs_layout.py | 79 | paddle2onnx PIR 格式不兼容（paddle 3.3.1） |
| A2 | `segment` mask decoder | vs_segment_decoder.py | 96 | MobileSAM Transformer 数值一致性风险 |
| A3 | `wallpaper` | vs_wall.py | 197 | 壁纸程序化分类（主色/亮度/饱和度/宽高比） |
| A4 | `analyze` | vs_analyze.py | 163 | 任务引擎（JSON 配置编排传感器） |
| A5 | `cluster` | vs_cluster.py | 156 | CLIP 离线聚类（open_clip ViT-B-32） |
| A6 | `audit3d` | vs_audit3d.py | 321 | 3D 装配审计（scipy cKDTree 硬依赖） |
| A7 | `zoom`/`probe` | vs_protocol.py | 173 | 多轮反馈协议（analyze/zoom/probe） |
| A8 | `check`/`setup` | setup/vs_setup.py | 298 | 引导安装器（conda env 检测/创建） |

### B. 数值一致性差距：差分测试容差（8 处）

| # | 传感器 | 当前容差 | plan 目标 | 根因 |
| --- | --- | --- | --- | --- |
| B1 | S5 OCR bbox | IoU ≥ 0.85 | **1.0** | det 框几何差（unclip 1.4 vs pyclipper） |
| B2 | S5 OCR conf | ±0.3 | **±0.01** | bbox 几何差 → crop 不同 → rec 分数差 |
| B3 | S6 omniparser caption | 非空+长度 | **逐 token 一致** | Florence-2 greedy vs transformers 采样 |
| B4 | S7 pdf bbox | ±5pt | **±1pt** | pdfium vs PyMuPDF 引擎差异 |
| B5 | S7 blend bbox3d | ±0.01 | **±0.001** | blend-rs vs bpy 顶点序 |
| B6 | S1 pix Lab | <0.005 | **<0.001** | D65 矩阵 vs skimage 精确矩阵 |
| B7 | S4 saliency pixels | ±50 | **±5** | PIL LANCZOS vs image Lanczos3 |
| B8 | S4 depth q75 | ±0.03 | **±0.005** | resize 滤波器微差 |

### C. 架构差距（5 处）

| # | 差距 | 现状 | 目标 |
| --- | --- | --- | --- |
| C1 | vsd 沙箱未接入 | sandbox.rs 有实现，main.rs 未调用 | 推理路径全程沙箱化 |
| C2 | vs-mcp 单工具 | 1 个 `vs` 工具（action 分发） | 细粒度工具（每传感器一个） |
| C3 | capture/dom/a11y 无差分 | 环境依赖无法 golden | 契约测试（结构 + 类型 + 非空） |
| C4 | audit3d 未 Rust 化 | scipy cKDTree 硬依赖 | 纯 Rust（kiddo/kdtree crate） |
| C5 | vsd 调度器功耗感知 | 仅 battery/plugged 检测 | 完整功耗模型（CPU 降级 + 频率） |

### D. 代码级差距（"每一行"层面，5 处）

| # | 差距 | Python 实现 | Rust 现状 | 目标 |
| --- | --- | --- | --- | --- |
| D1 | resize 滤波器 | PIL LANCZOS/BILINEAR | image crate Lanczos3/Triangle | 逐像素一致（自研 LANCZOS 内核） |
| D2 | Lab 矩阵 | skimage 精确 D65 | 近似 D65 | 精确矩阵（0.95047/1.0/1.08883） |
| D3 | VTracer 版本 | Python 绑定（新） | Rust 1.0.0-alpha.4 | 升级到同版本或 fork 对齐 |
| D4 | Florence-2 采样 | transformers 采样 | greedy | 复刻采样参数（temperature/top_p） |
| D5 | 光栅化 | cv2.fillPoly | scanline（已修 S5） | 全传感器统一整数光栅化 |

---

## 2. 解决方案（按差距编号）

### A 类：消除全部 Python 回退

**A1 layout（PP-DocLayoutV3）**：

- 方案：paddle 3.3.1 PIR → 用 `paddle.jit.to_static` + `paddle2onnx` 新版本（2.1.1+ 支持 PIR），或
  降级 paddle 到 2.6.x（pdmodel 格式）导出 → ort 推理
- 备选：PP-DocLayoutV3 是 DETR 系，用 YOLOv9 同款导出路径（ultralytics 格式）
- 门禁：与 Python paddle 输出 IoU ≥ 0.95

**A2 segment mask decoder**：

- 方案：MobileSAM mask decoder 是轻量 Transformer（~4M 参数），ONNX 导出
  （prompt encoder + mask decoder 两个 ONNX）→ 纯 Rust 推理
- 门禁：mask 逐像素 IoU ≥ 0.99（vs torch）

**A3 wallpaper**：

- 方案：纯 Rust 程序化分类（主色直方图峰值 + 亮度档 + 饱和度档 + 宽高比）——全部确定性算法，
  无 DL 依赖，直接移植
- 门禁：与 vs_wall.py 逐字段一致（确定性输出精确）

**A4 analyze（任务引擎）**：

- 方案：Rust 任务引擎（JSON 配置 → 传感器编排 → 融合），vs-bin 子命令 `analyze`
- 门禁：与 vs_analyze.py 输出结构一致

**A5 cluster（CLIP 聚类）**：

- 方案：CLIP ViT-B-32 ONNX 导出（open_clip 权重）→ ort 编码 → 余弦相似度 → 贪心单链分组
- 门禁：与 vs_cluster.py 分组结果一致（确定性）

**A6 audit3d**：

- 方案：OBB-SAT 分离测试（纯 Rust 几何）+ 网格级距离（kiddo crate kd-tree 替代 cKDTree）
- 门禁：与 vs_audit3d.py findings 逐字段一致（mm 精度）

**A7 zoom/probe（协议）**：

- 方案：Rust 协议引擎（analyze/zoom/probe 三动作编排），vs-bin 子命令 `protocol`
- 门禁：与 vs_protocol.py 输出结构一致

**A8 check/setup**：

- 方案：Rust 自检（`vs check`：模型存在性 + 二进制完整性 + 环境检测）+ 安装器（`vs setup`：
  模型下载 + 目录创建）——纯 stdlib 逻辑直接移植
- 门禁：check 输出与 vs_setup.py 一致

### B 类：收紧全部差分容差

**B1/B2 OCR**：fork rapidocr-core 的 unclip 用 pyclipper round-join 等价实现（当前凸偏移 miter）
→ IoU 目标 1.0、conf 目标 ±0.01

**B3 Florence-2 caption**：复刻 transformers 采样（temperature=1.0, top_p=0.9, do_sample）——
ort 生成循环加采样参数；门禁逐 token 一致（同 seed）

**B4 pdf**：pdfium 字符级坐标 → 行聚合算法对齐 PyMuPDF（y 翻转 + 行高 = 字号 + 行合并阈值）
→ bbox ±1pt

**B5 blend**：blend-rs 顶点序对齐 bpy bound_box（8 角点顺序）→ bbox3d ±0.001

**B6 pix Lab**：color.rs 用 skimage 精确 D65 矩阵（0.95047/1.0/1.08883）→ <0.001

**B7/B8 saliency/depth**：自研 LANCZOS 内核（PIL 同款 3-lobe）替代 image crate → pixels ±5

### C 类：架构补齐

**C1 vsd 沙箱接入**：main.rs 推理路径调用 sandbox::apply（landlock + seccomp + namespace），
推理子进程全程沙箱化；门禁：沙箱逃逸测试（execve 拦截 + 外路径写入拒绝）

**C2 vs-mcp 细粒度工具**：每传感器一个 tool（capture/ocr/dom/a11y/pdf/pptx/blend/saliency/
depth/segment/omniparser/detect/layout/scene_stats/pix/edge/ascii/geometry/fusion/rules/audit）
→ 20+ 工具；门禁：tools/list 全量 + 每工具冒烟

**C3 capture/dom/a11y 契约测试**：结构（schema v3 字段）+ 类型（bbox 整数）+ 非空断言
（环境可用时）；门禁：契约测试全绿

**C4 audit3d**：见 A6

**C5 vsd 功耗模型**：CPU 频率读取（/sys/devices/system/cpu）+ 功耗档位（battery/plugged/
low-power）+ 调度器降级策略；门禁：功耗切换测试

### D 类：代码级对齐

**D1**：自研 LANCZOS 3-lobe 内核（PIL 同款）→ 全传感器统一 resize
**D2**：color.rs 精确 D65 矩阵
**D3**：VTracer 升级或 fork 对齐 Python 绑定版本
**D4**：Florence-2 采样参数复刻（见 B3）
**D5**：全传感器统一整数光栅化（fillPoly 等价）

---

## 3. 阶段划分（Gate-Based，每阶段独立交付）

### P1 · 纯算法差距清零（B6/D1/D2/D5 + A3/A8）

- 自研 LANCZOS 内核 + 精确 D65 矩阵 + 统一光栅化
- wallpaper 纯 Rust + check/setup 纯 Rust
- 门禁：S1/S4 差分容差收紧（pix Lab <0.001、saliency pixels ±5）+ 新差分测试

### P2 · DL 传感器差距清零（A1/A2/A5 + B1/B2/B3/D4）

- layout ONNX 导出（paddle 降级或 DETR 路径）
- segment mask decoder ONNX 导出
- cluster CLIP ONNX 导出
- OCR unclip pyclipper 等价 + Florence-2 采样复刻
- 门禁：S5/S6 差分容差收紧（IoU 1.0、caption 逐 token）+ 新差分测试

### P3 · 3D 与协议差距清零（A4/A6/A7 + B4/B5）

- analyze/zoom/probe 协议引擎 + audit3d 纯 Rust（kiddo）
- pdf 行聚合对齐 + blend 顶点序对齐
- 门禁：S7 差分容差收紧（pdf ±1pt、blend ±0.001）+ 新差分测试

### P4 · 架构差距清零（C1/C2/C3/C5 + D3）

- vsd 沙箱接入推理路径 + 功耗模型
- vs-mcp 细粒度工具 + capture/dom/a11y 契约测试
- VTracer 版本对齐
- 门禁：沙箱逃逸测试 + 功耗切换测试 + MCP 全工具冒烟

### P5 · 全面超越验证（收尾）

- 全量回归（全部差分容差收紧后）+ 性能基准卡更新
- 零 Python 回退验证（grep 确认无 Command::new python）
- 文档同步（README/知识库）
- 门禁：全绿 + kb-audit 通过 + 零回退点

---

## 4. 门禁标准（超越 = 逐行一致）

| 维度 | 标准 |
| --- | --- |
| 功能覆盖 | **零 Python 回退**（grep 无 `Command::new` python 调用） |
| 数值一致性 | 确定性输出**精确一致**（≤1e-6）；算法依赖输出容差收紧至 plan 目标 |
| 差分测试 | 每传感器 ≥1 差分测试；容差全部达到 B 类目标 |
| 性能 | Rust ≥ Python（目标 2-10×，基准卡全量更新） |
| 安全 | 沙箱全程接入（推理路径）+ 逃逸测试通过 |
| 可部署 | 单二进制 + 模型目录，无 conda env 依赖 |

---

## 5. 里程碑

| 里程碑 | 内容 | 预计 |
| --- | --- | --- |
| M1 | P1 完成（纯算法差距清零） | 第一批可交付 |
| M2 | P2 完成（DL 传感器差距清零） | 核心能力逐行一致 |
| M3 | P3-P4 完成（3D/协议/架构差距清零） | 全面超越达成 |
| M4 | P5 完成（全面超越验证） | 待定 |

---

## 6. 风险与缓解

| 风险 | 等级 | 缓解 |
| --- | --- | --- |
| paddle PIR 导出仍不兼容 | 中 | 降级 paddle 2.6.x 或 DETR 替代路径（A1 备选） |
| MobileSAM decoder ONNX 数值漂移 | 中 | 逐层对比 torch，容差 0.99 起步 |
| Florence-2 采样复刻不精确 | 中 | 同 seed + 同参数；失败则记录为已知（非阻塞） |
| CLIP ONNX 导出（open_clip 权重） | 中 | 用 transformers CLIP 权重（标准格式） |
| VTracer 版本对齐 | 低 | fork 或升级；元素内容比较可跳过（S2 已容差） |
| 工作量（~1,300 行 Python → Rust） | 高 | 分阶段 Gate-Based，每阶段独立交付 |
