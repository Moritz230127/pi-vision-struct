# vision-situation — 视觉技能：何时用哪个动作

> pi-vision-struct 提供**单端口工具 `vs`**（v3.0.0 动作集，严格本地化），同时接入两个宿主：
> **pi-coding-agent**（`/vs` 命令 + `vs` 工具）与 **Claude Code**（MCP server，`vs` 工具）。
> 两个宿主共享同一份核心，参数与 action 完全一致。
> 除 dom/a11y/check/setup 外，全部动作在内核沙箱（bwrap --unshare-net：零网络+只读根）内运行。
> 输出均为 schema v3 JSON（数字/坐标/hex，可直接推理）。全部本地、只读、无网络外发、**零第三方 API、零其他模型**。
> 本文件是完整命令参考——**遇到视觉任务时按此选择 action**。

## 一、任务 → action 决策表

| 任务 | action |
|---|---|
| 截屏 / 取色 / 颜色直方图 / 双图 diff / 对比度 | `pixels`（截图先 `capture`） |
| 图片/截图里的文字 + 精确坐标 | `ocr` |
| 网页 DOM 结构（无需截图的布局真值） | `dom`（需 URL） |
| PPTX 结构（形状/字体/颜色/坐标） | `pptx` |
| 任意截图图标级元素（无 DOM 也可） | `omniparser` |
| 文档/论文版式（标题/正文/图表/表格） | `layout` |
| PDF 文本块抽取（pt 坐标） | `pdf` |
| 桌面原生应用结构（角色/名称/屏幕坐标） | `a11y`（先 `list=true` 列应用） |
| **复杂背景图"看哪里"（显著性候选区）** | `saliency`（→ 再 `zoom`） |
| **前景/背景分离（复杂背景破解）** | `segment`（建议先 `saliency` 联动） |
| **单目深度（近/中/远分布）** | `depth_midas` |
| **亚像素边缘/精密测量** | `edge` |
| **纯文本 LLM 粗看（ASCII 栅格）** | `ascii` |
| **形状原语（矩形/圆/多边形）** | `geometry` |
| 自然图像开放词表物体检测 | `detect`（classes 必填） |
| 跨时间截图 diff | `analyze` task=diff-screenshots（传 input+compare） |
| 整页诊断管线（DOM+OCR+像素融合） | `analyze` task=diagnose-screenshot |
| **多传感器证据融合（D-S 理论）** | `fusion`（reports 必填） |
| **多轮协议：粗报告（候选+栅格+统计）** | `analyze`（image 必填） |
| **多轮协议：细报告（区域高倍+边缘+深度）** | `zoom`（image+region 必填） |
| **多轮协议：定向取证（单传感器）** | `probe`（image+bbox+sensor 必填） |
| 布局审计（重叠/出界/对比度） | `audit`（先有 report JSON） |
| 设计准则（对齐/间距/安全区/对比度） | `rules`（report JSON） |
| 3D 间隙/干涉审计 | `audit3d`（report 必填） |
| 相似图片分组（壁纸/截图聚类） | `cluster` |
| 环境自检 | `env` 或 `check`（Claude Code）/ `/vs check`（pi） |
| 安装/修复依赖 | `setup`（setup_args 传 --with-omniparser 等；Claude Code）/ `/vs setup`（pi） |

> **多轮反馈协议（SeeingEye 模式）**：复杂场景先 `analyze` 粗报告（saliency 候选区 + ascii 栅格 + 全局统计）
> → 从 candidates 选区域 `zoom` 细看（OCR 高倍 + edge 亚像素 + depth + segment 前景）
> → 对疑点 `probe` 定向取证（单传感器在 bbox 区域）。推理时直接用报告里的
> `[bbox: x1,y1,x2,y2]` / `[point: x,y]` 原语记法引用位置，不要用"左边的元素"这类相对方位描述。

## 二、action 参考

### 测量（pi-vision env，多数走 bwrap）
- `capture`：`out` 必填（PNG 路径）、可选 `region`。grim 截屏
- `pixels`：`image` 必填；`regions`/`compare`(diff)/`colors`/`wcag`/`threshold`(默认30)。**SLIC 超像素主色、精确 hex+百分比、ΔE、对比度**
- `ocr`：`image` 必填；`region`/`upscale`/`max_items`/`min_conf`/`backend`(rapidocr 默认 ~2-5s | paddle 高召回)/`preprocess`(contrast 低对比度用)/`daemon`。**text+4点bbox+conf+词典纠错**
- `wallpaper`：`dir` 必填；`colors`/`max_files`/`ext`
- `env`：环境自检（无参数）

### 结构（L0 + DL 感知）
- `dom`：`url` 必填；`max_elements`(默认60)、`screenshot`。**DOM+computed style：tag/role/text/bbox/color/font/z-index**
- `pptx`：`file` 必填；`max_shapes`、`slide`。**pt 坐标、填充 hex、字号/色**
- `omniparser`：`image` 必填；`max_items`、`no_ocr`。**图标级元素 + Florence-2 语义；本地常驻服务自动拉起**
- `layout`：`image` 必填；`max_items`、`min_conf`。**PP-DocLayoutV3 版式**
- `pdf`：`file` 必填；`pages`(如 1-3)、`render_dir`。**PyMuPDF 文本块(pt 坐标)**
- `a11y`：可选 `app`/`list=true`/`max_elements`(默认80)/`with_text`。**AT-SPI 无障碍树：原生应用的 DOM 等价物，screen_px 坐标**

### L2 轻量感知（vsensor env，GPU）
- `saliency`：`image` 必填；`top_n`(默认5)、`min_score`(默认0.3)、`device`(cuda/cpu)。**U²-Net 显著性 → top-N 候选区域 bbox+score（复杂背景"看哪里"）**
- `segment`：`image` 必填；`saliency`(联动报告 JSON，box 提示)、`device`。**MobileSAM 前景分割 → 前景 bbox+面积比+实例数**
- `depth_midas`：`image` 必填；`region`、`device`。**MiDaS 单目深度 → 近/中/远分布 + 区域深度**
- `edge`：`image` 必填；`max_lines`。**Devernay 亚像素边缘（像素值半值插值，精度 0.0000px）**
- `ascii`：`image` 必填；`cols`(默认64)、`rows`(默认36)、`color`。**多分辨率 ASCII 栅格（纯文本 LLM 粗看通道）**
- `geometry`：`image` 必填；`max_shapes`(默认50)。**VTracer SVG 化 → 形状原语（矩形/圆/多边形）**

### 融合（F1 引擎 + F2 协议；pi-vision env）
- `fusion`：`reports`(多传感器 JSON 路径) 必填。**D-S 证据融合：匈牙利匹配 → mass 组合 → belief/plausibility/uncertainty → verdict**
- `analyze`：`image` 必填（多轮协议粗报告）；或 `task` 必填（diagnose-screenshot/audit-pptx/classify-images/diff-screenshots）
- `zoom`：`image`+`region`(x1,y1,x2,y2) 必填。**细报告：区域 OCR 高倍 + edge 亚像素 + depth + segment 前景**
- `probe`：`image`+`bbox`+`sensor`(ocr/edge/depth/segment/pix) 必填。**定向取证：单传感器在 bbox 区域的结果**
- `audit`：`report` 必填；`canvas`(WxH)、`overlap_threshold`
- `rules`：`report` 必填；`canvas`、`align_tol`(默认4px)、`margin`。**R1对比度/R2重叠/R3对齐/R4间距/R5安全区**
- `audit3d`：`report`(blender_dump 场景图) 必填；`gap_threshold`(默认15mm)、`method`(auto/obb/mesh/aabb)。**3D 间隙/干涉，最大精度档（OBB-SAT+网格点云 KDTree，multiprocessing 并行）**

### 理解扩展
- `detect`：`image`+`classes`(逗号分隔开放词表) 必填；`threshold`(默认0.25)、`max_items`。**OWLv2 zero-shot 物体检测，image_px bbox；注意：只能检测提供的类别，类别词选择影响结果**

### 聚类（omniparser env）
- `cluster`：`dir` 或 `files`(逗号分隔)；`threshold`(默认0.75)、`max_files`。**CLIP ViT-B-32 输出 clusters[]+top_pairs[]**

## 三、输出解读要点

- 坐标系：schema v3 `coordsys` 字段（css_px/device_px/image_px/pt/screen_px）；跨动作比较先换算
- 颜色：hex + 百分比 + CIELAB ΔE（漂移判定 ΔE76，阈值默认 5）
- 对比度：WCAG gamma 校正，AA 要求 4.5:1（大字号 3:1）
- findings 一律带 `belief`/`plausibility`/`uncertainty`/`verdict`（confirmed/conflict/needs_review）+ `evidence`（来源传感器+原始数值）
- candidates（saliency 候选区）是 zoom-in 入口；foreground（segment 前景）是复杂背景分离结果
- 模型自身不能看图：**永远先调工具取结构化数据，再基于数字推理**

## 四、示例调用（JSON 参数形态）

> pi 宿主下用 `/vs` 命令或 `vs({...})` 工具；Claude Code 宿主下用 `vs({...})` 工具（MCP）。
> 两种形态参数完全一致。

```json
{"action": "capture", "out": "/tmp/vs_cap.png"}
{"action": "ocr", "image": "/tmp/vs_cap.png", "upscale": 2}
{"action": "pixels", "image": "/tmp/vs_cap.png", "colors": 8}
{"action": "saliency", "image": "/tmp/vs_cap.png", "top_n": 5}
{"action": "segment", "image": "/tmp/vs_cap.png", "saliency": "/tmp/sal.json"}
{"action": "edge", "image": "/tmp/vs_cap.png"}
{"action": "ascii", "image": "/tmp/vs_cap.png", "cols": 64, "rows": 36}
{"action": "analyze", "image": "/tmp/vs_cap.png"}
{"action": "zoom", "image": "/tmp/vs_cap.png", "region": "100,100,400,300"}
{"action": "probe", "image": "/tmp/vs_cap.png", "bbox": "100,100,400,300", "sensor": "ocr"}
{"action": "fusion", "reports": "/tmp/ocr.json /tmp/pix.json"}
{"action": "check"}
{"action": "setup", "setup_args": "--with-omniparser"}
```
