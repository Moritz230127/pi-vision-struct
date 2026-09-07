# vision-situation — 视觉技能：何时用哪个动作

> pi-vision-struct **V4 全面 Rust 化**：单二进制 `vs-bin`（clap 子命令 = action），
> 同时接入两个宿主：**pi-coding-agent**（`extensions/index.ts` exec 胶水）与
> **Claude Code**（Rust MCP server `vs-mcp`）。
> 全部动作本地、只读、无网络外发（dom/a11y 除外）、**零第三方 API、零其他模型**。
> 输出均为 schema v3 JSON（数字/坐标/hex，可直接推理）。
> 本文件是完整命令参考——**遇到视觉任务时按此选择 action**。

## 一、任务 → action 决策表

| 任务 | action（vs-bin 子命令） |
| --- | --- |
| 截屏 / 取色 / 颜色直方图 / 双图 diff / 对比度 | `pix`（截图先 `capture`） |
| 图片/截图里的文字 + 精确坐标 | `ocr` |
| 网页 DOM 结构（无需截图的布局真值） | `dom`（需 URL） |
| PPTX 结构（形状/字体/颜色/坐标） | `pptx` |
| 任意截图图标级元素（无 DOM 也可） | `omniparser` |
| 文档/论文版式（标题/正文/图表/表格） | `layout` |
| PDF 文本块抽取（pt 坐标） | `pdf` |
| 桌面原生应用结构（角色/名称/屏幕坐标） | `a11y`（先 `--list` 列应用） |
| **复杂背景图"看哪里"（显著性候选区）** | `saliency`（→ 再 zoom 细看） |
| **前景/背景分离（复杂背景破解）** | `segment`（建议先 `saliency` 联动） |
| **单目深度（近/中/远分布）** | `depth` |
| **亚像素边缘/精密测量** | `edge` |
| **纯文本 LLM 粗看（ASCII 栅格）** | `ascii` |
| **形状原语（矩形/圆/多边形）** | `geometry` |
| 自然图像开放词表物体检测 | `detect`（`--classes` 必填） |
| **多传感器证据融合（D-S 理论）** | `fusion`（`--reports` 必填） |
| 布局审计（重叠/出界/对比度） | `audit`（先有 report JSON） |
| 设计准则（对齐/间距/安全区/对比度） | `rules`（report JSON） |
| 3D 场景图（.blend 解析） | `blend`（`--file` 必填） |
| 环境自检 | `schema`（输出契约验证） |

> **多轮反馈协议（SeeingEye 模式）**：复杂场景先 `saliency` 粗报告（候选区 + 全局统计）
> → 从 candidates 选区域 zoom 细看（OCR 高倍 + edge 亚像素 + depth + segment 前景）
> → 对疑点 probe 定向取证（单传感器在 bbox 区域）。推理时直接用报告里的
> `[bbox: x1,y1,x2,y2]` / `[point: x,y]` 原语记法引用位置，不要用"左边的元素"这类相对方位描述。

## 二、action 参考（vs-bin 子命令）

### 测量

| 子命令 | 参数 | 说明 |
| --- | --- | --- |
| `capture` | `--out` 必填 | Wayland 截屏（xcap，10s 超时保护） |
| `pix` | `--image` + `--colors`/`--regions`/`--compare`/`--wcag` | SLIC 超像素主色、区域取色、双图 diff、WCAG 对比度 |
| `ocr` | `--image` + `--region`/`--upscale`/`--max-items`/`--min-conf` | 文字 + 4点bbox + conf + 词典纠错 |
| `scene_stats` | `--image` + `--region`/`--colors` | 真 Otsu 面积比、颜色直方图、对比度统计、空间布局 |

### 结构（L0 源码 + DL 感知）

| 子命令 | 参数 | 说明 |
| --- | --- | --- |
| `dom` | `--url` + `--max-elements` | DOM+computed style（chromiumoxide） |
| `pptx` | `--file` + `--max-shapes`/`--slide` | shapes/字体/颜色/pt 坐标（OOXML 自研解析） |
| `omniparser` | `--image` + `--max-items` | 图标级元素（YOLOv9 + Florence-2 生成循环） |
| `layout` | `--image` + `--max-items`/`--min-conf` | 文档版式（PP-DocLayoutV3，Python 回退） |
| `pdf` | `--file` + `--pages` | PDF 文本块抽取（pdfium-render） |
| `a11y` | `--app`/`--max-elements`/`--with-text` | 无障碍树（atspi，screen_px） |
| `detect` | `--image` + `--classes`(逗号分隔开放词表) | OWLv2 zero-shot 物体检测（三 ONNX + 自研 NMS） |

### L2 轻量感知

| 子命令 | 参数 | 说明 |
| --- | --- | --- |
| `saliency` | `--image` + `--top-n`/`--min-score` | U²-Net 显著性 → top-N 候选区域 |
| `segment` | `--image` + `--saliency`(联动) | MobileSAM 前景分割（Rust encoder + Python decoder 回退） |
| `depth` | `--image` + `--region` | MiDaS 单目深度（近/中/远分布） |
| `edge` | `--image` + `--max-lines` | Devernay 亚像素边缘（精度 0.0000px） |
| `ascii` | `--image` + `--cols`/`--rows`/`--color` | 多分辨率 ASCII 栅格 |
| `geometry` | `--image` + `--max-shapes` | VTracer SVG 化 → 形状原语 |

### 三维 + 融合

| 子命令 | 参数 | 说明 |
| --- | --- | --- |
| `blend` | `--file` 必填 | .blend 场景图（blend-rs 解析） |
| `fusion` | `--reports`(多传感器 JSON) | D-S 证据融合：匈牙利匹配 → mass 组合 → belief/plausibility |
| `audit` | `--report` + `--canvas`/`--overlap-threshold` | 2D 重叠/出界/对比度 |
| `rules` | `--report` + `--canvas`/`--align-tol`/`--margin` | 设计准则（对齐/间距/安全区） |

## 三、构建与验证

```bash
cd rust && cargo build --release -p vs-bin   # 单二进制
./target/release/vs-bin <子命令> --help      # 参数参考
cargo test --workspace                       # 全量测试（105+ 全绿）
```

> 模型权重本地缓存：`~/.cache/vsensor/`（u2net / mobile_sam / midas / rapidocr / florence2 / owlv2）。
> 仅 Linux；layout / detect 首次需联网下载权重。
