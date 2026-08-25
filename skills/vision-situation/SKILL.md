# vision-situation — 视觉技能：何时用哪个动作

> pi-vision-struct 扩展提供**单端口工具 `vs`**（18 个 action，严格本地化）：
> 除 dom/critic/semantic 外全部动作在内核沙箱（bwrap --unshare-net：零网络+只读根）内运行。
> 输出均为 schema v2 JSON（数字/坐标/hex，可直接推理）。全部本地、只读、无网络外发。
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
| 跨时间截图 diff | `analyze` task=diff-screenshots（传 input+compare） |
| 整页诊断管线（DOM+OCR+像素融合） | `analyze` task=diagnose-screenshot |
| DOM↔OCR↔像素三方互验 | `crosscheck` |
| 布局审计（重叠/出界/对比度） | `audit`（先有 report JSON） |
| 设计准则（对齐/间距/安全区/对比度） | `rules`（report JSON） |
| 对可疑 finding 做 VLM 复核 | `critic`（enable=true，慢） |
| 相似图片分组（壁纸/截图聚类） | `cluster` |
| 环境自检 | `env` 或 `/vs check` |

> **高分辨率 zoom-in 工作流**：先整图 `ocr`/`pixels` 粗扫 → 对疑点区域带
> `region=x1,y1,x2,y2` 重跑同动作细查（小字叠加 `upscale=3`）。两轮坐标同一坐标系，
> 推理时直接用报告里的 `[bbox: x1,y1,x2,y2]` / `[point: x,y]` 原语记法引用位置，
> 不要用"左边的元素"这类相对方位描述。

## 二、action 参考

### 测量（pi-vision env，多数走 bwrap）
- `capture`：`out` 必填（PNG 路径）、可选 `region`。grim 截屏
- `pixels`：`image` 必填；`regions`/`compare`(diff)/`colors`/`wcag`/`threshold`(默认30)。**精确 hex+百分比、ΔE、对比度**
- `ocr`：`image` 必填；`region`/`upscale`/`max_items`/`min_conf`/`backend`(rapidocr 默认 ~2-5s | paddle 高召回，默认走常驻服务 ocrserver 自动拉起、单图亚秒级)/`preprocess`(contrast 低对比度用)/`daemon`(auto|always|never)。**text+4点bbox+conf**
- `wallpaper`：`dir` 必填；`colors`/`max_files`/`ext`/`semantic`(opt-in)
- `semantic`：`image` 必填；`enable=true` 才执行（L2 有思考成本）；`prompt` 可自定义
- `env`：环境自检（无参数）

### 结构（L0 + DL 感知）
- `dom`：`url` 必填；`max_elements`(默认60)、`screenshot`。**DOM+computed style：tag/role/text/bbox/color/font/z-index**
- `pptx`：`file` 必填；`max_shapes`、`slide`。**pt 坐标、填充 hex、字号/色**
- `omniparser`：`image` 必填；`max_items`、`no_ocr`。**图标级元素 + Florence-2 语义；本地常驻服务自动拉起（稀疏图 ~4s）**
- `layout`：`image` 必填；`max_items`、`min_conf`。**PP-DocLayoutV3 版式；首次下模型 ~30MB**
- `pdf`：`file` 必填；`pages`(如 1-3)、`render_dir`。**PyMuPDF 文本块(pt 坐标)**
- `a11y`：可选 `app`/`list=true`/`max_elements`(默认80)/`with_text`。**AT-SPI 无障碍树：原生应用的 DOM 等价物，screen_px 坐标**

### 融合（确定性；pi-vision env）
- `analyze`：`task` 必填（diagnose-screenshot/audit-pptx/classify-images/diff-screenshots）；`input`/`url`/`dpr` 可选
- `crosscheck`：`image` 必填；`dom`/`ocr` 报告 JSON、`dpr`、`color_threshold`(ΔE 默认5)。**颜色漂移/文本互验/重叠**
- `audit`：`report` 必填；`canvas`(WxH)、`overlap_threshold`
- `rules`：`report` 必填；`canvas`、`align_tol`(默认4px)、`margin`。**R1对比度/R2重叠/R3对齐/R4间距/R5安全区**
- `critic`：`report`+`image` 必填；`enable=true` 才调 VLM；`max_critic`(默认8)、`margin`。**裁剪可疑区 → qwen3-vl 裁决。注意：出界/安全区等全局属性缺陷在裁剪视图会误拒**

### 聚类（omniparser env）
- `cluster`：`dir` 或 `files`(逗号分隔)；`threshold`(默认0.75)、`max_files`。**CLIP ViT-B-32 输出 clusters[]+top_pairs[]；首次下模型 ~350MB 后离线**

## 三、输出解读要点

- 坐标系：schema v2 `coordsys` 字段（css_px/device_px/image_px/pt/screen_px）；跨动作比较先换算
- 颜色：hex + 百分比 + CIELAB ΔE（漂移判定 ΔE76，阈值默认 5）
- 对比度：WCAG gamma 校正，AA 要求 4.5:1（大字号 3:1）
- findings/anomalies 一律带 `evidence`（数值+阈值）、`primitive`（[bbox:…] 记法）与 `suggested_cause`
- 模型自身不能看图：**永远先调工具取结构化数据，再基于数字推理**

## 四、示例调用（JSON 参数形态）

```json
{"action": "capture", "out": "/tmp/vs_cap.png"}
{"action": "ocr", "image": "/tmp/vs_cap.png", "upscale": 2}
{"action": "pixels", "image": "/tmp/vs_cap.png", "colors": 8}
{"action": "analyze", "task": "diagnose-screenshot", "input": "/tmp/vs_cap.png"}
{"action": "rules", "report": "/tmp/report.json", "canvas": "1920x1067"}
{"action": "critic", "report": "/tmp/report.json", "image": "/tmp/vs_cap.png", "enable": true, "max_critic": 4}
```
