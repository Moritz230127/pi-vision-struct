# vision-situation — 视觉技能：何时用哪个命令

> pi-vision-struct 扩展提供 4 个分组工具（`vs_measure` / `vs_struct` / `vs_fuse` / `vs_cluster`），
> **严格本地化**：除 dom/critic/semantic 外全部工具在内核沙箱（bwrap --unshare-net：零网络+只读根）内运行；
> 每个工具用 `action` 枚举选择子命令。输出均为 schema v2 JSON（数字/坐标/hex，可直接推理）。
> 全部本地、只读、无网络外发。本文件是完整命令参考——**遇到视觉任务时按此选择工具**。

## 一、任务 → 工具 决策表

| 任务 | 工具 |
|---|---|
| 截屏 / 取色 / 颜色直方图 / 双图 diff / 对比度 | `vs_measure` action=pixels（截图先 action=capture） |
| 图片/截图里的文字 + 精确坐标 | `vs_measure` action=ocr |
| 网页 DOM 结构（无需截图的布局真值） | `vs_struct` action=dom（需 URL） |
| PPTX 结构（形状/字体/颜色/坐标） | `vs_struct` action=pptx |
| 任意截图图标级元素（无 DOM 也可） | `vs_struct` action=omniparser |
| 文档/论文版式（标题/正文/图表/表格） | `vs_struct` action=layout |
| 整页诊断管线（DOM+OCR+像素融合） | `vs_fuse` action=analyze task=diagnose-screenshot |
| DOM↔OCR↔像素三方互验 | `vs_fuse` action=crosscheck |
| 布局审计（重叠/出界/对比度） | `vs_fuse` action=audit（先有 report JSON） |
| 设计准则（对齐/间距/安全区/对比度） | `vs_fuse` action=rules（report JSON 或元素报告） |
| 对可疑 finding 做 VLM 复核 | `vs_fuse` action=critic（opt-in enable，慢） |
| 相似图片分组（壁纸/截图聚类） | `vs_cluster` |
| 环境自检 | `vs_measure` action=env 或 `/vs check` |

## 二、命令参考

### vs_measure（本地测量/感知；pi-vision env）
- `capture`：Wayland 截屏 → `out`（PNG 路径必填）、`region`（x1,y1,x2,y2 可选）
- `pixels`：`image` 必填；`regions`(逗号分隔 x1,y1,x2,y2)、`compare`(diff 对比图)、`colors`(主色数)、`wcag`(前景hex,背景hex 逗号分隔)、`threshold`(diff 阈值 默认30)。**返回精确数字：hex+百分比、ΔE、对比度**
- `ocr`：`image` 必填；`region`/`upscale`/`max_items`/`min_conf`/`backend`/`preprocess`(contrast 低对比度用)。**双后端：rapidocr**(默认, 快 ~2-5s) / **paddle**(PP-OCRv6 medium, 慢 ~7-40s)；返回 text+4点 bbox+conf
- `wallpaper`：`dir` 必填；`colors`/`max_files`/`ext`/`semantic`(opt-in L2)
- `semantic`：`image` 必填；`enable=true` 才执行（L2 有思考成本）；`prompt` 可自定义
- `env`：环境自检（无参数）

### vs_struct（L0 源码 + DL 图标；dom/pptx 用 pi-vision，omniparser 用独立 env）
- `dom`：`url` 必填；`max_elements`(默认60)、`screenshot`(会话截图路径)。**DOM+computed style：tag/role/text/bbox/color/font/z-index**
- `pptx`：`file` 必填；`max_shapes`、`slide`。**pt 坐标、填充 hex、字号/色**
- `omniparser`：`image` 必填；`max_items`、`no_ocr`。**图标级元素 + Florence-2 语义描述；走本地常驻服务（模型驻留：稀疏图 ~4s，密集图 ~37s；服务自动拉起，无需手动管理）**
- `layout`：`image` 必填；`max_items`、`min_conf`。**文档版式 PP-DocLayoutV3：标题/正文/图表/表格区域；首次下模型 ~30MB**

### vs_fuse（确定性融合/准则；pi-vision env）
- `analyze`：`task` 必填（diagnose-screenshot/audit-pptx/classify-images）；`input`/`url`/`dpr` 可选。**多步传感器+融合合并报告**
- `crosscheck`：`image` 必填；`dom`/`ocr`(报告 JSON 路径)、`dpr`、`color_threshold`(ΔE 默认5)。**颜色漂移/文本互验/重叠**
- `audit`：`report` 必填（pptx/dom 报告）；`canvas`(WxH)、`overlap_threshold`
- `rules`：`report` 必填；`canvas`、`align_tol`(默认4px)、`margin`(默认2px)。**R1 对比度/R2 重叠/R3 对齐漂移/R4 间距/R5 安全区；仅评估设计元素，OCR 自然文本不误报；每个 finding 带证据+阈值**
- `critic`：`report`+`image` 必填；`enable=true` 才调 VLM；`max_critic`(默认8)、`margin`。**裁剪可疑区 → 本地 qwen3-vl 裁决 → 证据并入 finding。注意：出界/安全区等全局属性缺陷在裁剪视图会误拒，需豁免或结合 rules 证据解读**

### vs_cluster（CLIP 离线聚类；omniparser env）
- `dir` 或 `files`(逗号分隔)；`threshold`(默认0.75 越大越细)、`max_files`。**输出 clusters[]（代表+成员相似度）+ top_pairs[]；确定性**

## 三、输出解读要点

- 坐标系：schema v2 `coordsys` 字段（css_px/device_px/image_px/pt）；跨工具比较先换算
- 颜色：hex + 百分比 + CIELAB ΔE（漂移判定用 ΔE76，阈值默认 5）
- 对比度：WCAG gamma 校正公式，AA 要求 4.5:1（大字号 3:1）
- findings/anomalies 一律带 `evidence`（数值+阈值）与 `suggested_cause`——推理时引用具体数值而非泛泛而谈
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
