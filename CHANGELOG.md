# Changelog

## 0.2.0 — 2026-08-03（分层无损 v2 + Phase 2.1–2.4）

### 架构 v2（vision-report/schema v2）

- 统一元素模型 `{id,type,bbox[x1,y1,x2,y2],text,conf,color,font,z,source[],coordsys}`；坐标系 css_px/device_px/image_px/pt + 转换
- 分层无损通道：L0 源码（DOM/PPTX）→ L1 确定性测量（颜色/OCR/diff/WCAG）→ L2 语义（opt-in）

### 新传感器/算子

- `vs_schema`：元素模型、CIELAB ΔE76、WCAG gamma 对比度、bbox 几何
- `vs_crosscheck`：颜色漂移（密集采样最小 ΔE）、OCR↔DOM 文本互验、元素重叠
- `vs_audit`：重叠/出界/对比度审计
- `vs_analyze`：配置驱动任务引擎（3 内置任务）+ `vs_omniparser`（OmniParser V2，CPU，图标级元素 + Florence-2 语义）
- `vs_rules`：确定性设计准则规则引擎（对比度/重叠/对齐漂移/间距一致/安全区；仅评估设计元素，OCR 不误报）
- `vs_cluster`：CLIP 离线相似图聚类（确定性）
- `vs_critic`：VLM-as-critic 闭环（裁剪可疑区 → 本地 qwen3-vl 复核 → 裁决并入 evidence；opt-in）

### 验证

- 基准套件 `bench/`：颜色 ΔE=0.0、OCR CER=0.0、diff 定位 recall=1.0、漂移检出 ΔE=76.3
- 验收抽样集 `bench/samples`（12 确定性样本）：规则臂一致率 100%、critic 臂 75%（分歧已实证解释）
- 全量回归：17+8+3+19+7+10 = 64 断言全绿
- 真实数据：Firefox 截图（OCR 60 项 / OmniParser 101 元素）、46 张壁纸分类与聚类

### 分发

- 引导安装器 `/vs setup`（conda env 检测/创建、依赖、自测；--check/--dry-run/--with-omniparser/--with-dom）
- docs/omniparser-setup.md（OmniParser 环境维护手册：版本钉死、HF 缓存坑、config 补丁）

## 0.1.0 — 2026-08-02（v1 传感器）

- 基础工具：vs_env_check、screen_capture（grim）、pix_analyze、ocr_boxes（RapidOCR PP-OCRv6）、dom_dump（Playwright Firefox）、pptx_dump、wallpaper_classify、semantic_tag（L2 opt-in）
- 自测 17/17；真实场景验证（Firefox 截图、文件截图、46 张壁纸）

### 待发布前补

- repository 字段（git 仓库 URL，需创建仓库后填入）
