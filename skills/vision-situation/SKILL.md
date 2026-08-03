# Vision Situation Awareness（视觉态势感知 · schema v2）

结构化视觉引擎：让纯文本模型用**精确数字**（而非自然语言转述）理解图像。
信息只在**测量与融合**中流动；DeepSeek 是融合层的决策者。

## 核心原则

1. **能读源码就不读渲染图**：网页问题优先 `dom_dump`（DOM 是布局真值），PPT 优先 `pptx_dump`。
2. **测量优先于描述**：颜色取 hex、差异用像素 diff、文字用带坐标框的 OCR。
3. **融合优先于人工对照**：优先用 `vs_crosscheck`（三方互验）和 `vs_audit`（几何审计）自动检出异常——**不要自己把两份 JSON 对照**。
4. **不要凭截图猜数字**：所有坐标/颜色/字体来自工具 JSON，直接引用。
5. **首选任务引擎**：`vs_analyze --task <name>` 一条命令跑完传感器+融合，新任务=新配置。

## 工具

| 场景 | 工具 |
| --- | --- |
| 一键任务（首选） | `vs_analyze`（tasks: diagnose-screenshot / audit-pptx / classify-images） |
| 多源互验（DOM vs OCR vs 像素） | `vs_crosscheck`（颜色漂移 ΔE、文本缺失/多余、重叠） |
| 几何/样式审计（任意元素） | `vs_audit`（重叠、出界、WCAG 对比度） |
| 截图/区域 | `screen_capture` |
| 颜色/直方图/diff/WCAG | `pix_analyze` |
| 图上文字（带框） | `ocr_boxes` |
| 网页 DOM+computed style | `dom_dump` |
| PPTX 结构 | `pptx_dump` |
| 任意截图图标级元素（无需 DOM） | `vs_omniparser` | OmniParser V2（YOLO+Florence-2，CPU）；图标语义描述 + 文本；独立 conda env，首载慢 |
| 设计准则规则引擎 | `vs_rules` | 确定性规则：对比度/重叠/对齐漂移/间距一致/安全区；每个 finding 带证据 + suggested_cause + design_score；仅评估设计元素（dom/pptx），OCR 自然文本不误报 |
| CLIP 相似图聚类 | `vs_cluster` | ViT-B-32 CPU 离线；余弦阈值贪心分组（确定性）；输出 clusters[] + top_pairs[] + 相似度证据 |
| VLM-as-critic | `vs_critic` | 裁剪可疑区 → 本地 qwen3-vl 复核 → 裁决并入 findings（opt-in，--enable）；单区约 20s；全局属性缺陷（出界/安全区）裁剪视图会误判，需豁免 |
| 验收抽样集 | `bench/run_acceptance.py` | 12 确定性样本（6 缺陷+6 干净）；规则臂 100% 一致率；critic 臂 75% 确认率（实测，分歧含规则误报识别与裁剪局限） |
| 壁纸批量 | `wallpaper_classify` |
| 语义标签（L2 opt-in） | `semantic_tag`（本地 qwen3-vl，仅不可测属性） |
| 自检 | `vs_env_check` |

## Schema v2（统一元素模型）

```jsonc
{"schema": "vision-report/v2", "task": "...", "sensors": ["pix","ocr","dom"],
 "coordsys": "css_px | device_px | image_px | pt",
 "source": {"dpr": 1.0, "viewport_px": [...], "scroll": [...]},
 "elements": [{"id":0, "type":"text|button|panel|region|div...", "bbox":[x1,y1,x2,y2],
               "text":"...", "conf":0.99, "color":{"fill":"#FFF","text":"#111"},
               "font":{...}, "z":5, "source":["dom"], "coordsys":"css_px"}],
 "anomalies": [{"type":"color_drift|text_missing_in_ocr|text_not_in_dom|element_overlap|off_canvas|contrast_fail",
                "bbox":[...], "confidence":0.9,
                "evidence": {"dom_color":"#111111","pixel_color":"#8A898A","delta_e76":42.3},
                "suggested_cause":"..."}],
 "metrics": {"dominant_colors":[...], "brightness":230, "anomaly_count":2},
 "truncated": false}
```

- `bbox` 一律 `[x1,y1,x2,y2]`；DOM 是 `css_px`（含 `bbox_device_px`=×DPR），OCR/截图是 `image_px`，PPT 是 `pt`
- **anomalies 带证据链**：优先直接引用 evidence 数值，不要重新估算
- ΔE76 色差阈值参考：<1 不可感知，2-10 可感知，>10 明显，>50 巨大

## 典型工作流

### Firefox 显示异常诊断（融合闭环）

1. `screen_capture` 抓当前屏幕 → 得 PNG
2. `vs_analyze --task diagnose-screenshot --input <PNG> --url <页面URL>`：pix + ocr + dom + crosscheck 一条命令
   （无 URL 时自动跳过 DOM 步骤）
3. **读 `anomalies` 的证据**：`color_drift`（DOM 声明色 vs 像素实测 ΔE）、`text_missing_in_ocr`（渲染失败）
4. 依据证据定位并修复 → 修复后复截图 → `pix_analyze --compare` 验证 diff 归零

### PPT 优化

1. `vs_analyze --task audit-pptx --input <file.pptx>`：pptx_dump + vs_audit（重叠/出界/对比度自动检出）
2. 引用 evidence 中的 bbox/面积/ratio（"卡片B 与卡片A 重叠 72×72pt，对比度 3.95 < 4.5"）

### 壁纸/图片分类

1. `vs_analyze --task classify-images --input <目录>`（程序化：色相族/冷暖/亮度/饱和度/宽高比 + 分组）
2. 风格/主题等主观标签：`semantic_tag`（opt-in：`enable: true`；qwen3-vl 思考型，单张 1-2 分钟）

## L2 语义（opt-in）

- 只处理**不可测量**属性（风格/主题/类型）；颜色/坐标/尺寸一律走 L0/L1
- 默认关闭：不传 `enable` 时返回 `enabled:false`；只连 localhost:11434，无外发

## 注意

- 所有工具**只读**、本地、输出上限；`truncated=true` 时缩小范围重查
- 坐标系：截图/OCR=像素，PPT=pt（72pt=1in），DOM=css_px（乘 DPR 得设备像素）
- 工具失败返回 `{"error": ...}`，先读 error 再重试
- 实景截图存在每屏噪声（显示器亮度差异、亚像素、焦点态），跨截图 diff 需先亮度对齐；严格 diff 归零在受控渲染下断言
