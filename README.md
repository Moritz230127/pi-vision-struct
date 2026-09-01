# pi-vision-struct

**给纯文本模型的数理化视觉通道** —— 全部输出为精确数值（坐标 / hex / 矩阵 / 统计），零主观描述。
DeepSeek 等纯文本 LLM 不需要"看懂"图片，它需要**最高精度的数理数据**来做**数值推理**。

> **平台：仅 Linux x86_64。** 全部工具只读、本地、无网络外发（`a11y` 的会话 DBus 除外）；
> 多数动作在 bwrap 内核沙箱中运行。**零第三方 API、零其他模型**（v3.0.0 起移除 qwen3-vl/ollama 依赖）。

---

## 一、原理：为什么必须是数理化

### 1.1 核心命题

> 图片不是"看"的对象，是"转化为数理数据"的原材料。
> LLM 多模态本身不具备工程级能力——因为多模态是在"猜"，而工程需要"算"。

传统视觉扩展让 VLM 输出"一张风景照"、"一个停车场"、"扁平 / 科技风格"——这是**主观噪声**。
DeepSeek 在文本推理上是最强模型之一，它不需要别人替它"理解"图片，它需要：

- `area_ratio = 0.4967` 而非"约一半面积"
- `contrast_ratio = 2.09` 而非"对比度偏低"
- `matrix_world = [16 个数]` 而非"在机匣右侧"
- `gap_mm = -0.02` 而非"离得很近"

**传感器输出主观描述 = 给最强推理引擎喂噪声。传感器输出精确数值 = 让最强推理引擎做它最擅长的事。**

### 1.2 数理化分层模型（v3.0.0）

```
物理世界（像素 / 网格 / 顶点 / 法线 / 材质）
         │
    ┌────┼────────────────────────────┐
    │ L0 源层                          │
    │ dom(Playwright) · pptx · pdf    │
    │ a11y(AT-SPI) · capture · 文件    │
    └────┼────────────────────────────┘
         │
    ┌────▼────────────────────────────┐
    │ L1 确定性测量（纯算法）           │
    │ pixels → hex, ΔE (SLIC超像素)    │
    │ ocr → text, bbox[4] (词典纠错)   │
    │ scene_stats → Otsu/直方图/空间    │
    │ edge → 亚像素边缘 (0.0000px)      │
    │ ascii → 多分辨率文本栅格          │
    │ geometry → VTracer 形状原语      │
    └────┼────────────────────────────┘
         │
    ┌────▼────────────────────────────┐
    │ L2 轻量感知（≤25M 参数，GPU）     │
    │ saliency → U²-Net 候选区域       │
    │ segment → MobileSAM 前景分割     │
    │ depth → MiDaS 单目深度           │
    │ detect → OWLv2 开放词表检测      │
    └────┼────────────────────────────┘
         │
    ┌────▼────────────────────────────┐
    │ F1 融合引擎（纯算法）             │
    │ align → 匈牙利全局匹配           │
    │ evidence → D-S mass 映射         │
    │ combine → 正交和组合             │
    │ decide → belief/plausibility    │
    └────┬────────────────────────────┘
         │
    ┌────▼────────────────────────────┐
    │ F2 推理接口（schema v3）         │
    │ analyze → 粗报告（候选+栅格+统计）│
    │ zoom → 细报告（区域高倍+边缘+深度）│
    │ probe → 单传感器定向证据         │
    └────┬────────────────────────────┘
         │
    ┌────▼────────────────────────────┐
    │ DeepSeek（文本推理）             │
    │ 输入：上述全部数值               │
    │ 推理：数值逻辑 + 多轮反馈         │
    │ 输出：可执行数值建议             │
    └─────────────────────────────────┘
```

**关键区别（v2 → v3）：**

| v2.2.2 | v3.0.0 |
|---|---|
| `crosscheck` 硬阈值（IoU>0.05, ΔE>5.0） | `fusion` **D-S 证据理论**（mass 组合 + belief/plausibility/uncertainty） |
| 朴素 max-IoU 匹配 | **匈牙利全局匹配**（scipy linear_sum_assignment） |
| MEDIANCUT 颜色量化 | **SLIC 超像素 + K-means** |
| median 伪 Otsu | **真 Otsu 多阈值** |
| 纯 Python BFS 连通域 | **numpy 向量化**（157ms） |
| L2 语义层（qwen3-vl:8b） | **移除**（零其他模型约束） |
| 无显著性/分割/深度传感器 | **saliency/segment/depth**（U²-Net/MobileSAM/MiDaS） |
| 无调度 | **vsched**（显存预算 + 并发 + 功耗感知 + CPU 降级） |
| schema v2 | **schema v3**（findings/candidates/foreground/evidence/uncertainty） |

### 1.3 精度保证

- **坐标系**：`coordsys` 字段声明（`css_px` / `device_px` / `image_px` / `screen_px` / `world_m` / `pt`），跨动作比较先用 schema 变换换算
- **单位**：全部数值带单位（mm / px / hex / conf 0-1 / WCAG 比值）
- **可复算**：每个输出都能在原图/原模型上用独立代码验证
- **零幻觉**：数值主链无 VLM 热路径；融合层全部纯算法（D-S 证据理论）

---

## 二、架构

### 2.1 单端口工具（v3.0.0 动作集）

全部能力收敛为单一注册工具 `vs`，`action` 枚举分发到内部分发表：

| 分发表 | 动作 | 传感器/引擎 |
|---|---|---|
| MEASURE | `capture` `pixels` `ocr` `wallpaper` `scene_stats` `env` | PIL / rapidocr / paddle / skimage |
| STRUCT | `dom` `pptx` `omniparser` `layout` `pdf` `a11y` `detect` | playwright / CLIP / OWLv2 |
| 3D | `blender_dump` `depth` `depth_geom` `audit3d` | bpy headless / 真几何深度 / OBB-SAT+KDTree |
| L2 感知 | `saliency` `segment` `depth_midas` `edge` `ascii` `geometry` | U²-Net / MobileSAM / MiDaS / Devernay / VTracer |
| FUSE | `analyze` `fusion` `audit` `rules` `zoom` `probe` | D-S 证据理论 / 匈牙利匹配 / 多轮协议 |

> 单端口设计使上下文注入最小化，且分发表即真相——新增动作 = 表加一行 + 枚举加一项。

### 2.2 调度管理器（vsched + vsd）

8GB VRAM 硬约束下，任何并发 GPU 推理都可能 OOM。`vsched` 是 GPU/CPU 推理请求的唯一入口：

- **显存预算表**：每模型注册 `vram_reserve` + `vram_peak`，总预算 6.5GB（留 1.5GB 安全水位）
- **并发信号量**：GPU 推理槽位 = 1（串行），杜绝同刻双 GPU 推理
- **模型 LRU 淘汰**：最近未用模型卸载（冷载 1-3s / 热 <100ms）
- **优先级队列**：analyze > zoom > probe > bg
- **功耗感知**：检测电源状态（battery/plugged），电池模式大模型自动 CPU 降级，接电源全速
- **CPU 降级**：显存紧张时自动 CPU 推理，任务永不因 OOM 失败
- **子进程隔离**：每个推理请求独立子进程，用完即释放显存（并发 5 任务显存峰值 816MB）

### 2.3 自稳定层

- **preflight**：每次调用前两级预检（解释器可执行 → 核心依赖可导入）；ok 进程内缓存，失败 30s 后可重试
- **常驻服务**：`vsd`（统一 daemon，unix socket 行 JSON 协议）+ `omniserver` + `ocrserver`，自动拉起、崩溃自动清理
- **git 完整性**：`/vs check` 输出 tag/commit 与未提交改动计数

### 2.4 安全

- bwrap `--unshare-net`（零网络）+ 系统根只读；可写白名单仅 `/tmp`、`$HOME`、`~/.cache/*`、`$XDG_RUNTIME_DIR`
- 豁免项均注明原因：`dom`（加载用户 URL）、`a11y`（会话 DBus）
- 依赖版本冻结，`pip check` 零冲突；`VS_NO_SANDBOX=1` 可关闭沙箱

### 2.5 双宿主接入（pi-coding-agent + Claude Code）

代码已重架构为**双宿主一等扩展**：框架无关核心 `extensions/pi-vision-core.ts` 是唯一事实源，
pi 与 Claude Code 各写一个薄适配器壳，复用同一份 action 与同一份 Python 传感器（零改动）。

| 宿主 | 入口 | 接入方式 |
|---|---|---|
| **pi-coding-agent** | `extensions/index.ts`（`@earendil-works/pi-coding-agent` 的 `ExtensionAPI`） | `package.json` 的 `pi.extensions` 自动加载 |
| **Claude Code** | `extensions/server.mcp.ts`（MCP stdio server） | `claude mcp add pi-vision-struct -- node dist/extensions/server.mcp.js` |

详见 **[docs/claude-code.md](docs/claude-code.md)**。

---

## 三、工具参考

### 测量（pi-vision env）

| 动作 | 参数 | 说明 |
|---|---|---|
| `capture` | `out` 必填 | Wayland 截屏（grim）|
| `pixels` | `image` + `colors`/`regions`/`compare`/`wcag`/`threshold` | SLIC 超像素主色、区域取色、双图 diff（向量化连通域）、WCAG 对比度 |
| `ocr` | `image` + `region`/`upscale`/`max_items`/`min_conf`/`backend`/`preprocess`/`daemon` | 文字 + 4点bbox + conf + **词典纠错后处理** |
| `scene_stats` | `image` + `region`/`colors` | 真 Otsu 面积比、颜色直方图、对比度统计、空间布局 |
| `wallpaper` | `dir` + `colors`/`max_files`/`ext` | 壁纸分类 |
| `env` | — | 环境自检 |

### 结构（L0 源码 + DL 感知）

| 动作 | 参数 | 说明 |
|---|---|---|
| `dom` | `url` + `max_elements`/`screenshot` | DOM+computed style，真值 |
| `pptx` | `file` + `max_shapes`/`slide` | shapes/字体/颜色/pt 坐标 |
| `omniparser` | `image` + `max_items`/`no_ocr` | 图标级元素（YOLOv9 + Florence-2）|
| `layout` | `image` + `max_items`/`min_conf` | 文档版式（PP-DocLayoutV3）|
| `pdf` | `file` + `pages`/`render_dir` | PDF 文本块抽取 |
| `a11y` | `list`/`app`/`max_elements`/`with_text` | 无障碍树（AT-SPI，screen_px）|
| `detect` | `image` + `classes`(逗号分隔开放词表) | OWLv2 zero-shot 物体检测 |

### L2 轻量感知（vsensor env，GPU）

| 动作 | 参数 | 说明 |
|---|---|---|
| `saliency` | `image` + `top_n`/`min_score`/`device` | U²-Net 显著性 → top-N 候选区域（破解复杂背景 zoom-in 死锁）|
| `segment` | `image` + `saliency`(联动)/`device` | MobileSAM 前景分割（saliency bbox 提示）|
| `depth_midas` | `image` + `region`/`device` | MiDaS 单目深度（近/中/远分布）|
| `edge` | `image` + `max_lines` | Devernay 亚像素边缘（像素值半值插值，精度 0.0000px）|
| `ascii` | `image` + `cols`/`rows`/`color` | 多分辨率 ASCII 栅格（纯文本 LLM 粗看通道）|
| `geometry` | `image` + `max_shapes` | VTracer SVG 化 → 形状原语（矩形/圆/多边形）|

### 三维（Blender 数理化）

| 动作 | 参数 | 说明 |
|---|---|---|
| `blender_dump` | `blend` 必填 | Blender 场景图：`objects3d[]`（matrix_world 4×4 + bbox3d 8×3 + collection/material/visible）+ cameras |
| `depth` | `image` 必填 | 深度矩阵统计（near/far/median/std/histogram/center_depth）|
| `depth_geom` | `blend`+`camera`/`image` | **真几何深度**：Blender 内栅格化相机空间距离（米）|
| `audit3d` | `report` + `gap_threshold`(默认15mm) | OBB-SAT + 网格级 KDTree 干涉/间隙检测（multiprocessing 并行）|

### 融合（F1 引擎 + F2 协议）

| 动作 | 参数 | 说明 |
|---|---|---|
| `fusion` | `reports`(多传感器 JSON) | **D-S 证据融合**：匈牙利匹配 → mass 组合 → belief/plausibility/uncertainty |
| `analyze` | `image` | **粗报告**：saliency 候选 + ascii 栅格 + scene_stats + ocr + 融合 findings |
| `zoom` | `image` + `region` | **细报告**：区域 OCR 高倍 + edge 亚像素 + depth + segment 前景 |
| `probe` | `image` + `bbox` + `sensor` | **定向取证**：单传感器在 bbox 区域的结果 |
| `audit` | `report` + `canvas`/`overlap_threshold` | 2D 重叠/出界/对比度 |
| `rules` | `report` + `canvas`/`align_tol`/`margin` | 设计准则（对齐/间距/安全区）|

#### 融合引擎（D-S 证据理论）

对多传感器输出做证据融合：

1. **align**：匈牙利算法全局最优匹配（代价 = 1 - IoU - α·text_sim）
2. **evidence**：传感器输出 → mass 函数（OCR conf / ΔE 似然 / 检测置信度 / 前景面积比 / 显著性分数 / 深度一致性）
3. **combine**：D-S 正交和组合 `m₁₂(A) = Σ_{B∩C=A} m₁(B)m₂(C) / (1-K)`
4. **decide**：belief>0.6 → confirmed；plausibility<0.4 → conflict；K>0.7 → needs_review

> 实证：边界场景（ΔE∈[3,8] 正常 vs [8,20] 异常），v2 硬阈值误报 70% → v3 D-S 融合零误报。

#### 多轮反馈协议（SeeingEye 模式）

```
DeepSeek ──vs analyze──→ 粗报告（saliency 候选 + ascii 粗栅格 + 全局统计）
DeepSeek 选候选区 ──vs zoom [region]──→ 细报告（ocr 高倍 + segment 前景 + edge 亚像素）
DeepSeek 追问 ──vs probe [bbox, sensor]──→ 单传感器定向证据
DeepSeek 综合推理 → 最终答案
```

---

## 四、验证

| 场景 | 指标 | 结果 |
|---|---|---|
| 融合回归 | 10 断言（D-S 数学/匈牙利/冲突/确定性） | 全部通过 |
| 全功能测试 | 31 项（L0/L1/L2/F1/F2/3D/调度/自诊断） | 31/31 PASS |
| 亚像素边缘 | 5 偏移场景（0.3/0.5/0.7/1.3px） | 误差 0.0000px |
| 复杂背景 | saliency 候选区命中有效内容 | score 0.98（精确命中）|
| 前景分割 | MobileSAM + saliency 联动 | IoU 精确（area_ratio 0.0833=理论值）|
| 并发调度 | 5 任务并发 | 显存峰值 816MB，零 OOM |
| 全流水线 | analyze（saliency+ascii+stats+ocr+fusion） | 2.9s（GPU）|
| 涡扇 3D | 196 物体 / 16110 对 | 4 真干涉 + 1056 已验证间隙 |

```bash
mamba run -n pi-vision python -u tests/test_fusion.py
mamba run -n pi-vision python -u tests/run_full_test.py
```

---

## 五、环境

三 conda env 隔离（互不污染）：

| env | 内容 | 用途 |
|---|---|---|
| `pi-vision` | python 3.12 + rapidocr/paddle/playwright/scipy | L1 确定性测量 + F1 融合 |
| `vsensor` | GPU torch cu124 + skimage + onnxruntime-gpu + vtracer + mobile-sam | L2 轻量 DL 传感器 |
| `omniparser` | CPU torch + OmniParser/OWLv2/CLIP | detect/omniparser/cluster |

模型权重（本地缓存，离线可用）：`~/.cache/vsensor/`（u2net.onnx 176MB / mobile_sam.pt 40MB / midas_v21_small_256.onnx 66MB）

---

## 六、已知限制

- 点云距离精度取决于导出顶点密度（默认每物体 ≤4000 点下采样）
- `scipy` 为 `audit3d` 网格级精度的硬依赖（cKDTree）
- 模型自身不能看图：**永远先调工具取结构化数据，再基于数字推理**
- **语义级理解受限**：扩展提供精确结构证据（色调/深度/布局/物体类别），但"这是什么场景"的语义判断依赖 DeepSeek 推理 + detect 类别词选择（开放词表检测只能检测提供的类别）
- 仅 Linux；layout / cluster / detect 首次需联网下载权重
