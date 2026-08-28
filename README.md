# pi-vision-struct

**给纯文本模型的数理化视觉通道** —— 全部输出为精确数值（坐标 / hex / 矩阵 / 统计），零主观描述。
DeepSeek 等纯文本 LLM 不需要"看懂"图片，它需要**最高精度的数理数据**来做**数值推理**。

> **平台：仅 Linux x86_64。** 全部工具只读、本地、无网络外发（`a11y` 的会话 DBus 除外）；
> 多数动作在 bwrap 内核沙箱中运行。

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

### 1.2 数理化分层模型

```
物理世界（像素 / 网格 / 顶点 / 法线 / 材质）
         │
    ┌────┼────┐
    │ 2D 传感器              3D 传感器
    │ pixels → hex, ΔE       blender_dump → matrix_world[16], bbox3d[24]
    │ ocr → text, bbox[4]    depth → depth_matrix, stats{near/far/median}
    │ detect → class, bbox   audit3d → gap_mm, interference_bool
    │ scene_stats → 直方图/面积比/对比度/空间布局
    └────┼───────────────────┘
         │
    ┌────▼────┐
    │ 数值融合（全部数学运算） │
    │ crosscheck: Δδ 差异    │
    │ rules: 阈值判定         │
    │ audit3d: bbox3d 交集测试│
    └────┬────┘
         │
    ┌────▼──────────────────┐
    │ DeepSeek（文本推理）   │
    │ 输入：上述全部数值     │
    │ 推理：数值逻辑         │
    │ 输出：可执行数值建议   │
    └───────────────────────┘
```

**关键区别：**

| 旧架构（v1.0.0） | 新架构（v2.0.0 数理化）|
|---|---|
| `semantic` 输出"扁平 / 科技" | `scene_stats` 输出面积比、颜色直方图、文字密度、对比度统计 |
| `critic` 用 VLM 判定 finding 真伪 | `crosscheck` 用两组测量的 Δδ 数值差异 |
| `chart` 用 VLM 提取数据 | OCR 坐标提取 + 网格采样（纯数学）|
| L2 语义层存在 | **删除**——中间不能有"语义理解"层 |

### 1.3 精度保证

- **坐标系**：`coordsys` 字段声明（`css_px` / `device_px` / `image_px` / `screen_px` / `world_m` / `pt`），跨动作比较先用 schema 变换换算
- **单位**：全部数值带单位（mm / px / hex / conf 0-1 / WCAG 比值）
- **可复算**：每个输出都能在原图/原模型上用独立代码验证
- **零幻觉**：无 VLM 热路径（env 自检除外），依赖网络的模型调用全部移除

---

## 二、架构

### 2.1 单端口工具（22 动作，v1.0.0 为 20）

全部能力收敛为单一注册工具 `vs`，`action` 枚举分发到内部分发表：

| 分发表 | 动作 | 传感器/引擎 |
|---|---|---|
| MEASURE | `capture` `pixels` `ocr` `wallpaper` `scene_stats` `env` | PIL / rapidocr / paddle |
| STRUCT | `dom` `pptx` `omniparser` `layout` `pdf` `a11y` `detect` | playwright / CLIP / OWLv2 |
| 3D | `blender_dump` `depth` | bpy headless / 亮度梯度 |
| FUSE | `analyze` `crosscheck` `audit` `rules` `audit3d` | 纯矩阵/阈值运算 |

> 单端口设计使上下文注入从 1359 tokens 降至约 850（-38%），且分发表即真相——新增动作 = 表加一行 + 枚举加一项。

### 2.2 自稳定层

- **preflight**：每次调用前两级预检（解释器可执行 → 核心依赖可导入）；ok 进程内缓存，失败 30s 后可重试；结构化错误 + 修复指引；自诊断路径永久豁免
- **常驻服务**：`omniserver`（OmniParser 权重驻留）与 `ocrserver`（PaddleOCR 驻留），unix socket 行 JSON 协议，自动拉起、崩溃自动清理
- **git 完整性**：`/vs check` 输出 tag/commit 与未提交改动计数

### 2.3 安全

- bwrap `--unshare-net`（零网络）+ 系统根只读；可写白名单仅 `/tmp`、`$HOME`、`~/.cache/*`、`$XDG_RUNTIME_DIR`
- 豁免项均注明原因：`dom`（加载用户 URL）、`a11y`（会话 DBus）
- 依赖版本冻结，`pip check` 零冲突；`VS_NO_SANDBOX=1` 可关闭沙箱

### 2.4 扩展机制

新增传感器 = 在对应分发表中加一行 `Act{script, timeout, build}` + 在 `vs` 工具的 `Type.Union` 中加一个 `Type.Literal`。无需注册新工具。

---

## 三、工具参考

### 测量（pi-vision env）

| 动作 | 参数 | 说明 |
|---|---|---|
| `capture` | `out` 必填 | Wayland 截屏（grim）|
| `pixels` | `image` + `colors`/`regions`/`compare`/`wcag`/`threshold` | 颜色直方图、区域取色、双图 diff、WCAG 对比度 |
| `ocr` | `image` + `region`/`upscale`/`max_items`/`min_conf`/`backend`/`preprocess`/`daemon` | 文字 + 4点bbox + conf；rapidocr(快) / paddle(召回高，常驻服务) |
| `scene_stats` | `image` + `region`/`colors` | 颜色直方图、面积比、对比度统计、空间布局（全部数值，替代旧 semantic）|
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

### 三维（Blender 数理化）

| 动作 | 参数 | 说明 |
|---|---|---|
| `blender_dump` | `blend` 必填 | Blender 场景图：`objects3d[]`（matrix_world 4×4 + bbox3d 8×3 + collection/material/visible）+ cameras（fov/矩阵）|
| `depth` | `image` 必填 | 深度矩阵统计（near/far/median/std/histogram/center_depth），相对亮度梯度 |
| `audit3d` | `report` + `gap_threshold`(默认15mm) | AABB 干涉/间隙检测：interference(相交) + tight_gap(<阈值) |

### 融合（deterministic）

| 动作 | 参数 | 说明 |
|---|---|---|
| `analyze` | `task` + `input`/`url`/`compare` | 任务引擎（整页/多视角）|
| `crosscheck` | `image` + `dom`/`ocr`/`dpr` | DOM↔OCR↔像素三方互验（Δδ 数值）|
| `audit` | `report` + `canvas`/`overlap_threshold` | 2D 重叠/出界/对比度 |
| `rules` | `report` + `canvas`/`align_tol`/`margin` | 设计准则（对齐/间距/安全区）|
| `audit3d` | `report` + `gap_threshold`/`method` | 3D 间隙/干涉，最大精度档 |

#### audit3d 精度模型（最大精度）

对每对 MESH/CURVE 物体逐级判定：

1. **AABB 预筛**：轴对齐包围盒已分离（间隙 > 阈值）→ 直接跳过（不计入任何计数）
2. **OBB-SAT**：用 `bbox3d` 的 8 个世界角点直接构造有向包围盒，分离轴定理（15 轴）判定。旋转感知，比 AABB 紧致，消除因旋转产生的假干涉
3. **网格级 surface-to-surface 距离**（最大精度核心）：导出两物体世界坐标点云，用 `scipy.cKDTree` 双向最近邻求最小表面间距。
   - 同心/嵌套结构（转子在静子内、机匣包裹压气机）因存在环形间隙 → 得到**正间隙** → 判为 `clearance`（非干涉），正确保留装配间隙
   - 真实穿透/接触 → 距离 ≈ 0 → 判为 `interference`

`--method`：`auto`（默认，OBB+Mesh 级最大精度）/ `obb`（仅有向包围盒）/ `mesh`（仅点云距离）/ `aabb`（原 AABB 回退）。

> 涡扇验证：AABB=1042 假干涉 → OBB=952 → auto(mesh)=**4 真干涉** + 1056 已验证真实间隙。剩余 4 例为结构件真实接触（spinner↔tip_cap、nacelle↔bypass_duct、core_casing↔bypass_duct、elbow↔elbow，表面距离均 0.0mm）。

---

## 四、验证

| 场景 | 指标 | 结果 |
|---|---|---|
| 涡扇 4K 渲染 10 图 | 颜色/对比度/空间布局 | 主色 #ECF0F2 14.77% 等，对比度 2.09~18.64 |
| 涡扇 depth 10 图 | median 深度分布 | 0.08~0.68 相对深度，全图产出 |
| 涡扇场景图 196物体 | 8相机 / 17集合 / 547万面 / 最大精度审计 | 180 MESH+8 CURVE / 16110对 / **4 真干涉** / 1056 已验证间隙 |
| 融合回归 | 8断言 | 全部通过 |

```bash
mamba run -n pi-vision python -u tests/test_fusion.py
mamba run -n pi-vision python -u tests/run_self_tests.py
```

---

## 五、已知限制

- 点云距离精度取决于导出顶点密度（默认每物体 ≤4000 点下采样）；极高密度需求可上调 `export_world_verts` 的 `max_verts`
- `scipy` 为 `audit3d` 网格级精度的硬依赖（cKDTree）；缺失时自动回退 OBB/AABB 档
- 模型自身不能看图：**永远先调工具取结构化数据，再基于数字推理**
- 仅 Linux；layout / cluster / detect 首次需联网下载权重
