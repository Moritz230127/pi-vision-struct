# pi-vision-struct

**给纯文本模型的数理化视觉通道** —— 全部输出为精确数值（坐标 / hex / 矩阵 / 统计），零主观描述。
DeepSeek 等纯文本 LLM 不需要"看懂"图片，它需要**最高精度的数理数据**来做**数值推理**。

> **平台：仅 Linux x86_64。** 全部工具只读、本地、无网络外发（`a11y` 的会话 DBus 除外）。
> **V4.1：全面超越**（P1-P4 完成）——传感器/融合/规则/调度/沙箱全部 Rust 实现，
> 与 Python 版逐字段差分门禁验证（容差全收紧）。**零 Python 回退**（仅 layout 因 paddle PIR 导出不兼容保留回退）。

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

### 1.2 数理化分层模型（v4.0.0）

```
物理世界（像素 / 网格 / 顶点 / 法线 / 材质）
         │
    ┌────┼────────────────────────────┐
    │ L0 源层（Rust）                  │
    │ dom(chromiumoxide) · pptx(OOXML)│
    │ a11y(atspi) · capture(xcap)     │
    │ pdf(pdfium) · blend(blend-rs)   │
    └────┼────────────────────────────┘
         │
    ┌────▼────────────────────────────┐
    │ L1 确定性测量（Rust 纯算法）      │
    │ pix → hex, ΔE (SLIC超像素)       │
    │ ocr → text, bbox[4] (rapidocr)  │
    │ scene_stats → Otsu/直方图/空间    │
    │ edge → 亚像素边缘 (0.0000px)      │
    │ ascii → 多分辨率文本栅格          │
    │ geometry → VTracer 形状原语      │
    └────┼────────────────────────────┘
         │
    ┌────▼────────────────────────────┐
    │ L2 轻量感知（Rust ort ONNX）      │
    │ saliency → U²-Net 候选区域       │
    │ segment → MobileSAM 前景分割     │
    │ depth → MiDaS 单目深度           │
    │ detect → OWLv2 开放词表检测      │
    │ omniparser → YOLOv9+Florence-2   │
    └────┼────────────────────────────┘
         │
    ┌────▼────────────────────────────┐
    │ F1 融合引擎（Rust 纯算法）         │
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

**关键区别（v3 → v4）：**

| v3.0.0 | v4.0.0 |
| --- | --- |
| Python 传感器（PIL/skimage/rapidocr） | **Rust 全实现**（vs-core crate，差分门禁逐字段一致） |
| bwrap 沙箱 | **纯 Rust 沙箱**（landlock + seccomp + mount namespace） |
| vsched（Python 调度） | **vsd daemon**（tokio + unix socket + 显存预算 + LRU + 优先级 + 功耗感知） |
| playwright DOM | **chromiumoxide**（Rust 驱动 headless Chrome） |
| paddle OCR 后端 | **rapidocr-core**（fork 修复 det parity：BGR + fillPoly + unclip 1.4） |
| OWLv2 Python 推理 | **OWLv2 三 ONNX 导出**（text/vision encoder + detection head + 自研 NMS） |
| OmniParser Python | **Florence-2 拆分 ONNX**（vision/language encoder + decoder 自回归生成循环） |
| 三 conda env | **单 Rust 二进制**（vs-bin）+ 1 处 Python 回退（layout） |

### 1.3 精度保证

- **坐标系**：`coordsys` 字段声明（`css_px` / `device_px` / `image_px` / `screen_px` / `world_m` / `pt`），跨动作比较先用 schema 变换换算
- **单位**：全部数值带单位（mm / px / hex / conf 0-1 / WCAG 比值）
- **可复算**：每个输出都能在原图/原模型上用独立代码验证
- **零幻觉**：数值主链无 VLM 热路径；融合层全部纯算法（D-S 证据理论）
- **差分门禁**：每阶段 Rust 输出 vs Python golden 逐字段对比（确定性输出精确 ≤1e-6，算法依赖输出容差）

---

## 二、架构

### 2.1 Rust 单二进制（vs-bin）

全部能力收敛为单一 Rust 二进制 `vs-bin`，clap 子命令 = action 枚举：

| 分发表 | 子命令 | 传感器/引擎 |
| --- | --- | --- |
| MEASURE | `capture` `pix` `ocr` `scene_stats` | xcap / rapidocr-core / 纯算法 |
| STRUCT | `dom` `pptx` `omniparser` `layout` `pdf` `a11y` `detect` | chromiumoxide / OOXML / YOLOv9+Florence-2 / OWLv2 |
| 3D | `blend` `depth` | blend-rs / MiDaS |
| L2 感知 | `saliency` `segment` `depth` `edge` `ascii` `geometry` | U²-Net / MobileSAM / MiDaS / Devernay / VTracer |
| FUSE | `fusion` `audit` `rules` | D-S 证据理论 / 匈牙利匹配 |
| 其他 | `schema` | schema v3 输出契约验证 |

> 单端口设计使上下文注入最小化，且分发表即真相——新增动作 = 表加一行 + 枚举加一项。

### 2.2 调度管理器（vsd daemon）

8GB VRAM 硬约束下，任何并发 GPU 推理都可能 OOM。`vsd` 是 GPU/CPU 推理请求的唯一入口
（tokio + unix socket 行 JSON 协议，与 Python `vsd.py` 兼容）：

- **显存预算表**：每模型注册 `vram_reserve` + `vram_peak`，总预算 6.5GB（留 1.5GB 安全水位）
- **并发信号量**：GPU 推理槽位 = 1（串行），杜绝同刻双 GPU 推理
- **模型 LRU 淘汰**：最近未用模型卸载（冷载 1-3s / 热 <100ms）
- **优先级队列**：analyze > zoom > probe > bg
- **功耗感知**：检测电源状态（battery/plugged），电池模式大模型自动 CPU 降级，接电源全速
- **CPU 降级**：显存紧张时自动 CPU 推理，任务永不因 OOM 失败
- **超时保护**：推理超时自动终止（capture 10s 超时防 Wayland 阻塞）

### 2.3 纯 Rust 沙箱（替代 bwrap）

三层防护（`vsd::sandbox`）：

1. **Landlock**：系统路径只读，仅缓存目录可写
2. **seccomp**：禁止 execve/execveat/ptrace 等逃逸向量
3. **mount namespace**：私有传播隔离

### 2.4 双宿主接入（pi-coding-agent + Claude Code）

| 宿主 | 入口 | 接入方式 |
| --- | --- | --- |
| **pi-coding-agent** | `extensions/index.ts`（~160 行 exec 胶水，零业务逻辑） | `package.json` 的 `pi.extensions` 自动加载 |
| **Claude Code** | `rust/crates/vs-mcp`（Rust MCP server，rmcp crate） | `claude mcp add pi-vision-struct -- vs-mcp` |

---

## 三、工具参考（vs-bin 子命令）

### 测量

| 子命令 | 参数 | 说明 |
| --- | --- | --- |
| `capture` | `--out` 必填 | Wayland 截屏（xcap，10s 超时保护） |
| `pix` | `--image` + `--colors`/`--regions`/`--compare`/`--wcag` | SLIC 超像素主色、区域取色、双图 diff、WCAG 对比度 |
| `ocr` | `--image` + `--region`/`--upscale`/`--max-items`/`--min-conf` | 文字 + 4点bbox + conf + 词典纠错（rapidocr-core） |
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
| `segment` | `--image` + `--saliency`(联动) | MobileSAM 前景分割（Rust encoder + mask decoder ONNX 全 Rust） |
| `depth` | `--image` + `--region` | MiDaS 单目深度（近/中/远分布） |
| `edge` | `--image` + `--max-lines` | Devernay 亚像素边缘（精度 0.0000px） |
| `ascii` | `--image` + `--cols`/`--rows`/`--color` | 多分辨率 ASCII 栅格 |
| `geometry` | `--image` + `--max-shapes` | VTracer SVG 化 → 形状原语 |

### 三维

| 子命令 | 参数 | 说明 |
| --- | --- | --- |
| `blend` | `--file` 必填 | .blend 场景图（blend-rs 解析：objects3d[] + matrix_world + bbox3d） |
| `depth` | `--image` 必填 | 深度矩阵统计（near/far/median/std/histogram） |

### 融合（F1 引擎）

| 子命令 | 参数 | 说明 |
| --- | --- | --- |
| `fusion` | `--reports`(多传感器 JSON) | D-S 证据融合：匈牙利匹配 → mass 组合 → belief/plausibility/uncertainty |
| `audit` | `--report` + `--canvas`/`--overlap-threshold` | 2D 重叠/出界/对比度 |
| `rules` | `--report` + `--canvas`/`--align-tol`/`--margin` | 设计准则（对齐/间距/安全区） |

#### 融合引擎（D-S 证据理论）

1. **align**：匈牙利算法全局最优匹配（代价 = 1 - IoU - α·text_sim）
2. **evidence**：传感器输出 → mass 函数（OCR conf / ΔE 似然 / 检测置信度 / 前景面积比 / 显著性分数 / 深度一致性）
3. **combine**：D-S 正交和组合 `m₁₂(A) = Σ_{B∩C=A} m₁(B)m₂(C) / (1-K)`
4. **decide**：belief>0.6 → confirmed；plausibility<0.4 → conflict；K>0.7 → needs_review

#### 多轮反馈协议（SeeingEye 模式）

```
DeepSeek ──vs analyze──→ 粗报告（saliency 候选 + ascii 粗栅格 + 全局统计）
DeepSeek 选候选区 ──vs zoom [region]──→ 细报告（ocr 高倍 + segment 前景 + edge 亚像素）
DeepSeek 追问 ──vs probe [bbox, sensor]──→ 单传感器定向证据
DeepSeek 综合推理 → 最终答案
```

---

## 四、验证（差分门禁）

| 阶段 | 内容 | 门禁 |
| --- | --- | --- |
| S0 | schema/bbox/color/util 基础设施 | 3 差分测试全绿 |
| S1 | ascii/scene_stats/pix/edge 纯算法 | 38 单元 + 差分全绿 |
| S2 | fusion/rules/audit/geometry | 57 单元 + 4 差分全绿 |
| S3 | vsd daemon + 调度器 + 沙箱 | 并发压测零 OOM + 沙箱逃逸测试 |
| S4 | saliency/depth/segment ONNX | 候选区 IoU ≥ 0.95 |
| S5 | OCR（rapidocr-core fork） | CER=0 + bbox IoU ≥ 0.85 |
| S6 | omniparser + layout | 元素 IoU ≥ 0.9（实测 1.0） |
| S7 | capture/dom/a11y/pdf/pptx/blend | 端到端真实图测试 |
| S8 | 宿主接入（TS 薄壳 + vs-mcp） | 全量回归全绿 |

**当前状态：105+ 测试全绿；clippy 0 警告；fmt 干净。**

### 性能基准卡（release vs Python，s03_alignment.png，2026-09-07 实测）

| 传感器 | Rust release | Python | 加速比 |
| --- | --- | --- | --- |
| scene_stats | 19ms | 323ms | **17.0×** |
| pix | 203ms | 829ms | **4.1×** |
| ocr | 1077ms | 1969ms | **1.8×** |
| cluster（3 文件） | 552ms | — | 纯 Rust（无 Python 对应） |

```bash
cd rust && cargo test --workspace          # 全量测试
cargo clippy --workspace --all-targets     # 0 警告
cargo build --release -p vs-bin            # 单二进制
./target/release/vs-bin <子命令> --help    # 工具参考
```

---

## 五、环境

- **Rust 工具链**：cargo workspace（vs-core / vs-bin / vsd / vs-mcp 四 crate）
- **ONNX 模型**（本地缓存，离线可用）：`~/.cache/vsensor/`（u2net.onnx / mobile_sam_encoder.onnx / midas_v21_small_256.onnx / rapidocr/ / florence2/ / owlv2/）
- **Python 回退**（1 处）：`layout`（PP-DocLayoutV3 paddle 3.x PIR 导出不兼容，plan2 §6 已知风险）
- 仅 Linux；layout / detect 首次需联网下载权重

---

## 六、已知限制

- `layout` 为 Python 回退（paddle 3.x PIR 导出不兼容 paddle2onnx，plan2 §6 已知风险）
- Florence-2 caption 生成（ort greedy vs transformers 采样）内容不同（门禁用非空 + 长度容差）
- 模型自身不能看图：**永远先调工具取结构化数据，再基于数字推理**
- 仅 Linux；layout / detect 首次需联网下载权重
