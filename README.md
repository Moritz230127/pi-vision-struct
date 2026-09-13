> ## ⛔ 本项目已停止维护 / This project is no longer maintained
>
> **状态：2026-09-13 正式封库（End of Life）。** 仓库已归档为**只读**，不再接受
> issue、PR，不再有任何维护与更新。
>
> **原因**：本项目的前提是「纯文本模型需要外部视觉通道」。DeepSeek-V4.1 已原生支持
> 多模态输入，该前提不再成立，本项目随之失去意义。
>
> **仍然可用**：代码、文档、完整提交历史与 MIT 许可全部保留，可自由使用、fork、
> 二次开发；GitHub Release 中的预编译二进制（`vs-bin` / `vsd` / `vs-mcp`）仍可下载，
> 已有的 npm 安装（postinstall 从 Release 拉二进制）不会中断。
> npm 包 `pi-version-and-gui` 已标记 deprecated，安装时会显示弃用警告。
>
> **请注意**：不再有任何安全修复或兼容性更新。涉及 AT-SPI / Wayland / niri 的
> 环境变更可能导致工具失效，请自行评估后使用。
>
> ---
>
> **Status: archived and End-of-Life as of 2026-09-13.** The repository is read-only;
> issues, pull requests, updates and maintenance have all stopped.
>
> **Why**: this project was built on the premise that text-only models need an external
> vision channel. DeepSeek-V4.1 handles multimodal input natively, so that premise no
> longer holds and the project has lost its purpose.
>
> **What still works**: the source, docs, full commit history and MIT license remain
> available for use, forking and further development. Prebuilt binaries in GitHub
> Releases stay downloadable, so existing npm installs keep functioning. The npm
> package `pi-version-and-gui` is marked deprecated.
>
> **Please note**: no security or compatibility fixes will be issued. Environment
> changes around AT-SPI / Wayland / niri may break the tooling — evaluate accordingly.

# pi-version-and-gui

**给纯文本模型的数理化视觉通道 + GUI 控制** —— 全部输出为精确数值（坐标 / hex / 矩阵 / 统计），零主观描述。
DeepSeek 等纯文本 LLM 不需要"看懂"图片，它需要**最高精度的数理数据**来做**数值推理**，并需要**元素编号引用**来做**精确 GUI 控制**。

> **平台：仅 Linux x86_64（niri Wayland 桌面）。** 全部工具只读、本地、无网络外发（`a11y` 的会话 DBus 除外）。
> **V4：全面 Rust 化 + GUI 控制** —— 传感器/融合/规则/调度/沙箱全部 Rust 实现，
> 与 Python 版逐字段差分门禁验证（容差全收紧）。**零 Python 回退**。
> GUI 控制：a11y_action 直达（AT-SPI DoAction，零坐标误差）+ 五级降级链 + 闭环验证。

---

## 一、原理：为什么必须是数理化

### 1.1 核心命题

> 图片不是"看"的对象，是"转化为数理数据"的原材料。
> LLM 多模态本身不具备工程级能力——因为多模态是在"猜"，而工程需要"算"。

传统视觉扩展让 VLM 输出"一张风景照"、"一个停车场"、"扁平 / 科技风格"——这是**主观噪声**。
DeepSeek 在文本推理上是最强模型之一，它不需要别人替它"理解"图片，它需要：

- **精确的数值**（坐标 / hex / 矩阵 / 统计）来做**数值推理**
- **编号的元素清单**（`click [3]`）来做**精确 GUI 控制**

### 1.2 GUI 控制：元素优先、坐标兜底（Codex 同款架构）

```text
感知（a11y 树 / 融合元素）→ 编号 [0]...[N] → 决策（click [3]）→ 行动（五级降级链）→ 验证（闭环）
```

- **a11y_action**：AT-SPI DoAction 直达（零坐标误差，~100ms）
- **element_click**：编号 → GetExtents 屏幕坐标 → 绝对定位（±0px）
- **vision_click**：融合元素 bbox center → 绝对定位（±2px，无 AT-SPI 时）
- **global**：wlrctl serve 兜底
- **闭环验证**：点击后截图对比 → 失败降级重试

---

## 二、安装

### 2.1 pi 扩展（推荐）

```bash
pi install npm:pi-version-and-gui
```text

### 2.2 从源码构建

```bash
git clone git@github.com:Moritz230127/pi-vision-struct.git
cd pi-vision-struct/rust
cargo build --release
# 二进制: target/release/vs-bin / vsd / vs-mcp
```

### 2.3 模型

模型存 `~/.cache/vsensor/`（~4GB，**独立于工作区**，npm 扩展直接读）：

| 模型 | 用途 | 大小 | 下载方式 |
| --- | --- | --- | --- |
| u2net.onnx | saliency 显著性 | ~170MB | PyTorch 导出 |
| midas_v21_small_256.onnx | depth 深度 | ~100MB | PyTorch 导出 |
| mobile_sam_encoder/decoder.onnx | segment 分割 | ~80MB | PyTorch 导出 |
| rapidocr/ | OCR（det/rec/cls） | ~30MB | **install.js 自动下载**（modelscope） |
| oclv2/ | detect 检测 | ~1.2GB | HuggingFace（需认证） |
| florence2/ | omniparser 图标标注 | ~1.5GB | HuggingFace（需认证） |
| icon_detect.onnx | omniparser 图标检测 | ~40MB | HuggingFace（需认证） |
| doclayout.onnx | layout 版式 | ~100MB | PyTorch 导出 |
| clip_vitb32.onnx | cluster 聚类 | ~350MB | PyTorch 导出 |

**自动下载**：`pi install npm:pi-version-and-gui` 的 postinstall 自动下载
rapidocr 3 个 ONNX（OCR 必需）。其余模型需 PyTorch 导出或 HuggingFace 下载
（见 `python/export_*.py` 导出脚本）。

**检查缺失**：`vs check` 列出缺失模型清单。

> ⚠️ **二进制问题说明**：
> - 二进制 151MB 不打包进 npm（包保持轻量 248KB），postinstall 从 GitHub Release 下载
> - npm 11 的 install-scripts 安全策略可能阻止 postinstall 自动执行：
>   `npm install-scripts approve pi-version-and-gui` 或手动 `node scripts/install.js`
> - **approve 必须在包安装目录运行**（pi 的 npm 根）：
>   ```bash
>   cd ~/.pi/agent/npm   # pi 的 npm 安装根
>   npm install-scripts approve pi-version-and-gui
>   ```
>   在 `~` 运行会报 `ENOMATCH`（找不到包）
> - 模型 4.1GB 不打包进 npm（太大），存 `~/.cache/vsensor/`（独立于工作区）

---

## 三、传感器（vs-bin 子命令）

| 子命令 | 功能 | 输出 |
| --- | --- | --- |
| `ascii` | ASCII 栅格（元素感知） | grid + 元素区域高分辨率 |
| `scene_stats` | 全局统计 | 亮度/对比度/色直方图/布局摘要 |
| `pix` | 像素分析 | 主色/亮度/饱和度 |
| `edge` | 边缘检测 | 密度热图（16×16）+ top5 线段 |
| `ocr` | 文本识别 | 文本 + 4 点 bbox + conf |
| `saliency` | 显著性 | 候选区 + score |
| `depth` | 深度估计 | 深度分箱统计 |
| `segment` | 前景分割 | 实例 + bbox + 面积比 |
| `omniparser` | 图标解析 | 图标 + Florence-2 语义标注 |
| `detect` | 开放词表检测 | 类别 + bbox + conf |
| `layout` | 文档版式 | 布局区域 |
| `cluster` | CLIP 聚类 | 感知相似度分组 |
| `wall` | 壁纸分类 | 主色/亮度/饱和度档 |
| `fusion` | D-S 证据融合 | 融合置信度 |
| `rules` | 规则引擎 | 规则判定 |
| `audit` | 审计 | findings |
| `geometry` | 几何分析 | 形状 + 元素 |
| `analyze` | 任务引擎 | JSON 配置编排 |
| `protocol` | 多轮反馈协议 | analyze/zoom/probe/click |
| `gui` | GUI 行动 | click/type/key/drag/scroll/focus/chain |
| `pdf` | PDF 解析 | 文本 + bbox |
| `pptx` | PPTX 解析 | 形状 + 文本 |
| `blend` | 3D 解析 | 场景图 + bbox3d |
| `capture` | 截屏 | 窗口级/全屏 |
| `dom` | DOM 提取 | 网页元素 |
| `a11y` | 无障碍树 | AT-SPI 元素 + bus/path |
| `hit_test` | 命中测试 | 点击点 → 元素 |
| `check` | 自检 | 模型/二进制/环境 |
| `setup` | 安装器 | 模型下载 + 目录创建 |

---

## 四、GUI 控制协议

### 4.1 编号引用（click [N]）

```bash
# 1. 感知：分析窗口 → 融合元素清单（编号 + bbox + bus/path）
vs-bin protocol --action analyze --image cap.png --window 12

# 2. 行动：编号引用点击（a11y_action 直达）
vs-bin protocol --action click --image cap.png --confirm 3 --window 12

# 3. 闭环：点击后截图对比验证
vs-bin gui --action verify_task --before a.png --after b.png --viewport "100,100,500,400"
```text

### 4.2 chain 原语（多步动作，减少模型轮次）

```bash
vs-bin gui --action chain --steps "key ctrl+l; type bilibili.com; key Return; sleep 1"
```

### 4.3 五级降级链

```text
a11y_action（AT-SPI DoAction，零误差）→ element_click（GetExtents ±0px）
→ vision_click（融合元素 ±2px）→ global（wlrctl serve 兜底）
```

---

## 五、性能基准（Rust release vs Python，s03_alignment.png）

| 传感器 | Rust release | Python | 加速比 |
| --- | --- | --- | --- |
| scene_stats | 19ms | 323ms | **17.0×** |
| pix | 203ms | 829ms | **4.1×** |
| ocr | 1077ms | 1969ms | **1.8×** |

### GUI 控制延迟（真实 Firefox 实测）

| 操作 | 耗时 |
| --- | --- |
| 窗口 capture | 40ms |
| analyze（a11y 优先） | 226ms |
| click（a11y_action） | 216ms |
| chain（5 步） | 232ms |

---

## 六、架构

```text
pi-version-and-gui
├── extensions/index.ts        # TS 薄壳（exec vs-bin，零业务逻辑）
├── skills/                    # skill（编号引用协议教学）
├── rust/
│   ├── vs-core/               # 37 个传感器/模块（全 Rust）
│   ├── vs-bin/                # CLI（30+ 子命令）
│   ├── vsd/                   # 常驻 daemon（调度器 + 沙箱 + 事件驱动）
│   └── vs-mcp/                # MCP server（28 工具）
└── python/                    # 参考实现（V3，仅文档）
```

### vsd daemon（单进程闭环）

```text
感知层（a11y 树 / 融合元素）→ 行动层（五级降级链）→ 协议层（click [N]）→ 事件层（niri event-stream）
```

---

## 七、验证

- **241 测试全绿**（单元 + 差分门禁 + 契约测试）
- **clippy 0 警告 + fmt 干净**
- **零 Python 回退**（grep 确认无 Command::new python）
- **差分门禁**：12 个差分测试（schema/sensors/S2/S4/S5/S6/S7/detect/cluster/layout/p3）
- **真实 GUI 测试**：Firefox Google 搜索闭环（a11y_action 直达 + chain 原语 + 闭环验证）

---

## 八、已知限制

- **Blender 无 AT-SPI 树**（自绘 UI）——click 走 vision_click 兜底（~15s DL）
- **DeepSeek 纯文本模型**——每轮思考 5-15s（chain 原语缓解）
- **中文输入法**：微软双拼（type 中文按双拼键序，优先用 URL/英文）
- **仅 Linux x86_64 + niri Wayland**（与 V3 一致）

---

## 九、开发

```bash
cd rust
cargo test --workspace    # 全量测试
cargo clippy --workspace --all-targets  # 0 警告
cargo fmt                 # 格式
```text

## 十、许可

MIT
