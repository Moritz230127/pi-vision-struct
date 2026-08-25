# pi-vision-struct

**给纯文本模型的像素级精确视觉通道** —— 分层无损：L0 源码 → L1 确定性测量 → L2 opt-in 语义。
DeepSeek 等纯文本模型经此获得视觉能力：输出的是**数字/坐标/hex**（可复算、可审计），而非自然语言转述。

> **平台支持：仅 Linux x86_64**（Wayland/X11）。macOS / Windows 不受支持。
> 全部工具只读、本地、无网络外发（Ollama 仅 localhost）；多数动作在 bwrap 内核沙箱中运行。

---

## 一、架构设计

### 1.1 分层无损通道（schema `vision-report/v2`）

```text
L0 源码    dom(dom_dump) / pptx(pptx_dump) / a11y(AT-SPI 无障碍树)     ← 无损结构真值
L1 测量    pixels(颜色/diff/WCAG) / ocr(rapidocr|paddle, 4点bbox)      ← 确定性数字
L2 语义    semantic(标签) / chart(图表→数据) / critic(复核)            ← opt-in，本地 VLM
DL 感知    omniparser(图标级) / detect(OWLv2 开放词表) / cluster(CLIP)  ← 仅感知
融合       analyze(任务引擎) / crosscheck(互验) / audit / rules(准则)   ← 确定性符号运算
```

原则：**测量只做精确计算（PIL/像素数学）；DL 只做感知（OCR/图标/语义/检测）；融合保持确定性（无学习式融合）**。
每个异常项携带 `evidence`（数值+阈值）、`primitive`（`[bbox: x1,y1,x2,y2]` 记法）与 `suggested_cause`，
下游文本模型以坐标原语推理，消除"左边的元素"式指代歧义。

### 1.2 单端口工具（20 动作）

全部能力收敛为单一注册工具 `vs`，`action` 枚举分发到内部分发表
（MEASURE/STRUCT/FUSE + cluster/detect/chart）。上下文注入 ≈704 tokens
（四分组工具时代为 1359）。分发表即真相：新增动作 = 表加一行 + 枚举加一项。

### 1.3 自稳定层

- **preflight**：每次调用前两级预检（解释器可执行 → 核心依赖可导入）；
  ok 进程内缓存，失败缓存 30s 后可重试；失败返回结构化错误 + 修复指引；
  自诊断路径（inline -c、vs_setup.py）永久豁免——诊断永不被诊断阻断
- **常驻服务 ×2**：`omniserver`（OmniParser 权重驻留）与 `ocrserver`
  （PaddleOCR 引擎驻留），unix socket 行 JSON 协议，自动拉起、崩溃后残留 socket 自动清理
- **git 完整性**：`/vs check` 输出当前 tag/commit 与未提交改动计数，工作树漂移用户可见

### 1.4 安全与本地化

- 运行时零网络外发（审计确认）+ 内核级强制：bwrap `--unshare-net`（无任何网络）+
  系统根只读；可写白名单仅 `/tmp`、`$HOME`、`~/.cache/omniparser`、`$XDG_RUNTIME_DIR`
- 沙箱豁免（均注明原因）：`dom`（本职加载用户 URL）、`semantic`/`critic`/`chart`
  （宿主 Ollama 127.0.0.1，硬编码目标）、`a11y`（会话 DBus）
- 常驻服务走 unix socket（文件系统 IPC，无 TCP 端口）
- 依赖版本精确钉死（`requirements*.txt` 冻结），`pip check` 零冲突
- `VS_NO_SANDBOX=1` 可关闭沙箱（不推荐）

### 1.5 坐标系与单位

`coordsys` 字段声明：`css_px`（DOM）/ `device_px`（截图）/ `image_px`（图像文件）/
`screen_px`（a11y）/ `pt`（PPTX/PDF，1pt=1/72in）。跨动作比较先用 schema 提供的变换换算。

---

## 二、安装教程（Linux，手工步骤）

无需一键脚本——以下每步都可独立执行与验证。任一步骤失败，先解决再继续。

### 步骤 0 · 系统前置包

```bash
# Arch Linux
sudo pacman -S --needed git curl bzip2 grim at-spi2-core python-gobject
# Debian / Ubuntu
sudo apt install -y git curl bzip2 libatspi2.0-0 python3-gi
# 可选（Wayland 截屏需要 compositor；X11 用户另备 scrot/mss 路径）
```

验证：`which git curl && echo ok`

### 步骤 1 · 安装 Miniforge（conda/mamba）

```bash
curl -L -o /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh
bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
"$HOME/miniforge3/bin/conda" init bash   # 或你的 shell；重开终端生效
```

验证：新终端里 `mamba --version` 有输出。

### 步骤 2 · 获取扩展

```bash
# 方式一：pi 扩展机制
pi install git:github.com/Moritz230127/pi-vision-struct
# 方式二：直接克隆
git clone https://github.com/Moritz230127/pi-vision-struct.git ~/.pi-extensions/pi-vision-struct
```

### 步骤 3 · 核心环境（必需）

```bash
cd ~/.pi-extensions/pi-vision-struct
mamba create -y -n pi-vision python=3.12
mamba run -n pi-vision pip install -r python/requirements.txt
```

验证：

```bash
mamba run -n pi-vision python -c "import onnxruntime, rapidocr, pptx, PIL; print('core OK')"
python3 python/setup/vs_setup.py --check    # 应输出 JSON："pi-vision": {"complete": true}
```

### 步骤 4 · OmniParser 环境（可选：图标级元素 / detect / cluster）

```bash
mamba create -y -n omniparser python=3.12
mamba run -n omniparser pip install torch --index-url https://download.pytorch.org/whl/cpu
mamba run -n omniparser pip install -r python/requirements-omniparser.txt transformers
```

验证：`mamba run -n omniparser python -c "import torch, transformers, openai; print('omni OK')"`

### 步骤 5 · 写运行时配置

`~/.config/pi-vision-struct.json`：

```json
{
  "pi_vision_python": "<HOME>/conda-envs/pi-vision/bin/python",
  "omniparser_python": "<HOME>/conda-envs/omniparser/bin/python",
  "l2_model": "qwen3-vl:8b"
}
```

`<HOME>` 替换为你的家目录绝对路径。`l2_model` 是 semantic/critic/chart 的 VLM 档位，
可随时换成更大的模型（如 qwen3-vl:235b）。

### 步骤 6 · L2 语义后端（可选：semantic/critic/chart）

安装 [Ollama](https://ollama.com) 并拉取视觉模型：

```bash
ollama pull qwen3-vl:8b
```

不装也不影响 L0/L1/融合全部功能——L2 全部 opt-in。

### 步骤 7 · 注册进 pi 并验证

重启 pi 后，对它说一句"截个图看看屏幕上有什么"，或手动验证：

```bash
# 五条冒烟（按序，前一条产物喂后一条）
mamba run -n pi-vision python python/vs_capture.py --out /tmp/s.png
mamba run -n pi-vision python python/vs_ocr.py   --image /tmp/s.png | head -c 200
mamba run -n pi-vision python python/vs_pix.py   --image /tmp/s.png --colors 3 | head -c 200
mamba run -n pi-vision python python/vs_a11y.py  --list
mamba run -n pi-vision python tests/test_fusion.py
```

预期：五条全部输出合法 JSON / `结果: 8 通过 / 0 失败`。

### 替代方案 · Docker（零安装）

```bash
docker pull ghcr.io/moritz230127/pi-vision-struct:latest
docker run --rm -v "$PWD":/work pi-vision-struct:latest ocr --image /work/a.png
docker run --rm -v "$PWD":/work pi-vision-struct:latest help
```

镜像已烘焙 OmniParser 权重 / Florence-2 / CLIP；layout 首用自动经内置 hf-mirror 拉 paddle 模型。

---

## 三、使用

单端口 `vs` 工具，20 个 action：

| 组 | action |
|---|---|
| 测量 | `capture` `pixels` `ocr` `wallpaper` `semantic` `env` |
| 结构 | `dom` `pptx` `omniparser` `layout` `pdf` `a11y` `detect` |
| 融合 | `analyze` `crosscheck` `audit` `rules` `critic` |
| 其他 | `cluster` `chart` |

完整参数与决策表见技能文件 `skills/vision-situation/SKILL.md`。

---

## 四、验证体系

- 基准报告卡 `bench/report_card.json`：颜色 ΔE=0.0、OCR CER=0.0、diff 定位 recall=1.0
- 验收 `bench/run_acceptance.py`：12 样本规则臂一致率 100%、critic 臂 75%
- 回归：`tests/` 六套（基础/融合/用例/规则/聚类/critic）
- 发布门禁（v1.0.0）：静态全绿 + 单元 25/25 + 客户视角 19 例零报错一次过 +
  Docker 最坏路径双场景实测

```bash
mamba run -n pi-vision python -u tests/run_self_tests.py
mamba run -n pi-vision python -u tests/test_fusion.py
```

---

## 五、已知限制

- 仅 Linux；macOS / Windows 无适配（安装器与文档均已移除相关指引）
- a11y 依赖桌面会话的 AT-SPI 总线；纯 CLI 会话只能看到 portal 进程
- layout / cluster / detect 首次使用需联网下载权重（~30MB / ~350MB / ~1.2GB），此后离线
- critic 对全局属性缺陷（出界/安全区）在裁剪视图会误判，需结合 rules 证据解读
- L2 语义有思考成本（默认 opt-in）
- ultralytics YOLO-World 未采用（8.3.x/8.4.x set_classes 嵌入异常，实测记录于 CHANGELOG 0.4.0）

## 文档索引

- `skills/vision-situation/SKILL.md` — 20 动作完整参考与决策表
- `docs/omniparser-setup.md` — OmniParser 环境维护手册
- `CHANGELOG.md` — 版本记录（含选型实录与决策依据）
