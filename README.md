# pi-vision-struct

**给纯文本模型的超高精度视觉通道** —— 分层无损：L0 源码 → L1 确定性测量 → L2 opt-in 语义。DeepSeek 等文本模型经此获得像素级精确的视觉能力（数字/坐标/hex，而非自然语言转述）。

全部工具**只读、本地、无网络外发**（Ollama 仅 localhost）。Python 运行于 conda env。

## 安装

```bash
pi install npm:pi-vision-struct      # 或 pi install /path/to/pi-vision-struct
```

首次使用前运行引导安装（自动检测/创建 conda env + 依赖 + 自测）：

```text
/vs setup            # 核心安装（默认 dry-run 展示计划；env 已存在时直接执行）
/vs setup --with-omniparser   # 附加 OmniParser env（图标级元素）
/vs setup --with-dom          # 附加 playwright firefox（dom_dump）
/vs check            # 只读健康检查
```

## 架构：分层无损通道（vision-report/v2）

```text
L0 源码    dom_dump (DOM+computed style) / pptx_dump (PPTX XML)   ← 无损真值
L1 测量    pix_analyze (颜色/diff/WCAG) / ocr_boxes (带坐标)       ← 确定性数字
L2 语义    semantic_tag / vs_critic (本地 qwen3-vl, opt-in)        ← 不可测属性
融合       vs_crosscheck (互验) / vs_audit / vs_rules (准则)       ← 符号/确定性
DL 感知    vs_omniparser (图标级, CPU) / vs_cluster (CLIP, CPU)    ← 仅感知
```

原则：**测量工具只用精确计算（PIL/像素数学）；DL 只做感知（OCR/图标/语义）；融合保持确定性（无学习式融合）**。

## 工具（4 分组 · CLI 风格，action 枚举）

| 工具 | action | 用途 | env |
|---|---|---|---|
| `vs_measure` | capture / pixels / ocr / wallpaper / semantic / env | 截屏、像素测量（颜色/diff/WCAG）、OCR、壁纸分类、L2 语义、环境自检 | pi-vision |
| `vs_struct` | dom / pptx / omniparser | DOM 结构、PPTX 结构、任意截图图标级元素 | pi-vision / omniparser |
| `vs_fuse` | analyze / crosscheck / audit / rules / critic | 任务引擎、三方互验、布局审计、设计准则、VLM 复核 | pi-vision |
| `vs_cluster` | — | CLIP 相似图聚类（确定性） | omniparser |

> 完整命令参考在技能 `skills/vision-situation/SKILL.md`（模型按需加载，不进 AGENTS.md）。
> 设计权衡：15 个细粒度工具 → 4 分组，每轮工具 schema 约 **3.5K → 1.3K token**（省 ~63%），
> 且模型路由选择面变小、准确率提升；action 用 enum 校验保留参数安全。

## 验证

- **基准报告卡** `bench/report_card.json`：颜色 ΔE=0.0、OCR CER=0.0、diff 定位 recall=1.0
- **验收** `bench/run_acceptance.py`：12 样本规则臂一致率 100%、critic 臂 75%（分歧已实证）
- 回归：`tests/` 下 6 套 64 断言（基础/融合/用例/规则/聚类/critic）
- 真实数据：Reddit 截图（OCR 60 项、OmniParser 101 元素）、46 张壁纸

```bash
/home/Arch/conda-envs/pi-vision/bin/python -u tests/run_self_tests.py
/home/Arch/conda-envs/pi-vision/bin/python -u tests/test_fusion.py
/home/Arch/conda-envs/pi-vision/bin/python -u tests/run_case_tests.py
/home/Arch/conda-envs/pi-vision/bin/python -u tests/test_rules.py
/home/Arch/conda-envs/pi-vision/bin/python -u tests/test_critic.py
/home/Arch/conda-envs/omniparser/bin/python -u tests/test_cluster.py
/home/Arch/conda-envs/pi-vision/bin/python -u bench/run_benchmark.py
```

## 环境

- `pi-vision` env（python 3.12）：onnxruntime / rapidocr / playwright / python-pptx / pillow / numpy —— `python/requirements.txt`
- `omniparser` env（python 3.12，CPU torch）：OmniParser + CLIP —— `python/requirements-omniparser.txt`；版本钉死与模型/补丁详见 **docs/omniparser-setup.md**
- env 目录：`$HOME/conda-envs`（`PI_VISION_PYTHON` 可覆盖 python 路径）；下载需代理时：`/vs setup --proxy http://127.0.0.1:10808`

## 已知限制（残余差距）

- 原生多模态基线对比需免费 Gemini/GLM 密钥（用户侧）
- critic 仅复核规则发现的 finding；全局属性缺陷（出界/安全区）裁剪视图会误判（豁免或带画布边界）
- 语义 L2 有思考成本（qwen3-vl，单区约 20s），默认 opt-in
- 发布前需补 repository 字段（npm publish 前置）

## 文档

- `docs/omniparser-setup.md` — OmniParser 环境维护手册
- `skills/vision-situation/SKILL.md` — 融合阅读技能
- `CHANGELOG.md` — 版本记录
