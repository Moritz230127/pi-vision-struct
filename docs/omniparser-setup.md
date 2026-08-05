# OmniParser 传感器维护手册（vs_omniparser.py）

## 用途

对**任意截图**（无需 DOM）提取图标级 UI 元素 + 语义描述（OmniParser V2：YOLOv9-E 检测 + Florence-2 图标描述）+ RapidOCR 文本。CPU 运行（不占 8GB 显存），首次加载模型 10-20s，单图 30-60s。

## 环境

- conda env：`/home/Arch/conda-envs/omniparser`（python 3.12）
- 依赖：`torch(cpu) transformers==4.46.1 huggingface_hub==0.28.1 ultralytics onnxruntime rapidocr supervision timm einops opencv-python-headless openai azure-identity`
- **版本钉死原因**：transformers 5.x 的 Florence2 配置结构变了（`Florence2LanguageConfig` KeyError）；4.56 无语言配置类；**4.46.1 与检查点（transformers_version=4.46.1）精确匹配**，走远程代码路径
- 下载走代理：`export HTTPS_PROXY=http://127.0.0.1:10808 HTTP_PROXY=http://127.0.0.1:10808`

## 模型（/tmp/OmniParser/weights/）

| 模型 | 路径 | 来源 |
| --- | --- | --- |
| 检测器 | `icon_detect_v3/model.pt` (268MB) | `hf_hub_download("microsoft/OmniParser-v2.0", "icon_detect_v3/model.pt", revision="refs/pr/37")` |
| 图标描述 | `icon_caption_florence/` (1.1GB) | `hf_hub_download` 三个文件后 `mv icon_caption icon_caption_florence` |
| 处理器 | `microsoft/Florence-2-base` (887MB, HF 缓存) | `snapshot_download`（离线需要） |

## 关键坑（均已解决，重装时按此操作）

1. **transformers 版本**：必须 4.46.1（检查点匹配）；hub 0.28.1（0.36 的 local_files_only 校验过严）
2. **HF 缓存 refs/main 不能有尾随换行**：`printf '<hash>' > .../refs/main`（echo 会加 \n 导致离线解析失败）
3. **检查点 config.json 两处补丁**（/tmp/OmniParser/weights/icon_caption_florence/config.json）：
   - `_name_or_path`: `microsoft/Florence-2-base-ft` → `microsoft/Florence-2-base`（已缓存的仓库）
   - `auto_map` 前缀：`microsoft/Florence-2-base-ft--...` → `microsoft/Florence-2-base--...`
4. **离线加载**：vs_omniparser.py 顶部设 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` / `HF_MODULES_CACHE=/tmp/OmniParser/transformers_modules`
5. **easyocr/paddleocr 桩**：util/utils.py 顶层实例化它们，脚本用模块桩跳过（文本由 RapidOCR 提供）
6. **stdout 纯净**：库的进度打印用 `contextlib.redirect_stdout(sys.stderr)` 隔离，stdout 只输出 JSON

## 重装脚本（快照）

```bash
mamba create -n omniparser python=3.12 -y
# 代理下：
/home/Arch/conda-envs/omniparser/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
/home/Arch/conda-envs/omniparser/bin/python -m pip install "transformers==4.46.1" "huggingface_hub==0.28.1" ultralytics onnxruntime rapidocr supervision timm einops opencv-python-headless openai azure-identity
# 模型 + 补丁见上文；Florence-2-base 用 snapshot_download 缓存
```

## 重启后一键修复（推荐）

/tmp 重启即清空（权重 1.3GB 丢失），一键重建：

```bash
~/.pi-extensions/pi-vision-struct/python/setup/repair_omniparser.sh
# 后台执行: nohup .../repair_omniparser.sh > /tmp/repair_omni.log 2>&1 &
# 完成后验证: tail -3 /tmp/repair_omni.log（应有 patched 与完成标记）
```

脚本自动: clone → 下载检测器+图标权重(代理 10808) → 打 config 补丁。约 2-5 分钟。
env 依赖与 HF 缓存(~/.cache)重启不丢，无需重装。
