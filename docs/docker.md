# Docker 镜像使用说明

## 镜像内容

- 基础：python:3.12-slim (Debian trixie)
- 依赖：torch/torchvision (CPU)、onnxruntime、RapidOCR (PP-OCRv6)、paddlepaddle + PaddleOCR + PaddleX、
  ultralytics (YOLO)、transformers、open_clip、PyMuPDF、python-pptx、mss
- **已烘焙模型**（运行时完全离线）：
  - OmniParser 权重：`~/.cache/omniparser/weights/`（icon_detect_v3 + icon_caption_florence）
  - Florence-2-base（含远程代码快照 + config 补丁）
  - CLIP ViT-B-32 (laion2b_s34b_b79k) → `vs cluster` 离线可用
- 未烘焙：paddle OCR/layout 模型（~300MB）——首次运行 `layout` 自动下载，内置 `HF_ENDPOINT=https://hf-mirror.com` 加速；已有缓存可挂载宿主 `~/.paddlex`。

## 构建

```bash
docker build -t pi-vision-struct:latest .
# 国内网络构建建议直连（PyPI/pytorch 直连可达，HF 走内置镜像）；如遇网络波动：
docker build --network=host --build-arg HTTPS_PROXY=http://127.0.0.1:10808 -t pi-vision-struct:latest .
```

构建注意事项（踩坑记录）：

1. `antlr4-python3-runtime==4.9.3` 无 cp312 wheel，必须 `pip install --no-build-isolation`（预装 setuptools）从 sdist 构建，否则构建隔离连不上 PyPI 报 `No matching distribution`。
2. 构建期模型烘焙用 `hf_hub_download`/`snapshot_download`（内置 hf-mirror），不要用 COPY 宿主缓存——符号链接不进 docker 构建上下文（报 `COPY failed: file does not exist`）。
3. 构建代理：容器内 `127.0.0.1` 是容器自身，宿主机代理必须 `--network=host` 才能用。
4. pip 大文件经代理易超时：加 `--retries 10 --timeout 180`。

## 发布（GHCR）

```bash
docker tag pi-vision-struct:latest ghcr.io/moritz230127/pi-vision-struct:latest
echo "$(gh auth token)" | docker login ghcr.io -u <user> --password-stdin
docker push ghcr.io/moritz230127/pi-vision-struct:latest
```

注意：`gh auth token` 需含 `write:packages` scope（`gh auth refresh -h github.com -s write:packages`，浏览器设备码授权）。
push 失败 `permission_denied: The token provided does not match expected scopes` = scope 不足。

## 日常使用

```bash
# 环境自检
docker run --rm pi-vision-struct:latest env

# 结构化测量（输出 vision-report/v2 JSON）
docker run --rm -v "$PWD":/work pi-vision-struct:latest pix --image /work/a.png --colors 3
docker run --rm -v "$PWD":/work pi-vision-struct:latest ocr --image /work/a.png
docker run --rm -v "$PWD":/work pi-vision-struct:latest omniparser --image /work/a.png
docker run --rm -v "$PWD":/work pi-vision-struct:latest layout --image /work/a.png
docker run --rm -v "$PWD":/work pi-vision-struct:latest pdf --file /work/doc.pdf --render-dir /work/pages
docker run --rm -v "$PWD":/work pi-vision-struct:latest cluster --files "/work/a.png,/work/b.png"

# 直接跑脚本
docker run --rm -v "$PWD":/work pi-vision-struct:latest python/vs_pix.py --image /work/a.png
```

所有 action 对应 `python/vs_*.py`（入口 `docker-entry.sh` 分发）。
