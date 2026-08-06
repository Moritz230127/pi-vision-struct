# pi-vision-struct — 视觉工具套件容器镜像
# 构建:  docker build --network=host --build-arg HTTPS_PROXY=http://127.0.0.1:10808 -t pi-vision-struct:latest .
# 用法:  docker run --rm -v "$PWD":/work pi-vision-struct:latest <action> --image /work/x.png
# 发布:  docker tag pi-vision-struct:latest ghcr.io/moritz230127/pi-vision-struct:latest
#        docker push ghcr.io/moritz230127/pi-vision-struct:latest
FROM python:3.12-slim

# 构建代理（可选）：--build-arg HTTPS_PROXY=http://127.0.0.1:10808
ARG HTTPS_PROXY
ARG HTTP_PROXY
ENV HTTPS_PROXY=${HTTPS_PROXY} HTTP_PROXY=${HTTP_PROXY} \
    HF_ENDPOINT=https://hf-mirror.com \
    PYTHONUNBUFFERED=1

# 系统依赖（opencv/paddle 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/pi-vision-struct
COPY . .

# 依赖（torch CPU + 核心 + paddle + pymupdf + mss）
# antlr4-python3-runtime 无 cp312 wheel，需 --no-build-isolation 从 sdist 构建
RUN pip install --no-cache-dir --retries 10 --timeout 180 setuptools wheel \
    && pip install --no-cache-dir --retries 10 --timeout 180 --no-build-isolation antlr4-python3-runtime==4.9.3 \
    && pip install --no-cache-dir --retries 10 --timeout 180 \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --retries 10 --timeout 180 -r python/requirements.txt \
    && pip install --no-cache-dir --retries 10 --timeout 180 -r python/requirements-omniparser.txt \
    && pip install --no-cache-dir --retries 10 --timeout 180 pymupdf mss

# 模型烘焙（构建期联网下载 → 镜像自包含，运行时完全离线）
RUN mkdir -p /root/.cache/huggingface /root/.cache/omniparser && \
    python - << 'EOF'
import json, os, shutil
from huggingface_hub import hf_hub_download, snapshot_download

# OmniParser 权重（检测器 + 图标描述）
w = "/root/.cache/omniparser/weights"
os.makedirs(f"{w}/icon_detect_v3", exist_ok=True)
hf_hub_download("microsoft/OmniParser-v2.0", "icon_detect_v3/model.pt",
                revision="refs/pr/37", local_dir=f"{w}/icon_detect_v3")
for f in ["config.json", "generation_config.json", "model.safetensors"]:
    hf_hub_download("microsoft/OmniParser-v2.0", f"icon_caption/{f}",
                    revision="refs/pr/37", local_dir=f"{w}/icon_caption_florence")

# Florence-2-base（处理器 + 远程代码 + 权重）
snapshot_download("microsoft/Florence-2-base")

# config 补丁（离线远程代码加载必需）
p = f"{w}/icon_caption_florence/config.json"
c = json.load(open(p))
c["_name_or_path"] = "microsoft/Florence-2-base"
c["auto_map"] = {
    "AutoConfig": "microsoft/Florence-2-base--configuration_florence2.Florence2Config",
    "AutoModelForCausalLM": "microsoft/Florence-2-base--modeling_florence2.Florence2ForConditionalGeneration",
}
json.dump(c, open(p, "w"), indent=2)

# CLIP（vs_cluster 用）
snapshot_download("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
print("models baked")
EOF

# 自检（构建时验证依赖可导入）
RUN python -c "import PIL,numpy,onnxruntime,rapidocr,pptx,fitz,mss,torch,transformers,ultralytics,paddle,paddleocr,paddlex,open_clip; print('deps OK')"

COPY docker-entry.sh /usr/local/bin/docker-entry.sh
RUN chmod +x /usr/local/bin/docker-entry.sh
ENTRYPOINT ["/usr/local/bin/docker-entry.sh"]
WORKDIR /work
