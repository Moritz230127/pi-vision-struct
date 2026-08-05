#!/usr/bin/env bash
# repair_omniparser.sh — 重启后重建 /tmp/OmniParser 权重（走代理 10808）
set -e
export HTTPS_PROXY=http://127.0.0.1:10808 HTTP_PROXY=http://127.0.0.1:10808
OMNI=/tmp/OmniParser
WEIGHTS=$OMNI/weights
PY=/home/Arch/conda-envs/omniparser/bin/python

echo "[1/4] clone OmniParser"
if [ ! -d "$OMNI/.git" ]; then
  git clone -q --depth 1 https://github.com/microsoft/OmniParser.git "$OMNI" 2>&1 | tail -1
fi
mkdir -p "$WEIGHTS"

echo "[2/4] download icon_detect_v3/model.pt"
if [ ! -f "$WEIGHTS/icon_detect_v3/model.pt" ]; then
  mkdir -p "$WEIGHTS/icon_detect_v3"
  $PY - << 'EOF'
from huggingface_hub import hf_hub_download
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
p = hf_hub_download(repo_id="microsoft/OmniParser-v2.0", filename="icon_detect_v3/model.pt", revision="refs/pr/37")
import shutil
shutil.copy(p, "/tmp/OmniParser/weights/icon_detect_v3/model.pt")
print("OK detect:", os.path.getsize(p))
EOF
fi

echo "[3/4] download icon_caption_florence (3 files)"
if [ ! -d "$WEIGHTS/icon_caption_florence" ]; then
  mkdir -p "$WEIGHTS/icon_caption_florence"
  $PY - << 'EOF'
from huggingface_hub import hf_hub_download
import os, shutil
for f in ["config.json", "model.safetensors", "generation_config.json"]:
    p = hf_hub_download(repo_id="microsoft/OmniParser-v2.0", filename=f"icon_caption/{f}", revision="refs/pr/37")
    shutil.copy(p, f"/tmp/OmniParser/weights/icon_caption_florence/{f}")
    print("OK", f, os.path.getsize(p))
EOF
fi

echo "[4/4] patch config.json (_name_or_path + auto_map)"
$PY - << 'EOF'
import json
p = "/tmp/OmniParser/weights/icon_caption_florence/config.json"
c = json.load(open(p))
c["_name_or_path"] = "microsoft/Florence-2-base"
c["auto_map"] = {
    "AutoConfig": "microsoft/Florence-2-base--configuration_florence2.Florence2Config",
    "AutoModelForCausalLM": "microsoft/Florence-2-base--modeling_florence2.Florence2ForConditionalGeneration",
}
json.dump(c, open(p, "w"), indent=2)
print("patched:", c["_name_or_path"])
EOF

echo "=== 完成 ==="
du -sh "$WEIGHTS"
