#!/usr/bin/env bash
# apply_omniparser_patch.sh — 重放 OmniParser 部署补丁（VIS-P22）
#
# 背景：OmniParser 在 niri + Arch 上的部署需要 3 处补丁，其中 modeling_florence2.py
# 补丁位于 HF 缓存（transformers_modules），transformers 升级或缓存清理后会丢失。
# 本脚本将全部补丁做成可重放形式，升级后一键恢复。
#
# 用法: bash apply_omniparser_patch.sh [--check]
#   --check  只检查补丁状态，不应用

set -euo pipefail

OMNI="$HOME/.cache/omniparser"
WEIGHTS="$OMNI/weights/icon_caption_florence"
REPO="$HOME/.cache/omniparser-repo"
MODELING="$OMNI/transformers_modules/transformers_modules/icon_caption_florence/modeling_florence2.py"
PY=/home/Arch/conda-envs/omniparser/bin/python

CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

echo "== OmniParser patch check =="

# ---- 1. util/utils.py: processor 从本地路径加载 ----
UTILS="$REPO/util/utils.py"
if grep -q 'AutoProcessor.from_pretrained(model_name_or_path' "$UTILS" 2>/dev/null; then
    echo "  [OK] utils.py processor 本地路径"
else
    echo "  [MISS] utils.py processor 本地路径"
    if [ "$CHECK_ONLY" = 0 ]; then
        sed -i 's|processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)|processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)|' "$UTILS"
        echo "  [FIXED] utils.py"
    fi
fi

# ---- 2. config.json: _name_or_path 指向本地 + auto_map 去前缀 ----
CFG="$WEIGHTS/config.json"
if [ -f "$CFG" ] && grep -q 'omniparser/weights/icon_caption_florence' "$CFG" 2>/dev/null; then
    echo "  [OK] config.json 本地路径"
else
    echo "  [MISS] config.json 本地路径"
    if [ "$CHECK_ONLY" = 0 ]; then
        "$PY" - <<'EOF'
import json, os
p = os.path.expanduser("~/.cache/omniparser/weights/icon_caption_florence/config.json")
with open(p) as f:
    cfg = json.load(f)
cfg["_name_or_path"] = os.path.expanduser("~/.cache/omniparser/weights/icon_caption_florence")
am = cfg.get("auto_map", {})
for k, v in am.items():
    am[k] = v.replace("microsoft/Florence-2-base-ft--", "")
with open(p, "w") as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)
print("  [FIXED] config.json")
EOF
    fi
fi

# ---- 3. modeling_florence2.py: 注意力掩码扩展补丁 ----
if [ -f "$MODELING" ] && grep -q "Compatibility fix (transformers >= 4.4x)" "$MODELING" 2>/dev/null; then
    echo "  [OK] modeling_florence2.py 掩码补丁"
else
    echo "  [MISS] modeling_florence2.py 掩码补丁"
    if [ "$CHECK_ONLY" = 0 ]; then
        "$PY" - <<'EOF'
import os, re
p = os.path.expanduser("~/.cache/omniparser/transformers_modules/transformers_modules/icon_caption_florence/modeling_florence2.py")
with open(p) as f:
    src = f.read()
old = """        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)"""
new = """        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                # Compatibility fix (transformers >= 4.4x): the processor produces a
                # text-only 4D mask; image tokens are prepended by the model, so the
                # mask must be expanded to cover them (image part unmasked).
                if attention_mask.size(-1) < src_len:
                    pad = src_len - attention_mask.size(-1)
                    ones = torch.ones(
                        bsz, 1, tgt_len, pad,
                        device=attention_mask.device, dtype=attention_mask.dtype,
                    )
                    attention_mask = torch.cat([ones, attention_mask], dim=-1)
                    ones2 = torch.ones(
                        bsz, 1, pad, src_len,
                        device=attention_mask.device, dtype=attention_mask.dtype,
                    )
                    attention_mask = torch.cat([ones2, attention_mask], dim=-2)
                else:
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                    )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)"""
if old in src:
    src = src.replace(old, new)
    with open(p, "w") as f:
        f.write(src)
    print("  [FIXED] modeling_florence2.py")
else:
    print("  [WARN] modeling_florence2.py 补丁模式未匹配（可能已应用或结构变化）")
EOF
    fi
fi

echo "== done =="
