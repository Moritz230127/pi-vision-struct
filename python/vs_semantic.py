#!/usr/bin/env python3
"""vs_semantic.py — L2 语义标签（opt-in，本地 Ollama qwen3-vl:8b）

只处理不可测量的属性（风格/主题/类型）；可测量的属性一律走 L0/L1 工具。
默认 opt-in：需要显式 --enable 或环境变量 PI_VISION_SEMANTIC=1，否则拒绝执行。
只连 localhost:11434，无网络外发。
8GB VRAM 约束：图片最长边缩到 768px、num_ctx=8192、num_predict=300。

用法:
  vs_semantic.py --image PATH [--prompt '...'] [--enable] [--model qwen3-vl:8b]
                 [--max-tokens 300] [--timeout 120]
"""
import argparse
import base64
import io
import json
import os
import sys
import urllib.request

from PIL import Image  # type: ignore[import-not-found]

DEFAULT_PROMPT = (
    "分析这张图，只输出 JSON，不要输出其他任何内容。"
    '字段：style(风格,如 极简/扁平/插画/摄影/渐变/动漫/抽象)、'
    "theme(主题,如 风景/人物/几何/科技/自然/城市/星空/文字)、"
    "type(类型,如 照片/插画/矢量/文字/界面截图)、"
    "keywords(3-5 个逗号分隔的关键词)。"
    '输出格式：{"style":"","theme":"","type":"","keywords":""} 只填值。'
)

# qwen3-vl:8b 为思考型变体：复杂提示词会先输出大量 <think> 再回答。
# 实测（Ollama 0.32.5）：num_predict 需 ≥2048 才能让思考结束并产出最终 JSON。
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TIMEOUT = 300


def _try_parse_json(text: str) -> dict | None:
    """从模型输出提取 JSON。先整体解析；失败则剥掉 markdown 围栏再试。"""
    text = text.strip()
    if not text:
        return None
    candidates = [text]
    if text.startswith("```"):
        lines = text.splitlines()
        candidates.append("\n".join(l for l in lines if not l.strip().startswith("```")))
    for cand in candidates:
        try:
            parsed = json.loads(cand)
            if isinstance(parsed, dict):
                return parsed
        except ValueError:
            continue
    return None


def classify(
    image_path: str,
    prompt: str | None = None,
    model: str = "qwen3-vl:8b",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_TIMEOUT,
    num_ctx: int = 8192,
    max_side: int = 768,
) -> dict:
    """调用本地 Ollama qwen3-vl:8b 生成语义标签。永不抛异常，失败返回 {"ok": False, ...}。"""
    try:
        im = Image.open(image_path).convert("RGB")
        w, h = im.size
        scale = max_side / max(w, h)
        if scale < 1:
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)  # type: ignore[attr-defined]
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        body = {
            "model": model,
            "prompt": prompt or DEFAULT_PROMPT,
            "images": [b64],
            "stream": False,
            "think": False,  # 此 Ollama 版本对该模型无效（思考型变体），保留无害
            "options": {
                "num_ctx": num_ctx,
                "num_predict": max_tokens,
                "temperature": 0.2,
            },
        }
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = str(data.get("response", "")).strip()
        return {"ok": True, "raw": text, "parsed": _try_parse_json(text), "model": model,
                "eval_count": data.get("eval_count"), "done_reason": data.get("done_reason")}
    except Exception as e:
        return {"ok": False, "error": str(e)[:500]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt")
    ap.add_argument("--enable", action="store_true", help="显式开启 L2 语义（默认 opt-in 关闭）")
    ap.add_argument("--model", default="qwen3-vl:8b")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    args = ap.parse_args()

    try:
        im = Image.open(args.image)
        size_px = list(im.size)
    except Exception as e:
        print(json.dumps({"error": "vs_semantic failed", "detail": str(e)[:300]}, ensure_ascii=False))
        return 1

    if not (args.enable or os.environ.get("PI_VISION_SEMANTIC") == "1"):
        print(json.dumps({
            "schema": "vision-report/v1",
            "source": {"type": "image", "path": args.image, "size_px": size_px},
            "semantic": {"enabled": False,
                         "reason": "opt-in: 传 --enable 或设 PI_VISION_SEMANTIC=1"},
        }, ensure_ascii=False))
        return 0

    result = classify(args.image, prompt=args.prompt, model=args.model,
                      max_tokens=args.max_tokens, timeout=args.timeout)
    print(json.dumps({
        "schema": "vision-report/v1",
        "source": {"type": "image", "path": args.image, "size_px": size_px},
        "semantic": {"enabled": True, **result},
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
