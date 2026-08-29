#!/usr/bin/env python3
"""vs_semantic_v2.py — 可选 VLM 语义兜底（不进入数理化主链）。

设计原则：
  - 本扩展主链路全部输出精确数值（坐标/hex/矩阵/统计）。
  - 仅当用户明确需要"语义理解"（如"这图讲了什么故事"）时，才走此兜底通道。
  - 走本地 Ollama（127.0.0.1，硬编码，无外发）；输出 JSON 带 semantic_fallback 标记，
    下游文本模型应当将其与数值报告区分对待。

输入：image + prompt（可选）
输出：{ task: "semantic_fallback", note: "...", response: "..." }

依赖：ollama 服务 + 视觉模型（默认 qwen3-vl:8b，可配）。
"""
import argparse
import json
import sys
import urllib.request
import base64
import os


def read_config() -> dict:
    cfg_path = os.path.expanduser("~/.config/pi-vision-struct.json")
    try:
        if os.path.exists(cfg_path):
            import json as _j
            return _j.load(open(cfg_path))
    except Exception:
        pass
    return {}


def call_ollama(model: str, prompt: str, image_b64: str, host: str = "127.0.0.1:11434") -> str:
    url = f"http://{host}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode())
    return data.get("response", "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default="用一句话客观描述这张图的主要内容和可见元素。不要臆测，只描述可见事实。")
    ap.add_argument("--model", default="")
    ap.add_argument("--host", default="127.0.0.1:11434")
    args = ap.parse_args()

    cfg = read_config()
    model = args.model or cfg.get("l2_model") or "qwen3-vl:8b"

    try:
        with open(args.image, "rb") as f:
            raw = f.read()
        # 下采样：避免超出 VLM 上下文（4K 图 token 过多）
        try:
            from PIL import Image
            import io
            im = Image.open(io.BytesIO(raw))
            max_edge = 1024
            if max(im.size) > max_edge:
                im.thumbnail((max_edge, max_edge))
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode()
        except Exception:
            img_b64 = base64.b64encode(raw).decode()
    except Exception as e:
        print(json.dumps({"error": "semantic_fallback failed", "detail": f"image read: {e}"}))
        return 1

    try:
        response = call_ollama(model, args.prompt, img_b64, args.host)
    except Exception as e:
        print(json.dumps({
            "error": "semantic_fallback failed",
            "detail": str(e)[:300],
            "hint": "确保 ollama 已启动且模型已拉取：ollama pull " + model,
        }, ensure_ascii=False))
        return 1

    result = {
        "schema": "vision-report/v2",
        "task": "semantic_fallback",
        "semantic_fallback": True,
        "note": "此通道为可选 VLM 语义兜底，不参与数值主链；请与 scene_stats/pixels 等数值报告区分对待。",
        "model": model,
        "source": {"type": "image", "path": args.image},
        "prompt": args.prompt,
        "response": response,
        "truncated": False,
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
