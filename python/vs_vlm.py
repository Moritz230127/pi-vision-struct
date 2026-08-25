#!/usr/bin/env python3
"""vs_vlm.py — L2 共享 VLM 网关：本地 Ollama 调用唯一出口。

vs_semantic（语义标签）与 vs_critic（VLM 复核裁决）共用同一后端调用，
消除重复的 HTTP 拼装与编码逻辑。只读、仅 localhost、永不抛异常。
"""
import base64
import io
import json
import urllib.request

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3-vl:8b"


def default_model() -> str:
    """L2 模型解析顺序：config `l2_model` > 内置默认。U1：换档只改配置。"""
    import json
    import os

    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    if os.name == "nt":
        base = os.environ.get("APPDATA", base)
    try:
        with open(os.path.join(base, "pi-vision-struct.json"), encoding="utf-8") as f:
            m = json.load(f).get("l2_model")
        if isinstance(m, str) and m.strip():
            return m.strip()
    except Exception:
        pass
    return DEFAULT_MODEL


def generate(prompt: str, image_b64: str | None = None, *,
             model: str | None = None,
             base_url: str = DEFAULT_BASE_URL,
             num_ctx: int = 8192, max_tokens: int = 512,
             temperature: float = 0.2, timeout: int = 120) -> dict:
    """单次生成。成功返回 {ok,text,model,eval_count,done_reason}；失败 {ok:False,error}。"""
    model = model or default_model()
    try:
        body: dict = {"model": model, "prompt": prompt, "stream": False,
                      "options": {"num_ctx": num_ctx, "num_predict": max_tokens,
                                  "temperature": temperature}}
        if image_b64:
            body["images"] = [image_b64]
        req = urllib.request.Request(
            base_url.rstrip("/") + "/api/generate",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return {"ok": True, "text": str(data.get("response", "")).strip(),
                "eval_count": data.get("eval_count"),
                "done_reason": data.get("done_reason")}
    except Exception as e:
        return {"ok": False, "error": str(e)[:500]}


def image_to_b64(path: str, max_side: int = 768) -> str:
    """读图 → RGB → 可选等比缩放（LANCZOS）→ PNG base64。失败抛异常，调用方处理。"""
    from PIL import Image  # type: ignore[import-not-found]

    im = Image.open(path).convert("RGB")
    w, h = im.size
    scale = max_side / max(w, h)
    if scale < 1:
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                       Image.LANCZOS)  # type: ignore[attr-defined]
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
