#!/usr/bin/env python3
"""omniserver.py — OmniParser 常驻服务（本地 localhost，模型只加载一次）。

vs_omniparser 每次调用都冷载 YOLO + Florence-2（~10s）。本服务把模型驻留内存，
调用降为纯推理。仅绑定 127.0.0.1，无任何外发。

API:
  GET  /health         → {"ok": true, "pid": N, "models": ["yolo","florence2"]}
  POST /parse          → body JSON: {image, max_items?, no_ocr?}
                         → schema v2 elements（与 vs_omniparser 输出一致）
  POST /shutdown       → 优雅退出

启动:
  nohup /home/Arch/conda-envs/omniparser/bin/python -u python/omniserver.py \
        > ~/.cache/omniparser/daemon.log 2>&1 &

停止: pkill -f omniserver.py
"""
import argparse
import base64
import io
import json
import os
import sys
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
OMNI_DIR = f"{os.path.expanduser('~')}/.cache/omniparser"
os.environ["HF_MODULES_CACHE"] = f"{OMNI_DIR}/transformers_modules"

PORT = 8765
MODELS: dict = {}


def load_models() -> None:
    """惰性加载（与 vs_omniparser.py 相同的桩/补丁逻辑）。"""
    sys.path.insert(0, OMNI_DIR)

    def _stub(name: str, attrs: dict) -> types.ModuleType:
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        return m

    class _DummyReader:  # easyocr 桩
        def __init__(self, *a, **k):
            pass

    class _DummyPaddleOCR:  # paddleocr 桩
        def __init__(self, *a, **k):
            pass

    sys.modules.setdefault("easyocr", _stub("easyocr", {"Reader": _DummyReader}))
    sys.modules.setdefault("paddleocr", _stub("paddleocr", {"PaddleOCR": _DummyPaddleOCR}))
    from PIL import Image  # type: ignore[import-not-found]
    from util.utils import (  # type: ignore[import-not-found]
        get_caption_model_processor,
        get_som_labeled_img,
        get_yolo_model,
    )

    MODELS["yolo"] = get_yolo_model(
        model_path=f"{OMNI_DIR}/weights/icon_detect_v3/model.pt", device="cpu")
    MODELS["caption"] = get_caption_model_processor(
        model_name="florence2",
        model_name_or_path=f"{OMNI_DIR}/weights/icon_caption_florence",
        device="cpu")
    MODELS["image"] = Image
    MODELS["get_som"] = get_som_labeled_img


def parse_image(image_path: str, max_items: int, no_ocr: bool) -> dict:
    import contextlib
    import sys as _sys

    if "get_som" not in MODELS:
        load_models()
    im = MODELS["image"].open(image_path).convert("RGB")
    ocr_bbox, ocr_text = None, []
    if not no_ocr:
        from rapidocr import RapidOCR  # type: ignore[import-not-found]

        tmp = "/tmp/vs_omni_ocr.png"
        im.save(tmp)
        out = RapidOCR()(tmp)
        boxes = getattr(out, "boxes", None)
        if boxes is not None:
            ocr_bbox = []
            for box in boxes.tolist():
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                ocr_bbox.append([min(xs), min(ys), max(xs), max(ys)])
            ocr_text = [t for t in out.txts]
    with contextlib.redirect_stdout(_sys.stderr):
        _, _, parsed = MODELS["get_som"](
            im, MODELS["yolo"], BOX_TRESHOLD=0.05, output_coord_in_ratio=True,
            ocr_bbox=ocr_bbox, ocr_text=ocr_text,
            caption_model_processor=MODELS["caption"],
            use_local_semantics=True, iou_threshold=0.7, scale_img=False,
            batch_size=32)
    w, h = im.size
    els = []
    for i, it in enumerate(parsed):
        if i >= max_items:
            break
        bb = it.get("bbox")
        if not bb:
            continue
        bx = [round(float(bb[0]) * w), round(float(bb[1]) * h),
              round(float(bb[2]) * w), round(float(bb[3]) * h)]
        els.append({"id": i, "type": it.get("type", "icon"),
                    "bbox": bx, "text": it.get("content"),
                    "conf": None, "color": None, "font": None, "z": None,
                    "source": ["omniparser"], "coordsys": "image_px",
                    "interactivity": bool(it.get("interactivity"))})
    return {"schema": "vision-report/v2", "task": "omniparser",
            "sensors": ["omniparser"], "coordsys": "image_px",
            "source": {"type": "image", "path": image_path, "size_px": [w, h],
                       "engine": "OmniParser-v2-daemon", "box_threshold": 0.05},
            "elements": els, "anomalies": [], "metrics": {},
            "truncated": len(els) >= max_items}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # noqa: A002
        pass

    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/health"):
            self._send(200, {"ok": True, "pid": os.getpid(),
                             "models": sorted(MODELS.keys())})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, TypeError):
            self._send(400, {"error": "bad request"})
            return
        if self.path.startswith("/parse"):
            try:
                self._send(200, parse_image(
                    str(req.get("image", "")),
                    int(req.get("max_items", 60)),
                    bool(req.get("no_ocr", False))))
            except Exception as e:
                self._send(500, {"error": "parse failed", "detail": str(e)[:400]})
        elif self.path.startswith("/shutdown"):
            self._send(200, {"ok": True})
            import threading
            threading.Thread(target=lambda: (os._exit(0))).start()  # noqa: PLR5501
        else:
            self._send(404, {"error": "not found"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    load_models()  # 启动即加载，首次调用不再等待
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(json.dumps({"ok": True, "pid": os.getpid(), "port": args.port,
                       "models": sorted(MODELS.keys())}), flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
