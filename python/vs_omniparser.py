#!/usr/bin/env python3
"""vs_omniparser.py — OmniParser V2 传感器：任意截图的图标/UI 元素结构化（schema v2）。

OmniParser（微软）把 UI 截图解析为带语义的元素：YOLOv9-E 检测 + Florence-2 图标描述。
文本部分用本系统的 RapidOCR（跳过 easyocr，避免重复依赖）。
CPU 运行（torch CPU 版，不占 8GB 显存）；首载模型约 10-20s。

用法:
  vs_omniparser.py --image PATH [--box-threshold 0.05] [--max-items 60] [--no-ocr]

依赖环境: conda env `omniparser`（torch-cpu + ultralytics + transformers + rapidocr）
模型: ~/.cache/omniparser/weights/icon_detect_v3/model.pt + weights/icon_caption_florence/
（持久目录，重启不丢；重建脚本: python/setup/repair_omniparser.sh）
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# 模型已预下载到 HF 缓存（Florence-2-base + 微调权重），强制离线加载
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_MODULES_CACHE"] = f"{Path.home()}/.cache/omniparser/transformers_modules"  # trust_remote_code 模块缓存固定路径

OMNI_DIR = f"{Path.home()}/.cache/omniparser"
WEIGHTS = f"{OMNI_DIR}/weights"
DAEMON_SOCKET = f"{OMNI_DIR}/omniserver.sock"


def _unix_http(method: str, path: str, body: bytes | None = None,
               timeout: float = 3.0) -> dict:
    """经 unix socket 的 HTTP 客户端（文件系统 IPC，无任何 TCP）。"""
    import http.client
    import socket

    class _UnixConn(http.client.HTTPConnection):
        def connect(self):
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.settimeout(timeout)
            self.sock.connect(DAEMON_SOCKET)

    conn = _UnixConn("unix")
    try:
        conn.request(method, path, body=body,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        return json.loads(resp.read() or b"{}")
    finally:
        conn.close()


def daemon_health(timeout: float = 3.0) -> bool:
    try:
        d = _unix_http("GET", "/health", timeout=timeout)
        return bool(d.get("ok"))
    except Exception:
        return False


def daemon_parse(image_path: str, max_items: int, no_ocr: bool) -> dict | None:
    """调用常驻服务；失败返回 None（调用方回退直连）。"""
    try:
        body = json.dumps({"image": image_path, "max_items": max_items,
                           "no_ocr": no_ocr}).encode("utf-8")
        return _unix_http("POST", "/parse", body=body, timeout=1800)
    except Exception:
        return None


def daemon_spawn() -> bool:
    """拉起常驻服务（detached）。等待就绪最多 90s（含模型加载）。"""
    import subprocess
    import time

    log = open(f"{OMNI_DIR}/daemon.log", "a", encoding="utf-8")
    cmd = [
        sys.executable, "-u",
        f"{Path(__file__).resolve().parent}/omniserver.py",
        "--socket", DAEMON_SOCKET,
    ]
    try:
        subprocess.Popen(cmd, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                         start_new_session=True, close_fds=True)
    except Exception:
        return False
    deadline = time.time() + 90
    while time.time() < deadline:
        if daemon_health(timeout=1.0):
            return True
        time.sleep(1)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--box-threshold", type=float, default=0.05)
    ap.add_argument("--max-items", type=int, default=60)
    ap.add_argument("--no-ocr", action="store_true", help="跳过 OCR（仅图标）")
    ap.add_argument("--daemon", choices=["auto", "always", "never"], default="auto",
                    help="auto=有服务则用否则拉起；always=必须服务；never=直连")
    args = ap.parse_args()

    # 常驻服务优先（模型驻留，去掉 ~10s 冷载）
    if args.daemon != "never" and os.environ.get("VS_OMNI_DAEMON") != "0":
        if daemon_health() or daemon_spawn():
            out = daemon_parse(args.image, args.max_items, args.no_ocr)
            if out is not None:
                print(json.dumps(out, ensure_ascii=False))
                return 0
            if args.daemon == "always":
                print(json.dumps({"error": "vs_omniparser failed",
                                  "detail": "常驻服务不可用(--daemon=always)"},
                                 ensure_ascii=False))
                return 1
        # 服务失败 → 回退直连

    try:
        sys.path.insert(0, OMNI_DIR)
        # OmniParser 的 util/utils.py 顶层 import easyocr / paddleocr 并实例化（check_ocr_box 用），
        # 我们从不调用它（文本由 RapidOCR 提供）——用带占位类的模块桩避免安装整套 OCR 依赖。
        import types as _types

        def _stub(name: str, attrs: dict) -> _types.ModuleType:
            m = _types.ModuleType(name)
            for k, v in attrs.items():
                setattr(m, k, v)
            return m

        class _DummyReader:
            def __init__(self, *a, **kw):
                pass

        class _DummyPaddleOCR:
            def __init__(self, *a, **kw):
                pass

        sys.modules.setdefault("easyocr", _stub("easyocr", {"Reader": _DummyReader}))
        sys.modules.setdefault("paddleocr", _stub("paddleocr", {"PaddleOCR": _DummyPaddleOCR}))
        from PIL import Image  # type: ignore[import-not-found]
        from util.utils import get_caption_model_processor, get_som_labeled_img, get_yolo_model  # type: ignore[import-not-found]

        image = Image.open(args.image).convert("RGB")
        w, h = image.size

        som_model = get_yolo_model(model_path=f"{WEIGHTS}/icon_detect_v3/model.pt", device="cpu")
        caption_processor = get_caption_model_processor(
            model_name="florence2",
            model_name_or_path=f"{WEIGHTS}/icon_caption_florence",
            device="cpu",
        )

        ocr_bbox, ocr_text = None, []
        if not args.no_ocr:
            from rapidocr import RapidOCR  # type: ignore[import-not-found]

            engine = RapidOCR()
            tmp = "/tmp/vs_omni_ocr.png"
            image.save(tmp)
            out = engine(tmp)
            boxes = getattr(out, "boxes", None)
            if boxes is not None:
                # RapidOCR 输出 4 点四边形 → xyxy
                ocr_bbox = []
                for box in boxes.tolist():
                    xs = [p[0] for p in box]
                    ys = [p[1] for p in box]
                    ocr_bbox.append([min(xs), min(ys), max(xs), max(ys)])
                ocr_text = [t for t in out.txts]

        import contextlib

        with contextlib.redirect_stdout(sys.stderr):  # 库的进度打印不进 stdout，保持纯 JSON
            _, _, parsed = get_som_labeled_img(
            image,
            som_model,
            BOX_TRESHOLD=args.box_threshold,
            output_coord_in_ratio=True,
            ocr_bbox=ocr_bbox,
            ocr_text=ocr_text,
            caption_model_processor=caption_processor,
            use_local_semantics=True,
            iou_threshold=0.7,
            scale_img=False,
            batch_size=64,
        )

        elements = []
        for i, item in enumerate(parsed[: args.max_items]):
            bb = item.get("bbox")
            if not bb:
                continue
            bx = [round(float(bb[0]) * w), round(float(bb[1]) * h),
                  round(float(bb[2]) * w), round(float(bb[3]) * h)]
            elements.append({
                "id": i, "type": item.get("type", "icon"), "bbox": bx,
                "text": item.get("content"), "conf": None,
                "color": None, "font": None, "z": None,
                "source": ["omniparser"], "coordsys": "image_px",
                "interactivity": item.get("interactivity"),
            })

        print(json.dumps({
            "schema": "vision-report/v2", "task": "omniparser", "sensors": ["omniparser"],
            "coordsys": "image_px",
            "source": {"type": "image", "path": args.image, "size_px": [w, h],
                       "engine": "OmniParser-v2", "box_threshold": args.box_threshold},
            "elements": elements, "anomalies": [], "metrics": {"element_count": len(elements)},
            "truncated": len(elements) >= args.max_items,
        }, ensure_ascii=False))
        return 0
    except Exception as e:
        print(json.dumps({"error": "vs_omniparser failed", "detail": str(e)[:500]}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
