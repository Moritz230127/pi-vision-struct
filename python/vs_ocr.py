#!/usr/bin/env python3
"""vs_ocr.py — 带坐标框的文本提取（双后端，schema v2）。

后端：
  rapidocr (默认)  PP-OCRv6 small ONNX，CPU 快（~2-5s），召回中
  paddle           PaddleOCR PP-OCRv6 medium，CPU 慢（~7-40s），召回高
                 （实测真实截图 96 vs 60 元素；需 enable_mkldnn=False 避开
                   paddlepaddle oneDNN bug）

用法:
  vs_ocr.py --image PATH [--region x1,y1,x2,y2] [--upscale 2]
            [--max-items 100] [--min-conf 0.5] [--backend rapidocr|paddle]

--region 先裁剪再识别（配合 --upscale 放大提高小字号召回率）；
坐标始终换算回原始图像空间。输出 JSON。
"""
import argparse
import json
import os
import sys
from pathlib import Path

import vs_schema as S

from PIL import Image


# ---- OCR 后处理：词典纠错（编辑距离）----

# 常用中文/英文词典（轻量版，可扩展）
COMMON_WORDS = {
    "提交", "取消", "确定", "保存", "删除", "编辑", "搜索", "设置", "登录", "注册",
    "退出", "返回", "下一步", "上一步", "完成", "取消", "确认", "关闭", "打开", "新建",
    "submit", "cancel", "ok", "save", "delete", "edit", "search", "settings",
    "login", "logout", "back", "next", "done", "close", "open", "new", "yes", "no",
}


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def dict_correct(text: str, max_dist: int = 1) -> tuple[str, float]:
    """词典纠错：编辑距离 ≤ max_dist 的词典词替换。返回 (修正文本, 置信度)。"""
    t = text.strip()
    if not t or len(t) < 2:
        return text, 1.0
    best_word, best_dist = None, max_dist + 1
    for word in COMMON_WORDS:
        d = _levenshtein(t.lower(), word.lower())
        if d < best_dist:
            best_dist = d
            best_word = word
    if best_word and best_dist <= max_dist:
        return best_word, 0.9  # 纠错后置信度 0.9
    return text, 1.0


def postprocess_items(items: list[dict]) -> list[dict]:
    """OCR 后处理：词典纠错 + 置信度加权。"""
    out = []
    for it in items:
        text = it.get("text", "")
        corrected, conf_factor = dict_correct(text)
        it["text"] = corrected
        if conf_factor < 1.0:
            it["conf"] = round(it.get("conf", 0.5) * conf_factor, 3)
            it["corrected"] = True
        out.append(it)
    return out


_PADDLE_ENGINE = None


def get_paddle_engine():
    """PaddleOCR 引擎进程内单例（ocrserver 驻留时全程只载一次）。"""
    global _PADDLE_ENGINE
    if _PADDLE_ENGINE is None:
        from paddleocr import PaddleOCR  # type: ignore[import-not-found]

        _PADDLE_ENGINE = PaddleOCR(lang="ch", enable_mkldnn=False)
    return _PADDLE_ENGINE


def paddle_predict(img_path: str, min_conf: float = 0.5) -> list[dict]:
    """PP-OCRv6 medium 高召回推理 → items（与 rapid 路径同构）。"""
    engine = get_paddle_engine()
    res = engine.predict(img_path)
    first = res[0] if isinstance(res, list) else res
    texts = first.get("rec_texts", []) or []
    polys = first.get("dt_polys", []) or []
    scores = first.get("rec_scores", []) or []
    items = []
    for i, txt in enumerate(texts):
        score = float(scores[i]) if i < len(scores) else 0.0
        if score < min_conf:
            continue
        if i < len(polys) and polys[i] is not None:
            box = [[round(float(p[0])), round(float(p[1]))] for p in polys[i]]
        else:
            box = None
        xs = [p[0] for p in box] if box else [0]
        ys = [p[1] for p in box] if box else [0]
        items.append({
            "text": txt, "conf": round(score, 3),
            "bbox": [min(xs), min(ys), max(xs), max(ys)] if box else None,
            "quad": box, "center": [round((min(xs) + max(xs)) / 2),
                                     round((min(ys) + max(ys)) / 2)] if box else None,
        })
    return items


def _ocr_paddle(img_path: str, args) -> list[dict]:
    """CLI 兼容入口。惰性导入，避免拖慢默认路径。"""
    return paddle_predict(img_path, min_conf=args.min_conf)


# ---- paddle 常驻服务客户端（unix socket 行分隔 JSON，与 omniserver 同模式）----
DAEMON_DIR = f"{os.path.expanduser('~')}/.cache/vs-ocr"
DAEMON_SOCKET = f"{DAEMON_DIR}/ocrserver.sock"


def daemon_health(timeout: float = 1.0) -> bool:
    import socket

    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect(DAEMON_SOCKET)
        s.sendall(b'{"ping": true}\n')
        line = s.makefile("rb").readline()
        s.close()
        return b'"ok"' in line
    except Exception:
        return False


def daemon_spawn() -> bool:
    """拉起常驻服务（detached）。等待就绪最多 90s（含模型加载）。"""
    import subprocess
    import time

    os.makedirs(DAEMON_DIR, exist_ok=True)
    log = open(f"{DAEMON_DIR}/daemon.log", "a", encoding="utf-8")
    cmd = [sys.executable, "-u",
           f"{Path(__file__).resolve().parent}/ocrserver.py"]
    try:
        subprocess.Popen(cmd, stdout=log, stderr=log,
                         stdin=subprocess.DEVNULL,
                         start_new_session=True, close_fds=True)
    except Exception:
        return False
    deadline = time.time() + 90
    while time.time() < deadline:
        if daemon_health():
            return True
        time.sleep(1)
    return False


def daemon_ocr(image: str, min_conf: float) -> list[dict] | None:
    """经常驻服务推理；任何失败返回 None（调用方回退冷载）。"""
    import socket

    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(120.0)
        s.connect(DAEMON_SOCKET)
        payload = json.dumps({"image": image, "min_conf": min_conf}) + "\n"
        s.sendall(payload.encode())
        with s.makefile("rb") as f:
            data = json.loads(f.readline().decode())
        s.close()
        return data["items"] if data.get("ok") else None
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--region")
    ap.add_argument("--upscale", type=int, default=2)
    ap.add_argument("--max-items", type=int, default=100)
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--backend", default="rapidocr", choices=["rapidocr", "paddle"])
    ap.add_argument("--preprocess", default="none", choices=["none", "contrast"],
                    help="none=原图; contrast=自动对比度拉伸（低对比度文字用）")
    ap.add_argument("--daemon", choices=["auto", "always", "never"], default="auto",
                    help="auto=有服务则用否则拉起；always=必须服务；never=直连冷载")
    args = ap.parse_args()

    try:
        im = Image.open(args.image).convert("RGB")
        w, h = im.size
        ox = oy = 0
        crop = None
        if args.region:
            x1, y1, x2, y2 = (int(v) for v in args.region.split(","))
            ox, oy = min(x1, x2), min(y1, y2)
            crop = (ox, oy, max(x1, x2), max(y1, y2))
            im = im.crop(crop)
        if args.preprocess == "contrast":
            from PIL import ImageOps  # type: ignore[import-not-found]

            im = ImageOps.autocontrast(im, cutoff=1)
        if args.upscale and args.upscale > 1:
            im = im.resize((im.width * args.upscale, im.height * args.upscale), Image.LANCZOS)  # type: ignore[attr-defined]

        tmp = "/tmp/vs_ocr_input.png"
        im.save(tmp)

        if args.backend == "paddle":
            served = None
            if args.daemon != "never" and os.environ.get("VS_OCR_DAEMON") != "0":
                if daemon_health() or daemon_spawn():
                    served = daemon_ocr(tmp, args.min_conf)
            if served is not None:
                items = served
            elif args.daemon == "always":
                raise RuntimeError("ocrdaemon unavailable (--daemon always)")
            else:
                items = paddle_predict(tmp, min_conf=args.min_conf)
        else:
            from rapidocr import RapidOCR  # type: ignore[import-not-found]

            engine = RapidOCR()
            out = engine(tmp)
            items = []
            boxes = getattr(out, "boxes", None)
            if boxes is not None:
                for box, text, score in zip(boxes, out.txts, out.scores):
                    score = float(score)
                    if score < args.min_conf:
                        continue
                    # 换算回原始坐标
                    box = [[round(p[0] / args.upscale) + ox, round(p[1] / args.upscale) + oy] for p in box.tolist()]
                    xs = [p[0] for p in box]
                    ys = [p[1] for p in box]
                    cx = round(sum(xs) / 4)
                    cy = round(sum(ys) / 4)
                    items.append({"text": text, "conf": round(score, 3),
                                  "bbox": [min(xs), min(ys), max(xs), max(ys)],
                                  "quad": box, "center": [cx, cy]})
                    if len(items) >= args.max_items:
                        break

        items = items[: args.max_items]
        items = postprocess_items(items)  # 词典纠错 + 置信度加权
        els = [{"id": i, "type": "text", "bbox": it["bbox"], "text": it["text"],
                "conf": it["conf"], "color": None, "font": None, "z": None,
                "source": ["ocr"], "coordsys": "image_px", "center": it["center"], "quad": it["quad"]}
               for i, it in enumerate(items)]
        print(S.dump_json({"schema": "vision-report/v3", "task": "ocr", "sensors": ["ocr"],
                          "coordsys": "image_px",
                          "source": {"type": "image", "path": args.image, "size_px": [w, h],
                                      "backend": args.backend},
                          "elements": els, "anomalies": [], "metrics": {},
                          "truncated": len(items) >= args.max_items}))
        return 0
    except Exception as e:
        print(json.dumps({"error": "vs_ocr failed", "detail": str(e)[:500]}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
