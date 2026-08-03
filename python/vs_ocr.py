#!/usr/bin/env python3
"""vs_ocr.py — RapidOCR（PP-OCRv4, onnxruntime）带坐标框的文本提取。

用法:
  vs_ocr.py --image PATH [--region x1,y1,x2,y2] [--upscale 2] [--max-items 100] [--min-conf 0.5]

--region 先裁剪再识别（配合 --upscale 放大提高小字号召回率）；
坐标始终换算回原始图像空间。输出 JSON。
"""
import argparse
import json
import sys

from PIL import Image


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--region")
    ap.add_argument("--upscale", type=int, default=2)
    ap.add_argument("--max-items", type=int, default=100)
    ap.add_argument("--min-conf", type=float, default=0.5)
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
        if args.upscale and args.upscale > 1:
            im = im.resize((im.width * args.upscale, im.height * args.upscale), Image.LANCZOS)  # type: ignore[attr-defined]

        tmp = "/tmp/vs_ocr_input.png"
        im.save(tmp)

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

        els = [{"id": i, "type": "text", "bbox": it["bbox"], "text": it["text"],
                "conf": it["conf"], "color": None, "font": None, "z": None,
                "source": ["ocr"], "coordsys": "image_px", "center": it["center"], "quad": it["quad"]}
               for i, it in enumerate(items)]
        print(json.dumps({"schema": "vision-report/v2", "task": "ocr", "sensors": ["ocr"],
                          "coordsys": "image_px",
                          "source": {"type": "image", "path": args.image, "size_px": [w, h]},
                          "elements": els, "anomalies": [], "metrics": {},
                          "truncated": len(items) >= args.max_items},
                         ensure_ascii=False))
        return 0
    except Exception as e:
        print(json.dumps({"error": "vs_ocr failed", "detail": str(e)[:500]}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
