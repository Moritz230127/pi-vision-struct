#!/usr/bin/env python3
"""vs_detect.py — 自然图像开放词表检测传感器（U2）。

OWLv2 zero-shot 检测（transformers，omniparser env）：任意文本类别 → bbox+置信度。
权重 google/owlv2-base-patch16-ensemble 首用经 HF 下载（~1.2GB，支持 --proxy/
HF_ENDPOINT 镜像），此后离线。选型备注：ultralytics YOLO-World 在 8.4.x/8.3.x
实测 set_classes 嵌入异常（开放词分数≈0.01 噪声），故采用 OWLv2。

用法:
  vs_detect.py --image IMG --classes "person,car,dog" [--threshold 0.25] [--max-items 50]
输出:
  schema v2 elements：type="object", text=类别名, bbox=[x1,y1,x2,y2], conf
"""
import argparse
import json
import sys

import vs_schema as S

MODEL_ID = "google/owlv2-base-patch16-ensemble"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--classes", required=True,
                    help="逗号分隔的开放词表类别，如 person,car,dog")
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--max-items", type=int, default=50)
    args = ap.parse_args()

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not classes:
        print(json.dumps({"error": "vs_detect failed", "detail": "classes 为空"},
                         ensure_ascii=False))
        return 1

    try:
        import torch
        from PIL import Image
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
    except ImportError as e:
        print(json.dumps(
            {"error": "vs_detect failed", "detail": f"依赖缺失: {e}",
             "hint": "pip install transformers torch Pillow （omniparser env）"},
            ensure_ascii=False))
        return 1

    try:
        proc = Owlv2Processor.from_pretrained(MODEL_ID)
        model = Owlv2ForObjectDetection.from_pretrained(MODEL_ID)
        model.eval()
        im = Image.open(args.image).convert("RGB")
        inputs = proc(text=[classes], images=im, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs)
        res = proc.post_process_object_detection(
            out, threshold=args.threshold,
            target_sizes=torch.tensor([(im.height, im.width)]))[0]

        elements = []
        ranked = sorted(range(len(res["scores"])),
                        key=lambda i: -float(res["scores"][i]))
        for i, idx in enumerate(ranked):
            if i >= args.max_items:
                break
            x1, y1, x2, y2 = (round(float(v)) for v in res["boxes"][idx].tolist())
            el = S.element(i, "object", [x1, y1, x2, y2],
                           text=classes[int(res["labels"][idx])],
                           conf=round(float(res["scores"][idx]), 3),
                           coordsys="image_px")
            el["primitive"] = S.bbox_primitive([x1, y1, x2, y2])
            elements.append(el)

        report = S.envelope(task="detect", sensors=["owlv2"],
                            coordsys="image_px",
                            source={"type": "image", "path": args.image,
                                    "classes": classes})
        report["elements"] = elements
        report["notation"] = S.NOTATION_GUIDE
        report["metrics"] = {"detections": len(elements),
                             "classes": len(classes),
                             "threshold": args.threshold}
        print(S.dump_json(report))
        return 0
    except Exception as e:
        print(json.dumps({"error": "vs_detect failed", "detail": str(e)[:500]},
                         ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
