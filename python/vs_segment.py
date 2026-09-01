#!/usr/bin/env python3
"""vs_segment.py — V3 前景分割传感器（MobileSAM，GPU）。

MobileSAM（5M 参数）→ 前景/背景 mask → 前景 bbox + 面积比 + 实例数。
破解"复杂背景无法识别有效内容"：先分离前景，再对前景做测量/OCR。

依赖: vsensor env（GPU torch + MobileSAM 权重 ~/.cache/vsensor/mobile_sam.pt）
用法:
  vs_segment.py --image PATH [--min-area-ratio 0.01] [--max-instances 10]
"""
import argparse
import json
import sys
from pathlib import Path

import vs_schema as S

WEIGHTS = f"{Path.home()}/.cache/vsensor/mobile_sam.pt"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--saliency", help="saliency 报告 JSON（候选区 bbox 作为 box 提示）")
    ap.add_argument("--min-area-ratio", type=float, default=0.01)
    ap.add_argument("--max-instances", type=int, default=10)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    try:
        import torch
        import numpy as np
        from PIL import Image
        from mobile_sam import sam_model_registry, SamPredictor

        device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
        model = sam_model_registry["vit_t"](checkpoint=WEIGHTS)
        model.to(device)
        model.eval()
        predictor = SamPredictor(model)

        im = Image.open(args.image).convert("RGB")
        w, h = im.size
        arr = np.array(im)
        predictor.set_image(arr)

        # 提示来源：saliency 候选区 bbox（优先）或网格点
        prompts = []
        if args.saliency:
            sal = S.load_json(args.saliency)
            for c in sal.get("candidates", [])[: args.max_instances]:
                prompts.append({"box": c["bbox"]})
        if not prompts:
            grid = 3
            for gy in range(grid):
                for gx in range(grid):
                    px = int((gx + 0.5) * w / grid)
                    py = int((gy + 0.5) * h / grid)
                    prompts.append({"point": [px, py]})

        instances = []
        for prompt in prompts:
            if "box" in prompt:
                box = np.array([prompt["box"]], dtype=np.float32)
                masks, scores, _ = predictor.predict(
                    box=box, multimask_output=False)
            else:
                input_point = np.array([prompt["point"]])
                input_label = np.array([1])
                masks, scores, _ = predictor.predict(
                    point_coords=input_point, point_labels=input_label,
                    multimask_output=False)
            mask = masks[0]  # (H, W) bool
            ys, xs = np.nonzero(mask)
            if len(ys) < 50:
                continue
            area_ratio = len(ys) / (w * h)
            if area_ratio < args.min_area_ratio:
                continue
            instances.append({
                "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                "area_ratio": round(area_ratio, 4),
                "pixels": int(len(ys)),
                "score": round(float(scores[0]), 3),
                "prompt": prompt,
            })

        # 合并重叠实例（IoU > 0.5 取面积大的）
        instances.sort(key=lambda i: -i["pixels"])
        merged = []
        for inst in instances:
            dup = False
            for m in merged:
                if S.bbox_iou(inst["bbox"], m["bbox"]) > 0.5:
                    dup = True
                    break
            if not dup:
                merged.append(inst)
            if len(merged) >= args.max_instances:
                break

        # 前景 = 最大实例
        foreground = merged[0] if merged else None

        elements = []
        for i, inst in enumerate(merged):
            elements.append(S.element(i, "foreground", inst["bbox"],
                                      conf=min(1.0, inst["area_ratio"] * 2.0),
                                      source=["segment"], coordsys="image_px"))
            elements[-1]["segment"] = inst

        report = S.envelope(task="segment", sensors=["segment"], coordsys="image_px",
                            source={"type": "image", "path": args.image,
                                    "size_px": [w, h], "device": device})
        report["schema"] = "vision-report/v3"
        report["elements"] = elements
        report["foreground"] = foreground
        report["metrics"] = {
            "instances": len(merged),
            "foreground_ratio": foreground["area_ratio"] if foreground else 0.0,
            "device": device,
        }
        print(S.dump_json(report))
        return 0
    except Exception as e:
        print(json.dumps({"error": "vs_segment failed", "detail": str(e)[:500]},
                         ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
