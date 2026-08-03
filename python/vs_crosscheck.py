#!/usr/bin/env python3
"""vs_crosscheck.py — 融合算子：多传感器交叉验证（schema v2）。

对同一画面的多个独立传感器输出做一致性检查，不一致自动检出为 anomaly（带证据）：

  1. color_drift   : DOM 声明颜色 vs 像素实测色（CIELAB ΔE76）→ 渲染异常/样式未生效
  2. text_missing  : DOM 有文本但 OCR 在对应区域未读到 → canvas/GPU 渲染问题或元素未绘制
  3. text_extra    : OCR 读到文本但 DOM 无 → canvas/图片内文字、或 DOM 取自不同状态
  4. overlap       : DOM 元素两两重叠超阈值

用法:
  vs_crosscheck.py --image IMG [--dom dom.json] [--ocr ocr.json]
                   [--dpr 1.0] [--color-threshold 5.0] [--overlap-threshold 0.05]
"""
import argparse
import json
import sys
from typing import Any, cast

import vs_schema as S


def load(path: str | None) -> dict[str, Any] | None:
    return S.load_json(path) if path else None


def sample_color_dense(image_path: str, bbox: list[int], max_samples: int = 300) -> list[tuple[tuple[int, int, int], tuple[int, int]]]:
    """bbox 内密集步进采样，返回 [(rgb, 采样点), ...]（上限 max_samples，跳过越界）。"""
    from PIL import Image  # type: ignore[import-not-found]

    im = Image.open(image_path).convert("RGB")
    w, h = im.size
    x1, y1 = max(0, bbox[0]), max(0, bbox[1])
    x2, y2 = min(bbox[2], w - 1), min(bbox[3], h - 1)
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    step = max(1, round((bw * bh / max_samples) ** 0.5))
    pts: list[tuple[tuple[int, int, int], tuple[int, int]]] = []
    for yy in range(y1, y2 + 1, step):
        for xx in range(x1, x2 + 1, step):
            rgb: tuple[int, int, int] = cast(Any, im.getpixel((xx, yy)))
            pts.append((rgb, (xx, yy)))
            if len(pts) >= max_samples:
                break
        if len(pts) >= max_samples:
            break
    return pts


def dom_bbox_device(el: dict[str, Any], dpr: float) -> list[int]:
    if "bbox_device_px" in el:
        return [round(v) for v in el["bbox_device_px"]]
    return [round(v * dpr) for v in el["bbox"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--dom")
    ap.add_argument("--ocr")
    ap.add_argument("--dpr", type=float, default=1.0)
    ap.add_argument("--color-threshold", type=float, default=5.0)
    ap.add_argument("--overlap-threshold", type=float, default=0.05)
    args = ap.parse_args()

    try:
        dom = load(args.dom)
        ocr = load(args.ocr)
        anomalies: list[dict[str, Any]] = []
        dpr = args.dpr if args.dom and args.dom else 1.0

        # ---------- 1) 颜色漂移：DOM 声明色 vs 像素实测（网格采样取最小 ΔE） ----------
        if dom and args.image:
            for el in dom.get("elements", []):
                col = (el.get("color") or {})
                declared = col.get("text") or col.get("fill")
                if not declared:
                    continue
                bbox = dom_bbox_device(el, dpr)
                samples = sample_color_dense(args.image, bbox)
                if not samples:
                    continue
                best_de, best_pt, best_rgb = None, None, None
                for rgb, pt in samples:
                    de = S.delta_e76(S.hex_to_rgb(declared), rgb)
                    if best_de is None or de < best_de:
                        best_de, best_pt, best_rgb = de, pt, rgb
                if best_de is not None and best_pt is not None and best_rgb is not None \
                        and best_de > args.color_threshold:
                    anomalies.append({
                        "type": "color_drift",
                        "bbox": bbox,
                        "confidence": round(min(0.99, best_de / 100 + 0.5), 2),
                        "evidence": {"dom_color": declared, "pixel_color": S.rgb_to_hex(best_rgb),
                                     "delta_e76": round(best_de, 1),
                                     "sample_point": list(best_pt), "samples": len(samples)},
                        "suggested_cause": "computed style 未生效或被覆盖，或截图与 DOM 状态不一致",
                    })

        # ---------- 2) 文本交叉验证：DOM vs OCR ----------
        if dom and ocr:
            dom_texts = [el for el in dom.get("elements", []) if el.get("text")]
            ocr_texts = [el for el in ocr.get("elements", []) if el.get("text")]
            for el in dom_texts:
                bbox = dom_bbox_device(el, dpr)
                area = S.bbox_area(bbox)
                if area < 20:
                    continue
                best = 0.0
                for o in ocr_texts:
                    ob = [round(v) for v in o["bbox"]]
                    best = max(best, S.bbox_iou(bbox, ob))
                if best == 0.0:
                    anomalies.append({
                        "type": "text_missing_in_ocr",
                        "bbox": bbox,
                        "confidence": 0.6,
                        "evidence": {"dom_text": el["text"][:100], "max_iou_with_ocr": best},
                        "suggested_cause": "元素未绘制/被遮挡/为 canvas 渲染，或 OCR 漏读",
                    })
            # OCR 文本不在 DOM（采样前 20 个避免噪音）
            for o in ocr_texts[:20]:
                ob = [round(v) for v in o["bbox"]]
                best = 0.0
                for el in dom_texts:
                    best = max(best, S.bbox_iou(ob, dom_bbox_device(el, dpr)))
                if best == 0.0 and len(o.get("text", "")) > 1:
                    anomalies.append({
                        "type": "text_not_in_dom",
                        "bbox": ob,
                        "confidence": 0.5,
                        "evidence": {"ocr_text": o["text"][:100], "max_iou_with_dom": best},
                        "suggested_cause": "canvas/图片内文字，或 DOM 取自不同页面状态",
                    })

        # ---------- 3) DOM 元素重叠 ----------
        if dom:
            els = dom.get("elements", [])
            for i in range(len(els)):
                for j in range(i + 1, len(els)):
                    a, b = els[i], els[j]
                    if not (a.get("text") or b.get("text")):
                        continue
                    box_a = dom_bbox_device(a, dpr)
                    box_b = dom_bbox_device(b, dpr)
                    inter = S.bbox_intersection(box_a, box_b)
                    if inter is None:
                        continue
                    iou = S.bbox_iou(box_a, box_b)
                    if iou > args.overlap_threshold:
                        anomalies.append({
                            "type": "element_overlap",
                            "bbox": inter,
                            "confidence": round(min(0.99, iou + 0.5), 2),
                            "evidence": {"iou": round(iou, 2),
                                         "a": (a.get("text") or a.get("type"))[:60],
                                         "b": (b.get("text") or b.get("type"))[:60]},
                        })

        report = S.envelope(task="crosscheck", sensors=[s for s, p in (("dom", dom), ("ocr", ocr)) if p],
                            coordsys="device_px",
                            source={"type": "fused", "image": args.image, "dpr": dpr})
        report["anomalies"] = anomalies
        report["metrics"] = {"anomaly_count": len(anomalies)}
        print(S.dump_json(report))
        return 0
    except Exception as e:
        print(json.dumps({"error": "vs_crosscheck failed", "detail": str(e)[:500]}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
