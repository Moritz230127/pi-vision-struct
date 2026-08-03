#!/usr/bin/env python3
"""vs_audit.py — 融合算子：元素级几何/样式审计（schema v2）。

对任意元素列表（DOM 元素 / PPTX 形状）计算确定性缺陷：

  1. element_overlap : 两两 bbox 重叠（面积与 IoU）
  2. off_canvas      : 元素超出画布（PPT 出界 / DOM 视口外）
  3. contrast        : 文本色 vs 填充色的 WCAG 对比度不达标（AA <4.5）

用法:
  vs_audit.py --report report.json [--canvas WxH] [--overlap-threshold 0.05]
  元素来源: report.elements[]（需 bbox,color{text,fill}）或 report.slides[].shapes[]（pptx）
"""
import argparse
import json
import sys
from typing import Any

import vs_schema as S


def extract_elements(report: dict[str, Any]) -> list[dict[str, Any]]:
    els = report.get("elements") or []
    if els:
        return els
    out = []
    for slide in report.get("slides", []):
        for sh in slide.get("shapes", []):
            if sh.get("pos_pt") is not None and sh.get("size_pt") is not None:
                x, y = sh["pos_pt"][0] or 0, sh["pos_pt"][1] or 0
                w, h = sh["size_pt"][0] or 0, sh["size_pt"][1] or 0
                texts = []
                fill = sh.get("fill")
                for t in sh.get("texts", []):
                    for run in t.get("runs", []):
                        texts.append({"text": run.get("text", ""), "color": run.get("color"),
                                      "size_pt": run.get("size_pt")})
                out.append({"bbox": [x, y, x + w, y + h], "fill": fill,
                            "texts": texts, "name": sh.get("name")})
    return out


def first_text_color(el: dict[str, Any]):
    texts = el.get("texts")
    if texts:
        for t in texts:
            if t.get("color"):
                return t["color"], t.get("size_pt")
    color = (el.get("color") or {})
    return (color.get("text"), None)


def label(el: dict[str, Any], fallback: str) -> str:
    return str(el.get("name") or (el.get("text") or el.get("type") or "") or fallback)[:60]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--canvas")
    ap.add_argument("--overlap-threshold", type=float, default=0.05)
    ap.add_argument("--min-overlap-area", type=float, default=400.0)
    args = ap.parse_args()

    try:
        report = S.load_json(args.report)
        els = extract_elements(report)
        anomalies: list[dict[str, Any]] = []
        canvas = None
        if args.canvas:
            w, h = (int(v) for v in args.canvas.split("x"))
            canvas = (w, h)
        elif report.get("source", {}).get("slide_size_pt"):
            # pptx 报告自带画布尺寸（pt）
            sw, sh = report["source"]["slide_size_pt"]
            canvas = (int(sw), int(sh))

        # 1) 两两重叠
        for i in range(len(els)):
            for j in range(i + 1, len(els)):
                a, b = els[i], els[j]
                box_a, box_b = a["bbox"], b["bbox"]
                inter = S.bbox_intersection(box_a, box_b)
                if inter is None:
                    continue
                ia = S.bbox_area(inter)
                if ia < args.min_overlap_area:
                    continue
                iou = S.bbox_iou(box_a, box_b)
                if iou > args.overlap_threshold:
                    anomalies.append({
                        "type": "element_overlap", "bbox": inter,
                        "confidence": round(min(0.99, iou + 0.5), 2),
                        "evidence": {"iou": round(iou, 2), "area": round(ia),
                                     "a": label(a, f"el{i}"), "b": label(b, f"el{j}")},
                    })

        # 2) 出界
        if canvas:
            cw, ch = canvas
            for el in els:
                x1, y1, x2, y2 = el["bbox"]
                if x1 < -1 or y1 < -1 or x2 > cw + 1 or y2 > ch + 1:
                    anomalies.append({
                        "type": "off_canvas", "bbox": el["bbox"],
                        "confidence": 0.95,
                        "evidence": {"canvas": [cw, ch], "name": label(el, "?")},
                    })

        # 3) 对比度
        for el in els:
            fg, _ = first_text_color(el)
            fill = el.get("fill")
            if not fg or not fill or fill in ("BACKGROUND (5)", "SOLID (1)"):
                continue
            try:
                ratio = S.contrast_ratio(S.hex_to_rgb(fg), S.hex_to_rgb(fill))
            except ValueError:
                continue
            if ratio < 4.5:
                anomalies.append({
                    "type": "contrast_fail", "bbox": el["bbox"],
                    "confidence": 1.0,
                    "evidence": {"fg": fg, "bg": fill, "ratio": round(ratio, 2), "required_aa": 4.5,
                                 "name": label(el, "?")},
                })

        report_out = S.envelope(task="audit",
                                sensors=report.get("sensors") or ["audit"],
                                coordsys=report.get("coordsys"),
                                source={"type": "fused", "from": args.report})
        report_out["anomalies"] = anomalies
        report_out["metrics"] = {"element_count": len(els), "anomaly_count": len(anomalies)}
        print(S.dump_json(report_out))
        return 0
    except Exception as e:
        print(json.dumps({"error": "vs_audit failed", "detail": str(e)[:500]}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
