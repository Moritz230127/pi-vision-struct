#!/usr/bin/env python3
"""vs_geometry.py — V3 几何原语传感器（VTracer SVG 化 + 原语解析）。

把图像转 SVG（VTracer），再解析为形状原语（矩形/圆/椭圆/线段/多边形），
供纯文本 LLM 直接推理几何结构（VDLM 思路，无需训练）。

用法:
  vs_geometry.py --image PATH [--mode polygon|spline] [--max-shapes 50]
"""
import argparse
import json
import re
import sys
from typing import Any

import vs_schema as S


def svg_to_primitives(svg: str, max_shapes: int = 50) -> list[dict[str, Any]]:
    """SVG → 形状原语列表。"""
    primitives: list[dict[str, Any]] = []
    # 解析 <rect>
    for m in re.finditer(r'<rect[^>]*>', svg):
        attrs = _parse_attrs(m.group(0))
        if "x" in attrs and "y" in attrs and "width" in attrs and "height" in attrs:
            x, y = float(attrs["x"]), float(attrs["y"])
            w, h = float(attrs["width"]), float(attrs["height"])
            primitives.append({
                "type": "rect", "bbox": [x, y, x + w, y + h],
                "width": w, "height": h, "fill": attrs.get("fill"),
            })
    # 解析 <circle>
    for m in re.finditer(r'<circle[^>]*>', svg):
        attrs = _parse_attrs(m.group(0))
        if "cx" in attrs and "cy" in attrs and "r" in attrs:
            cx, cy, r = float(attrs["cx"]), float(attrs["cy"]), float(attrs["r"])
            primitives.append({
                "type": "circle", "bbox": [cx - r, cy - r, cx + r, cy + r],
                "center": [cx, cy], "radius": r, "fill": attrs.get("fill"),
            })
    # 解析 <ellipse>
    for m in re.finditer(r'<ellipse[^>]*>', svg):
        attrs = _parse_attrs(m.group(0))
        if "cx" in attrs and "cy" in attrs and "rx" in attrs and "ry" in attrs:
            cx, cy, rx, ry = (float(attrs[k]) for k in ("cx", "cy", "rx", "ry"))
            primitives.append({
                "type": "ellipse", "bbox": [cx - rx, cy - ry, cx + rx, cy + ry],
                "center": [cx, cy], "rx": rx, "ry": ry, "fill": attrs.get("fill"),
            })
    # 解析 <line>
    for m in re.finditer(r'<line[^>]*>', svg):
        attrs = _parse_attrs(m.group(0))
        if all(k in attrs for k in ("x1", "y1", "x2", "y2")):
            x1, y1, x2, y2 = (float(attrs[k]) for k in ("x1", "y1", "x2", "y2"))
            primitives.append({
                "type": "line", "bbox": [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)],
                "p1": [x1, y1], "p2": [x2, y2],
            })
    # 解析 <path>（VTracer 主输出：M/L 多边形 + transform translate）
    for m in re.finditer(r'<path[^>]*>', svg):
        tag = m.group(0)
        dm = re.search(r'd="([^"]*)"', tag)
        if not dm:
            continue
        d = dm.group(1)
        # transform translate 偏移
        tx = ty = 0.0
        tm = re.search(r'translate\(([-\d.e]+)[, ]+([-\d.e]+)\)', tag)
        if tm:
            tx, ty = float(tm.group(1)), float(tm.group(2))
        # 解析 M/L 命令（局部坐标）
        pts = []
        for pm in re.finditer(r'([ML])\s*([-\d.e]+)[, ]+([-\d.e]+)', d):
            pts.append((float(pm.group(2)) + tx, float(pm.group(3)) + ty))
        if len(pts) >= 3:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            primitives.append({
                "type": "polygon", "bbox": [min(xs), min(ys), max(xs), max(ys)],
                "points": len(pts), "fill": _parse_attrs(tag).get("fill"),
            })
        elif len(pts) == 2:
            primitives.append({
                "type": "line", "bbox": [pts[0][0], pts[0][1], pts[1][0], pts[1][1]],
                "p1": list(pts[0]), "p2": list(pts[1]),
            })
    return primitives[:max_shapes]


def _parse_attrs(tag: str) -> dict[str, str]:
    """解析 SVG 标签属性。"""
    attrs: dict[str, str] = {}
    for m in re.finditer(r'(\w[\w-]*)\s*=\s*"([^"]*)"', tag):
        attrs[m.group(1)] = m.group(2)
    return attrs


def _path_bbox(d: str) -> list[float]:
    """path 数据 → 粗略 bbox（解析 M/L/H/V 命令）。"""
    xs, ys = [], []
    for m in re.finditer(r'([MmLlHhVv])\s*([-\d.e]+)(?:[,\s]+([-\d.e]+))?', d):
        cmd, a, b = m.group(1), m.group(2), m.group(3)
        if cmd in "Mm" and b:
            xs.append(float(a)); ys.append(float(b))
        elif cmd in "Hh":
            xs.append(float(a))
        elif cmd in "Vv":
            ys.append(float(a))
    if not xs or not ys:
        return [0, 0, 0, 0]
    return [min(xs), min(ys), max(xs), max(ys)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--mode", default="polygon", choices=["polygon", "spline"])
    ap.add_argument("--max-shapes", type=int, default=50)
    args = ap.parse_args()

    try:
        import vtracer  # type: ignore[import-not-found]
        from PIL import Image
        im = Image.open(args.image).convert("RGB")
        w, h = im.size

        # VTracer 光栅 → SVG
        svg = _vtracer_bytes(im, args.mode)

        primitives = svg_to_primitives(svg, args.max_shapes)

        elements = []
        for i, p in enumerate(primitives):
            elements.append(S.element(i, p["type"], p["bbox"],
                                      conf=1.0, source=["geometry"], coordsys="image_px"))
            elements[-1]["shape"] = p

        report = S.envelope(task="geometry", sensors=["geometry"], coordsys="image_px",
                            source={"type": "image", "path": args.image,
                                    "size_px": [w, h], "mode": args.mode})
        report["schema"] = "vision-report/v3"
        report["elements"] = elements
        report["metrics"] = {
            "shapes": len(primitives),
            "types": {t: sum(1 for p in primitives if p["type"] == t) for t in
                      set(p["type"] for p in primitives)},
        }
        print(S.dump_json(report))
        return 0
    except Exception as e:
        print(json.dumps({"error": "vs_geometry failed", "detail": str(e)[:500]},
                         ensure_ascii=False))
        return 1


def _vtracer_bytes(im, mode: str) -> str:
    """PIL 图像 → VTracer SVG（bytes 路径）。"""
    import io
    import vtracer  # type: ignore[import-not-found]
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return vtracer.convert_raw_image_to_svg(buf.getvalue(), img_format="png",
                                            colormode="color", mode=mode)


if __name__ == "__main__":
    sys.exit(main())
