#!/usr/bin/env python3
"""vs_pix.py — 确定性像素测量（vision-report/v1 片段）

直方图 / 区域取色 / 亮度饱和度 / 像素级 diff / WCAG 2.x 对比度（gamma 校正）。
纯 PIL 实现，无模型、无 numpy。

用法:
  vs_pix.py --image PATH [--regions x1,y1,x2,y2 ...] [--compare PATH] [--colors N]
            [--wcag fghex,bghex ...] [--diff-threshold N]
"""
import argparse
import json
import sys
from typing import Any, cast

from PIL import Image, ImageChops, ImageStat


def srgb_to_linear(c: float) -> float:
    c = c / 255.0
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(rgb) -> float:
    r, g, b = (srgb_to_linear(ch) for ch in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(fg, bg) -> float:
    l1, l2 = relative_luminance(fg), relative_luminance(bg)
    if l1 < l2:
        l1, l2 = l2, l1
    return (l1 + 0.05) / (l2 + 0.05)


def hex_of(rgb) -> str:
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def parse_region(s: str):
    try:
        x1, y1, x2, y2 = (int(v) for v in s.split(","))
    except ValueError as e:
        raise ValueError(f"bad region: {s!r}") from e
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def connected_components(mask_img, min_pixels: int = 25, max_comps: int = 20):
    """diff 掩码的 8-连通域分析 → 独立异常区域 bbox（去噪，上限保护）。"""
    from collections import deque

    w, h = mask_img.size
    data = mask_img.tobytes()
    visited = bytearray(w * h)
    comps = []
    for y in range(h):
        for x in range(w):
            idx = y * w + x
            if visited[idx] or data[idx] == 0:
                continue
            q = deque([(x, y)])
            visited[idx] = 1
            xs, ys, n = [], [], 0
            while q:
                cx, cy = q.popleft()
                xs.append(cx)
                ys.append(cy)
                n += 1
                x0, x1 = max(0, cx - 1), min(w - 1, cx + 1)
                y0, y1 = max(0, cy - 1), min(h - 1, cy + 1)
                for ny in range(y0, y1 + 1):
                    for nx in range(x0, x1 + 1):
                        ni = ny * w + nx
                        if not visited[ni] and data[ni] > 0:
                            visited[ni] = 1
                            q.append((nx, ny))
            if n >= min_pixels:
                comps.append({"bbox": [min(xs), min(ys), max(xs), max(ys)], "count": n})
                if len(comps) >= max_comps:
                    break
    return comps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--regions", nargs="*", default=[])
    ap.add_argument("--compare")
    ap.add_argument("--colors", type=int, default=8)
    ap.add_argument("--wcag", nargs="*", default=[])
    ap.add_argument("--diff-threshold", type=int, default=30)
    args = ap.parse_args()

    try:
        im = Image.open(args.image).convert("RGB")
        w, h = im.size
        total = w * h
        out = {"schema": "vision-report/v2", "task": "pix", "sensors": ["pix"],
               "coordsys": "image_px",
               "source": {"type": "image", "path": args.image, "size_px": [w, h]},
               "elements": [], "anomalies": [], "metrics": {}}

        # 1) 主色直方图
        q = im.quantize(colors=args.colors, method=Image.Quantize.MEDIANCUT).convert("RGB")  # type: ignore[attr-defined]
        dom = sorted(q.getcolors(maxcolors=total) or [], key=lambda e: -e[0])
        metrics: dict[str, object] = {
            "dominant_colors": [
                {
                    "hex": hex_of(cast(Any, entry[1])),
                    "rgb": [v for v in cast(Any, entry[1])],
                    "pct": round(entry[0] / total * 100, 1),
                }
                for entry in dom
            ]
        }
        stat_hsv = ImageStat.Stat(im.convert("HSV"))
        metrics["brightness"] = round(stat_hsv.mean[2])
        metrics["saturation"] = round(stat_hsv.mean[1])
        out["metrics"] = metrics

        # 2) 区域取色（中心像素）→ 同时输出为 v2 元素
        regions = []
        region_elements = []
        for i, s in enumerate(args.regions):
            x1, y1, x2, y2 = parse_region(s)
            sample = cast(Any, im.getpixel(((x1 + x2) // 2, (y1 + y2) // 2)))
            regions.append({"bbox": [x1, y1, x2, y2], "hex": hex_of(sample), "rgb": [v for v in sample]})
            region_elements.append({"id": i, "type": "region", "bbox": [x1, y1, x2, y2],
                                    "text": None, "conf": 1.0,
                                    "color": {"fill": hex_of(sample)}, "font": None, "z": None,
                                    "source": ["pix"], "coordsys": "image_px"})
        if regions:
            out["regions"] = regions
            out["elements"] = region_elements

        # 3) WCAG 对比度
        wcag = []
        for pair in args.wcag:
            fg_s, bg_s = pair.split(",")
            fg = tuple(int(fg_s[i : i + 2], 16) for i in (0, 2, 4))
            bg = tuple(int(bg_s[i : i + 2], 16) for i in (0, 2, 4))
            ratio = contrast_ratio(fg, bg)
            wcag.append({"fg": hex_of(fg), "bg": hex_of(bg), "ratio": round(ratio, 2),
                         "passes_aa": ratio >= 4.5, "passes_aaa": ratio >= 7.0})
        if wcag:
            out["wcag"] = wcag

        # 4) 像素级 diff（ImageChops 差异 + 阈值 + 连通域定位）
        if args.compare:
            im2 = Image.open(args.compare).convert("RGB")
            if im2.size != (w, h):
                im2 = im2.resize((w, h), Image.LANCZOS)  # type: ignore[attr-defined]
            diff_img = ImageChops.difference(im, im2).convert("L")
            mask = diff_img.point(lambda v: 255 if v > args.diff_threshold else 0)
            comps = connected_components(mask)
            out["anomalies"] = [
                {"type": "pixel_diff", "bbox": c["bbox"], "count": c["count"],
                 "threshold": args.diff_threshold}
                for c in comps
            ]

        print(json.dumps(out, ensure_ascii=False))
        return 0
    except Exception as e:  # 永不崩溃：任何错误都以 JSON 形式输出
        print(json.dumps({"error": "vs_pix failed", "detail": str(e)[:500]}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
