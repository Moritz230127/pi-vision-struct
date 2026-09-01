#!/usr/bin/env python3
"""vs_scene_stats.py — 数理化场景统计（替代原 vs_semantic 主观描述）。

全部输出为带单位的数值，无任何自然语言描述。
输入：任意图片
输出：颜色直方图（hex+count+percent）、面积比（前景/背景/区域）、
      对比度统计（mean/std/min/max）、文字密度（字符数/总像素）、
      空间布局量化（重心x/y、包围盒扩展度）

用法: vs_scene_stats.py --image PATH [--region x1,y1,x2,y2] [--colors 8]
"""
import argparse
import json
import math
import os
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageStat

import vs_schema as S


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02X}{g:02X}{b:02X}"


def luminance(rgb: tuple[int, int, int]) -> float:
    r, g, b = (c / 255.0 for c in rgb)
    r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
    g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
    b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def color_histogram(im: Image.Image, n_colors: int) -> list[dict]:
    """MEDIANCUT 量化 → 按面积降序排列的颜色表"""
    total = im.width * im.height
    q = im.quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT).convert("RGB")
    colors = q.getcolors(maxcolors=total) or []
    colors.sort(key=lambda e: -e[0])
    result = []
    for count, rgb in colors:
        result.append({
            "hex": rgb_to_hex(*rgb),
            "rgb": list(rgb),
            "count": count,
            "percent": round(count / total * 100, 2),
        })
    return result


def area_ratios(im: Image.Image) -> dict:
    """基于 Otsu 阈值的前景/背景面积比（替代 median 伪 Otsu）。"""
    import numpy as np
    from skimage.filters import threshold_otsu

    gray = np.array(im.convert("L"))
    total = gray.size
    threshold = float(threshold_otsu(gray))
    foreground = int(np.sum(gray < threshold))
    background = total - foreground
    return {
        "foreground_ratio": round(foreground / total, 4),
        "background_ratio": round(background / total, 4),
        "threshold": round(threshold, 1),
        "method": "otsu",
    }


def contrast_stats(im: Image.Image) -> dict:
    """亮度对比度统计"""
    gray = im.convert("L")
    arr = np.array(gray, dtype=np.float64)
    lum_min = float(arr.min())
    lum_max = float(arr.max())
    lum_mean = float(arr.mean())
    lum_std = float(arr.std())
    # WCAG contrast ratio of extremes
    def rel_lum(v):
        v = v / 255.0
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    l1, l2 = rel_lum(lum_max), rel_lum(lum_min)
    wcag_ratio = (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)
    return {
        "luminance_mean": round(lum_mean, 2),
        "luminance_std": round(lum_std, 2),
        "luminance_min": round(lum_min, 2),
        "luminance_max": round(lum_max, 2),
        "wcag_contrast_ratio": round(wcag_ratio, 2),
    }


def spatial_layout(im: Image.Image) -> dict:
    """空间布局量化：重心、包围盒扩展度"""
    gray = im.convert("L")
    arr = np.array(gray, dtype=np.float64)
    h, w = arr.shape
    # 重心（亮度加权）
    total_lum = arr.sum()
    if total_lum > 0:
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        center_x = float(np.sum(x_coords * arr) / total_lum)
        center_y = float(np.sum(y_coords * arr) / total_lum)
    else:
        center_x, center_y = w / 2, h / 2
    # 非黑区域的包围盒
    threshold = 10
    mask = arr > threshold
    if mask.any():
        ys, xs = np.where(mask)
        spread_x = float(xs.max() - xs.min())
        spread_y = float(ys.max() - ys.min())
        coverage = float(mask.sum()) / mask.size
    else:
        spread_x = spread_y = 0.0
        coverage = 0.0
    return {
        "center_of_mass_x": round(center_x / w, 4),  # 归一化 0-1
        "center_of_mass_y": round(center_y / h, 4),
        "spread_x_normalized": round(spread_x / w, 4),
        "spread_y_normalized": round(spread_y / h, 4),
        "coverage_ratio": round(coverage, 4),
        "aspect_ratio": round(w / h, 4) if h > 0 else 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--region", help="x1,y1,x2,y2")
    ap.add_argument("--colors", type=int, default=8, help="主色数量")
    args = ap.parse_args()

    try:
        im = Image.open(args.image).convert("RGB")
        w, h = im.size

        # 区域裁剪
        if args.region:
            x1, y1, x2, y2 = (int(v) for v in args.region.split(","))
            im = im.crop((x1, y1, x2, y2))
            w, h = im.size

        report = S.envelope(task="scene_stats", sensors=["pix", "PIL"],
                            coordsys="image_px",
                            source={"type": "image", "path": args.image,
                                    "size_px": [w, h]})

        # 数理化统计（全部数值）
        report["metrics"] = {
            "color_histogram": color_histogram(im, args.colors),
            "area": area_ratios(im),
            "contrast": contrast_stats(im),
            "spatial": spatial_layout(im),
            "pixel_count": w * h,
            "width": w, "height": h,
        }

        report["notation"] = S.NOTATION_GUIDE
        print(S.dump_json(report))
        return 0
    except Exception as e:
        print(json.dumps({"error": "vs_scene_stats failed", "detail": str(e)[:500]},
                         ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
