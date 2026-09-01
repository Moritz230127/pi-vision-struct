#!/usr/bin/env python3
"""vs_ascii.py — V3 ASCII 栅格传感器（纯算法，零模型）。

把图像转成多分辨率 ASCII 文本栅格，供纯文本 LLM（DeepSeek）"粗看"：
  - 粗栅格 64×36：全图结构概览
  - 细栅格 128×72：候选区域细节（配合 zoom 协议）

字符映射（亮度 → 字符密度）+ 可选颜色前缀（ANSI 或 hex 标注）。

用法:
  vs_ascii.py --image PATH [--cols 64] [--rows 36] [--color]
"""
import argparse
import json
import sys

import numpy as np

import vs_schema as S

# 亮度 → 字符（从密到疏）
RAMP = "@%#*+=-:. "
# 颜色字符（用于颜色标注）
COLOR_CHARS = "RGBYMCW"


def luminance(arr: np.ndarray) -> np.ndarray:
    """RGB → 亮度（0-255）。"""
    r, g, b = arr[..., 0].astype(np.float64), arr[..., 1].astype(np.float64), arr[..., 2].astype(np.float64)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def dominant_color(arr: np.ndarray) -> str:
    """区域主色（量化到 8 色）。"""
    r, g, b = arr[..., 0].mean(), arr[..., 1].mean(), arr[..., 2].mean()
    # 简单量化
    if r > 200 and g < 100 and b < 100:
        return "R"
    if g > 200 and r < 100 and b < 100:
        return "G"
    if b > 200 and r < 100 and g < 100:
        return "B"
    if r > 150 and g > 150 and b < 100:
        return "Y"
    if r > 150 and g < 100 and b > 150:
        return "M"
    if r < 100 and g > 150 and b > 150:
        return "C"
    if r > 200 and g > 200 and b > 200:
        return "W"
    return "K"


def to_ascii_grid(arr: np.ndarray, cols: int, rows: int, use_color: bool = False) -> list[str]:
    """图像 → ASCII 栅格行列表。"""
    h, w = arr.shape[:2]
    lum = luminance(arr)
    # 分块采样
    lines = []
    for ry in range(rows):
        y0 = int(ry * h / rows)
        y1 = int((ry + 1) * h / rows)
        line = []
        for rx in range(cols):
            x0 = int(rx * w / cols)
            x1 = int((rx + 1) * w / cols)
            block = arr[y0:y1, x0:x1]
            l = lum[y0:y1, x0:x1].mean()
            idx = int((255 - l) / 255 * (len(RAMP) - 1))
            idx = max(0, min(len(RAMP) - 1, idx))
            ch = RAMP[idx]
            if use_color:
                c = dominant_color(block)
                if c != "K":
                    ch = c
            line.append(ch)
        lines.append("".join(line))
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--cols", type=int, default=64)
    ap.add_argument("--rows", type=int, default=36)
    ap.add_argument("--color", action="store_true", help="用颜色字符标注")
    ap.add_argument("--region", help="x1,y1,x2,y2 裁剪区域")
    args = ap.parse_args()

    try:
        from PIL import Image
        im = Image.open(args.image).convert("RGB")
        w, h = im.size
        if args.region:
            x1, y1, x2, y2 = (int(v) for v in args.region.split(","))
            im = im.crop((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
        arr = np.array(im)

        grid = to_ascii_grid(arr, args.cols, args.rows, args.color)

        report = S.envelope(task="ascii", sensors=["ascii"], coordsys="image_px",
                            source={"type": "image", "path": args.image,
                                    "size_px": [w, h], "cols": args.cols, "rows": args.rows,
                                    "color": args.color})
        report["schema"] = "vision-report/v3"
        report["ascii"] = {"cols": args.cols, "rows": args.rows,
                           "grid": grid, "color": args.color}
        report["metrics"] = {"chars": args.cols * args.rows,
                             "ramp": RAMP}
        print(S.dump_json(report))
        return 0
    except Exception as e:
        print(json.dumps({"error": "vs_ascii failed", "detail": str(e)[:500]},
                         ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
