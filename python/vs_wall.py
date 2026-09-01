#!/usr/bin/env python3
"""vs_wall.py — 壁纸批量程序化分类（主色/亮度/饱和度）+ opt-in 语义标签。

程序化维度全部确定性：色相族（主色直方图峰值）、亮度档、饱和度档、宽高比。
（v3.0.0 起 semantic 功能废弃，仅程序化分类）

用法:
  vs_wall.py --dir PATH [--colors 5] [--max-files 200] [--ext png jpg jpeg webp]
             [--semantic] [--semantic-max 10]
"""
import argparse
import json
import vs_schema as S
import sys
from collections import Counter
from pathlib import Path
from typing import Any, cast

from PIL import Image, ImageStat  # type: ignore[import-not-found]

DEFAULT_EXTS = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]


def _downsample(im: Image.Image, max_side: int = 512) -> Image.Image:
    """缩小到最长边 max_side（统计加速）。失败时原样返回。"""
    try:
        w, h = im.size
        scale = max_side / max(w, h)
        if scale < 1:
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)  # type: ignore[attr-defined]
    except Exception:
        pass
    return im


def dominant_hue(im: Image.Image) -> int | None:
    """主色相：HSV 色相直方图峰值（只统计饱和度 ≥25 的像素）。纯灰图返回 None。"""
    hsv = _downsample(im).convert("HSV")
    data = hsv.tobytes()  # 每像素 3 字节 (H,S,V)
    cnt: Counter = Counter()
    for i in range(0, len(data), 3):
        s = data[i + 1]
        if s >= 25:
            cnt[data[i]] += 1
    if not cnt:
        return None
    # PIL 的 HSV 色相字节为 0-255 缩放（360° → 255），换算为角度
    return cast(int, round(cnt.most_common(1)[0][0] * 360 / 255))


def hue_family(hue: int | None, sat: int) -> str:
    if hue is None or sat < 25:
        return "灰/中性"
    if hue < 15 or hue >= 330:
        return "红"
    if hue < 45:
        return "橙"
    if hue < 65:
        return "黄"
    if hue < 160:
        return "绿"
    if hue < 200:
        return "青"
    if hue < 250:
        return "蓝"
    if hue < 285:
        return "紫"
    return "品红"


def warm_cool(hue: int | None, sat: int) -> str:
    if hue is None or sat < 25:
        return "中性"
    if hue < 70 or hue >= 330:
        return "暖色"
    return "冷色"


def brightness_tier(v: int) -> str:
    if v < 85:
        return "暗"
    if v < 170:
        return "中"
    return "亮"


def saturation_tier(s: int) -> str:
    if s < 25:
        return "低饱和"
    if s < 60:
        return "中饱和"
    return "高饱和"


def aspect_of(w: int, h: int) -> str:
    r = w / h if h else 1.0
    if r > 1.2:
        return "横版"
    if r < 0.83:
        return "竖版"
    return "方形"


def analyze_image(path: Path, colors: int) -> dict:
    """单张壁纸的程序化特征。永不抛异常：失败返回 {"error": ...}。"""
    try:
        im = Image.open(path).convert("RGB")
        w, h = im.size
        total = w * h
        q = im.quantize(colors=colors, method=Image.Quantize.MEDIANCUT).convert("RGB")  # type: ignore[attr-defined]
        dom = sorted(q.getcolors(maxcolors=total) or [], key=lambda e: -e[0])
        dominant_colors = [
            {
                "hex": f"#{cast(Any, e[1])[0]:02X}{cast(Any, e[1])[1]:02X}{cast(Any, e[1])[2]:02X}",
                "pct": round(e[0] / total * 100, 1),
            }
            for e in dom
        ]
        stat = ImageStat.Stat(_downsample(im).convert("HSV"))
        brightness = int(round(stat.mean[2]))
        saturation = int(round(stat.mean[1]))
        hue = dominant_hue(im)
        family = hue_family(hue, saturation)
        wc = warm_cool(hue, saturation)
        bt = brightness_tier(brightness)
        st = saturation_tier(saturation)
        return {
            "size_px": [w, h],
            "aspect": aspect_of(w, h),
            "metrics": {
                "brightness": brightness,
                "saturation": saturation,
                "dominant_colors": dominant_colors,
            },
            "programmatic": {
                "hue_deg": hue,
                "family": family,
                "warm_cool": wc,
                "tone": bt,
                "sat_tier": st,
                "category": f"{wc}·{bt}·{st}",
            },
        }
    except Exception as e:
        return {"error": str(e)[:200]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--colors", type=int, default=5)
    ap.add_argument("--max-files", type=int, default=200)
    ap.add_argument("--ext", nargs="*", default=[])
    ap.add_argument("--semantic", action="store_true", help="opt-in: 对每个文件附加 L2 语义标签")
    ap.add_argument("--semantic-max", type=int, default=10)
    args = ap.parse_args()

    try:
        exts = {e if e.startswith(".") else "." + e for e in args.ext} or set(DEFAULT_EXTS)
        files = sorted(
            p for p in Path(args.dir).iterdir()
            if p.is_file() and p.suffix.lower() in exts
        )[: args.max_files]

        items: list[dict[str, Any]] = []
        semantic_failures = 0
        for path in files:
            item = analyze_image(path, args.colors)
            item["file"] = path.name
            if args.semantic:
                # V3: vs_semantic 已移除（零其他模型约束）；semantic 功能废弃
                item["semantic"] = {"error": "semantic 已废弃（v3.0.0 移除 L2 语义层）"}
                semantic_failures += 1
            items.append(item)

        groups = {
            "by_category": dict(Counter(str(i.get("programmatic", {}).get("category", "?")) for i in items if "programmatic" in i)),
            "by_tone": dict(Counter(str(i.get("programmatic", {}).get("tone", "?")) for i in items if "programmatic" in i)),
            "by_family": dict(Counter(str(i.get("programmatic", {}).get("family", "?")) for i in items if "programmatic" in i)),
        }

        print(S.dump_json({
            "schema": "vision-report/v1",
            "source": {"type": "wallpaper_batch", "dir": str(args.dir),
                       "count": len(items), "truncated": len(items) >= args.max_files},
            "items": items,
            "groups": groups,
            "semantic": {"enabled": args.semantic, "failures": semantic_failures},
        }))
        return 0
    except Exception as e:
        print(json.dumps({"error": "vs_wall failed", "detail": str(e)[:500]}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
