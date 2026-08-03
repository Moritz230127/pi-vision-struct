#!/usr/bin/env python3
"""vs_schema.py — schema v2 统一元素模型 + 坐标系 + 色差度量。

所有传感器输出同构 element，融合算子与 DeepSeek 按同一模型消费：
  element = {id, type, bbox[x1,y1,x2,y2], text, conf, color{fill,text}, font, z,
             source:[传感器名], coordsys}

坐标系:
  css_px     CSS 像素（DOM，含 DPR 缩放前）
  device_px  物理设备像素（grim 截图）
  image_px   图像文件像素（经缩放后的图片）
  pt         印刷点（PPTX：1pt = 1/72 英寸）
"""
import math
import json
from typing import Any, Sequence

SCHEMA = "vision-report/v2"

# ---------------------------------------------------------------- 元素模型

def element(eid: int, etype: str, bbox: Sequence[float], *, text=None, conf=None,
            color=None, font=None, z=None, source=None, coordsys: str = "image_px") -> dict[str, Any]:
    bbox_int = list(bbox)
    return {
        "id": eid, "type": etype, "bbox": bbox_int,
        "text": text, "conf": conf, "color": color, "font": font, "z": z,
        "source": list(source or []), "coordsys": coordsys,
    }


def envelope(task: str | None = None, sensors: list[str] | None = None,
             coordsys: str | None = None, source: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema": SCHEMA, "task": task, "sensors": list(sensors or []),
        "coordsys": coordsys, "source": source or {},
        "elements": [], "anomalies": [], "metrics": {}, "truncated": False,
    }


# ---------------------------------------------------------------- 坐标变换

def css_to_device(bbox: Sequence[float], dpr: float, scroll: tuple[int, int] = (0, 0)) -> list[int]:
    """CSS 像素（含滚动偏移）→ 设备像素：device = (css - scroll) × dpr"""
    sx, sy = scroll
    return [round((bbox[0] - sx) * dpr), round((bbox[1] - sy) * dpr),
            round((bbox[2] - sx) * dpr), round((bbox[3] - sy) * dpr)]


def device_to_css(bbox: Sequence[float], dpr: float, scroll: tuple[int, int] = (0, 0)) -> list[int]:
    sx, sy = scroll
    return [round(bbox[0] / dpr + sx), round(bbox[1] / dpr + sy),
            round(bbox[2] / dpr + sx), round(bbox[3] / dpr + sy)]


def scale_bbox(bbox: Sequence[float], k: float) -> list[int]:
    """按缩放系数 k 变换 bbox（图像缩放/区域放大回原图）。"""
    return [round(v * k) for v in bbox]


def clip_bbox(bbox: Sequence[float], w: int, h: int) -> list[int]:
    x1, y1, x2, y2 = (max(0, round(v)) for v in bbox)
    return [x1, y1, min(x2, w), min(y2, h)]


def bbox_area(bbox: Sequence[float]) -> float:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def bbox_intersection(a: Sequence[float], b: Sequence[float]) -> list[int] | None:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return [round(x1), round(y1), round(x2), round(y2)] if x2 > x1 and y2 > y1 else None


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    inter = bbox_intersection(a, b)
    if inter is None:
        return 0.0
    ia = bbox_area(inter)
    ua = bbox_area(a) + bbox_area(b) - ia
    return ia / ua if ua > 0 else 0.0


# ---------------------------------------------------------------- 色差（CIE Lab ΔE76）

def srgb_to_linear(c: float) -> float:
    c = c / 255.0
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(rgb) -> float:
    r, g, b = (srgb_to_linear(ch) for ch in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(fg, bg) -> float:
    """WCAG 2.x 对比度（gamma 校正相对亮度）。AA 文本需 ≥4.5。"""
    l1, l2 = relative_luminance(fg), relative_luminance(bg)
    if l1 < l2:
        l1, l2 = l2, l1
    return (l1 + 0.05) / (l2 + 0.05)


def rgb_to_lab(rgb) -> tuple[float, float, float]:
    r, g, b = (srgb_to_linear(ch) for ch in rgb)
    # sRGB → XYZ (D65)
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    x, y, z = x / 0.95047, y / 1.0, z / 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 0.008856 else (7.787 * t + 16 / 116)

    fx, fy, fz = f(x), f(y), f(z)
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))


def delta_e76(c1, c2) -> float:
    """CIEDE2000 的经典简化版（ΔE76）：L*a*b* 空间欧氏距离。
    <1 不可感知；1-2 极微弱；2-10 可感知；10-50 明显；>50 巨大。"""
    l1, a1, b1 = rgb_to_lab(c1)
    l2, a2, b2 = rgb_to_lab(c2)
    return math.sqrt((l1 - l2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2)


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def rgb_to_hex(rgb) -> str:
    return "#{:02X}{:02X}{:02X}".format(*(round(v) for v in rgb))


def parse_hex(s: str) -> tuple[int, int, int]:
    try:
        return hex_to_rgb(s)
    except ValueError as e:
        raise ValueError(f"bad hex color: {s!r}") from e


def load_json(path: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        raise ValueError(f"cannot read {path}: {e}") from e


def dump_json(data, path: str | None = None) -> str:
    s = json.dumps(data, ensure_ascii=False)
    if path:
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(s)
        except OSError as e:
            raise ValueError(f"cannot write {path}: {e}") from e
    return s
