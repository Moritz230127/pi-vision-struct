#!/usr/bin/env python3
"""bench/gen_fixtures.py — 生成确定性真值夹具（基准套件用）。

产物（tests/bench_fixtures/）:
  colors.png          12 个已知 hex 色块（位置固定）
  text_lines.png      已知文本行（CJK+ASCII+数字，bbox 已知）
  layout_page.png     合成 UI 页面（已知元素矩形）
  diff_base.png / diff_injected.png  注入 N 个已知异常矩形
  drift_scene.png     DOM 声明色 vs 渲染色不一致的场景（crosscheck 基准）
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont  # type: ignore[import-not-found]

OUT = Path(__file__).resolve().parent.parent / "tests" / "bench_fixtures"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = ["#E74C3C", "#F39C12", "#2ECC71", "#3498DB", "#9B59B6", "#1ABC9C",
          "#E91E63", "#FF9800", "#4CAF50", "#2196F3", "#795548", "#607D8B"]

TEXT_LINES = [
    "中文测试行 Hello World 12345",
    "QWEN3VL8B-GT42-K7Z9",
    "页面加载失败 ERR_CONNECTION_REFUSED",
    "缓存命中率 97.99% pixelRatio=2.0",
    "The quick brown fox 42",
    "图纸编号 GT-DWG-042 比例 1:100",
    "RapidOCR PP-OCRv6 坐标框",
    "WCAG 对比度 4.48 不达标",
    "壁纸分类 蓝 27 橙 6 红 5",
    "0123456789 ABCDEFGHIJKLMNOPQRSTUVWXYZ",
]


def font(size: int):
    for p in ["/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"]:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


def gen_colors() -> None:
    im = Image.new("RGB", (640, 480), (255, 255, 255))
    d = ImageDraw.Draw(im)
    rects = []
    for i, hexc in enumerate(COLORS):
        x = (i % 4) * 160
        y = (i // 4) * 160
        d.rectangle([x + 20, y + 20, x + 140, y + 140], fill=tuple(int(hexc[j:j+2], 16) for j in (1, 3, 5)))
        rects.append({"hex": hexc, "bbox": [x + 20, y + 20, x + 140, y + 140]})
    im.save(OUT / "colors.png")
    (OUT / "colors_gt.json").write_text(json.dumps(rects), encoding="utf-8")


def gen_text_lines() -> None:
    f = font(22)
    im = Image.new("RGB", (1000, 320), (255, 255, 255))
    d = ImageDraw.Draw(im)
    items = []
    for i, line in enumerate(TEXT_LINES):
        y = 10 + i * 30
        d.text((10, y), line, fill=(20, 20, 20), font=f)
        bbox = d.textbbox((10, y), line, font=f)
        items.append({"text": line, "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]]})
    im.save(OUT / "text_lines.png")
    (OUT / "text_gt.json").write_text(json.dumps(items), encoding="utf-8")


def gen_layout_page() -> None:
    f_big, f_mid = font(26), font(16)
    im = Image.new("RGB", (800, 500), (245, 246, 247))
    d = ImageDraw.Draw(im)
    els = []
    # 标题
    d.rectangle([20, 20, 780, 70], fill=(255, 255, 255))
    d.text((30, 28), "仪表盘 Dashboard", fill=(34, 120, 210), font=f_big)
    els.append({"type": "title", "bbox": [20, 20, 780, 70], "text": "仪表盘 Dashboard"})
    # 左面板
    d.rectangle([20, 90, 250, 460], fill=(34, 120, 210))
    d.text((30, 100), "导航", fill=(255, 255, 255), font=f_mid)
    d.text((30, 130), "首页", fill=(230, 240, 255), font=f_mid)
    d.text((30, 160), "设置", fill=(230, 240, 255), font=f_mid)
    els.append({"type": "panel", "bbox": [20, 90, 250, 460], "fill": "#2278D2"})
    # 主卡片
    d.rectangle([270, 90, 780, 260], fill=(255, 255, 255))
    d.text((290, 100), "统计卡片", fill=(40, 40, 40), font=f_mid)
    d.rectangle([290, 140, 420, 240], fill=(240, 240, 240))
    d.text((300, 150), "CPU 42%", fill=(60, 60, 60), font=f_mid)
    els.append({"type": "card", "bbox": [270, 90, 780, 260]})
    # 按钮
    d.rectangle([270, 280, 430, 320], fill=(34, 120, 210))
    d.text((285, 290), "提交", fill=(255, 255, 255), font=f_mid)
    els.append({"type": "button", "bbox": [270, 280, 430, 320], "fill": "#2278D2"})
    im.save(OUT / "layout_page.png")
    (OUT / "layout_gt.json").write_text(json.dumps(els), encoding="utf-8")


def gen_diff() -> None:
    im = Image.new("RGB", (600, 400), (240, 240, 240))
    d = ImageDraw.Draw(im)
    for x in range(0, 600, 60):
        d.line([x, 0, x, 400], fill=(220, 220, 220))
    for y in range(0, 400, 60):
        d.line([0, y, 600, y], fill=(220, 220, 220))
    im.save(OUT / "diff_base.png")
    injected = [(100, 80, 200, 140), (350, 200, 480, 300), (50, 250, 150, 340)]
    im2 = im.copy()
    d2 = ImageDraw.Draw(im2)
    for r in injected:
        d2.rectangle(r, fill=(200, 60, 60))
    im2.save(OUT / "diff_injected.png")
    (OUT / "diff_gt.json").write_text(json.dumps(injected), encoding="utf-8")


def gen_drift_scene() -> None:
    """DOM 声明色 #111111，实际渲染为红色 → crosscheck 颜色漂移基准。"""
    f = font(20)
    im = Image.new("RGB", (500, 200), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.text((100, 60), "标题测试", fill=(224, 0, 0), font=f)
    im.save(OUT / "drift_scene.png")
    dom = {"schema": "vision-report/v2", "coordsys": "css_px",
           "elements": [{"id": 0, "type": "text", "bbox": [100, 60, 260, 90],
                         "text": "标题测试", "color": {"fill": "#FFFFFF", "text": "#111111"},
                         "source": ["dom"]}]}
    (OUT / "drift_dom.json").write_text(json.dumps(dom), encoding="utf-8")


def main() -> int:
    gen_colors()
    gen_text_lines()
    gen_layout_page()
    gen_diff()
    gen_drift_scene()
    print(f"fixtures written to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
