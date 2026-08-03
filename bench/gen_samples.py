#!/usr/bin/env python3
"""gen_samples.py — 美学判断抽样集（Phase 2.3 验收用，确定性合成）。

12 个样本（800x600 画布）：6 缺陷 + 6 干净。每个样本生成：
  - <id>.png          渲染图像（缺陷真实可见，供 VLM critic 裁剪复核）
  - <id>.report.json  schema-v2 元素报告（供规则引擎）
  - <id>.gt.json      人工真值标注 {expected:[{rule,severity}], clean:bool}

缺陷样本（每样本单类缺陷，真值可判）:
  s01 low_contrast  文本 #777/#CCC 对比度不足 → text_contrast
  s02 overlap       两元素重叠 → overlap
  s03 alignment     按钮左缘漂移 7px → alignment_drift
  s04 spacing       卡片行 200px 离群间距 → spacing_anomaly
  s05 offcanvas     元素出界 → safe_area(critical)
  s06 edge_text     文本贴边 → safe_area(warn)
干净样本: s07 网格 / s08 列 / s09 卡片 / s10 侧栏 / s11 表单 / s12 横幅

用法: gen_samples.py [--out bench/samples]
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw  # type: ignore[import-not-found]

W, H = 800, 600
OUT_DEFAULT = Path(__file__).resolve().parent / "samples"


def element(bbox, *, source=("dom",), text="", fill=None, color=None, size_pt=14.0, z=0):
    el = {"bbox": list(bbox), "source": list(source), "text": text, "z": z}
    if fill or color:
        el["color"] = {}
        if fill:
            el["fill"] = fill
            el["color"]["fill"] = fill
        if color:
            el["color"]["text"] = color
            el["texts"] = [{"text": text, "color": color, "size_pt": size_pt}]
    return el


def box(im: Image.Image, bbox, fill, outline="#333333", text="", text_color="#111111", size=16):
    d = ImageDraw.Draw(im)
    d.rectangle(bbox, fill=fill, outline=outline, width=2)
    if text:
        d.text((bbox[0] + 8, bbox[1] + 6), text, fill=text_color)


def sample_s01():
    els = [element([80, 60, 720, 110], text="浅灰文字低对比度示例", fill="#CCCCCC",
                   color="#777777", size_pt=18.0)]
    im = Image.new("RGB", (W, H), "#FFFFFF")
    box(im, [80, 60, 720, 110], "#CCCCCC", text="浅灰文字低对比度示例", text_color="#777777", size=24)
    gt = {"expected": [{"rule": "text_contrast", "severity": "critical"}], "clean": False}
    return im, els, gt


def sample_s02():
    els = [element([100, 80, 400, 260], text="A"), element([300, 200, 620, 380], text="B")]
    im = Image.new("RGB", (W, H), "#FFFFFF")
    box(im, [100, 80, 400, 260], "#DDEEFF", text="A")
    box(im, [300, 200, 620, 380], "#FFEEDD", text="B")
    gt = {"expected": [{"rule": "overlap", "severity": "warn"}], "clean": False}
    return im, els, gt


def sample_s03():
    els = [element([100, 80, 300, 130], text="btn1"),
           element([100, 160, 300, 210], text="btn2"),
           element([100, 240, 300, 290], text="btn3"),
           element([107, 320, 307, 370], text="btn4")]  # 左缘漂移 7px
    im = Image.new("RGB", (W, H), "#FFFFFF")
    for i, (b, t) in enumerate(zip([e["bbox"] for e in els], ["btn1", "btn2", "btn3", "btn4"])):
        box(im, b, "#EEEEEE", text=t)
    gt = {"expected": [{"rule": "alignment_drift", "severity": "info"}], "clean": False}
    return im, els, gt


def sample_s04():
    els = [element([40, 80, 180, 200], text="c1"),
           element([220, 80, 360, 200], text="c2"),
           element([400, 80, 540, 200], text="c3"),
           element([660, 80, 780, 200], text="c4")]  # 间距 120（离群 > 2.5×40=100），且不出界
    im = Image.new("RGB", (W, H), "#FFFFFF")
    for b in [e["bbox"] for e in els]:
        box(im, b, "#E8F0FF", text="card")
    gt = {"expected": [{"rule": "spacing_anomaly", "severity": "info"}], "clean": False}
    return im, els, gt


def sample_s05():
    els = [element([-40, 100, 240, 260], text="出界元素"), element([300, 100, 500, 200], text="正常")]
    im = Image.new("RGB", (W, H), "#FFFFFF")
    box(im, [-40, 100, 240, 260], "#FFEEDD", text="出界元素")
    box(im, [300, 100, 500, 200], "#EEEEEE", text="正常")
    gt = {"expected": [{"rule": "safe_area", "severity": "critical"}], "clean": False}
    return im, els, gt


def sample_s06():
    els = [element([100, 0, 400, 8], text="贴边文本")]
    im = Image.new("RGB", (W, H), "#FFFFFF")
    box(im, [100, 0, 400, 8], "#EEEEEE", text="贴边文本", size=8)
    gt = {"expected": [{"rule": "safe_area", "severity": "warn"}], "clean": False}
    return im, els, gt


def sample_s07():
    els = []
    for r in range(3):
        for c in range(3):
            els.append(element([60 + c * 240, 60 + r * 180, 60 + c * 240 + 160, 60 + r * 180 + 100],
                               text=f"g{r}{c}"))
    im = Image.new("RGB", (W, H), "#FFFFFF")
    for b in [e["bbox"] for e in els]:
        box(im, b, "#E8F0FF", text="")
    gt = {"expected": [], "clean": True}
    return im, els, gt


def sample_s08():
    els = [element([340, 40 + i * 70, 460, 80 + i * 70], text=f"row{i}") for i in range(6)]
    im = Image.new("RGB", (W, H), "#FFFFFF")
    for b in [e["bbox"] for e in els]:
        box(im, b, "#EEEEEE")
    gt = {"expected": [], "clean": True}
    return im, els, gt


def sample_s09():
    els = [element([80, 60, 720, 130], text="卡片标题", z=1),
           element([100, 150, 700, 420], text="正文内容", z=0)]
    im = Image.new("RGB", (W, H), "#FFFFFF")
    box(im, [80, 60, 720, 130], "#224488", text="卡片标题", text_color="#FFFFFF")
    box(im, [100, 150, 700, 420], "#F0F0F0")
    gt = {"expected": [], "clean": True}
    return im, els, gt


def sample_s10():
    els = [element([20, 20, 180, 580], text="侧栏", z=0),
           element([220, 20, 780, 300], text="主区", z=0),
           element([220, 320, 780, 580], text="详情", z=0)]
    im = Image.new("RGB", (W, H), "#FFFFFF")
    box(im, [20, 20, 180, 580], "#F5F5F5")
    box(im, [220, 20, 780, 300], "#E8F0FF")
    box(im, [220, 320, 780, 580], "#F0F8F0")
    gt = {"expected": [], "clean": True}
    return im, els, gt


def sample_s11():
    els = [element([200, 80, 600, 120], text="姓名"), element([200, 150, 600, 190], text="邮箱"),
           element([200, 220, 600, 260], text="密码"), element([200, 290, 600, 330], text="提交")]
    im = Image.new("RGB", (W, H), "#FFFFFF")
    for i, b in enumerate([e["bbox"] for e in els]):
        box(im, b, "#EEEEEE", text=["姓名", "邮箱", "密码", "提交"][i])
    gt = {"expected": [], "clean": True}
    return im, els, gt


def sample_s12():
    els = [element([0, 0, 800, 120], text="深色横幅白色文字", fill="#222222",
                   color="#FFFFFF", size_pt=28.0)]
    im = Image.new("RGB", (W, H), "#FFFFFF")
    box(im, [0, 0, 800, 120], "#222222", text="深色横幅白色文字", text_color="#FFFFFF", size=30)
    gt = {"expected": [], "clean": True}
    return im, els, gt


SAMPLES = {
    "s01_low_contrast": sample_s01,
    "s02_overlap": sample_s02,
    "s03_alignment": sample_s03,
    "s04_spacing": sample_s04,
    "s05_offcanvas": sample_s05,
    "s06_edge_text": sample_s06,
    "s07_clean_grid": sample_s07,
    "s08_clean_column": sample_s08,
    "s09_clean_card": sample_s09,
    "s10_clean_sidebar": sample_s10,
    "s11_clean_form": sample_s11,
    "s12_clean_banner": sample_s12,
}


def build(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for sid, fn in SAMPLES.items():
        im, els, gt = fn()
        im.save(out_dir / f"{sid}.png")
        report = {
            "schema": "vision-report/v2", "task": "sample", "sensors": ["sample"],
            "coordsys": "css_px",
            "source": {"type": "synthetic", "size_px": [W, H]},
            "elements": els,
        }
        (out_dir / f"{sid}.report.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        (out_dir / f"{sid}.gt.json").write_text(json.dumps(gt, ensure_ascii=False), encoding="utf-8")
    print(f"生成 {len(SAMPLES)} 个样本 -> {out_dir}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()
    try:
        build(args.out)
        return 0
    except Exception as e:
        print(json.dumps({"error": "gen_samples failed", "detail": str(e)[:300]}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
