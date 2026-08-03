#!/usr/bin/env python3
"""Phase 2 用例3 — 壁纸批量程序化分类 + opt-in 语义标签。

生成 4 张合成壁纸（暖色亮/冷色暗/中性灰/绿色中饱和）→ vs_wall 批量分类 →
断言程序化归类（色相族/冷暖/亮度档/饱和度档/宽高比）。
语义标签（L2, opt-in）不在本自动测试内依赖 GPU，单独在 pi 会话中验收。

运行: /home/Arch/conda-envs/pi-vision/bin/python tests/case3_wallpaper.py
"""
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

from PIL import Image  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / "python"
OUT = ROOT / "tests" / "_out" / "case3"
PYBIN = "/home/Arch/conda-envs/pi-vision/bin/python"

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name} {detail}")


def gradient(w: int, h: int, top: tuple, bottom: tuple) -> Image.Image:
    im = Image.new("RGB", (w, h))
    px = cast(Any, im.load())
    for y in range(h):
        t = y / max(h - 1, 1)
        c = tuple(round(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
        for x in range(w):
            px[x, y] = c
    return im


def make_fixtures() -> Path:
    d = OUT
    d.mkdir(parents=True, exist_ok=True)
    gradient(1920, 1080, (255, 170, 0), (255, 220, 120)).save(d / "warm_bright.png")
    gradient(1080, 1920, (5, 12, 50), (20, 40, 100)).save(d / "cool_dark.png")
    gradient(1920, 1080, (200, 200, 200), (245, 245, 245)).save(d / "neutral_gray.png")
    gradient(1600, 1200, (60, 140, 70), (90, 180, 110)).save(d / "green_med.png")
    return d


def main() -> int:
    wall_dir = make_fixtures()

    r = subprocess.run([PYBIN, str(PY / "vs_wall.py"), "--dir", str(wall_dir),
                        "--colors", "5"], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise AssertionError(f"vs_wall exit {r.returncode}: {r.stderr[:300]}")
    try:
        d = json.loads(r.stdout)
    except ValueError as e:
        raise AssertionError(f"bad JSON from vs_wall: {r.stdout[:200]}") from e

    items = {i["file"]: i for i in d["items"]}
    print("[case3] vs_wall 程序化分类结果")
    for fname, it in items.items():
        print(f"  {fname}: {it['programmatic']} aspect={it['aspect']} "
              f"b={it['metrics']['brightness']} s={it['metrics']['saturation']}")

    print("[case3] 断言")
    wb = items["warm_bright.png"]["programmatic"]
    check("warm_bright: 色相族=橙/黄", wb["family"] in ("橙", "黄"), str(wb["family"]))
    check("warm_bright: 暖色·亮·高饱和·横版",
          wb["warm_cool"] == "暖色" and wb["tone"] == "亮" and wb["sat_tier"] == "高饱和"
          and items["warm_bright.png"]["aspect"] == "横版",
          str(wb))

    cd = items["cool_dark.png"]["programmatic"]
    check("cool_dark: 色相族=蓝", cd["family"] == "蓝", str(cd["family"]))
    check("cool_dark: 冷色·暗·竖版",
          cd["warm_cool"] == "冷色" and cd["tone"] == "暗" and items["cool_dark.png"]["aspect"] == "竖版",
          str(cd))

    ng = items["neutral_gray.png"]["programmatic"]
    check("neutral_gray: 灰/中性（无主色相）", ng["family"] == "灰/中性" and ng["warm_cool"] == "中性",
          str(ng))
    check("neutral_gray: 低饱和", ng["sat_tier"] == "低饱和", str(ng))

    gm = items["green_med.png"]["programmatic"]
    check("green_med: 色相族=绿", gm["family"] == "绿", str(gm["family"]))

    groups = d["groups"]
    check("分组 by_family 含 4 族", len(groups["by_family"]) == 4, str(groups["by_family"]))
    check("分组 by_tone 含 暗/中/亮", set(groups["by_tone"]) >= {"暗", "中", "亮"}, str(groups["by_tone"]))

    # 语义开关默认关闭（opt-in 验证）
    r2 = subprocess.run([PYBIN, str(PY / "vs_semantic.py"), "--image", str(wall_dir / "warm_bright.png")],
                        capture_output=True, text=True, timeout=60)
    try:
        sem = json.loads(r2.stdout)
    except ValueError as e:
        raise AssertionError(f"bad JSON from vs_semantic: {r2.stdout[:200]}") from e
    check("L2 语义默认 opt-in 关闭（enabled:false）",
          sem.get("semantic", {}).get("enabled") == False, str(sem))  # noqa: E712

    print(f"\ncase3 结果: {PASS} 通过 / {FAIL} 失败")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
