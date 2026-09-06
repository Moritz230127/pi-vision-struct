#!/usr/bin/env python3
"""gen_sensor_golden.py — 生成 S1 传感器 golden 输出（Python 基线）。

用法: python3 rust/tests/gen_sensor_golden.py > rust/tests/sensor_golden.json
输出: 与 Rust vs-core::ascii/scene_stats/pix/edge 逐字段对比的基准 JSON。

依赖: pi-vision conda env（skimage/PIL/cv2/numpy）。解释器解析顺序：
      PI_VISION_PYTHON 环境变量 > ~/conda-envs/pi-vision/bin/python > python3。
"""
import json
import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
PY_DIR = os.path.join(ROOT, "python")
SAMPLES = os.path.join(ROOT, "bench", "samples")

# 测试图像（bench/samples 合成 UI 样本，确定性）
IMAGES = ["s01_low_contrast.png", "s02_overlap.png", "s03_alignment.png"]


def resolve_python() -> str:
    env = os.environ.get("PI_VISION_PYTHON")
    if env and os.path.exists(env):
        return env
    cand = os.path.expanduser("~/conda-envs/pi-vision/bin/python")
    if os.path.exists(cand):
        return cand
    return sys.executable


def run(script: str, *args: str) -> dict:
    r = subprocess.run(
        [PYBIN, os.path.join(PY_DIR, script), *args],
        capture_output=True, text=True, timeout=180,
    )
    if r.returncode != 0:
        raise RuntimeError(f"{script} failed: {r.stderr[:400]}")
    return json.loads(r.stdout)


def main() -> None:
    golden = {}
    for img in IMAGES:
        path = os.path.join(SAMPLES, img)
        key = img.replace(".png", "")
        golden[key] = {
            "ascii": run("vs_ascii.py", "--image", path, "--cols", "64", "--rows", "36"),
            "ascii_color": run("vs_ascii.py", "--image", path, "--cols", "32", "--rows", "18", "--color"),
            "scene_stats": run("vs_scene_stats.py", "--image", path, "--colors", "8"),
            "pix": run("vs_pix.py", "--image", path, "--colors", "8"),
            "edge": run("vs_edge.py", "--image", path, "--max-lines", "20"),
        }
    print(json.dumps(golden, ensure_ascii=False))


if __name__ == "__main__":
    PYBIN = resolve_python()
    main()
