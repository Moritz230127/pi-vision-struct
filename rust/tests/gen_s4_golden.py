#!/usr/bin/env python3
"""gen_s4_golden.py — 生成 S4 DL 传感器 golden 输出（Python 基线）。

用法: python3 rust/tests/gen_s4_golden.py > rust/tests/s4_golden.json
输出: 与 Rust vs-core::saliency/depth/segment 对比的基准 JSON。

依赖: pi-vision env（onnxruntime + u2net/midas）+ vsensor env（mobile_sam）。
解释器解析顺序：PI_VISION_PYTHON / PI_VISION_VSENSOR_PYTHON 环境变量 >
~/conda-envs/{pi-vision,vsensor}/bin/python > python3。
"""
import json
import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
PY_DIR = os.path.join(ROOT, "python")
SAMPLES = os.path.join(ROOT, "bench", "samples")

IMAGES = ["s01_low_contrast.png", "s02_overlap.png", "s03_alignment.png"]


def resolve_python(env_name: str) -> str:
    env = os.environ.get(f"PI_VISION_{env_name.upper()}_PYTHON")
    if env and os.path.exists(env):
        return env
    cand = os.path.expanduser(f"~/conda-envs/{env_name}/bin/python")
    if os.path.exists(cand):
        return cand
    return sys.executable


def run(pybin: str, script: str, *args: str) -> dict:
    r = subprocess.run(
        [pybin, os.path.join(PY_DIR, script), *args],
        capture_output=True, text=True, timeout=300,
    )
    if r.returncode != 0:
        raise RuntimeError(f"{script} failed: {r.stderr[:400]}")
    return json.loads(r.stdout)


def main() -> None:
    pv = resolve_python("pi-vision")
    vs = resolve_python("vsensor")
    golden = {}
    for img in IMAGES:
        path = os.path.join(SAMPLES, img)
        key = img.replace(".png", "")
        golden[key] = {
            "saliency": run(pv, "vs_saliency.py", "--image", path, "--device", "cpu"),
            "depth": run(pv, "vs_depth.py", "--image", path, "--device", "cpu"),
            "segment": run(vs, "vs_segment.py", "--image", path, "--device", "cpu"),
        }
    print(json.dumps(golden, ensure_ascii=False))


if __name__ == "__main__":
    main()
