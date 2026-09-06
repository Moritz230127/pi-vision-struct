#!/usr/bin/env python3
"""gen_detect_golden.py — 生成 detect（OWLv2）golden 输出（Python 基线）。

用法: python3 rust/tests/gen_detect_golden.py > rust/tests/detect_golden.json
输出: 与 Rust vs-core::detect 对比的基准 JSON。

依赖: vsensor env（transformers + torch + OWLv2 权重）。
解释器解析顺序：PI_VISION_VSENSOR_PYTHON 环境变量 >
~/conda-envs/vsensor/bin/python > python3。

图像选择：
  text_lines.png（tests/bench_fixtures）— 有 3 个 text 检测（主验证）
  s01/s02（bench/samples）— UI 截图，person,car,text 下 0 检测（验证空输出一致）
"""
import json
import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
PY_DIR = os.path.join(ROOT, "python")
SAMPLES = os.path.join(ROOT, "bench", "samples")
FIXTURES = os.path.join(ROOT, "tests", "bench_fixtures")

CLASSES = "person,car,text"


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
        capture_output=True, text=True, timeout=600,
    )
    if r.returncode != 0:
        raise RuntimeError(f"{script} failed: {r.stderr[:400]}")
    return json.loads(r.stdout)


def main() -> None:
    vs = resolve_python("vsensor")
    golden = {}
    # text_lines：有检测（主验证）；s01/s02：0 检测（空输出一致性）
    images = [
        (os.path.join(FIXTURES, "text_lines.png"), "text_lines"),
        (os.path.join(SAMPLES, "s01_low_contrast.png"), "s01_low_contrast"),
        (os.path.join(SAMPLES, "s02_overlap.png"), "s02_overlap"),
    ]
    for path, key in images:
        golden[key] = {
            "detect": run(
                vs, "vs_detect.py",
                "--image", path, "--classes", CLASSES,
            )
        }
    print(json.dumps(golden, ensure_ascii=False))


if __name__ == "__main__":
    main()
