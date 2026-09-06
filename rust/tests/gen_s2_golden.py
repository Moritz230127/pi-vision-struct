#!/usr/bin/env python3
"""gen_s2_golden.py — 生成 S2 传感器 golden 输出（Python 基线）。

用法: python3 rust/tests/gen_s2_golden.py > rust/tests/s2_golden.json
覆盖: fusion（报告对融合）/ rules（规则引擎）/ audit（审计）/ geometry（VTracer）。
依赖: pi-vision env（vs_fusion/vs_rules/vs_audit）+ vsensor env（vs_geometry/vtracer）。
"""
import json
import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
PY_DIR = os.path.join(ROOT, "python")
SAMPLES = os.path.join(ROOT, "bench", "samples")

# 规则基准样本（12 个合成 UI 报告）
REPORTS = sorted(
    f for f in os.listdir(SAMPLES) if f.endswith(".report.json")
)[:6]  # 取前 6 个控制测试时长


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
        capture_output=True, text=True, timeout=180,
    )
    if r.returncode != 0:
        raise RuntimeError(f"{script} failed: {r.stderr[:400]}")
    return json.loads(r.stdout)


def main() -> None:
    pv = resolve_python("pi-vision")
    vs = resolve_python("vsensor")
    golden = {}

    # rules：每个样本报告
    golden["rules"] = {}
    for rep in REPORTS:
        path = os.path.join(SAMPLES, rep)
        key = rep.replace(".report.json", "")
        golden["rules"][key] = run(pv, "vs_rules.py", "--report", path)

    # audit：前 3 个样本
    golden["audit"] = {}
    for rep in REPORTS[:3]:
        path = os.path.join(SAMPLES, rep)
        key = rep.replace(".report.json", "")
        golden["audit"][key] = run(pv, "vs_audit.py", "--report", path)

    # fusion：报告对（同源元素融合）
    golden["fusion"] = {}
    pairs = [(REPORTS[0], REPORTS[1]), (REPORTS[2], REPORTS[3])]
    for i, (ra, rb) in enumerate(pairs):
        pa = os.path.join(SAMPLES, ra)
        pb = os.path.join(SAMPLES, rb)
        golden["fusion"][f"pair{i}"] = run(pv, "vs_fusion.py", "--reports", pa, pb)

    # geometry：样本图像（vsensor env 有 vtracer）
    golden["geometry"] = {}
    for img in ["s01_low_contrast.png", "s02_overlap.png"]:
        path = os.path.join(SAMPLES, img)
        key = img.replace(".png", "")
        golden["geometry"][key] = run(vs, "vs_geometry.py", "--image", path, "--max-shapes", "20")

    print(json.dumps(golden, ensure_ascii=False))


if __name__ == "__main__":
    main()
