#!/usr/bin/env python3
"""tune_rules.py — 规则引擎阈值的数据驱动寻优（"训练"的确定性形式）。

方法：
  1. 生成 24 个边界样本（每个带人工真值: 该不该报）
  2. 网格搜索规则参数（overlap_threshold / min_overlap_area / align_tol /
     spacing_k / margin）
  3. 用一致率（agreement）作为目标函数，选出最优参数
  4. 对比默认参数 vs 最优参数的提升

完全本地、确定性、零成本。结果写 bench/tune_report.json。

用法: /home/Arch/conda-envs/pi-vision/bin/python bench/tune_rules.py
"""
import itertools
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / "python"
OUT = ROOT / "bench" / "_tune"
OUT.mkdir(parents=True, exist_ok=True)
PYBIN = "/home/Arch/conda-envs/pi-vision/bin/python"

W, H = 800, 600
DEFAULTS = {"overlap_threshold": 0.05, "min_overlap_area": 400.0,
            "align_tol": 4.0, "spacing_k": 2.5, "margin": 2.0}


def el(bbox, src=("dom",), text=None):
    e = {"bbox": list(bbox), "source": list(src)}
    if text:
        e["text"] = text
    return e


def make_samples() -> dict[str, tuple[list[dict], dict]]:
    """id -> (elements, gt)。gt: {expect_flag: bool, rule: str}。"""
    S: dict[str, tuple[list[dict], dict]] = {}

    # ---- overlap: 触碰(不报) / 轻叠(边界) / 实质重叠(报) ----
    S["ov_touch"] = ([el([100, 100, 300, 300]), el([300, 100, 500, 300])],
                     {"expect": False, "rule": "overlap"})  # 边贴边
    S["ov_slight"] = ([el([100, 100, 300, 300]), el([296, 100, 496, 300])],
                      {"expect": True, "rule": "overlap"})  # 4px 重叠(4x300=1200)
    S["ov_real"] = ([el([100, 100, 300, 300]), el([250, 250, 450, 450])],
                    {"expect": True, "rule": "overlap"})  # 明显重叠

    # ---- alignment: 0(对齐) / 3px(不报) / 5px(报) / 12px(报) ----
    base = [el([100, 40, 200, 70]), el([100, 90, 200, 120]), el([100, 140, 200, 170])]
    S["al_ok"] = (base + [el([100, 190, 200, 220])], {"expect": False, "rule": "alignment_drift"})
    S["al_3px"] = (base + [el([103, 190, 203, 220])], {"expect": False, "rule": "alignment_drift"})
    S["al_5px"] = (base + [el([105, 190, 205, 220])], {"expect": True, "rule": "alignment_drift"})
    S["al_12px"] = (base + [el([112, 190, 212, 220])], {"expect": True, "rule": "alignment_drift"})

    # ---- spacing: 比率 1.5x(不报) / 2.5x(边界) / 4x(报) ----
    def row(gap_ratio: float) -> list[dict]:
        # 基准间距 40，末位间距 = 40*ratio
        g = 40 * gap_ratio
        return [el([40, 80, 180, 200]), el([220, 80, 360, 200]),
                el([400, 80, 540, 200]), el([540 + g, 80, 540 + g + 140, 200])]
    S["sp_1_5x"] = (row(1.5), {"expect": False, "rule": "spacing_anomaly"})
    S["sp_3x"] = (row(3.0), {"expect": True, "rule": "spacing_anomaly"})
    S["sp_4x"] = (row(4.0), {"expect": True, "rule": "spacing_anomaly"})

    # ---- safe_area: 距边 5px(不报) / 1px(报) / 出界(报) ----
    S["sa_5px"] = ([el([100, 5, 300, 60], text="t")], {"expect": False, "rule": "safe_area"})
    S["sa_1px"] = ([el([100, 1, 300, 60], text="t")], {"expect": True, "rule": "safe_area"})
    S["sa_off"] = ([el([-30, 100, 200, 260])], {"expect": True, "rule": "safe_area"})

    return S


def run_rules(report_path: str, params: dict) -> list[dict]:
    args = [str(PY / "vs_rules.py"), "--report", report_path,
            "--overlap-threshold", str(params["overlap_threshold"]),
            "--min-overlap-area", str(params["min_overlap_area"]),
            "--align-tol", str(params["align_tol"]),
            "--spacing-k", str(params["spacing_k"]),
            "--margin", str(params["margin"])]
    r = subprocess.run([PYBIN, *args], capture_output=True, text=True, timeout=120)
    return json.loads(r.stdout).get("findings", [])


def evaluate(params: dict, samples: dict) -> float:
    correct = 0
    for sid, (els, gt) in samples.items():
        report = {"schema": "vision-report/v2", "task": "tune", "sensors": ["tune"],
                  "coordsys": "css_px",
                  "source": {"type": "synthetic", "size_px": [W, H]},
                  "elements": els}
        p = OUT / f"{sid}.json"
        p.write_text(json.dumps(report), encoding="utf-8")
        findings = run_rules(str(p), params)
        hit = any(f["rule"] == gt["rule"] for f in findings)
        if hit == gt["expect"]:
            correct += 1
    return correct / len(samples)


def main() -> int:
    samples = make_samples()
    n = len(samples)
    print(f"边界样本: {n} 个（overlap 3 / alignment 4 / spacing 3 / safe_area 3）")

    base = evaluate(DEFAULTS, samples)
    print(f"默认参数一致率: {base:.1%} ({int(base*n)}/{n})")

    grid = {
        "overlap_threshold": [0.01, 0.03, 0.05, 0.08, 0.12],
        "min_overlap_area": [200.0, 400.0, 800.0, 1500.0],
        "align_tol": [2.0, 4.0, 6.0, 8.0],
        "spacing_k": [1.8, 2.5, 3.5, 5.0],
        "margin": [1.0, 2.0, 4.0, 6.0],
    }
    keys = list(grid.keys())
    best = {"params": dict(DEFAULTS), "score": base}
    results = []
    total = 1
    for k in keys:
        total *= len(grid[k])
    print(f"搜索空间: {total} 组合（贪心坐标下降，5×{len(keys)} 次评估）...")

    # 坐标下降：轮流优化每个参数（确定性，避免全网格爆炸）
    cur = dict(DEFAULTS)
    results = []
    for _round in range(3):
        improved = False
        for k in keys:
            scores = []
            for v in grid[k]:
                trial = dict(cur)
                trial[k] = v
                scores.append((evaluate(trial, samples), v, trial))
            best_v = max(scores, key=lambda t: t[0])[1]
            if best_v != cur[k]:
                cur[k] = best_v
                improved = True
            results.append(max(scores, key=lambda t: t[0]))
        if not improved:
            break

    final_score = evaluate(cur, samples)
    print(f"寻优后一致率: {final_score:.1%} ({int(final_score*n)}/{n})")
    print(f"默认参数    : {DEFAULTS}")
    print(f"最优参数    : {cur}")

    report = {"default_score": base, "tuned_score": final_score,
              "defaults": DEFAULTS, "tuned": cur,
              "sample_count": n,
              "n_evals": len(results)}
    (Path(__file__).resolve().parent / "tune_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告 -> bench/tune_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
