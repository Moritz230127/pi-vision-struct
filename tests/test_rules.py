#!/usr/bin/env python3
"""test_rules.py — 规则引擎自测：contrast / overlap / alignment / spacing / safe_area。

运行（conda env）:
  /home/Arch/conda-envs/pi-vision/bin/python tests/test_rules.py
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / "python"
OUT = ROOT / "tests" / "_out"
OUT.mkdir(parents=True, exist_ok=True)
PYBIN = "/home/Arch/conda-envs/pi-vision/bin/python"

PASS = 0
FAIL = 0


def run_tool(script: str, args: list[str]) -> dict:
    r = subprocess.run([PYBIN, str(PY / script), *args], capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        raise AssertionError(f"tool exit {r.returncode}: {r.stdout[:300]} {r.stderr[:200]}")
    try:
        return json.loads(r.stdout)
    except ValueError as e:
        raise AssertionError(f"not json: {r.stdout[:300]}") from e


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name} {detail}")


def make_report(elements: list[dict], source: dict | None = None) -> dict:
    return {
        "schema": "vision-report/v2", "task": "test", "sensors": ["test"],
        "coordsys": "css_px",
        "source": source or {"type": "fused", "size_px": [1000, 800]},
        "elements": elements,
    }


def findings_of(report: dict) -> list[dict]:
    return report.get("findings", [])


def test_contrast() -> None:
    print("[vs_rules] R1 对比度")
    rep = make_report([
        {"bbox": [0, 0, 50, 20], "texts": [{"text": "t", "color": "#777777", "size_pt": 12}], "fill": "#CCCCCC"},
        {"bbox": [0, 30, 60, 50], "texts": [{"text": "ok", "color": "#000000", "size_pt": 12}], "fill": "#FFFFFF"},
    ])
    p = OUT / "rules_contrast.json"
    p.write_text(json.dumps(rep), encoding="utf-8")
    out = run_tool("vs_rules.py", ["--report", str(p)])
    fs = [f for f in findings_of(out) if f["rule"] == "text_contrast"]
    check("低对比度 (#777/#CCC) 命中", len(fs) == 1 and fs[0]["element_ids"] == [0])
    check("高对比度 (黑/白) 不误报", all(f["element_ids"] != [1] for f in fs))
    check("严重度分级存在", bool(fs) and fs[0]["severity"] in ("warn", "critical"))
    check("证据含 ratio/required", bool(fs) and "ratio" in fs[0]["evidence"] and "required" in fs[0]["evidence"])


def test_overlap() -> None:
    print("[vs_rules] R2 重叠")
    rep = make_report([
        {"bbox": [0, 0, 100, 100]},
        {"bbox": [50, 50, 150, 150]},
        {"bbox": [300, 300, 400, 400]},
    ])
    p = OUT / "rules_overlap.json"
    p.write_text(json.dumps(rep), encoding="utf-8")
    out = run_tool("vs_rules.py", ["--report", str(p)])
    fs = [f for f in findings_of(out) if f["rule"] == "overlap"]
    check("重叠对 (0,1) 命中", len(fs) == 1 and set(fs[0]["element_ids"]) == {0, 1})
    check("远距元素不误报", all(2 not in f["element_ids"] for f in fs))


def test_alignment() -> None:
    print("[vs_rules] R3 对齐漂移")
    rep = make_report([
        {"bbox": [100, 10, 200, 30], "source": ["dom"]},
        {"bbox": [100, 50, 200, 70], "source": ["dom"]},
        {"bbox": [100, 90, 200, 110], "source": ["dom"]},
        {"bbox": [107, 130, 207, 150], "source": ["dom"]},  # 左缘 107：距簇均值 100 为 7px ∈ (4, 16]
    ])
    p = OUT / "rules_align.json"
    p.write_text(json.dumps(rep), encoding="utf-8")
    out = run_tool("vs_rules.py", ["--report", str(p)])
    fs = [f for f in findings_of(out) if f["rule"] == "alignment_drift"]
    check("漂移元素 (idx 3) 命中", any(f["element_ids"] == [3] for f in fs))
    check("对齐组成员不误报", all(f["element_ids"] != [0] for f in fs))
    check("证据含 axis/offset", bool(fs) and "offset" in fs[0]["evidence"] and fs[0]["evidence"]["offset"] >= 4.0)


def test_spacing() -> None:
    print("[vs_rules] R4 间距一致性")
    # 同行 4 设计元素，间距 10 / 12 / 200 → 中位 12，200 > 2.5×12=30 → 命中
    rep = make_report([
        {"bbox": [0, 10, 50, 30], "source": ["dom"]},
        {"bbox": [60, 10, 80, 30], "source": ["dom"]},
        {"bbox": [92, 10, 112, 30], "source": ["dom"]},
        {"bbox": [312, 10, 332, 30], "source": ["dom"]},
    ])
    p = OUT / "rules_spacing.json"
    p.write_text(json.dumps(rep), encoding="utf-8")
    out = run_tool("vs_rules.py", ["--report", str(p)])
    fs = [f for f in findings_of(out) if f["rule"] == "spacing_anomaly"]
    check("大间距 (200px) 命中", len(fs) == 1 and fs[0]["evidence"]["gap"] == 200.0)
    check("证据含 median", bool(fs) and "median" in fs[0]["evidence"])


def test_safe_area() -> None:
    print("[vs_rules] R5 安全区")
    rep = make_report([
        {"bbox": [-10, 0, 30, 20], "text": "clipped"},
        {"bbox": [100, 0, 300, 5], "text": "edge"},
        {"bbox": [100, 100, 300, 200]},
    ])
    p = OUT / "rules_safe.json"
    p.write_text(json.dumps(rep), encoding="utf-8")
    out = run_tool("vs_rules.py", ["--report", str(p)])
    fs = findings_of(out)
    crit = [f for f in fs if f["rule"] == "safe_area" and f["evidence"].get("kind") == "off_canvas"]
    warn = [f for f in fs if f["rule"] == "safe_area" and f["evidence"].get("kind") == "edge_text"]
    check("出界元素 critical", len(crit) == 1 and crit[0]["element_ids"] == [0] and crit[0]["severity"] == "critical")
    check("贴边文本元素 warn", len(warn) == 1 and warn[0]["element_ids"] == [1])
    check("正常元素不误报", all(2 not in f["element_ids"] for f in fs))


def test_ocr_scope() -> None:
    print("[vs_rules] 作用域：OCR 自然文本不误报对齐/间距")
    # 同左缘 100 的 3 个 OCR 元素 + 一个左缘 107 的 OCR 元素（自然文本参差）
    rep = make_report([
        {"bbox": [100, 10, 200, 30], "source": ["ocr"]},
        {"bbox": [100, 50, 200, 70], "source": ["ocr"]},
        {"bbox": [100, 90, 200, 110], "source": ["ocr"]},
        {"bbox": [107, 130, 207, 150], "source": ["ocr"]},
    ])
    p = OUT / "rules_ocr.json"
    p.write_text(json.dumps(rep), encoding="utf-8")
    out = run_tool("vs_rules.py", ["--report", str(p)])
    fs = [f for f in findings_of(out) if f["rule"] in ("alignment_drift", "spacing_anomaly")]
    check("OCR 元素零对齐/间距命中", len(fs) == 0, f"got {len(fs)}")


def test_output_shape() -> None:
    print("[vs_rules] 输出契约")
    rep = make_report([{"bbox": [0, 0, 50, 20]}])
    p = OUT / "rules_shape.json"
    p.write_text(json.dumps(rep), encoding="utf-8")
    out = run_tool("vs_rules.py", ["--report", str(p)])
    check("schema v2 envelope", out.get("schema") == "vision-report/v2" and out.get("task") == "rules")
    check("rules 清单（5 条）", len(out.get("rules", [])) == 5)
    check("metrics 含 design_score", "design_score" in out.get("metrics", {}))
    check("severity 计数", set(out["metrics"]["severity"]) <= {"critical", "warn", "info"})


def main() -> int:
    print(f"rules self-tests (python {sys.version.split()[0]})\n")
    test_contrast()
    test_overlap()
    test_alignment()
    test_spacing()
    test_safe_area()
    test_ocr_scope()
    test_output_shape()
    print(f"\n结果: {PASS} 通过 / {FAIL} 失败")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
