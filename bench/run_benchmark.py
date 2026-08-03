#!/usr/bin/env python3
"""bench/run_benchmark.py — 基准套件：跑工具 + 计算指标 + 产出报告卡。

指标:
  color_dE_max/mean   颜色精度（实测 hex vs 真值 ΔE76，PIL 合成应为 0）
  ocr_cer_mean        文本识别 CER（字符编辑距离/长度，去空白比较）
  ocr_bbox_iou_mean   OCR bbox 与真值 IoU
  diff_iou_mean       diff 异常定位 IoU（检出框 vs 注入框）
  diff_recall         注入异常检出率
  drift_detected      crosscheck 颜色漂移检出

用法: /home/Arch/conda-envs/pi-vision/bin/python bench/run_benchmark.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / "python"
FIX = ROOT / "tests" / "bench_fixtures"
PYBIN = "/home/Arch/conda-envs/pi-vision/bin/python"

sys.path.insert(0, str(PY))
import vs_schema as S  # type: ignore[import-not-found]


def load_gt(name: str) -> list[dict]:
    try:
        return json.loads((FIX / name).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise RuntimeError(f"cannot load fixture {name}: {e}") from e


def run_tool(script: str, args: list[str], timeout: int = 120) -> dict:
    r = subprocess.run([PYBIN, str(PY / script), *args], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{script} exit {r.returncode}: {r.stdout[:200]} {r.stderr[:150]}")
    try:
        return json.loads(r.stdout)
    except ValueError as e:
        raise RuntimeError(f"{script} bad JSON: {r.stdout[:200]}") from e


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def norm(s: str) -> str:
    return "".join(ch for ch in s if not ch.isspace())


def bench_color() -> dict:
    gt = load_gt("colors_gt.json")
    regions = [f"{r['bbox'][0]},{r['bbox'][1]},{r['bbox'][2]},{r['bbox'][3]}" for r in gt]
    d = run_tool("vs_pix.py", ["--image", str(FIX / "colors.png"), "--regions", *regions])
    regs = {tuple(r["bbox"]): r["hex"] for r in d["regions"]}
    des = []
    for r in gt:
        key = tuple(r["bbox"])
        if key in regs:
            des.append(S.delta_e76(S.hex_to_rgb(r["hex"]), S.hex_to_rgb(regs[key])))
    return {"color_dE_max": round(max(des), 3) if des else None,
            "color_dE_mean": round(sum(des) / len(des), 3) if des else None,
            "color_regions_measured": len(des), "color_regions_gt": len(gt)}


def bench_ocr() -> dict:
    gt = load_gt("text_gt.json")
    d = run_tool("vs_ocr.py", ["--image", str(FIX / "text_lines.png"), "--max-items", "30"])
    ocr_texts = [it["text"] for it in d["elements"]]
    # CER：逐行与真值匹配（OCR 行合并/拆分处理：取编辑距离最小）
    cers, ious = [], []
    for g in gt:
        gnorm = norm(g["text"])
        best = min((levenshtein(gnorm, norm(o)) / max(len(gnorm), 1) for o in ocr_texts), default=1.0)
        cers.append(best)
        # bbox IoU：OCR 项与真值行匹配
        best_iou = 0.0
        for o in d["elements"]:
            best_iou = max(best_iou, S.bbox_iou(g["bbox"], o["bbox"]))
        ious.append(best_iou)
    return {"ocr_cer_mean": round(sum(cers) / len(cers), 4),
            "ocr_bbox_iou_mean": round(sum(ious) / len(ious), 3)}


def bench_diff() -> dict:
    injected = load_gt("diff_gt.json")
    d = run_tool("vs_pix.py", ["--image", str(FIX / "diff_base.png"),
                               "--compare", str(FIX / "diff_injected.png")])
    detected = [a["bbox"] for a in d.get("anomalies", []) if a["type"] == "pixel_diff"]
    ious = []
    if detected:
        for inj in injected:
            ious.append(max(S.bbox_iou(inj, det) for det in detected))
    return {"diff_iou_mean": round(sum(ious) / len(ious), 3) if ious else 0.0,
            "diff_recall": round(len([i for i in ious if i > 0.3]) / len(injected), 2),
            "diff_detected_count": len(detected)}


def bench_drift() -> dict:
    d = run_tool("vs_crosscheck.py", ["--image", str(FIX / "drift_scene.png"),
                                      "--dom", str(FIX / "drift_dom.json")])
    drift = [a for a in d["anomalies"] if a["type"] == "color_drift"]
    return {"drift_detected": len(drift) > 0,
            "drift_delta_e76": drift[0]["evidence"]["delta_e76"] if drift else None}


def main() -> int:
    t0 = time.time()
    results = {}
    print("running benchmark...")
    for name, fn in [("color", bench_color), ("ocr", bench_ocr), ("diff", bench_diff), ("drift", bench_drift)]:
        try:
            results[name] = fn()
            print(f"  {name}: {results[name]}")
        except Exception as e:
            results[name] = {"error": str(e)[:200]}
            print(f"  {name}: ERROR {e}")
    results["_meta"] = {"elapsed_s": round(time.time() - t0, 1)}

    card = FIX / ".." / ".." / "bench" / "report_card.json"
    card = (ROOT / "bench" / "report_card.json")
    card.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreport card: {card}")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
