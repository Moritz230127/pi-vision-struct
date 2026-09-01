#!/usr/bin/env python3
"""vs_protocol.py — V3 多轮反馈协议（analyze/zoom/probe）。

SeeingEye 模式：DeepSeek 通过三个动作驱动多轮视觉推理：
  analyze [image]      → 粗报告（saliency 候选 + ascii 粗栅格 + 全局统计 + 融合）
  zoom [region]        → 细报告（region 内 ocr 高倍 + segment 前景 + edge + depth）
  probe [bbox, sensor] → 单传感器定向证据

用法:
  vs_protocol.py analyze --image PATH
  vs_protocol.py zoom --image PATH --region x1,y1,x2,y2
  vs_protocol.py probe --image PATH --bbox x1,y1,x2,y2 --sensor ocr|edge|depth|segment
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import vs_schema as S

ROOT = Path(__file__).resolve().parent
PYBIN = f"{Path.home()}/conda-envs/pi-vision/bin/python"
VSENSOR = f"{Path.home()}/conda-envs/vsensor/bin/python"


def run(pybin: str, script: str, *args: str, timeout: int = 180) -> dict:
    """运行传感器脚本，返回 JSON。"""
    cmd = [pybin, str(ROOT / script), *args]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return {"error": f"{script} 输出非 JSON", "detail": (r.stdout or r.stderr)[:300]}


def analyze(image: str, dpr: float = 1.0) -> dict:
    """粗报告：saliency 候选 + ascii 粗栅格 + scene_stats + ocr + 融合。"""
    # 并行传感器（子进程）
    sal = run(VSENSOR, "vs_saliency.py", "--image", image, "--device", "cuda")
    ascii_r = run(PYBIN, "vs_ascii.py", "--image", image, "--cols", "64", "--rows", "36")
    stats = run(PYBIN, "vs_scene_stats.py", "--image", image)
    ocr = run(PYBIN, "vs_ocr.py", "--image", image, "--max-items", "60")

    # 融合（ocr + saliency 候选）
    fusion = None
    if "error" not in ocr and "error" not in sal:
        # 构造融合输入
        els = (ocr.get("elements") or []) + (sal.get("elements") or [])
        fusion = run(PYBIN, "vs_fusion.py", "--reports", _tmp_report(ocr), _tmp_report(sal))

    report = S.envelope(task="analyze", sensors=["saliency", "ascii", "scene_stats", "ocr", "fusion"],
                        coordsys="image_px", source={"type": "image", "path": image})
    report["candidates"] = sal.get("candidates", []) if "error" not in sal else []
    report["ascii"] = ascii_r.get("ascii", {}) if "error" not in ascii_r else {}
    report["metrics"] = {
        "scene": stats.get("metrics", {}) if "error" not in stats else {},
        "ocr_items": len(ocr.get("elements", [])) if "error" not in ocr else 0,
        "saliency_candidates": len(report["candidates"]),
    }
    if fusion and "error" not in fusion:
        report["findings"] = fusion.get("findings", [])
    report["notation"] = S.NOTATION_GUIDE
    report["note"] = ("粗报告：先用 candidates 定位有效内容，再用 zoom 放大细看，"
                      "或 probe 定向取证。")
    return report


def zoom(image: str, region: str) -> dict:
    """细报告：region 内 ocr 高倍 + segment 前景 + edge + depth。"""
    x1, y1, x2, y2 = (int(v) for v in region.split(","))
    region_str = f"{x1},{y1},{x2},{y2}"

    ocr = run(PYBIN, "vs_ocr.py", "--image", image, "--region", region_str,
              "--upscale", "2", "--max-items", "40")
    edge = run(PYBIN, "vs_edge.py", "--image", image)
    depth = run(VSENSOR, "vs_depth.py", "--image", image, "--region", region_str,
                "--device", "cuda")
    seg = run(VSENSOR, "vs_segment.py", "--image", image, "--device", "cuda")

    report = S.envelope(task="zoom", sensors=["ocr", "edge", "depth", "segment"],
                        coordsys="image_px",
                        source={"type": "image", "path": image, "region": [x1, y1, x2, y2]})
    report["elements"] = (ocr.get("elements", []) if "error" not in ocr else []) + \
                         (edge.get("elements", []) if "error" not in edge else [])
    report["foreground"] = seg.get("foreground") if "error" not in seg else None
    report["depth"] = depth.get("depth", {}) if "error" not in depth else {}
    report["metrics"] = {
        "ocr_items": len(ocr.get("elements", [])) if "error" not in ocr else 0,
        "edge_points": edge.get("metrics", {}).get("edge_points", 0) if "error" not in edge else 0,
        "region": [x1, y1, x2, y2],
    }
    report["notation"] = S.NOTATION_GUIDE
    report["note"] = ("细报告：region 内 OCR 已 2× 放大；edge 给出亚像素边缘；"
                      "depth 给出区域深度；foreground 给出前景分割。")
    return report


def probe(image: str, bbox: str, sensor: str) -> dict:
    """单传感器定向证据。"""
    x1, y1, x2, y2 = (int(v) for v in bbox.split(","))
    region_str = f"{x1},{y1},{x2},{y2}"

    if sensor == "ocr":
        r = run(PYBIN, "vs_ocr.py", "--image", image, "--region", region_str,
                "--upscale", "3", "--max-items", "20")
    elif sensor == "edge":
        r = run(PYBIN, "vs_edge.py", "--image", image)
    elif sensor == "depth":
        r = run(VSENSOR, "vs_depth.py", "--image", image, "--region", region_str,
                "--device", "cuda")
    elif sensor == "segment":
        r = run(VSENSOR, "vs_segment.py", "--image", image, "--device", "cuda")
    elif sensor == "pix":
        r = run(VSENSOR, "vs_pix.py", "--image", image, "--regions", region_str)
    else:
        return {"error": f"unknown sensor: {sensor}", "detail": "可选: ocr|edge|depth|segment|pix"}

    report = S.envelope(task="probe", sensors=[sensor], coordsys="image_px",
                        source={"type": "image", "path": image, "bbox": [x1, y1, x2, y2]})
    report["elements"] = r.get("elements", []) if "error" not in r else []
    report["metrics"] = r.get("metrics", {}) if "error" not in r else {}
    if "regions" in r:
        report["regions"] = r["regions"]
    if "depth" in r:
        report["depth"] = r["depth"]
    report["notation"] = S.NOTATION_GUIDE
    report["note"] = f"定向证据：{sensor} 传感器在 bbox 区域的结果。"
    return report


def _tmp_report(data: dict) -> str:
    """把传感器输出写临时文件（供 vs_fusion 读取）。"""
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".json", prefix="vs_sensor_")
    with open(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="action", required=True)

    p_an = sub.add_parser("analyze")
    p_an.add_argument("--image", required=True)

    p_zoom = sub.add_parser("zoom")
    p_zoom.add_argument("--image", required=True)
    p_zoom.add_argument("--region", required=True, help="x1,y1,x2,y2")

    p_probe = sub.add_parser("probe")
    p_probe.add_argument("--image", required=True)
    p_probe.add_argument("--bbox", required=True, help="x1,y1,x2,y2")
    p_probe.add_argument("--sensor", required=True)

    args = ap.parse_args()
    try:
        if args.action == "analyze":
            print(S.dump_json(analyze(args.image)))
        elif args.action == "zoom":
            print(S.dump_json(zoom(args.image, args.region)))
        elif args.action == "probe":
            print(S.dump_json(probe(args.image, args.bbox, args.sensor)))
        return 0
    except Exception as e:
        print(json.dumps({"error": "vs_protocol failed", "detail": str(e)[:500]},
                         ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
