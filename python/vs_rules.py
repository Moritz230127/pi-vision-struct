#!/usr/bin/env python3
"""vs_rules.py — 确定性设计准则规则引擎（Phase 2.3，schema v2）。

对任意 schema-v2 元素列表（DOM / PPTX / OmniParser / OCR 融合报告）执行
确定性设计准则检查。所有规则基于测量值 + 显式阈值，不做学习式判断：

  R1 text_contrast   文本色 vs 背景色 WCAG 对比度（AA 4.5:1；大字号 3:1）
  R2 overlap         元素两两重叠（IoU > 阈值）
  R3 alignment_drift 左/右/水平中线对齐聚类中的近邻漂移
  R4 spacing_anomaly 同行/同列连续间距的离群值（> k × 中位间距）
  R5 safe_area       元素贴边（≤ margin px）或出界

输出: schema-v2 envelope + findings[] + metrics（checks/passed/failed/design_score）
      + rules[]（规则清单与阈值，保证可审计性）。

用法:
  vs_rules.py --report report.json [--canvas WxH]
    [--align-tol 4] [--align-drift 4] [--margin 2] [--spacing-k 2.5]
    [--overlap-threshold 0.05] [--min-overlap-area 400]
"""
import argparse
import json
import sys
from typing import Any

import vs_schema as S


RULES = [
    {"code": "text_contrast", "desc": "WCAG AA 对比度（正常文本≥4.5:1，大文本≥3:1）",
     "thresholds": {"aa_normal": 4.5, "aa_large": 3.0}},
    {"code": "overlap", "desc": "两两 bbox 重叠（IoU 与最小重叠面积）",
     "thresholds": {"iou": 0.05, "min_area": 400.0}},
    {"code": "alignment_drift", "desc": "边缘/中线聚类内的近邻漂移（> tol 且 ≤ drift×tol）；仅设计元素（source∈dom/pptx 或带 font/z 元数据）",
     "thresholds": {"tol": 4.0, "drift": 4.0, "scope": "designed"}},
    {"code": "spacing_anomaly", "desc": "同行/列连续间距离群（> k × 中位间距）；仅设计元素",
     "thresholds": {"k": 2.5, "scope": "designed"}},
    {"code": "safe_area", "desc": "元素贴边（≤ margin px）或出界",
     "thresholds": {"margin": 2.0}},
]


def is_designed(el: dict[str, Any]) -> bool:
    """设计意图判定：DOM/PPTX 来源或带 font/z 元数据 = 有明确布局意图；
    OCR/OmniParser 元素是自然文本的像素测量，边缘本就参差，不适用对齐/间距准则。"""
    src = el.get("source") or []
    if isinstance(src, str):
        src = [src]
    if any(s in ("dom", "pptx") for s in src):
        return True
    if el.get("font") or el.get("z") is not None:
        return True
    return False


def extract_elements(report: dict[str, Any]) -> list[dict[str, Any]]:
    """复用 vs_audit 的元素提取逻辑（elements[] 优先，PPTX shapes[] 兜底）。"""
    els = report.get("elements") or []
    if els:
        return els
    out = []
    for slide in report.get("slides", []):
        for sh in slide.get("shapes", []):
            if sh.get("pos_pt") is not None and sh.get("size_pt") is not None:
                x, y = sh["pos_pt"][0] or 0, sh["pos_pt"][1] or 0
                w, h = sh["size_pt"][0] or 0, sh["size_pt"][1] or 0
                texts = []
                fill = sh.get("fill")
                for t in sh.get("texts", []):
                    for run in t.get("runs", []):
                        texts.append({"text": run.get("text", ""), "color": run.get("color"),
                                      "size_pt": run.get("size_pt")})
                out.append({"bbox": [x, y, x + w, y + h], "fill": fill,
                            "texts": texts, "name": sh.get("name")})
    return out


def label(el: dict[str, Any], idx: int) -> str:
    return str(el.get("name") or (el.get("text") or el.get("type") or "") or f"el{idx}")[:60]


def canvas_of(report: dict[str, Any], args) -> tuple[int, int] | None:
    def to_int(v: Any) -> int | None:
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    if args.canvas:
        parts = args.canvas.split("x")
        if len(parts) == 2:
            w, h = to_int(parts[0]), to_int(parts[1])
            if w is not None and h is not None:
                return (w, h)
    src = report.get("source", {})
    for key in ("size_px", "viewport", "slide_size_pt"):
        v = src.get(key)
        if v and len(v) == 2:
            w, h = to_int(v[0]), to_int(v[1])
            if w is not None and h is not None:
                return (w, h)
    return None


def cluster(values: list[float], tol: float) -> list[list[float]]:
    """1D 密度聚类：按 tol 容差归并（确定性，输入有序）。"""
    groups: list[list[float]] = []
    for v in values:
        placed = False
        for g in groups:
            if abs(v - g[0]) <= tol:  # 与聚类首元素比较（保持确定性）
                g.append(v)
                placed = True
                break
        if not placed:
            groups.append([v])
    return groups


def check_contrast(els: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for idx, el in enumerate(els):
        texts = el.get("texts")
        pairs = []
        if texts:
            for t in texts:
                if t.get("color"):
                    pairs.append((t["color"], t.get("size_pt") or 11.0))
        else:
            color = el.get("color") or {}
            if color.get("text"):
                pairs.append((color["text"], 11.0))
        fill = el.get("fill") or (el.get("color") or {}).get("fill")
        if not pairs or not fill or fill in ("BACKGROUND (5)", "SOLID (1)"):
            continue
        for fg, size_pt in pairs:
            try:
                ratio = S.contrast_ratio(S.hex_to_rgb(fg), S.hex_to_rgb(fill))
            except (ValueError, TypeError):
                continue
            is_large = size_pt >= 18.0
            required = 3.0 if is_large else 4.5
            if ratio < required:
                out.append({
                    "rule": "text_contrast", "element_ids": [idx],
                    "severity": "critical" if ratio < 2.0 else "warn",
                    "bbox": el["bbox"],
                    "evidence": {"fg": fg, "bg": fill, "ratio": round(ratio, 2),
                                 "size_pt": size_pt, "required": required},
                    "suggested_cause": "文本色与背景色对比度不足（WCAG AA 未达标）",
                })
    return out


def check_overlap(els: list[dict[str, Any]], iou_th: float, min_area: float) -> list[dict[str, Any]]:
    out = []
    for i in range(len(els)):
        for j in range(i + 1, len(els)):
            box_a, box_b = els[i]["bbox"], els[j]["bbox"]
            inter = S.bbox_intersection(box_a, box_b)
            if inter is None:
                continue
            ia = S.bbox_area(inter)
            if ia < min_area:
                continue
            iou = S.bbox_iou(box_a, box_b)
            if iou > iou_th:
                out.append({
                    "rule": "overlap", "element_ids": [i, j],
                    "severity": "warn",
                    "bbox": inter,
                    "evidence": {"iou": round(iou, 2), "area": round(ia),
                                 "a": label(els[i], i), "b": label(els[j], j)},
                    "suggested_cause": "两个元素区域重叠（可能遮挡内容或布局冲突）",
                })
    return out


def check_alignment(els: list[dict[str, Any]], tol: float, drift_k: float) -> list[dict[str, Any]]:
    """对左缘/右缘/水平中线做聚类，标记落在 (tol, drift_k×tol] 的漂移元素。
    仅评估设计元素（OCR 自然文本边缘参差不是设计缺陷）。"""
    designed = [(idx, el) for idx, el in enumerate(els) if is_designed(el)]
    out = []
    for axis, key in (("left", lambda b: b[0]), ("right", lambda b: b[2]),
                      ("hcenter", lambda b: (b[0] + b[2]) / 2.0)):
        groups = cluster([key(el["bbox"]) for _, el in designed], tol)
        for g in groups:
            if len(g) < 2:
                continue
            mean = sum(g) / len(g)
            for idx, el in designed:
                v = key(el["bbox"])
                if v in g:
                    continue
                d = abs(v - mean)
                if tol < d <= drift_k * tol:
                    out.append({
                        "rule": "alignment_drift", "element_ids": [idx],
                        "severity": "info",
                        "bbox": el["bbox"],
                        "evidence": {"axis": axis, "value": round(v, 1),
                                     "cluster_mean": round(mean, 1),
                                     "offset": round(d, 1), "tol": tol},
                        "suggested_cause": f"{axis} 边缘偏离对齐簇 {round(d, 1)}px（组内有 {len(g)} 个元素对齐）",
                    })
    return out


def check_spacing(els: list[dict[str, Any]], k: float) -> list[dict[str, Any]]:
    """按行/列分组后检测连续间距离群（> k × 中位间距，且间距≥8px 才报告）。
    仅评估设计元素。"""
    out = []
    designed = [(idx, el) for idx, el in enumerate(els) if is_designed(el)]
    rows: list[list[tuple[int, dict[str, Any]]]] = []  # (原索引, 元素)，行内按 x 排序
    for idx, el in sorted(designed, key=lambda t: t[1]["bbox"][1]):
        y1, y2 = el["bbox"][1], el["bbox"][3]
        placed = False
        for row in rows:
            ry1, ry2 = row[0][1]["bbox"][1], row[0][1]["bbox"][3]
            if min(y2, ry2) - max(y1, ry1) > 0.5 * min(y2 - y1, ry2 - ry1):
                row.append((idx, el))
                placed = True
                break
        if not placed:
            rows.append([(idx, el)])
    for row in rows:
        if len(row) < 3:
            continue
        row_sorted = sorted(row, key=lambda t: t[1]["bbox"][0])
        gaps = [row_sorted[i + 1][1]["bbox"][0] - row_sorted[i][1]["bbox"][2]
                for i in range(len(row_sorted) - 1)]
        gaps = [g for g in gaps if g > 0]
        if not gaps:
            continue
        median = sorted(gaps)[len(gaps) // 2]
        if median <= 0:
            continue
        for i, g in enumerate(gaps):
            if g > k * median and g >= 8.0:
                out.append({
                    "rule": "spacing_anomaly",
                    "element_ids": [row_sorted[i][0], row_sorted[i + 1][0]],
                    "severity": "info",
                    "bbox": [row_sorted[i][1]["bbox"][2], row_sorted[i][1]["bbox"][1],
                             row_sorted[i + 1][1]["bbox"][0], row_sorted[i][1]["bbox"][3]],
                    "evidence": {"gap": round(g, 1), "median": round(median, 1),
                                 "k": k, "orientation": "horizontal"},
                    "suggested_cause": f"水平间距 {round(g, 1)}px 明显大于同行中位间距 {round(median, 1)}px",
                })
    return out


def check_safe_area(els: list[dict[str, Any]], canvas: tuple[int, int] | None, margin: float) -> list[dict[str, Any]]:
    out = []
    for idx, el in enumerate(els):
        x1, y1, x2, y2 = el["bbox"]
        if canvas:
            cw, ch = canvas
            clipped = x1 < 0 or y1 < 0 or x2 > cw or y2 > ch
            if clipped:
                out.append({
                    "rule": "safe_area", "element_ids": [idx], "severity": "critical",
                    "bbox": el["bbox"],
                    "evidence": {"canvas": [cw, ch], "kind": "off_canvas"},
                    "suggested_cause": "元素超出画布边界（可能被裁切或坐标错误）",
                })
                continue
            # 通栏/通高元素（占满整条边）= 有意的全出血设计，不告警
            full_bleed = (x1 <= 0 and x2 >= cw) or (y1 <= 0 and y2 >= ch)
            if full_bleed:
                continue
            edge = min(x1, y1, cw - x2, ch - y2)
            if edge <= margin and (el.get("text") or el.get("texts")):
                out.append({
                    "rule": "safe_area", "element_ids": [idx], "severity": "warn",
                    "bbox": el["bbox"],
                    "evidence": {"canvas": [cw, ch], "edge_margin": round(edge, 1),
                                 "kind": "edge_text"},
                    "suggested_cause": f"文本元素距画布边缘仅 {round(edge, 1)}px（存在裁切/触边风险）",
                })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--canvas")
    # 默认值经数据寻优（bench/tune_rules.py，13 边界样本 84.6%→100%，原验收集零回归）
    ap.add_argument("--align-tol", type=float, default=4.0)
    ap.add_argument("--align-drift", type=float, default=4.0)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--spacing-k", type=float, default=1.8)
    ap.add_argument("--overlap-threshold", type=float, default=0.01)
    ap.add_argument("--min-overlap-area", type=float, default=200.0)
    args = ap.parse_args()

    try:
        report = S.load_json(args.report)
        els = extract_elements(report)
        for idx, el in enumerate(els):
            el["_idx"] = idx
        canvas = canvas_of(report, args)

        findings: list[dict[str, Any]] = []
        findings += check_contrast(els)
        findings += check_overlap(els, args.overlap_threshold, args.min_overlap_area)
        findings += check_alignment(els, args.align_tol, args.align_drift)
        findings += check_spacing(els, args.spacing_k)
        findings += check_safe_area(els, canvas, args.margin)

        report_out = S.envelope(task="rules",
                                sensors=report.get("sensors") or ["rules"],
                                coordsys=report.get("coordsys"),
                                source={"type": "fused", "from": args.report})
        report_out["rules"] = RULES
        report_out["findings"] = findings
        sev = {"critical": 0, "warn": 0, "info": 0}
        for f in findings:
            sev[f["severity"]] = sev.get(f["severity"], 0) + 1
        report_out["metrics"] = {
            "element_count": len(els), "finding_count": len(findings),
            "severity": sev,
            "design_score": round(100.0 * (1.0 - (1.0 * sev["critical"] + 0.5 * sev["warn"]
                                                + 0.1 * sev["info"]) / max(1, len(els))), 1),
        }
        print(S.dump_json(report_out))
        return 0
    except Exception as e:
        print(json.dumps({"error": "vs_rules failed", "detail": str(e)[:500]}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
