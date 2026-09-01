#!/usr/bin/env python3
"""vs_fusion.py — V3 融合引擎：多传感器证据融合（schema v3）。

三阶段流水线：
  1. align    : 匈牙利算法全局最优匹配（替代朴素 max-IoU）
  2. evidence : 传感器输出 → mass 函数（Dempster-Shafer 证据质量）
  3. combine  : D-S 正交和组合 → belief/plausibility → verdict

输出 findings[]（带 belief/plausibility/uncertainty/evidence/verdict），
DeepSeek 据此决定：直接采信 / 请求 zoom 复查 / 标记存疑。

纯算法，零模型，确定性可复算。依赖：scipy（匈牙利匹配）。
"""
import argparse
import json
import math
import sys
from typing import Any, Sequence

import vs_schema as S

# ---------------------------------------------------------------- 辨识框架

# Θ = {consistent, conflict, uncertain}
CONSISTENT = "consistent"
CONFLICT = "conflict"
UNCERTAIN = "uncertain"
FRAME = (CONSISTENT, CONFLICT, UNCERTAIN)

# 决策阈值
BELIEF_CONFIRM = 0.60   # belief(consistent) > 0.6 → confirmed
PLAUSIBILITY_CONFLICT = 0.40  # plausibility(consistent) < 0.4 → conflict
K_MAX = 0.70            # 冲突系数上限：K>0.7 → needs_review（防误报）

# mass 映射参数
TAU_COLOR = 5.0         # ΔE → 似然 的衰减常数
OCR_CONF_SCALE = 0.8    # OCR conf → consistent 的缩放
SEG_AREA_SCALE = 2.0    # 前景面积比 → consistent 的缩放


# ---------------------------------------------------------------- mass 映射

def mass_from_ocr(conf: float) -> dict[str, float]:
    """OCR 文本存在性：conf 高 → consistent 证据强。"""
    c = min(1.0, max(0.0, conf)) * OCR_CONF_SCALE
    return {CONSISTENT: c, CONFLICT: 0.0, UNCERTAIN: 1.0 - c}


def mass_from_color_delta(delta_e: float) -> dict[str, float]:
    """DOM 声明色 vs 像素实测：ΔE 小 → consistent 强。"""
    c = math.exp(-delta_e / TAU_COLOR)
    return {CONSISTENT: c, CONFLICT: 1.0 - c, UNCERTAIN: 0.0}


def mass_from_detect(conf: float) -> dict[str, float]:
    """检测器置信度：conf 即 consistent。"""
    c = min(1.0, max(0.0, conf))
    return {CONSISTENT: c, CONFLICT: 0.0, UNCERTAIN: 1.0 - c}


def mass_from_segment(area_ratio: float) -> dict[str, float]:
    """前景面积比：大 → 前景明确。"""
    c = min(1.0, max(0.0, area_ratio) * SEG_AREA_SCALE)
    return {CONSISTENT: c, CONFLICT: 0.0, UNCERTAIN: 1.0 - c}


def mass_from_saliency(score: float) -> dict[str, float]:
    """显著性分数：高 → 候选区有效。"""
    c = min(1.0, max(0.0, score))
    return {CONSISTENT: c, CONFLICT: 0.0, UNCERTAIN: 1.0 - c}


def mass_from_depth(depth_delta: float, sigma: float = 0.1) -> dict[str, float]:
    """深度一致性：|Δdepth| 小 → 一致。"""
    c = 1.0 - min(1.0, abs(depth_delta) / sigma)
    return {CONSISTENT: c, CONFLICT: 1.0 - c, UNCERTAIN: 0.0}


def mass_from_rule(rule_conf: float) -> dict[str, float]:
    """规则引擎置信度（audit/rules 输出）。"""
    c = min(1.0, max(0.0, rule_conf))
    return {CONSISTENT: c, CONFLICT: 0.0, UNCERTAIN: 1.0 - c}


# 传感器 → mass 工厂注册表
MASS_FACTORIES = {
    "ocr": mass_from_ocr,
    "color_delta": mass_from_color_delta,
    "detect": mass_from_detect,
    "segment": mass_from_segment,
    "saliency": mass_from_saliency,
    "depth": mass_from_depth,
    "rule": mass_from_rule,
}


# ---------------------------------------------------------------- D-S 组合

def ds_combine(m1: dict[str, float], m2: dict[str, float]) -> dict[str, float]:
    """Dempster 正交和：m₁₂(A) = Σ_{B∩C=A} m₁(B)m₂(C) / (1-K)。

    K = Σ_{B∩C=∅} m₁(B)m₂(C)（冲突系数）。
    三元素框架 {consistent, conflict, uncertain} 的闭式解。
    """
    # 冲突系数 K：consistent∩conflict = ∅ 等
    k = (m1[CONSISTENT] * m2[CONFLICT] + m1[CONFLICT] * m2[CONSISTENT])
    denom = 1.0 - k
    if denom <= 1e-12:
        # 完全冲突 → 全不确定（防除零）
        return {CONSISTENT: 0.0, CONFLICT: 0.0, UNCERTAIN: 1.0}
    # 组合后各子集质量
    m_cons = (m1[CONSISTENT] * m2[CONSISTENT] +
              m1[CONSISTENT] * m2[UNCERTAIN] +
              m1[UNCERTAIN] * m2[CONSISTENT]) / denom
    m_conf = (m1[CONFLICT] * m2[CONFLICT] +
              m1[CONFLICT] * m2[UNCERTAIN] +
              m1[UNCERTAIN] * m2[CONFLICT]) / denom
    m_unc = (m1[UNCERTAIN] * m2[UNCERTAIN]) / denom
    return {CONSISTENT: m_cons, CONFLICT: m_conf, UNCERTAIN: m_unc}


def ds_combine_all(masses: list[dict[str, float]]) -> dict[str, float]:
    """顺序组合多个 mass 函数。空列表 → 全不确定。"""
    if not masses:
        return {CONSISTENT: 0.0, CONFLICT: 0.0, UNCERTAIN: 1.0}
    acc = masses[0]
    for m in masses[1:]:
        acc = ds_combine(acc, m)
    return acc


def belief_plausibility(m: dict[str, float]) -> tuple[float, float]:
    """belief(A) = m(A)；plausibility(A) = 1 - belief(¬A) = m(A) + m(uncertain)。"""
    bel = m[CONSISTENT]
    pla = m[CONSISTENT] + m[UNCERTAIN]
    return bel, pla


def decide(bel: float, pla: float, k: float) -> str:
    """决策规则：confirmed / conflict / needs_review。"""
    if k > K_MAX:
        return "needs_review"  # 证据高度冲突 → 需复查而非误报
    if bel > BELIEF_CONFIRM:
        return "confirmed"
    if pla < PLAUSIBILITY_CONFLICT:
        return "conflict"
    return "needs_review"


# ---------------------------------------------------------------- 匈牙利匹配

def hungarian_match(boxes_a: list[list[float]], boxes_b: list[list[float]],
                    texts_a: list[str | None] | None = None,
                    texts_b: list[str | None] | None = None,
                    alpha: float = 0.3) -> list[tuple[int, int, float]]:
    """全局最优一对一匹配（匈牙利算法）。

    代价 C[i][j] = 1 - IoU(box_i, box_j) - α·text_sim(i, j)
    返回 [(i, j, iou), ...]（仅 IoU > 0 的匹配对）。
    """
    from scipy.optimize import linear_sum_assignment

    n, m = len(boxes_a), len(boxes_b)
    if n == 0 or m == 0:
        return []
    cost = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            iou = S.bbox_iou(boxes_a[i], boxes_b[j])
            sim = 0.0
            if texts_a and texts_b and texts_a[i] and texts_b[j]:
                sim = _text_sim(texts_a[i], texts_b[j])
            cost[i][j] = 1.0 - iou - alpha * sim
    row_idx, col_idx = linear_sum_assignment(cost)
    pairs = []
    for i, j in zip(row_idx, col_idx):
        iou = S.bbox_iou(boxes_a[i], boxes_b[j])
        if iou > 0.0:
            pairs.append((int(i), int(j), iou))
    return pairs


def _text_sim(a: str, b: str) -> float:
    """文本相似度：精确匹配 1.0；包含 0.7；否则编辑距离归一化。"""
    a, b = a.strip().lower(), b.strip().lower()
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.7
    # 编辑距离（Levenshtein）归一化
    d = _levenshtein(a, b)
    return max(0.0, 1.0 - d / max(len(a), len(b)))


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# ---------------------------------------------------------------- 融合主流程

def fuse_elements(elements: list[dict[str, Any]],
                  match_groups: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] | None = None,
                  ) -> list[dict[str, Any]]:
    """对元素列表执行融合，输出 findings。

    match_groups: [(group_a_elements, group_b_elements), ...] 预定义匹配组
    （如 DOM↔OCR、DOM↔像素）。None 时按传感器自动分组。
    """
    findings: list[dict[str, Any]] = []
    if match_groups is None:
        match_groups = _auto_groups(elements)
    for group_a, group_b in match_groups:
        boxes_a = [el["bbox"] for el in group_a]
        boxes_b = [el["bbox"] for el in group_b]
        texts_a = [el.get("text") for el in group_a]
        texts_b = [el.get("text") for el in group_b]
        pairs = hungarian_match(boxes_a, boxes_b, texts_a, texts_b)
        matched_b = {j for _, j, _ in pairs}
        # 已匹配对 → 一致性证据
        for i, j, iou in pairs:
            masses = _element_masses(group_a[i], group_b[j], iou)
            m = ds_combine_all(masses)
            bel, pla = belief_plausibility(m)
            k = _conflict_coef(masses)
            findings.append(_finding(
                "match_consistent", group_a[i], group_b[j], m, bel, pla, k,
                {"iou": round(iou, 3), "sensors": [group_a[i].get("source"), group_b[j].get("source")]}))
        # 未匹配 → 缺失/多余证据
        for i, el in enumerate(group_a):
            if i not in {p[0] for p in pairs}:
                findings.append(_finding(
                    "unmatched_a", el, None, None, 0.0, 0.0, 0.0,
                    {"reason": "no match in group_b", "sensor": el.get("source")}))
        for j, el in enumerate(group_b):
            if j not in matched_b:
                findings.append(_finding(
                    "unmatched_b", None, el, None, 0.0, 0.0, 0.0,
                    {"reason": "no match in group_a", "sensor": el.get("source")}))
    return findings


def _auto_groups(elements: list[dict[str, Any]]) -> list[tuple[list[dict], list[dict]]]:
    """按 source 自动分组：同 source 元素归一组，两两组合。"""
    groups: dict[str, list[dict]] = {}
    for el in elements:
        src = ",".join(el.get("source") or ["unknown"])
        groups.setdefault(src, []).append(el)
    keys = list(groups.keys())
    out = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            out.append((groups[keys[i]], groups[keys[j]]))
    return out


def _element_masses(el_a: dict[str, Any], el_b: dict[str, Any], iou: float) -> list[dict[str, float]]:
    """从两个元素提取 mass 证据。"""
    masses: list[dict[str, float]] = []
    # IoU 本身是证据：高 IoU → 一致
    masses.append({CONSISTENT: min(1.0, iou * 2.0), CONFLICT: 0.0, UNCERTAIN: 1.0 - min(1.0, iou * 2.0)})
    # 各自 conf
    for el in (el_a, el_b):
        conf = el.get("conf")
        if conf is not None:
            masses.append(mass_from_ocr(conf))
    # 颜色一致性（若都有 color）
    ca = (el_a.get("color") or {}).get("fill") or (el_a.get("color") or {}).get("text")
    cb = (el_b.get("color") or {}).get("fill") or (el_b.get("color") or {}).get("text")
    if ca and cb:
        try:
            de = S.delta_e76(S.hex_to_rgb(ca), S.hex_to_rgb(cb))
            masses.append(mass_from_color_delta(de))
        except ValueError:
            pass
    return masses


def _conflict_coef(masses: list[dict[str, float]]) -> float:
    """组合过程中的最大冲突系数（近似：两两组合的 K 最大值）。"""
    k_max = 0.0
    for i in range(len(masses)):
        for j in range(i + 1, len(masses)):
            k = (masses[i][CONSISTENT] * masses[j][CONFLICT] +
                 masses[i][CONFLICT] * masses[j][CONSISTENT])
            k_max = max(k_max, k)
    return k_max


def _finding(ftype: str, el_a: dict | None, el_b: dict | None,
             m: dict[str, float] | None, bel: float, pla: float, k: float,
             evidence: dict[str, Any]) -> dict[str, Any]:
    """构造 finding（schema v3）。"""
    bbox = (el_a or el_b or {}).get("bbox")
    f: dict[str, Any] = {
        "type": ftype,
        "bbox": list(bbox) if bbox else None,
        "belief": round(bel, 3),
        "plausibility": round(pla, 3),
        "uncertainty": round(1.0 - (pla - bel), 3),
        "verdict": decide(bel, pla, k) if m else "needs_review",
        "evidence": evidence,
    }
    if el_a is not None:
        f["a_id"] = el_a.get("id")
    if el_b is not None:
        f["b_id"] = el_b.get("id")
    return f


# ---------------------------------------------------------------- CLI

def main() -> int:
    ap = argparse.ArgumentParser(description="V3 融合引擎：多传感器证据融合")
    ap.add_argument("--reports", nargs="+", required=True,
                    help="传感器报告 JSON 路径（schema v3 元素）")
    ap.add_argument("--groups", help="匹配组定义 JSON：[[[a_ids],[b_ids]], ...]")
    args = ap.parse_args()

    try:
        all_elements: list[dict[str, Any]] = []
        sensors: list[str] = []
        for path in args.reports:
            rep = S.load_json(path)
            all_elements.extend(rep.get("elements") or [])
            sensors.extend(rep.get("sensors") or [])

        match_groups = None
        if args.groups:
            gdef = json.loads(args.groups)
            by_id = {el.get("id"): el for el in all_elements}
            match_groups = []
            for ga, gb in gdef:
                match_groups.append(([by_id[i] for i in ga if i in by_id],
                                     [by_id[i] for i in gb if i in by_id]))

        findings = fuse_elements(all_elements, match_groups)

        report = S.envelope(task="fusion", sensors=list(dict.fromkeys(sensors)),
                            coordsys="image_px",
                            source={"type": "fused", "reports": args.reports})
        report["schema"] = "vision-report/v3"
        report["findings"] = findings
        report["notation"] = S.NOTATION_GUIDE
        report["metrics"] = {
            "element_count": len(all_elements),
            "finding_count": len(findings),
            "confirmed": sum(1 for f in findings if f["verdict"] == "confirmed"),
            "conflict": sum(1 for f in findings if f["verdict"] == "conflict"),
            "needs_review": sum(1 for f in findings if f["verdict"] == "needs_review"),
        }
        print(S.dump_json(report))
        return 0
    except Exception as e:
        print(json.dumps({"error": "vs_fusion failed", "detail": str(e)[:500]},
                         ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
