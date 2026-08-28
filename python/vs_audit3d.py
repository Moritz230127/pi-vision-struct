#!/usr/bin/env python3
"""vs_audit3d.py — 三维装配审计（融合层，最大精度）。

输入：blender_dump 产生的 JSON（含 objects3d[].bbox3d 8角点 + .verts 世界坐标点云）
输出：findings[]（干涉/间隙，带真实世界距离 mm）+ metrics

精度递进（对每对 MESH 物体）：
  1. OBB-SAT 分离测试：用 8 个 bbox3d 角点直接构造有向包围盒（OBB），
     分离轴定理判定——旋转感知，比 AABB 紧致得多，天然消除因旋转产生的假干涉。
  2. 若 OBB 重叠 → 网格级 surface-to-surface 距离（cKDTree 点云最近邻）：
     取两物体世界坐标顶点点云，计算最小表面间距。
     - 同心/嵌套结构（转子在静子内、机匣包裹压气机）因存在环形间隙，
       此处得到**正间隙**而判为非干涉，正确保留装配间隙。
     - 真正穿透/接触 → 距离≈0 → 判为干涉。
  3. 无点云回退：用 AABB 穿透深度（保守）。

全部数值（世界坐标米），零主观描述。
"""
import argparse
import json
import sys

import numpy as np

try:
    from scipy.spatial import cKDTree
    HAVE_SCIPY = True
except Exception:  # pragma: no cover
    HAVE_SCIPY = False

import vs_schema as S

# 只检查这些类型（跳过 Camera/Empty/Light 等辅助对象；CURVE=管道/线缆须纳入点云距离）
MESH_TYPES = {"MESH", "CURVE", "SURFACE", "META", "FONT"}


# ------------------------------------------------------------------ OBB / SAT

def _corners(bbox3d: list) -> np.ndarray:
    """8 角点 → (8,3) 数组"""
    return np.asarray(bbox3d, dtype=float).reshape(8, 3)


def _obb_axes(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """从 bbox3d 8 角点还原 OBB：中心 + 三条局部轴（单位向量）+ 半长。

    Blender bound_box 顺序：index 0..7 对应 (x,y,z)∈{0,1}^3 的二进制编码，
    即角点 i 的符号位为 (i>>0 &1, i>>1 &1, i>>2 &1)。
    """
    c0 = corners[0]
    # 三条边方向：从角点0出发到角点1,2,4（对应 x,y,z 位翻转）
    ex = corners[1] - corners[0]
    ey = corners[2] - corners[0]
    ez = corners[4] - corners[0]
    center = (corners.max(axis=0) + corners.min(axis=0)) / 2.0
    axes = []
    half = []
    for e in (ex, ey, ez):
        length = float(np.linalg.norm(e))
        if length < 1e-12:
            axes.append(np.zeros(3))
            half.append(0.0)
        else:
            axes.append(e / length)
            half.append(length / 2.0)
    return center, np.array(axes), np.array(half)


def _sat_separation(center_a, axes_a, half_a, center_b, axes_b, half_b) -> float:
    """分离轴定理：返回两 OBB 的最小分离距离（米）。

    参数：center (3,) 轴中心；axes (3,3) 三条单位局部轴；half (3,) 半长。
    返回值：
      > 0  → 两 OBB 分离，此值为最小间隙（米）
      <= 0 → OBB 重叠（穿透深度取负，但此处我们只判重叠，不取穿透量）

    轴集：3 条 a 局部轴 + 3 条 b 局部轴 + 9 条叉积轴 = 15 轴。
    """
    # 15 条候选分离轴
    test_axes = []
    for ax in axes_a:
        test_axes.append(ax)
    for ax in axes_b:
        test_axes.append(ax)
    for ax in axes_a:
        for bx in axes_b:
            cr = np.cross(ax, bx)
            if np.linalg.norm(cr) > 1e-9:
                test_axes.append(cr / np.linalg.norm(cr))

    max_separation = -np.inf
    for axis in test_axes:
        axis = np.asarray(axis, dtype=float)
        n = np.linalg.norm(axis)
        if n < 1e-12:
            continue
        axis = axis / n
        # a 在轴上的投影半径
        ra = sum(abs(float(np.dot(ai, axis))) * hi for ai, hi in zip(axes_a, half_a))
        rb = sum(abs(float(np.dot(bi, axis))) * hi for bi, hi in zip(axes_b, half_b))
        # 中心距在该轴上的投影
        dist_on_axis = float(np.dot(center_a - center_b, axis))
        overlap = ra + rb - abs(dist_on_axis)
        if overlap < 0:
            # 此轴为分离轴，间隙 = overlap 的绝对值
            max_separation = max(max_separation, -overlap)
    # 若所有轴 overlap >= 0 → OBB 重叠，返回 0（表示无分离轴）
    if max_separation < 0:
        return 0.0  # 重叠
    return max_separation


# ------------------------------------------------------------------ 点云距离

def _surface_distance(va: np.ndarray, vb: np.ndarray) -> float:
    """两世界坐标点云的最小表面间距（米）。用 cKDTree 双向最近邻取较小值。"""
    if va.shape[0] == 0 or vb.shape[0] == 0:
        return float("nan")
    ta = cKDTree(va)
    tb = cKDTree(vb)
    d_ab, _ = tb.query(va, k=1)
    d_ba, _ = ta.query(vb, k=1)
    return float(min(d_ab.min(), d_ba.min()))


# ------------------------------------------------------------------ 主流程

def bbox3d_distance(a, b) -> float:
    """两个 bbox3d 的 AABB 最小分离距离（相交时为负——穿透深度），用于回退。"""
    gap = 0.0
    intersecting = True
    for i in range(3):
        a_min = min(p[i] for p in a)
        a_max = max(p[i] for p in a)
        b_min = min(p[i] for p in b)
        b_max = max(p[i] for p in b)
        if a_max < b_min:
            gap = max(gap, b_min - a_max)
            intersecting = False
        elif b_max < a_min:
            gap = max(gap, a_min - b_max)
            intersecting = False
    if intersecting:
        penetration = min(
            min(max(p[i] for p in a), max(p[i] for p in b)) -
            max(min(p[i] for p in a), min(p[i] for p in b))
            for i in range(3)
        )
        return -penetration
    return gap


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--gap-threshold", type=float, default=15.0,
                    help="最小间隙阈值（mm），小于此值告警")
    ap.add_argument("--method", choices=["auto", "obb", "mesh", "aabb"],
                    default="auto",
                    help="精度档：auto=OBB+Mesh 级最大精度；obb=仅OBB-SAT；"
                         "mesh=仅点云距离；aabb=原AABB回退")
    args = ap.parse_args()

    try:
        data = json.loads(open(args.report, encoding="utf-8").read())
        objs = data.get("objects3d") or []

        mesh_objs = [o for o in objs if o.get("type") in MESH_TYPES]
        threshold_m = args.gap_threshold / 1000.0

        findings = []
        interference_count = 0
        gap_warnings = 0
        verified_clearance = 0  # 经网格级确认存在真实正间隙（嵌套结构非干涉）

        method_used = set()

        for i in range(len(mesh_objs)):
            for j in range(i + 1, len(mesh_objs)):
                a, b = mesh_objs[i], mesh_objs[j]
                if not a.get("bbox3d") or not b.get("bbox3d"):
                    continue

                # 预筛：AABB 若已分离则直接跳过（加速）
                aabb_d = bbox3d_distance(a["bbox3d"], b["bbox3d"])
                if aabb_d > threshold_m:
                    # AABB 已明显分离，无需精细判定
                    method_used.add("aabb-skip")
                    continue

                ca, axes_a, half_a = _obb_axes(_corners(a["bbox3d"]))
                cb, axes_b, half_b = _obb_axes(_corners(b["bbox3d"]))

                sep = None
                if args.method in ("auto", "obb"):
                    sep = _sat_separation(ca, axes_a, half_a, cb, axes_b, half_b)
                    method_used.add("obb-sat")

                if sep is not None and sep > threshold_m:
                    # OBB 明确分离
                    findings.append({
                        "rule": "clearance",
                        "element_ids": [i, j],
                        "severity": "info",
                        "distance_mm": round(sep * 1000, 4),
                        "method": "obb-sat",
                        "evidence": {"a": a["name"], "b": b["name"],
                                     "collection_a": a.get("collection"),
                                     "collection_b": b.get("collection"),
                                     "clearance_mm": round(sep * 1000, 4)},
                    })
                    verified_clearance += 1
                    continue

                # OBB 重叠或仅 mesh 档 → 网格级点云距离（最大精度）
                va = np.asarray(a["verts"]) if a.get("verts") else None
                vb = np.asarray(b["verts"]) if b.get("verts") else None
                if args.method in ("auto", "mesh") and va is not None and vb is not None and HAVE_SCIPY:
                    dist = _surface_distance(va, vb)
                    method_used.add("mesh-kdtree")
                    if dist != dist:  # nan
                        continue
                    if dist <= 1e-6:
                        interference_count += 1
                        findings.append({
                            "rule": "interference",
                            "element_ids": [i, j],
                            "severity": "critical",
                            "distance_mm": 0.0,
                            "method": "mesh-kdtree",
                            "evidence": {"a": a["name"], "b": b["name"],
                                         "collection_a": a.get("collection"),
                                         "collection_b": b.get("collection"),
                                         "penetration_mm": 0.0},
                        })
                    elif dist < threshold_m:
                        gap_warnings += 1
                        findings.append({
                            "rule": "tight_gap",
                            "element_ids": [i, j],
                            "severity": "warn",
                            "distance_mm": round(dist * 1000, 4),
                            "method": "mesh-kdtree",
                            "evidence": {"a": a["name"], "b": b["name"],
                                         "collection_a": a.get("collection"),
                                         "collection_b": b.get("collection"),
                                         "gap_mm": round(dist * 1000, 4)},
                        })
                    else:
                        verified_clearance += 1
                        findings.append({
                            "rule": "clearance",
                            "element_ids": [i, j],
                            "severity": "info",
                            "distance_mm": round(dist * 1000, 4),
                            "method": "mesh-kdtree",
                            "evidence": {"a": a["name"], "b": b["name"],
                                         "collection_a": a.get("collection"),
                                         "collection_b": b.get("collection"),
                                         "clearance_mm": round(dist * 1000, 4)},
                        })
                else:
                    # 回退 AABB
                    method_used.add("aabb-fallback")
                    if aabb_d < 0:
                        interference_count += 1
                        findings.append({
                            "rule": "interference",
                            "element_ids": [i, j],
                            "severity": "critical",
                            "distance_mm": round(aabb_d * 1000, 4),
                            "method": "aabb-fallback",
                            "evidence": {"a": a["name"], "b": b["name"],
                                         "collection_a": a.get("collection"),
                                         "collection_b": b.get("collection"),
                                         "penetration_mm": round(-aabb_d * 1000, 4)},
                        })
                    elif 0 <= aabb_d < threshold_m:
                        gap_warnings += 1
                        findings.append({
                            "rule": "tight_gap",
                            "element_ids": [i, j],
                            "severity": "warn",
                            "distance_mm": round(aabb_d * 1000, 4),
                            "method": "aabb-fallback",
                            "evidence": {"a": a["name"], "b": b["name"],
                                         "collection_a": a.get("collection"),
                                         "collection_b": b.get("collection"),
                                         "gap_mm": round(aabb_d * 1000, 4)},
                        })

        metrics = {
            "total_objects": len(objs),
            "mesh_objects": len(mesh_objs),
            "pair_checks": len(mesh_objs) * (len(mesh_objs) - 1) // 2,
            "interference_count": interference_count,
            "gap_warnings": gap_warnings,
            "verified_clearance": verified_clearance,
            "gap_threshold_mm": args.gap_threshold,
            "methods": sorted(method_used),
            "scipy_available": HAVE_SCIPY,
        }

        report = S.envelope(task="audit3d", sensors=["bpy", "audit3d"],
                            coordsys="world_m",
                            source={"type": "blender", "from": args.report})
        report["findings"] = findings
        report["metrics"] = metrics

        print(S.dump_json(report))
        return 0
    except Exception as e:
        import traceback
        print(json.dumps({"error": "vs_audit3d failed", "detail": str(e)[:500],
                          "traceback": traceback.format_exc()[:500]}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
