#!/usr/bin/env python3
"""vs_audit3d.py — 三维装配审计（融合层，纯矩阵运算）。

输入：blender_dump 产生的 JSON
输出：findings[]（间隙值mm、干涉判定）+ metrics

只检查 MESH 类型的可见物体（过滤 Camera/Empty/Light 等辅助对象）。
AABB 交集测试 + 距离计算，全部数值运算。
"""
import argparse
import json
import math
import sys

import numpy as np

import vs_schema as S

# 只检查这些类型（跳过 Camera/Empty/Light 等辅助对象）
MESH_TYPES = {"MESH", "CURVE", "SURFACE", "META", "FONT"}


def bbox3d_distance(a, b) -> float:
    """两个 bbox3d 的最小分离距离（相交时为负——穿透深度）"""
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
    args = ap.parse_args()

    try:
        data = json.loads(open(args.report, encoding="utf-8").read())
        objs = data.get("objects3d") or []

        # 只保留 MESH 类型物体（过滤 Camera/Empty/Light）
        mesh_objs = [o for o in objs if o.get("type") in MESH_TYPES]
        threshold_m = args.gap_threshold / 1000.0

        findings = []
        interference_count = 0
        gap_warnings = 0

        for i in range(len(mesh_objs)):
            for j in range(i + 1, len(mesh_objs)):
                a, b = mesh_objs[i], mesh_objs[j]
                if not a.get("bbox3d") or not b.get("bbox3d"):
                    continue

                dist = bbox3d_distance(a["bbox3d"], b["bbox3d"])

                if dist < 0:
                    interference_count += 1
                    findings.append({
                        "rule": "interference",
                        "element_ids": [i, j],
                        "severity": "critical",
                        "distance_mm": round(dist * 1000, 4),
                        "evidence": {"a": a["name"], "b": b["name"],
                                     "penetration_mm": round(-dist * 1000, 4)},
                    })
                elif 0 <= dist < threshold_m:
                    gap_warnings += 1
                    findings.append({
                        "rule": "tight_gap",
                        "element_ids": [i, j],
                        "severity": "warn",
                        "distance_mm": round(dist * 1000, 4),
                        "evidence": {"a": a["name"], "b": b["name"],
                                     "gap_mm": round(dist * 1000, 4)},
                    })

        metrics = {
            "total_objects": len(objs),
            "mesh_objects": len(mesh_objs),
            "pair_checks": len(mesh_objs) * (len(mesh_objs) - 1) // 2,
            "interference_count": interference_count,
            "gap_warnings": gap_warnings,
            "gap_threshold_mm": args.gap_threshold,
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
