//! 三维装配审计（对应 Python `vs_audit3d.py`，纯 Rust，schema v3 输出）。
//!
//! 输入：blender_dump 产生的 JSON（objects3d[].bbox3d 8角点 + .verts 世界坐标点云）。
//! 输出：findings[]（干涉/间隙，带真实世界距离 mm）+ metrics。
//!
//! 精度递进（对每对 MESH 物体）：
//!   1. OBB-SAT 分离测试：8 角点构造有向包围盒（OBB），分离轴定理判定。
//!   2. OBB 重叠 → 网格级 surface-to-surface 距离（kiddo kd-tree 点云最近邻，
//!      替代 Python scipy cKDTree）。
//!   3. 无点云回退：AABB 穿透深度（保守）。

use serde_json::{json, Value};

/// 只检查这些类型（与 Python MESH_TYPES 一致）。
const MESH_TYPES: &[&str] = &["MESH", "CURVE", "SURFACE", "META", "FONT"];

/// audit3d 参数（对应 Python `--report/--gap-threshold/--method`）。
pub struct Audit3dParams {
    /// blender_dump 报告 JSON 路径。
    pub report: String,
    /// 最小间隙阈值（mm），小于此值告警。
    pub gap_threshold: f64,
    /// 精度档：auto | obb | mesh | aabb。
    pub method: String,
}

impl Default for Audit3dParams {
    fn default() -> Self {
        Self {
            report: String::new(),
            gap_threshold: 15.0,
            method: "auto".into(),
        }
    }
}

/// 8 角点 → (8,3) 数组。
fn corners(bbox3d: &Value) -> Vec<[f64; 3]> {
    bbox3d
        .as_array()
        .map(|arr| {
            arr.iter()
                .map(|c| {
                    let v = c.as_array().unwrap();
                    [
                        v[0].as_f64().unwrap_or(0.0),
                        v[1].as_f64().unwrap_or(0.0),
                        v[2].as_f64().unwrap_or(0.0),
                    ]
                })
                .collect()
        })
        .unwrap_or_default()
}

/// 从 bbox3d 8 角点还原 OBB：中心 + 三条局部轴（单位向量）+ 半长。
///
/// Blender bound_box 顺序：角点 0 出发到角点 1,2,4 为 x,y,z 方向。
fn obb_axes(c: &[[f64; 3]]) -> ([f64; 3], [[f64; 3]; 3], [f64; 3]) {
    let c0 = c[0];
    let ex = sub(c[1], c0);
    let ey = sub(c[2], c0);
    let ez = sub(c[4], c0);
    let mut center = [0.0; 3];
    let mut mn = [f64::MAX; 3];
    let mut mx = [f64::MIN; 3];
    for p in c {
        for k in 0..3 {
            mn[k] = mn[k].min(p[k]);
            mx[k] = mx[k].max(p[k]);
        }
    }
    for k in 0..3 {
        center[k] = (mn[k] + mx[k]) / 2.0;
    }
    let mut axes = [[0.0; 3]; 3];
    let mut half = [0.0; 3];
    for (i, e) in [ex, ey, ez].iter().enumerate() {
        let len = norm(*e);
        if len < 1e-12 {
            axes[i] = [0.0; 3];
            half[i] = 0.0;
        } else {
            axes[i] = scale(*e, 1.0 / len);
            half[i] = len / 2.0;
        }
    }
    (center, axes, half)
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// 分离轴定理：返回两 OBB 的最小分离距离（米）。
///
/// > 0 → 分离（最小间隙）；<= 0 → 重叠。
fn sat_separation(
    center_a: [f64; 3],
    axes_a: [[f64; 3]; 3],
    half_a: [f64; 3],
    center_b: [f64; 3],
    axes_b: [[f64; 3]; 3],
    half_b: [f64; 3],
) -> f64 {
    // 15 条候选分离轴：3 a 轴 + 3 b 轴 + 9 叉积轴
    let mut test_axes: Vec<[f64; 3]> = Vec::with_capacity(15);
    for ax in axes_a {
        test_axes.push(ax);
    }
    for bx in axes_b {
        test_axes.push(bx);
    }
    for ax in axes_a {
        for bx in axes_b {
            let cr = cross(ax, bx);
            let n = norm(cr);
            if n > 1e-9 {
                test_axes.push(scale(cr, 1.0 / n));
            }
        }
    }

    let mut max_separation = f64::NEG_INFINITY;
    let d = sub(center_a, center_b);
    for axis in test_axes {
        let n = norm(axis);
        if n < 1e-12 {
            continue;
        }
        let axis = scale(axis, 1.0 / n);
        let mut ra = 0.0;
        for (ai, hi) in axes_a.iter().zip(half_a.iter()) {
            ra += dot(*ai, axis).abs() * hi;
        }
        let mut rb = 0.0;
        for (bi, hi) in axes_b.iter().zip(half_b.iter()) {
            rb += dot(*bi, axis).abs() * hi;
        }
        let dist_on_axis = dot(d, axis);
        let overlap = ra + rb - dist_on_axis.abs();
        if overlap < 0.0 {
            max_separation = max_separation.max(-overlap);
        }
    }
    if max_separation < 0.0 {
        return 0.0; // 重叠
    }
    max_separation
}

/// 两世界坐标点云的最小表面间距（米）。kiddo kd-tree 双向最近邻取较小值。
fn surface_distance(va: &[[f64; 3]], vb: &[[f64; 3]]) -> Option<f64> {
    use kiddo::{ImmutableKdTree, SquaredEuclidean};
    if va.is_empty() || vb.is_empty() {
        return None;
    }
    let ta = ImmutableKdTree::new_from_slice(va).ok()?;
    let tb = ImmutableKdTree::new_from_slice(vb).ok()?;
    let mut min_ab = f64::MAX;
    for p in va {
        let r = tb.query(p).nearest_one::<SquaredEuclidean<f64>>().execute();
        min_ab = min_ab.min(r.distance.sqrt());
    }
    let mut min_ba = f64::MAX;
    for p in vb {
        let r = ta.query(p).nearest_one::<SquaredEuclidean<f64>>().execute();
        min_ba = min_ba.min(r.distance.sqrt());
    }
    Some(min_ab.min(min_ba))
}

/// 两个 bbox3d 的 AABB 最小分离距离（相交时为负——穿透深度）。
fn bbox3d_distance(a: &Value, b: &Value) -> f64 {
    let ca = corners(a);
    let cb = corners(b);
    let mut gap = 0.0f64;
    let mut intersecting = true;
    for i in 0..3 {
        let a_min = ca.iter().map(|p| p[i]).fold(f64::MAX, f64::min);
        let a_max = ca.iter().map(|p| p[i]).fold(f64::MIN, f64::max);
        let b_min = cb.iter().map(|p| p[i]).fold(f64::MAX, f64::min);
        let b_max = cb.iter().map(|p| p[i]).fold(f64::MIN, f64::max);
        if a_max < b_min {
            gap = gap.max(b_min - a_max);
            intersecting = false;
        } else if b_max < a_min {
            gap = gap.max(a_min - b_max);
            intersecting = false;
        }
    }
    if intersecting {
        let mut penetration = f64::MAX;
        for i in 0..3 {
            let a_max = ca.iter().map(|p| p[i]).fold(f64::MIN, f64::max);
            let a_min = ca.iter().map(|p| p[i]).fold(f64::MAX, f64::min);
            let b_max = cb.iter().map(|p| p[i]).fold(f64::MIN, f64::max);
            let b_min = cb.iter().map(|p| p[i]).fold(f64::MAX, f64::min);
            let p = a_max.min(b_max) - a_min.max(b_min);
            penetration = penetration.min(p);
        }
        return -penetration;
    }
    gap
}

/// 分析一对物体 → findings 列表。
fn analyze_pair(
    a: &Value,
    b: &Value,
    idx_a: usize,
    idx_b: usize,
    threshold_m: f64,
    method: &str,
) -> Vec<Value> {
    let mut findings = Vec::new();
    if a.get("bbox3d").is_none() || b.get("bbox3d").is_none() {
        return findings;
    }
    let name_a = a["name"].as_str().unwrap_or("").to_string();
    let name_b = b["name"].as_str().unwrap_or("").to_string();
    let coll_a = a.get("collection").and_then(|v| v.as_str()).unwrap_or("");
    let coll_b = b.get("collection").and_then(|v| v.as_str()).unwrap_or("");

    // 预筛：AABB 已分离则跳过
    let aabb_d = bbox3d_distance(&a["bbox3d"], &b["bbox3d"]);
    if aabb_d > threshold_m {
        return findings;
    }

    let ca = corners(&a["bbox3d"]);
    let cb = corners(&b["bbox3d"]);
    let (center_a, axes_a, half_a) = obb_axes(&ca);
    let (center_b, axes_b, half_b) = obb_axes(&cb);

    let mut sep: Option<f64> = None;
    if method == "auto" || method == "obb" {
        sep = Some(sat_separation(
            center_a, axes_a, half_a, center_b, axes_b, half_b,
        ));
    }

    if let Some(s) = sep {
        if s > threshold_m {
            findings.push(json!({
                "rule": "clearance",
                "element_ids": [idx_a, idx_b],
                "severity": "info",
                "distance_mm": round4(s * 1000.0),
                "method": "obb-sat",
                "evidence": {
                    "a": name_a, "b": name_b,
                    "collection_a": coll_a, "collection_b": coll_b,
                    "clearance_mm": round4(s * 1000.0),
                },
            }));
            return findings;
        }
    }

    // OBB 重叠或仅 mesh 档 → 网格级点云距离
    let va: Option<Vec<[f64; 3]>> = a.get("verts").and_then(|v| {
        v.as_array().map(|arr| {
            arr.iter()
                .map(|p| {
                    let v = p.as_array().unwrap();
                    [
                        v[0].as_f64().unwrap_or(0.0),
                        v[1].as_f64().unwrap_or(0.0),
                        v[2].as_f64().unwrap_or(0.0),
                    ]
                })
                .collect()
        })
    });
    let vb: Option<Vec<[f64; 3]>> = b.get("verts").and_then(|v| {
        v.as_array().map(|arr| {
            arr.iter()
                .map(|p| {
                    let v = p.as_array().unwrap();
                    [
                        v[0].as_f64().unwrap_or(0.0),
                        v[1].as_f64().unwrap_or(0.0),
                        v[2].as_f64().unwrap_or(0.0),
                    ]
                })
                .collect()
        })
    });

    if (method == "auto" || method == "mesh") && va.is_some() && vb.is_some() {
        if let (Some(va), Some(vb)) = (va, vb) {
            if let Some(dist) = surface_distance(&va, &vb) {
                if dist <= 1e-6 {
                    findings.push(json!({
                        "rule": "interference",
                        "element_ids": [idx_a, idx_b],
                        "severity": "critical",
                        "distance_mm": 0.0,
                        "method": "mesh-kdtree",
                        "evidence": {
                            "a": name_a, "b": name_b,
                            "collection_a": coll_a, "collection_b": coll_b,
                            "penetration_mm": 0.0,
                        },
                    }));
                } else if dist < threshold_m {
                    findings.push(json!({
                        "rule": "tight_gap",
                        "element_ids": [idx_a, idx_b],
                        "severity": "warn",
                        "distance_mm": round4(dist * 1000.0),
                        "method": "mesh-kdtree",
                        "evidence": {
                            "a": name_a, "b": name_b,
                            "collection_a": coll_a, "collection_b": coll_b,
                            "gap_mm": round4(dist * 1000.0),
                        },
                    }));
                } else {
                    findings.push(json!({
                        "rule": "clearance",
                        "element_ids": [idx_a, idx_b],
                        "severity": "info",
                        "distance_mm": round4(dist * 1000.0),
                        "method": "mesh-kdtree",
                        "evidence": {
                            "a": name_a, "b": name_b,
                            "collection_a": coll_a, "collection_b": coll_b,
                            "clearance_mm": round4(dist * 1000.0),
                        },
                    }));
                }
                return findings;
            }
        }
    }

    // 无点云回退：AABB 穿透深度（保守）
    findings.push(json!({
        "rule": "interference",
        "element_ids": [idx_a, idx_b],
        "severity": "critical",
        "distance_mm": 0.0,
        "method": "aabb-fallback",
        "evidence": {
            "a": name_a, "b": name_b,
            "collection_a": coll_a, "collection_b": coll_b,
            "penetration_mm": round4(-aabb_d * 1000.0),
        },
    }));
    findings
}

/// Python round(x, 4) 语义。
fn round4(v: f64) -> f64 {
    (v * 10000.0).round() / 10000.0
}

/// 运行 audit3d：解析 blender_dump 报告 → findings + metrics。
pub fn run(p: &Audit3dParams) -> Result<Value, String> {
    let data: Value = serde_json::from_str(
        &std::fs::read_to_string(&p.report).map_err(|e| format!("read {}: {e}", p.report))?,
    )
    .map_err(|e| format!("parse {}: {e}", p.report))?;
    let objs = data
        .get("objects3d")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let mesh_objs: Vec<Value> = objs
        .iter()
        .filter(|o| {
            o.get("type")
                .and_then(|t| t.as_str())
                .map(|t| MESH_TYPES.contains(&t))
                .unwrap_or(false)
        })
        .cloned()
        .collect();
    let threshold_m = p.gap_threshold / 1000.0;

    let mut findings = Vec::new();
    let mut interference_count = 0;
    let mut gap_warnings = 0;
    let mut verified_clearance = 0;
    let mut method_used: Vec<String> = Vec::new();

    for i in 0..mesh_objs.len() {
        for j in (i + 1)..mesh_objs.len() {
            let res = analyze_pair(&mesh_objs[i], &mesh_objs[j], i, j, threshold_m, &p.method);
            for f in &res {
                match f["rule"].as_str().unwrap_or("") {
                    "interference" => interference_count += 1,
                    "tight_gap" => gap_warnings += 1,
                    "clearance" => verified_clearance += 1,
                    _ => {}
                }
                let m = f["method"].as_str().unwrap_or("").to_string();
                if !method_used.contains(&m) {
                    method_used.push(m);
                }
            }
            findings.extend(res);
        }
    }
    method_used.sort();

    let metrics = json!({
        "total_objects": objs.len(),
        "mesh_objects": mesh_objs.len(),
        "pair_checks": mesh_objs.len() * mesh_objs.len().saturating_sub(1) / 2,
        "interference_count": interference_count,
        "gap_warnings": gap_warnings,
        "verified_clearance": verified_clearance,
        "gap_threshold_mm": p.gap_threshold,
        "methods": method_used,
        "scipy_available": false,
    });

    let mut report = crate::schema::envelope(
        Some("audit3d"),
        &["bpy", "audit3d"],
        Some("world_m"),
        Some(json!({"type": "blender", "from": p.report})),
    );
    report["findings"] = json!(findings);
    report["metrics"] = metrics;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube(min: f64, max: f64) -> Value {
        // 8 角点（bpy bound_box 顺序）
        let pts = [
            [min, min, min],
            [min, min, max],
            [min, max, max],
            [min, max, min],
            [max, min, min],
            [max, min, max],
            [max, max, max],
            [max, max, min],
        ];
        json!(pts.iter().map(|p| json!(p)).collect::<Vec<_>>())
    }

    #[test]
    fn sat_separated_boxes() {
        // 两个分离的立方体：间隙 1m
        let (ca, axa, ha) = obb_axes(&corners(&cube(0.0, 1.0)));
        let (cb, axb, hb) = obb_axes(&corners(&cube(2.0, 3.0)));
        let sep = sat_separation(ca, axa, ha, cb, axb, hb);
        assert!((sep - 1.0).abs() < 1e-9, "分离间隙应为 1.0，实际 {sep}");
    }

    #[test]
    fn sat_overlapping_boxes() {
        // 重叠立方体
        let (ca, axa, ha) = obb_axes(&corners(&cube(0.0, 2.0)));
        let (cb, axb, hb) = obb_axes(&corners(&cube(1.0, 3.0)));
        let sep = sat_separation(ca, axa, ha, cb, axb, hb);
        assert_eq!(sep, 0.0, "重叠应返回 0");
    }

    #[test]
    fn aabb_distance_separated() {
        let d = bbox3d_distance(&cube(0.0, 1.0), &cube(2.0, 3.0));
        assert!((d - 1.0).abs() < 1e-9);
    }

    #[test]
    fn aabb_distance_overlap() {
        let d = bbox3d_distance(&cube(0.0, 2.0), &cube(1.0, 3.0));
        assert!(d < 0.0, "重叠应为负（穿透深度），实际 {d}");
    }

    #[test]
    fn surface_distance_points() {
        let va = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let vb = vec![[0.0, 2.0, 0.0], [1.0, 2.0, 0.0]];
        let d = surface_distance(&va, &vb).unwrap();
        assert!((d - 2.0).abs() < 1e-9, "点云间距应为 2.0，实际 {d}");
    }

    #[test]
    fn analyze_pair_clearance() {
        // 分离立方体 + 空点云 → AABB 回退（interference，保守）
        let a = json!({
            "name": "A", "type": "MESH", "collection": "C",
            "bbox3d": cube(0.0, 1.0), "verts": json!([]),
        });
        let b = json!({
            "name": "B", "type": "MESH", "collection": "C",
            "bbox3d": cube(2.0, 3.0), "verts": json!([]),
        });
        // 阈值 1.5：预筛不跳过（aabb_d=1.0 ≤ 1.5），OBB sep=1.0 < 1.5 →
        // 无点云 → AABB 回退
        let f = analyze_pair(&a, &b, 0, 1, 1.5, "auto");
        assert_eq!(f.len(), 1);
        assert_eq!(f[0]["rule"], "interference");
        assert_eq!(f[0]["method"], "aabb-fallback");
    }

    #[test]
    fn analyze_pair_prefilter_skips() {
        // 分离立方体 + 小阈值 → 预筛直接跳过（与 Python 一致）
        let a = json!({
            "name": "A", "type": "MESH", "collection": "C",
            "bbox3d": cube(0.0, 1.0), "verts": json!([]),
        });
        let b = json!({
            "name": "B", "type": "MESH", "collection": "C",
            "bbox3d": cube(2.0, 3.0), "verts": json!([]),
        });
        let f = analyze_pair(&a, &b, 0, 1, 0.5, "auto");
        assert_eq!(f.len(), 0, "aabb_d=1.0 > 0.5 → 预筛跳过");
    }
}
