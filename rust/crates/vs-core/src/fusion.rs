//! 融合引擎（对应 Python `vs_fusion.py`，schema v3 输出逐字段一致）。
//!
//! 三阶段流水线：匈牙利全局匹配 → 传感器 mass 映射（Dempster-Shafer）→
//! 正交和组合 → belief/plausibility → verdict。纯算法，零模型，确定性可复算。

use std::path::PathBuf;

use serde_json::{json, Value};

use crate::bbox::bbox_iou;
use crate::color::delta_e76;
use crate::util::py_round_n;

// ---------------------------------------------------------------- 辨识框架

/// 三元素框架 Θ = {consistent, conflict, uncertain} 的索引。
pub const CONSISTENT: usize = 0;
pub const CONFLICT: usize = 1;
pub const UNCERTAIN: usize = 2;

/// mass 函数：`[consistent, conflict, uncertain]`。
type Mass = [f64; 3];

// 决策阈值（与 Python 一致）
const BELIEF_CONFIRM: f64 = 0.60;
const PLAUSIBILITY_CONFLICT: f64 = 0.40;
const K_MAX: f64 = 0.70;
const TAU_COLOR: f64 = 5.0;
const OCR_CONF_SCALE: f64 = 0.8;
const SEG_AREA_SCALE: f64 = 2.0;

// ---------------------------------------------------------------- mass 映射

/// OCR 文本存在性：conf 高 → consistent 证据强。
pub fn mass_from_ocr(conf: f64) -> Mass {
    let c = conf.clamp(0.0, 1.0) * OCR_CONF_SCALE;
    [c, 0.0, 1.0 - c]
}

/// DOM 声明色 vs 像素实测：ΔE 小 → consistent 强。
pub fn mass_from_color_delta(delta_e: f64) -> Mass {
    let c = (-delta_e / TAU_COLOR).exp();
    [c, 1.0 - c, 0.0]
}

/// 检测器置信度：conf 即 consistent。
pub fn mass_from_detect(conf: f64) -> Mass {
    let c = conf.clamp(0.0, 1.0);
    [c, 0.0, 1.0 - c]
}

/// 前景面积比：大 → 前景明确。
pub fn mass_from_segment(area_ratio: f64) -> Mass {
    let c = (area_ratio.clamp(0.0, 1.0) * SEG_AREA_SCALE).min(1.0);
    [c, 0.0, 1.0 - c]
}

/// 显著性分数：高 → 候选区有效。
pub fn mass_from_saliency(score: f64) -> Mass {
    let c = score.clamp(0.0, 1.0);
    [c, 0.0, 1.0 - c]
}

/// 深度一致性：|Δdepth| 小 → 一致。
pub fn mass_from_depth(depth_delta: f64, sigma: f64) -> Mass {
    let c = 1.0 - (depth_delta.abs() / sigma).min(1.0);
    [c, 1.0 - c, 0.0]
}

/// 规则引擎置信度。
pub fn mass_from_rule(rule_conf: f64) -> Mass {
    let c = rule_conf.clamp(0.0, 1.0);
    [c, 0.0, 1.0 - c]
}

// ---------------------------------------------------------------- D-S 组合

/// Dempster 正交和：三元素框架的闭式解。
fn ds_combine(m1: Mass, m2: Mass) -> Mass {
    let k = m1[CONSISTENT] * m2[CONFLICT] + m1[CONFLICT] * m2[CONSISTENT];
    let denom = 1.0 - k;
    if denom <= 1e-12 {
        // 完全冲突 → 全不确定（防除零）
        return [0.0, 0.0, 1.0];
    }
    let m_cons = (m1[CONSISTENT] * m2[CONSISTENT]
        + m1[CONSISTENT] * m2[UNCERTAIN]
        + m1[UNCERTAIN] * m2[CONSISTENT])
        / denom;
    let m_conf =
        (m1[CONFLICT] * m2[CONFLICT] + m1[CONFLICT] * m2[UNCERTAIN] + m1[UNCERTAIN] * m2[CONFLICT])
            / denom;
    let m_unc = (m1[UNCERTAIN] * m2[UNCERTAIN]) / denom;
    [m_cons, m_conf, m_unc]
}

/// 顺序组合多个 mass 函数；空列表 → 全不确定。
fn ds_combine_all(masses: &[Mass]) -> Mass {
    if masses.is_empty() {
        return [0.0, 0.0, 1.0];
    }
    let mut acc = masses[0];
    for &m in &masses[1..] {
        acc = ds_combine(acc, m);
    }
    acc
}

/// belief(A) = m(A)；plausibility(A) = m(A) + m(uncertain)。
fn belief_plausibility(m: Mass) -> (f64, f64) {
    (m[CONSISTENT], m[CONSISTENT] + m[UNCERTAIN])
}

/// 决策规则：confirmed / conflict / needs_review。
fn decide(bel: f64, pla: f64, k: f64) -> &'static str {
    if k > K_MAX {
        "needs_review" // 证据高度冲突 → 需复查而非误报
    } else if bel > BELIEF_CONFIRM {
        "confirmed"
    } else if pla < PLAUSIBILITY_CONFLICT {
        "conflict"
    } else {
        "needs_review"
    }
}

// ---------------------------------------------------------------- 匈牙利匹配

/// 文本相似度：精确匹配 1.0；包含 0.7；否则编辑距离归一化。
fn text_sim(a: &str, b: &str) -> f64 {
    let (a, b) = (a.trim().to_lowercase(), b.trim().to_lowercase());
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    if a == b {
        return 1.0;
    }
    if a.contains(&b) || b.contains(&a) {
        return 0.7;
    }
    let d = levenshtein(&a, &b);
    (1.0 - d as f64 / a.len().max(b.len()) as f64).max(0.0)
}

/// Levenshtein 编辑距离（标准 DP，与 Python 一致）。
fn levenshtein(a: &str, b: &str) -> usize {
    let (a, b) = if a.len() < b.len() { (b, a) } else { (a, b) };
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut cur = vec![i + 1];
        for (j, cb) in b.chars().enumerate() {
            cur.push(
                (prev[j + 1] + 1)
                    .min(cur[j] + 1)
                    .min(prev[j] + usize::from(ca != cb)),
            );
        }
        prev = cur;
    }
    prev[b.len()]
}

/// 匈牙利算法（Kuhn-Munkres，O(n³)）：最小代价一对一匹配。
///
/// 支持矩形矩阵（补齐为方阵，padding 用大代价）；返回匹配对 `(row, col)`
/// 按行序（与 scipy `linear_sum_assignment` 一致，差分测试逐对验证）。
fn hungarian(cost: &[Vec<f64>]) -> Vec<(usize, usize)> {
    let (n, m) = (cost.len(), cost[0].len());
    if n == 0 || m == 0 {
        return vec![];
    }
    // 补齐为方阵：padding 行/列用大代价（不参与最优匹配）
    let size = n.max(m);
    let pad = 1e9f64;
    let mut sq = vec![vec![pad; size]; size];
    for i in 0..n {
        for j in 0..m {
            sq[i][j] = cost[i][j];
        }
    }
    // 经典 Hungarian：u/v 势能 + 交替路径
    let mut u = vec![0.0f64; size + 1];
    let mut v = vec![0.0f64; size + 1];
    let mut p = vec![0usize; size + 1]; // p[j] = 匹配的行（1-based）
    let mut way = vec![0usize; size + 1];
    for i in 1..=size {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![f64::INFINITY; size + 1];
        let mut used = vec![false; size + 1];
        loop {
            used[j0] = true;
            let (i0, mut j1) = (p[j0], 0usize);
            let mut delta = f64::INFINITY;
            for j in 1..=size {
                if !used[j] {
                    let cur = sq[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=size {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        // 增广路径回溯
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    // 收集匹配（行 → 列），按行序；过滤 padding 匹配
    let mut pairs: Vec<(usize, usize)> = p
        .iter()
        .enumerate()
        .skip(1)
        .filter(|(j, &row)| row != 0 && row - 1 < n && j - 1 < m)
        .map(|(j, &row)| (row - 1, j - 1))
        .collect();
    pairs.sort_by_key(|&(i, _)| i);
    pairs
}

/// 全局最优一对一匹配（对应 Python `hungarian_match`）。
fn hungarian_match(
    boxes_a: &[Vec<f64>],
    boxes_b: &[Vec<f64>],
    texts_a: &[Option<String>],
    texts_b: &[Option<String>],
    alpha: f64,
) -> Vec<(usize, usize, f64)> {
    let (n, m) = (boxes_a.len(), boxes_b.len());
    if n == 0 || m == 0 {
        return vec![];
    }
    let mut cost = vec![vec![0.0f64; m]; n];
    for i in 0..n {
        for j in 0..m {
            let iou = bbox_iou(&boxes_a[i], &boxes_b[j]);
            let sim = match (&texts_a[i], &texts_b[j]) {
                (Some(a), Some(b)) if !a.is_empty() && !b.is_empty() => text_sim(a, b),
                _ => 0.0,
            };
            cost[i][j] = 1.0 - iou - alpha * sim;
        }
    }
    let mut pairs = hungarian(&cost);
    pairs.sort_by_key(|&(i, _)| i); // scipy 按行序返回
    pairs
        .into_iter()
        .filter_map(|(i, j)| {
            let iou = bbox_iou(&boxes_a[i], &boxes_b[j]);
            (iou > 0.0).then_some((i, j, iou))
        })
        .collect()
}

// ---------------------------------------------------------------- 融合主流程

/// 按 source 自动分组：同 source 元素归一组，两两组合。
fn auto_groups(elements: &[Value]) -> Vec<(Vec<Value>, Vec<Value>)> {
    let mut groups: Vec<(String, Vec<Value>)> = Vec::new();
    for el in elements {
        let src = el
            .get("source")
            .and_then(|s| s.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str())
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .unwrap_or_else(|| "unknown".to_string());
        match groups.iter_mut().find(|(k, _)| *k == src) {
            Some((_, v)) => v.push(el.clone()),
            None => groups.push((src, vec![el.clone()])),
        }
    }
    let mut out = Vec::new();
    for i in 0..groups.len() {
        for j in (i + 1)..groups.len() {
            out.push((groups[i].1.clone(), groups[j].1.clone()));
        }
    }
    out
}

/// 从两个元素提取 mass 证据。
fn element_masses(el_a: &Value, el_b: &Value, iou: f64) -> Vec<Mass> {
    let mut masses = Vec::new();
    // IoU 本身是证据：高 IoU → 一致
    let iou_ev = (iou * 2.0).min(1.0);
    masses.push([iou_ev, 0.0, 1.0 - iou_ev]);
    // 各自 conf
    for el in [el_a, el_b] {
        if let Some(conf) = el.get("conf").and_then(|c| c.as_f64()) {
            masses.push(mass_from_ocr(conf));
        }
    }
    // 颜色一致性（若都有 color）
    let color_of = |el: &Value| -> Option<String> {
        let color = el.get("color")?;
        color
            .get("fill")
            .or_else(|| color.get("text"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    };
    if let (Some(ca), Some(cb)) = (color_of(el_a), color_of(el_b)) {
        if let (Ok(ra), Ok(rb)) = (crate::color::parse_hex(&ca), crate::color::parse_hex(&cb)) {
            let de = delta_e76(
                [ra[0] as f64, ra[1] as f64, ra[2] as f64],
                [rb[0] as f64, rb[1] as f64, rb[2] as f64],
            );
            masses.push(mass_from_color_delta(de));
        }
    }
    masses
}

/// 组合过程中的最大冲突系数（近似：两两组合的 K 最大值）。
fn conflict_coef(masses: &[Mass]) -> f64 {
    let mut k_max = 0.0f64;
    for i in 0..masses.len() {
        for j in (i + 1)..masses.len() {
            let k = masses[i][CONSISTENT] * masses[j][CONFLICT]
                + masses[i][CONFLICT] * masses[j][CONSISTENT];
            k_max = k_max.max(k);
        }
    }
    k_max
}

/// 构造 finding（schema v3）。参数镜像 Python `_finding` 签名。
#[allow(clippy::too_many_arguments)]
fn finding(
    ftype: &str,
    el_a: Option<&Value>,
    el_b: Option<&Value>,
    m: Option<Mass>,
    bel: f64,
    pla: f64,
    k: f64,
    evidence: Value,
) -> Value {
    let bbox = el_a
        .or(el_b)
        .and_then(|el| el.get("bbox"))
        .and_then(|b| b.as_array())
        .cloned();
    let mut f = json!({
        "type": ftype,
        "bbox": bbox,
        "belief": py_round_n(bel, 3),
        "plausibility": py_round_n(pla, 3),
        "uncertainty": py_round_n(1.0 - (pla - bel), 3),
        "verdict": if m.is_some() { decide(bel, pla, k) } else { "needs_review" },
        "evidence": evidence,
    });
    if let Some(a) = el_a {
        if let Some(id) = a.get("id") {
            f["a_id"] = id.clone();
        }
    }
    if let Some(b) = el_b {
        if let Some(id) = b.get("id") {
            f["b_id"] = id.clone();
        }
    }
    f
}

/// 对元素列表执行融合，输出 findings。
fn fuse_elements(
    elements: &[Value],
    match_groups: Option<&[(Vec<Value>, Vec<Value>)]>,
) -> Vec<Value> {
    let groups: Vec<(Vec<Value>, Vec<Value>)> = match match_groups {
        Some(g) => g.to_vec(),
        None => auto_groups(elements),
    };
    let mut findings = Vec::new();
    for (group_a, group_b) in groups {
        let boxes_a: Vec<Vec<f64>> = group_a
            .iter()
            .filter_map(|el| {
                el.get("bbox")
                    .and_then(|b| b.as_array())
                    .map(|b| b.iter().filter_map(|v| v.as_f64()).collect())
            })
            .collect();
        let boxes_b: Vec<Vec<f64>> = group_b
            .iter()
            .filter_map(|el| {
                el.get("bbox")
                    .and_then(|b| b.as_array())
                    .map(|b| b.iter().filter_map(|v| v.as_f64()).collect())
            })
            .collect();
        let texts_a: Vec<Option<String>> = group_a
            .iter()
            .map(|el| {
                el.get("text")
                    .and_then(|t| t.as_str())
                    .map(|s| s.to_string())
            })
            .collect();
        let texts_b: Vec<Option<String>> = group_b
            .iter()
            .map(|el| {
                el.get("text")
                    .and_then(|t| t.as_str())
                    .map(|s| s.to_string())
            })
            .collect();
        let pairs = hungarian_match(&boxes_a, &boxes_b, &texts_a, &texts_b, 0.3);
        let matched_b: std::collections::HashSet<usize> =
            pairs.iter().map(|&(_, j, _)| j).collect();
        // 已匹配对 → 一致性证据
        for (i, j, iou) in &pairs {
            let masses = element_masses(&group_a[*i], &group_b[*j], *iou);
            let m = ds_combine_all(&masses);
            let (bel, pla) = belief_plausibility(m);
            let k = conflict_coef(&masses);
            findings.push(finding(
                "match_consistent",
                Some(&group_a[*i]),
                Some(&group_b[*j]),
                Some(m),
                bel,
                pla,
                k,
                json!({
                    "iou": py_round_n(*iou, 3),
                    "sensors": [
                        group_a[*i].get("source"),
                        group_b[*j].get("source"),
                    ],
                }),
            ));
        }
        // 未匹配 → 缺失/多余证据
        let matched_a: std::collections::HashSet<usize> =
            pairs.iter().map(|&(i, _, _)| i).collect();
        for (i, el) in group_a.iter().enumerate() {
            if !matched_a.contains(&i) {
                findings.push(finding(
                    "unmatched_a",
                    Some(el),
                    None,
                    None,
                    0.0,
                    0.0,
                    0.0,
                    json!({"reason": "no match in group_b", "sensor": el.get("source")}),
                ));
            }
        }
        for (j, el) in group_b.iter().enumerate() {
            if !matched_b.contains(&j) {
                findings.push(finding(
                    "unmatched_b",
                    None,
                    Some(el),
                    None,
                    0.0,
                    0.0,
                    0.0,
                    json!({"reason": "no match in group_a", "sensor": el.get("source")}),
                ));
            }
        }
    }
    findings
}

/// 运行融合引擎（对应 Python `main()`）。
pub fn run(reports: &[PathBuf], groups: Option<&str>) -> Result<Value, String> {
    let mut all_elements: Vec<Value> = Vec::new();
    let mut sensors: Vec<String> = Vec::new();
    for path in reports {
        let rep: Value = serde_json::from_str(
            &std::fs::read_to_string(path).map_err(|e| format!("cannot read {path:?}: {e}"))?,
        )
        .map_err(|e| format!("invalid JSON in {path:?}: {e}"))?;
        if let Some(els) = rep.get("elements").and_then(|e| e.as_array()) {
            all_elements.extend(els.iter().cloned());
        }
        if let Some(ss) = rep.get("sensors").and_then(|s| s.as_array()) {
            for s in ss {
                if let Some(name) = s.as_str() {
                    if !sensors.iter().any(|x| x == name) {
                        sensors.push(name.to_string());
                    }
                }
            }
        }
    }
    let match_groups = match groups {
        Some(gdef) => {
            let gdef: Value = serde_json::from_str(gdef).map_err(|e| format!("bad groups: {e}"))?;
            let by_id: std::collections::HashMap<i64, Value> = all_elements
                .iter()
                .filter_map(|el| {
                    el.get("id")
                        .and_then(|id| id.as_i64())
                        .map(|id| (id, el.clone()))
                })
                .collect();
            let mut out = Vec::new();
            if let Some(arr) = gdef.as_array() {
                for pair in arr {
                    let ga = pair.get(0).and_then(|v| v.as_array()).map(|a| {
                        a.iter()
                            .filter_map(|i| i.as_i64())
                            .filter_map(|i| by_id.get(&i).cloned())
                            .collect::<Vec<_>>()
                    });
                    let gb = pair.get(1).and_then(|v| v.as_array()).map(|a| {
                        a.iter()
                            .filter_map(|i| i.as_i64())
                            .filter_map(|i| by_id.get(&i).cloned())
                            .collect::<Vec<_>>()
                    });
                    if let (Some(ga), Some(gb)) = (ga, gb) {
                        out.push((ga, gb));
                    }
                }
            }
            Some(out)
        }
        None => None,
    };
    let findings = fuse_elements(&all_elements, match_groups.as_deref());
    let mut report = crate::schema::envelope(
        Some("fusion"),
        &sensors.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        Some("image_px"),
        Some(
            json!({"type": "fused", "reports": reports.iter().map(|p| p.display().to_string()).collect::<Vec<_>>()}),
        ),
    );
    report["findings"] = json!(findings);
    report["notation"] = json!(crate::schema::NOTATION_GUIDE);
    let confirmed = findings
        .iter()
        .filter(|f| f["verdict"] == "confirmed")
        .count();
    let conflict = findings
        .iter()
        .filter(|f| f["verdict"] == "conflict")
        .count();
    let needs_review = findings
        .iter()
        .filter(|f| f["verdict"] == "needs_review")
        .count();
    report["metrics"] = json!({
        "element_count": all_elements.len(),
        "finding_count": findings.len(),
        "confirmed": confirmed,
        "conflict": conflict,
        "needs_review": needs_review,
    });
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mass_functions() {
        let m = mass_from_ocr(0.5);
        assert!((m[CONSISTENT] - 0.4).abs() < 1e-9); // 0.5 * 0.8
        assert!((m[UNCERTAIN] - 0.6).abs() < 1e-9);
        let m = mass_from_color_delta(0.0);
        assert!((m[CONSISTENT] - 1.0).abs() < 1e-9);
        let m = mass_from_segment(0.8);
        assert!((m[CONSISTENT] - 1.0).abs() < 1e-9); // 0.8*2 截断到 1
    }

    #[test]
    fn ds_combine_identity() {
        // 与全不确定组合 = 自身
        let m = [0.6, 0.1, 0.3];
        let u = [0.0, 0.0, 1.0];
        let r = ds_combine(m, u);
        assert!((r[CONSISTENT] - 0.6).abs() < 1e-9);
        assert!((r[CONFLICT] - 0.1).abs() < 1e-9);
        assert!((r[UNCERTAIN] - 0.3).abs() < 1e-9);
    }

    #[test]
    fn ds_combine_conflict() {
        // 完全冲突 → 全不确定
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        let r = ds_combine(a, b);
        assert_eq!(r, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn decide_thresholds() {
        assert_eq!(decide(0.8, 0.9, 0.1), "confirmed");
        assert_eq!(decide(0.1, 0.2, 0.1), "conflict");
        assert_eq!(decide(0.5, 0.6, 0.8), "needs_review"); // K > K_MAX
        assert_eq!(decide(0.5, 0.6, 0.1), "needs_review");
    }

    #[test]
    fn levenshtein_known() {
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("abc", "abc"), 0);
        assert_eq!(levenshtein("", "abc"), 3);
    }

    #[test]
    fn text_sim_rules() {
        assert_eq!(text_sim("Hello", "hello"), 1.0); // 精确（忽略大小写）
        assert_eq!(text_sim("abc", "abcdef"), 0.7); // 包含
        assert_eq!(text_sim("", "x"), 0.0);
        assert!(text_sim("kitten", "sitting") > 0.0);
    }

    #[test]
    fn hungarian_small() {
        // 与 scipy linear_sum_assignment 实测一致
        let cost = vec![
            vec![1.0, 0.5, 0.8],
            vec![0.5, 1.0, 0.3],
            vec![0.7, 0.2, 0.9],
        ];
        let pairs = hungarian(&cost);
        assert_eq!(pairs, vec![(0, 1), (1, 2), (2, 0)]);
        // 平局：全相等 → 对角
        let cost2 = vec![vec![0.5, 0.5], vec![0.5, 0.5]];
        assert_eq!(hungarian(&cost2), vec![(0, 0), (1, 1)]);
        // 非方阵 3x2
        let cost3 = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        assert_eq!(hungarian(&cost3), vec![(0, 0), (1, 1)]);
    }
}
