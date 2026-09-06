//! 确定性设计准则规则引擎（对应 Python `vs_rules.py`，schema v3 输出逐字段一致）。
//!
//! 所有规则基于测量值 + 显式阈值：文本对比度 / 重叠 / 对齐漂移 / 间距异常 / 安全区。
//! 对齐/间距仅评估设计元素（DOM/PPTX 或带 font/z 元数据）。

use std::path::Path;

use serde_json::{json, Value};

use crate::bbox::{bbox_area, bbox_intersection, bbox_iou};
use crate::color::{contrast_ratio, parse_hex};
use crate::util::py_round_n;

/// 规则清单（与 Python `RULES` 一致，保证可审计性）。
pub fn rules() -> Vec<Value> {
    vec![
        json!({"code": "text_contrast", "desc": "WCAG AA 对比度（正常文本≥4.5:1，大文本≥3:1）", "thresholds": {"aa_normal": 4.5, "aa_large": 3.0}}),
        json!({"code": "overlap", "desc": "两两 bbox 重叠（IoU 与最小重叠面积）", "thresholds": {"iou": 0.05, "min_area": 400.0}}),
        json!({"code": "alignment_drift", "desc": "边缘/中线聚类内的近邻漂移（> tol 且 ≤ drift×tol）；仅设计元素（source∈dom/pptx 或带 font/z 元数据）", "thresholds": {"tol": 4.0, "drift": 4.0, "scope": "designed"}}),
        json!({"code": "spacing_anomaly", "desc": "同行/列连续间距离群（> k × 中位间距）；仅设计元素", "thresholds": {"k": 2.5, "scope": "designed"}}),
        json!({"code": "safe_area", "desc": "元素贴边（≤ margin px）或出界", "thresholds": {"margin": 2.0}}),
    ]
}

/// 规则引擎参数（对应 Python CLI 参数，默认值经 bench/tune_rules.py 寻优）。
pub struct RulesParams {
    pub canvas: Option<String>,
    pub align_tol: f64,
    pub align_drift: f64,
    pub margin: f64,
    pub spacing_k: f64,
    pub overlap_threshold: f64,
    pub min_overlap_area: f64,
}

impl Default for RulesParams {
    fn default() -> Self {
        Self {
            canvas: None,
            align_tol: 4.0,
            align_drift: 4.0,
            margin: 1.0,
            spacing_k: 1.8,
            overlap_threshold: 0.01,
            min_overlap_area: 200.0,
        }
    }
}

/// 设计意图判定：DOM/PPTX 来源或带 font/z 元数据。
fn is_designed(el: &Value) -> bool {
    let src = el.get("source");
    let designed_src = src
        .and_then(|s| s.as_array())
        .map(|a| a.iter().any(|v| matches!(v.as_str(), Some("dom" | "pptx"))))
        .or_else(|| {
            src.and_then(|s| s.as_str())
                .map(|s| s == "dom" || s == "pptx")
        })
        .unwrap_or(false);
    if designed_src {
        return true;
    }
    el.get("font").is_some() || el.get("z").is_some()
}

/// 从报告提取元素（elements[] 优先，PPTX shapes[] 兜底）。
fn extract_elements(report: &Value) -> Vec<Value> {
    if let Some(els) = report.get("elements").and_then(|e| e.as_array()) {
        return els.clone();
    }
    let mut out = Vec::new();
    if let Some(slides) = report.get("slides").and_then(|s| s.as_array()) {
        for slide in slides {
            if let Some(shapes) = slide.get("shapes").and_then(|s| s.as_array()) {
                for sh in shapes {
                    let (Some(pos), Some(size)) = (sh.get("pos_pt"), sh.get("size_pt")) else {
                        continue;
                    };
                    let (Some(px), Some(py)) = (
                        pos.get(0).and_then(|v| v.as_f64()),
                        pos.get(1).and_then(|v| v.as_f64()),
                    ) else {
                        continue;
                    };
                    let (Some(sx), Some(sy)) = (
                        size.get(0).and_then(|v| v.as_f64()),
                        size.get(1).and_then(|v| v.as_f64()),
                    ) else {
                        continue;
                    };
                    let mut texts = Vec::new();
                    if let Some(ts) = sh.get("texts").and_then(|t| t.as_array()) {
                        for t in ts {
                            if let Some(runs) = t.get("runs").and_then(|r| r.as_array()) {
                                for run in runs {
                                    texts.push(json!({
                                        "text": run.get("text").and_then(|v| v.as_str()).unwrap_or(""),
                                        "color": run.get("color"),
                                        "size_pt": run.get("size_pt"),
                                    }));
                                }
                            }
                        }
                    }
                    out.push(json!({
                        "bbox": [px, py, px + sx, py + sy],
                        "fill": sh.get("fill"),
                        "texts": texts,
                        "name": sh.get("name"),
                    }));
                }
            }
        }
    }
    out
}

/// 元素标签（name/text/type 或 el{idx}），截断 60 字符。
fn label(el: &Value, idx: usize) -> String {
    let fallback = format!("el{idx}");
    let s = el
        .get("name")
        .and_then(|v| v.as_str())
        .or_else(|| el.get("text").and_then(|v| v.as_str()))
        .or_else(|| el.get("type").and_then(|v| v.as_str()))
        .unwrap_or(&fallback);
    s.chars().take(60).collect()
}

/// 画布尺寸：参数 "WxH" 或报告 source.size_px/viewport/slide_size_pt。
fn canvas_of(report: &Value, canvas_arg: &Option<String>) -> Option<(i64, i64)> {
    let to_int = |v: &Value| v.as_f64().map(|f| f as i64);
    if let Some(c) = canvas_arg {
        let parts: Vec<&str> = c.split('x').collect();
        if parts.len() == 2 {
            if let (Ok(w), Ok(h)) = (parts[0].parse(), parts[1].parse()) {
                return Some((w, h));
            }
        }
    }
    let src = report.get("source");
    for key in ["size_px", "viewport", "slide_size_pt"] {
        if let Some(v) = src.and_then(|s| s.get(key)).and_then(|v| v.as_array()) {
            if v.len() == 2 {
                if let (Some(w), Some(h)) = (to_int(&v[0]), to_int(&v[1])) {
                    return Some((w, h));
                }
            }
        }
    }
    None
}

/// Python `str(round(v, n))` 格式：保留 n 位小数（如 7.0 → "7.0"，非 Rust 的 "7"）。
fn py_fmt(v: f64, n: i32) -> String {
    format!("{:.*}", n as usize, py_round_n(v, n))
}

/// 1D 密度聚类：按 tol 容差归并（与聚类首元素比较，确定性）。
fn cluster(values: &[f64], tol: f64) -> Vec<Vec<f64>> {
    let mut groups: Vec<Vec<f64>> = Vec::new();
    for &v in values {
        let mut placed = false;
        for g in groups.iter_mut() {
            if (v - g[0]).abs() <= tol {
                g.push(v);
                placed = true;
                break;
            }
        }
        if !placed {
            groups.push(vec![v]);
        }
    }
    groups
}

/// R1 文本对比度（WCAG AA：正常 ≥4.5，大文本 ≥3.0）。
fn check_contrast(els: &[Value]) -> Vec<Value> {
    let mut out = Vec::new();
    for (idx, el) in els.iter().enumerate() {
        let mut pairs: Vec<(String, f64)> = Vec::new();
        if let Some(texts) = el.get("texts").and_then(|t| t.as_array()) {
            for t in texts {
                if let Some(c) = t.get("color").and_then(|c| c.as_str()) {
                    pairs.push((
                        c.to_string(),
                        t.get("size_pt").and_then(|v| v.as_f64()).unwrap_or(11.0),
                    ));
                }
            }
        } else if let Some(c) = el
            .get("color")
            .and_then(|c| c.get("text"))
            .and_then(|t| t.as_str())
        {
            pairs.push((c.to_string(), 11.0));
        }
        let fill = el
            .get("fill")
            .and_then(|f| f.as_str())
            .map(|s| s.to_string())
            .or_else(|| {
                el.get("color")
                    .and_then(|c| c.get("fill"))
                    .and_then(|f| f.as_str())
                    .map(|s| s.to_string())
            });
        let Some(fill) = fill else { continue };
        if pairs.is_empty() || fill == "BACKGROUND (5)" || fill == "SOLID (1)" {
            continue;
        }
        for (fg, size_pt) in pairs {
            let (Ok(fgrgb), Ok(fillrgb)) = (parse_hex(&fg), parse_hex(&fill)) else {
                continue;
            };
            let ratio = contrast_ratio(
                [fgrgb[0] as f64, fgrgb[1] as f64, fgrgb[2] as f64],
                [fillrgb[0] as f64, fillrgb[1] as f64, fillrgb[2] as f64],
            );
            let required = if size_pt >= 18.0 { 3.0 } else { 4.5 };
            if ratio < required {
                out.push(json!({
                    "rule": "text_contrast",
                    "element_ids": [idx],
                    "severity": if ratio < 2.0 { "critical" } else { "warn" },
                    "bbox": el.get("bbox"),
                    "evidence": {
                        "fg": fg, "bg": fill,
                        "ratio": py_round_n(ratio, 2),
                        "size_pt": size_pt, "required": required,
                    },
                    "suggested_cause": "文本色与背景色对比度不足（WCAG AA 未达标）",
                }));
            }
        }
    }
    out
}

/// R2 两两重叠（IoU > 阈值，最小重叠面积）。
fn check_overlap(els: &[Value], iou_th: f64, min_area: f64) -> Vec<Value> {
    let mut out = Vec::new();
    for i in 0..els.len() {
        for j in (i + 1)..els.len() {
            let (Some(ba), Some(bb)) = (
                els[i].get("bbox").and_then(|v| v.as_array()),
                els[j].get("bbox").and_then(|v| v.as_array()),
            ) else {
                continue;
            };
            let ba: Vec<f64> = ba.iter().filter_map(|v| v.as_f64()).collect();
            let bb: Vec<f64> = bb.iter().filter_map(|v| v.as_f64()).collect();
            let Some(inter) = bbox_intersection(&ba, &bb) else {
                continue;
            };
            let ia = bbox_area(&[
                inter[0] as f64,
                inter[1] as f64,
                inter[2] as f64,
                inter[3] as f64,
            ]);
            if ia < min_area {
                continue;
            }
            let iou = bbox_iou(&ba, &bb);
            if iou > iou_th {
                out.push(json!({
                    "rule": "overlap",
                    "element_ids": [i, j],
                    "severity": "warn",
                    "bbox": [inter[0], inter[1], inter[2], inter[3]],
                    "evidence": {
                        "iou": py_round_n(iou, 2),
                        "area": ia.round(),
                        "a": label(&els[i], i),
                        "b": label(&els[j], j),
                    },
                    "suggested_cause": "两个元素区域重叠（可能遮挡内容或布局冲突）",
                }));
            }
        }
    }
    out
}

/// R3 对齐漂移：左缘/右缘/水平中线聚类，标记落在 (tol, drift_k×tol] 的漂移。
fn check_alignment(els: &[Value], tol: f64, drift_k: f64) -> Vec<Value> {
    let designed: Vec<(usize, &Value)> = els
        .iter()
        .enumerate()
        .filter(|(_, el)| is_designed(el))
        .collect();
    let mut out = Vec::new();
    // 对齐轴：左缘/右缘/水平中线（键函数取 bbox 坐标）
    type AxisKey = fn(&[f64]) -> f64;
    let axes: [(&str, AxisKey); 3] = [
        ("left", |b| b[0]),
        ("right", |b| b[2]),
        ("hcenter", |b| (b[0] + b[2]) / 2.0),
    ];
    for (axis, key) in axes {
        // 直接取 bbox 数值
        let values: Vec<f64> = designed
            .iter()
            .map(|(_, el)| {
                let b: Vec<f64> = el
                    .get("bbox")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|v| v.as_f64()).collect())
                    .unwrap_or_default();
                key(&b)
            })
            .collect();
        let groups = cluster(&values, tol);
        for g in &groups {
            if g.len() < 2 {
                continue;
            }
            let mean = g.iter().sum::<f64>() / g.len() as f64;
            for (idx, el) in &designed {
                let b: Vec<f64> = el
                    .get("bbox")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|v| v.as_f64()).collect())
                    .unwrap_or_default();
                let v = key(&b);
                if g.contains(&v) {
                    continue;
                }
                let d = (v - mean).abs();
                if tol < d && d <= drift_k * tol {
                    out.push(json!({
                        "rule": "alignment_drift",
                        "element_ids": [idx],
                        "severity": "info",
                        "bbox": el.get("bbox"),
                        "evidence": {
                            "axis": axis,
                            "value": py_round_n(v, 1),
                            "cluster_mean": py_round_n(mean, 1),
                            "offset": py_round_n(d, 1),
                            "tol": tol,
                        },
                        "suggested_cause": format!("{axis} 边缘偏离对齐簇 {}px（组内有 {} 个元素对齐）", py_fmt(d, 1), g.len()),
                    }));
                }
            }
        }
    }
    out
}

/// R4 间距异常：按行分组后检测连续间距离群（> k × 中位间距，且 ≥8px）。
fn check_spacing(els: &[Value], k: f64) -> Vec<Value> {
    let designed: Vec<(usize, &Value)> = els
        .iter()
        .enumerate()
        .filter(|(_, el)| is_designed(el))
        .collect();
    let mut out = Vec::new();
    // 按 y 排序后分组（垂直重叠 > 0.5 × min 高度）
    let mut sorted: Vec<(usize, &Value)> = designed.clone();
    sorted.sort_by(|a, b| {
        let ya =
            a.1.get("bbox")
                .and_then(|v| v.as_array())
                .and_then(|b| b.get(1))
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
        let yb =
            b.1.get("bbox")
                .and_then(|v| v.as_array())
                .and_then(|b| b.get(1))
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
        ya.partial_cmp(&yb).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut rows: Vec<Vec<(usize, &Value)>> = Vec::new();
    for item in sorted {
        let (y1, y2) = bbox_y(item.1);
        let mut placed = false;
        for row in rows.iter_mut() {
            let (ry1, ry2) = bbox_y(row[0].1);
            if (y2.min(ry2) - y1.max(ry1)) > 0.5 * (y2 - y1).min(ry2 - ry1) {
                row.push(item);
                placed = true;
                break;
            }
        }
        if !placed {
            rows.push(vec![item]);
        }
    }
    for row in &rows {
        if row.len() < 3 {
            continue;
        }
        let mut row_sorted = row.clone();
        row_sorted.sort_by(|a, b| {
            let xa =
                a.1.get("bbox")
                    .and_then(|v| v.as_array())
                    .and_then(|b| b.first())
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
            let xb =
                b.1.get("bbox")
                    .and_then(|v| v.as_array())
                    .and_then(|b| b.first())
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
            xa.partial_cmp(&xb).unwrap_or(std::cmp::Ordering::Equal)
        });
        let gaps: Vec<(f64, bool)> = row_sorted
            .windows(2)
            .map(|w| {
                // 保留整数算术（Python int bbox → int gap → str "120" 而非 "120.0"）
                let (x2a, x2a_int) = bbox_val(w[0].1, 2);
                let (x1b, x1b_int) = bbox_val(w[1].1, 0);
                let gap = x1b - x2a;
                (gap, x2a_int && x1b_int && gap.fract() == 0.0)
            })
            .filter(|(g, _)| *g > 0.0)
            .collect();
        if gaps.is_empty() {
            continue;
        }
        let mut sorted_gaps: Vec<f64> = gaps.iter().map(|(g, _)| *g).collect();
        sorted_gaps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted_gaps[sorted_gaps.len() / 2];
        if median <= 0.0 {
            continue;
        }
        for (i, &(g, g_int)) in gaps.iter().enumerate() {
            if g > k * median && g >= 8.0 {
                let (a, b) = (row_sorted[i], row_sorted[i + 1]);
                let (_ax1, ay1, ax2, ay2) = bbox_xy(a.1);
                let (bx1, _, _, _) = bbox_xy(b.1);
                let fmt_gap = if g_int {
                    format!("{}", g as i64)
                } else {
                    py_fmt(g, 1)
                };
                let fmt_median = if g_int {
                    format!("{}", median as i64)
                } else {
                    py_fmt(median, 1)
                };
                out.push(json!({
                    "rule": "spacing_anomaly",
                    "element_ids": [a.0, b.0],
                    "severity": "info",
                    "bbox": [ax2, ay1, bx1, ay2],
                    "evidence": {
                        "gap": py_round_n(g, 1),
                        "median": py_round_n(median, 1),
                        "k": k,
                        "orientation": "horizontal",
                    },
                    "suggested_cause": format!("水平间距 {fmt_gap}px 明显大于同行中位间距 {fmt_median}px"),
                }));
            }
        }
    }
    out
}

/// bbox 的 (y1, y2)。
fn bbox_y(el: &Value) -> (f64, f64) {
    let b: Vec<f64> = el
        .get("bbox")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_f64()).collect())
        .unwrap_or_default();
    (
        b.get(1).copied().unwrap_or(0.0),
        b.get(3).copied().unwrap_or(0.0),
    )
}

/// bbox 的 (x1, y1, x2, y2)。
fn bbox_xy(el: &Value) -> (f64, f64, f64, f64) {
    let b: Vec<f64> = el
        .get("bbox")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_f64()).collect())
        .unwrap_or_default();
    (
        b.first().copied().unwrap_or(0.0),
        b.get(1).copied().unwrap_or(0.0),
        b.get(2).copied().unwrap_or(0.0),
        b.get(3).copied().unwrap_or(0.0),
    )
}

/// bbox 第 `idx` 个值 + 是否整数（保留 Python 整数算术语义）。
fn bbox_val(el: &Value, idx: usize) -> (f64, bool) {
    let v = el
        .get("bbox")
        .and_then(|b| b.as_array())
        .and_then(|a| a.get(idx));
    match v {
        Some(Value::Number(n)) if n.as_i64().is_some() => (n.as_f64().unwrap(), true),
        Some(Value::Number(n)) => (n.as_f64().unwrap(), false),
        _ => (0.0, false),
    }
}

/// R5 安全区：出界（critical）或贴边文本（warn）。
fn check_safe_area(els: &[Value], canvas: Option<(i64, i64)>, margin: f64) -> Vec<Value> {
    let mut out = Vec::new();
    for (idx, el) in els.iter().enumerate() {
        let (x1, y1, x2, y2) = bbox_xy(el);
        if let Some((cw, ch)) = canvas {
            let (cw, ch) = (cw as f64, ch as f64);
            if x1 < 0.0 || y1 < 0.0 || x2 > cw || y2 > ch {
                out.push(json!({
                    "rule": "safe_area",
                    "element_ids": [idx],
                    "severity": "critical",
                    "bbox": el.get("bbox"),
                    "evidence": {"canvas": [cw as i64, ch as i64], "kind": "off_canvas"},
                    "suggested_cause": "元素超出画布边界（可能被裁切或坐标错误）",
                }));
                continue;
            }
            // 通栏/通高元素 = 有意的全出血设计，不告警
            let full_bleed = (x1 <= 0.0 && x2 >= cw) || (y1 <= 0.0 && y2 >= ch);
            if full_bleed {
                continue;
            }
            let edge = x1.min(y1).min(cw - x2).min(ch - y2);
            let has_text = el.get("text").is_some() || el.get("texts").is_some();
            if edge <= margin && has_text {
                // 保留整数算术语义（Python int bbox → edge int → "0px"）
                let all_int =
                    (0..4).all(|i| bbox_val(el, i).1) && cw.fract() == 0.0 && ch.fract() == 0.0;
                let fmt_edge = if all_int && edge.fract() == 0.0 {
                    format!("{}", edge as i64)
                } else {
                    py_fmt(edge, 1)
                };
                out.push(json!({
                    "rule": "safe_area",
                    "element_ids": [idx],
                    "severity": "warn",
                    "bbox": el.get("bbox"),
                    "evidence": {
                        "canvas": [cw as i64, ch as i64],
                        "edge_margin": py_round_n(edge, 1),
                        "kind": "edge_text",
                    },
                    "suggested_cause": format!("文本元素距画布边缘仅 {fmt_edge}px（存在裁切/触边风险）"),
                }));
            }
        }
    }
    out
}

/// 运行规则引擎（对应 Python `main()`）。
pub fn run(report_path: &Path, p: &RulesParams) -> Result<Value, String> {
    let report: Value = serde_json::from_str(
        &std::fs::read_to_string(report_path)
            .map_err(|e| format!("cannot read {report_path:?}: {e}"))?,
    )
    .map_err(|e| format!("invalid JSON in {report_path:?}: {e}"))?;
    let els = extract_elements(&report);
    let canvas = canvas_of(&report, &p.canvas);

    let mut findings = Vec::new();
    findings.extend(check_contrast(&els));
    findings.extend(check_overlap(&els, p.overlap_threshold, p.min_overlap_area));
    findings.extend(check_alignment(&els, p.align_tol, p.align_drift));
    findings.extend(check_spacing(&els, p.spacing_k));
    findings.extend(check_safe_area(&els, canvas, p.margin));

    let mut report_out = crate::schema::envelope(
        Some("rules"),
        &report
            .get("sensors")
            .and_then(|s| s.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
            .unwrap_or_else(|| vec!["rules"]),
        report.get("coordsys").and_then(|c| c.as_str()),
        Some(json!({"type": "fused", "from": report_path.display().to_string()})),
    );
    report_out["rules"] = json!(rules());
    report_out["findings"] = json!(findings);
    let mut sev = json!({"critical": 0, "warn": 0, "info": 0});
    for f in &findings {
        if let Some(s) = f.get("severity").and_then(|v| v.as_str()) {
            let cur = sev.get(s).and_then(|v| v.as_i64()).unwrap_or(0);
            sev[s] = json!(cur + 1);
        }
    }
    let n = els.len().max(1) as f64;
    let score = 100.0
        * (1.0
            - (1.0 * sev["critical"].as_i64().unwrap_or(0) as f64
                + 0.5 * sev["warn"].as_i64().unwrap_or(0) as f64
                + 0.1 * sev["info"].as_i64().unwrap_or(0) as f64)
                / n);
    report_out["metrics"] = json!({
        "element_count": els.len(),
        "finding_count": findings.len(),
        "severity": sev,
        "design_score": py_round_n(score, 1),
    });
    Ok(report_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn el(bbox: [f64; 4], source: &str) -> Value {
        json!({"bbox": bbox, "source": [source], "text": "x"})
    }

    #[test]
    fn cluster_groups() {
        let v = vec![1.0, 2.0, 10.0, 11.0, 20.0];
        let g = cluster(&v, 2.0);
        assert_eq!(g.len(), 3);
        assert_eq!(g[0], vec![1.0, 2.0]);
    }

    #[test]
    fn is_designed_rules() {
        assert!(is_designed(&el([0.0, 0.0, 1.0, 1.0], "dom")));
        assert!(is_designed(
            &json!({"bbox": [0.0, 0.0, 1.0, 1.0], "font": "Arial"})
        ));
        assert!(!is_designed(&el([0.0, 0.0, 1.0, 1.0], "ocr")));
    }

    #[test]
    fn overlap_rule() {
        let els = vec![
            el([0.0, 0.0, 100.0, 100.0], "dom"),
            el([50.0, 50.0, 150.0, 150.0], "dom"),
        ];
        let f = check_overlap(&els, 0.01, 200.0);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0]["rule"], "overlap");
    }

    #[test]
    fn safe_area_off_canvas() {
        let els = vec![el([-5.0, 0.0, 50.0, 50.0], "dom")];
        let f = check_safe_area(&els, Some((100, 100)), 1.0);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0]["severity"], "critical");
    }
}

#[cfg(test)]
mod pyfmt_tests {
    use super::*;
    #[test]
    fn py_fmt_direct() {
        assert_eq!(py_fmt(120.0, 1), "120.0");
        assert_eq!(py_fmt(40.0, 1), "40.0");
        assert_eq!(py_fmt(7.0, 1), "7.0");
        assert_eq!(py_fmt(7.5, 1), "7.5");
    }
}
