//! 元素级几何/样式审计（对应 Python `vs_audit.py`，schema v3 输出逐字段一致）。
//!
//! 对任意元素列表（DOM 元素 / PPTX 形状）计算确定性缺陷：
//! 两两重叠 / 出界 / WCAG 对比度不达标。

use std::path::Path;

use serde_json::{json, Value};

use crate::bbox::{bbox_area, bbox_intersection, bbox_iou};
use crate::color::{contrast_ratio, parse_hex};
use crate::util::py_round_n;

/// 审计参数（对应 Python CLI 参数）。
pub struct AuditParams {
    pub canvas: Option<String>,
    pub overlap_threshold: f64,
    pub min_overlap_area: f64,
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

/// 元素首个文本色（texts[].color 或 color.text）。
fn first_text_color(el: &Value) -> Option<(String, Option<f64>)> {
    if let Some(texts) = el.get("texts").and_then(|t| t.as_array()) {
        for t in texts {
            if let Some(c) = t.get("color").and_then(|c| c.as_str()) {
                return Some((c.to_string(), t.get("size_pt").and_then(|v| v.as_f64())));
            }
        }
    }
    let color = el.get("color");
    color
        .and_then(|c| c.get("text"))
        .and_then(|t| t.as_str())
        .map(|t| (t.to_string(), None))
}

/// 元素标签（name/text/type 或 fallback），截断 60 字符。
fn label(el: &Value, fallback: &str) -> String {
    let s = el
        .get("name")
        .and_then(|v| v.as_str())
        .or_else(|| el.get("text").and_then(|v| v.as_str()))
        .or_else(|| el.get("type").and_then(|v| v.as_str()))
        .unwrap_or(fallback);
    s.chars().take(60).collect()
}

/// 运行审计（对应 Python `main()`）。
pub fn run(report_path: &Path, p: &AuditParams) -> Result<Value, String> {
    let report: Value = serde_json::from_str(
        &std::fs::read_to_string(report_path)
            .map_err(|e| format!("cannot read {report_path:?}: {e}"))?,
    )
    .map_err(|e| format!("invalid JSON in {report_path:?}: {e}"))?;
    let els = extract_elements(&report);
    let mut anomalies: Vec<Value> = Vec::new();
    // 画布：参数或报告自带
    let canvas: Option<(i64, i64)> = p
        .canvas
        .as_deref()
        .and_then(|c| {
            let parts: Vec<&str> = c.split('x').collect();
            if parts.len() == 2 {
                Some((parts[0].parse().ok()?, parts[1].parse().ok()?))
            } else {
                None
            }
        })
        .or_else(|| {
            report
                .get("source")
                .and_then(|s| s.get("slide_size_pt"))
                .and_then(|v| v.as_array())
                .and_then(|v| Some((v.first()?.as_f64()? as i64, v.get(1)?.as_f64()? as i64)))
        });

    // 1) 两两重叠
    for i in 0..els.len() {
        for j in (i + 1)..els.len() {
            let (a, b) = (&els[i], &els[j]);
            let (Some(ba), Some(bb)) = (
                a.get("bbox").and_then(|v| v.as_array()),
                b.get("bbox").and_then(|v| v.as_array()),
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
            if ia < p.min_overlap_area {
                continue;
            }
            let iou = bbox_iou(&ba, &bb);
            if iou > p.overlap_threshold {
                anomalies.push(crate::schema::anomaly(
                    "element_overlap",
                    &[
                        inter[0] as f64,
                        inter[1] as f64,
                        inter[2] as f64,
                        inter[3] as f64,
                    ],
                    json!({
                        "iou": py_round_n(iou, 2),
                        "area": ia.round(),
                        "a": label(a, &format!("el{i}")),
                        "b": label(b, &format!("el{j}")),
                    }),
                    py_round_n((0.99f64).min(iou + 0.5), 2),
                    None,
                ));
            }
        }
    }

    // 2) 出界
    if let Some((cw, ch)) = canvas {
        for el in &els {
            let Some(b) = el.get("bbox").and_then(|v| v.as_array()) else {
                continue;
            };
            let vals: Vec<f64> = b.iter().filter_map(|v| v.as_f64()).collect();
            if vals.len() < 4 {
                continue;
            }
            let (x1, y1, x2, y2) = (vals[0], vals[1], vals[2], vals[3]);
            if x1 < -1.0 || y1 < -1.0 || x2 > cw as f64 + 1.0 || y2 > ch as f64 + 1.0 {
                anomalies.push(crate::schema::anomaly(
                    "off_canvas",
                    &vals,
                    json!({"canvas": [cw, ch], "name": label(el, "?")}),
                    0.95,
                    None,
                ));
            }
        }
    }

    // 3) 对比度
    for el in &els {
        let Some((fg, _)) = first_text_color(el) else {
            continue;
        };
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
        if fill == "BACKGROUND (5)" || fill == "SOLID (1)" {
            continue;
        }
        let (Ok(fgrgb), Ok(fillrgb)) = (parse_hex(&fg), parse_hex(&fill)) else {
            continue;
        };
        let ratio = contrast_ratio(
            [fgrgb[0] as f64, fgrgb[1] as f64, fgrgb[2] as f64],
            [fillrgb[0] as f64, fillrgb[1] as f64, fillrgb[2] as f64],
        );
        if ratio < 4.5 {
            anomalies.push(crate::schema::anomaly(
                "contrast_fail",
                &el.get("bbox")
                    .and_then(|v| v.as_array())
                    .map(|b| b.iter().filter_map(|v| v.as_f64()).collect::<Vec<_>>())
                    .unwrap_or_default(),
                json!({
                    "fg": fg,
                    "bg": fill,
                    "ratio": py_round_n(ratio, 2),
                    "required_aa": 4.5,
                    "name": label(el, "?"),
                }),
                1.0,
                None,
            ));
        }
    }

    let mut report_out = crate::schema::envelope(
        Some("audit"),
        &report
            .get("sensors")
            .and_then(|s| s.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
            .unwrap_or_else(|| vec!["audit"]),
        report.get("coordsys").and_then(|c| c.as_str()),
        Some(json!({"type": "fused", "from": report_path.display().to_string()})),
    );
    report_out["anomalies"] = json!(anomalies);
    report_out["notation"] = json!(crate::schema::NOTATION_GUIDE);
    report_out["metrics"] = json!({
        "element_count": els.len(),
        "anomaly_count": anomalies.len(),
    });
    Ok(report_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn el(bbox: [f64; 4], fill: &str, text: &str) -> Value {
        json!({
            "bbox": bbox,
            "fill": fill,
            "text": text,
            "color": {"text": text},
        })
    }

    #[test]
    fn overlap_detected() {
        let report = json!({
            "elements": [
                el([0.0, 0.0, 100.0, 100.0], "#FFFFFF", "a"),
                el([50.0, 50.0, 150.0, 150.0], "#000000", "b"),
            ],
            "sensors": ["test"],
        });
        let path = std::env::temp_dir().join("audit_test.json");
        std::fs::write(&path, report.to_string()).unwrap();
        let r = run(
            &path,
            &AuditParams {
                canvas: None,
                overlap_threshold: 0.05,
                min_overlap_area: 400.0,
            },
        )
        .unwrap();
        let anoms = r["anomalies"].as_array().unwrap();
        assert_eq!(anoms.len(), 1);
        assert_eq!(anoms[0]["type"], "element_overlap");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn off_canvas_detected() {
        let report = json!({
            "elements": [el([-10.0, 0.0, 50.0, 50.0], "#FFFFFF", "a")],
            "sensors": ["test"],
        });
        let path = std::env::temp_dir().join("audit_test2.json");
        std::fs::write(&path, report.to_string()).unwrap();
        let r = run(
            &path,
            &AuditParams {
                canvas: Some("100x100".to_string()),
                overlap_threshold: 0.05,
                min_overlap_area: 400.0,
            },
        )
        .unwrap();
        let anoms = r["anomalies"].as_array().unwrap();
        assert_eq!(anoms.len(), 1);
        assert_eq!(anoms[0]["type"], "off_canvas");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn contrast_fail_detected() {
        // 白字白底 → 对比度 1.0 < 4.5
        let report = json!({
            "elements": [el([0.0, 0.0, 50.0, 50.0], "#FFFFFF", "#FFFFFF")],
            "sensors": ["test"],
        });
        let path = std::env::temp_dir().join("audit_test3.json");
        std::fs::write(&path, report.to_string()).unwrap();
        let r = run(
            &path,
            &AuditParams {
                canvas: None,
                overlap_threshold: 0.05,
                min_overlap_area: 400.0,
            },
        )
        .unwrap();
        let anoms = r["anomalies"].as_array().unwrap();
        assert_eq!(anoms.len(), 1);
        assert_eq!(anoms[0]["type"], "contrast_fail");
        std::fs::remove_file(&path).ok();
    }
}
