//! 元素融合层（plan3 E1）：多传感器 → 一份融合元素清单。
//!
//! 流水线：
//!   1. 跨传感器对齐：bbox IoU ≥ 阈值 判定同一元素（不同传感器元素两两比较）
//!   2. 属性合并：文本取 ocr（conf 高）、语义取 omniparser（caption）、类别取 detect
//!   3. 置信度仲裁：D-S 证据融合（复用 fusion.rs 的 mass_from_* + ds_combine_all）
//!   4. 统一编号：融合后元素重新编号 [0]...[N]
//!   5. 溯源：每元素带 sources: ["ocr", "omniparser"]
//!
//! 输出：schema v3 envelope + elements（id/bbox/center/text/caption/type/conf/sources/interactive）。
//! 目标：DeepSeek 只需读这一份清单——文本、语义、类别、置信度、来源全部合并。

use serde_json::{json, Value};

use crate::bbox::bbox_iou;
use crate::fusion::{ds_combine_all, mass_from_detect, mass_from_ocr};
use crate::schema::envelope;

/// 融合参数。
pub struct FusionParams {
    /// bbox IoU 阈值：≥ 此值判定为同一元素。
    pub min_iou: f64,
    /// 最低置信度：低于此值的元素过滤。
    pub min_conf: f64,
}

impl Default for FusionParams {
    fn default() -> Self {
        Self {
            min_iou: 0.5,
            min_conf: 0.0,
        }
    }
}

/// 融合中间元素（跨传感器合并前）。
#[derive(Clone)]
struct RawElement {
    /// 来源传感器名。
    source: String,
    /// bbox [x1, y1, x2, y2]。
    bbox: [f64; 4],
    /// 文本（ocr）。
    text: Option<String>,
    /// 语义描述（omniparser caption）。
    caption: Option<String>,
    /// 类别（detect）。
    class: Option<String>,
    /// 置信度（可能为 None）。
    conf: Option<f64>,
    /// 可交互（omniparser）。
    interactive: bool,
}

/// 从报告提取元素（兼容不同传感器输出结构）。
fn extract_elements(report: &Value) -> Vec<RawElement> {
    let mut out = Vec::new();
    let source = report
        .get("sensors")
        .and_then(|s| s.as_array())
        .and_then(|a| a.first())
        .and_then(|s| s.as_str())
        .unwrap_or("unknown")
        .to_string();
    let Some(elements) = report.get("elements").and_then(|e| e.as_array()) else {
        return out;
    };
    for e in elements {
        let Some(bbox) = e.get("bbox").and_then(|b| b.as_array()) else {
            continue;
        };
        if bbox.len() < 4 {
            continue;
        }
        let to_f64 = |v: &Value| v.as_f64().unwrap_or_else(|| v.as_i64().unwrap_or(0) as f64);
        let b = [
            to_f64(&bbox[0]),
            to_f64(&bbox[1]),
            to_f64(&bbox[2]),
            to_f64(&bbox[3]),
        ];
        let text = e.get("text").and_then(|t| t.as_str()).map(String::from);
        let caption = e
            .get("caption")
            .and_then(|t| t.as_str())
            .map(String::from)
            .or_else(|| {
                // omniparser 把语义放在 text 字段
                if source == "omniparser" {
                    text.clone()
                } else {
                    None
                }
            });
        let class = e.get("class").and_then(|t| t.as_str()).map(String::from);
        let conf = e.get("conf").and_then(|c| c.as_f64());
        let interactive = e
            .get("interactivity")
            .and_then(|i| i.as_bool())
            .unwrap_or(false);
        out.push(RawElement {
            source: source.clone(),
            bbox: b,
            text,
            caption,
            class,
            conf,
            interactive,
        });
    }
    out
}

/// 计算两个 bbox 的 IoU（bbox_iou 接受 &[f64]）。
fn iou(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    bbox_iou(a, b)
}

/// 中心点是否在 bbox 内。
fn center_inside(center: (f64, f64), b: &[f64; 4]) -> bool {
    center.0 >= b[0] && center.0 <= b[2] && center.1 >= b[1] && center.1 <= b[3]
}

/// 中心点坐标。
fn center_of(b: &[f64; 4]) -> (f64, f64) {
    ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
}

/// 对齐判据：IoU ≥ 阈值，或小框中心在大框内（ocr 文本框被 omniparser 图标框包含）。
fn aligned(a: &[f64; 4], b: &[f64; 4], min_iou: f64) -> bool {
    if iou(a, b) >= min_iou {
        return true;
    }
    // 单向中心包含：小框中心在大框内 → 同一元素（按钮 = 图标框 + 文本）
    let (ca, cb) = (center_of(a), center_of(b));
    let area = |b: &[f64; 4]| (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    if area(a) <= area(b) {
        center_inside(ca, b)
    } else {
        center_inside(cb, a)
    }
}

/// 融合后的元素。
#[derive(Clone)]
struct FusedElement {
    bbox: [f64; 4],
    text: Option<String>,
    caption: Option<String>,
    class: Option<String>,
    conf: f64,
    sources: Vec<String>,
    interactive: bool,
}

/// 元素融合主入口。
///
/// `reports`：多个传感器报告（schema v3 envelope，含 elements）。
pub fn run(reports: &[Value], p: &FusionParams) -> Result<Value, String> {
    // 1) 提取全部原始元素
    let mut raws: Vec<RawElement> = Vec::new();
    for r in reports {
        raws.extend(extract_elements(r));
    }
    if raws.is_empty() {
        return Ok(envelope(
            Some("fusion_elements"),
            &["element_fusion"],
            Some("image_px"),
            None,
        ));
    }

    // 2) 跨传感器对齐：贪心聚类（按 conf 降序，每个元素并入 IoU 最高的簇）
    //    簇 = 同一物理元素的多个传感器观测
    let mut clusters: Vec<Vec<RawElement>> = Vec::new();
    // 按 conf 降序（conf None 排最后）
    let mut sorted = raws.clone();
    sorted.sort_by(|a, b| {
        let ca = a.conf.unwrap_or(-1.0);
        let cb = b.conf.unwrap_or(-1.0);
        cb.partial_cmp(&ca).unwrap_or(std::cmp::Ordering::Equal)
    });
    for raw in sorted {
        let mut best: Option<(usize, f64)> = None;
        for (ci, cluster) in clusters.iter().enumerate() {
            // 与簇内所有元素的最大对齐度（IoU 或中心包含）
            let max_iou = cluster
                .iter()
                .map(|c| {
                    if aligned(&c.bbox, &raw.bbox, p.min_iou) {
                        1.0
                    } else {
                        0.0
                    }
                })
                .fold(0.0_f64, f64::max);
            if max_iou > 0.0 && best.is_none_or(|(_, b)| max_iou > b) {
                best = Some((ci, max_iou));
            }
        }
        match best {
            Some((ci, _)) => clusters[ci].push(raw),
            None => clusters.push(vec![raw]),
        }
    }

    // 3) 属性合并 + 置信度仲裁 + 编号
    let mut fused: Vec<FusedElement> = Vec::new();
    for cluster in &clusters {
        // bbox：取 conf 最高的观测（确定性）
        let rep = cluster
            .iter()
            .max_by(|a, b| {
                let ca = a.conf.unwrap_or(-1.0);
                let cb = b.conf.unwrap_or(-1.0);
                ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        // 文本：取 ocr 的 text（conf 高优先）
        let text = cluster
            .iter()
            .filter(|c| c.source == "ocr")
            .max_by(|a, b| {
                let ca = a.conf.unwrap_or(0.0);
                let cb = b.conf.unwrap_or(0.0);
                ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .and_then(|c| c.text.clone())
            .or_else(|| rep.text.clone());
        // 语义：取 omniparser 的 caption
        let caption = cluster
            .iter()
            .find(|c| c.source == "omniparser")
            .and_then(|c| c.caption.clone())
            .or_else(|| rep.caption.clone());
        // 类别：取 detect 的 class
        let class = cluster
            .iter()
            .find(|c| c.source == "detect")
            .and_then(|c| c.class.clone())
            .or_else(|| rep.class.clone());
        // 置信度仲裁：D-S 证据融合
        let masses: Vec<[f64; 3]> = cluster
            .iter()
            .map(|c| match c.source.as_str() {
                "ocr" => mass_from_ocr(c.conf.unwrap_or(0.5)),
                "detect" => mass_from_detect(c.conf.unwrap_or(0.5)),
                // omniparser 无 conf：检测到元素本身是弱证据（0.5 中性）
                _ => mass_from_detect(c.conf.unwrap_or(0.5)),
            })
            .collect();
        let m = ds_combine_all(&masses);
        let conf = m[0]; // consistent mass = 融合置信度
        let interactive = cluster.iter().any(|c| c.interactive);
        let sources: Vec<String> = {
            let mut s: Vec<String> = cluster.iter().map(|c| c.source.clone()).collect();
            s.sort();
            s.dedup();
            s
        };
        fused.push(FusedElement {
            bbox: rep.bbox,
            text,
            caption,
            class,
            conf,
            sources,
            interactive,
        });
    }

    // 4) 过滤低置信度
    fused.retain(|e| e.conf >= p.min_conf);

    // 5) 输出 schema v3 envelope
    let mut report = envelope(
        Some("fusion_elements"),
        &["element_fusion"],
        Some("image_px"),
        None,
    );
    let elements: Vec<Value> = fused
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let cx = ((e.bbox[0] + e.bbox[2]) / 2.0).round() as i64;
            let cy = ((e.bbox[1] + e.bbox[3]) / 2.0).round() as i64;
            let conf = (e.conf * 1000.0).round() / 1000.0;
            // 存在性断言（plan3 E6）：conf ≥ 0.7 为"已确认"，否则"推测"
            let status = if conf >= 0.7 { "已确认" } else { "推测" };
            let assertion = format!(
                "此元素由 {} 检测，conf {conf}（{status}）",
                e.sources.join(" + ")
            );
            json!({
                "id": i,
                "bbox": e.bbox,
                "center": [cx, cy],
                "text": e.text,
                "caption": e.caption,
                "class": e.class,
                "type": if e.interactive { "interactive" } else { "element" },
                "conf": conf,
                "assertion": assertion,
                "sources": e.sources,
                "interactive": e.interactive,
                "coordsys": "image_px",
            })
        })
        .collect();
    report["elements"] = json!(elements);
    report["metrics"] = json!({
        "input_elements": raws.len(),
        "fused_elements": fused.len(),
        "dedup_ratio": if raws.is_empty() { 0.0 } else {
            (raws.len() as f64 - fused.len() as f64) / raws.len() as f64
        },
    });
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ocr_report(elements: Vec<Value>) -> Value {
        json!({
            "schema": "vision-report/v3",
            "sensors": ["ocr"],
            "coordsys": "image_px",
            "elements": elements,
        })
    }

    fn omniparser_report(elements: Vec<Value>) -> Value {
        json!({
            "schema": "vision-report/v3",
            "sensors": ["omniparser"],
            "coordsys": "image_px",
            "elements": elements,
        })
    }

    #[test]
    fn cross_sensor_alignment_merges_overlap() {
        // ocr 检测 btn（小框），omniparser 检测同一区域（大框，IoU 0.8）
        let ocr = ocr_report(vec![json!({
            "bbox": [100.0, 100.0, 200.0, 150.0],
            "text": "btn1", "conf": 0.99, "type": "text",
        })]);
        let omni = omniparser_report(vec![json!({
            "bbox": [90.0, 95.0, 210.0, 155.0],
            "text": "a button", "interactivity": true,
        })]);
        let out = run(&[ocr, omni], &FusionParams::default()).unwrap();
        let els = out["elements"].as_array().unwrap();
        assert_eq!(els.len(), 1, "重叠元素应合并为 1");
        let e = &els[0];
        assert_eq!(e["text"], "btn1", "文本取 ocr");
        assert_eq!(e["caption"], "a button", "语义取 omniparser");
        assert_eq!(e["sources"], json!(["ocr", "omniparser"]), "溯源");
        assert_eq!(e["interactive"], true, "interactivity 合并");
        assert!(e["conf"].as_f64().unwrap() > 0.85, "D-S 融合置信度");
    }

    #[test]
    fn non_overlapping_elements_stay_separate() {
        let ocr = ocr_report(vec![
            json!({"bbox": [0.0, 0.0, 50.0, 50.0], "text": "a", "conf": 0.9}),
            json!({"bbox": [200.0, 200.0, 250.0, 250.0], "text": "b", "conf": 0.9}),
        ]);
        let out = run(&[ocr], &FusionParams::default()).unwrap();
        let els = out["elements"].as_array().unwrap();
        assert_eq!(els.len(), 2, "不重叠元素保持独立");
        assert_eq!(els[0]["id"], 0);
        assert_eq!(els[1]["id"], 1, "编号连续");
    }

    #[test]
    fn ds_fusion_single_conf() {
        // 同一元素被 ocr + detect 检测 → 单一融合 conf
        let ocr = ocr_report(vec![json!({
            "bbox": [100.0, 100.0, 200.0, 150.0], "text": "btn", "conf": 0.8,
        })]);
        let det = json!({
            "schema": "vision-report/v3",
            "sensors": ["detect"],
            "coordsys": "image_px",
            "elements": [json!({
                "bbox": [105.0, 100.0, 200.0, 150.0], "class": "button", "conf": 0.7,
            })],
        });
        let out = run(&[ocr, det], &FusionParams::default()).unwrap();
        let els = out["elements"].as_array().unwrap();
        assert_eq!(els.len(), 1);
        let conf = els[0]["conf"].as_f64().unwrap();
        assert!(conf > 0.8, "D-S 融合应提升置信度: {conf}");
        assert_eq!(els[0]["class"], "button", "类别取 detect");
    }

    #[test]
    fn min_conf_filters_low_confidence() {
        let ocr = ocr_report(vec![
            json!({"bbox": [0.0, 0.0, 50.0, 50.0], "text": "a", "conf": 0.9}),
            json!({"bbox": [200.0, 200.0, 250.0, 250.0], "text": "b", "conf": 0.3}),
        ]);
        let out = run(
            &[ocr],
            &FusionParams {
                min_conf: 0.5,
                ..FusionParams::default()
            },
        )
        .unwrap();
        let els = out["elements"].as_array().unwrap();
        assert_eq!(els.len(), 1, "低置信度元素被过滤");
        assert_eq!(els[0]["text"], "a");
    }

    #[test]
    fn empty_inputs_ok() {
        let out = run(&[], &FusionParams::default()).unwrap();
        assert_eq!(out["elements"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn assertion_high_conf_confirmed() {
        // E6 门禁：conf ≥ 0.7 → "已确认"
        let ocr = ocr_report(vec![json!({
            "bbox": [100.0, 100.0, 200.0, 150.0],
            "text": "btn1", "conf": 0.9, "type": "text",
        })]);
        let out = run(&[ocr], &FusionParams::default()).unwrap();
        let e = &out["elements"].as_array().unwrap()[0];
        let a = e["assertion"].as_str().unwrap();
        assert!(a.contains("已确认"), "conf 0.9 应为已确认: {a}");
        assert!(a.contains("ocr"), "断言应含来源: {a}");
        // D-S 融合会映射 conf（0.9 → 0.72），断言应含融合后 conf 值
        let conf = e["conf"].as_f64().unwrap();
        assert!(
            a.contains(&format!("{conf}")),
            "断言应含融合后 conf {conf}: {a}"
        );
    }

    #[test]
    fn assertion_low_conf_speculative() {
        // E6 门禁：conf < 0.7 → "推测"
        let ocr = ocr_report(vec![json!({
            "bbox": [100.0, 100.0, 200.0, 150.0],
            "text": "btn1", "conf": 0.5, "type": "text",
        })]);
        let out = run(&[ocr], &FusionParams::default()).unwrap();
        let e = &out["elements"].as_array().unwrap()[0];
        let a = e["assertion"].as_str().unwrap();
        assert!(a.contains("推测"), "conf 0.5 应为推测: {a}");
    }
}
