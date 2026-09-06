//! schema v3 统一元素模型 + 报告信封 + 视觉原语记法。
//!
//! 与 Python `vs_schema.py` 输出逐字段一致：
//! `element = {id, type, bbox[x1,y1,x2,y2], text, conf, color{fill,text}, font, z,
//!             source:[传感器名], coordsys}`

use serde_json::{json, Value};

/// schema 版本标识（与 Python `SCHEMA` 一致）。
pub const SCHEMA: &str = "vision-report/v3";

/// DeepSeek 式坐标锚定推理的记法指南（随报告输出）。
pub const NOTATION_GUIDE: &str = "primitive notation: in reasoning chains, reference locations as \
[bbox: x1,y1,x2,y2] or [point: x,y]; never use relative phrases like \
'the element on the left'. Coordinates follow this report's coordsys.";

/// 统一元素（对应 Python `element()` 输出，逐字段一致）。
///
/// `bbox` 原样序列化（不取整）；`source` 为空时输出 `[]`。
#[derive(serde::Serialize)]
pub struct Element<'a> {
    pub id: i64,
    #[serde(rename = "type")]
    pub etype: &'a str,
    pub bbox: &'a [f64],
    pub text: Option<&'a str>,
    pub conf: Option<f64>,
    pub color: Option<Value>,
    pub font: Option<&'a str>,
    pub z: Option<i64>,
    pub source: &'a [&'a str],
    pub coordsys: &'a str,
}

/// 构造报告信封（对应 Python `envelope()`）：空容器 + 元数据。
pub fn envelope(
    task: Option<&str>,
    sensors: &[&str],
    coordsys: Option<&str>,
    source: Option<Value>,
) -> Value {
    json!({
        "schema": SCHEMA,
        "task": task,
        "sensors": sensors,
        "coordsys": coordsys,
        "source": source.unwrap_or_else(|| json!({})),
        "elements": [],
        "findings": [],
        "candidates": [],
        "foreground": Value::Null,
        "metrics": {},
        "truncated": false,
    })
}

/// DeepSeek "Thinking with Visual Primitives" 记法：`[bbox: x1,y1,x2,y2]`。
///
/// 坐标经 Python 语义取整（银行家舍入）后拼接。
pub fn bbox_primitive(bbox: &[f64]) -> String {
    let parts: Vec<String> = bbox
        .iter()
        .map(|v| crate::util::py_round(*v).to_string())
        .collect();
    format!("[bbox: {}]", parts.join(", "))
}

/// DeepSeek "Thinking with Visual Primitives" 记法：`[point: x,y]`。
pub fn point_primitive(x: f64, y: f64) -> String {
    format!(
        "[point: {}, {}]",
        crate::util::py_round(x),
        crate::util::py_round(y)
    )
}

/// 统一异常项构造（唯一出口）：type + bbox + primitive 原语记法 + 证据。
///
/// `suggested_cause` 为 `Some` 时才输出该字段（与 Python 一致）。
pub fn anomaly(
    atype: &str,
    bbox: &[f64],
    evidence: Value,
    confidence: f64,
    suggested_cause: Option<&str>,
) -> Value {
    let mut a = json!({
        "type": atype,
        "bbox": bbox,
        "primitive": bbox_primitive(bbox),
        "confidence": confidence,
        "evidence": evidence,
    });
    if let Some(cause) = suggested_cause {
        a["suggested_cause"] = Value::String(cause.to_string());
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn element_shape() {
        let el = serde_json::to_value(Element {
            id: 1,
            etype: "text",
            bbox: &[1.0, 2.0, 3.0, 4.0],
            text: Some("hi"),
            conf: Some(0.9),
            color: None,
            font: None,
            z: None,
            source: &["ocr"],
            coordsys: "image_px",
        })
        .unwrap();
        assert_eq!(el["id"], 1);
        assert_eq!(el["type"], "text");
        assert_eq!(el["bbox"], json!([1.0, 2.0, 3.0, 4.0]));
        assert_eq!(el["text"], "hi");
        assert_eq!(el["source"], json!(["ocr"]));
        assert_eq!(el["coordsys"], "image_px");
        assert!(el["color"].is_null());
    }

    #[test]
    fn envelope_shape() {
        let env = envelope(Some("test"), &["pix"], Some("css_px"), None);
        assert_eq!(env["schema"], "vision-report/v3");
        assert_eq!(env["sensors"], json!(["pix"]));
        assert_eq!(env["source"], json!({}));
        assert_eq!(env["elements"], json!([]));
        assert!(env["foreground"].is_null());
        assert_eq!(env["truncated"], false);
    }

    #[test]
    fn primitives() {
        assert_eq!(bbox_primitive(&[1.0, 2.5, 3.0, 4.5]), "[bbox: 1, 2, 3, 4]");
        assert_eq!(point_primitive(1.5, 2.5), "[point: 2, 2]");
    }

    #[test]
    fn anomaly_with_and_without_cause() {
        let a = anomaly(
            "overlap",
            &[0.0, 0.0, 10.0, 10.0],
            json!({"iou": 0.5}),
            0.8,
            Some("too close"),
        );
        assert_eq!(a["type"], "overlap");
        assert_eq!(a["primitive"], "[bbox: 0, 0, 10, 10]");
        assert_eq!(a["suggested_cause"], "too close");
        let b = anomaly("overlap", &[0.0, 0.0, 10.0, 10.0], json!({}), 0.5, None);
        assert!(b.get("suggested_cause").is_none());
    }
}
