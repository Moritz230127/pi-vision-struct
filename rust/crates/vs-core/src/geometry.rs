//! 几何原语传感器（对应 Python `vs_geometry.py`，schema v3 输出逐字段一致）。
//!
//! VTracer 光栅 → SVG → 解析为形状原语（矩形/圆/椭圆/线段/多边形），
//! 供纯文本 LLM 直接推理几何结构。VTracer 本身就是 Rust crate，内联调用。

use std::path::Path;

use regex::Regex;
use serde_json::{json, Value};
use vtracer::{ColorImage, Config, FitMode, Session};

use crate::image::load_rgb;

/// 几何参数（对应 Python `--mode/--max-shapes`）。
pub struct GeometryParams {
    pub mode: String, // "polygon" | "spline"
    pub max_shapes: u32,
}

/// 解析 SVG 标签属性。
fn parse_attrs(tag: &str) -> std::collections::HashMap<String, String> {
    let re = Regex::new(r#"(\w[\w-]*)\s*=\s*"([^"]*)""#).unwrap();
    re.captures_iter(tag)
        .map(|c| (c[1].to_string(), c[2].to_string()))
        .collect()
}

/// SVG → 形状原语列表（对应 Python `svg_to_primitives`）。
fn svg_to_primitives(svg: &str, max_shapes: u32) -> Vec<Value> {
    let mut primitives: Vec<Value> = Vec::new();
    let f = |s: &str| s.parse::<f64>().unwrap_or(0.0);

    // <rect>
    let re_rect = Regex::new(r#"<rect[^>]*>"#).unwrap();
    for m in re_rect.find_iter(svg) {
        let attrs = parse_attrs(m.as_str());
        if let (Some(x), Some(y), Some(w), Some(h)) = (
            attrs.get("x"),
            attrs.get("y"),
            attrs.get("width"),
            attrs.get("height"),
        ) {
            let (x, y, w, h) = (f(x), f(y), f(w), f(h));
            primitives.push(json!({
                "type": "rect", "bbox": [x, y, x + w, y + h],
                "width": w, "height": h, "fill": attrs.get("fill"),
            }));
        }
    }
    // <circle>
    let re_circle = Regex::new(r#"<circle[^>]*>"#).unwrap();
    for m in re_circle.find_iter(svg) {
        let attrs = parse_attrs(m.as_str());
        if let (Some(cx), Some(cy), Some(r)) = (attrs.get("cx"), attrs.get("cy"), attrs.get("r")) {
            let (cx, cy, r) = (f(cx), f(cy), f(r));
            primitives.push(json!({
                "type": "circle", "bbox": [cx - r, cy - r, cx + r, cy + r],
                "center": [cx, cy], "radius": r, "fill": attrs.get("fill"),
            }));
        }
    }
    // <ellipse>
    let re_ellipse = Regex::new(r#"<ellipse[^>]*>"#).unwrap();
    for m in re_ellipse.find_iter(svg) {
        let attrs = parse_attrs(m.as_str());
        if let (Some(cx), Some(cy), Some(rx), Some(ry)) = (
            attrs.get("cx"),
            attrs.get("cy"),
            attrs.get("rx"),
            attrs.get("ry"),
        ) {
            let (cx, cy, rx, ry) = (f(cx), f(cy), f(rx), f(ry));
            primitives.push(json!({
                "type": "ellipse", "bbox": [cx - rx, cy - ry, cx + rx, cy + ry],
                "center": [cx, cy], "rx": rx, "ry": ry, "fill": attrs.get("fill"),
            }));
        }
    }
    // <line>
    let re_line = Regex::new(r#"<line[^>]*>"#).unwrap();
    for m in re_line.find_iter(svg) {
        let attrs = parse_attrs(m.as_str());
        if ["x1", "y1", "x2", "y2"]
            .iter()
            .all(|k| attrs.contains_key(*k))
        {
            let (x1, y1, x2, y2) = (
                f(&attrs["x1"]),
                f(&attrs["y1"]),
                f(&attrs["x2"]),
                f(&attrs["y2"]),
            );
            primitives.push(json!({
                "type": "line",
                "bbox": [x1.min(x2), y1.min(y2), x1.max(x2), y1.max(y2)],
                "p1": [x1, y1], "p2": [x2, y2],
            }));
        }
    }
    // <path>（VTracer 主输出：M/L 多边形 + transform translate）
    let re_path = Regex::new(r#"<path[^>]*>"#).unwrap();
    let re_d = Regex::new(r#"d="([^"]*)""#).unwrap();
    let re_translate = Regex::new(r#"translate\(([-\d.e]+)[, ]+([-\d.e]+)\)"#).unwrap();
    let re_ml = Regex::new(r#"([ML])\s*([-\d.e]+)[, ]+([-\d.e]+)"#).unwrap();
    for m in re_path.find_iter(svg) {
        let tag = m.as_str();
        let Some(dm) = re_d.captures(tag) else {
            continue;
        };
        let d = &dm[1];
        let (mut tx, mut ty) = (0.0f64, 0.0f64);
        if let Some(tm) = re_translate.captures(tag) {
            tx = f(&tm[1]);
            ty = f(&tm[2]);
        }
        let mut pts: Vec<(f64, f64)> = Vec::new();
        for pm in re_ml.captures_iter(d) {
            pts.push((f(&pm[2]) + tx, f(&pm[3]) + ty));
        }
        if pts.len() >= 3 {
            let xs: Vec<f64> = pts.iter().map(|p| p.0).collect();
            let ys: Vec<f64> = pts.iter().map(|p| p.1).collect();
            primitives.push(json!({
                "type": "polygon",
                "bbox": [
                    xs.iter().cloned().fold(f64::INFINITY, f64::min),
                    ys.iter().cloned().fold(f64::INFINITY, f64::min),
                    xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                    ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                ],
                "points": pts.len(),
                "fill": parse_attrs(tag).get("fill"),
            }));
        } else if pts.len() == 2 {
            primitives.push(json!({
                "type": "line",
                "bbox": [pts[0].0.min(pts[1].0), pts[0].1.min(pts[1].1), pts[0].0.max(pts[1].0), pts[0].1.max(pts[1].1)],
                "p1": [pts[0].0, pts[0].1], "p2": [pts[1].0, pts[1].1],
            }));
        }
    }
    primitives.truncate(max_shapes as usize);
    primitives
}

/// 运行几何传感器（对应 Python `main()`）。
pub fn run(path: &Path, p: &GeometryParams) -> Result<Value, String> {
    let img = load_rgb(path)?;
    let (w, h) = (img.width as usize, img.height as usize);
    // 构造 vtracer ColorImage（RGBA）
    let mut cimg = ColorImage::new_w_h(w, h);
    for (i, px) in img.data.as_chunks::<3>().0.iter().enumerate() {
        cimg.pixels[i * 4..i * 4 + 4].copy_from_slice(&[px[0], px[1], px[2], 255]);
    }
    // Config：mode 由参数决定，其余默认（与 Python 绑定一致）
    let mode = if p.mode == "spline" {
        FitMode::Spline
    } else {
        FitMode::Polygon
    };
    let cfg = Config {
        mode,
        ..Config::default()
    };
    let mut session = Session::new(cimg);
    let svg = session
        .render_svg(&cfg)
        .map_err(|e| format!("vtracer failed: {e}"))?;

    let primitives = svg_to_primitives(&svg, p.max_shapes);
    let elements: Vec<Value> = primitives
        .iter()
        .enumerate()
        .map(|(i, prim)| {
            let bbox: Vec<f64> = prim
                .get("bbox")
                .and_then(|b| b.as_array())
                .map(|b| b.iter().filter_map(|v| v.as_f64()).collect())
                .unwrap_or_default();
            let mut el = serde_json::to_value(crate::schema::Element {
                id: i as i64,
                etype: prim.get("type").and_then(|t| t.as_str()).unwrap_or("shape"),
                bbox: &bbox,
                text: None,
                conf: Some(1.0),
                color: None,
                font: None,
                z: None,
                source: &["geometry"],
                coordsys: "image_px",
            })
            .unwrap();
            el["shape"] = prim.clone();
            el
        })
        .collect();
    let mut report = crate::schema::envelope(
        Some("geometry"),
        &["geometry"],
        Some("image_px"),
        Some(json!({
            "type": "image",
            "path": path.display().to_string(),
            "size_px": [w, h],
            "mode": p.mode,
        })),
    );
    report["elements"] = json!(elements);
    let mut types: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for prim in &primitives {
        let t = prim
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("shape")
            .to_string();
        *types.entry(t).or_insert(0) += 1;
    }
    report["metrics"] = json!({
        "shapes": primitives.len(),
        "types": types,
    });
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_rect() {
        let svg = r##"<svg><rect x="10" y="20" width="30" height="40" fill="#FF0000"/></svg>"##;
        let p = svg_to_primitives(svg, 50);
        assert_eq!(p.len(), 1);
        assert_eq!(p[0]["type"], "rect");
        assert_eq!(p[0]["bbox"], json!([10.0, 20.0, 40.0, 60.0]));
        assert_eq!(p[0]["fill"], "#FF0000");
    }

    #[test]
    fn parse_circle() {
        let svg = r#"<svg><circle cx="50" cy="50" r="10"/></svg>"#;
        let p = svg_to_primitives(svg, 50);
        assert_eq!(p[0]["type"], "circle");
        assert_eq!(p[0]["bbox"], json!([40.0, 40.0, 60.0, 60.0]));
    }

    #[test]
    fn parse_polygon_path() {
        let svg = r##"<svg><path d="M 0 0 L 10 0 L 10 10 L 0 10 Z" fill="#000"/></svg>"##;
        let p = svg_to_primitives(svg, 50);
        assert_eq!(p[0]["type"], "polygon");
        assert_eq!(p[0]["points"], 4);
        assert_eq!(p[0]["bbox"], json!([0.0, 0.0, 10.0, 10.0]));
    }

    #[test]
    fn parse_path_translate() {
        let svg = r#"<svg><path transform="translate(5, 7)" d="M 0 0 L 10 0 L 10 10 Z"/></svg>"#;
        let p = svg_to_primitives(svg, 50);
        assert_eq!(p[0]["type"], "polygon");
        assert_eq!(p[0]["bbox"], json!([5.0, 7.0, 15.0, 17.0]));
    }
}
