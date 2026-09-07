//! 多轮反馈协议（对应 Python `vs_protocol.py`，纯 Rust，schema v3 输出）。
//!
//! SeeingEye 模式：三个动作驱动多轮视觉推理：
//!   analyze [image]      → 粗报告（saliency 候选 + ascii 粗栅格 + 全局统计 + 融合）
//!   zoom [region]        → 细报告（region 内 ocr 高倍 + segment 前景 + edge + depth）
//!   probe [bbox, sensor] → 单传感器定向证据

use std::path::Path;

use serde_json::{json, Value};

use crate::schema::{envelope, NOTATION_GUIDE};

/// 协议参数。
pub struct ProtocolParams {
    /// analyze | zoom | probe
    pub action: String,
    pub image: String,
    pub region: Option<String>,
    pub bbox: Option<String>,
    pub sensor: Option<String>,
}

/// analyze：粗报告（saliency 候选 + ascii 粗栅格 + scene_stats + ocr + 融合）。
fn analyze(image: &str) -> Result<Value, String> {
    let img = Path::new(image);
    let sal = crate::saliency::run(
        img,
        &crate::saliency::SaliencyParams {
            top_n: 5,
            min_score: 0.3,
        },
    );
    let ascii_r = crate::ascii::run(
        img,
        &crate::ascii::AsciiParams {
            cols: 64,
            rows: 36,
            color: false,
        },
    );
    let stats = crate::scene_stats::run(img, &crate::scene_stats::SceneStatsParams { colors: 8 });
    let ocr = crate::ocr::run(
        img,
        &crate::ocr::OcrParams {
            max_items: 60,
            min_conf: 0.5,
        },
    );

    // 融合（ocr + saliency 候选）
    let mut fusion: Option<Value> = None;
    if let (Ok(ocr_v), Ok(sal_v)) = (&ocr, &sal) {
        let dir = std::env::temp_dir();
        let mut paths = Vec::new();
        for (i, v) in [ocr_v, sal_v].iter().enumerate() {
            let path = dir.join(format!("vs_protocol_fusion_{i}.json"));
            let _ = std::fs::write(&path, serde_json::to_string(v).unwrap());
            paths.push(path);
        }
        fusion = crate::fusion::run(&paths, None).ok();
        for path in &paths {
            let _ = std::fs::remove_file(path);
        }
    }

    let mut report = envelope(
        Some("analyze"),
        &["saliency", "ascii", "scene_stats", "ocr", "fusion"],
        Some("image_px"),
        Some(json!({"type": "image", "path": image})),
    );
    report["candidates"] = sal
        .as_ref()
        .map(|v| v.get("candidates").cloned().unwrap_or(json!([])))
        .unwrap_or(json!([]));
    report["ascii"] = ascii_r
        .as_ref()
        .map(|v| v.get("ascii").cloned().unwrap_or(json!({})))
        .unwrap_or(json!({}));
    report["metrics"] = json!({
        "scene": stats.as_ref().map(|v| v.get("metrics").cloned().unwrap_or(json!({}))).unwrap_or(json!({})),
        "ocr_items": ocr.as_ref().map(|v| v.get("elements").and_then(|e| e.as_array()).map(|a| a.len()).unwrap_or(0)).unwrap_or(0),
        "saliency_candidates": report["candidates"].as_array().map(|a| a.len()).unwrap_or(0),
    });
    if let Some(f) = &fusion {
        report["findings"] = f.get("findings").cloned().unwrap_or(json!([]));
    }
    report["notation"] = json!(NOTATION_GUIDE);
    report["note"] =
        json!("粗报告：先用 candidates 定位有效内容，再用 zoom 放大细看，或 probe 定向取证。");
    Ok(report)
}

/// zoom：细报告（region 内 ocr 高倍 + segment 前景 + edge + depth）。
fn zoom(image: &str, region: &str) -> Result<Value, String> {
    let coords: Vec<i64> = region
        .split(',')
        .map(|v| v.trim().parse().unwrap_or(0))
        .collect();
    if coords.len() != 4 {
        return Err(format!("invalid region: {region}"));
    }
    let (x1, y1, x2, y2) = (coords[0], coords[1], coords[2], coords[3]);
    let img = Path::new(image);

    let ocr = crate::ocr::run(
        img,
        &crate::ocr::OcrParams {
            max_items: 40,
            min_conf: 0.5,
        },
    );
    let edge = crate::edge::run(
        img,
        &crate::edge::EdgeParams {
            sigma: 1.0,
            hough_threshold: 40,
            max_lines: 50,
            max_points: 2000,
        },
    );
    let depth = crate::depth::run(
        img,
        &crate::depth::DepthParams {
            bins: 3,
            region: Some(region.to_string()),
        },
    );
    let seg = crate::segment::run(
        img,
        &crate::segment::SegmentParams {
            points: Vec::new(),
            labels: Vec::new(),
            min_area: 0.0,
            max_instances: 10,
        },
    );

    let mut report = envelope(
        Some("zoom"),
        &["ocr", "edge", "depth", "segment"],
        Some("image_px"),
        Some(json!({
            "type": "image",
            "path": image,
            "region": [x1, y1, x2, y2],
        })),
    );
    let mut els: Vec<Value> = Vec::new();
    if let Ok(o) = &ocr {
        if let Some(e) = o.get("elements").and_then(|v| v.as_array()) {
            els.extend(e.iter().cloned());
        }
    }
    if let Ok(e) = &edge {
        if let Some(es) = e.get("elements").and_then(|v| v.as_array()) {
            els.extend(es.iter().cloned());
        }
    }
    report["elements"] = json!(els);
    report["foreground"] = seg
        .as_ref()
        .map(|v| v.get("foreground").cloned().unwrap_or(Value::Null))
        .unwrap_or(Value::Null);
    report["depth"] = depth
        .as_ref()
        .map(|v| v.get("depth").cloned().unwrap_or(json!({})))
        .unwrap_or(json!({}));
    report["metrics"] = json!({
        "ocr_items": ocr.as_ref().map(|v| v.get("elements").and_then(|e| e.as_array()).map(|a| a.len()).unwrap_or(0)).unwrap_or(0),
        "edge_points": edge.as_ref().map(|v| v.get("metrics").and_then(|m| m.get("edge_points")).and_then(|p| p.as_u64()).unwrap_or(0)).unwrap_or(0),
        "region": [x1, y1, x2, y2],
    });
    report["notation"] = json!(NOTATION_GUIDE);
    report["note"] = json!(
        "细报告：region 内 OCR 已 2× 放大；edge 给出亚像素边缘；depth 给出区域深度；foreground 给出前景分割。"
    );
    Ok(report)
}

/// 裁剪图像到 region 并保存临时文件（Python `--region` 语义：先裁剪再识别）。
fn crop_to_temp(image: &str, region: &str) -> Result<String, String> {
    let coords: Vec<i64> = region
        .split(',')
        .map(|v| v.trim().parse().unwrap_or(0))
        .collect();
    if coords.len() != 4 {
        return Err(format!("invalid region: {region}"));
    }
    let (x, y) = (
        coords[0].min(coords[2]) as u32,
        coords[1].min(coords[3]) as u32,
    );
    let (w, h) = (
        coords[2].abs_diff(coords[0]) as u32,
        coords[3].abs_diff(coords[1]) as u32,
    );
    let img = image::open(image).map_err(|e| format!("open {image}: {e}"))?;
    let cropped = image::imageops::crop_imm(&img, x, y, w, h).to_image();
    let tmp = std::env::temp_dir().join(format!("vs_protocol_crop_{}.png", std::process::id()));
    cropped.save(&tmp).map_err(|e| format!("save crop: {e}"))?;
    Ok(tmp.to_string_lossy().into_owned())
}

/// probe：单传感器定向证据。
fn probe(image: &str, bbox: &str, sensor: &str) -> Result<Value, String> {
    let coords: Vec<i64> = bbox
        .split(',')
        .map(|v| v.trim().parse().unwrap_or(0))
        .collect();
    if coords.len() != 4 {
        return Err(format!("invalid bbox: {bbox}"));
    }
    let (x1, y1, x2, y2) = (coords[0], coords[1], coords[2], coords[3]);
    let img = Path::new(image);

    let r: Result<Value, String> = match sensor {
        "ocr" => {
            // Python 语义：先裁剪 region 再识别，坐标偏移回原图
            let coords: Vec<i64> = bbox
                .split(',')
                .map(|v| v.trim().parse().unwrap_or(0))
                .collect();
            let (ox, oy) = (coords[0].min(coords[2]), coords[1].min(coords[3]));
            let cropped = crop_to_temp(image, bbox)?;
            let res = crate::ocr::run(
                Path::new(&cropped),
                &crate::ocr::OcrParams {
                    max_items: 20,
                    min_conf: 0.5,
                },
            );
            let _ = std::fs::remove_file(&cropped);
            // 偏移回原图坐标
            res.map(|mut v| {
                if let Some(els) = v.get_mut("elements").and_then(|e| e.as_array_mut()) {
                    for el in els {
                        if let Some(b) = el.get_mut("bbox").and_then(|b| b.as_array_mut()) {
                            if b.len() >= 4 {
                                b[0] = json!(b[0].as_f64().unwrap_or(0.0) + ox as f64);
                                b[1] = json!(b[1].as_f64().unwrap_or(0.0) + oy as f64);
                                b[2] = json!(b[2].as_f64().unwrap_or(0.0) + ox as f64);
                                b[3] = json!(b[3].as_f64().unwrap_or(0.0) + oy as f64);
                            }
                        }
                        if let Some(c) = el.get_mut("center").and_then(|c| c.as_array_mut()) {
                            if c.len() >= 2 {
                                c[0] = json!(c[0].as_f64().unwrap_or(0.0) + ox as f64);
                                c[1] = json!(c[1].as_f64().unwrap_or(0.0) + oy as f64);
                            }
                        }
                    }
                }
                v
            })
        }
        "edge" => crate::edge::run(
            img,
            &crate::edge::EdgeParams {
                sigma: 1.0,
                hough_threshold: 40,
                max_lines: 50,
                max_points: 2000,
            },
        ),
        "depth" => crate::depth::run(
            img,
            &crate::depth::DepthParams {
                bins: 3,
                region: Some(bbox.to_string()),
            },
        ),
        "segment" => crate::segment::run(
            img,
            &crate::segment::SegmentParams {
                points: Vec::new(),
                labels: Vec::new(),
                min_area: 0.0,
                max_instances: 10,
            },
        ),
        "pix" => crate::pix::run(
            img,
            &crate::pix::PixParams {
                colors: 8,
                regions: vec![bbox.to_string()],
                compare: None,
                wcag: Vec::new(),
                diff_threshold: 0,
            },
        ),
        other => Err(format!("unknown sensor: {other}")),
    };

    let mut report = envelope(
        Some("probe"),
        &[sensor],
        Some("image_px"),
        Some(json!({
            "type": "image",
            "path": image,
            "bbox": [x1, y1, x2, y2],
        })),
    );
    match r {
        Ok(v) => {
            report["elements"] = v.get("elements").cloned().unwrap_or(json!([]));
            report["metrics"] = v.get("metrics").cloned().unwrap_or(json!({}));
            if let Some(regions) = v.get("regions") {
                report["regions"] = regions.clone();
            }
            if let Some(depth) = v.get("depth") {
                report["depth"] = depth.clone();
            }
        }
        Err(e) => {
            report["error"] = json!(e);
        }
    }
    report["notation"] = json!(NOTATION_GUIDE);
    report["note"] = json!(format!("定向证据：{sensor} 传感器在 bbox 区域的结果。"));
    Ok(report)
}

/// 运行协议动作。
pub fn run(p: &ProtocolParams) -> Result<Value, String> {
    match p.action.as_str() {
        "analyze" => analyze(&p.image),
        "zoom" => {
            let region = p
                .region
                .clone()
                .ok_or_else(|| "zoom 需要 --region x1,y1,x2,y2".to_string())?;
            zoom(&p.image, &region)
        }
        "probe" => {
            let bbox = p
                .bbox
                .clone()
                .ok_or_else(|| "probe 需要 --bbox x1,y1,x2,y2".to_string())?;
            let sensor = p
                .sensor
                .clone()
                .ok_or_else(|| "probe 需要 --sensor ocr|edge|depth|segment|pix".to_string())?;
            probe(&p.image, &bbox, &sensor)
        }
        other => Err(format!("unknown action: {other}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_action_rejected() {
        let p = ProtocolParams {
            action: "bogus".into(),
            image: "x.png".into(),
            region: None,
            bbox: None,
            sensor: None,
        };
        assert!(run(&p).is_err());
    }

    #[test]
    fn zoom_requires_region() {
        let p = ProtocolParams {
            action: "zoom".into(),
            image: "x.png".into(),
            region: None,
            bbox: None,
            sensor: None,
        };
        assert!(run(&p).is_err());
    }

    #[test]
    fn probe_requires_bbox_sensor() {
        let p = ProtocolParams {
            action: "probe".into(),
            image: "x.png".into(),
            region: None,
            bbox: None,
            sensor: None,
        };
        assert!(run(&p).is_err());
    }

    #[test]
    fn invalid_region_rejected() {
        let p = ProtocolParams {
            action: "zoom".into(),
            image: "x.png".into(),
            region: Some("1,2,3".into()),
            bbox: None,
            sensor: None,
        };
        assert!(run(&p).is_err());
    }
}
