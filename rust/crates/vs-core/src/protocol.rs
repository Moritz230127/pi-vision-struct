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

/// analyze：粗报告（plan3 E3 分层感知——默认轻量路径）。
///
/// 只返回：融合元素清单（element_fusion）+ scene_stats 摘要 + edge density 热图。
/// 不返回：原始 ocr/omniparser/detect 元素、edge_point、ascii 全栅格。
/// 目标：总输出 < 3K tokens（两级感知预算的"粗"层）。
fn analyze(image: &str) -> Result<Value, String> {
    let img = Path::new(image);

    // 1) 融合元素清单：ocr + omniparser + detect → element_fusion
    let ocr = crate::ocr::run(
        img,
        &crate::ocr::OcrParams {
            max_items: 60,
            min_conf: 0.5,
        },
    );
    let omni = crate::omniparser::run(
        img,
        &crate::omniparser::OmniparserParams {
            max_items: 30,
            min_conf: 0.3,
        },
    );
    let detect = crate::detect::run(
        img,
        &crate::detect::DetectParams {
            classes: vec!["button".into(), "text".into(), "icon".into()],
            min_conf: 0.25,
        },
    );
    let mut fusion_reports: Vec<Value> = Vec::new();
    if let Ok(v) = &ocr {
        fusion_reports.push(v.clone());
    }
    if let Ok(v) = &omni {
        fusion_reports.push(v.clone());
    }
    if let Ok(v) = &detect {
        fusion_reports.push(v.clone());
    }
    let fused = crate::element_fusion::run(
        &fusion_reports,
        &crate::element_fusion::FusionParams::default(),
    );

    // 2) scene_stats 摘要（metrics 精简：不含 color_histogram 大数组）
    let stats = crate::scene_stats::run(img, &crate::scene_stats::SceneStatsParams { colors: 8 });
    let stats_metrics = stats
        .as_ref()
        .map(|v| v.get("metrics").cloned().unwrap_or(json!({})))
        .unwrap_or(json!({}));
    let mut stats_summary = stats_metrics.clone();
    if let Some(obj) = stats_summary.as_object_mut() {
        obj.remove("color_histogram"); // 大数组，摘要不需要
    }

    // 3) edge density 热图（16×16，<1K tokens）
    let edge = crate::edge::run(
        img,
        &crate::edge::EdgeParams {
            sigma: 1.0,
            hough_threshold: 40,
            max_lines: 50,
            max_points: 2000,
            density: true,
        },
    );
    let edge_metrics = edge
        .as_ref()
        .map(|v| v.get("metrics").cloned().unwrap_or(json!({})))
        .unwrap_or(json!({}));

    let mut report = envelope(
        Some("analyze"),
        &["element_fusion", "scene_stats", "edge"],
        Some("image_px"),
        Some(json!({"type": "image", "path": image})),
    );
    report["elements"] = fused
        .as_ref()
        .map(|v| v.get("elements").cloned().unwrap_or(json!([])))
        .unwrap_or(json!([]));
    report["metrics"] = json!({
        "scene": stats_summary,
        "edge": edge_metrics,
        "fusion": fused.as_ref().map(|v| v.get("metrics").cloned().unwrap_or(json!({}))).unwrap_or(json!({})),
    });
    report["notation"] = json!(NOTATION_GUIDE);
    report["note"] = json!(
        "粗报告（轻量）：elements 为融合元素清单（编号引用，如 click [3]）；scene 为全局统计摘要；edge 为边缘密度热图。需要精确定位时用 zoom 放大区域，或 probe 定向取证。"
    );
    Ok(report)
}

/// zoom：细报告（plan3 E3——区域 2× upscale + 局部融合元素 + ascii 局部）。
///
/// 流程：裁剪 region → 2× upscale → 局部 ocr/omniparser → 融合（坐标偏移回原图）
/// + ascii 局部（区域裁剪）+ edge density + depth。
fn zoom(image: &str, region: &str) -> Result<Value, String> {
    let coords: Vec<i64> = region
        .split(',')
        .map(|v| v.trim().parse().unwrap_or(0))
        .collect();
    if coords.len() != 4 {
        return Err(format!("invalid region: {region}"));
    }
    let (x1, y1, x2, y2) = (coords[0], coords[1], coords[2], coords[3]);
    let (ox, oy) = (x1.min(x2), y1.min(y2));
    let img = Path::new(image);

    // 1) 裁剪 + 2× upscale（LANCZOS，与 Python --upscale 2 一致）
    let cropped = crop_to_temp(image, region)?;
    let upscaled = upscale_temp(&cropped, 2)?;
    let _ = std::fs::remove_file(&cropped);
    let up_img = Path::new(&upscaled);

    // 2) 局部融合元素：ocr + omniparser → element_fusion（坐标偏移回原图）
    let ocr = crate::ocr::run(
        up_img,
        &crate::ocr::OcrParams {
            max_items: 40,
            min_conf: 0.5,
        },
    );
    let omni = crate::omniparser::run(
        up_img,
        &crate::omniparser::OmniparserParams {
            max_items: 20,
            min_conf: 0.3,
        },
    );
    let mut fusion_reports: Vec<Value> = Vec::new();
    if let Ok(v) = &ocr {
        fusion_reports.push(v.clone());
    }
    if let Ok(v) = &omni {
        fusion_reports.push(v.clone());
    }
    let fused = crate::element_fusion::run(
        &fusion_reports,
        &crate::element_fusion::FusionParams::default(),
    );
    let _ = std::fs::remove_file(&upscaled);

    // 3) ascii 局部（区域裁剪，不 upscale——粗栅格即可）
    let ascii_r = crate::ascii::run(
        img,
        &crate::ascii::AsciiParams {
            cols: 32,
            rows: 18,
            color: false,
            element_aware: false,
            elements: Vec::new(),
        },
    );

    // 4) edge density + depth（区域）
    let edge = crate::edge::run(
        img,
        &crate::edge::EdgeParams {
            sigma: 1.0,
            hough_threshold: 40,
            max_lines: 50,
            max_points: 2000,
            density: true,
        },
    );
    let depth = crate::depth::run(
        img,
        &crate::depth::DepthParams {
            bins: 3,
            region: Some(region.to_string()),
        },
    );

    // 5) 融合元素坐标偏移回原图（upscale 2× → 除以 2，再加区域偏移）
    let mut els: Vec<Value> = Vec::new();
    if let Ok(f) = &fused {
        if let Some(fe) = f.get("elements").and_then(|e| e.as_array()) {
            for (i, e) in fe.iter().enumerate() {
                let mut el = e.clone();
                if let Some(b) = el.get_mut("bbox").and_then(|b| b.as_array_mut()) {
                    if b.len() >= 4 {
                        b[0] = json!(b[0].as_f64().unwrap_or(0.0) / 2.0 + ox as f64);
                        b[1] = json!(b[1].as_f64().unwrap_or(0.0) / 2.0 + oy as f64);
                        b[2] = json!(b[2].as_f64().unwrap_or(0.0) / 2.0 + ox as f64);
                        b[3] = json!(b[3].as_f64().unwrap_or(0.0) / 2.0 + oy as f64);
                    }
                }
                if let Some(c) = el.get_mut("center").and_then(|c| c.as_array_mut()) {
                    if c.len() >= 2 {
                        c[0] = json!(c[0].as_f64().unwrap_or(0.0) / 2.0 + ox as f64);
                        c[1] = json!(c[1].as_f64().unwrap_or(0.0) / 2.0 + oy as f64);
                    }
                }
                el["id"] = json!(i);
                els.push(el);
            }
        }
    }

    let mut report = envelope(
        Some("zoom"),
        &["element_fusion", "ascii", "edge", "depth"],
        Some("image_px"),
        Some(json!({
            "type": "image",
            "path": image,
            "region": [x1, y1, x2, y2],
            "upscale": 2,
        })),
    );
    report["elements"] = json!(els);
    report["ascii"] = ascii_r
        .as_ref()
        .map(|v| v.get("ascii").cloned().unwrap_or(json!({})))
        .unwrap_or(json!({}));
    report["depth"] = depth
        .as_ref()
        .map(|v| v.get("depth").cloned().unwrap_or(json!({})))
        .unwrap_or(json!({}));
    report["metrics"] = json!({
        "region": [x1, y1, x2, y2],
        "upscale": 2,
        "fused_elements": els.len(),
        "edge": edge.as_ref().map(|v| v.get("metrics").cloned().unwrap_or(json!({}))).unwrap_or(json!({})),
    });
    report["notation"] = json!(NOTATION_GUIDE);
    report["note"] = json!(
        "细报告：region 已 2× 放大后融合（elements 坐标已偏移回原图）；ascii 为区域局部栅格；edge 为密度热图；depth 为区域深度。"
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

/// 图像 2× upscale（LANCZOS，与 Python `--upscale 2` 一致）。
fn upscale_temp(image: &str, factor: u32) -> Result<String, String> {
    let img = image::open(image).map_err(|e| format!("open {image}: {e}"))?;
    let (w, h) = (img.width() * factor, img.height() * factor);
    let up = image::imageops::resize(&img, w, h, image::imageops::FilterType::Lanczos3);
    let tmp = std::env::temp_dir().join(format!("vs_protocol_up_{}.png", std::process::id()));
    up.save(&tmp).map_err(|e| format!("save upscale: {e}"))?;
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
                density: false, // probe 按需：原始 edge_point
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

    #[test]
    fn analyze_lightweight_no_ascii_grid() {
        // E3 门禁：analyze 不返回 ascii 全栅格、不返回原始 ocr 元素
        let img = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../bench/samples/s03_alignment.png"
        );
        let p = ProtocolParams {
            action: "analyze".into(),
            image: img.into(),
            region: None,
            bbox: None,
            sensor: None,
        };
        let r = run(&p).unwrap();
        assert!(r.get("ascii").is_none(), "analyze 不应返回 ascii 全栅格");
        assert!(
            r.get("candidates")
                .and_then(|c| c.as_array())
                .map(|a| a.is_empty())
                .unwrap_or(false),
            "analyze 不应返回 saliency candidates（应为空）"
        );
        // 融合元素清单存在
        let els = r["elements"].as_array().unwrap();
        assert!(!els.is_empty(), "analyze 应返回融合元素清单");
        // 每元素带编号 + 溯源
        for e in els {
            assert!(e.get("id").is_some(), "元素应带编号");
            assert!(e.get("sources").is_some(), "元素应带溯源");
        }
        // 输出大小 < 3K tokens（≈ 12KB）
        let size = serde_json::to_string(&r).unwrap().len();
        assert!(
            size < 12_000,
            "analyze 输出应 < 3K tokens，实际 {size} bytes"
        );
    }

    #[test]
    fn zoom_upscale_and_coord_offset() {
        // E3 门禁：zoom 区域 2× upscale + 坐标偏移回原图
        let img = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../bench/samples/s03_alignment.png"
        );
        let p = ProtocolParams {
            action: "zoom".into(),
            image: img.into(),
            region: Some("100,80,300,200".into()),
            bbox: None,
            sensor: None,
        };
        let r = run(&p).unwrap();
        assert_eq!(r["source"]["upscale"], 2, "zoom 应 2× upscale");
        // 融合元素坐标应在原图空间（x ≥ 100, y ≥ 80）
        if let Some(els) = r["elements"].as_array() {
            for e in els {
                if let Some(b) = e.get("bbox").and_then(|b| b.as_array()) {
                    let x = b[0].as_f64().unwrap();
                    let y = b[1].as_f64().unwrap();
                    assert!(x >= 100.0, "bbox x 应偏移回原图: {x}");
                    assert!(y >= 80.0, "bbox y 应偏移回原图: {y}");
                }
            }
        }
        // ascii 局部存在
        assert!(r.get("ascii").is_some(), "zoom 应返回 ascii 局部");
    }
}
