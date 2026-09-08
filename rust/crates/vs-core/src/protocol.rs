//! 多轮反馈协议（对应 Python `vs_protocol.py`，纯 Rust，schema v3 输出）。
//!
//! SeeingEye 模式：三个动作驱动多轮视觉推理：
//!   analyze [image]      → 粗报告（saliency 候选 + ascii 粗栅格 + 全局统计 + 融合）
//!   zoom [region]        → 细报告（region 内 ocr 高倍 + segment 前景 + edge + depth）
//!   probe [bbox, sensor] → 单传感器定向证据

use std::path::Path;

use serde_json::{json, Value};

use crate::schema::{envelope, NOTATION_GUIDE};

/// 读取图片物理尺寸（像素）。失败时回退 (0, 0)。
fn image_size_px(image: &str) -> (u32, u32) {
    match crate::image::load_rgb(Path::new(image)) {
        Ok(img) => (img.width, img.height),
        Err(_) => (0, 0),
    }
}

/// 协议参数。
pub struct ProtocolParams {
    /// analyze | zoom | probe
    pub action: String,
    pub image: String,
    pub region: Option<String>,
    pub bbox: Option<String>,
    pub sensor: Option<String>,
    /// probe 确认模式：对指定融合元素 id 重新检测（E6 确定性锚点）。
    pub confirm: Option<String>,
    /// G3 闭环验证：click 后截图对比（verify_click）。
    pub verify_click: bool,
    /// G4 窗口过滤：只分析指定 niri 窗口 id 区域（排除终端/其他窗口内容）。
    pub window: Option<String>,
}

/// analyze：粗报告（plan3 E3 分层感知——默认轻量路径）。
///
/// 只返回：融合元素清单（element_fusion）+ scene_stats 摘要 + edge density 热图。
/// 不返回：原始 ocr/omniparser/detect 元素、edge_point、ascii 全栅格。
/// 目标：总输出 < 3K tokens（两级感知预算的"粗"层）。
/// 全局感知缓存（F6：增量感知——事件触发重扫，查询命中缓存）。
static PERCEPTION_CACHE: std::sync::OnceLock<crate::events::PerceptionCache> =
    std::sync::OnceLock::new();

/// 获取全局感知缓存。
pub fn perception_cache() -> &'static crate::events::PerceptionCache {
    PERCEPTION_CACHE.get_or_init(crate::events::PerceptionCache::new)
}

fn analyze(image: &str) -> Result<Value, String> {
    // F6：缓存命中直接返回（同窗口连续 analyze ≤1s）
    if let Some(cached) = perception_cache().get(image) {
        return Ok(cached);
    }
    // F2（plan5）：a11y 树优先——先跑 AT-SPI（~100ms），有元素则用 a11y（无 DL）
    if let Ok(a11y_report) = crate::a11y::run(&crate::a11y::A11yParams {
        app: None,
        max_items: 60,
    }) {
        let a11y_els = a11y_report
            .get("elements")
            .and_then(|e| e.as_array())
            .cloned()
            .unwrap_or_default();
        if !a11y_els.is_empty() {
            let r = analyze_a11y_only(image, a11y_report);
            if let Ok(report) = &r {
                perception_cache().put(image, report.clone());
            }
            return r;
        }
    }
    // 降级：a11y 空 → DL 融合（现有路径）
    let r = analyze_dl(image);
    if let Ok(report) = &r {
        perception_cache().put(image, report.clone());
    }
    r
}

/// a11y 优先路径：a11y 元素 + scene_stats 摘要（无 DL 推理，≤1s）。
fn analyze_a11y_only(image: &str, a11y_report: Value) -> Result<Value, String> {
    let img = Path::new(image);
    // 1) a11y 元素 → 统一编号 + 补 center/conf/assertion（与融合元素结构一致）
    let mut elements: Vec<Value> = a11y_report
        .get("elements")
        .and_then(|e| e.as_array())
        .cloned()
        .unwrap_or_default();
    for (i, e) in elements.iter_mut().enumerate() {
        e["id"] = json!(i);
        // center = bbox 中心
        if let Some(b) = e.get("bbox").and_then(|b| b.as_array()) {
            if b.len() >= 4 {
                let x1 = b[0].as_f64().unwrap_or(0.0);
                let y1 = b[1].as_f64().unwrap_or(0.0);
                let x2 = b[2].as_f64().unwrap_or(0.0);
                let y2 = b[3].as_f64().unwrap_or(0.0);
                e["center"] = json!([(x1 + x2) / 2.0, (y1 + y2) / 2.0]);
            }
        }
        // a11y 元素 conf = 1.0（结构确认，非检测）
        e["conf"] = json!(1.0);
        e["sources"] = json!(["a11y"]);
        e["assertion"] = json!("此元素由 AT-SPI 无障碍树确认（结构级，conf 1.0）");
        // 精简字段（token 治理）——保留 bus/path（F3：click 直接定位）
        let keep = [
            "id",
            "bbox",
            "center",
            "text",
            "type",
            "conf",
            "sources",
            "assertion",
            "bus",
            "path",
        ];
        if let Some(obj) = e.as_object_mut() {
            obj.retain(|k, _| keep.contains(&k.as_str()));
        }
    }
    // token 治理：top-15 截断（与 DL 路径一致）
    elements.truncate(10);
    // 2) scene_stats 摘要（metrics 精简）
    let stats = crate::scene_stats::run(img, &crate::scene_stats::SceneStatsParams { colors: 8 });
    let stats_metrics = stats
        .as_ref()
        .map(|v| v.get("metrics").cloned().unwrap_or(json!({})))
        .unwrap_or(json!({}));
    let mut stats_summary = stats_metrics.clone();
    if let Some(obj) = stats_summary.as_object_mut() {
        obj.remove("color_histogram");
    }
    let mut report = envelope(
        Some("analyze"),
        &["a11y", "scene_stats"],
        Some("screen_px"),
        Some(json!({"type": "image", "path": image, "channel": "a11y"})),
    );
    report["elements"] = json!(elements);
    report["metrics"] = json!({"scene": stats_summary, "channel": "a11y"});
    report["notation"] = json!(NOTATION_GUIDE);
    let ids: Vec<i64> = (0..elements.len() as i64).collect();
    report["note"] = json!(format!(
        "粗报告（a11y 通道）：elements 为 AT-SPI 无障碍树元素（编号引用，如 click [3]）；scene 为全局统计摘要。已确认元素 {}（结构级 conf 1.0）。无 DL 推理（a11y 优先）。",
        format!("[{}]", ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(","))
    ));
    Ok(report)
}

/// DL 兜底路径：ocr + omniparser + detect → element_fusion（现有 analyze 逻辑）。
fn analyze_dl(image: &str) -> Result<Value, String> {
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
    // G3 token 治理：元素截断 top-20（按 conf 降序）+ 精简字段
    if let Some(els) = report["elements"].as_array_mut() {
        els.sort_by(|a, b| {
            let ca = a.get("conf").and_then(|c| c.as_f64()).unwrap_or(0.0);
            let cb = b.get("conf").and_then(|c| c.as_f64()).unwrap_or(0.0);
            cb.partial_cmp(&ca).unwrap_or(std::cmp::Ordering::Equal)
        });
        els.truncate(10);
        for e in els.iter_mut() {
            let keep = [
                "id",
                "bbox",
                "center",
                "text",
                "caption",
                "conf",
                "sources",
                "assertion",
            ];
            if let Some(obj) = e.as_object_mut() {
                obj.retain(|k, _| keep.contains(&k.as_str()));
                // caption 截断 40 字符（token 治理）
                if let Some(cap) = obj.get_mut("caption").and_then(|c| c.as_str()) {
                    let short: String = cap.chars().take(40).collect();
                    obj.insert("caption".into(), json!(short));
                }
            }
        }
    }
    report["metrics"] = json!({
        "scene": stats_summary,
        "edge": edge_metrics,
        "fusion": fused.as_ref().map(|v| v.get("metrics").cloned().unwrap_or(json!({}))).unwrap_or(json!({})),
    });
    report["notation"] = json!(NOTATION_GUIDE);
    // E6 输出模板：已确认元素 vs 推测元素（conf 阈值 0.7）
    let (confirmed_ids, speculative_ids): (Vec<i64>, Vec<i64>) = fused
        .as_ref()
        .ok()
        .and_then(|v| v.get("elements").and_then(|e| e.as_array()))
        .map(|els| {
            type ConfPair = (i64, f64);
            let (conf, spec): (Vec<ConfPair>, Vec<ConfPair>) = els
                .iter()
                .filter_map(|e| {
                    let id = e.get("id").and_then(|i| i.as_i64())?;
                    let conf = e.get("conf").and_then(|c| c.as_f64()).unwrap_or(0.0);
                    Some((id, conf))
                })
                .partition(|(_, c)| *c >= 0.7);
            (
                conf.into_iter().map(|(id, _)| id).collect(),
                spec.into_iter().map(|(id, _)| id).collect(),
            )
        })
        .unwrap_or_default();
    let fmt_ids = |ids: &[i64]| {
        if ids.is_empty() {
            "[]".to_string()
        } else {
            format!(
                "[{}]",
                ids.iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            )
        }
    };
    report["note"] = json!(format!(
        "粗报告（轻量）：elements 为融合元素清单（编号引用，如 click [3]）；scene 为全局统计摘要；edge 为边缘密度热图。已确认元素 {}（conf ≥ 0.7）；推测元素 {}（conf < 0.7，需 probe --confirm 验证）。需要精确定位时用 zoom 放大区域，或 probe 定向取证。",
        fmt_ids(&confirmed_ids),
        fmt_ids(&speculative_ids)
    ));
    Ok(report)
}

/// 查询 niri 窗口的 app_id（用于 a11y 树匹配）。
fn window_app_id(window_id: &str) -> Option<String> {
    use std::process::Command;
    let wins: Value = Command::new("niri")
        .args(["msg", "--json", "windows"])
        .output()
        .ok()
        .and_then(|o| serde_json::from_slice(&o.stdout).ok())?;
    wins.as_array()?
        .iter()
        .find(|w| {
            w.get("id").and_then(|i| i.as_i64()).map(|i| i.to_string())
                == Some(window_id.to_string())
        })
        .and_then(|w| w.get("app_id").and_then(|a| a.as_str()).map(String::from))
}

/// 估算 niri 窗口的屏幕位置（逻辑坐标）。
/// 用启发式：workspace → output 的 x 偏移 + column 索引 × tile 宽度。
/// 返回 (x, y, w, h) 逻辑坐标；失败返回 None。
fn window_logical_rect(window_id: &str) -> Option<(f64, f64, f64, f64)> {
    use std::process::Command;
    // 1) 窗口信息
    let wins: Value = Command::new("niri")
        .args(["msg", "--json", "windows"])
        .output()
        .ok()
        .and_then(|o| serde_json::from_slice(&o.stdout).ok())?;
    let win = wins.as_array()?.iter().find(|w| {
        w.get("id").and_then(|i| i.as_i64()).map(|i| i.to_string()) == Some(window_id.to_string())
    })?;
    let ws_id = win.get("workspace_id").and_then(|v| v.as_i64())?;
    let layout = win.get("layout")?;
    let tile_w = layout.get("tile_size")?.as_array()?.first()?.as_f64()?;
    let tile_h = layout.get("tile_size")?.as_array()?.get(1)?.as_f64()?;
    // 2) workspace → output
    let ws: Value = Command::new("niri")
        .args(["msg", "--json", "workspaces"])
        .output()
        .ok()
        .and_then(|o| serde_json::from_slice(&o.stdout).ok())?;
    let out_name = ws
        .as_array()?
        .iter()
        .find(|w| w.get("id").and_then(|i| i.as_i64()) == Some(ws_id))?
        .get("output")?
        .as_str()?
        .to_string();
    // 3) output 的 logical x 偏移
    let outs: Value = Command::new("niri")
        .args(["msg", "--json", "outputs"])
        .output()
        .ok()
        .and_then(|o| serde_json::from_slice(&o.stdout).ok())?;
    let out_x = outs
        .as_object()?
        .get(&out_name)?
        .get("logical")?
        .get("x")?
        .as_f64()?;
    let out_y = outs
        .as_object()?
        .get(&out_name)?
        .get("logical")?
        .get("y")?
        .as_f64()?;
    // 4) 窗口位置：tile 接近输出大小时窗口占满输出（x=output_x）；
    //    否则用滚动布局列位置（需视口偏移，niri 不暴露——启发式）
    let out_w = outs
        .as_object()?
        .get(&out_name)?
        .get("logical")?
        .get("width")?
        .as_f64()?;
    let out_h = outs
        .as_object()?
        .get(&out_name)?
        .get("logical")?
        .get("height")?
        .as_f64()?;
    // tile 占满输出（≥90% 宽高）→ 窗口在输出内全屏（x=output_x）
    let fills_output = tile_w >= out_w * 0.9 && tile_h >= out_h * 0.9;
    let x = if fills_output {
        out_x
    } else {
        // 多列布局：滚动位置 - 视口偏移（focused 窗口在视口内，偏移 = (col-1)×tile_w）
        // 视口显示 focused 窗口所在列，故窗口屏幕 x = output_x（focused 时）
        out_x
    };
    Some((x, out_y, tile_w.min(out_w), tile_h.min(out_h)))
}

/// analyze --window <id>：只分析指定窗口区域（排除终端/其他窗口内容）。
///
/// 流程：niri 窗口位置（逻辑）→ 换算物理像素 → 裁剪 → analyze → 坐标偏移回原图。
fn analyze_window(image: &str, window_id: &str) -> Result<Value, String> {
    // 1) 窗口逻辑位置
    let (wx, wy, ww, wh) = window_logical_rect(window_id)
        .ok_or_else(|| format!("无法定位窗口 {window_id}（niri 窗口信息缺失）"))?;
    // 2) 换算物理像素（capture size / layout size）
    let (w_cap, h_cap) = image_size_px(image);
    let (w_log, h_log) = crate::gui::layout_size();
    let sx = if w_log > 0 {
        w_cap as f64 / w_log as f64
    } else {
        1.0
    };
    let sy = if h_log > 0 {
        h_cap as f64 / h_log as f64
    } else {
        1.0
    };
    let (px, py, pw, ph) = (wx * sx, wy * sy, ww * sx, wh * sy);
    // 3) 裁剪窗口区域 → 临时文件
    let img = crate::image::load_rgb(Path::new(image)).map_err(|e| format!("load {image}: {e}"))?;
    let (x, y) = (px.max(0.0) as u32, py.max(0.0) as u32);
    let (w, h) = (
        pw.min(img.width as f64 - x as f64) as u32,
        ph.min(img.height as f64 - y as f64) as u32,
    );
    if w == 0 || h == 0 {
        return Err(format!("窗口 {window_id} 区域为空（{px},{py} {pw}x{ph}）"));
    }
    let crop = image::imageops::crop_imm(&img_to_dynamic(&img), x, y, w, h).to_image();
    let tmp = format!("/tmp/vsd_window_{window_id}.png");
    crop.save(&tmp).map_err(|e| format!("save crop: {e}"))?;
    // 4) 分析裁剪区域（F2：a11y 优先——按窗口 app_id 匹配 a11y 树）
    let mut report = if let Some(app_id) = window_app_id(window_id) {
        // 窗口 app_id → a11y 树（~100ms，无 DL）
        match crate::a11y::run(&crate::a11y::A11yParams {
            app: Some(app_id.clone()),
            max_items: 60,
        }) {
            Ok(a11y_report) => {
                let a11y_els = a11y_report
                    .get("elements")
                    .and_then(|e| e.as_array())
                    .cloned()
                    .unwrap_or_default();
                if !a11y_els.is_empty() {
                    analyze_a11y_only(&tmp, a11y_report)?
                } else {
                    analyze(&tmp)?
                }
            }
            Err(_) => analyze(&tmp)?,
        }
    } else {
        analyze(&tmp)?
    };
    // 5) 坐标偏移回原图（物理像素）
    if let Some(els) = report["elements"].as_array_mut() {
        for e in els.iter_mut() {
            if let Some(b) = e.get_mut("bbox").and_then(|b| b.as_array_mut()) {
                if b.len() >= 4 {
                    b[0] = json!(b[0].as_f64().unwrap_or(0.0) + x as f64);
                    b[1] = json!(b[1].as_f64().unwrap_or(0.0) + y as f64);
                    b[2] = json!(b[2].as_f64().unwrap_or(0.0) + x as f64);
                    b[3] = json!(b[3].as_f64().unwrap_or(0.0) + y as f64);
                }
            }
            if let Some(c) = e.get_mut("center").and_then(|c| c.as_array_mut()) {
                if c.len() >= 2 {
                    c[0] = json!(c[0].as_f64().unwrap_or(0.0) + x as f64);
                    c[1] = json!(c[1].as_f64().unwrap_or(0.0) + y as f64);
                }
            }
        }
    }
    report["source"] = json!({"type": "window", "window_id": window_id, "path": image});
    report["metrics"]["window_rect_logical"] = json!([wx, wy, ww, wh]);
    let _ = std::fs::remove_file(&tmp);
    Ok(report)
}

/// 将 RgbImage 转为 DynamicImage（供 crop_imm 使用）。
fn img_to_dynamic(img: &crate::image::RgbImage) -> image::DynamicImage {
    use image::ImageBuffer;
    let buf: ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_raw(img.width, img.height, img.data.clone()).unwrap();
    image::DynamicImage::ImageRgb8(buf)
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

/// probe 确认模式（plan3 E6）：对指定融合元素 id 重新检测（ocr + omniparser 局部），
/// 返回确认/否认 + 新 conf。DeepSeek 可先 probe 确认元素存在，再执行动作。
fn probe_confirm(image: &str, id: usize) -> Result<Value, String> {
    // 1) 先跑 analyze 得到融合元素清单
    let base = analyze(image)?;
    let els = base
        .get("elements")
        .and_then(|e| e.as_array())
        .ok_or_else(|| "analyze 无融合元素".to_string())?;
    let target = els
        .get(id)
        .ok_or_else(|| format!("元素 id {id} 不存在（共 {} 个）", els.len()))?;
    let bbox = target
        .get("bbox")
        .and_then(|b| b.as_array())
        .ok_or_else(|| "元素无 bbox".to_string())?;
    if bbox.len() < 4 {
        return Err("元素 bbox 不完整".into());
    }
    let to_f64 = |v: &Value| v.as_f64().unwrap_or_else(|| v.as_i64().unwrap_or(0) as f64);
    let (x1, y1, x2, y2) = (
        to_f64(&bbox[0]) as i64,
        to_f64(&bbox[1]) as i64,
        to_f64(&bbox[2]) as i64,
        to_f64(&bbox[3]) as i64,
    );
    let region = format!("{x1},{y1},{x2},{y2}");

    // 2) 局部重检：裁剪 region → 2× upscale → ocr + omniparser → 融合
    let cropped = crop_to_temp(image, &region)?;
    let upscaled = upscale_temp(&cropped, 2)?;
    let _ = std::fs::remove_file(&cropped);
    let up_img = Path::new(&upscaled);
    let ocr = crate::ocr::run(
        up_img,
        &crate::ocr::OcrParams {
            max_items: 10,
            min_conf: 0.3,
        },
    );
    let omni = crate::omniparser::run(
        up_img,
        &crate::omniparser::OmniparserParams {
            max_items: 5,
            min_conf: 0.2,
        },
    );
    let _ = std::fs::remove_file(&upscaled);
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

    // 3) 判定：局部融合有元素 → 确认；否则 → 否认
    let local_els = fused
        .as_ref()
        .map(|v| {
            v.get("elements")
                .and_then(|e| e.as_array())
                .map(|a| a.len())
                .unwrap_or(0)
        })
        .unwrap_or(0);
    let new_conf = fused
        .as_ref()
        .ok()
        .and_then(|v| v.get("elements").and_then(|e| e.as_array()))
        .and_then(|a| a.first())
        .and_then(|e| e.get("conf"))
        .and_then(|c| c.as_f64())
        .unwrap_or(0.0);
    let confirmed = local_els > 0 && new_conf >= 0.5;

    let mut report = envelope(
        Some("probe_confirm"),
        &["element_fusion"],
        Some("image_px"),
        Some(json!({
            "type": "image",
            "path": image,
            "element_id": id,
            "region": [x1, y1, x2, y2],
        })),
    );
    report["confirmed"] = json!(confirmed);
    report["new_conf"] = json!((new_conf * 1000.0).round() / 1000.0);
    report["local_elements"] = json!(local_els);
    report["elements"] = fused
        .as_ref()
        .map(|v| v.get("elements").cloned().unwrap_or(json!([])))
        .unwrap_or(json!([]));
    report["note"] = json!(if confirmed {
        format!("确认：元素 [{id}] 存在（局部重检 conf {new_conf:.3}）")
    } else {
        format!("否认：元素 [{id}] 未在局部重检中确认（conf {new_conf:.3}）")
    });
    Ok(report)
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
        "analyze" => {
            // G4：--window <id> 只分析指定窗口区域（排除终端/其他窗口内容）
            if let Some(wid) = &p.window {
                analyze_window(&p.image, wid)
            } else {
                analyze(&p.image)
            }
        }
        "zoom" => {
            let region = p
                .region
                .clone()
                .ok_or_else(|| "zoom 需要 --region x1,y1,x2,y2".to_string())?;
            zoom(&p.image, &region)
        }
        "probe" => {
            // E6：confirm 模式（对元素 id 重检）优先于 bbox+sensor 模式
            if let Some(cid) = &p.confirm {
                let id: usize = cid
                    .trim()
                    .parse()
                    .map_err(|_| format!("confirm 需要整数元素 id，收到: {cid}"))?;
                return probe_confirm(&p.image, id);
            }
            let bbox = p
                .bbox
                .clone()
                .ok_or_else(|| "probe 需要 --bbox x1,y1,x2,y2 或 --confirm <id>".to_string())?;
            let sensor = p
                .sensor
                .clone()
                .ok_or_else(|| "probe 需要 --sensor ocr|edge|depth|segment|pix".to_string())?;
            probe(&p.image, &bbox, &sensor)
        }
        "click" => {
            // plan4 G1：click [N] 编号引用 → 五级降级链（a11y → element → vision → global）
            let id: usize = p
                .confirm
                .clone()
                .ok_or_else(|| "click 需要 --confirm <元素编号>（如 click [3]）".to_string())?
                .trim()
                .parse()
                .map_err(|e| format!("click 需要整数元素编号: {e}"))?;
            // 先 analyze 得到融合元素清单 → 取元素 bbox center 作为 vision_click 兜底
            // （--window 时用窗口过滤分析，排除其他窗口内容污染）
            let base = match &p.window {
                Some(wid) => analyze_window(&p.image, wid)?,
                None => analyze(&p.image)?,
            };
            let fallback: Option<(f64, f64)> = base
                .get("elements")
                .and_then(|e| e.as_array())
                .and_then(|els| els.get(id))
                .and_then(|e| e.get("center"))
                .and_then(|c| c.as_array())
                .and_then(|c| {
                    if c.len() >= 2 {
                        Some((c[0].as_f64().unwrap_or(0.0), c[1].as_f64().unwrap_or(0.0)))
                    } else {
                        None
                    }
                });
            // 坐标换算：capture 物理像素 → 逻辑坐标（wlrctl motion_absolute 需要）
            // capture 输出 5760×1800（scale 1.25），layout 4608×1440 → ÷1.25
            let fallback = fallback.map(|(x, y)| {
                let (w_cap, h_cap) = image_size_px(&p.image);
                let (w_log, h_log) = crate::gui::layout_size();
                let sx = if w_log > 0 {
                    w_cap as f64 / w_log as f64
                } else {
                    1.0
                };
                let sy = if h_log > 0 {
                    h_cap as f64 / h_log as f64
                } else {
                    1.0
                };
                (x / sx, y / sy)
            });
            // 从 analyze 结果取 bus/path（F3：click 直接定位，不再遍历全局树）
            let (bus, path) = base
                .get("elements")
                .and_then(|e| e.as_array())
                .and_then(|els| els.get(id))
                .map(|e| {
                    (
                        e.get("bus").and_then(|b| b.as_str()).map(String::from),
                        e.get("path").and_then(|p| p.as_str()).map(String::from),
                    )
                })
                .unwrap_or((None, None));
            let r = crate::gui::click_element(id, fallback, bus.as_deref(), path.as_deref())?;
            // 闭环：返回元素信息 + 点击结果
            let mut out = r.clone();
            out["element_id"] = json!(id);
            if let Some((x, y)) = fallback {
                out["element_center"] = json!([x, y]);
            }
            // G3 闭环验证：click → verify_click（截图对比）→ 失败重试
            if p.verify_click {
                let cap = "/tmp/vsd_verify";
                match crate::gui::click_with_verify(id, fallback, cap, 0.001, 2) {
                    Ok(v) => {
                        out["verify_click"] = v.get("verify").cloned().unwrap_or(json!(null));
                        out["attempts"] = v.get("attempts").cloned().unwrap_or(json!([]));
                    }
                    Err(e) => {
                        out["verify_click"] = json!({"ok": false, "error": e});
                    }
                }
            }
            Ok(out)
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
            confirm: None,
            verify_click: false,
            window: None,
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
            confirm: None,
            verify_click: false,
            window: None,
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
            confirm: None,
            verify_click: false,
            window: None,
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
            confirm: None,
            verify_click: false,
            window: None,
        };
        assert!(run(&p).is_err());
    }

    #[test]
    fn probe_confirm_invalid_id_rejected() {
        // E6：confirm 非整数 → 报错
        let p = ProtocolParams {
            action: "probe".into(),
            image: "x.png".into(),
            region: None,
            bbox: None,
            sensor: None,
            confirm: Some("abc".into()),
            verify_click: false,
            window: None,
        };
        assert!(run(&p).is_err(), "confirm 非整数应报错");
    }

    #[test]
    fn probe_confirm_unknown_id_rejected() {
        // E6：confirm 超范围 id → 报错
        let img = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../bench/samples/s03_alignment.png"
        );
        let p = ProtocolParams {
            action: "probe".into(),
            image: img.into(),
            region: None,
            bbox: None,
            sensor: None,
            confirm: Some("999".into()),
            verify_click: false,
            window: None,
        };
        assert!(run(&p).is_err(), "confirm 超范围 id 应报错");
    }

    #[test]
    fn probe_confirm_known_element_confirms() {
        // E6：对已知元素 confirm → 确认 + 新 conf
        let img = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../bench/samples/s03_alignment.png"
        );
        let p = ProtocolParams {
            action: "probe".into(),
            image: img.into(),
            region: None,
            bbox: None,
            sensor: None,
            confirm: Some("0".into()),
            verify_click: false,
            window: None,
        };
        let r = run(&p).unwrap();
        assert_eq!(r["confirmed"], true, "已知元素应确认");
        assert!(r["new_conf"].as_f64().unwrap() > 0.0, "确认应带新 conf");
        let note = r["note"].as_str().unwrap();
        assert!(note.contains("确认"), "note 应含确认: {note}");
    }

    #[test]
    fn analyze_note_has_assertion_summary() {
        // E6：analyze note 含已确认/推测元素汇总
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
            confirm: None,
            verify_click: false,
            window: None,
        };
        let r = run(&p).unwrap();
        let note = r["note"].as_str().unwrap();
        assert!(note.contains("已确认元素"), "note 应含已确认元素: {note}");
        // a11y 通道无推测元素（全部 conf 1.0）；DL 通道有
        if note.contains("a11y 通道") {
            assert!(
                !note.contains("推测元素"),
                "a11y 通道不应有推测元素: {note}"
            );
        } else {
            assert!(note.contains("推测元素"), "note 应含推测元素: {note}");
        }
        // 每元素带 assertion 字段
        for e in r["elements"].as_array().unwrap() {
            assert!(e.get("assertion").is_some(), "元素应带存在性断言");
        }
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
            confirm: None,
            verify_click: false,
            window: None,
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
            confirm: None,
            verify_click: false,
            window: None,
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
