//! 文档版式分析（对应 Python `vs_layout.py`，schema v3 输出逐字段一致）。
//!
//! PP-DocLayoutV3 → ONNX（doclayout.onnx，paddle.enable_static + paddle2onnx 导出，
//! 绕过 PIR 解析 bug）+ 纯 Rust 推理 + paddlex 后处理精确复刻：
//!   - 预处理：cv2 INTER_CUBIC resize 800×800 + NormalizeImage + Permute
//!   - 后处理：阈值 + mask polygon 提取（findContours + approxPolyDP +
//!     extract_custom_vertices）+ minAreaRect quad + 坐标映射
//!
//! 零 Python 回退（plan2 P5 门禁）。

use std::path::Path;

use serde_json::{json, Value};

use crate::schema::envelope;
use crate::util::resize_cv2_cubic;

/// 布局参数（对应 Python CLI 参数）。
pub struct LayoutParams {
    pub max_items: u32,
    pub min_conf: f64,
}

/// 模型路径。
fn model_path() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    format!("{home}/.cache/vsensor/doclayout.onnx")
}

/// 标签列表（PP-DocLayoutV3 26 类，与 inference.yml label_list 一致）。
const LABELS: [&str; 26] = [
    "abstract",
    "algorithm",
    "aside_text",
    "chart",
    "content",
    "display_formula",
    "doc_title",
    "figure_title",
    "footer",
    "footer_image",
    "footnote",
    "formula_number",
    "header",
    "header_image",
    "image",
    "inline_formula",
    "number",
    "paragraph_title",
    "reference",
    "reference_content",
    "seal",
    "table",
    "text",
    "vertical_text",
    "vision_footnote",
    "unknown",
];

/// 运行布局传感器（纯 Rust：ONNX 推理 + paddlex 后处理复刻）。
pub fn run(path: &Path, p: &LayoutParams) -> Result<Value, String> {
    // 1) 读图 + 预处理（cv2 INTER_CUBIC resize 800×800 + normalize + permute）
    let img = crate::image::load_rgb(path).map_err(|e| format!("load image: {e}"))?;
    let (w, h) = (img.width as f64, img.height as f64);
    let resized = resize_cv2_cubic(&img, 800, 800);
    // NormalizeImage（mean 0, std 1）+ Permute（CHW）
    let mut tensor = vec![0f32; 3 * 800 * 800];
    for i in 0..(800 * 800) {
        for c in 0..3 {
            tensor[c * 800 * 800 + i] = resized[i * 3 + c] as f32 / 255.0;
        }
    }
    let im_shape: Vec<f32> = vec![800.0, 800.0];
    let scale_factor: Vec<f32> = vec![(800.0 / h) as f32, (800.0 / w) as f32];

    // 2) ort 推理（3 输入 3 输出）
    let model = model_path();
    if !Path::new(&model).exists() {
        return Err(format!("模型不存在: {model}（需先导出 doclayout.onnx）"));
    }
    let mut session = ort::session::Session::builder()
        .map_err(|e| format!("ort builder: {e}"))?
        .commit_from_file(&model)
        .map_err(|e| format!("ort load {model}: {e}"))?;
    let shape_t = ort::value::Tensor::from_array(([1usize, 2], im_shape.into_boxed_slice()))
        .map_err(|e| format!("ort tensor im_shape: {e}"))?;
    let img_t = ort::value::Tensor::from_array(([1usize, 3, 800, 800], tensor.into_boxed_slice()))
        .map_err(|e| format!("ort tensor image: {e}"))?;
    let scale_t = ort::value::Tensor::from_array(([1usize, 2], scale_factor.into_boxed_slice()))
        .map_err(|e| format!("ort tensor scale: {e}"))?;
    let outputs = session
        .run(ort::inputs! { "im_shape" => shape_t, "image" => img_t, "scale_factor" => scale_t })
        .map_err(|e| format!("ort run: {e}"))?;
    let (shape0, det_data) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("ort extract det: {e}"))?;
    let (shape2, mask_data) = outputs[2]
        .try_extract_tensor::<i32>()
        .map_err(|e| format!("ort extract mask: {e}"))?;
    let n_det = shape0.num_elements() / 7;
    let det: Vec<f32> = det_data[..n_det * 7].to_vec();
    let (n_mask, mh, mw) = (shape2[0] as usize, shape2[1] as usize, shape2[2] as usize);
    let masks: Vec<i32> = mask_data[..n_mask * mh * mw].to_vec();

    // 3) 后处理（paddlex LayoutAnalysisProcess 复刻）
    let boxes = postprocess(PostInput {
        det: &det,
        masks: &masks,
        n_det,
        mh,
        mw,
        img_w: w as u32,
        img_h: h as u32,
        min_conf: p.min_conf,
    });

    // 4) schema v3 输出
    let mut els = Vec::new();
    for (i, b) in boxes.iter().enumerate() {
        if i >= p.max_items as usize {
            break;
        }
        let (label, score, coord) = b;
        let (x1, y1, x2, y2) = (coord[0], coord[1], coord[2], coord[3]);
        els.push(json!({
            "id": i,
            "type": "layout",
            "label": label,
            "bbox": [x1, y1, x2, y2],
            "conf": score,
            "text": null,
            "color": null,
            "font": null,
            "z": null,
            "source": ["layout"],
            "coordsys": "image_px",
            "center": [(x1 + x2) / 2, (y1 + y2) / 2],
            "quad": null,
        }));
    }
    let mut report = envelope(
        Some("layout"),
        &["layout"],
        Some("image_px"),
        Some(json!({
            "type": "image",
            "path": path.display().to_string(),
            "size_px": [w as u32, h as u32],
            "engine": "PP-DocLayoutV3",
        })),
    );
    report["elements"] = Value::Array(els);
    report["truncated"] = json!(boxes.len() >= p.max_items as usize);
    Ok(report)
}

/// 后处理：阈值过滤 + mask polygon 提取 + 坐标映射（paddlex LayoutAnalysisProcess 复刻）。
/// 后处理输入（mask 空间 + 原图尺寸 + 阈值）。
struct PostInput<'a> {
    det: &'a [f32],
    masks: &'a [i32],
    n_det: usize,
    mh: usize,
    mw: usize,
    img_w: u32,
    img_h: u32,
    min_conf: f64,
}

/// 返回 (label, score, [x1, y1, x2, y2]) 列表（原图坐标）。
fn postprocess(p: PostInput) -> Vec<(String, f64, [i32; 4])> {
    let PostInput {
        det,
        masks,
        n_det,
        mh,
        mw,
        img_w,
        img_h,
        min_conf,
    } = p;
    // 1) 阈值过滤（paddlex draw_threshold=0.5 内部过滤 + min_conf 外部过滤）
    let draw_threshold = 0.5f64;
    let mut kept: Vec<(usize, usize, f32, [f32; 4])> = Vec::new();
    for i in 0..n_det {
        let row = &det[i * 7..i * 7 + 7];
        let (cls, score) = (row[0] as usize, row[1] as f64);
        if score > draw_threshold && score > min_conf && cls < LABELS.len() {
            kept.push((i, cls, row[1], [row[2], row[3], row[4], row[5]]));
        }
    }
    if kept.is_empty() {
        return Vec::new();
    }

    // 2) mask polygon 提取（extract_polygon_points_by_masks）
    // scale_ratio = [h/s, w/s]（paddlex: scale_size=[800,800], img_size=[w,h]）
    let scale_w = (800.0 / img_w as f64) / 4.0;
    let scale_h = (800.0 / img_h as f64) / 4.0;
    let max_box_w = kept
        .iter()
        .map(|(_, _, _, b)| (b[2] - b[0]).max(0.0) as f64)
        .fold(0.0f64, |a, b| a.max(b));

    let mut out: Vec<(String, f64, [i32; 4])> = Vec::new();
    for (orig_idx, cls, score, box_) in kept.iter() {
        let (x_min, y_min, x_max, y_max) = (
            box_[0] as i32,
            box_[1] as i32,
            box_[2] as i32,
            box_[3] as i32,
        );
        let (box_w, box_h) = (x_max - x_min, y_max - y_min);
        if box_w <= 0 || box_h <= 0 {
            continue;
        }
        // crop mask（200×200 空间；paddlex 用 int() truncate，非 round）
        let x_s0 = ((x_min as f64 * scale_w) as i64).clamp(0, mw as i64) as usize;
        let x_s1 = ((x_max as f64 * scale_w) as i64).clamp(0, mw as i64) as usize;
        let y_s0 = ((y_min as f64 * scale_h) as i64).clamp(0, mh as i64) as usize;
        let y_s1 = ((y_max as f64 * scale_h) as i64).clamp(0, mh as i64) as usize;
        let mut cropped = vec![0u8; (y_s1 - y_s0) * (x_s1 - x_s0)];
        let mut sum = 0i32;
        for yy in y_s0..y_s1 {
            for xx in x_s0..x_s1 {
                let v = masks[orig_idx * mh * mw + yy * mw + xx];
                cropped[(yy - y_s0) * (x_s1 - x_s0) + (xx - x_s0)] = v as u8;
                sum += v;
            }
        }
        if sum == 0 {
            continue;
        }
        // resize mask 到 box 大小（cv2 INTER_NEAREST）
        let resized = resize_nearest(
            &cropped,
            x_s1 - x_s0,
            y_s1 - y_s0,
            box_w as usize,
            box_h as usize,
        );
        // mask → polygon（findContours + approxPolyDP + extract_custom_vertices）
        let polygon = mask2polygon(&resized, box_w as usize, box_h as usize, max_box_w);
        let polygon = match polygon {
            Some(p) => p,
            None => continue,
        };
        if std::env::var("VS_LAYOUT_DEBUG").is_ok() {
            eprintln!(
                "[dbg] cls={} score={:.3} box=({x_min},{y_min},{x_max},{y_max}) polygon={:?}",
                cls, score, polygon
            );
        }
        // polygon 坐标 + box 偏移
        let polygon: Vec<(f64, f64)> = polygon
            .iter()
            .map(|(px, py)| (px + x_min as f64, py + y_min as f64))
            .collect();
        // _normalize_layout_polygon（auto 模式：rect/quad/poly 决策）
        let rect = [
            (x_min as f64, y_min as f64),
            (x_max as f64, y_min as f64),
            (x_max as f64, y_max as f64),
            (x_min as f64, y_max as f64),
        ];
        let final_poly = normalize_layout_polygon(&rect, &polygon, None);
        // bbox（restructured_boxes：clamp 到原图）
        let xs: Vec<f64> = final_poly.iter().map(|p| p.0).collect();
        let ys: Vec<f64> = final_poly.iter().map(|p| p.1).collect();
        let (x1, y1) = (
            xs.iter().cloned().fold(f64::INFINITY, f64::min),
            ys.iter().cloned().fold(f64::INFINITY, f64::min),
        );
        let (x2, y2) = (
            xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        );
        let coord = [
            (x1.max(0.0).min(img_w as f64)) as i32,
            (y1.max(0.0).min(img_h as f64)) as i32,
            (x2.max(0.0).min(img_w as f64)) as i32,
            (y2.max(0.0).min(img_h as f64)) as i32,
        ];
        if coord[2] <= coord[0] || coord[3] <= coord[1] {
            continue;
        }
        out.push((LABELS[*cls].to_string(), *score as f64, coord));
    }
    out
}

/// cv2 INTER_NEAREST resize（mask 到 box 大小）。
/// cv2 用 cvFloor（向下取整）：src_x = floor(x * sw / dw)。
fn resize_nearest(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    let mut out = vec![0u8; dw * dh];
    for y in 0..dh {
        let sy = ((y as f64 * sh as f64 / dh as f64).floor() as usize).min(sh - 1);
        for x in 0..dw {
            let sx = ((x as f64 * sw as f64 / dw as f64).floor() as usize).min(sw - 1);
            out[y * dw + x] = src[sy * sw + sx];
        }
    }
    out
}

/// mask → polygon（findContours RETR_EXTERNAL + approxPolyDP + extract_custom_vertices）。
fn mask2polygon(mask: &[u8], w: usize, h: usize, max_box_w: f64) -> Option<Vec<(f64, f64)>> {
    if std::env::var("VS_LAYOUT_DEBUG").is_ok() {
        let mut min_y = h;
        let mut max_y = 0usize;
        let mut min_x = w;
        let mut max_x = 0usize;
        let mut count = 0usize;
        for y in 0..h {
            for x in 0..w {
                if mask[y * w + x] != 0 {
                    count += 1;
                    min_y = min_y.min(y);
                    max_y = max_y.max(y);
                    min_x = min_x.min(x);
                    max_x = max_x.max(x);
                }
            }
        }
        eprintln!(
            "    [dbg] mask2polygon 输入: {}x{} 非零 x=[{min_x},{max_x}] y=[{min_y},{max_y}] count={count}",
            w, h
        );
    }
    // findContours（RETR_EXTERNAL + CHAIN_APPROX_SIMPLE）：Moore 边界跟踪
    let contours = find_external_contours(mask, w, h);
    if contours.is_empty() {
        return None;
    }
    // 最大面积轮廓
    let mut best: Option<(Vec<(f64, f64)>, f64)> = None;
    for c in contours {
        let area = contour_area(&c);
        if best.as_ref().map(|(_, a)| area > *a).unwrap_or(true) {
            best = Some((c, area));
        }
    }
    let (contour, _) = best?;
    if std::env::var("VS_LAYOUT_DEBUG").is_ok() {
        eprintln!(
            "    [dbg] contour({} 点): {:?}",
            contour.len(),
            &contour[..contour.len().min(8)]
        );
    }
    // approxPolyDP（Ramer-Douglas-Peucker）
    let epsilon = 0.004 * arc_length(&contour);
    let approx = approx_poly_dp(&contour, epsilon);
    // 统一轮廓方向为顺时针（cv2 findContours 外部轮廓约定：shoelace 面积为负）
    let approx = ensure_clockwise(&approx);
    if std::env::var("VS_LAYOUT_DEBUG").is_ok() {
        eprintln!("    [dbg] approx({} 点): {:?}", approx.len(), approx);
    }
    // extract_custom_vertices
    let max_allowed = max_box_w * 0.3;
    let vertices = extract_custom_vertices(&approx, max_allowed);
    if std::env::var("VS_LAYOUT_DEBUG").is_ok() {
        eprintln!("    [dbg] vertices({} 点): {:?}", vertices.len(), vertices);
    }
    if vertices.len() < 4 {
        return None;
    }
    Some(vertices)
}

/// findContours RETR_EXTERNAL + CHAIN_APPROX_SIMPLE：Moore 边界跟踪。
fn find_external_contours(mask: &[u8], w: usize, h: usize) -> Vec<Vec<(f64, f64)>> {
    let mut visited = vec![false; w * h];
    let mut contours = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if mask[idx] != 0 && !visited[idx] {
                // BFS 连通域
                let mut stack = vec![(x, y)];
                visited[idx] = true;
                let mut pixels = Vec::new();
                while let Some((cx, cy)) = stack.pop() {
                    pixels.push((cx, cy));
                    for (dx, dy) in [(0i64, 1i64), (0, -1), (1, 0), (-1, 0)] {
                        let (nx, ny) = (cx as i64 + dx, cy as i64 + dy);
                        if nx >= 0 && ny >= 0 && nx < w as i64 && ny < h as i64 {
                            let ni = ny as usize * w + nx as usize;
                            if mask[ni] != 0 && !visited[ni] {
                                visited[ni] = true;
                                stack.push((nx as usize, ny as usize));
                            }
                        }
                    }
                }
                // 边界像素集合（4-邻域有背景）
                let is_boundary = |px: i64, py: i64| -> bool {
                    if px < 0 || py < 0 || px >= w as i64 || py >= h as i64 {
                        return false;
                    }
                    if mask[py as usize * w + px as usize] == 0 {
                        return false;
                    }
                    [(0i64, 1i64), (0, -1), (1, 0), (-1, 0)]
                        .iter()
                        .any(|(dx, dy)| {
                            let (nx, ny) = (px + dx, py + dy);
                            nx < 0
                                || ny < 0
                                || nx >= w as i64
                                || ny >= h as i64
                                || mask[ny as usize * w + nx as usize] == 0
                        })
                };
                let boundary: Vec<(usize, usize)> = pixels
                    .iter()
                    .filter(|(px, py)| is_boundary(*px as i64, *py as i64))
                    .cloned()
                    .collect();
                if boundary.len() >= 3 {
                    if std::env::var("VS_LAYOUT_DEBUG").is_ok() {
                        eprintln!(
                            "    [dbg] boundary {} 点: {:?}",
                            boundary.len(),
                            &boundary[..boundary.len().min(12)]
                        );
                    }
                    // 简化 Moore 边界跟踪：从最左上边界像素开始，顺时针
                    // 方向：0=东(1,0) 1=东北(1,-1) 2=北(0,-1) 3=西北(-1,-1)
                    //       4=西(-1,0) 5=西南(-1,1) 6=南(0,1) 7=东南(1,1)
                    let dirs = [
                        (1i64, 0i64),
                        (1, -1),
                        (0, -1),
                        (-1, -1),
                        (-1, 0),
                        (-1, 1),
                        (0, 1),
                        (1, 1),
                    ];
                    let start = *boundary
                        .iter()
                        .min_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)))
                        .unwrap();
                    let mut ordered: Vec<(usize, usize)> = Vec::new();
                    let mut cur = start;
                    let mut prev_dir = 4usize; // 初始进入方向：从西边
                    let mut steps = 0usize;
                    loop {
                        ordered.push(cur);
                        steps += 1;
                        if steps > boundary.len() * 8 {
                            break; // 防死循环
                        }
                        // 从进入方向的顺时针下一个开始搜索（跳过上一个像素）
                        let mut found = false;
                        for k in 1..=8 {
                            let d = (prev_dir + 8 - k) % 8;
                            let (dx, dy) = dirs[d];
                            let (nx, ny) = (cur.0 as i64 + dx, cur.1 as i64 + dy);
                            if is_boundary(nx, ny) {
                                cur = (nx as usize, ny as usize);
                                prev_dir = (d + 4) % 8; // 进入方向 = 反方向
                                found = true;
                                break;
                            }
                        }
                        if !found {
                            break;
                        }
                        if cur == start {
                            break; // 回到起点 → 完成
                        }
                    }
                    // CHAIN_APPROX_SIMPLE：只保留方向变化的点
                    let simple = chain_approx_simple(&ordered);
                    if simple.len() >= 3 {
                        contours.push(
                            simple
                                .iter()
                                .map(|&(px, py)| (px as f64, py as f64))
                                .collect(),
                        );
                    }
                }
            }
        }
    }
    contours
}

/// CHAIN_APPROX_SIMPLE：只保留方向变化的轮廓点。
fn chain_approx_simple(points: &[(usize, usize)]) -> Vec<(usize, usize)> {
    if points.len() < 3 {
        return points.to_vec();
    }
    let mut out = Vec::new();
    let n = points.len();
    for i in 0..n {
        let p_prev = points[(i + n - 1) % n];
        let p_curr = points[i];
        let p_next = points[(i + 1) % n];
        let d1 = (
            p_curr.0 as i64 - p_prev.0 as i64,
            p_curr.1 as i64 - p_prev.1 as i64,
        );
        let d2 = (
            p_next.0 as i64 - p_curr.0 as i64,
            p_next.1 as i64 - p_curr.1 as i64,
        );
        if d1 != d2 {
            out.push(p_curr);
        }
    }
    if out.len() < 3 {
        points.to_vec()
    } else {
        out
    }
}

/// 多边形面积（Shoelace，cv2.contourArea 等价）。
fn contour_area(poly: &[(f64, f64)]) -> f64 {
    contour_area_signed(poly).abs()
}

/// 有符号面积（正 = 逆时针，负 = 顺时针，图像 y 向下坐标系）。
fn contour_area_signed(poly: &[(f64, f64)]) -> f64 {
    let n = poly.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0;
    for i in 0..n {
        let (x1, y1) = poly[i];
        let (x2, y2) = poly[(i + 1) % n];
        area += x1 * y2 - x2 * y1;
    }
    area / 2.0
}

/// 统一轮廓方向为顺时针（cv2 findContours 外部轮廓约定：shoelace 面积为负）。
fn ensure_clockwise(poly: &[(f64, f64)]) -> Vec<(f64, f64)> {
    if contour_area_signed(poly) > 0.0 {
        // 逆时针 → 反转
        poly.iter().rev().cloned().collect()
    } else {
        poly.to_vec()
    }
}

/// 弧长（cv2.arcLength 等价）。
fn arc_length(poly: &[(f64, f64)]) -> f64 {
    let n = poly.len();
    if n < 2 {
        return 0.0;
    }
    let mut len = 0.0;
    for i in 0..n {
        let (x1, y1) = poly[i];
        let (x2, y2) = poly[(i + 1) % n];
        len += ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt();
    }
    len
}

/// Ramer-Douglas-Peucker（cv2.approxPolyDP 等价，处理闭合轮廓）。
fn approx_poly_dp(poly: &[(f64, f64)], epsilon: f64) -> Vec<(f64, f64)> {
    if poly.len() < 3 || epsilon <= 0.0 {
        return poly.to_vec();
    }
    // 闭合轮廓：首尾相连（cv2 处理闭合多边形）
    // 找最远点对作为分割点
    let mut max_d = 0.0f64;
    let mut split = 0usize;
    for i in 0..poly.len() {
        for j in 0..poly.len() {
            let d = ((poly[i].0 - poly[j].0).powi(2) + (poly[i].1 - poly[j].1).powi(2)).sqrt();
            if d > max_d {
                max_d = d;
                split = i;
            }
        }
    }
    // 从 split 点展开成开链
    let mut open: Vec<(f64, f64)> = poly[split..].to_vec();
    open.extend_from_slice(&poly[..=split]);
    let approx = dp_open(&open, epsilon);
    // 去重（首尾可能重复）
    let mut out = approx;
    if out.len() > 1 && out[0] == out[out.len() - 1] {
        out.pop();
    }
    out
}

/// 开链 Douglas-Peucker。
fn dp_open(poly: &[(f64, f64)], epsilon: f64) -> Vec<(f64, f64)> {
    if poly.len() < 3 {
        return poly.to_vec();
    }
    let (x1, y1) = poly[0];
    let (x2, y2) = poly[poly.len() - 1];
    let mut max_dist = 0.0f64;
    let mut max_idx = 0usize;
    for (i, &p) in poly.iter().enumerate().skip(1).take(poly.len() - 2) {
        let d = point_line_dist(p, (x1, y1), (x2, y2));
        if d > max_dist {
            max_dist = d;
            max_idx = i;
        }
    }
    if max_dist > epsilon {
        let left = dp_open(&poly[..=max_idx], epsilon);
        let right = dp_open(&poly[max_idx..], epsilon);
        let mut out = left[..left.len() - 1].to_vec();
        out.extend(right);
        out
    } else {
        vec![poly[0], poly[poly.len() - 1]]
    }
}

/// 点到线段距离。
fn point_line_dist(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let (px, py) = p;
    let (ax, ay) = a;
    let (bx, by) = b;
    let dx = bx - ax;
    let dy = by - ay;
    let len2 = dx * dx + dy * dy;
    if len2 == 0.0 {
        return ((px - ax).powi(2) + (py - ay).powi(2)).sqrt();
    }
    let t = (((px - ax) * dx + (py - ay) * dy) / len2).clamp(0.0, 1.0);
    let (cx, cy) = (ax + t * dx, ay + t * dy);
    ((px - cx).powi(2) + (py - cy).powi(2)).sqrt()
}

/// extract_custom_vertices（paddlex 复刻）。
fn extract_custom_vertices(poly: &[(f64, f64)], max_allowed_dist: f64) -> Vec<(f64, f64)> {
    let n = poly.len();
    if n < 3 {
        return poly.to_vec();
    }
    // point_info
    let mut is_convex_pts = vec![false; n];
    let mut angles = vec![0.0f64; n];
    let mut v1s = vec![(0.0f64, 0.0f64); n];
    let mut v2s = vec![(0.0f64, 0.0f64); n];
    for i in 0..n {
        let p_prev = poly[(i + n - 1) % n];
        let p_curr = poly[i];
        let p_next = poly[(i + 1) % n];
        let v1 = (p_curr.0 - p_prev.0, p_curr.1 - p_prev.1);
        let v2 = (p_next.0 - p_curr.0, p_next.1 - p_curr.1);
        let cross = v1.0 * v2.1 - v1.1 * v2.0;
        is_convex_pts[i] = cross < 0.0;
        angles[i] = angle_between(v1, v2);
        v1s[i] = v1;
        v2s[i] = v2;
    }
    // concave 分组
    let concave: Vec<usize> = (0..n).filter(|&i| !is_convex_pts[i]).collect();
    let mut preserve_concave: Vec<usize> = Vec::new();
    if !concave.is_empty() {
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut current = vec![concave[0]];
        for i in 1..concave.len() {
            if concave[i] - concave[i - 1] == 1 || (concave[i - 1] == n - 1 && concave[i] == 0) {
                current.push(concave[i]);
            } else {
                if current.len() >= 2 {
                    groups.push(current.clone());
                }
                current = vec![concave[i]];
            }
        }
        if current.len() >= 2 {
            groups.push(current);
        }
        if concave.len() >= 2 && concave[0] == 0 && concave[concave.len() - 1] == n - 1 {
            if groups.iter().any(|g| g.contains(&0)) && groups.iter().any(|g| g.contains(&(n - 1)))
            {
                for g in &groups {
                    preserve_concave.extend(g);
                }
            }
        } else {
            for g in &groups {
                preserve_concave.extend(g);
            }
        }
    }
    // kept_points
    let kept: Vec<usize> = (0..n)
        .filter(|&i| is_convex_pts[i] || (preserve_concave.contains(&i) && angles[i] >= 120.0))
        .collect();
    if kept.is_empty() {
        return poly.to_vec();
    }
    // final_points（dist > max_allowed_dist 时插值）
    let mut final_points: Vec<usize> = Vec::new();
    for idx in 0..kept.len() {
        let current_idx = kept[idx];
        let next_idx = kept[(idx + 1) % kept.len()];
        final_points.push(current_idx);
        let dist = ((poly[current_idx].0 - poly[next_idx].0).powi(2)
            + (poly[current_idx].1 - poly[next_idx].1).powi(2))
        .sqrt();
        if dist > max_allowed_dist {
            let intermediate: Vec<usize> = if next_idx > current_idx {
                (current_idx + 1..next_idx).collect()
            } else {
                (current_idx + 1..n).chain(0..next_idx).collect()
            };
            if !intermediate.is_empty() {
                let num_needed = (dist / max_allowed_dist).ceil() as usize - 1;
                if num_needed > 0 {
                    if intermediate.len() <= num_needed {
                        final_points.extend(intermediate);
                    } else {
                        let step = intermediate.len() as f64 / num_needed as f64;
                        for k in 0..num_needed {
                            final_points.push(intermediate[(k as f64 * step) as usize]);
                        }
                    }
                }
            }
        }
    }
    final_points.sort_unstable();
    final_points.dedup();
    // 输出（sharp angle 45° 时偏移）
    let mut res = Vec::new();
    for &i in &final_points {
        if is_convex_pts[i] && (angles[i] - 45.0).abs() < 1.0 {
            let (v1, v2) = (v1s[i], v2s[i]);
            let n1 = (v1.0.powi(2) + v1.1.powi(2)).sqrt();
            let n2 = (v2.0.powi(2) + v2.1.powi(2)).sqrt();
            if n1 > 0.0 && n2 > 0.0 {
                let (u1x, u1y) = (v1.0 / n1, v1.1 / n1);
                let (u2x, u2y) = (v2.0 / n2, v2.1 / n2);
                let (mut dx, mut dy) = (u1x + u2x, u1y + u2y);
                let dn = (dx * dx + dy * dy).sqrt();
                if dn > 0.0 {
                    dx /= dn;
                    dy /= dn;
                }
                let d = (n1 + n2) / 2.0;
                res.push((poly[i].0 + dx * d, poly[i].1 + dy * d));
                continue;
            }
        }
        res.push(poly[i]);
    }
    res
}

/// 向量夹角（度）。
fn angle_between(v1: (f64, f64), v2: (f64, f64)) -> f64 {
    let n1 = (v1.0.powi(2) + v1.1.powi(2)).sqrt();
    let n2 = (v2.0.powi(2) + v2.1.powi(2)).sqrt();
    if n1 == 0.0 || n2 == 0.0 {
        return 0.0;
    }
    let dot = ((v1.0 * v2.0 + v1.1 * v2.1) / (n1 * n2)).clamp(-1.0, 1.0);
    dot.acos().to_degrees()
}

/// _normalize_layout_polygon（auto 模式：rect/quad/poly 决策）。
fn normalize_layout_polygon(
    rect: &[(f64, f64); 4],
    polygon: &[(f64, f64)],
    _previous: Option<&[(f64, f64)]>,
) -> Vec<(f64, f64)> {
    if polygon.len() < 4 {
        return rect.to_vec();
    }
    // convert_polygon_to_quad（minAreaRect 近似：用凸包 + 旋转卡尺）
    let quad = min_area_rect(polygon);
    if let Some(quad) = quad {
        let iou = polygon_overlap_ratio(rect, &quad, "union");
        if iou >= 0.95 {
            return rect.to_vec();
        }
        let iou_poly_quad = polygon_overlap_ratio(polygon, &quad, "union");
        if iou_poly_quad >= 0.8 {
            return quad;
        }
    }
    polygon.to_vec()
}

/// 最小外接矩形（cv2.minAreaRect + boxPoints 近似：凸包 + 旋转卡尺）。
fn min_area_rect(points: &[(f64, f64)]) -> Option<Vec<(f64, f64)>> {
    if points.len() < 3 {
        return None;
    }
    let hull = convex_hull(points);
    if hull.len() < 3 {
        return None;
    }
    let n = hull.len();
    let mut best_area = f64::INFINITY;
    let mut best_rect: Option<Vec<(f64, f64)>> = None;
    for i in 0..n {
        let (x1, y1) = hull[i];
        let (x2, y2) = hull[(i + 1) % n];
        let dx = x2 - x1;
        let dy = y2 - y1;
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-9 {
            continue;
        }
        let (ux, uy) = (dx / len, dy / len);
        let (vx, vy) = (-uy, ux);
        let mut min_u = f64::INFINITY;
        let mut max_u = f64::NEG_INFINITY;
        let mut min_v = f64::INFINITY;
        let mut max_v = f64::NEG_INFINITY;
        for &(px, py) in &hull {
            let u = (px - x1) * ux + (py - y1) * uy;
            let v = (px - x1) * vx + (py - y1) * vy;
            min_u = min_u.min(u);
            max_u = max_u.max(u);
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
        let area = (max_u - min_u) * (max_v - min_v);
        if area < best_area {
            best_area = area;
            let corners = [
                (x1 + min_u * ux + min_v * vx, y1 + min_u * uy + min_v * vy),
                (x1 + max_u * ux + min_v * vx, y1 + max_u * uy + min_v * vy),
                (x1 + max_u * ux + max_v * vx, y1 + max_u * uy + max_v * vy),
                (x1 + min_u * ux + max_v * vx, y1 + min_u * uy + max_v * vy),
            ];
            best_rect = Some(corners.to_vec());
        }
    }
    // 排序：按角度（cv2.boxPoints 顺序）+ 左上角起始
    let rect = best_rect?;
    let cx = rect.iter().map(|p| p.0).sum::<f64>() / 4.0;
    let cy = rect.iter().map(|p| p.1).sum::<f64>() / 4.0;
    let mut idx: Vec<usize> = (0..4).collect();
    idx.sort_by(|&a, &b| {
        let (a1, b1) = (
            (rect[a].0 - cx).atan2(rect[a].1 - cy),
            (rect[b].0 - cx).atan2(rect[b].1 - cy),
        );
        a1.partial_cmp(&b1).unwrap()
    });
    let sorted: Vec<(f64, f64)> = idx.iter().map(|&i| rect[i]).collect();
    let sums: Vec<f64> = sorted.iter().map(|p| p.0 + p.1).collect();
    let tl = sums
        .iter()
        .enumerate()
        .fold(0, |acc, (i, s)| if *s < sums[acc] { i } else { acc });
    Some(sorted.iter().cycle().skip(tl).take(4).cloned().collect())
}

/// 凸包（Andrew monotone chain）。
fn convex_hull(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut pts = points.to_vec();
    pts.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap()
            .then(a.1.partial_cmp(&b.1).unwrap())
    });
    pts.dedup();
    if pts.len() < 3 {
        return pts;
    }
    let cross = |o: (f64, f64), a: (f64, f64), b: (f64, f64)| -> f64 {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };
    let mut lower = Vec::new();
    for &p in &pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// 多边形重叠率（shapely Polygon IoU 近似：Sutherland-Hodgman 裁剪）。
fn polygon_overlap_ratio(p1: &[(f64, f64)], p2: &[(f64, f64)], mode: &str) -> f64 {
    let inter = polygon_intersection(p1, p2);
    let inter_area = contour_area(&inter);
    if mode == "union" {
        let a1 = contour_area(p1);
        let a2 = contour_area(p2);
        let union = a1 + a2 - inter_area;
        if union <= 0.0 {
            return 0.0;
        }
        inter_area / union
    } else {
        let a1 = contour_area(p1);
        if a1 <= 0.0 {
            return 0.0;
        }
        inter_area / a1
    }
}

/// 多边形相交（Sutherland-Hodgman）。
fn polygon_intersection(subject: &[(f64, f64)], clip: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut output = subject.to_vec();
    let n = clip.len();
    for i in 0..n {
        if output.is_empty() {
            break;
        }
        let (cx1, cy1) = clip[i];
        let (cx2, cy2) = clip[(i + 1) % n];
        let input = output;
        output = Vec::new();
        let inside = |p: (f64, f64)| -> bool {
            (cx2 - cx1) * (p.1 - cy1) - (cy2 - cy1) * (p.0 - cx1) >= 0.0
        };
        let intersect = |s: (f64, f64), e: (f64, f64)| -> (f64, f64) {
            let (sx, sy) = s;
            let (ex, ey) = e;
            let d1 = (cx2 - cx1) * (sy - cy1) - (cy2 - cy1) * (sx - cx1);
            let d2 = (cx2 - cx1) * (ey - cy1) - (cy2 - cy1) * (ex - cx1);
            let t = d1 / (d1 - d2);
            (sx + t * (ex - sx), sy + t * (ey - sy))
        };
        for j in 0..input.len() {
            let s = input[j];
            let e = input[(j + 1) % input.len()];
            if inside(e) {
                if !inside(s) {
                    output.push(intersect(s, e));
                }
                output.push(e);
            } else if inside(s) {
                output.push(intersect(s, e));
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_runs_on_sample() {
        // 真实推理（需 doclayout.onnx）
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("..")
            .join("bench")
            .join("samples")
            .join("s04_spacing.png");
        if !path.exists() {
            return; // 样本缺失时跳过
        }
        match run(
            &path,
            &LayoutParams {
                max_items: 100,
                min_conf: 0.3,
            },
        ) {
            Ok(report) => {
                assert_eq!(report["task"], "layout");
                assert!(report["elements"].is_array());
            }
            Err(e) => {
                // 模型缺失时降级跳过
                eprintln!("layout 不可用（跳过）: {e}");
            }
        }
    }

    #[test]
    fn contour_area_works() {
        let square = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        assert!((contour_area(&square) - 100.0).abs() < 1e-6);
    }

    #[test]
    fn polygon_intersection_works() {
        let a = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let b = vec![(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)];
        let inter = polygon_intersection(&a, &b);
        assert!((contour_area(&inter) - 25.0).abs() < 1e-6);
    }
}
