//! 前景分割传感器（对应 Python `vs_segment.py`：MobileSAM，schema v3 输出一致）。
//!
//! 混合架构（S4 阶段）：
//! - **Rust**：预处理（resize longest side 1024 + 归一化 + pad）+ image encoder
//!   （MobileSAM TinyViT ONNX）→ embeddings
//! - **Python 回退**：mask decoder（prompt encoder + transformer + 上采样）经
//!   `vs_segment_decoder.py` 子进程执行（完整移植 Transformer 数值一致性风险高，
//!   见 plan.md S4 风险表；后续可替换为纯 Rust 实现）
//! - **Rust**：后处理（bbox / 面积比 / 实例合并 / 输出结构）

use std::path::Path;

use image::imageops::FilterType;
use serde_json::{json, Value};

use crate::bbox::bbox_iou;
use crate::image::load_rgb;

/// 分割参数（对应 Python `--min-area-ratio/--max-instances`）。
pub struct SegmentParams {
    /// 点提示（像素坐标）；空则用 3×3 网格点
    pub points: Vec<[f64; 2]>,
    /// 点标签（1=前景，0=背景）
    pub labels: Vec<i64>,
    /// 最小面积比（过滤小实例）
    pub min_area: f64,
    /// 最大实例数
    pub max_instances: u32,
}

impl Default for SegmentParams {
    fn default() -> Self {
        Self {
            points: vec![],
            labels: vec![],
            min_area: 0.01,
            max_instances: 10,
        }
    }
}

/// 模型路径（与 Python `WEIGHTS` 一致）。
fn weights_path() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    format!("{home}/.cache/vsensor/mobile_sam_encoder.onnx")
}

/// prompt encoder ONNX 路径。
fn prompt_encoder_path() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    format!("{home}/.cache/vsensor/mobile_sam_prompt_encoder.onnx")
}

/// mask decoder ONNX 路径。
fn mask_decoder_path() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    format!("{home}/.cache/vsensor/mobile_sam_mask_decoder.onnx")
}

/// 预处理：resize longest side 1024（round half up，BICUBIC）→
/// 归一化（SAM pixel_mean/std）→ pad 到 1024×1024 → CHW。
/// 返回 (CHW 数据, 原图宽高)。
fn preprocess(img: &crate::image::RgbImage) -> Result<(Vec<f32>, u32, u32), String> {
    let (w, h) = (img.width, img.height);
    // resize longest side 到 1024（PIL ResizeLongestSide：round half up）
    let scale = 1024.0f64 / f64::from(w.max(h));
    let neww = (f64::from(w) * scale + 0.5) as u32;
    let newh = (f64::from(h) * scale + 0.5) as u32;
    let resized = image::imageops::resize(
        &image::RgbImage::from_raw(w, h, img.data.clone()).ok_or("image decode")?,
        neww,
        newh,
        FilterType::Triangle, // torchvision resize 默认 BILINEAR（非 PIL BICUBIC）
    );
    // 归一化 + pad 到 1024×1024 + CHW
    let mean = [123.675f32, 116.28, 103.53];
    let std = [58.395f32, 57.12, 57.375];
    let mut data = vec![0.0f32; 3 * 1024 * 1024];
    for y in 0..newh as usize {
        for x in 0..neww as usize {
            let px = resized.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let v = f32::from(px[c]);
                data[c * 1024 * 1024 + y * 1024 + x] = (v - mean[c]) / std[c];
            }
        }
    }
    Ok((data, w, h))
}

/// 双线性 resize 2D 单通道（torch F.interpolate bilinear align_corners=False 等价）。
fn bilinear_resize_2d(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; dw * dh];
    if sw == dw && sh == dh {
        out.copy_from_slice(src);
        return out;
    }
    for y in 0..dh {
        // torch F.interpolate bilinear align_corners=False：
        //   src_idx = scale*(dst+0.5) - 0.5；src_idx < 0 → clamp 到 0
        let sy = ((y as f64 + 0.5) * sh as f64 / dh as f64 - 0.5).max(0.0);
        let y0 = sy.floor() as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let fy = (sy - y0 as f64) as f32;
        for x in 0..dw {
            let sx = ((x as f64 + 0.5) * sw as f64 / dw as f64 - 0.5).max(0.0);
            let x0 = sx.floor() as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let fx = (sx - x0 as f64) as f32;
            let p00 = src[y0 * sw + x0];
            let p01 = src[y0 * sw + x1];
            let p10 = src[y1 * sw + x0];
            let p11 = src[y1 * sw + x1];
            out[y * dw + x] =
                (p00 * (1.0 - fx) + p01 * fx) * (1.0 - fy) + (p10 * (1.0 - fx) + p11 * fx) * fy;
        }
    }
    out
}

/// 裁剪 2D 单通道到 (ch, cw)（左上角对齐）。
fn crop_2d(src: &[f32], sw: usize, sh: usize, ch: usize, cw: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; ch * cw];
    for y in 0..ch.min(sh) {
        for x in 0..cw.min(sw) {
            out[y * cw + x] = src[y * sw + x];
        }
    }
    out
}

/// 运行分割传感器（对应 Python `main()`）。
pub fn run(path: &Path, p: &SegmentParams) -> Result<Value, String> {
    let img = load_rgb(path)?;
    let (w, h) = (img.width, img.height);
    let (data, _, _) = preprocess(&img)?;

    // 1) image encoder ONNX → embeddings
    let model_path = weights_path();
    if !Path::new(&model_path).exists() {
        return Err(format!(
            "模型不存在: {model_path}（需先导出 mobile_sam_encoder.onnx）"
        ));
    }
    let mut session = ort::session::Session::builder()
        .map_err(|e| format!("ort builder: {e}"))?
        .commit_from_file(&model_path)
        .map_err(|e| format!("ort load {model_path}: {e}"))?;
    let input_name = session
        .inputs()
        .first()
        .map(|i| i.name().to_string())
        .ok_or("模型无输入")?;
    let tensor = ort::value::Tensor::from_array(([1usize, 3, 1024, 1024], data.into_boxed_slice()))
        .map_err(|e| format!("ort tensor: {e}"))?;
    let outputs = session
        .run(ort::inputs! { input_name => tensor })
        .map_err(|e| format!("ort run: {e}"))?;
    let (shape, emb_data) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("ort extract: {e}"))?;
    let n = shape.num_elements();

    // 2) 纯 Rust 解码：prompt_encoder + mask_decoder（ONNX，替代 Python 回退）
    let emb_data_f32: Vec<f32> = emb_data[..n].to_vec();

    // 提示：点或网格
    let (points, labels): (Vec<Vec<f64>>, Vec<i64>) = if !p.points.is_empty() {
        (
            p.points.iter().map(|pt| pt.to_vec()).collect(),
            if p.labels.is_empty() {
                vec![1; p.points.len()]
            } else {
                p.labels.clone()
            },
        )
    } else {
        // 3×3 网格点（对应 Python 默认）
        let grid = 3;
        let mut pts = Vec::new();
        let mut lbs = Vec::new();
        for gy in 0..grid {
            for gx in 0..grid {
                pts.push(vec![
                    ((gx as f64 + 0.5) * f64::from(w) / grid as f64) as i64 as f64,
                    ((gy as f64 + 0.5) * f64::from(h) / grid as f64) as i64 as f64,
                ]);
                lbs.push(1);
            }
        }
        (pts, lbs)
    };

    // 坐标缩放到 1024 空间（对应 SamPredictor.apply_coords）
    let scale = 1024.0f64 / f64::from(w.max(h));
    let neww = (f64::from(w) * scale + 0.5) as u32;
    let newh = (f64::from(h) * scale + 0.5) as u32;

    // 加载 prompt_encoder + mask_decoder
    let pe_path = prompt_encoder_path();
    let md_path = mask_decoder_path();
    if !Path::new(&pe_path).exists() || !Path::new(&md_path).exists() {
        return Err(format!(
            "decoder 模型不存在: {pe_path} / {md_path}（需先导出 mobile_sam_prompt_encoder/mask_decoder.onnx）"
        ));
    }
    let mut pe_sess = ort::session::Session::builder()
        .map_err(|e| format!("ort builder: {e}"))?
        .commit_from_file(&pe_path)
        .map_err(|e| format!("ort load prompt_encoder: {e}"))?;
    let mut md_sess = ort::session::Session::builder()
        .map_err(|e| format!("ort builder: {e}"))?
        .commit_from_file(&md_path)
        .map_err(|e| format!("ort load mask_decoder: {e}"))?;

    // image_pe = prompt_encoder.get_dense_pe()（位置编码，与输入无关）
    // prompt_encoder.onnx 第三个输出 image_pe
    let image_pe: Vec<f32> = {
        let dummy_pts = ort::value::Tensor::from_array((
            [1usize, 1, 2],
            vec![512.0f32, 512.0].into_boxed_slice(),
        ))
        .map_err(|e| format!("ort tensor: {e}"))?;
        let dummy_lbl =
            ort::value::Tensor::from_array(([1usize, 1], vec![1i64].into_boxed_slice()))
                .map_err(|e| format!("ort tensor: {e}"))?;
        let pe_out = pe_sess
            .run(ort::inputs! { "points" => dummy_pts, "labels" => dummy_lbl })
            .map_err(|e| format!("prompt_encoder run: {e}"))?;
        let (pe_shape, pe_data) = pe_out[2]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("ort extract: {e}"))?;
        let pe_n = pe_shape.num_elements();
        pe_data[..pe_n].to_vec() // (1,256,64,64)
    };

    // 逐 prompt 解码（对应 Python 逐 prompt 循环）
    let mut masks_out: Vec<Vec<bool>> = Vec::new();
    let mut scores: Vec<f64> = Vec::new();
    for pi in 0..points.len() {
        // prompt_encoder：点 → sparse + dense
        let sx = points[pi][0] * (f64::from(neww) / f64::from(w));
        let sy = points[pi][1] * (f64::from(newh) / f64::from(h));
        let lb = if pi < labels.len() { labels[pi] } else { 1 };
        let pt_t = ort::value::Tensor::from_array((
            [1usize, 1, 2],
            vec![sx as f32, sy as f32].into_boxed_slice(),
        ))
        .map_err(|e| format!("ort tensor: {e}"))?;
        let lb_t = ort::value::Tensor::from_array(([1usize, 1], vec![lb].into_boxed_slice()))
            .map_err(|e| format!("ort tensor: {e}"))?;
        let pe_out = pe_sess
            .run(ort::inputs! { "points" => pt_t, "labels" => lb_t })
            .map_err(|e| format!("prompt_encoder run: {e}"))?;
        let (s_shape, s_data) = pe_out[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("ort extract: {e}"))?;
        let s_n = s_shape.num_elements();
        let sparse: Vec<f32> = s_data[..s_n].to_vec(); // (1,2,256)
        let (d_shape, d_data) = pe_out[1]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("ort extract: {e}"))?;
        let d_n = d_shape.num_elements();
        let dense: Vec<f32> = d_data[..d_n].to_vec(); // (1,256,64,64)

        // mask_decoder：embeddings + image_pe + sparse + dense → masks + iou
        let emb_t = ort::value::Tensor::from_array((
            [1usize, 256, 64, 64],
            emb_data_f32.clone().into_boxed_slice(),
        ))
        .map_err(|e| format!("ort tensor: {e}"))?;
        let pe_t = ort::value::Tensor::from_array((
            [1usize, 256, 64, 64],
            image_pe.clone().into_boxed_slice(),
        ))
        .map_err(|e| format!("ort tensor: {e}"))?;
        let sp_t = ort::value::Tensor::from_array(([1usize, 2, 256], sparse.into_boxed_slice()))
            .map_err(|e| format!("ort tensor: {e}"))?;
        let dn_t =
            ort::value::Tensor::from_array(([1usize, 256, 64, 64], dense.into_boxed_slice()))
                .map_err(|e| format!("ort tensor: {e}"))?;
        let md_out = md_sess
            .run(ort::inputs! {
                "image_embeddings" => emb_t,
                "image_pe" => pe_t,
                "sparse_embeddings" => sp_t,
                "dense_embeddings" => dn_t,
            })
            .map_err(|e| format!("mask_decoder run: {e}"))?;
        let (m_shape, m_data) = md_out[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("ort extract: {e}"))?;
        let m_n = m_shape.num_elements();
        let masks_raw: Vec<f32> = m_data[..m_n].to_vec(); // (1,1,256,256)
        let (i_shape, i_data) = md_out[1]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("ort extract: {e}"))?;
        let i_n = i_shape.num_elements();
        let iou: f64 = i_data[..i_n].iter().copied().fold(0.0f32, |a, b| a + b) as f64;
        scores.push(iou);

        // postprocess_masks：1024×1024 → 裁剪 input_size → 原图尺寸（bilinear）
        // 注意：bilinear_resize_2d 参数为 (src, sw=宽度, sh=高度, dw=宽度, dh=高度)
        let mask_1024 = bilinear_resize_2d(&masks_raw, 256, 256, 1024, 1024);
        let mask_crop = crop_2d(&mask_1024, 1024, 1024, newh as usize, neww as usize);
        let mask_orig = bilinear_resize_2d(
            &mask_crop,
            neww as usize,
            newh as usize,
            w as usize,
            h as usize,
        );
        // masks > 0 → bool
        masks_out.push(mask_orig.iter().map(|&v| v > 0.0).collect());
    }

    // 3) 后处理（逐 mask 提取实例，对应 Python 逐 prompt 循环）
    let (_mh, mw) = (h as usize, w as usize);
    let n_masks = masks_out.len();
    let mut instances: Vec<Value> = Vec::new();
    for mi in 0..n_masks {
        let mask = &masks_out[mi];
        let mut ys: Vec<usize> = Vec::new();
        let mut xs: Vec<usize> = Vec::new();
        for (i, &m) in mask.iter().enumerate() {
            if m {
                ys.push(i / mw);
                xs.push(i % mw);
            }
        }
        if ys.len() < 50 {
            continue; // 去噪（对应 Python `if len(ys) < 50`）
        }
        let area_ratio = ys.len() as f64 / (f64::from(w) * f64::from(h));
        if area_ratio < p.min_area {
            continue;
        }
        let bbox = [
            *xs.iter().min().unwrap() as i64,
            *ys.iter().min().unwrap() as i64,
            *xs.iter().max().unwrap() as i64,
            *ys.iter().max().unwrap() as i64,
        ];
        let prompt = if !points.is_empty() {
            json!({"point": [points[mi.min(points.len() - 1)][0] as i64, points[mi.min(points.len() - 1)][1] as i64]})
        } else {
            json!(null)
        };
        instances.push(json!({
            "bbox": bbox,
            "area_ratio": (area_ratio * 10000.0).round() / 10000.0,
            "pixels": ys.len(),
            "score": scores.get(mi).copied().map(|v| (v * 1000.0).round() / 1000.0).unwrap_or(0.0),
            "prompt": prompt,
        }));
    }

    // 合并重叠实例（IoU > 0.5 取面积大的，对应 Python）
    instances.sort_by(|a, b| {
        b["pixels"]
            .as_i64()
            .unwrap_or(0)
            .cmp(&a["pixels"].as_i64().unwrap_or(0))
    });
    let mut merged: Vec<Value> = Vec::new();
    for inst in &instances {
        let mut dup = false;
        for m in &merged {
            let ia: Vec<f64> = inst["bbox"]
                .as_array()
                .map(|b| b.iter().filter_map(|v| v.as_f64()).collect())
                .unwrap_or_default();
            let ib: Vec<f64> = m["bbox"]
                .as_array()
                .map(|b| b.iter().filter_map(|v| v.as_f64()).collect())
                .unwrap_or_default();
            if bbox_iou(&ia, &ib) > 0.5 {
                dup = true;
                break;
            }
        }
        if !dup {
            merged.push(inst.clone());
        }
        if merged.len() >= p.max_instances as usize {
            break;
        }
    }
    let foreground = merged.first().cloned();

    // 输出（schema v3）
    let elements: Vec<Value> = merged
        .iter()
        .enumerate()
        .map(|(i, inst)| {
            let bbox: Vec<f64> = inst["bbox"]
                .as_array()
                .map(|b| b.iter().filter_map(|v| v.as_f64()).collect())
                .unwrap_or_default();
            let conf = (inst["area_ratio"].as_f64().unwrap_or(0.0) * 2.0).min(1.0);
            let mut el = serde_json::to_value(crate::schema::Element {
                id: i as i64,
                etype: "foreground",
                bbox: &bbox,
                text: None,
                conf: Some(conf),
                color: None,
                font: None,
                z: None,
                source: &["segment"],
                coordsys: "image_px",
            })
            .unwrap();
            el["segment"] = inst.clone();
            el
        })
        .collect();
    let mut report = crate::schema::envelope(
        Some("segment"),
        &["segment"],
        Some("image_px"),
        Some(json!({
            "type": "image",
            "path": path.display().to_string(),
            "size_px": [w, h],
            "device": "cpu",
        })),
    );
    report["elements"] = json!(elements);
    report["foreground"] = foreground.clone().unwrap_or(Value::Null);
    report["metrics"] = json!({
        "instances": merged.len(),
        "foreground_ratio": foreground.as_ref().and_then(|f| f["area_ratio"].as_f64()).unwrap_or(0.0),
        "device": "cpu",
    });
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preprocess_shape_and_values() {
        // 构造 2×2 纯红图 → 1024×1024 张量（longest side 1024）
        let img = crate::image::RgbImage {
            width: 2,
            height: 2,
            data: vec![255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0],
        };
        let (data, w, h) = preprocess(&img).unwrap();
        assert_eq!(data.len(), 3 * 1024 * 1024);
        assert_eq!((w, h), (2, 2));
        // 纯红像素：R 通道 (255 - 123.675)/58.395
        let expected_r = (255.0f32 - 123.675) / 58.395;
        assert!((data[0] - expected_r).abs() < 1e-3);
        // G 通道 (0 - 116.28)/57.12
        let expected_g = (0.0f32 - 116.28) / 57.12;
        assert!((data[1024 * 1024] - expected_g).abs() < 1e-3);
    }

    #[test]
    fn bilinear_resize_roundtrip() {
        // 2×2 → 4×4 双线性 resize（torch interpolate 语义）
        let src = vec![0.0f32, 1.0, 1.0, 0.0];
        let out = bilinear_resize_2d(&src, 2, 2, 4, 4);
        assert_eq!(out.len(), 16);
        // 中心像素（双线性插值 0.375）
        assert!((out[5] - 0.375).abs() < 0.01);
        // 裁剪
        let cropped = crop_2d(&out, 4, 4, 2, 2);
        assert_eq!(cropped.len(), 4);
    }

    #[test]
    fn model_missing_returns_error() {
        // 模型不存在时返回 Err（不 panic）
        let path = Path::new("/nonexistent.png");
        let r = run(path, &SegmentParams::default());
        assert!(r.is_err());
    }
}
