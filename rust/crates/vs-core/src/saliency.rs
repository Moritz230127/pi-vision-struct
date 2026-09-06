//! 显著性传感器（对应 Python `vs_saliency.py`：U²-Net ONNX，schema v3 输出一致）。
//!
//! U²-Net 显著性检测 → 显著性图 → top-N 候选区域 bbox + score。
//! 候选区域是 zoom-in 协议的入口。

use std::path::Path;

use image::imageops::FilterType;
use serde_json::{json, Value};

use crate::image::load_rgb;

/// 显著性参数（对应 Python `--top-n/--min-score`）。
pub struct SaliencyParams {
    pub top_n: u32,
    pub min_score: f64,
}

/// 模型路径（与 Python `WEIGHTS` 一致）。
fn weights_path() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    format!("{home}/.cache/vsensor/u2net.onnx")
}

/// 预处理：resize 320×320 LANCZOS → /255 → ImageNet 归一化 → CHW → batch 1。
/// 返回 (CHW 数据, 原图宽高)。
fn preprocess(img: &crate::image::RgbImage) -> Result<(Vec<f32>, u32, u32), String> {
    let (w, h) = (img.width, img.height);
    // resize 到 320×320（PIL LANCZOS 近似）
    let resized = image::imageops::resize(
        &image::RgbImage::from_raw(w, h, img.data.clone()).ok_or("image decode")?,
        320,
        320,
        FilterType::Lanczos3,
    );
    let mut data = vec![0.0f32; 3 * 320 * 320];
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];
    for y in 0..320usize {
        for x in 0..320usize {
            let px = resized.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let v = f32::from(px[c]) / 255.0;
                data[c * 320 * 320 + y * 320 + x] = (v - mean[c]) / std[c];
            }
        }
    }
    Ok((data, w, h))
}

/// 4-连通域标记（对应 scipy `ndimage.label` 默认 connectivity=1）。
/// 返回 (labels, 连通域数)。
fn connected_components(binary: &[bool], w: usize, h: usize) -> (Vec<u32>, u32) {
    let mut labels = vec![0u32; w * h];
    let mut next = 0u32;
    let mut stack: Vec<usize> = Vec::new();
    for i in 0..w * h {
        if binary[i] && labels[i] == 0 {
            next += 1;
            labels[i] = next;
            stack.push(i);
            while let Some(p) = stack.pop() {
                let (x, y) = (p % w, p / w);
                // 4-连通邻域
                for (dx, dy) in [(1i64, 0i64), (-1, 0), (0, 1), (0, -1)] {
                    let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                    if nx >= 0 && ny >= 0 && nx < w as i64 && ny < h as i64 {
                        let ni = (ny as usize) * w + nx as usize;
                        if binary[ni] && labels[ni] == 0 {
                            labels[ni] = next;
                            stack.push(ni);
                        }
                    }
                }
            }
        }
    }
    (labels, next)
}

/// 运行显著性传感器（对应 Python `main()`）。
pub fn run(path: &Path, p: &SaliencyParams) -> Result<Value, String> {
    let img = load_rgb(path)?;
    let (w, h) = (img.width, img.height);
    let (data, _, _) = preprocess(&img)?;

    // ort 推理（CPU provider；CUDA 需额外库，S4 阶段 CPU 即可）
    let model_path = weights_path();
    if !Path::new(&model_path).exists() {
        return Err(format!("模型不存在: {model_path}（需先下载 u2net.onnx）"));
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
    let tensor = ort::value::Tensor::from_array(([1usize, 3, 320, 320], data.into_boxed_slice()))
        .map_err(|e| format!("ort tensor: {e}"))?;
    let outputs = session
        .run(ort::inputs! { input_name => tensor })
        .map_err(|e| format!("ort run: {e}"))?;
    let (shape, sal_data) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("ort extract: {e}"))?;
    let n = shape.num_elements();
    let sal_data = &sal_data[..n];

    // 后处理：模型输出 (1, 1, 320, 320) → 取 (320, 320)；clip [0,1]
    let (sh, sw) = (320usize, 320usize);
    let sal_clipped: Vec<f32> = sal_data.iter().map(|v| v.clamp(0.0, 1.0)).collect();

    // 上采样回原图（BILINEAR）
    let sal_img = image::GrayImage::from_raw(
        sw as u32,
        sh as u32,
        sal_clipped.iter().map(|v| (v * 255.0) as u8).collect(),
    )
    .ok_or("sal 图构造失败")?;
    let sal_full = image::imageops::resize(&sal_img, w, h, FilterType::Triangle);
    let sal_full_f: Vec<f32> = sal_full.iter().map(|v| f32::from(*v) / 255.0).collect();

    // 连通域提取候选区域
    let binary: Vec<bool> = sal_full_f.iter().map(|v| *v > p.min_score as f32).collect();
    let (labels, n) = connected_components(&binary, w as usize, h as usize);
    let mut candidates: Vec<Value> = Vec::new();
    for i in 1..=n {
        let mut xs_min = usize::MAX;
        let mut ys_min = usize::MAX;
        let mut xs_max = 0usize;
        let mut ys_max = 0usize;
        let mut count = 0usize;
        let mut score_sum = 0.0f64;
        for (idx, &l) in labels.iter().enumerate() {
            if l == i {
                let (x, y) = (idx % w as usize, idx / w as usize);
                xs_min = xs_min.min(x);
                ys_min = ys_min.min(y);
                xs_max = xs_max.max(x);
                ys_max = ys_max.max(y);
                count += 1;
                score_sum += f64::from(sal_full_f[idx]);
            }
        }
        if count < 50 {
            continue; // 去噪
        }
        let score = score_sum / count as f64;
        candidates.push(json!({
            "bbox": [xs_min as i64, ys_min as i64, xs_max as i64, ys_max as i64],
            "score": (score * 1000.0).round() / 1000.0,
            "pixels": count,
        }));
    }
    candidates.sort_by(|a, b| {
        b["score"]
            .as_f64()
            .unwrap_or(0.0)
            .partial_cmp(&a["score"].as_f64().unwrap_or(0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    candidates.truncate(p.top_n as usize);

    // 输出（schema v3）
    let elements: Vec<Value> = candidates
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let bbox: Vec<f64> = c["bbox"]
                .as_array()
                .map(|b| b.iter().filter_map(|v| v.as_f64()).collect())
                .unwrap_or_default();
            let mut el = serde_json::to_value(crate::schema::Element {
                id: i as i64,
                etype: "candidate",
                bbox: &bbox,
                text: None,
                conf: c["score"].as_f64(),
                color: None,
                font: None,
                z: None,
                source: &["saliency"],
                coordsys: "image_px",
            })
            .unwrap();
            el["saliency"] = c.clone();
            el
        })
        .collect();
    let mut report = crate::schema::envelope(
        Some("saliency"),
        &["saliency"],
        Some("image_px"),
        Some(json!({
            "type": "image",
            "path": path.display().to_string(),
            "size_px": [w, h],
            "device": "cpu",
        })),
    );
    report["elements"] = json!(elements);
    report["candidates"] = json!(candidates);
    report["metrics"] = json!({
        "candidates": candidates.len(),
        "top_score": candidates.first().and_then(|c| c["score"].as_f64()).unwrap_or(0.0),
        "device": "cpu",
    });
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preprocess_shape_and_values() {
        // 构造 2×2 纯红图 → 320×320 张量
        let img = crate::image::RgbImage {
            width: 2,
            height: 2,
            data: vec![255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0],
        };
        let (data, w, h) = preprocess(&img).unwrap();
        assert_eq!(data.len(), 3 * 320 * 320);
        assert_eq!((w, h), (2, 2));
        // 纯红像素：R 通道 (1.0 - 0.485)/0.229
        let expected_r = (1.0f32 - 0.485) / 0.229;
        assert!((data[0] - expected_r).abs() < 1e-4);
        // G 通道 (0.0 - 0.456)/0.224
        let expected_g = (0.0f32 - 0.456) / 0.224;
        assert!((data[320 * 320] - expected_g).abs() < 1e-4);
    }

    #[test]
    fn connected_components_4conn() {
        // 对角两个像素（4-连通下不相连）
        let binary = vec![true, false, false, false, true, false, false, false, false];
        let (labels, n) = connected_components(&binary, 3, 3);
        assert_eq!(n, 2);
        assert_ne!(labels[0], labels[4]);
        // 相邻像素（4-连通下相连）
        let binary2 = vec![true, true, false, false, false, false, false, false, false];
        let (_, n2) = connected_components(&binary2, 3, 3);
        assert_eq!(n2, 1);
    }

    #[test]
    fn model_missing_returns_error() {
        // 模型不存在时返回 Err（不 panic）
        let path = Path::new("/nonexistent.png");
        let r = run(
            path,
            &SaliencyParams {
                top_n: 5,
                min_score: 0.3,
            },
        );
        assert!(r.is_err());
    }
}
