//! 单目深度传感器（对应 Python `vs_depth.py`：MiDaS-small ONNX，schema v3 输出一致）。
//!
//! MiDaS-small → 逆深度图 → 相对深度统计（近/中/远分布）+ 区域深度。

use std::path::Path;

use image::imageops::FilterType;
use serde_json::{json, Value};

use crate::image::load_rgb;

/// 深度参数（对应 Python `--bins/--region`）。
pub struct DepthParams {
    pub bins: u32,
    pub region: Option<String>,
}

/// 模型路径（与 Python `WEIGHTS` 一致）。
fn weights_path() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    format!("{home}/.cache/vsensor/midas_v21_small_256.onnx")
}

/// 深度统计（对应 Python `depth_stats`）。
fn depth_stats(d: &[f32]) -> Value {
    let mut sorted = d.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let q = |f: f64| -> f64 {
        if n == 0 {
            return 0.0;
        }
        let idx = ((n as f64 - 1.0) * f).round() as usize;
        f64::from(sorted[idx.min(n - 1)])
    };
    let min = sorted.first().copied().unwrap_or(0.0);
    let max = sorted.last().copied().unwrap_or(0.0);
    let mean = if n > 0 {
        d.iter().sum::<f32>() / n as f32
    } else {
        0.0
    };
    json!({
        "min": (f64::from(min) * 1000.0).round() / 1000.0,
        "max": (f64::from(max) * 1000.0).round() / 1000.0,
        "mean": (f64::from(mean) * 1000.0).round() / 1000.0,
        "q25": (q(0.25) * 1000.0).round() / 1000.0,
        "median": (q(0.5) * 1000.0).round() / 1000.0,
        "q75": (q(0.75) * 1000.0).round() / 1000.0,
    })
}

/// 运行深度传感器（对应 Python `main()`）。
pub fn run(path: &Path, p: &DepthParams) -> Result<Value, String> {
    let img = load_rgb(path)?;
    let (w, h) = (img.width, img.height);

    // 预处理：resize 256×256 LANCZOS → /255 → ImageNet 归一化 → CHW → batch 1
    let resized = image::imageops::resize(
        &image::RgbImage::from_raw(w, h, img.data.clone()).ok_or("image decode")?,
        256,
        256,
        FilterType::Lanczos3,
    );
    let mut data = vec![0.0f32; 3 * 256 * 256];
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];
    for y in 0..256usize {
        for x in 0..256usize {
            let px = resized.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let v = f32::from(px[c]) / 255.0;
                data[c * 256 * 256 + y * 256 + x] = (v - mean[c]) / std[c];
            }
        }
    }

    // ort 推理
    let model_path = weights_path();
    if !Path::new(&model_path).exists() {
        return Err(format!(
            "模型不存在: {model_path}（需先下载 midas_v21_small_256.onnx）"
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
    let tensor = ort::value::Tensor::from_array(([1usize, 3, 256, 256], data.into_boxed_slice()))
        .map_err(|e| format!("ort tensor: {e}"))?;
    let outputs = session
        .run(ort::inputs! { input_name => tensor })
        .map_err(|e| format!("ort run: {e}"))?;
    let (shape, depth_data) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("ort extract: {e}"))?;
    let n = shape.num_elements();
    let depth_data = &depth_data[..n];

    // 后处理：模型输出 (1, 1, 256, 256) → 取 (256, 256) 逆深度
    let (dh, dw) = (256usize, 256usize);
    let dmin = depth_data.iter().cloned().fold(f32::INFINITY, f32::min);
    let dmax = depth_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    // 归一化到 [0,1] → uint8 → resize 回原图（BILINEAR）
    let norm: Vec<u8> = depth_data
        .iter()
        .map(|v| (((v - dmin) / (dmax - dmin + 1e-9)) * 255.0) as u8)
        .collect();
    let depth_img =
        image::GrayImage::from_raw(dw as u32, dh as u32, norm).ok_or("depth 图构造失败")?;
    let depth_full = image::imageops::resize(&depth_img, w, h, FilterType::Triangle);
    let depth_full_f: Vec<f32> = depth_full.iter().map(|v| f32::from(*v) / 255.0).collect();

    // 全局深度统计
    let global_stats = depth_stats(&depth_full_f);

    // 深度分箱（近/中/远）
    let mut bins = Vec::new();
    for i in 0..p.bins {
        let lo = i as f32 / p.bins as f32;
        let hi = (i + 1) as f32 / p.bins as f32;
        let mask_count = depth_full_f.iter().filter(|&&v| v >= lo && v < hi).count();
        let area_ratio = mask_count as f64 / depth_full_f.len() as f64;
        bins.push(json!({
            "bin": i,
            "range": [(lo as f64 * 100.0).round() / 100.0, (hi as f64 * 100.0).round() / 100.0],
            "area_ratio": (area_ratio * 10000.0).round() / 10000.0,
        }));
    }

    // 区域深度
    let region_stats = match &p.region {
        Some(region) => {
            let parts: Vec<i64> = region
                .split(',')
                .filter_map(|v| v.trim().parse().ok())
                .collect();
            if parts.len() == 4 {
                let (x1, y1, x2, y2) = (parts[0], parts[1], parts[2], parts[3]);
                let mut region_vals = Vec::new();
                for y in y1..y2 {
                    for x in x1..x2 {
                        if y >= 0 && x >= 0 && y < h as i64 && x < w as i64 {
                            region_vals.push(depth_full_f[(y as usize) * w as usize + x as usize]);
                        }
                    }
                }
                if !region_vals.is_empty() {
                    Some(depth_stats(&region_vals))
                } else {
                    None
                }
            } else {
                None
            }
        }
        None => None,
    };

    let mut report = crate::schema::envelope(
        Some("depth"),
        &["depth"],
        Some("image_px"),
        Some(json!({
            "type": "image",
            "path": path.display().to_string(),
            "size_px": [w, h],
            "device": "cpu",
        })),
    );
    report["depth"] = json!({
        "global": global_stats,
        "bins": bins,
        "region": region_stats,
        "inverse": true,
    });
    report["metrics"] = json!({
        "bins": p.bins,
        "depth_range": [
            (f64::from(depth_full_f.iter().cloned().fold(f32::INFINITY, f32::min)) * 1000.0).round() / 1000.0,
            (f64::from(depth_full_f.iter().cloned().fold(f32::NEG_INFINITY, f32::max)) * 1000.0).round() / 1000.0,
        ],
        "device": "cpu",
    });
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_stats_known() {
        // 已知数组的统计
        let d = vec![0.0f32, 0.25, 0.5, 0.75, 1.0];
        let s = depth_stats(&d);
        assert_eq!(s["min"], 0.0);
        assert_eq!(s["max"], 1.0);
        assert_eq!(s["median"], 0.5);
        assert_eq!(s["q25"], 0.25);
        assert_eq!(s["q75"], 0.75);
    }

    #[test]
    fn model_missing_returns_error() {
        let path = Path::new("/nonexistent.png");
        let r = run(
            path,
            &DepthParams {
                bins: 3,
                region: None,
            },
        );
        assert!(r.is_err());
    }
}
