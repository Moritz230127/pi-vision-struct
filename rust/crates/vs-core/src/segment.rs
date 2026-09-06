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

/// Python 解码助手路径。
fn decoder_script() -> String {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
        .join("python");
    root.join("vs_segment_decoder.py").display().to_string()
}

/// vsensor env python。
fn vsensor_python() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    format!("{home}/conda-envs/vsensor/bin/python")
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

/// 写 npy 文件（float32 或 bool，简单格式）。
fn write_npy(path: &Path, shape: &[usize], data: &[u8], dtype: &str) -> Result<(), String> {
    // dtype: "f4" (float32) 或 "b1" (bool)
    let descr = match dtype {
        "f4" => "<f4",
        "b1" => "|b1",
        _ => return Err(format!("unsupported npy dtype {dtype}")),
    };
    let shape_str = shape
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let header =
        format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({shape_str},), }}");
    // header 对齐到 64 字节（含 magic 6 + version 2 + len 2）
    let mut header = header;
    let base = 6 + 2 + 2;
    while (base + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut out = Vec::with_capacity(base + header.len() + data.len());
    out.extend_from_slice(b"\x93NUMPY");
    out.extend_from_slice(&[1, 0]);
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(data);
    std::fs::write(path, out).map_err(|e| format!("write npy {path:?}: {e}"))
}

/// 读 npy 文件（float32 或 bool）。
fn read_npy(path: &Path) -> Result<(Vec<usize>, Vec<u8>), String> {
    let raw = std::fs::read(path).map_err(|e| format!("read npy {path:?}: {e}"))?;
    if raw.len() < 10 || &raw[0..6] != b"\x93NUMPY" {
        return Err("bad npy magic".into());
    }
    let header_len = u16::from_le_bytes([raw[8], raw[9]]) as usize;
    let header = String::from_utf8_lossy(&raw[10..10 + header_len]).to_string();
    // 解析 shape
    let shape_part = header
        .split("'shape': (")
        .nth(1)
        .and_then(|s| s.split(')').next())
        .unwrap_or("");
    let shape: Vec<usize> = shape_part
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let data = raw[10 + header_len..].to_vec();
    Ok((shape, data))
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
    let emb_bytes: Vec<u8> = emb_data[..n].iter().flat_map(|v| v.to_le_bytes()).collect();

    // 2) 保存 embeddings → Python 解码助手
    let tmp_dir = std::env::temp_dir();
    let emb_path = tmp_dir.join(format!("vsd_emb_{}.npy", std::process::id()));
    let mask_path = tmp_dir.join(format!("vsd_masks_{}.npy", std::process::id()));
    write_npy(&emb_path, &[1, 256, 64, 64], &emb_bytes, "f4")?;

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

    let req = json!({
        "embeddings": emb_path.display().to_string(),
        "image": path.display().to_string(),
        "points": points,
        "labels": labels,
        "out_masks": mask_path.display().to_string(),
    });
    let out = std::process::Command::new(vsensor_python())
        .arg(decoder_script())
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child
                .stdin
                .take()
                .unwrap()
                .write_all(req.to_string().as_bytes())?;
            child.wait_with_output()
        })
        .map_err(|e| format!("spawn decoder: {e}"))?;
    let resp: Value =
        serde_json::from_slice(&out.stdout).map_err(|e| format!("decoder 输出非 JSON: {e}"))?;
    if !resp.get("ok").and_then(|v| v.as_bool()).unwrap_or(false) {
        return Err(format!(
            "decoder 失败: {}",
            resp.get("error")
                .and_then(|e| e.as_str())
                .unwrap_or("unknown")
        ));
    }
    let _ = std::fs::remove_file(&emb_path);

    // 3) 读 masks → 后处理（逐 mask 提取实例，对应 Python 逐 prompt 循环）
    let (mask_shape, mask_bytes) = read_npy(&mask_path)?;
    let _ = std::fs::remove_file(&mask_path);
    let scores: Vec<f64> = resp
        .get("scores")
        .and_then(|s| s.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_f64()).collect())
        .unwrap_or_default();
    let n_masks = if mask_shape.len() == 3 {
        mask_shape[0]
    } else {
        1
    };
    let (mh, mw) = if mask_shape.len() == 3 {
        (mask_shape[1], mask_shape[2])
    } else {
        (mask_shape[0], mask_shape[1])
    };
    let mask_stride = mh * mw;
    let mut instances: Vec<Value> = Vec::new();
    for mi in 0..n_masks {
        let mask: Vec<bool> = mask_bytes[mi * mask_stride..(mi + 1) * mask_stride]
            .iter()
            .map(|b| *b != 0)
            .collect();
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
    fn npy_roundtrip() {
        let path = std::env::temp_dir().join("vsd_npy_test.npy");
        let data: Vec<u8> = vec![1u8, 2, 3, 4];
        write_npy(&path, &[2, 2], &data, "f4").unwrap();
        let (shape, out) = read_npy(&path).unwrap();
        assert_eq!(shape, vec![2, 2]);
        assert_eq!(out, data);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn model_missing_returns_error() {
        // 模型不存在时返回 Err（不 panic）
        let path = Path::new("/nonexistent.png");
        let r = run(path, &SegmentParams::default());
        assert!(r.is_err());
    }
}
