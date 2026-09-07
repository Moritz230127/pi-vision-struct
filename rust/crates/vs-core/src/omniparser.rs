//! OmniParser 传感器（对应 Python `vs_omniparser.py`，schema v3 输出逐字段一致）。
//!
//! 流水线：YOLOv9 icon_detect（ONNX）检测图标 → 裁剪 64×64 →
//! Florence-2 caption（vision_encoder + embed_tokens + language_encoder +
//! decoder 自回归生成循环，全部 ONNX）。文本部分由 OCR 传感器提供（S5）。

use std::path::Path;

use serde_json::{json, Value};

use crate::image::load_rgb;
use crate::util::resize_lanczos;

/// OmniParser 参数（对应 Python `--box-threshold/--max-items`）。
pub struct OmniparserParams {
    pub max_items: u32,
    pub min_conf: f64,
}

/// 模型目录。
fn model_dir() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    std::path::Path::new(&home).join(".cache").join("vsensor")
}

/// ort Shape → 维度数组（绕过 ndarray 版本冲突：ort 用 0.17，vs-core 用 0.16）。
fn shape_dims(shape: &ort::value::Shape) -> Vec<usize> {
    let mut dims = Vec::new();
    let remaining = shape.num_elements();
    // 从 Shape 的 Debug 输出解析维度（无公开 API）
    let dbg = format!("{shape:?}");
    let inner = dbg.trim_start_matches('[').trim_end_matches(']');
    if !inner.is_empty() {
        for part in inner.split(',') {
            if let Ok(d) = part.trim().parse::<i64>() {
                if d >= 0 {
                    dims.push(d as usize);
                }
            }
        }
    }
    if dims.is_empty() {
        dims.push(remaining);
    }
    dims
}

// ---------------------------------------------------------------- YOLOv9 检测

/// YOLOv9 检测结果：xyxy + score。
#[allow(dead_code)]
struct DetBox {
    bbox: [f64; 4],
    score: f64,
}

/// YOLOv9 解码（与 OmniParser `YOLOv9Detector._decode` 一致）：
/// 3 层（stride 8/16/32）→ class_logits + box_distances → sigmoid + anchor 解码。
fn decode_yolov9(
    outputs: &[(Vec<usize>, Vec<f32>)],
    img_w: usize,
    img_h: usize,
    conf: f64,
    scale: f64,
    pad_left: f64,
    pad_top: f64,
) -> Vec<DetBox> {
    if outputs.is_empty() {
        return Vec::new();
    }
    let strides = [8usize, 16, 32];
    let mut all_scores: Vec<f32> = Vec::new();
    let mut all_boxes: Vec<[f32; 4]> = Vec::new();
    for (layer, stride) in strides.iter().enumerate() {
        let (lshape, ldata) = &outputs[layer * 2]; // (1, 1, H, W)
        let (_, ddata) = &outputs[layer * 2 + 1]; // (1, 4, H, W)
        let h = lshape[2];
        let w = lshape[3];
        for y in 0..h {
            for x in 0..w {
                // sigmoid（Python `class_logits.sigmoid()`）
                let raw = ldata[y * w + x];
                let score = 1.0 / (1.0 + (-raw).exp());
                if (score as f64) < conf {
                    continue;
                }
                // dists 布局 (1, 4, H, W)：索引 (c*H+y)*W+x
                let (lt, tp, rt, bt) = (
                    ddata[y * w + x],
                    ddata[(h + y) * w + x],
                    ddata[(2 * h + y) * w + x],
                    ddata[(3 * h + y) * w + x],
                );
                let stride = *stride as f32;
                let cx = (x as f32 + 0.5) * stride;
                let cy = (y as f32 + 0.5) * stride;
                let x1 = cx - lt * stride;
                let y1 = cy - tp * stride;
                let x2 = cx + rt * stride;
                let y2 = cy + bt * stride;
                all_scores.push(score);
                all_boxes.push([x1, y1, x2, y2]);
            }
        }
    }
    // NMS（IoU 0.7，与 Python batched_nms 一致）
    let mut idx: Vec<usize> = (0..all_scores.len()).collect();
    idx.sort_by(|a, b| {
        all_scores[*b]
            .partial_cmp(&all_scores[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut keep: Vec<usize> = Vec::new();
    for &i in &idx {
        let mut ok = true;
        for &j in &keep {
            if iou(&all_boxes[i], &all_boxes[j]) > 0.7 {
                ok = false;
                break;
            }
        }
        if ok {
            keep.push(i);
        }
    }
    keep.iter()
        .map(|&i| {
            let b = all_boxes[i];
            // 坐标换算回原图（减 pad 除 scale，与 Python predict 一致）
            DetBox {
                bbox: [
                    ((b[0] as f64 - pad_left) / scale).clamp(0.0, img_w as f64),
                    ((b[1] as f64 - pad_top) / scale).clamp(0.0, img_h as f64),
                    ((b[2] as f64 - pad_left) / scale).clamp(0.0, img_w as f64),
                    ((b[3] as f64 - pad_top) / scale).clamp(0.0, img_h as f64),
                ],
                score: all_scores[i] as f64,
            }
        })
        .collect()
}

/// IoU（xyxy）。
fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);
    inter / (area_a + area_b - inter).max(1e-6)
}

/// YOLOv9 检测：加载 ONNX → 推理 → 解码 + NMS。
fn detect_icons(img: &crate::image::RgbImage, min_conf: f64) -> Result<Vec<DetBox>, String> {
    let model = model_dir().join("icon_detect.onnx");
    if !model.exists() {
        return Err(format!(
            "icon_detect.onnx 不存在: {model:?}（先运行 export_icon_detect_onnx.py）"
        ));
    }
    let mut session = ort::session::Session::builder()
        .map_err(|e| format!("ort builder: {e}"))?
        .commit_from_file(&model)
        .map_err(|e| format!("icon_detect 加载失败: {e}"))?;
    // 预处理：letterbox 到 640×640（与 Python `_preprocess` 一致：scale + pad 114）
    let (w, h) = (img.width as usize, img.height as usize);
    let scale = (640.0 / w as f64).min(640.0 / h as f64);
    let rw = (w as f64 * scale) as usize;
    let rh = (h as f64 * scale) as usize;
    let pad_left = (640 - rw) / 2;
    let pad_top = (640 - rh) / 2;
    let mut data = vec![114.0f32 / 255.0; 3 * 640 * 640]; // 灰边（channel 优先布局）
    // LANCZOS resize（PIL 精确复刻）
    let resized_data = resize_lanczos(img, rw as u32, rh as u32);
    for y in 0..rh {
        for x in 0..rw {
            let i = (y * rw + x) * 3;
            let di = (y + pad_top) * 640 + (x + pad_left);
            data[di] = resized_data[i] as f32 / 255.0;
            data[640 * 640 + di] = resized_data[i + 1] as f32 / 255.0;
            data[2 * 640 * 640 + di] = resized_data[i + 2] as f32 / 255.0;
        }
    }
    let tensor = ort::value::Tensor::from_array(([1usize, 3, 640, 640], data.into_boxed_slice()))
        .map_err(|e| format!("tensor: {e}"))?;
    let input_name = session.inputs()[0].name().to_string();
    let outputs = session
        .run(ort::inputs! { input_name => tensor })
        .map_err(|e| format!("icon_detect 推理失败: {e}"))?;
    let mut arrays: Vec<(Vec<usize>, Vec<f32>)> = Vec::new();
    for out in outputs {
        let (shape, data) = out
            .1
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("输出提取失败: {e}"))?;
        let n = shape.num_elements();
        let dims: Vec<usize> = shape_dims(shape);
        arrays.push((dims, data[..n].to_vec()));
    }
    Ok(decode_yolov9(
        &arrays,
        w,
        h,
        min_conf,
        scale,
        pad_left as f64,
        pad_top as f64,
    ))
}

// ---------------------------------------------------------------- Florence-2 caption

/// Florence-2 caption：裁剪图标 → 64×64 → vision_encoder → language_encoder → decoder 自回归。
fn caption_icon(
    img: &crate::image::RgbImage,
    box_xyxy: &[f64; 4],
    tokenizer: &tokenizers::Tokenizer,
    sessions: &FlorenceSessions,
) -> Result<String, String> {
    // 1) 裁剪 + resize 64×64（双线性，与 cv2.resize 近似）
    let (x1, y1, x2, y2) = (
        box_xyxy[0].max(0.0) as usize,
        box_xyxy[1].max(0.0) as usize,
        (box_xyxy[2] as usize).min(img.width as usize),
        (box_xyxy[3] as usize).min(img.height as usize),
    );
    if x2 <= x1 || y2 <= y1 {
        return Ok(String::new());
    }
    let cw = x2 - x1;
    let ch = y2 - y1;
    let mut px = vec![0.0f32; 3 * 64 * 64];
    for y in 0..64 {
        for x in 0..64 {
            let sx = x1 + (x as f64 * cw as f64 / 64.0) as usize;
            let sy = y1 + (y as f64 * ch as f64 / 64.0) as usize;
            let si = (sy * img.width as usize + sx) * 3;
            let di = (y * 64 + x) * 3;
            // CLIP normalize（ImageNet mean/std）
            px[di] = (img.data[si] as f32 / 255.0 - 0.485) / 0.229;
            px[di + 1] = (img.data[si + 1] as f32 / 255.0 - 0.456) / 0.224;
            px[di + 2] = (img.data[si + 2] as f32 / 255.0 - 0.406) / 0.225;
        }
    }
    // 2) vision_encoder
    let tensor = ort::value::Tensor::from_array(([1usize, 3, 64, 64], px.into_boxed_slice()))
        .map_err(|e| format!("tensor: {e}"))?;
    let mut vision_guard = sessions.vision.lock().unwrap();
    let out = vision_guard
        .run(ort::inputs! { "pixel_values" => tensor })
        .map_err(|e| format!("vision_encoder: {e}"))?;
    let (shape, data) = out[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("image_features: {e}"))?;
    let n = shape.num_elements();
    let image_features: Vec<f32> = data[..n].to_vec();
    // 3) tokenize "<CAPTION>"
    let encoding = tokenizer
        .encode("<CAPTION>", true)
        .map_err(|e| format!("tokenize: {e}"))?;
    let ids: Vec<i64> = encoding.get_ids().iter().map(|&i| i as i64).collect();
    let ids_len = ids.len();
    // 4) embed_tokens
    let ids_tensor = ort::value::Tensor::from_array(([1usize, ids_len], ids.into_boxed_slice()))
        .map_err(|e| format!("ids tensor: {e}"))?;
    let mut embed_guard = sessions.embed.lock().unwrap();
    let out = embed_guard
        .run(ort::inputs! { "input_ids" => ids_tensor })
        .map_err(|e| format!("embed_tokens: {e}"))?;
    let (shape, data) = out[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("embeddings: {e}"))?;
    let n = shape.num_elements();
    let embeddings: Vec<f32> = data[..n].to_vec();
    // 5) 拼接 image_features + text_embeddings（Python `_merge_input_ids_with_image_features`）
    let mut merged: Vec<f32> = Vec::with_capacity(image_features.len() + embeddings.len());
    merged.extend_from_slice(&image_features);
    merged.extend_from_slice(&embeddings);
    let merged_len = image_features.len() / 768 + ids_len;
    let emb_tensor =
        ort::value::Tensor::from_array(([1usize, merged_len, 768], merged.into_boxed_slice()))
            .map_err(|e| format!("emb tensor: {e}"))?;
    let mut lang_guard = sessions.lang.lock().unwrap();
    let out = lang_guard
        .run(ort::inputs! { "inputs_embeds" => emb_tensor })
        .map_err(|e| format!("language_encoder: {e}"))?;
    let (shape, data) = out[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("encoder_outputs: {e}"))?;
    let n = shape.num_elements();
    let encoder_outputs: Vec<f32> = data[..n].to_vec();
    // 6) decoder 自回归（greedy，max_new_tokens=20）
    let enc_tensor = ort::value::Tensor::from_array((
        [1usize, merged_len, 768],
        encoder_outputs.into_boxed_slice(),
    ))
    .map_err(|e| format!("enc tensor: {e}"))?;
    let mut gen_ids: Vec<i64> = vec![2]; // </s> 起始（Florence-2 decoder 起始 token）
    let eos = 2; // </s>
    const MAX_DEC_SEQ: usize = 20;
    for _ in 0..20 {
        // decoder 输入固定 seq=20（padding + attention_mask）
        let mut dec_ids = vec![0i64; MAX_DEC_SEQ];
        let mut dec_mask = vec![0i64; MAX_DEC_SEQ];
        for (i, &id) in gen_ids.iter().enumerate() {
            dec_ids[i] = id;
            dec_mask[i] = 1;
        }
        let ids_tensor =
            ort::value::Tensor::from_array(([1usize, MAX_DEC_SEQ], dec_ids.into_boxed_slice()))
                .map_err(|e| format!("gen ids: {e}"))?;
        let mask_tensor =
            ort::value::Tensor::from_array(([1usize, MAX_DEC_SEQ], dec_mask.into_boxed_slice()))
                .map_err(|e| format!("gen mask: {e}"))?;
        let mut decoder_guard = sessions.decoder.lock().unwrap();
        let out = decoder_guard
            .run(ort::inputs! { "decoder_input_ids" => ids_tensor, "decoder_attention_mask" => mask_tensor, "encoder_outputs" => enc_tensor.clone() })
            .map_err(|e| format!("decoder: {e}"))?;
        let (shape, data) = out[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("logits: {e}"))?;
        let n = shape.num_elements();
        let logits: Vec<f32> = data[..n].to_vec();
        // 取最后一步 argmax（logits 扁平: (1, 20, vocab)）
        let seq = gen_ids.len();
        let vocab = 51289;
        let mut best = (0usize, f32::NEG_INFINITY);
        for v in 0..vocab {
            let val = logits[(seq - 1) * vocab + v];
            if val > best.1 {
                best = (v, val);
            }
        }
        let next = best.0 as i64;
        if next == eos {
            break;
        }
        gen_ids.push(next);
    }
    // 7) decode（跳过特殊 token）
    let gen_u32: Vec<u32> = gen_ids.iter().map(|&i| i as u32).collect();
    let text = tokenizer
        .decode(&gen_u32, true)
        .map_err(|e| format!("decode: {e}"))?;
    Ok(text.trim().to_string())
}

/// Florence-2 会话集合。
struct FlorenceSessions {
    vision: std::sync::Mutex<ort::session::Session>,
    embed: std::sync::Mutex<ort::session::Session>,
    lang: std::sync::Mutex<ort::session::Session>,
    decoder: std::sync::Mutex<ort::session::Session>,
}

fn load_florence() -> Result<(FlorenceSessions, tokenizers::Tokenizer), String> {
    let dir = model_dir().join("florence2");
    let load = |name: &str| {
        ort::session::Session::builder()
            .map_err(|e| format!("ort builder: {e}"))?
            .commit_from_file(dir.join(name))
            .map_err(|e| format!("{name} 加载失败: {e}"))
    };
    let sessions = FlorenceSessions {
        vision: std::sync::Mutex::new(load("vision_encoder.onnx")?),
        embed: std::sync::Mutex::new(load("embed_tokens.onnx")?),
        lang: std::sync::Mutex::new(load("language_encoder.onnx")?),
        decoder: std::sync::Mutex::new(load("decoder.onnx")?),
    };
    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| format!("tokenizer 加载失败: {e}"))?;
    Ok((sessions, tokenizer))
}

// ---------------------------------------------------------------- 主流程

/// 运行 OmniParser 传感器（对应 Python `main()`）。
pub fn run(path: &Path, p: &OmniparserParams) -> Result<Value, String> {
    let img = load_rgb(path)?;
    let (w, h) = (img.width as usize, img.height as usize);
    // 1) YOLOv9 检测图标
    let dets = detect_icons(&img, p.min_conf)?;
    // 2) Florence-2 caption（懒加载：无检测时跳过）
    let mut elements: Vec<Value> = Vec::new();
    if !dets.is_empty() {
        let (sessions, tokenizer) = load_florence()?;
        for (i, det) in dets.iter().take(p.max_items as usize).enumerate() {
            let caption = caption_icon(&img, &det.bbox, &tokenizer, &sessions)?;
            elements.push(json!({
                "id": i,
                "type": "icon",
                "bbox": [
                    det.bbox[0].round() as i64,
                    det.bbox[1].round() as i64,
                    det.bbox[2].round() as i64,
                    det.bbox[3].round() as i64,
                ],
                "text": caption,
                "conf": None::<f64>,
                "color": None::<String>,
                "font": None::<String>,
                "z": None::<i64>,
                "source": ["omniparser"],
                "coordsys": "image_px",
                "interactivity": true,
            }));
        }
    }
    let mut report = crate::schema::envelope(
        Some("omniparser"),
        &["omniparser"],
        Some("image_px"),
        Some(json!({
            "type": "image",
            "path": path.display().to_string(),
            "size_px": [w, h],
            "engine": "OmniParser-v2",
            "box_threshold": p.min_conf,
        })),
    );
    report["elements"] = json!(elements);
    report["metrics"] = json!({"element_count": elements.len()});
    report["truncated"] = json!(elements.len() >= p.max_items as usize);
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iou_basic() {
        let a = [0.0, 0.0, 10.0, 10.0];
        let b = [5.0, 5.0, 15.0, 15.0];
        let i = iou(&a, &b);
        assert!((i - 25.0 / 175.0).abs() < 1e-6);
        assert_eq!(iou(&a, &a), 1.0);
        let c = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(iou(&a, &c), 0.0);
    }

    #[test]
    fn decode_yolov9_empty() {
        // 空输出 → 空检测
        let out = decode_yolov9(&[], 100, 100, 0.05, 1.0, 0.0, 0.0);
        assert!(out.is_empty());
    }
}
