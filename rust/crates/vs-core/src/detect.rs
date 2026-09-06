//! detect 传感器 — OWLv2 开放词表检测（S6 Plan B：纯 Rust ONNX）。
//!
//! 与 Python `vs_detect.py`（transformers Owlv2ForObjectDetection）对齐：
//!   1. CLIP BPE tokenizer（vocab.json + merges.txt，特殊 token 49406/49407）
//!   2. 图像预处理：resize 960×960 BILINEAR + CLIP 归一化
//!   3. 三个 ONNX：text_encoder / vision_encoder / detection_head
//!   4. 后处理：sigmoid(max logits) + argmax + center_to_corners + 按最大边缩放
//!      （OWLv2 无 NMS——与 transformers post_process_object_detection 一致）

use std::path::Path;

use serde_json::{json, Value};

use crate::image::{load_rgb, RgbImage};

/// 检测参数。
pub struct DetectParams {
    /// 开放词表类别（如 ["button", "text"]）。
    pub classes: Vec<String>,
    /// 置信度阈值（默认 0.25，与 vs_detect.py 一致）。
    pub min_conf: f64,
}

impl Default for DetectParams {
    fn default() -> Self {
        Self {
            classes: vec!["button".into(), "text".into()],
            min_conf: 0.25,
        }
    }
}

/// 模型目录（~/.cache/vsensor/owlv2/）。
fn model_dir() -> String {
    format!(
        "{}/.cache/vsensor/owlv2",
        std::env::var("HOME").unwrap_or_default()
    )
}

/// HF 缓存中的 tokenizer 文件（snapshot 目录）。
fn tokenizer_files() -> (String, String) {
    let base = format!(
        "{}/.cache/huggingface/hub/models--google--owlv2-base-patch16-ensemble/snapshots",
        std::env::var("HOME").unwrap_or_default()
    );
    // 扫描 snapshot 目录（模型下载后固定）
    if let Ok(entries) = std::fs::read_dir(&base) {
        for e in entries.flatten() {
            let dir = e.path();
            let vocab = dir.join("vocab.json");
            let merges = dir.join("merges.txt");
            if vocab.exists() && merges.exists() {
                return (
                    vocab.to_string_lossy().into_owned(),
                    merges.to_string_lossy().into_owned(),
                );
            }
        }
    }
    (String::new(), String::new())
}

/// CLIP BPE tokenizer（与 transformers CLIPTokenizer 对齐）。
///
/// 流程：regex 分割 → bytes_to_unicode 转换 → BPE merges → 特殊 token。
struct ClipTokenizer {
    tok: tokenizers::Tokenizer,
    byte_encoder: std::collections::HashMap<u8, char>,
    re: regex::Regex,
}

impl ClipTokenizer {
    fn new() -> Result<Self, String> {
        let (vocab, merges) = tokenizer_files();
        if vocab.is_empty() {
            return Err("CLIP tokenizer 文件缺失（需先下载 OWLv2 模型）".into());
        }
        let bpe = tokenizers::models::bpe::BPE::from_file(&vocab, &merges)
            .end_of_word_suffix("</w>".to_string())
            .build()
            .map_err(|e| format!("BPE build: {e}"))?;
        let mut tok = tokenizers::Tokenizer::new(bpe);
        tok.add_special_tokens([
            tokenizers::AddedToken::from("<|startoftext|>", true),
            tokenizers::AddedToken::from("<|endoftext|>", true),
        ])
        .map_err(|e| format!("add_special_tokens: {e}"))?;
        // bytes_to_unicode 映射（GPT-2/CLIP 相同）
        let mut bs: Vec<u8> = (b'!'..=b'~').collect();
        bs.extend((0xA1u8..=0xAC).collect::<Vec<_>>());
        bs.extend((0xAEu8..=0xFF).collect::<Vec<_>>());
        let mut cs: Vec<char> = bs.iter().map(|&b| b as char).collect();
        let mut n: u32 = 0;
        for b in 0u16..256 {
            if !bs.contains(&(b as u8)) {
                bs.push(b as u8);
                cs.push(char::from_u32(0x100 + n).unwrap());
                n += 1;
            }
        }
        let byte_encoder: std::collections::HashMap<u8, char> =
            bs.iter().zip(cs.iter()).map(|(&b, &c)| (b, c)).collect();
        // CLIP pre-tokenizer regex（与 transformers CLIPTokenizer 一致）
        let re = regex::Regex::new(
            r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
        )
        .map_err(|e| format!("regex: {e}"))?;
        Ok(Self {
            tok,
            byte_encoder,
            re,
        })
    }

    /// 编码单个类别 → 16 长度 input_ids（含 49406/49407 + padding 0）。
    fn encode(&self, text: &str) -> Result<Vec<i64>, String> {
        let mut ids = vec![49406i64];
        for m in self.re.find_iter(text) {
            let t = m.as_str();
            let converted: String = t.as_bytes().iter().map(|b| self.byte_encoder[b]).collect();
            let enc = self
                .tok
                .encode(converted, false)
                .map_err(|e| format!("encode {text:?}: {e}"))?;
            ids.extend(enc.get_ids().iter().map(|&i| i as i64));
        }
        ids.push(49407);
        while ids.len() < 16 {
            ids.push(0);
        }
        Ok(ids)
    }
}

/// scipy.ndimage.zoom（order=1, grid_mode=True, mode=mirror）等价插值。
///
/// 图像预处理：pad 正方形（灰 0.5）→ scipy zoom 960×960 → CLIP 归一化。
///
/// 与 Owlv2ImageProcessor 一致：rescale → pad → resize → normalize。
fn preprocess(img: &RgbImage) -> Result<Vec<f32>, String> {
    const SIZE: u32 = 960;
    // 1) pad 到正方形（max(w,h)，灰色 0.5 = 127.5）
    let side = img.width.max(img.height);
    let mut square = vec![127.5f32; (side * side * 3) as usize];
    for y in 0..img.height {
        for x in 0..img.width {
            let p = img.pixel(x, y);
            let dst = ((y * side + x) * 3) as usize;
            square[dst] = p[0] as f32;
            square[dst + 1] = p[1] as f32;
            square[dst + 2] = p[2] as f32;
        }
    }
    // 2) scipy zoom 到 960×960（f32 输入）
    let zoomed = scipy_zoom_f32(&square, side, side, SIZE, SIZE);
    // 3) normalize（rescale 1/255 隐含在归一化中）
    let mean = [0.481_454_66f32, 0.457_827_5, 0.408_210_73];
    let std = [0.268_629_5f32, 0.261_302_6, 0.275_777_1];
    let mut out = vec![0f32; (SIZE * SIZE * 3) as usize];
    for i in 0..(SIZE * SIZE * 3) as usize {
        let c = i / (SIZE * SIZE) as usize;
        out[i] = (zoomed[i] / 255.0 - mean[c]) / std[c];
    }
    Ok(out)
}

/// scipy.ndimage.zoom（order=1, grid_mode=True, mode=mirror）等价插值（f32 输入）。
fn scipy_zoom_f32(src: &[f32], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Vec<f32> {
    let zy = src_h as f64 / dst_h as f64;
    let zx = src_w as f64 / dst_w as f64;
    let mut out = vec![0f32; (dst_w * dst_h * 3) as usize];
    let mirror = |v: f64, n: i64| -> i64 {
        let mut v = v;
        if v < 0.0 {
            let sz2 = 2 * n - 2;
            v = sz2 as f64 * ((-v / sz2 as f64) as i64) as f64 + v;
            v = if v <= (1 - n) as f64 {
                v + sz2 as f64
            } else {
                -v
            };
        } else if v >= n as f64 {
            let sz2 = 2 * n - 2;
            v -= sz2 as f64 * ((v / sz2 as f64) as i64) as f64;
            if v >= n as f64 {
                v = sz2 as f64 - v;
            }
        }
        v as i64
    };
    for oy in 0..dst_h {
        let iy = (oy as f64 + 0.5) * zy - 0.5;
        let y0 = iy.floor() as i64;
        let y1 = y0 + 1;
        let wy = iy - y0 as f64;
        let my0 = mirror(y0 as f64, src_h as i64);
        let my1 = mirror(y1 as f64, src_h as i64);
        for ox in 0..dst_w {
            let ix = (ox as f64 + 0.5) * zx - 0.5;
            let x0 = ix.floor() as i64;
            let x1 = x0 + 1;
            let wx = ix - x0 as f64;
            let mx0 = mirror(x0 as f64, src_w as i64);
            let mx1 = mirror(x1 as f64, src_w as i64);
            let (a, b, c, d) = (
                (my0 * src_w as i64 + mx0) as usize * 3,
                (my0 * src_w as i64 + mx1) as usize * 3,
                (my1 * src_w as i64 + mx0) as usize * 3,
                (my1 * src_w as i64 + mx1) as usize * 3,
            );
            let (w00, w01, w10, w11) = (
                (1.0 - wy) * (1.0 - wx),
                (1.0 - wy) * wx,
                wy * (1.0 - wx),
                wy * wx,
            );
            for ch in 0..3usize {
                let v = w00 * src[a + ch] as f64
                    + w01 * src[b + ch] as f64
                    + w10 * src[c + ch] as f64
                    + w11 * src[d + ch] as f64;
                out[(ch * (dst_w as usize) * (dst_h as usize))
                    + ((oy as usize) * (dst_w as usize) + ox as usize)] = v as f32;
            }
        }
    }
    out
}

/// 运行检测传感器（对应 Python `main()`）。
pub fn run(path: &Path, p: &DetectParams) -> Result<Value, String> {
    let img = load_rgb(path)?;
    let (w, h) = (img.width as f64, img.height as f64);
    let classes = if p.classes.is_empty() {
        vec!["button".to_string(), "text".to_string()]
    } else {
        p.classes.clone()
    };

    // 1) tokenize
    let tok = ClipTokenizer::new()?;
    let mut input_ids = Vec::with_capacity(classes.len() * 16);
    for c in &classes {
        input_ids.extend(tok.encode(c)?);
    }
    let nq = classes.len();

    // 2) 图像预处理
    let pixel_values = preprocess(&img)?;

    // 3) 三个 ONNX 推理
    let dir = model_dir();
    let mut text_sess = ort::session::Session::builder()
        .map_err(|e| format!("ort builder: {e}"))?
        .commit_from_file(format!("{dir}/text_encoder.onnx"))
        .map_err(|e| format!("ort load text_encoder: {e}"))?;
    let mut vision_sess = ort::session::Session::builder()
        .map_err(|e| format!("ort builder: {e}"))?
        .commit_from_file(format!("{dir}/vision_encoder.onnx"))
        .map_err(|e| format!("ort load vision_encoder: {e}"))?;
    let mut head_sess = ort::session::Session::builder()
        .map_err(|e| format!("ort builder: {e}"))?
        .commit_from_file(format!("{dir}/detection_head.onnx"))
        .map_err(|e| format!("ort load detection_head: {e}"))?;

    // text_encoder: input_ids (nq, 16) → text_embeds (nq, 512)
    let ids_tensor = ort::value::Tensor::from_array(([nq, 16], input_ids.into_boxed_slice()))
        .map_err(|e| format!("ort tensor ids: {e}"))?;
    let text_out = text_sess
        .run(ort::inputs! { "input_ids" => ids_tensor })
        .map_err(|e| format!("ort run text: {e}"))?;
    let (tshape, tdata) = text_out[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("ort extract text: {e}"))?;
    let tdata = &tdata[..tshape.num_elements()];
    // (nq, 512) → (1, nq, 512)
    let mut query_embeds = vec![0f32; nq * 512];
    for i in 0..nq {
        for j in 0..512 {
            query_embeds[i * 512 + j] = tdata[i * 512 + j];
        }
    }

    // vision_encoder: pixel_values (1, 3, 960, 960) → image_feats (1, 3600, 768)
    let px_tensor =
        ort::value::Tensor::from_array(([1usize, 3, 960, 960], pixel_values.into_boxed_slice()))
            .map_err(|e| format!("ort tensor px: {e}"))?;
    let vision_out = vision_sess
        .run(ort::inputs! { "pixel_values" => px_tensor })
        .map_err(|e| format!("ort run vision: {e}"))?;
    let (vshape, vdata) = vision_out[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("ort extract vision: {e}"))?;
    let vdata = &vdata[..vshape.num_elements()];
    let image_feats: Vec<f32> = vdata.to_vec();

    // detection_head: image_feats + query_embeds + query_mask → logits + boxes
    let qe_tensor =
        ort::value::Tensor::from_array(([1usize, nq, 512], query_embeds.into_boxed_slice()))
            .map_err(|e| format!("ort tensor qe: {e}"))?;
    let qm_tensor =
        ort::value::Tensor::from_array(([1usize, nq], vec![1i64; nq].into_boxed_slice()))
            .map_err(|e| format!("ort tensor qm: {e}"))?;
    let if_tensor =
        ort::value::Tensor::from_array(([1usize, 3600, 768], image_feats.into_boxed_slice()))
            .map_err(|e| format!("ort tensor if: {e}"))?;
    let head_out = head_sess
        .run(ort::inputs! { "image_feats" => if_tensor, "query_embeds" => qe_tensor, "query_mask" => qm_tensor })
        .map_err(|e| format!("ort run head: {e}"))?;
    let (lshape, ldata) = head_out[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("ort extract logits: {e}"))?;
    let ldata = &ldata[..lshape.num_elements()];
    let (bshape, bdata) = head_out[1]
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("ort extract boxes: {e}"))?;
    let bdata = &bdata[..bshape.num_elements()];

    // 4) 后处理（对齐 post_process_object_detection）
    // logits (1, 3600, nq)：max over last dim → scores = sigmoid(max)；labels = argmax
    // （OWLv2 无背景类——logits 就是 num_queries 维）
    const NP: usize = 3600;
    let mut scores = vec![0f32; NP];
    let mut labels = vec![0i64; NP];
    for i in 0..NP {
        let base = i * nq;
        let mut mx = f32::NEG_INFINITY;
        let mut arg = 0usize;
        for q in 0..nq {
            let v = ldata[base + q];
            if v > mx {
                mx = v;
                arg = q;
            }
        }
        scores[i] = 1.0 / (1.0 + (-mx).exp());
        labels[i] = arg as i64;
    }
    // boxes (1, 3600, 4)：cxcywh → xyxy，按最大边缩放
    let size = w.max(h);
    let mut boxes = vec![0f64; NP * 4];
    for i in 0..NP {
        let cx = bdata[i * 4] as f64;
        let cy = bdata[i * 4 + 1] as f64;
        let bw = bdata[i * 4 + 2] as f64;
        let bh = bdata[i * 4 + 3] as f64;
        boxes[i * 4] = (cx - bw / 2.0) * size;
        boxes[i * 4 + 1] = (cy - bh / 2.0) * size;
        boxes[i * 4 + 2] = (cx + bw / 2.0) * size;
        boxes[i * 4 + 3] = (cy + bh / 2.0) * size;
    }

    // 阈值过滤 + 分数降序排序 + max_items 截断
    let mut kept: Vec<(usize, f32)> = (0..NP)
        .filter(|&i| scores[i] > p.min_conf as f32)
        .map(|i| (i, scores[i]))
        .collect();
    kept.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let max_items = 50usize;
    let mut elements = Vec::new();
    for (rank, (i, _)) in kept.iter().take(max_items).enumerate() {
        let i = *i;
        let x1 = boxes[i * 4].round() as i64;
        let y1 = boxes[i * 4 + 1].round() as i64;
        let x2 = boxes[i * 4 + 2].round() as i64;
        let y2 = boxes[i * 4 + 3].round() as i64;
        let label = labels[i] as usize;
        let cls = if label < classes.len() {
            classes[label].clone()
        } else {
            "unknown".into()
        };
        let conf = (scores[i] as f64 * 1000.0).round() / 1000.0;
        elements.push(json!({
            "id": rank,
            "type": "object",
            "bbox": [x1, y1, x2, y2],
            "text": cls,
            "conf": conf,
            "color": null,
            "font": null,
            "z": null,
            "source": ["detect"],
            "coordsys": "image_px",
            "center": [(x1 + x2) / 2, (y1 + y2) / 2],
            "quad": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            "primitive": {"type": "bbox", "x1": x1, "y1": y1, "x2": x2, "y2": y2}
        }));
    }

    Ok(json!({
        "schema": "vision-report/v3",
        "task": "detect",
        "sensors": ["owlv2"],
        "coordsys": "image_px",
        "source": {"type": "image", "path": path.to_string_lossy(), "classes": classes},
        "elements": elements,
        "anomalies": [],
        "metrics": {
            "detections": elements.len(),
            "classes": classes.len(),
            "threshold": p.min_conf
        },
        "truncated": elements.len() >= max_items
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_matches_python() {
        // "a photo of a button" → [49406, 320, 1125, 539, 320, 8007, 49407, 0×9]
        let tok = ClipTokenizer::new().unwrap();
        let ids = tok.encode("a photo of a button").unwrap();
        assert_eq!(ids.len(), 16);
        assert_eq!(&ids[..7], &[49406, 320, 1125, 539, 320, 8007, 49407]);
        assert!(ids[7..].iter().all(|&i| i == 0));
    }

    #[test]
    fn tokenizer_handles_chinese() {
        let tok = ClipTokenizer::new().unwrap();
        let ids = tok.encode("按钮").unwrap();
        assert_eq!(ids.len(), 16);
        assert_eq!(ids[0], 49406);
        // "按钮" → 6 个 byte token，49407 在位置 7
        assert_eq!(ids[7], 49407);
        assert!(ids[8..].iter().all(|&i| i == 0));
    }

    #[test]
    fn params_default() {
        let p = DetectParams::default();
        assert_eq!(p.min_conf, 0.25);
        assert_eq!(p.classes, vec!["button", "text"]);
    }
}
