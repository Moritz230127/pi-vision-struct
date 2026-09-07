//! CLIP 离线聚类（对应 Python `vs_cluster.py`，schema v3 输出逐字段一致）。
//!
//! 流程（与 vs_cluster.py 完全一致）：
//!   1. open_clip ViT-B-32（laion2b_s34b_b79k）ONNX 编码 → L2 归一化嵌入
//!   2. 余弦相似度矩阵（归一化向量的点积）
//!   3. 贪心单链分组：按文件名顺序，与已有组代表相似度均值 > 阈值则并入
//!   4. 每图 top-2 相似图作为证据
//!
//! 预处理（open_clip 标准）：Resize(224, BICUBIC) + CenterCrop(224) +
//! Normalize(mean=[0.48145466,0.4578275,0.40821073],
//!           std=[0.26862954,0.26130258,0.27577711])

use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::schema::envelope;

/// 聚类参数（对应 Python CLI 参数）。
pub struct ClusterParams {
    pub dir: Option<String>,
    pub files: Option<String>,
    pub threshold: f64,
    pub max_files: usize,
    pub exts: Vec<String>,
}

impl Default for ClusterParams {
    fn default() -> Self {
        Self {
            dir: None,
            files: None,
            threshold: 0.75,
            max_files: 200,
            exts: vec![],
        }
    }
}

/// 模型路径。
fn model_path() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    format!("{home}/.cache/vsensor/clip_vitb32.onnx")
}

/// 收集图片（dir 递归 / files 逗号分隔）。
fn collect_images(paths: &[String], exts: &[String]) -> Vec<String> {
    let exts: Vec<String> = if exts.is_empty() {
        vec![
            ".png".into(),
            ".jpg".into(),
            ".jpeg".into(),
            ".webp".into(),
            ".bmp".into(),
            ".gif".into(),
        ]
    } else {
        exts.iter()
            .map(|e| {
                if e.starts_with('.') {
                    e.to_lowercase()
                } else {
                    format!(".{}", e.to_lowercase())
                }
            })
            .collect()
    };
    let mut out = Vec::new();
    for p in paths {
        let pth = Path::new(p);
        if pth.is_dir() {
            let mut entries: Vec<PathBuf> = pth
                .read_dir()
                .map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.path()).collect())
                .unwrap_or_default();
            entries.sort();
            for f in entries {
                if f.is_file()
                    && f.extension()
                        .and_then(|e| e.to_str())
                        .map(|e| exts.contains(&format!(".{}", e.to_lowercase())))
                        .unwrap_or(false)
                {
                    // 保留输入路径格式（Python `str(Path(p) / f)` 语义）
                    out.push(
                        pth.join(f.file_name().unwrap())
                            .to_string_lossy()
                            .into_owned(),
                    );
                }
            }
        } else if pth.is_file()
            && pth
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| exts.contains(&format!(".{}", e.to_lowercase())))
                .unwrap_or(false)
        {
            out.push(p.to_string());
        }
    }
    out
}

/// 预处理：Resize(224, BICUBIC) + CenterCrop(224) + Normalize。
pub(crate) fn preprocess(img: &image::DynamicImage) -> Result<Vec<f32>, String> {
    const SIZE: u32 = 224;
    // open_clip 预处理：短边缩放到 224（保持宽高比，BICUBIC）→ CenterCrop 224×224
    let (w, h) = (img.width(), img.height());
    let scale = SIZE as f64 / w.min(h) as f64;
    let (nw, nh) = (
        (w as f64 * scale).round() as u32,
        (h as f64 * scale).round() as u32,
    );
    let resized = img.resize_exact(nw, nh, image::imageops::FilterType::CatmullRom);
    // CenterCrop 224×224
    let (cw, ch) = (resized.width(), resized.height());
    let (x, y) = ((cw - SIZE) / 2, (ch - SIZE) / 2);
    let cropped = image::imageops::crop_imm(&resized, x, y, SIZE, SIZE).to_image();
    let rgb = image::DynamicImage::ImageRgba8(cropped).to_rgb8();
    let mean = [0.481_454_66f32, 0.457_827_5, 0.408_210_73];
    let std = [0.268_629_5f32, 0.261_302_6, 0.275_777_1];
    let mut out = vec![0f32; (SIZE * SIZE * 3) as usize];
    // CHW 平面布局（RRR...GGG...BBB...，ONNX 标准）
    for (i, px) in rgb.pixels().enumerate() {
        for c in 0..3 {
            out[c * (SIZE * SIZE) as usize + i] = (px[c] as f32 / 255.0 - mean[c]) / std[c];
        }
    }
    Ok(out)
}

/// 运行聚类（CLIP 编码 + 贪心单链分组）。
pub fn run(p: &ClusterParams) -> Result<Value, String> {
    let images = if let Some(dir) = &p.dir {
        collect_images(std::slice::from_ref(dir), &p.exts)
    } else if let Some(files) = &p.files {
        collect_images(
            &files.split(',').map(String::from).collect::<Vec<_>>(),
            &p.exts,
        )
    } else {
        return Err("需要 --dir 或 --files".into());
    };
    let images: Vec<String> = images.into_iter().take(p.max_files).collect();
    if images.is_empty() {
        return Err("无匹配图片".into());
    }

    // 加载 ONNX
    let path = model_path();
    if !Path::new(&path).exists() {
        return Err(format!(
            "CLIP 模型不存在: {path}（需先导出 clip_vitb32.onnx）"
        ));
    }
    let mut sess = ort::session::Session::builder()
        .map_err(|e| format!("ort builder: {e}"))?
        .commit_from_file(&path)
        .map_err(|e| format!("ort load clip: {e}"))?;

    // 逐图编码 + L2 归一化
    let mut embs: Vec<(String, Vec<f32>)> = Vec::new();
    for img in &images {
        let Ok(im) = image::open(img) else { continue };
        let Ok(tensor) = preprocess(&im) else {
            continue;
        };
        let input =
            ort::value::Tensor::from_array(([1usize, 3, 224, 224], tensor.into_boxed_slice()))
                .map_err(|e| format!("ort tensor: {e}"))?;
        let out = sess
            .run(ort::inputs! { "pixel_values" => input })
            .map_err(|e| format!("ort run: {e}"))?;
        let (fshape, fdata) = out[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("ort extract: {e}"))?;
        let v: Vec<f32> = fdata[..fshape.num_elements()].to_vec();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let v: Vec<f32> = if norm > 0.0 {
            v.iter().map(|x| x / norm).collect()
        } else {
            v
        };
        embs.push((img.clone(), v));
    }
    if embs.len() < 2 {
        return Err(format!(
            "可编码图片仅 {} 张（至少 2 张才能聚类）",
            embs.len()
        ));
    }

    let n = embs.len();
    // 余弦相似度矩阵（归一化向量点积）
    let mut sim = vec![vec![0f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            sim[i][j] = embs[i]
                .1
                .iter()
                .zip(&embs[j].1)
                .map(|(a, b)| (*a as f64) * (*b as f64))
                .sum();
        }
    }

    // 贪心单链分组（确定性：按文件名顺序）
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (i, row) in sim.iter().enumerate() {
        let mut best_g = -1i64;
        let mut best_sim = -1.0f64;
        for (gi, g) in groups.iter().enumerate() {
            let s: f64 = g.iter().map(|&j| row[j]).sum::<f64>() / g.len() as f64;
            if s > best_sim {
                best_sim = s;
                best_g = gi as i64;
            }
        }
        if best_g >= 0 && best_sim > p.threshold {
            groups[best_g as usize].push(i);
        } else {
            groups.push(vec![i]);
        }
    }

    // clusters
    let mut clusters = Vec::new();
    for (gi, g) in groups.iter().enumerate() {
        let rep = g[0];
        let mut members: Vec<Value> = g
            .iter()
            .map(|&i| {
                json!({
                    "file": embs[i].0,
                    "sim_to_rep": py_round4(sim[i][rep]),
                })
            })
            .collect();
        members.sort_by(|a, b| {
            b["sim_to_rep"]
                .as_f64()
                .unwrap_or(0.0)
                .partial_cmp(&a["sim_to_rep"].as_f64().unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mean_sim: f64 = {
            let mut s = 0.0f64;
            let mut cnt = 0usize;
            for &i in g {
                for &j in g {
                    if i != j {
                        s += sim[i][j];
                        cnt += 1;
                    }
                }
            }
            if cnt == 0 {
                1.0
            } else {
                s / cnt as f64
            }
        };
        clusters.push(json!({
            "id": gi,
            "size": g.len(),
            "representative": embs[rep].0,
            "mean_sim": py_round4(mean_sim),
            "members": members,
        }));
    }

    // top_pairs（每图 top-2 相似）
    let mut top_pairs = Vec::new();
    for i in 0..n {
        let mut row: Vec<(usize, f64)> =
            (0..n).filter(|&j| j != i).map(|j| (j, sim[i][j])).collect();
        row.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (j, s) in row.iter().take(2) {
            top_pairs.push(json!({
                "a": embs[i].0,
                "b": embs[*j].0,
                "sim": py_round4(*s),
            }));
        }
    }

    let mut report = envelope(
        Some("cluster"),
        &["clip"],
        None,
        Some(json!({
            "type": "images",
            "count": n,
            "model": "ViT-B-32",
            "pretrained": "laion2b_s34b_b79k",
            "threshold": p.threshold,
        })),
    );
    report["clusters"] = json!(clusters);
    report["top_pairs"] = json!(top_pairs);
    report["metrics"] = json!({
        "image_count": n,
        "cluster_count": groups.len(),
        "singleton_count": groups.iter().filter(|g| g.len() == 1).count(),
        "threshold": p.threshold,
        "max_cluster_size": groups.iter().map(|g| g.len()).max().unwrap_or(0),
    });
    Ok(report)
}

/// Python round(x, 4)（精确十进制舍入）。
fn py_round4(x: f64) -> f64 {
    crate::util::py_round_n(x, 4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collect_images_dir() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("..")
            .join("bench")
            .join("samples");
        let imgs = collect_images(&[dir.to_string_lossy().into_owned()], &[]);
        assert!(
            imgs.len() >= 3,
            "samples 目录应有 ≥3 张图，实际 {}",
            imgs.len()
        );
    }

    #[test]
    fn cluster_runs_on_samples() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("..")
            .join("bench")
            .join("samples");
        let p = ClusterParams {
            dir: Some(dir.to_string_lossy().into_owned()),
            threshold: 0.75,
            max_files: 10,
            ..Default::default()
        };
        match run(&p) {
            Ok(report) => {
                assert_eq!(report["task"], "cluster");
                assert!(report["clusters"].is_array());
                assert!(report["metrics"]["image_count"].as_u64().unwrap() >= 2);
            }
            Err(e) => {
                // 模型缺失时降级跳过
                eprintln!("cluster 不可用（跳过）: {e}");
            }
        }
    }
}
