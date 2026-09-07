//! OCR 传感器（对应 Python `vs_ocr.py`：rapidocr 后端，schema v3 输出一致）。
//!
//! rapidocr-core（PP-OCRv6 small ONNX）det→cls→rec 流水线 → 4 点 bbox +
//! 词典纠错（编辑距离 ≤1 替换 + 置信度 ×0.9）→ schema v3 elements。

use std::path::Path;

use serde_json::{json, Value};

use crate::image::load_rgb;
use crate::util::py_round;

/// OCR 参数（对应 Python `--max-items/--min-conf`）。
pub struct OcrParams {
    pub max_items: u32,
    pub min_conf: f64,
}

/// 模型目录（与 Python rapidocr 同模型：PP-OCRv6 small）。
pub fn model_dir_debug() -> String {
    model_dir()
}

fn model_dir() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    format!("{home}/.cache/vsensor/rapidocr")
}

/// 常用中文/英文词典（与 Python `COMMON_WORDS` 一致；固定顺序保证确定性）。
const COMMON_WORDS: &[&str] = &[
    "提交",
    "取消",
    "确定",
    "保存",
    "删除",
    "编辑",
    "搜索",
    "设置",
    "登录",
    "注册",
    "退出",
    "返回",
    "下一步",
    "上一步",
    "完成",
    "确认",
    "关闭",
    "打开",
    "新建",
    "submit",
    "cancel",
    "ok",
    "save",
    "delete",
    "edit",
    "search",
    "settings",
    "login",
    "logout",
    "back",
    "next",
    "done",
    "close",
    "open",
    "new",
    "yes",
    "no",
];

/// Levenshtein 编辑距离（标准 DP，与 Python 一致；按字符计数，中文安全）。
fn levenshtein(a: &str, b: &str) -> usize {
    let (a, b) = if a.chars().count() < b.chars().count() {
        (b, a)
    } else {
        (a, b)
    };
    let bc = b.chars().count();
    let mut prev: Vec<usize> = (0..=bc).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut cur = vec![i + 1];
        for (j, cb) in b.chars().enumerate() {
            cur.push(
                (prev[j + 1] + 1)
                    .min(cur[j] + 1)
                    .min(prev[j] + usize::from(ca != cb)),
            );
        }
        prev = cur;
    }
    prev[bc]
}

/// 词典纠错：编辑距离 ≤ max_dist 的词典词替换。返回 (修正文本, 置信度因子)。
fn dict_correct(text: &str, max_dist: usize) -> (String, f64) {
    let t = text.trim();
    if t.is_empty() || t.chars().count() < 2 {
        return (text.to_string(), 1.0);
    }
    let tl = t.to_lowercase();
    let mut best_word: Option<&str> = None;
    let mut best_dist = max_dist + 1;
    for word in COMMON_WORDS {
        let d = levenshtein(&tl, &word.to_lowercase());
        if d < best_dist {
            best_dist = d;
            best_word = Some(word);
        }
    }
    match best_word {
        Some(w) if best_dist <= max_dist => (w.to_string(), 0.9), // 纠错后置信度 0.9
        _ => (text.to_string(), 1.0),
    }
}

/// 全局单例：rapidocr engine 只加载一次（det/cls/rec 三模型，避免并行竞争）。
static OCR_ENGINE: std::sync::OnceLock<Result<std::sync::Mutex<rapidocr_core::RapidOcr>, String>> =
    std::sync::OnceLock::new();

/// 运行 OCR 传感器（对应 Python `main()` rapidocr 路径）。
pub fn run(path: &Path, p: &OcrParams) -> Result<Value, String> {
    let img = load_rgb(path)?;
    let (w, h) = (img.width, img.height);

    // rapidocr-core 推理（det→cls→rec）
    let dir = model_dir();
    if !Path::new(&dir).exists() {
        return Err(format!(
            "模型目录不存在: {dir}（需放置 PP-OCRv6 small 模型 + dict）"
        ));
    }
    let engine = OCR_ENGINE
        .get_or_init(|| {
            let mut cfg = rapidocr_core::config::RapidOcrConfig::ppocr_v6_small(&dir);
            // unclip_ratio 1.6（Python 默认）：round-join（geo-clipper）与 pyclipper 一致，
            // 无需再像 miter 偏移那样调低。
            if let Some(det) = cfg.det.as_mut() {
                det.unclip_ratio = 1.6;
            }
            rapidocr_core::RapidOcr::new(cfg)
                .map_err(|e| format!("rapidocr init: {e}"))
                .map(std::sync::Mutex::new)
        })
        .as_ref()
        .map_err(|e| e.clone())?;
    let mut engine = engine.lock().unwrap();
    let rgb = image::RgbImage::from_raw(w, h, img.data.clone()).ok_or("image decode")?;
    let out = engine
        .run_image(&rgb)
        .map_err(|e| format!("rapidocr run: {e}"))?;

    // 后处理（对应 vs_ocr.py：min_conf 过滤 + 4 点 bbox + 词典纠错）
    let mut items: Vec<Value> = Vec::new();
    for line in &out.lines {
        let score = f64::from(line.score);
        if score < p.min_conf {
            continue;
        }
        // 4 点 bbox（round 到整数，Python 语义）
        let quad: Vec<Value> = line
            .bbox
            .points
            .iter()
            .map(|pt| {
                json!([
                    py_round(f64::from(pt[0])) as i64,
                    py_round(f64::from(pt[1])) as i64
                ])
            })
            .collect();
        let xs: Vec<i64> = quad.iter().map(|p| p[0].as_i64().unwrap_or(0)).collect();
        let ys: Vec<i64> = quad.iter().map(|p| p[1].as_i64().unwrap_or(0)).collect();
        let cx = py_round(xs.iter().sum::<i64>() as f64 / 4.0) as i64;
        let cy = py_round(ys.iter().sum::<i64>() as f64 / 4.0) as i64;
        let bbox = [
            *xs.iter().min().unwrap_or(&0),
            *ys.iter().min().unwrap_or(&0),
            *xs.iter().max().unwrap_or(&0),
            *ys.iter().max().unwrap_or(&0),
        ];
        items.push(json!({
            "text": line.text.clone(),
            "conf": (score * 1000.0).round() / 1000.0,
            "bbox": bbox,
            "quad": quad,
            "center": [cx, cy],
        }));
        if items.len() >= p.max_items as usize {
            break;
        }
    }

    // 词典纠错 + 置信度加权（对应 postprocess_items）
    for it in items.iter_mut() {
        let text = it["text"].as_str().unwrap_or("").to_string();
        let (corrected, conf_factor) = dict_correct(&text, 1);
        it["text"] = json!(corrected);
        if conf_factor < 1.0 {
            let conf = it["conf"].as_f64().unwrap_or(0.5) * conf_factor;
            it["conf"] = json!((conf * 1000.0).round() / 1000.0);
        }
    }

    // 输出（schema v3 elements，与 vs_ocr.py 逐字段一致）
    let elements: Vec<Value> = items
        .iter()
        .enumerate()
        .map(|(i, it)| {
            let bbox: Vec<f64> = it["bbox"]
                .as_array()
                .map(|b| b.iter().filter_map(|v| v.as_f64()).collect())
                .unwrap_or_default();
            let mut el = serde_json::to_value(crate::schema::Element {
                id: i as i64,
                etype: "text",
                bbox: &bbox,
                text: it["text"].as_str(),
                conf: it["conf"].as_f64(),
                color: None,
                font: None,
                z: None,
                source: &["ocr"],
                coordsys: "image_px",
            })
            .unwrap();
            el["center"] = it["center"].clone();
            el["quad"] = it["quad"].clone();
            el
        })
        .collect();
    let mut report = crate::schema::envelope(
        Some("ocr"),
        &["ocr"],
        Some("image_px"),
        Some(json!({
            "type": "image",
            "path": path.display().to_string(),
            "size_px": [w, h],
            "backend": "rapidocr",
        })),
    );
    report["elements"] = json!(elements);
    report["metrics"] = json!({
        "items": items.len(),
        "truncated": items.len() >= p.max_items as usize,
    });
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levenshtein_known() {
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("abc", "abc"), 0);
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("提交", "提交"), 0);
    }

    #[test]
    fn dict_correct_known() {
        // 精确匹配词典词 → 返回 0.9（Python quirk：精确匹配也走纠错路径）
        assert_eq!(dict_correct("提交", 1), ("提交".to_string(), 0.9));
        // 编辑距离 1 → 替换 + 0.9
        let (t, f) = dict_correct("提父", 1);
        assert_eq!(t, "提交");
        assert_eq!(f, 0.9);
        // 短文本不纠错
        assert_eq!(dict_correct("a", 1), ("a".to_string(), 1.0));
        // 距离 > 1 → 不纠错
        let (t2, f2) = dict_correct("完全无关词", 1);
        assert_eq!(t2, "完全无关词");
        assert_eq!(f2, 1.0);
    }

    #[test]
    fn dict_correct_english() {
        // "submti"→"submit" 距离 2（交换算 2 次替换）→ 不纠错（与 Python 实测一致）
        let (t, f) = dict_correct("submti", 1);
        assert_eq!(t, "submti");
        assert_eq!(f, 1.0);
        // 精确匹配 → 0.9
        let (t2, f2) = dict_correct("save", 1);
        assert_eq!(t2, "save");
        assert_eq!(f2, 0.9);
    }

    #[test]
    fn model_missing_returns_error() {
        // 模型目录不存在时返回 Err（不 panic）
        let r = run(
            Path::new("/nonexistent.png"),
            &OcrParams {
                max_items: 100,
                min_conf: 0.5,
            },
        );
        assert!(r.is_err());
    }
}
