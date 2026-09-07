//! layout 差分测试：Rust 纯 Rust 推理 vs Python paddlex golden。
//!
//! 门禁：元素 IoU ≥ 0.9（bbox 差 1-2px 来自 findContours/minAreaRect 实现差异）。

use std::path::Path;

use serde_json::Value;
use vs_core::layout::{run, LayoutParams};

fn load_golden(name: &str) -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("layout")
        .join(name);
    let s = std::fs::read_to_string(&path).expect("golden 文件缺失");
    serde_json::from_str(&s).expect("golden JSON 解析失败")
}

fn iou(a: &[i64], b: &[i64]) -> f64 {
    let (x1, y1, x2, y2) = (
        a[0].max(b[0]),
        a[1].max(b[1]),
        a[2].min(b[2]),
        a[3].min(b[3]),
    );
    let inter = ((x2 - x1).max(0) * (y2 - y1).max(0)) as f64;
    let a1 = ((a[2] - a[0]) * (a[3] - a[1])) as f64;
    let b1 = ((b[2] - b[0]) * (b[3] - b[1])) as f64;
    let union = a1 + b1 - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// 元素匹配：Python 每元素在 Rust 中找最大 IoU 匹配。
fn assert_elements_paired(golden: &Value, rust: &Value, label: &str, min_iou: f64) {
    let ge = golden["elements"].as_array().unwrap();
    let re = rust["elements"].as_array().unwrap();
    let mut matched = vec![false; re.len()];
    let mut total_iou = 0.0;
    let mut count = 0;
    for g in ge {
        let gb: Vec<i64> = g["bbox"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        let mut best = 0.0f64;
        for r in re.iter() {
            let rb: Vec<i64> = r["bbox"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect();
            let i = iou(&gb, &rb);
            if i > best {
                best = i;
                matched[re.iter().position(|x| std::ptr::eq(x, r)).unwrap_or(0)] = true;
            }
        }
        total_iou += best;
        count += 1;
        assert!(
            best >= min_iou,
            "{label}: golden bbox {gb:?} 无匹配（best IoU {best:.3} < {min_iou}）"
        );
    }
    // 标签一致性（匹配元素）
    for (i, r) in re.iter().enumerate() {
        if matched[i] {
            let rl = r["label"].as_str().unwrap();
            // 找 golden 中 IoU 最大的元素
            let rb: Vec<i64> = r["bbox"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect();
            let mut best_iou = 0.0f64;
            let mut best_label = "";
            for g in ge {
                let gb: Vec<i64> = g["bbox"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_i64().unwrap())
                    .collect();
                let i = iou(&gb, &rb);
                if i > best_iou {
                    best_iou = i;
                    best_label = g["label"].as_str().unwrap();
                }
            }
            assert_eq!(rl, best_label, "{label}: 标签不一致 {rl} vs {best_label}");
        }
    }
    eprintln!(
        "{label}: {count} 元素匹配，平均 IoU {:.3}",
        total_iou / count.max(1) as f64
    );
}

#[test]
fn layout_matches_python() {
    let samples = [
        ("s02_overlap.png", "s02_overlap.json"),
        ("s04_spacing.png", "s04_spacing.json"),
        ("s05_offcanvas.png", "s05_offcanvas.json"),
        ("s07_clean_grid.png", "s07_clean_grid.json"),
        ("s09_clean_card.png", "s09_clean_card.json"),
    ];
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
        .join("bench")
        .join("samples");
    for (img, golden_name) in samples {
        let img_path = root.join(img);
        if !img_path.exists() {
            eprintln!("样本缺失，跳过: {img}");
            continue;
        }
        let golden = load_golden(golden_name);
        let rust = run(
            &img_path,
            &LayoutParams {
                max_items: 100,
                min_conf: 0.3,
            },
        )
        .unwrap_or_else(|e| panic!("{img} layout 失败: {e}"));
        assert_elements_paired(&golden, &rust, img, 0.9);
    }
}
