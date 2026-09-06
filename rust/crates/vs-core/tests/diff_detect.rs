//! S6 门禁：vs-core detect（OWLv2）传感器与 Python 版差分测试。
//!
//! 流程：`rust/tests/gen_detect_golden.py` 生成 golden（Python 基线），
//! 本测试用 Rust 计算同组输入并对比。
//!
//! 门禁（plan）：元素 bbox IoU ≥ 0.9 + 类别一致。
//! 容差策略：
//! - bbox：Python 每元素在 Rust 中找最大 IoU 匹配，≥ 0.9
//! - 类别：text 字段精确一致（OWLv2 确定性输出）
//! - conf：Python round-half-even vs Rust round-half-away → |Δ| ≤ 0.002
//! - 结构字段（schema/task/sensors/coordsys/metrics）精确；跳过 source.path
//! - 跳过：primitive（Python 字符串 vs Rust 对象）、center/quad（Rust 独有）、
//!   findings/candidates/foreground/notation/anomalies（envelope 差异）

use std::path::PathBuf;

use serde_json::Value;
use vs_core::detect::{self, DetectParams};

/// (路径, golden key, 类别)。
const IMAGES: [(&str, &str, &str); 3] = [
    ("text_lines.png", "text_lines", "person,car,text"),
    (
        "s01_low_contrast.png",
        "s01_low_contrast",
        "person,car,text",
    ),
    ("s02_overlap.png", "s02_overlap", "person,car,text"),
];

/// bbox IoU（[x1,y1,x2,y2]）。
fn bbox_iou(a: &[f64], b: &[f64]) -> f64 {
    let (ax1, ay1, ax2, ay2) = (a[0], a[1], a[2], a[3]);
    let (bx1, by1, bx2, by2) = (b[0], b[1], b[2], b[3]);
    let ix1 = ax1.max(bx1);
    let iy1 = ay1.max(by1);
    let ix2 = ax2.min(bx2);
    let iy2 = ay2.min(by2);
    if ix2 <= ix1 || iy2 <= iy1 {
        return 0.0;
    }
    let inter = (ix2 - ix1) * (iy2 - iy1);
    let union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// 递归对比：结构字段精确，跳过已知差异键。
fn assert_json_eq(a: &Value, b: &Value, path: &str) {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            assert_eq!(x, y, "{path}");
        }
        (Value::String(x), Value::String(y)) => assert_eq!(x, y, "{path}"),
        (Value::Bool(x), Value::Bool(y)) => assert_eq!(x, y, "{path}"),
        (Value::Null, Value::Null) => {}
        (Value::Array(x), Value::Array(y)) => {
            assert_eq!(
                x.len(),
                y.len(),
                "{path}: array len {} vs {}",
                x.len(),
                y.len()
            );
            for (i, (xe, ye)) in x.iter().zip(y).enumerate() {
                assert_json_eq(xe, ye, &format!("{path}[{i}]"));
            }
        }
        (Value::Object(x), Value::Object(y)) => {
            // 跳过键：envelope 差异 + 结构差异 + 路径
            let skip_keys = [
                "findings",
                "candidates",
                "foreground",
                "notation",
                "anomalies",
                "primitive",
                "center",
                "quad",
            ];
            let x_keys: Vec<&String> = x
                .keys()
                .filter(|k| !skip_keys.contains(&k.as_str()))
                .collect();
            let y_keys: Vec<&String> = y
                .keys()
                .filter(|k| !skip_keys.contains(&k.as_str()))
                .collect();
            assert_eq!(
                x_keys.len(),
                y_keys.len(),
                "{path}: object keys {} vs {}",
                x_keys.len(),
                y_keys.len()
            );
            for k in x_keys {
                if path.ends_with(".source") && k == "path" {
                    continue;
                }
                // elements 已由 assert_elements_matched 按 IoU 匹配比较
                if k == "elements" && path.ends_with(".detect") {
                    continue;
                }
                let yv = y
                    .get(k)
                    .unwrap_or_else(|| panic!("{path}.{k}: missing in golden"));
                assert_json_eq(&x[k], yv, &format!("{path}.{k}"));
            }
        }
        _ => panic!("{path}: type mismatch {:?} vs {:?}", a, b),
    }
}

/// 元素匹配：Python 每元素在 Rust 中找最大 IoU 匹配（≥ 0.9）+ 类别一致。
fn assert_elements_matched(py_els: &[Value], rust_els: &[Value], path: &str) -> (usize, usize) {
    let mut matched = 0usize;
    for (i, pe) in py_els.iter().enumerate() {
        let p = format!("{path}[{i}]");
        let pb: Vec<f64> = pe["bbox"]
            .as_array()
            .unwrap_or_else(|| panic!("{p}: missing bbox"))
            .iter()
            .filter_map(|v| v.as_f64())
            .collect();
        // 找 Rust 中最大 IoU 匹配
        let mut best_iou = 0.0f64;
        let mut best_idx: Option<usize> = None;
        for (j, re) in rust_els.iter().enumerate() {
            let Some(rb) = re.get("bbox").and_then(|v| v.as_array()) else {
                continue;
            };
            let rb: Vec<f64> = rb.iter().filter_map(|v| v.as_f64()).collect();
            let iou = bbox_iou(&pb, &rb);
            if iou > best_iou {
                best_iou = iou;
                best_idx = Some(j);
            }
        }
        assert!(
            best_iou >= 0.9,
            "{p}: 无 Rust 匹配（max IoU {best_iou} < 0.9，bbox {pb:?}）"
        );
        // 类别一致（OWLv2 确定性输出）
        let pc = pe["text"].as_str().unwrap_or("");
        let rc = rust_els[best_idx.unwrap()]["text"].as_str().unwrap_or("");
        assert_eq!(pc, rc, "{p}: 类别不一致 py={pc:?} rust={rc:?}");
        // conf 容差（round 语义差异）
        let pconf = pe["conf"].as_f64().unwrap_or(0.0);
        let rconf = rust_els[best_idx.unwrap()]["conf"].as_f64().unwrap_or(0.0);
        assert!(
            (pconf - rconf).abs() <= 0.002,
            "{p}: conf 差异过大 py={pconf} rust={rconf}"
        );
        matched += 1;
    }
    (matched, py_els.len())
}

fn image_path(name: &str) -> PathBuf {
    if name == "text_lines.png" {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("..")
            .join("tests")
            .join("bench_fixtures")
            .join(name)
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("..")
            .join("bench")
            .join("samples")
            .join(name)
    }
}

fn load_golden() -> Value {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("gen_detect_golden.py");
    let out = std::process::Command::new("python3")
        .arg(&script)
        .output()
        .expect("failed to run gen_detect_golden.py (python3 required)");
    assert!(
        out.status.success(),
        "gen_detect_golden.py failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).expect("golden output is not valid JSON")
}

#[test]
fn detect_matches_python() {
    let g = load_golden();
    for (img, key, classes) in IMAGES {
        let path = image_path(img);
        let golden = &g[key]["detect"];
        let params = DetectParams {
            classes: classes.split(',').map(|s| s.to_string()).collect(),
            min_conf: 0.25,
        };
        let r = detect::run(&path, &params).unwrap();
        // 元素匹配：Python 每元素在 Rust 中 IoU ≥ 0.9 + 类别一致
        let ge = golden["elements"].as_array().unwrap();
        let re = r["elements"].as_array().unwrap();
        let (matched, total) = assert_elements_matched(ge, re, &format!("{key}.detect.elements"));
        assert_eq!(matched, total, "{key}: 全部 Python 元素应匹配");
        // 结构字段（schema/task/sensors/coordsys/source 除 path/metrics/truncated）
        assert_json_eq(golden, &r, &format!("{key}.detect"));
    }
}
