//! S5 门禁：vs-core OCR 传感器与 Python 版差分测试。
//!
//! 流程：`rust/tests/gen_s5_golden.py` 生成 golden（Python 基线），
//! 本测试用 Rust 计算同组输入并对比。
//!
//! 门禁（plan2 P2 收紧后）：CER=0（文本精确匹配）+ bbox IoU = 1.0 + conf ±0.01。
//!
//! 修复记录（2026-09）：
//! 1. det 分数：BGR 输入 + fillPoly 等价光栅化（s02 漏检修复）
//! 2. det 框几何：boundary 像素中心（cv2.findContours 等价）+ geo-clipper
//!    round-join（pyclipper JT_ROUND 等价）→ 全部 IoU 1.0

use std::path::PathBuf;

use serde_json::Value;
use vs_core::ocr;

const IMAGES: [&str; 4] = [
    "s01_low_contrast.png",
    "s02_overlap.png",
    "s03_alignment.png",
    "s04_spacing.png",
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

/// 结构字段比较：schema/task/sensors/coordsys 精确；公共键递归比较；
/// 跳过单侧独有键（Rust envelope 的 candidates/findings/foreground、Python 的 anomalies）与 source.path。
fn assert_structure(a: &Value, b: &Value, path: &str) {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            let (xf, yf) = (x.as_f64().unwrap(), y.as_f64().unwrap());
            assert!((xf - yf).abs() <= 1e-6, "{path}: {xf} != {yf}");
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
                assert_structure(xe, ye, &format!("{path}[{i}]"));
            }
        }
        (Value::Object(x), Value::Object(y)) => {
            for core in ["schema", "task", "sensors", "coordsys"] {
                if let (Some(xv), Some(yv)) = (x.get(core), y.get(core)) {
                    assert_structure(xv, yv, &format!("{path}.{core}"));
                }
            }
            for (k, xv) in x {
                if path.ends_with(".source") && k == "path" {
                    continue;
                }
                if matches!(
                    k.as_str(),
                    "candidates" | "findings" | "foreground" | "anomalies" | "elements"
                ) {
                    continue; // 单侧独有（envelope 扩展）或单独比较（elements）
                }
                if let Some(yv) = y.get(k) {
                    assert_structure(xv, yv, &format!("{path}.{k}"));
                }
            }
        }
        _ => panic!("{path}: type mismatch {:?} vs {:?}", a, b),
    }
}

/// 元素配对比较：对每个 Python 元素找 Rust 中 IoU 最大的元素。
/// 断言：文本 CER=0（精确匹配）+ bbox IoU ≥ 阈值 + conf 容差 0.05。
fn assert_elements_paired(py_els: &[Value], rs_els: &[Value], path: &str, _iou_th: f64) {
    assert_eq!(
        py_els.len(),
        rs_els.len(),
        "{path}: element count {} vs {}",
        py_els.len(),
        rs_els.len()
    );
    let mut used_rs = vec![false; rs_els.len()];
    for (i, pe) in py_els.iter().enumerate() {
        let p = format!("{path}[{i}]");
        let pb: Vec<f64> = pe
            .get("bbox")
            .and_then(|v| v.as_array())
            .map(|b| b.iter().filter_map(|v| v.as_f64()).collect())
            .expect("golden element missing bbox");
        // 找 Rust 中 IoU 最大的未用元素
        let mut best = None;
        for (j, re) in rs_els.iter().enumerate() {
            if used_rs[j] {
                continue;
            }
            let Some(rb) = re.get("bbox").and_then(|v| v.as_array()) else {
                continue;
            };
            let rb: Vec<f64> = rb.iter().filter_map(|v| v.as_f64()).collect();
            let iou = bbox_iou(&pb, &rb);
            if best.is_none_or(|(_, bi)| iou > bi) {
                best = Some((j, iou));
            }
        }
        let (j, iou) = best.expect("no Rust element to pair");
        used_rs[j] = true;
        let re = &rs_els[j];
        // 门禁 1：bbox IoU = 1.0（boundary 像素中心 + geo-clipper round-join 修复后）
        assert!(
            iou >= 1.0,
            "{p}: bbox IoU {iou} < 1.0 (py {pb:?} vs rs {:?})",
            re.get("bbox")
        );
        // 门禁 2：CER=0（文本精确匹配）
        let pt = pe.get("text").and_then(|v| v.as_str()).unwrap_or("");
        let rt = re.get("text").and_then(|v| v.as_str()).unwrap_or("");
        assert_eq!(pt, rt, "{p}: text mismatch (CER>0)");
        // conf 容差 0.01（bbox 一致 → crop 一致 → rec 分数一致）
        let pc = pe.get("conf").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let rc = re.get("conf").and_then(|v| v.as_f64()).unwrap_or(0.0);
        assert!(
            (pc - rc).abs() <= 0.01,
            "{p}: conf {pc} != {rc} (diff {})",
            (pc - rc).abs()
        );
    }
}

fn sample_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
        .join("bench")
        .join("samples")
        .join(name)
}

fn load_golden() -> Value {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("gen_s5_golden.py");
    let out = std::process::Command::new("python3")
        .arg(&script)
        .output()
        .expect("failed to run gen_s5_golden.py (python3 required)");
    assert!(
        out.status.success(),
        "gen_s5_golden.py failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).expect("golden output is not valid JSON")
}

#[test]
fn ocr_matches_python() {
    let g = load_golden();
    for img in IMAGES {
        let key = img.replace(".png", "");
        let path = sample_path(img);
        let golden = &g[&key]["ocr"];
        let r = ocr::run(
            &path,
            &ocr::OcrParams {
                max_items: 100,
                min_conf: 0.5,
            },
        )
        .unwrap();
        // 结构字段精确（schema/task/sensors/coordsys）
        assert_structure(golden, &r, &format!("{key}.ocr"));
        let ge = golden["elements"].as_array().unwrap();
        let re = r["elements"].as_array().unwrap();
        // 元素配对：CER=0 + bbox IoU = 1.0（boundary 像素中心 + geo-clipper 修复后）
        assert_elements_paired(ge, re, &format!("{key}.ocr.elements"), 1.0);
    }
}
