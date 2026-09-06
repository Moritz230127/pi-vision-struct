//! S0 门禁：vs-core 与 Python vs_schema.py 的差分测试。
//!
//! 流程：`rust/tests/gen_schema_golden.py` 生成 golden JSON（Python 基线），
//! 本测试用 Rust 计算同组输入并逐字段对比（数值容差 ≤1e-6）。
//!
//! 运行：`cargo test -p vs-core --test diff_schema`（需 python3 可用）。

use serde_json::{json, Value};
use vs_core::{bbox, color, schema};

/// 递归逐字段对比两个 JSON 值；数值按容差 1e-6 比较，其余严格相等。
fn assert_json_eq(a: &Value, b: &Value, path: &str) {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            let (xf, yf) = (x.as_f64().unwrap(), y.as_f64().unwrap());
            assert!(
                (xf - yf).abs() <= 1e-6,
                "{path}: {xf} != {yf} (diff {})",
                (xf - yf).abs()
            );
        }
        (Value::String(x), Value::String(y)) => assert_eq!(x, y, "{path}"),
        (Value::Bool(x), Value::Bool(y)) => assert_eq!(x, y, "{path}"),
        (Value::Null, Value::Null) => {}
        (Value::Array(x), Value::Array(y)) => {
            assert_eq!(x.len(), y.len(), "{path}: array len");
            for (i, (xe, ye)) in x.iter().zip(y).enumerate() {
                assert_json_eq(xe, ye, &format!("{path}[{i}]"));
            }
        }
        (Value::Object(x), Value::Object(y)) => {
            assert_eq!(x.len(), y.len(), "{path}: object keys");
            for (k, xv) in x {
                let yv = y
                    .get(k)
                    .unwrap_or_else(|| panic!("{path}.{k}: missing in golden"));
                assert_json_eq(xv, yv, &format!("{path}.{k}"));
            }
        }
        _ => panic!("{path}: type mismatch {:?} vs {:?}", a, b),
    }
}

/// 运行 golden 生成脚本并解析输出。
fn load_golden() -> Value {
    // CARGO_MANIFEST_DIR = rust/crates/vs-core → 上两级到 rust/ → tests/gen_schema_golden.py
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("gen_schema_golden.py");
    let out = std::process::Command::new("python3")
        .arg(&script)
        .output()
        .expect("failed to run gen_schema_golden.py (python3 required)");
    assert!(
        out.status.success(),
        "gen_schema_golden.py failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).expect("golden output is not valid JSON")
}

#[test]
fn schema_matches_python() {
    let g = load_golden();
    assert_json_eq(&g["SCHEMA"], &json!(schema::SCHEMA), "SCHEMA");

    // element（全字段 + 最小字段）
    assert_json_eq(
        &g["element"],
        &serde_json::to_value(schema::Element {
            id: 1,
            etype: "text",
            bbox: &[1.0, 2.0, 3.0, 4.0],
            text: Some("hi"),
            conf: Some(0.9),
            color: None,
            font: None,
            z: None,
            source: &["ocr"],
            coordsys: "image_px",
        })
        .unwrap(),
        "element",
    );
    assert_json_eq(
        &g["element_min"],
        &serde_json::to_value(schema::Element {
            id: 0,
            etype: "icon",
            bbox: &[0.0, 0.0, 10.0, 10.0],
            text: None,
            conf: None,
            color: None,
            font: None,
            z: None,
            source: &[],
            coordsys: "image_px",
        })
        .unwrap(),
        "element_min",
    );

    // envelope（最小 + 全字段）
    assert_json_eq(
        &g["envelope"],
        &schema::envelope(Some("test"), &["pix"], Some("css_px"), None),
        "envelope",
    );
    assert_json_eq(
        &g["envelope_full"],
        &schema::envelope(
            Some("t"),
            &["a", "b"],
            Some("pt"),
            Some(json!({"type": "file", "path": "/tmp/x.png"})),
        ),
        "envelope_full",
    );

    // 原语记法
    assert_eq!(
        g["bbox_primitive"],
        json!(schema::bbox_primitive(&[1.0, 2.5, 3.0, 4.5]))
    );
    assert_eq!(
        g["point_primitive"],
        json!(schema::point_primitive(1.5, 2.5))
    );

    // anomaly（带/不带 suggested_cause）
    assert_json_eq(
        &g["anomaly"],
        &schema::anomaly(
            "overlap",
            &[0.0, 0.0, 10.0, 10.0],
            json!({"iou": 0.5}),
            0.8,
            Some("too close"),
        ),
        "anomaly",
    );
    assert_json_eq(
        &g["anomaly_min"],
        &schema::anomaly("gap", &[1.0, 1.0, 2.0, 2.0], json!({}), 0.5, None),
        "anomaly_min",
    );
}

#[test]
fn bbox_matches_python() {
    let g = load_golden();
    let arr = |v: &[i64; 4]| json!(v);

    assert_json_eq(
        &g["css_to_device"],
        &arr(&bbox::css_to_device(
            &[10.0, 20.0, 110.0, 220.0],
            2.0,
            (0, 0),
        )),
        "css_to_device",
    );
    assert_json_eq(
        &g["css_to_device_scroll"],
        &arr(&bbox::css_to_device(
            &[10.0, 20.0, 110.0, 220.0],
            2.0,
            (5, 5),
        )),
        "css_to_device_scroll",
    );
    assert_json_eq(
        &g["device_to_css"],
        &arr(&bbox::device_to_css(
            &[20.0, 40.0, 220.0, 440.0],
            2.0,
            (0, 0),
        )),
        "device_to_css",
    );
    assert_json_eq(
        &g["scale_bbox"],
        &arr(&bbox::scale_bbox(&[1.0, 2.0, 3.0, 4.0], 2.5)),
        "scale_bbox",
    );
    assert_json_eq(
        &g["clip_bbox"],
        &arr(&bbox::clip_bbox(&[-5.0, -5.0, 200.0, 200.0], 100, 100)),
        "clip_bbox",
    );

    assert_json_eq(
        &g["bbox_area"],
        &json!(bbox::bbox_area(&[0.0, 0.0, 10.0, 10.0])),
        "bbox_area",
    );
    assert_json_eq(
        &g["bbox_area_neg"],
        &json!(bbox::bbox_area(&[5.0, 5.0, 3.0, 3.0])),
        "bbox_area_neg",
    );

    let inter = bbox::bbox_intersection(&[0.0, 0.0, 10.0, 10.0], &[5.0, 5.0, 15.0, 15.0]);
    assert_json_eq(&g["bbox_intersection"], &json!(inter), "bbox_intersection");
    let none = bbox::bbox_intersection(&[0.0, 0.0, 1.0, 1.0], &[5.0, 5.0, 6.0, 6.0]);
    assert_json_eq(
        &g["bbox_intersection_none"],
        &json!(none),
        "bbox_intersection_none",
    );

    assert_json_eq(
        &g["bbox_iou_full"],
        &json!(bbox::bbox_iou(
            &[0.0, 0.0, 10.0, 10.0],
            &[0.0, 0.0, 10.0, 10.0]
        )),
        "bbox_iou_full",
    );
    assert_json_eq(
        &g["bbox_iou_half"],
        &json!(bbox::bbox_iou(
            &[0.0, 0.0, 10.0, 10.0],
            &[0.0, 5.0, 10.0, 10.0]
        )),
        "bbox_iou_half",
    );
    assert_json_eq(
        &g["bbox_iou_none"],
        &json!(bbox::bbox_iou(&[0.0, 0.0, 1.0, 1.0], &[5.0, 5.0, 6.0, 6.0])),
        "bbox_iou_none",
    );
}

#[test]
fn color_matches_python() {
    let g = load_golden();
    let t = |v: [f64; 3]| json!(v);

    assert_json_eq(
        &g["srgb_to_linear_0"],
        &json!(color::srgb_to_linear(0.0)),
        "srgb_0",
    );
    assert_json_eq(
        &g["srgb_to_linear_255"],
        &json!(color::srgb_to_linear(255.0)),
        "srgb_255",
    );
    assert_json_eq(
        &g["srgb_to_linear_128"],
        &json!(color::srgb_to_linear(128.0)),
        "srgb_128",
    );

    assert_json_eq(
        &g["relative_luminance_white"],
        &json!(color::relative_luminance([255.0, 255.0, 255.0])),
        "lum_white",
    );
    assert_json_eq(
        &g["relative_luminance_black"],
        &json!(color::relative_luminance([0.0, 0.0, 0.0])),
        "lum_black",
    );

    assert_json_eq(
        &g["contrast_ratio_bw"],
        &json!(color::contrast_ratio(
            [0.0, 0.0, 0.0],
            [255.0, 255.0, 255.0]
        )),
        "cr_bw",
    );
    assert_json_eq(
        &g["contrast_ratio_same"],
        &json!(color::contrast_ratio(
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0]
        )),
        "cr_same",
    );

    assert_json_eq(
        &g["rgb_to_lab_white"],
        &t(color::rgb_to_lab([255.0, 255.0, 255.0])),
        "lab_white",
    );
    assert_json_eq(
        &g["rgb_to_lab_red"],
        &t(color::rgb_to_lab([255.0, 0.0, 0.0])),
        "lab_red",
    );

    assert_json_eq(
        &g["delta_e76_same"],
        &json!(color::delta_e76([10.0, 20.0, 30.0], [10.0, 20.0, 30.0])),
        "de_same",
    );
    assert_json_eq(
        &g["delta_e76_bw"],
        &json!(color::delta_e76([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])),
        "de_bw",
    );

    assert_json_eq(
        &g["hex_to_rgb"],
        &json!(color::hex_to_rgb("#FF8040")),
        "hex_to_rgb",
    );
    assert_json_eq(
        &g["rgb_to_hex"],
        &json!(color::rgb_to_hex([255.0, 128.0, 64.0])),
        "rgb_to_hex",
    );
    assert_json_eq(
        &g["rgb_to_hex_round"],
        &json!(color::rgb_to_hex([254.5, 127.5, 63.5])),
        "rgb_to_hex_round",
    );
    assert_json_eq(
        &g["parse_hex"],
        &json!(color::parse_hex("ff8040").unwrap()),
        "parse_hex",
    );
}
