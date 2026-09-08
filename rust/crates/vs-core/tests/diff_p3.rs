//! P3 门禁：3D/协议差分测试（audit3d/analyze/protocol vs Python golden）。
//!
//! 运行：`cargo test -p vs-core --test diff_p3`。

use std::path::PathBuf;

use serde_json::{json, Value};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/p3")
        .join(name)
}

fn load_golden(name: &str) -> Value {
    let s = std::fs::read_to_string(fixture(name)).expect("golden 文件缺失");
    serde_json::from_str(&s).expect("golden JSON 解析失败")
}

/// audit3d：findings 逐字段一致（rule/method/distance_mm/element_ids）+ metrics。
#[test]
fn audit3d_matches_python() {
    let golden = load_golden("audit3d_golden.json");
    let blend_report =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/s7/blend_golden.json");
    let out = vs_core::audit3d::run(&vs_core::audit3d::Audit3dParams {
        report: blend_report.to_string_lossy().into_owned(),
        gap_threshold: 15.0,
        method: "auto".into(),
    })
    .expect("audit3d run 失败");

    // 结构
    assert_eq!(out["task"], "audit3d");
    assert_eq!(out["coordsys"], "world_m");

    // findings 逐字段
    let g_findings = golden["findings"].as_array().unwrap();
    let r_findings = out["findings"].as_array().unwrap();
    assert_eq!(r_findings.len(), g_findings.len(), "findings 数一致");
    for (r, g) in r_findings.iter().zip(g_findings.iter()) {
        assert_eq!(r["rule"], g["rule"], "rule 一致");
        assert_eq!(r["method"], g["method"], "method 一致");
        assert_eq!(r["severity"], g["severity"], "severity 一致");
        assert_eq!(r["element_ids"], g["element_ids"], "element_ids 一致");
        // distance_mm 容差（mm 精度）
        let rd = r["distance_mm"].as_f64().unwrap();
        let gd = g["distance_mm"].as_f64().unwrap();
        assert!(
            (rd - gd).abs() <= 0.01,
            "distance_mm 容差 0.01: rust={rd} golden={gd}"
        );
        // evidence 逐字段
        assert_eq!(r["evidence"]["a"], g["evidence"]["a"]);
        assert_eq!(r["evidence"]["b"], g["evidence"]["b"]);
        assert_eq!(r["evidence"]["collection_a"], g["evidence"]["collection_a"]);
        assert_eq!(r["evidence"]["collection_b"], g["evidence"]["collection_b"]);
    }

    // metrics
    assert_eq!(
        out["metrics"]["total_objects"],
        golden["metrics"]["total_objects"]
    );
    assert_eq!(
        out["metrics"]["mesh_objects"],
        golden["metrics"]["mesh_objects"]
    );
    assert_eq!(
        out["metrics"]["pair_checks"],
        golden["metrics"]["pair_checks"]
    );
    assert_eq!(
        out["metrics"]["interference_count"],
        golden["metrics"]["interference_count"]
    );
    assert_eq!(
        out["metrics"]["gap_warnings"],
        golden["metrics"]["gap_warnings"]
    );
    assert_eq!(
        out["metrics"]["verified_clearance"],
        golden["metrics"]["verified_clearance"]
    );
    assert_eq!(
        out["metrics"]["gap_threshold_mm"],
        golden["metrics"]["gap_threshold_mm"]
    );
    assert_eq!(out["metrics"]["methods"], golden["metrics"]["methods"]);
}

/// analyze：任务引擎输出结构一致（task/sensors/elements/skipped_steps）。
#[test]
fn analyze_matches_python() {
    let golden = load_golden("analyze_golden.json");
    let img = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .join("bench/samples/s03_alignment.png");
    let out = vs_core::analyze::run(&vs_core::analyze::AnalyzeParams {
        task: "diagnose-screenshot".into(),
        input: Some(img.to_string_lossy().into_owned()),
        url: None,
        compare: None,
        prompt: None,
        dpr: 1.0,
    })
    .expect("analyze run 失败");

    // 结构
    assert_eq!(out["schema"], "vision-report/v3");
    assert_eq!(out["task"], golden["task"], "task 一致");
    assert_eq!(out["sensors"], golden["sensors"], "sensors 一致");
    assert_eq!(
        out["skipped_steps"], golden["skipped_steps"],
        "skipped_steps 一致"
    );

    // elements 数一致（pix + ocr 合并）
    let g_els = golden["elements"].as_array().unwrap();
    let r_els = out["elements"].as_array().unwrap();
    assert_eq!(r_els.len(), g_els.len(), "elements 数一致");

    // 每个 golden element 在 Rust 输出中找到匹配（按 bbox + type）
    for g in g_els {
        let gb = g["bbox"].as_array().unwrap();
        let gt = g["type"].as_str().unwrap();
        let found = r_els.iter().any(|r| {
            r["type"].as_str() == Some(gt)
                && r["bbox"]
                    .as_array()
                    .map(|rb| {
                        rb.iter()
                            .zip(gb.iter())
                            .all(|(a, b)| (a.as_f64().unwrap() - b.as_f64().unwrap()).abs() <= 1.0)
                    })
                    .unwrap_or(false)
        });
        assert!(found, "Rust 缺 element type={gt} bbox={gb:?}");
    }
}

/// protocol analyze：粗报告结构一致（a11y 优先通道，plan5 F2）。
///
/// 注：Python golden 为 F2 前的 DL 通道（saliency/ascii/ocr/fusion）；
/// F2 后 analyze 默认 a11y 优先（无 DL，≤1s），golden 已过时。
/// 本测试验证当前架构契约：a11y 通道或 DL 兜底通道的结构一致性。
#[test]
fn protocol_analyze_matches_python() {
    let img = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .join("bench/samples/s03_alignment.png");
    let out = vs_core::protocol::run(&vs_core::protocol::ProtocolParams {
        action: "analyze".into(),
        image: img.to_string_lossy().into_owned(),
        region: None,
        bbox: None,
        sensor: None,
        confirm: None,
        verify_click: false,
        window: None,
    })
    .expect("protocol analyze 失败");

    assert_eq!(out["task"], "analyze");
    // 通道：a11y 优先（F2）或 DL 兜底
    let sensors: Vec<String> = out["sensors"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|s| s.as_str().map(String::from))
        .collect();
    let is_a11y = sensors.iter().any(|s| s == "a11y");
    let is_dl = sensors.iter().any(|s| s == "element_fusion");
    assert!(
        is_a11y || is_dl,
        "analyze 应为 a11y 通道或 DL 兜底通道: {sensors:?}"
    );
    // 融合元素清单存在（编号引用）
    let els = out["elements"].as_array().unwrap();
    assert!(!els.is_empty(), "analyze 应返回融合元素清单");
    for e in els {
        assert!(e.get("id").is_some(), "元素应带编号");
        assert!(e.get("sources").is_some(), "元素应带溯源");
    }
    // note 含已确认元素汇总（E6 输出模板）
    let note = out["note"].as_str().unwrap();
    assert!(note.contains("已确认元素"), "note 应含已确认元素: {note}");
    // 输出 < 3K tokens（≈12KB）
    let size = serde_json::to_string(&out).unwrap().len();
    assert!(
        size < 12_000,
        "analyze 输出应 < 3K tokens，实际 {size} bytes"
    );
}

/// protocol zoom：细报告结构一致（element_fusion 通道，plan3 E3）。
///
/// 注：Python golden 为 E3 前的原始传感器（ocr/segment）；E3 后 zoom
/// 用 element_fusion 融合（坐标偏移回原图），golden 已过时。
#[test]
fn protocol_zoom_matches_python() {
    let img = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .join("bench/samples/s03_alignment.png");
    let out = vs_core::protocol::run(&vs_core::protocol::ProtocolParams {
        action: "zoom".into(),
        image: img.to_string_lossy().into_owned(),
        region: Some("100,80,320,300".into()),
        bbox: None,
        sensor: None,
        confirm: None,
        verify_click: false,
        window: None,
    })
    .expect("protocol zoom 失败");

    assert_eq!(out["task"], "zoom");
    // E3 通道：element_fusion + ascii + edge + depth
    let sensors: Vec<String> = out["sensors"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|s| s.as_str().map(String::from))
        .collect();
    assert!(
        sensors.iter().any(|s| s == "element_fusion"),
        "zoom 应含 element_fusion: {sensors:?}"
    );
    assert_eq!(out["source"]["upscale"], 2, "zoom 应 2× upscale");
    assert_eq!(out["metrics"]["region"], json!([100, 80, 320, 300]));
    // 融合元素坐标偏移回原图（x ≥ 100, y ≥ 80）
    for e in out["elements"].as_array().unwrap() {
        if let Some(b) = e.get("bbox").and_then(|b| b.as_array()) {
            let x = b[0].as_f64().unwrap();
            let y = b[1].as_f64().unwrap();
            assert!(x >= 100.0, "bbox x 应偏移回原图: {x}");
            assert!(y >= 80.0, "bbox y 应偏移回原图: {y}");
        }
    }
    // ascii 局部存在
    assert!(out.get("ascii").is_some(), "zoom 应返回 ascii 局部");
}

/// protocol probe：单传感器定向证据结构一致。
#[test]
fn protocol_probe_matches_python() {
    let golden = load_golden("protocol_probe_golden.json");
    let img = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .join("bench/samples/s03_alignment.png");
    let out = vs_core::protocol::run(&vs_core::protocol::ProtocolParams {
        action: "probe".into(),
        image: img.to_string_lossy().into_owned(),
        region: None,
        bbox: Some("100,80,320,300".into()),
        sensor: Some("ocr".into()),
        confirm: None,
        verify_click: false,
        window: None,
    })
    .expect("protocol probe 失败");

    assert_eq!(out["task"], "probe");
    assert_eq!(out["sensors"], golden["sensors"], "sensors 一致");
    assert_eq!(out["source"]["bbox"], golden["source"]["bbox"]);
    let g_els = golden["elements"].as_array().unwrap();
    let r_els = out["elements"].as_array().unwrap();
    assert_eq!(r_els.len(), g_els.len(), "elements 数一致");
    // 文本逐字段一致
    for (r, g) in r_els.iter().zip(g_els.iter()) {
        assert_eq!(r["text"], g["text"], "文本一致");
    }
}
