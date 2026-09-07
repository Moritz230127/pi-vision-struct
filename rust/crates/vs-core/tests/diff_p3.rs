//! P3 门禁：3D/协议差分测试（audit3d/analyze/protocol vs Python golden）。
//!
//! 运行：`cargo test -p vs-core --test diff_p3`。

use std::path::PathBuf;

use serde_json::Value;

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

/// protocol analyze：粗报告结构一致（candidates/ascii/metrics）。
#[test]
fn protocol_analyze_matches_python() {
    let golden = load_golden("protocol_analyze_golden.json");
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
    })
    .expect("protocol analyze 失败");

    assert_eq!(out["task"], "analyze");
    assert_eq!(out["sensors"], golden["sensors"], "sensors 一致");
    // candidates 数一致（saliency 候选）
    let g_cands = golden["candidates"].as_array().unwrap();
    let r_cands = out["candidates"].as_array().unwrap();
    assert_eq!(r_cands.len(), g_cands.len(), "candidates 数一致");
    // ascii 结构
    assert_eq!(out["ascii"]["cols"], golden["ascii"]["cols"]);
    assert_eq!(out["ascii"]["rows"], golden["ascii"]["rows"]);
    // metrics 键一致
    assert_eq!(out["metrics"]["ocr_items"], golden["metrics"]["ocr_items"]);
    assert_eq!(
        out["metrics"]["saliency_candidates"],
        golden["metrics"]["saliency_candidates"]
    );
}

/// protocol zoom：细报告结构一致（elements/metrics/region）。
#[test]
fn protocol_zoom_matches_python() {
    let golden = load_golden("protocol_zoom_golden.json");
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
    })
    .expect("protocol zoom 失败");

    assert_eq!(out["task"], "zoom");
    assert_eq!(out["sensors"], golden["sensors"], "sensors 一致");
    assert_eq!(out["metrics"]["region"], golden["metrics"]["region"]);
    // elements 数容差（edge 点数受 resize 滤波器影响）
    let g_els = golden["elements"].as_array().unwrap().len();
    let r_els = out["elements"].as_array().unwrap().len();
    assert!(
        (r_els as i64 - g_els as i64).abs() <= 5,
        "elements 数容差 ±5: rust={r_els} golden={g_els}"
    );
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
