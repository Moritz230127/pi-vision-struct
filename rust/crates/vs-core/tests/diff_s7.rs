//! S7 门禁：外部集成差分测试（pdf/pptx/blend vs Python golden）。
//!
//! 运行：`cargo test -p vs-core --test diff_s7`（需 fixtures + pdfium 库）。

use std::path::PathBuf;

use serde_json::Value;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/s7")
        .join(name)
}

fn load_golden(name: &str) -> Value {
    let s = std::fs::read_to_string(fixture(name)).expect("golden 文件缺失");
    serde_json::from_str(&s).expect("golden JSON 解析失败")
}

/// pdf：文本逐字段一致 + bbox 容差（pdfium vs PyMuPDF 引擎差异 ±5pt）。
#[test]
fn pdf_matches_python() {
    let golden = load_golden("pdf_golden.json");
    let out = vs_core::pdf::run(&vs_core::pdf::PdfParams {
        file: fixture("vsd_test.pdf").to_string_lossy().into_owned(),
        pages: None,
    })
    .expect("pdf run 失败");

    // 结构
    assert_eq!(out["task"], "pdf");
    assert_eq!(out["source"]["pages"], golden["source"]["pages"]);
    assert_eq!(
        out["metrics"]["page_count"],
        golden["metrics"]["page_count"]
    );

    // 元素数一致
    let g_els = golden["elements"].as_array().unwrap();
    let r_els = out["elements"].as_array().unwrap();
    assert_eq!(r_els.len(), g_els.len(), "元素数一致");

    // 文本逐字段一致 + bbox 容差
    for (r, g) in r_els.iter().zip(g_els.iter()) {
        assert_eq!(r["text"], g["text"], "文本一致");
        assert_eq!(r["page"], g["page"], "页码一致");
        let rb = r["bbox"].as_array().unwrap();
        let gb = g["bbox"].as_array().unwrap();
        for k in 0..4 {
            let diff = (rb[k].as_f64().unwrap() - gb[k].as_f64().unwrap()).abs();
            assert!(
                diff <= 5.0,
                "bbox[{k}] 容差 ±5pt: rust={} golden={}",
                rb[k],
                gb[k]
            );
        }
    }
}

/// pptx：结构逐字段一致（类型/位置/尺寸/文本）。
#[test]
fn pptx_matches_python() {
    let golden = load_golden("pptx_golden.json");
    let out = vs_core::pptx::run(&vs_core::pptx::PptxParams {
        file: fixture("vsd_test.pptx").to_string_lossy().into_owned(),
        slide: None,
    })
    .expect("pptx run 失败");

    assert_eq!(
        out["source"]["total_slides"],
        golden["source"]["total_slides"]
    );
    assert_eq!(
        out["source"]["slide_size_pt"],
        golden["source"]["slide_size_pt"]
    );

    let g_slides = golden["slides"].as_array().unwrap();
    let r_slides = out["slides"].as_array().unwrap();
    assert_eq!(r_slides.len(), g_slides.len(), "幻灯片数一致");

    for (rs, gs) in r_slides.iter().zip(g_slides.iter()) {
        assert_eq!(rs["slide"], gs["slide"]);
        assert_eq!(rs["shape_count"], gs["shape_count"], "形状数一致");
        let r_shapes = rs["shapes"].as_array().unwrap();
        let g_shapes = gs["shapes"].as_array().unwrap();
        for (r, g) in r_shapes.iter().zip(g_shapes.iter()) {
            assert_eq!(r["type"], g["type"], "形状类型一致");
            assert_eq!(r["pos_pt"], g["pos_pt"], "位置一致");
            assert_eq!(r["size_pt"], g["size_pt"], "尺寸一致");
            assert_eq!(r["rotation"], g["rotation"], "旋转一致");
            assert_eq!(r["texts"], g["texts"], "文本一致");
        }
    }
}

/// blend：对象数/名称/类型/tris 一致 + bbox3d 容差（blend-rs vs bpy）。
#[test]
fn blend_matches_python() {
    let golden = load_golden("blend_golden.json");
    let blend = "/home/Arch/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/blend-rs-0.3.0/examples/example-3.2.blend";
    let out = vs_core::blend::run(&vs_core::blend::BlendParams { file: blend.into() })
        .expect("blend run 失败");

    // metrics
    assert_eq!(
        out["metrics"]["object_count"],
        golden["metrics"]["object_count"]
    );
    assert_eq!(
        out["metrics"]["total_tris"],
        golden["metrics"]["total_tris"]
    );

    // 对象逐字段（按名称匹配，不依赖顺序——scene.objects 顺序与文件存储序不同）
    let g_objs = golden["objects3d"].as_array().unwrap();
    let r_objs = out["objects3d"].as_array().unwrap();
    assert_eq!(r_objs.len(), g_objs.len(), "对象数一致");
    for g in g_objs {
        let gname = g["name"].as_str().unwrap();
        let r = r_objs
            .iter()
            .find(|o| o["name"].as_str() == Some(gname))
            .unwrap_or_else(|| panic!("Rust 缺对象 {gname}"));
        assert_eq!(r["type"], g["type"], "对象类型一致");
        assert_eq!(r["tris"], g["tris"], "面数一致");
        // bbox3d 容差（blend-rs 顶点 vs bpy bound_box）
        let rb = r["bbox3d"].as_array().unwrap();
        let gb = g["bbox3d"].as_array().unwrap();
        assert_eq!(rb.len(), gb.len(), "bbox3d 角点数一致");
        for (rc, gc) in rb.iter().zip(gb.iter()) {
            let rv = rc.as_array().unwrap();
            let gv = gc.as_array().unwrap();
            for k in 0..3 {
                let diff = (rv[k].as_f64().unwrap() - gv[k].as_f64().unwrap()).abs();
                assert!(
                    diff <= 0.01,
                    "bbox3d[{k}] 容差 0.01: rust={} golden={}",
                    rv[k],
                    gv[k]
                );
            }
        }
        // matrix_world 容差
        let rm = r["matrix_world"].as_array().unwrap();
        let gm = g["matrix_world"].as_array().unwrap();
        for k in 0..16 {
            let diff = (rm[k].as_f64().unwrap() - gm[k].as_f64().unwrap()).abs();
            assert!(diff <= 1e-6, "matrix_world[{k}] 容差 1e-6");
        }
    }
}
