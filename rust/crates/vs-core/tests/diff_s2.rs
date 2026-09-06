//! S2 门禁：vs-core 融合/规则/审计/几何与 Python 版的差分测试。
//!
//! 流程：`rust/tests/gen_s2_golden.py` 生成 golden（Python 基线），
//! 本测试用 Rust 计算同组输入并逐字段对比。
//!
//! 容差策略：rules/audit/fusion 确定性输出精确（≤1e-6）；
//! geometry 的 VTracer 版本可能与 Python 绑定有差异 → 形状数容差。

use std::path::PathBuf;

use serde_json::Value;
use vs_core::{audit, fusion, geometry, rules};

/// 递归逐字段对比；数值按容差比较（默认 1e-6），其余严格相等。
/// `lenient` 路径下数组比较公共前缀；`num_tol` 为路径 → 数值容差。
fn assert_json_eq(
    a: &Value,
    b: &Value,
    path: &str,
    tol: f64,
    lenient: &[&str],
    num_tol: &[(&str, f64)],
) {
    let is_lenient = lenient.iter().any(|p| path.contains(p));
    let path_tol = num_tol
        .iter()
        .find(|(p, _)| path.contains(p))
        .map(|(_, t)| *t)
        .unwrap_or(tol);
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            let (xf, yf) = (x.as_f64().unwrap(), y.as_f64().unwrap());
            assert!(
                (xf - yf).abs() <= path_tol,
                "{path}: {xf} != {yf} (diff {}, tol {path_tol})",
                (xf - yf).abs()
            );
        }
        (Value::String(x), Value::String(y)) => assert_eq!(x, y, "{path}"),
        (Value::Bool(x), Value::Bool(y)) => assert_eq!(x, y, "{path}"),
        (Value::Null, Value::Null) => {}
        (Value::Array(x), Value::Array(y)) => {
            if is_lenient {
                let n = x.len().min(y.len());
                for i in 0..n {
                    assert_json_eq(&x[i], &y[i], &format!("{path}[{i}]"), tol, lenient, num_tol);
                }
            } else {
                assert_eq!(
                    x.len(),
                    y.len(),
                    "{path}: array len {} vs {}",
                    x.len(),
                    y.len()
                );
                for (i, (xe, ye)) in x.iter().zip(y).enumerate() {
                    assert_json_eq(xe, ye, &format!("{path}[{i}]"), tol, lenient, num_tol);
                }
            }
        }
        (Value::Object(x), Value::Object(y)) => {
            assert_eq!(
                x.len(),
                y.len(),
                "{path}: object keys {} vs {}",
                x.len(),
                y.len()
            );
            for (k, xv) in x {
                // source.path/from/reports 为环境相关路径，不可比
                if path.ends_with(".source") && matches!(k.as_str(), "path" | "from" | "reports") {
                    continue;
                }
                // geometry 元素/类型因 VTracer 版本差异不可逐项比（仅比结构/shapes 数）
                if path.starts_with("geometry.") && matches!(k.as_str(), "elements" | "types") {
                    continue;
                }
                let yv = y
                    .get(k)
                    .unwrap_or_else(|| panic!("{path}.{k}: missing in golden"));
                assert_json_eq(xv, yv, &format!("{path}.{k}"), tol, lenient, num_tol);
            }
        }
        _ => panic!("{path}: type mismatch {:?} vs {:?}", a, b),
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
        .join("gen_s2_golden.py");
    let out = std::process::Command::new("python3")
        .arg(&script)
        .output()
        .expect("failed to run gen_s2_golden.py (python3 required)");
    assert!(
        out.status.success(),
        "gen_s2_golden.py failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).expect("golden output is not valid JSON")
}

#[test]
fn rules_match_python() {
    let g = load_golden();
    for (key, golden) in g["rules"].as_object().unwrap() {
        let path = sample_path(&format!("{key}.report.json"));
        let r = rules::run(&path, &rules::RulesParams::default()).unwrap();
        assert_json_eq(golden, &r, &format!("rules.{key}"), 1e-6, &[], &[]);
    }
}

#[test]
fn audit_match_python() {
    let g = load_golden();
    for (key, golden) in g["audit"].as_object().unwrap() {
        let path = sample_path(&format!("{key}.report.json"));
        let r = audit::run(
            &path,
            &audit::AuditParams {
                canvas: None,
                overlap_threshold: 0.05,
                min_overlap_area: 400.0,
            },
        )
        .unwrap();
        assert_json_eq(golden, &r, &format!("audit.{key}"), 1e-6, &[], &[]);
    }
}

#[test]
fn fusion_match_python() {
    let g = load_golden();
    for (key, golden) in g["fusion"].as_object().unwrap() {
        let idx: usize = key.trim_start_matches("pair").parse().unwrap();
        let reports = [
            "s01_low_contrast",
            "s02_overlap",
            "s03_alignment",
            "s04_spacing",
        ];
        let (ra, rb) = match idx {
            0 => (reports[0], reports[1]),
            _ => (reports[2], reports[3]),
        };
        let pa = sample_path(&format!("{ra}.report.json"));
        let pb = sample_path(&format!("{rb}.report.json"));
        let r = fusion::run(&[pa, pb], None).unwrap();
        assert_json_eq(golden, &r, &format!("fusion.{key}"), 1e-6, &[], &[]);
    }
}

#[test]
fn geometry_match_python() {
    let g = load_golden();
    for (key, golden) in g["geometry"].as_object().unwrap() {
        let path = sample_path(&format!("{key}.png"));
        let r = geometry::run(
            &path,
            &geometry::GeometryParams {
                mode: "polygon".to_string(),
                max_shapes: 20,
            },
        )
        .unwrap();
        // VTracer 版本差异 → 形状数容差 ±3，其余精确
        let g_shapes = golden["metrics"]["shapes"].as_u64().unwrap();
        let r_shapes = r["metrics"]["shapes"].as_u64().unwrap();
        assert!(
            (g_shapes as i64 - r_shapes as i64).abs() <= 3,
            "geometry.{key}: shapes {} vs {}",
            g_shapes,
            r_shapes
        );
        assert_json_eq(
            golden,
            &r,
            &format!("geometry.{key}"),
            1e-6,
            &[".geometry.", ".elements"],
            &[(".shape.points", 2.0), (".metrics.shapes", 3.0)],
        );
    }
}
