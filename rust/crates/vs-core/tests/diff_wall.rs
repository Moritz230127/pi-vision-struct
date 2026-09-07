//! P1 门禁：wall（壁纸分类）与 Python `vs_wall.py` 逐字段一致 + setup 契约测试。
//!
//! 运行：`cargo test -p vs-core --test diff_wall`（需 pi-vision conda env 生成 golden）。

use std::path::PathBuf;

use serde_json::Value;
use vs_core::{setup, wall};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("wall")
}

/// 递归逐字段对比：数值精确（≤1e-9），字符串/布尔/结构严格相等。
fn assert_json_eq(a: &Value, b: &Value, path: &str) {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            let (xf, yf) = (x.as_f64().unwrap(), y.as_f64().unwrap());
            assert!(
                (xf - yf).abs() <= 1e-9,
                "{path}: {xf} != {yf} (diff {})",
                (xf - yf).abs()
            );
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
            assert_eq!(
                x.len(),
                y.len(),
                "{path}: object keys {} vs {}",
                x.len(),
                y.len()
            );
            for (k, xv) in x {
                // source.dir 为环境相关路径（Python 绝对路径 vs Rust 相对路径）
                if path.ends_with(".source") && k == "dir" {
                    continue;
                }
                let yv = y
                    .get(k)
                    .unwrap_or_else(|| panic!("{path}.{k}: missing in golden"));
                assert_json_eq(xv, yv, &format!("{path}.{k}"));
            }
        }
        _ => panic!("{path}: type mismatch {:?} vs {:?}", a, b),
    }
}

/// wall：与 vs_wall.py golden 逐字段一致（确定性输出精确）。
#[test]
fn wall_matches_python() {
    let golden: Value = serde_json::from_str(
        &std::fs::read_to_string(fixture_dir().join("wall_golden.json")).unwrap(),
    )
    .unwrap();
    let out = wall::run(&fixture_dir(), &wall::WallParams::default()).unwrap();
    assert_json_eq(&golden, &out, "wall");
}

/// setup check：契约测试（结构 + 类型 + 非空，环境相关字段不 golden）。
#[test]
fn setup_check_contract() {
    let r = setup::check();
    assert_eq!(r["mode"], "check");
    assert!(r["ok"].is_boolean());
    assert!(r["model_dir"].is_string());
    // 模型清单 ≥ 10 项
    let models = r["models"].as_object().unwrap();
    assert!(models.len() >= 10, "模型清单 {} 项", models.len());
    // 每项有 exists + desc
    for (name, v) in models {
        assert!(v["exists"].is_boolean(), "{name}: exists 缺失");
        assert!(v["desc"].is_string(), "{name}: desc 缺失");
    }
    // 二进制完整性
    assert!(r["binary"]["self_ok"].as_bool().unwrap());
    // 环境检测
    assert_eq!(r["env"]["os"], "linux");
    assert!(r["env"]["arch"].is_string());
    assert!(r["env"]["kernel"].is_string());
    // steps 非空
    assert!(!r["steps"].as_array().unwrap().is_empty());
}

/// setup：模型目录创建（临时 HOME）。
#[test]
fn setup_creates_model_dir() {
    let tmp = std::env::temp_dir().join("vsd_setup_contract");
    let _ = std::fs::remove_dir_all(&tmp);
    std::env::set_var("HOME", &tmp);
    let r = setup::setup();
    assert_eq!(r["ok"], true);
    assert!(std::path::Path::new(&tmp).join(".cache/vsensor").exists());
    let _ = std::fs::remove_dir_all(&tmp);
}
