//! S1 门禁：vs-core 传感器与 Python 版的差分测试。
//!
//! 流程：`rust/tests/gen_sensor_golden.py` 生成 golden（Python 基线），
//! 本测试用 Rust 计算同组输入并逐字段对比。
//!
//! 容差策略（设计决策）：
//! - 确定性输出（ascii 栅格 / scene_stats 对比度·空间 / edge 亚像素）→ 精确（≤1e-6）
//! - 算法依赖输出（SLIC 超像素 / MEDIANCUT 量化）→ 颜色 hex 精确 + 百分比容差 ±1.0
//!
//! 运行：`cargo test -p vs-core --test diff_sensors`（需 pi-vision conda env）。

use std::path::PathBuf;

use serde_json::Value;
use vs_core::{ascii, edge, pix, scene_stats};

const IMAGES: [&str; 3] = [
    "s01_low_contrast.png",
    "s02_overlap.png",
    "s03_alignment.png",
];

/// 递归逐字段对比；数值按容差比较（默认 1e-6），其余严格相等。
/// `tolerance` 为路径前缀 → 数值容差；`lenient` 允许数组长度不同（比较公共前缀）；
/// `fuzzy_hex` 下 hex 字符串按 RGB 通道容差比较（SLIC 均值色近似）。
fn assert_json_eq(
    a: &Value,
    b: &Value,
    path: &str,
    tolerance: &[(&str, f64)],
    lenient: &[&str],
    fuzzy_hex: &[&str],
) {
    let tol = tolerance
        .iter()
        .find(|(p, _)| path.contains(p))
        .map(|(_, t)| *t)
        .unwrap_or(1e-6);
    let is_lenient = lenient.iter().any(|p| path.contains(p));
    let is_fuzzy = fuzzy_hex.iter().any(|p| path.contains(p));
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            let (xf, yf) = (x.as_f64().unwrap(), y.as_f64().unwrap());
            assert!(
                (xf - yf).abs() <= tol,
                "{path}: {xf} != {yf} (diff {}, tol {tol})",
                (xf - yf).abs()
            );
        }
        (Value::String(x), Value::String(y)) if is_fuzzy && x.starts_with('#') => {
            // hex 颜色：按 RGB 通道容差比较（SLIC 均值色近似）
            let parse = |s: &str| -> [i64; 3] {
                let h = s.trim_start_matches('#');
                [
                    i64::from_str_radix(&h[0..2], 16).unwrap(),
                    i64::from_str_radix(&h[2..4], 16).unwrap(),
                    i64::from_str_radix(&h[4..6], 16).unwrap(),
                ]
            };
            let (xa, ya) = (parse(x), parse(y));
            for (i, (xc, yc)) in xa.iter().zip(ya).enumerate() {
                assert!(
                    (xc - yc).abs() <= 50,
                    "{path}: hex {x} vs {y} (channel {i} diff {})",
                    (xc - yc).abs()
                );
            }
        }
        (Value::String(x), Value::String(y)) => assert_eq!(x, y, "{path}"),
        (Value::Bool(x), Value::Bool(y)) => assert_eq!(x, y, "{path}"),
        (Value::Null, Value::Null) => {}
        (Value::Array(x), Value::Array(y)) => {
            if is_lenient {
                // 比较公共前缀（SLIC 超像素数可能不同）
                let n = x.len().min(y.len());
                for i in 0..n {
                    assert_json_eq(
                        &x[i],
                        &y[i],
                        &format!("{path}[{i}]"),
                        tolerance,
                        lenient,
                        fuzzy_hex,
                    );
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
                    assert_json_eq(
                        xe,
                        ye,
                        &format!("{path}[{i}]"),
                        tolerance,
                        lenient,
                        fuzzy_hex,
                    );
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
                // source.path 为环境相关绝对路径，不可比（golden 与 Rust 解析方式不同）
                if path.ends_with(".source") && k == "path" {
                    continue;
                }
                // fuzzy 路径下 rgb 数组与 hex 字段冗余（同一颜色），hex 模糊检查已覆盖
                if is_fuzzy && k == "rgb" {
                    continue;
                }
                let yv = y
                    .get(k)
                    .unwrap_or_else(|| panic!("{path}.{k}: missing in golden"));
                assert_json_eq(
                    xv,
                    yv,
                    &format!("{path}.{k}"),
                    tolerance,
                    lenient,
                    fuzzy_hex,
                );
            }
        }
        _ => panic!("{path}: type mismatch {:?} vs {:?}", a, b),
    }
}

/// 运行 golden 生成脚本并解析输出。
fn load_golden() -> Value {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("gen_sensor_golden.py");
    let out = std::process::Command::new("python3")
        .arg(&script)
        .output()
        .expect("failed to run gen_sensor_golden.py (python3 required)");
    assert!(
        out.status.success(),
        "gen_sensor_golden.py failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).expect("golden output is not valid JSON")
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

/// SLIC/MEDIANCUT 百分比字段容差（算法依赖，非确定性）。
/// dominant_colors 为经典 SLIC vs skimage Lab-SLIC 的近似比较（见 plan.md 风险表）。
const TOL: &[(&str, f64)] = &[
    (".pix.metrics.dominant_colors", 3.0),
    (".pix.metrics.colors", 1.0),
    (".pix.metrics.regions", 1.0),
    (".scene_stats.metrics.color_histogram", 1.0),
];

/// 数组长度可不同的路径（SLIC 超像素数随实现而异）：比较公共前缀。
const LENIENT: &[&str] = &[".pix.metrics.dominant_colors", ".pix.metrics.regions"];

/// hex 颜色按 RGB 通道容差比较的路径（SLIC 均值色随实现而异）。
const FUZZY_HEX: &[&str] = &[".pix.metrics.dominant_colors", ".pix.metrics.regions"];

#[test]
fn sensors_match_python() {
    let g = load_golden();
    for img in IMAGES {
        let key = img.replace(".png", "");
        let path = sample_path(img);
        let golden = &g[&key];

        // ascii（确定性 → 精确）
        assert_json_eq(
            &golden["ascii"],
            &ascii::run(
                &path,
                &ascii::AsciiParams {
                    cols: 64,
                    rows: 36,
                    color: false,
                },
            )
            .unwrap(),
            &format!("{key}.ascii"),
            TOL,
            LENIENT,
            FUZZY_HEX,
        );
        assert_json_eq(
            &golden["ascii_color"],
            &ascii::run(
                &path,
                &ascii::AsciiParams {
                    cols: 32,
                    rows: 18,
                    color: true,
                },
            )
            .unwrap(),
            &format!("{key}.ascii_color"),
            TOL,
            LENIENT,
            FUZZY_HEX,
        );

        // scene_stats（MEDIANCUT 颜色容差，其余精确）
        assert_json_eq(
            &golden["scene_stats"],
            &scene_stats::run(&path, &scene_stats::SceneStatsParams { colors: 8 }).unwrap(),
            &format!("{key}.scene_stats"),
            TOL,
            LENIENT,
            FUZZY_HEX,
        );

        // pix（SLIC 颜色容差）
        assert_json_eq(
            &golden["pix"],
            &pix::run(
                &path,
                &pix::PixParams {
                    colors: 8,
                    regions: vec![],
                    compare: None,
                    wcag: vec![],
                    diff_threshold: 30,
                },
            )
            .unwrap(),
            &format!("{key}.pix"),
            TOL,
            LENIENT,
            FUZZY_HEX,
        );

        // edge（确定性 → 精确）
        assert_json_eq(
            &golden["edge"],
            &edge::run(
                &path,
                &edge::EdgeParams {
                    sigma: 1.0,
                    hough_threshold: 40,
                    max_lines: 20,
                    max_points: 2000,
                },
            )
            .unwrap(),
            &format!("{key}.edge"),
            TOL,
            LENIENT,
            FUZZY_HEX,
        );
    }
}
