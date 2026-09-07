//! P2 门禁：cluster（CLIP 聚类）差分测试（vs Python vs_cluster.py golden）。
//!
//! 门禁（plan2 P2）：分组结果与 vs_cluster.py 完全一致（确定性输出精确）。

use std::path::Path;

use serde_json::Value;
use vs_core::cluster::{run, ClusterParams};

fn load_golden(name: &str) -> Value {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join(name);
    let s = std::fs::read_to_string(&p).expect("golden 文件缺失");
    serde_json::from_str(&s).expect("golden JSON 解析失败")
}

/// 聚类差分：clusters 分组 + metrics 与 Python 逐字段一致。
#[test]
fn cluster_matches_python() {
    let golden = load_golden("cluster_golden.json");
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
        .join("bench")
        .join("samples");
    let p = ClusterParams {
        dir: Some(dir.to_string_lossy().into_owned()),
        threshold: 0.75,
        max_files: 10,
        ..Default::default()
    };
    let out = run(&p).expect("cluster run 失败");

    // 文件名提取（Python golden 相对路径 vs Rust 绝对路径）
    let fname =
        |v: &Value| -> String { v.as_str().unwrap().rsplit('/').next().unwrap().to_string() };

    // metrics 逐字段一致
    let gm = &golden["metrics"];
    let rm = &out["metrics"];
    for key in [
        "image_count",
        "cluster_count",
        "singleton_count",
        "max_cluster_size",
    ] {
        assert_eq!(
            gm[key], rm[key],
            "metrics.{key}: py={} rust={}",
            gm[key], rm[key]
        );
    }
    assert!((gm["threshold"].as_f64().unwrap() - rm["threshold"].as_f64().unwrap()).abs() < 1e-9);

    // clusters 分组一致（representative + size + members 文件集合）
    let gc = golden["clusters"].as_array().unwrap();
    let rc = out["clusters"].as_array().unwrap();
    assert_eq!(gc.len(), rc.len(), "cluster 数不一致");
    for (gi, (g, r)) in gc.iter().zip(rc.iter()).enumerate() {
        assert_eq!(g["size"], r["size"], "cluster {gi} size 不一致");
        assert_eq!(
            fname(&g["representative"]),
            fname(&r["representative"]),
            "cluster {gi} representative 不一致"
        );
        // members 文件集合一致（顺序可能不同——Python 按 sim 降序，Rust 同）
        let gfiles: Vec<String> = g["members"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| fname(&m["file"]))
            .collect();
        let rfiles: Vec<String> = r["members"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| fname(&m["file"]))
            .collect();
        assert_eq!(gfiles, rfiles, "cluster {gi} members 不一致");
        // sim_to_rep 容差（CLIP 编码数值微差）
        for (gm_, rm_) in g["members"]
            .as_array()
            .unwrap()
            .iter()
            .zip(r["members"].as_array().unwrap())
        {
            let gs = gm_["sim_to_rep"].as_f64().unwrap();
            let rs = rm_["sim_to_rep"].as_f64().unwrap();
            assert!(
                (gs - rs).abs() <= 0.05,
                "cluster {gi} sim_to_rep: py={gs} rust={rs}"
            );
        }
    }

    // top_pairs 一致（a 文件 + b 在容差内候选 + sim 容差）
    let gp = golden["top_pairs"].as_array().unwrap();
    let rp = out["top_pairs"].as_array().unwrap();
    assert_eq!(gp.len(), rp.len(), "top_pairs 数不一致");
    for (pi, (g, r)) in gp.iter().zip(rp.iter()).enumerate() {
        assert_eq!(fname(&g["a"]), fname(&r["a"]), "top_pair {pi} a 不一致");
        // b 允许在容差内候选（sim 接近时排序可能翻转）
        let gb = fname(&g["b"]);
        let rb = fname(&r["b"]);
        let gs = g["sim"].as_f64().unwrap();
        let rs = r["sim"].as_f64().unwrap();
        if gb != rb {
            // 若 b 不同，则两者 sim 必须接近（排序翻转的容差内）
            assert!(
                (gs - rs).abs() <= 0.05,
                "top_pair {pi} b 不一致且 sim 差过大: py={gb}({gs}) rust={rb}({rs})"
            );
        } else {
            assert!(
                (gs - rs).abs() <= 0.05,
                "top_pair {pi} sim: py={gs} rust={rs}"
            );
        }
    }
}
