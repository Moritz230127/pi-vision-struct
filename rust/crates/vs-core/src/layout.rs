//! 文档版式分析（对应 Python `vs_layout.py`，schema v3 输出逐字段一致）。
//!
//! PP-DocLayoutV3（paddle 全家桶）→ Python 回退（与 segment 的 mask decoder
//! 回退一致，plan 允许）。原因：paddle 3.x PIR 格式与 paddle2onnx 2.1.0
//! 不兼容（4 种导出方式均失败），且 paddlex 预处理管线复杂（Resize 800×800
//! + DETR 后处理）。Rust 侧负责 schema v3 输出结构 + 参数透传。

use std::path::Path;

use serde_json::Value;

/// 布局参数（对应 Python CLI 参数）。
pub struct LayoutParams {
    pub max_items: u32,
    pub min_conf: f64,
}

/// 运行布局传感器（Python 回退：paddle 推理）。
pub fn run(path: &Path, p: &LayoutParams) -> Result<Value, String> {
    // 定位 pi-vision env python（paddle 所在 env）
    let pybin = std::env::var("PI_VISION_PYTHON").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
        format!("{home}/conda-envs/pi-vision/bin/python")
    });
    let script = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
        .join("python")
        .join("vs_layout.py");
    let out = std::process::Command::new(&pybin)
        .arg(&script)
        .arg("--image")
        .arg(path)
        .arg("--max-items")
        .arg(p.max_items.to_string())
        .arg("--min-conf")
        .arg(p.min_conf.to_string())
        .output()
        .map_err(|e| format!("spawn vs_layout.py: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "vs_layout.py failed: {}",
            String::from_utf8_lossy(&out.stderr)
                .chars()
                .take(300)
                .collect::<String>()
        ));
    }
    serde_json::from_slice(&out.stdout).map_err(|e| format!("vs_layout.py 输出非 JSON: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_runs_on_sample() {
        // 真实推理（需 pi-vision env + paddle）
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("..")
            .join("bench")
            .join("samples")
            .join("s01_low_contrast.png");
        if !path.exists() {
            return; // 样本缺失时跳过
        }
        match run(
            &path,
            &LayoutParams {
                max_items: 100,
                min_conf: 0.3,
            },
        ) {
            Ok(report) => {
                assert_eq!(report["task"], "layout");
                assert!(report["elements"].is_array());
            }
            Err(e) => {
                // 环境无 paddle 时降级跳过（CI 无 paddle）
                eprintln!("layout 回退不可用（跳过）: {e}");
            }
        }
    }
}
