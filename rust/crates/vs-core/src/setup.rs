//! 引导自检与安装（对应 Python `setup/vs_setup.py --check`，纯 Rust）。
//!
//! `check`：只读健康检查（模型存在性 + 二进制完整性 + 环境检测）。
//! `setup`：模型目录创建 + 下载 URL 提示（V4 单二进制无需 conda env）。

use std::path::Path;

use serde_json::{json, Value};

/// 模型清单（V4 需要的全部模型文件）。
pub const MODELS: &[(&str, &str)] = &[
    ("u2net.onnx", "saliency（U²-Net）"),
    ("midas_v21_small_256.onnx", "depth（MiDaS-small）"),
    (
        "mobile_sam_encoder.onnx",
        "segment（MobileSAM image encoder）",
    ),
    ("mobile_sam.pt", "segment（MobileSAM 权重，decoder 回退）"),
    ("rapidocr/PP-OCRv6_det_small.onnx", "ocr（PP-OCRv6 det）"),
    ("rapidocr/PP-OCRv6_rec_small.onnx", "ocr（PP-OCRv6 rec）"),
    (
        "rapidocr/ch_ppocr_mobile_v2.0_cls_mobile.onnx",
        "ocr（PP-OCRv6 cls）",
    ),
    (
        "florence2/vision_encoder.onnx",
        "omniparser（Florence-2 vision encoder）",
    ),
    (
        "florence2/language_encoder.onnx",
        "omniparser（Florence-2 language encoder）",
    ),
    ("florence2/decoder.onnx", "omniparser（Florence-2 decoder）"),
    (
        "florence2/embed_tokens.onnx",
        "omniparser（Florence-2 embed tokens）",
    ),
    ("icon_detect.onnx", "omniparser（YOLOv9 icon detect）"),
    ("owlv2/text_encoder.onnx", "detect（OWLv2 text encoder）"),
    (
        "owlv2/vision_encoder.onnx",
        "detect（OWLv2 vision encoder）",
    ),
    (
        "owlv2/detection_head.onnx",
        "detect（OWLv2 detection head）",
    ),
];

/// 模型目录（与 Python `WEIGHTS` 一致）。
pub fn model_dir() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    format!("{home}/.cache/vsensor")
}

/// 只读健康检查（对应 Python `vs_setup.py --check`）。
pub fn check() -> Value {
    let dir = model_dir();
    let mut models = serde_json::Map::new();
    let mut missing: Vec<String> = Vec::new();
    for (name, desc) in MODELS {
        let p = Path::new(&dir).join(name);
        let ok = p.exists();
        models.insert(name.replace('/', "_"), json!({"exists": ok, "desc": desc}));
        if !ok {
            missing.push(name.to_string());
        }
    }
    // 二进制完整性：当前可执行文件存在
    let self_ok = std::env::current_exe().map(|p| p.exists()).unwrap_or(false);
    // 环境检测：Linux + 内核版本
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let kernel = std::fs::read_to_string("/proc/sys/kernel/osrelease")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "unknown".into());
    json!({
        "ok": missing.is_empty(),
        "mode": "check",
        "model_dir": dir,
        "models": models,
        "missing": missing,
        "binary": {"self_ok": self_ok},
        "env": {"os": os, "arch": arch, "kernel": kernel},
        "steps": [{
            "name": "health_check",
            "status": if missing.is_empty() { "ok" } else { "fail" },
            "detail": if missing.is_empty() {
                format!("{} 个模型全部就绪", MODELS.len())
            } else {
                format!("缺失 {} 个模型: {}", missing.len(), missing.join(", "))
            },
        }],
    })
}

/// 安装引导（对应 Python `vs_setup.py` 的模型部分）：创建目录 + 下载 URL 提示。
pub fn setup() -> Value {
    let dir = model_dir();
    let created = match std::fs::create_dir_all(&dir) {
        Ok(()) => true,
        Err(e) => {
            return json!({
                "ok": false,
                "mode": "setup",
                "error": format!("创建目录失败: {e}"),
            })
        }
    };
    // 子目录
    for sub in ["rapidocr", "florence2", "owlv2"] {
        let _ = std::fs::create_dir_all(Path::new(&dir).join(sub));
    }
    let missing: Vec<String> = MODELS
        .iter()
        .filter(|(name, _)| !Path::new(&dir).join(name).exists())
        .map(|(name, _)| name.to_string())
        .collect();
    json!({
        "ok": true,
        "mode": "setup",
        "model_dir": dir,
        "created": created,
        "missing": missing,
        "hint": "模型需手动下载（见 README 模型清单）；V4 单二进制无需 conda env",
        "next": ["运行 vs check 验证模型就绪"],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_reports_models() {
        let r = check();
        assert_eq!(r["mode"], "check");
        assert!(r["models"].as_object().unwrap().len() >= 10);
        assert!(r["binary"]["self_ok"].as_bool().unwrap());
    }

    #[test]
    fn setup_creates_dir() {
        // 用临时 HOME 避免污染真实目录
        let tmp = std::env::temp_dir().join("vsd_setup_test");
        let _ = std::fs::remove_dir_all(&tmp);
        std::env::set_var("HOME", &tmp);
        let r = setup();
        assert_eq!(r["ok"], true);
        assert!(Path::new(&tmp).join(".cache/vsensor").exists());
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
