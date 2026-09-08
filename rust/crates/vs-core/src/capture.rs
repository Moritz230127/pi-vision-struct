//! 截屏传感器（对应 Python `vs_capture.py`：grim/mss → xcap，schema v3 输出）。
//!
//! xcap（Wayland screencopy / X11 / Windows / macOS）截屏 → 保存 PNG +
//! 输出 schema v3 envelope（source.type=screen + size_px + 图像路径）。

use serde_json::{json, Value};

use crate::schema::envelope;

/// 截屏参数（对应 Python `--region`）。
pub struct CaptureParams {
    /// 区域 [x1, y1, x2, y2]（None = 全屏）。
    pub region: Option<[u32; 4]>,
    /// 输出 PNG 路径。
    pub out: String,
}

/// 截屏并输出 schema v3 envelope。
///
/// 在独立线程执行 + 超时保护（Wayland screencopy 在无 portal 环境可能阻塞）。
pub fn run(p: &CaptureParams) -> Result<Value, String> {
    const TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);
    let p = CaptureParams {
        region: p.region,
        out: p.out.clone(),
    };
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(capture_inner(&p));
    });
    match rx.recv_timeout(TIMEOUT) {
        Ok(r) => r,
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            Err("capture timeout (Wayland screencopy 无响应，需 portal 会话)".into())
        }
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            Err("capture thread panicked".into())
        }
    }
}

fn capture_inner(p: &CaptureParams) -> Result<Value, String> {
    // 优先 grim（Wayland 可靠）；失败降级 xcap
    if let Ok(v) = capture_grim(p) {
        return Ok(v);
    }
    capture_xcap(p)
}

/// grim 截图（Wayland screencopy，niri 环境可靠）。
fn capture_grim(p: &CaptureParams) -> Result<Value, String> {
    let out = std::process::Command::new("grim")
        .arg(&p.out)
        .output()
        .map_err(|e| format!("spawn grim: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "grim failed: {}",
            String::from_utf8_lossy(&out.stderr)
                .chars()
                .take(200)
                .collect::<String>()
        ));
    }
    // 区域裁剪（grim 全屏 → 按 region 裁剪）
    let full = image::open(&p.out).map_err(|_e| format!("open {}", p.out))?;
    let img = match p.region {
        Some([x1, y1, x2, y2]) => {
            let (x, y) = (x1.min(x2), y1.min(y2));
            let (w, h) = (x2.abs_diff(x1), y2.abs_diff(y1));
            let cropped = image::imageops::crop_imm(&full, x, y, w, h).to_image();
            image::DynamicImage::ImageRgba8(cropped).to_rgb8()
        }
        None => full.to_rgb8(),
    };
    img.save(&p.out)
        .map_err(|e| format!("save {}: {e}", p.out))?;
    let (w, h) = (img.width(), img.height());
    let mut report = envelope(
        Some("capture"),
        &["grim"],
        Some("screen_px"),
        Some(json!({
            "type": "screen",
            "path": p.out,
            "size_px": [w, h],
            "backend": "grim",
        })),
    );
    report["metrics"] = json!({"size_px": [w, h]});
    Ok(report)
}

/// xcap 截图（跨平台兜底）。
fn capture_xcap(p: &CaptureParams) -> Result<Value, String> {
    use xcap::Monitor;

    let monitors = Monitor::all().map_err(|e| format!("xcap Monitor::all: {e}"))?;
    if monitors.is_empty() {
        return Err("xcap: no monitor found".into());
    }
    // 主显示器（索引 0，与 Python mss monitors[0] 一致）
    let mon = &monitors[0];
    let img = mon
        .capture_image()
        .map_err(|e| format!("xcap capture_image: {e}"))?;

    // 区域裁剪（region 坐标在屏幕空间）
    let cropped = match p.region {
        Some([x1, y1, x2, y2]) => {
            let (x, y) = (x1.min(x2), y1.min(y2));
            let (w, h) = (x2.abs_diff(x1), y2.abs_diff(y1));
            image::imageops::crop_imm(&img, x, y, w, h).to_image()
        }
        None => img,
    };

    // 保存 PNG
    cropped
        .save(&p.out)
        .map_err(|e| format!("save {}: {e}", p.out))?;

    let (w, h) = (cropped.width(), cropped.height());
    let mut report = envelope(
        Some("capture"),
        &["xcap"],
        Some("screen_px"),
        Some(json!({
            "type": "screen",
            "path": p.out,
            "size_px": [w, h],
            "backend": "xcap",
        })),
    );
    report["metrics"] = json!({"size_px": [w, h]});
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_params_default() {
        // 参数结构可用性（真实截屏在冒烟测试验证）
        let p = CaptureParams {
            region: None,
            out: "/tmp/vsd_capture_test.png".into(),
        };
        assert!(p.region.is_none());
        assert_eq!(p.out, "/tmp/vsd_capture_test.png");
    }
}
