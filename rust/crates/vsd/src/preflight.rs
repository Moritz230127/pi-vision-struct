//! preflight：daemon 启动前的环境检查 + 失败日志。
//!
//! 检查项：socket 目录可写 / 设备参数合法 / 关键路径存在。
//! 失败日志：统一写入 `~/.cache/vsensor/vsd.log`（追加，带时间戳）。

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// 失败日志路径。
#[allow(dead_code)] // 测试与外部调用用
pub fn log_path() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    Path::new(&home)
        .join(".cache")
        .join("vsensor")
        .join("vsd.log")
}

/// 追加失败日志（带时间戳）。
pub fn log_failure(context: &str, detail: &str) {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let line = format!("[{ts}] {context}: {detail}\n");
    if let Some(parent) = log_path().parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path())
    {
        let _ = f.write_all(line.as_bytes());
    }
}

/// preflight 检查：返回 Err 时 daemon 拒绝启动。
pub fn run(socket: &str, device: &str) -> Result<(), String> {
    // 1) socket 路径合法（父目录存在或可创建）
    let sock = Path::new(socket);
    if let Some(parent) = sock.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("socket 目录不可创建 {parent:?}: {e}"))?;
        }
    }
    // 2) 设备参数合法
    if device != "cuda" && device != "cpu" {
        return Err(format!("device 必须为 cuda/cpu，收到 {device}"));
    }
    // 3) 关键路径存在（vs-core 数据目录）
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    let cache = Path::new(&home).join(".cache").join("vsensor");
    if !cache.exists() {
        std::fs::create_dir_all(&cache).map_err(|e| format!("缓存目录不可创建: {e}"))?;
    }
    println!("[vsd] preflight OK (socket={socket}, device={device})");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preflight_valid() {
        let tmp = std::env::temp_dir().join("vsd_preflight_test.sock");
        assert!(run(tmp.to_str().unwrap(), "cuda").is_ok());
        assert!(run(tmp.to_str().unwrap(), "cpu").is_ok());
    }

    #[test]
    fn preflight_bad_device() {
        let tmp = std::env::temp_dir().join("vsd_preflight_bad.sock");
        assert!(run(tmp.to_str().unwrap(), "gpu").is_err());
    }

    #[test]
    fn failure_log_writes() {
        log_failure("test", "unit test failure");
        let content = std::fs::read_to_string(log_path()).unwrap_or_default();
        assert!(content.contains("test: unit test failure"));
    }
}
