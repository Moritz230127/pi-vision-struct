//! vsd — V4 常驻 daemon（S3 实现：调度器 + 热模型 + unix socket）。
//!
//! S0 阶段为占位桩：验证 tokio 运行时与 vs-core 链接正常。

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixListener;

const SOCKET_PATH: &str = "/tmp/vsd.sock";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 清理残留 socket（崩溃后重启）
    let _ = std::fs::remove_file(SOCKET_PATH);
    let listener = UnixListener::bind(SOCKET_PATH)?;
    println!("[vsd] listening on {SOCKET_PATH} (S3: scheduler + warm models pending)");

    loop {
        let (mut stream, _) = listener.accept().await?;
        // S3 实现：行 JSON 协议 + 调度器。当前仅回 pong。
        let mut buf = [0u8; 4096];
        let n = stream.read(&mut buf).await?;
        if n > 0 {
            stream
                .write_all(b"{\"ok\": true, \"pong\": true}\n")
                .await?;
        }
    }
}
