//! vsd — V4 常驻 daemon（S3：调度器 + 热模型 + unix socket）。
//!
//! 协议（行分隔 JSON，与 Python `vsd.py` 兼容）：
//!   {"model": "saliency", "args": {...}, "priority": 0}  → 推理请求
//!   {"ping": true}   → {"ok": true, "pong": true}
//!   {"stats": true}  → {"ok": true, "stats": {...}}
//!   → {"ok": true, "report": {...}} 或 {"ok": false, "error": "..."}

use std::sync::Arc;
use std::time::Duration;

use clap::Parser;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixListener;
use vsd::preflight;
use vsd::sandbox::{self, SandboxConfig};
use vsd::scheduler::{self, Scheduler};

/// 默认 socket 路径（与 Python vsd.py 一致）。
const DEFAULT_SOCKET: &str = "/tmp/vsd.sock";

#[derive(Parser)]
#[command(
    name = "vsd",
    version,
    about = "V4 常驻视觉 daemon（调度器 + unix socket + 沙箱）"
)]
struct Cli {
    /// unix socket 路径
    #[arg(long, default_value = DEFAULT_SOCKET)]
    socket: String,
    /// 设备（cuda/cpu）
    #[arg(long, default_value = "cuda")]
    device: String,
    /// 推理超时（秒）
    #[arg(long, default_value_t = 120.0)]
    timeout: f64,
    /// 沙箱（landlock + seccomp + namespace；默认开启）
    #[arg(long, default_value_t = true)]
    sandbox: bool,
    /// 事件驱动模式：监听 niri event-stream，变化即再感知（默认关闭）
    #[arg(long)]
    watch: bool,
    /// 事件节流（毫秒）
    #[arg(long, default_value_t = 100)]
    throttle_ms: u64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // preflight：环境检查（socket 目录 / 权限 / 设备）
    preflight::run(&cli.socket, &cli.device)?;

    // 沙箱：landlock + seccomp + namespace（推理路径全程沙箱化）
    if cli.sandbox {
        // --watch 模式需要 spawn niri event-stream（受控 exec），放行 execve
        let cfg = SandboxConfig {
            allow_exec: cli.watch,
            ..SandboxConfig::default()
        };
        match sandbox::apply(&cfg) {
            Ok(()) => println!("[vsd] sandbox applied (landlock + seccomp + namespace)"),
            Err(e) => {
                // 沙箱失败不阻断（无 CAP_SYS_ADMIN 环境降级），但记录
                eprintln!("[vsd] sandbox warning: {e}");
            }
        }
    }

    // 清理残留 socket（崩溃后重启）
    let _ = std::fs::remove_file(&cli.socket);
    let listener = UnixListener::bind(&cli.socket)?;
    println!("[vsd] listening on {} (device={})", cli.socket, cli.device);

    // 调度器（6.5GB 预算 + 1 GPU 槽位）
    let sched = scheduler::default_scheduler();
    // 注册推理函数：真实调用 vs-core 传感器（S4/S5 模型）
    sched.register(
        "saliency",
        Box::new(|_model, args, _use_gpu| {
            let image = args
                .get("image")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "缺少 args.image".to_string())?;
            let top_n = args.get("top_n").and_then(|v| v.as_u64()).unwrap_or(5) as u32;
            let min_score = args
                .get("min_score")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.3);
            vs_core::saliency::run(
                std::path::Path::new(image),
                &vs_core::saliency::SaliencyParams { top_n, min_score },
            )
        }),
        None,
    );
    sched.register(
        "segment",
        Box::new(|_model, args, _use_gpu| {
            let image = args
                .get("image")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "缺少 args.image".to_string())?;
            let min_area = args
                .get("min_area_ratio")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.01);
            let max_instances = args
                .get("max_instances")
                .and_then(|v| v.as_u64())
                .unwrap_or(10) as u32;
            vs_core::segment::run(
                std::path::Path::new(image),
                &vs_core::segment::SegmentParams {
                    points: vec![],
                    labels: vec![],
                    min_area,
                    max_instances,
                },
            )
        }),
        None,
    );
    sched.register(
        "depth",
        Box::new(|_model, args, _use_gpu| {
            let image = args
                .get("image")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "缺少 args.image".to_string())?;
            let bins = args.get("bins").and_then(|v| v.as_u64()).unwrap_or(3) as u32;
            let region = args
                .get("region")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            vs_core::depth::run(
                std::path::Path::new(image),
                &vs_core::depth::DepthParams { bins, region },
            )
        }),
        None,
    );
    // R3：analyze 模型（PerceptionCache 缓存命中 ≤1s，连续 analyze 同窗口不重复感知）
    sched.register(
        "analyze",
        Box::new(|_model, args, _use_gpu| {
            let image = args
                .get("image")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "缺少 args.image".to_string())?;
            let window = args
                .get("window")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            vs_core::protocol::run(&vs_core::protocol::ProtocolParams {
                action: "analyze".into(),
                image: image.to_string(),
                region: None,
                bbox: None,
                sensor: None,
                confirm: None,
                verify_click: false,
                window,
            })
        }),
        None,
    );

    // 事件驱动模式：监听 niri event-stream，变化即再感知
    if cli.watch {
        let throttle = cli.throttle_ms;
        std::thread::spawn(
            move || match vs_core::events::EventWatcher::start(throttle) {
                Ok(mut watcher) => {
                    println!("[vsd] event watch started (throttle={throttle}ms)");
                    while let Some(ev) = watcher.next_event() {
                        if ev.triggers_reperception() {
                            println!("[vsd] event: {ev:?} → 触发再感知");
                        }
                    }
                }
                Err(e) => eprintln!("[vsd] event watch failed: {e}"),
            },
        );
    }

    loop {
        let (stream, _) = listener.accept().await?;
        let sched = sched.clone();
        let timeout = Duration::from_secs_f64(cli.timeout);
        tokio::spawn(async move {
            handle_conn(stream, sched, timeout).await;
        });
    }
}

/// 处理单个连接：读一行 JSON → 响应。
async fn handle_conn(mut stream: tokio::net::UnixStream, sched: Arc<Scheduler>, timeout: Duration) {
    let mut reader = BufReader::new(&mut stream);
    let mut line = String::new();
    let n = match reader.read_line(&mut line).await {
        Ok(n) => n,
        Err(_) => return,
    };
    if n == 0 {
        return;
    }
    let resp = match serde_json::from_str::<serde_json::Value>(line.trim()) {
        Ok(req) => handle_request(req, &sched, timeout).await,
        Err(_) => serde_json::json!({"ok": false, "error": "invalid JSON"}),
    };
    let mut out = serde_json::to_string(&resp).unwrap_or_else(|_| "{}".into());
    out.push('\n');
    let _ = stream.write_all(out.as_bytes()).await;
}

/// 处理请求：ping / stats / 推理。
async fn handle_request(
    req: serde_json::Value,
    sched: &Arc<Scheduler>,
    timeout: Duration,
) -> serde_json::Value {
    if req.get("ping").and_then(|v| v.as_bool()).unwrap_or(false) {
        return serde_json::json!({"ok": true, "pong": true});
    }
    if req.get("stats").and_then(|v| v.as_bool()).unwrap_or(false) {
        return serde_json::json!({"ok": true, "stats": sched.stats()});
    }
    let Some(model) = req.get("model").and_then(|v| v.as_str()) else {
        return serde_json::json!({"ok": false, "error": "missing model"});
    };
    let args = req.get("args").cloned().unwrap_or(serde_json::json!({}));
    let priority = req
        .get("priority")
        .and_then(|v| v.as_u64())
        .unwrap_or(scheduler::PRIORITY_PROBE as u64) as u32;
    // 提交任务并等待完成（S3 简化：同步等待；S4 可改异步回调）
    let (tx, rx) = tokio::sync::oneshot::channel();
    let sched2 = sched.clone();
    let model2 = model.to_string();
    std::thread::spawn(move || {
        // 通过调度器执行（阻塞工作线程）
        let result = run_in_scheduler(&sched2, &model2, args, priority, timeout);
        let _ = tx.send(result);
    });
    match tokio::time::timeout(timeout + Duration::from_secs(5), rx).await {
        Ok(Ok(result)) => {
            serde_json::json!({"ok": result.get("error").is_none(), "report": result})
        }
        Ok(Err(_)) => serde_json::json!({"ok": false, "error": "channel closed"}),
        Err(_) => serde_json::json!({"ok": false, "error": "vsd timeout"}),
    }
}

/// 在调度器中执行推理（阻塞调用，供工作线程使用）。
fn run_in_scheduler(
    sched: &Arc<Scheduler>,
    model: &str,
    args: serde_json::Value,
    _priority: u32,
    _timeout: Duration,
) -> serde_json::Value {
    // 简化：直接调用推理函数（S3 桩；S4 接入真实调度队列）
    match sched.infer_sync(model, &args, true) {
        Ok(r) => r,
        Err(e) => {
            preflight::log_failure(&format!("infer {model}"), &e);
            serde_json::json!({"error": e})
        }
    }
}

/// 便捷入口：状态查询（对应 Python `vsched.py --status`）。
pub fn status(socket: &str) -> Result<serde_json::Value, String> {
    use std::io::Write;
    use std::os::unix::net::UnixStream;
    let mut stream = UnixStream::connect(socket).map_err(|e| format!("connect: {e}"))?;
    stream
        .write_all(b"{\"stats\": true}\n")
        .map_err(|e| format!("write: {e}"))?;
    let mut buf = String::new();
    use std::io::Read;
    stream
        .read_to_string(&mut buf)
        .map_err(|e| format!("read: {e}"))?;
    serde_json::from_str(&buf).map_err(|e| format!("parse: {e}"))
}

/// 便捷入口：ping。
pub fn ping(socket: &str) -> Result<serde_json::Value, String> {
    use std::io::{Read, Write};
    use std::os::unix::net::UnixStream;
    let mut stream = UnixStream::connect(socket).map_err(|e| format!("connect: {e}"))?;
    stream
        .write_all(b"{\"ping\": true}\n")
        .map_err(|e| format!("write: {e}"))?;
    let mut buf = String::new();
    stream
        .read_to_string(&mut buf)
        .map_err(|e| format!("read: {e}"))?;
    serde_json::from_str(&buf).map_err(|e| format!("parse: {e}"))
}

// 供 run_in_scheduler 访问推理函数注册表（S3 简化；S4 重构为公开 API）
// （infer_fns 已 pub(crate)，无需额外访问器）
