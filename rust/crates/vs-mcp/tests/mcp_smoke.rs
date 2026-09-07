//! C2 门禁：vs-mcp 细粒度工具冒烟测试。
//!
//! 通过 stdio MCP 协议 spawn vs-mcp 二进制：
//!   1. initialize → OK
//!   2. tools/list → 26 个细粒度工具
//!   3. 代表性工具 tools/call 冒烟（scene_stats 真实图片）

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::time::Duration;

const EXPECTED_TOOLS: &[&str] = &[
    "capture",
    "ocr",
    "dom",
    "a11y",
    "pdf",
    "pptx",
    "blend",
    "saliency",
    "depth",
    "segment",
    "omniparser",
    "detect",
    "layout",
    "scene_stats",
    "pix",
    "edge",
    "ascii",
    "geometry",
    "fusion",
    "rules",
    "audit",
    "analyze",
    "protocol",
    "audit3d",
    "cluster",
    "wall",
    "check",
    "setup",
];

/// 启动 vs-mcp 子进程，发送 MCP 请求，收集响应。
///
/// 注意：notifications/initialized 无响应（不读）；仅带 id 的请求读响应。
fn mcp_session(requests: &[&str]) -> Vec<serde_json::Value> {
    let bin = env!("CARGO_BIN_EXE_vs-mcp");
    let mut child = Command::new(bin)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn vs-mcp");
    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();

    // 后台线程读取 stdout（非阻塞收集）
    let (tx, rx) = std::sync::mpsc::channel::<serde_json::Value>();
    std::thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(line.trim()) {
                        if tx.send(v).is_err() {
                            break;
                        }
                    }
                }
                Err(_) => break,
            }
        }
    });

    let mut responses = Vec::new();
    for req in requests {
        writeln!(stdin, "{req}").unwrap();
        stdin.flush().unwrap();
        // 仅带 id 的请求有响应
        let has_id = serde_json::from_str::<serde_json::Value>(req)
            .map(|v| v.get("id").is_some())
            .unwrap_or(false);
        if !has_id {
            continue;
        }
        // 等待响应（超时保护）
        match rx.recv_timeout(Duration::from_secs(30)) {
            Ok(v) => responses.push(v),
            Err(_) => break,
        }
    }
    let _ = child.kill();
    let _ = child.wait();
    responses
}

#[test]
fn mcp_initialize_ok() {
    let resp = mcp_session(&[
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}"#,
    ]);
    assert!(!resp.is_empty(), "应有 initialize 响应");
    assert!(resp[0]["result"].is_object(), "initialize 成功: {resp:?}");
}

#[test]
fn mcp_tools_list_26() {
    let resp = mcp_session(&[
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}"#,
        r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}"#,
    ]);
    // 找到 tools/list 响应（id=2）
    let list = resp
        .iter()
        .find(|r| r.get("id").and_then(|v| v.as_u64()) == Some(2))
        .expect("tools/list 响应");
    let tools = list["result"]["tools"].as_array().expect("tools 数组");
    let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
    // 26+ 工具（含 check/setup）
    assert!(
        names.len() >= 26,
        "工具数 ≥ 26，实际 {}: {names:?}",
        names.len()
    );
    for expected in EXPECTED_TOOLS {
        assert!(
            names.contains(expected),
            "缺少工具 {expected}，实际: {names:?}"
        );
    }
}

#[test]
fn mcp_scene_stats_call() {
    let img = "/home/Arch/Pi工作区/pi-vision-struct-v4/bench/samples/s03_alignment.png";
    if !std::path::Path::new(img).exists() {
        eprintln!("[skip] 测试图片不存在: {img}");
        return;
    }
    let req = format!(
        r#"{{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{{"name":"scene_stats","arguments":{{"image":"{img}"}}}}}}"#
    );
    let resp = mcp_session(&[
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}"#,
        r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
        &req,
    ]);
    let call = resp
        .iter()
        .find(|r| r.get("id").and_then(|v| v.as_u64()) == Some(3))
        .expect("tools/call 响应");
    let content = call["result"]["content"][0]["text"].as_str().unwrap_or("");
    let report: serde_json::Value = serde_json::from_str(content).unwrap_or_default();
    assert_eq!(report["schema"], "vision-report/v3", "schema v3");
    assert_eq!(report["task"], "scene_stats", "task");
    assert!(report["metrics"].is_object(), "metrics 存在");
}
