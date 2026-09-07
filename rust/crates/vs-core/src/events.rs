//! 事件驱动感知（plan4 G2）：监听 niri event-stream → 变化即感知。
//!
//! niri `msg event-stream` 输出 Rust Debug 文本行（非 JSON），格式：
//!   Workspaces changed: [Workspace { id: 1, ... }]
//!   Windows changed: [Window { id: 93, title: Some("..."), ... }]
//!   Keyboard layouts changed: ...
//!   Overview toggled: false
//!
//! 本模块按行前缀解析事件类型（无需完整解析 Debug 文本），节流合并后
//! 输出事件序列。vsd daemon 集成：事件 → 触发增量再感知 → 缓存更新。

use serde_json::{json, Value};
use std::io::BufRead;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{Duration, Instant};

/// 事件类型（niri event-stream 行前缀）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NiriEvent {
    WorkspacesChanged,
    WindowsChanged,
    WindowOpenedOrChanged,
    WindowClosed,
    FocusChanged,
    KeyboardLayoutsChanged,
    OverviewToggled,
    Other(String),
}

impl NiriEvent {
    /// 从 event-stream 行解析事件类型（前缀匹配）。
    pub fn from_line(line: &str) -> NiriEvent {
        let l = line.trim();
        if l.starts_with("Workspaces changed") {
            NiriEvent::WorkspacesChanged
        } else if l.starts_with("Windows changed") {
            NiriEvent::WindowsChanged
        } else if l.starts_with("Window opened or changed") {
            NiriEvent::WindowOpenedOrChanged
        } else if l.starts_with("Window closed") {
            NiriEvent::WindowClosed
        } else if l.starts_with("Window focused") || l.starts_with("Focus changed") {
            NiriEvent::FocusChanged
        } else if l.starts_with("Keyboard layouts changed") {
            NiriEvent::KeyboardLayoutsChanged
        } else if l.starts_with("Overview toggled") {
            NiriEvent::OverviewToggled
        } else if !l.is_empty() && !l.starts_with("Started reading") {
            NiriEvent::Other(l.chars().take(80).collect())
        } else {
            NiriEvent::Other(String::new())
        }
    }

    /// 该事件是否需要触发再感知（窗口/焦点变化）。
    pub fn triggers_reperception(&self) -> bool {
        matches!(
            self,
            NiriEvent::WindowsChanged
                | NiriEvent::WindowOpenedOrChanged
                | NiriEvent::WindowClosed
                | NiriEvent::FocusChanged
        )
    }
}

/// 事件监听器：spawn niri event-stream 子进程 + 行解析 + 节流合并。
pub struct EventWatcher {
    child: Child,
    rx: Receiver<NiriEvent>,
    throttle: Duration,
    last_emit: Option<Instant>,
}

impl EventWatcher {
    /// 启动监听（spawn `niri msg event-stream`）。
    pub fn start(throttle_ms: u64) -> Result<Self, String> {
        let mut child = Command::new("niri")
            .args(["msg", "event-stream"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("spawn niri event-stream: {e}"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "niri event-stream stdout 不可用".to_string())?;
        let (tx, rx): (Sender<NiriEvent>, Receiver<NiriEvent>) = mpsc::channel();
        // 读线程：逐行解析 → 发送
        std::thread::spawn(move || {
            let reader = std::io::BufReader::new(stdout);
            for line in reader.lines() {
                let Ok(line) = line else { break };
                let ev = NiriEvent::from_line(&line);
                if tx.send(ev).is_err() {
                    break;
                }
            }
        });
        Ok(Self {
            child,
            rx,
            throttle: Duration::from_millis(throttle_ms),
            last_emit: None,
        })
    }

    /// 阻塞等待下一个事件（节流合并：throttle 窗口内的事件合并为 1 个）。
    pub fn next_event(&mut self) -> Option<NiriEvent> {
        // 等待第一个事件
        let first = self.rx.recv().ok()?;
        let mut merged = first;
        let window_start = Instant::now();
        // 节流窗口内继续收，合并
        while window_start.elapsed() < self.throttle {
            match self.rx.recv_timeout(self.throttle - window_start.elapsed()) {
                Ok(ev) => {
                    if ev.triggers_reperception() {
                        merged = ev;
                    }
                }
                Err(_) => break,
            }
        }
        self.last_emit = Some(Instant::now());
        Some(merged)
    }

    /// 非阻塞尝试接收（无事件返回 None）。
    pub fn try_next(&mut self) -> Option<NiriEvent> {
        self.rx.try_recv().ok()
    }

    /// 停止监听。
    pub fn stop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for EventWatcher {
    fn drop(&mut self) {
        self.stop();
    }
}

/// 事件驱动感知：监听 → 触发再感知回调 → 缓存更新。
///
/// `on_event` 回调在每次节流合并后的事件上调用（返回是否触发再感知）。
pub fn watch_loop(
    throttle_ms: u64,
    on_event: impl FnMut(NiriEvent) -> bool,
    max_events: Option<u64>,
) -> Result<Value, String> {
    let mut watcher = EventWatcher::start(throttle_ms)?;
    let mut seen: Vec<Value> = Vec::new();
    let mut count: u64 = 0;
    let mut on_event = on_event;
    while let Some(ev) = watcher.next_event() {
        count += 1;
        let trigger = on_event(ev.clone());
        seen.push(json!({
            "event": format!("{ev:?}"),
            "triggers_reperception": ev.triggers_reperception(),
            "reperceived": trigger,
        }));
        if let Some(max) = max_events {
            if count >= max {
                break;
            }
        }
    }
    Ok(json!({
        "events_seen": count,
        "events": seen,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_event_types() {
        assert_eq!(
            NiriEvent::from_line("Workspaces changed: [Workspace { id: 1 }]"),
            NiriEvent::WorkspacesChanged
        );
        assert_eq!(
            NiriEvent::from_line("Windows changed: [Window { id: 93 }]"),
            NiriEvent::WindowsChanged
        );
        assert_eq!(
            NiriEvent::from_line("Window opened or changed: Window { id: 1 }"),
            NiriEvent::WindowOpenedOrChanged
        );
        assert_eq!(
            NiriEvent::from_line("Window closed: Window { id: 1 }"),
            NiriEvent::WindowClosed
        );
        assert_eq!(
            NiriEvent::from_line("Keyboard layouts changed: KeyboardLayouts { names: [] }"),
            NiriEvent::KeyboardLayoutsChanged
        );
        assert_eq!(
            NiriEvent::from_line("Overview toggled: false"),
            NiriEvent::OverviewToggled
        );
        assert_eq!(
            NiriEvent::from_line("Started reading events."),
            NiriEvent::Other(String::new())
        );
    }

    #[test]
    fn triggers_reperception_only_for_window_changes() {
        assert!(NiriEvent::WindowsChanged.triggers_reperception());
        assert!(NiriEvent::WindowOpenedOrChanged.triggers_reperception());
        assert!(NiriEvent::WindowClosed.triggers_reperception());
        assert!(NiriEvent::FocusChanged.triggers_reperception());
        assert!(!NiriEvent::WorkspacesChanged.triggers_reperception());
        assert!(!NiriEvent::KeyboardLayoutsChanged.triggers_reperception());
        assert!(!NiriEvent::OverviewToggled.triggers_reperception());
    }

    #[test]
    fn throttle_merges_events() {
        // 节流逻辑：构造 3 个事件，throttle 窗口内合并
        let (tx, rx): (Sender<NiriEvent>, Receiver<NiriEvent>) = mpsc::channel();
        tx.send(NiriEvent::WindowsChanged).unwrap();
        tx.send(NiriEvent::FocusChanged).unwrap();
        tx.send(NiriEvent::WindowsChanged).unwrap();
        drop(tx);
        // 模拟 EventWatcher 的合并逻辑（直接测 rx 消费）
        let first = rx.recv().unwrap();
        let mut merged = first;
        while let Ok(ev) = rx.try_recv() {
            if ev.triggers_reperception() {
                merged = ev;
            }
        }
        // 合并后应保留最后一个触发再感知的事件
        assert_eq!(merged, NiriEvent::WindowsChanged);
    }
}
