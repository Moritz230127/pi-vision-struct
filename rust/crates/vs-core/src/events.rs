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
use std::collections::HashMap;
use std::io::BufRead;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// 感知缓存（G3 增量再感知）：image path → 最近 analyze 结果。
///
/// 事件触发时对缓存中的 image 重新 analyze（增量重扫），查询时返回最新。
#[derive(Clone, Default)]
pub struct PerceptionCache {
    inner: Arc<Mutex<HashMap<String, Value>>>,
}

impl PerceptionCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// 存入（image path → analyze 结果）。
    pub fn put(&self, image: &str, report: Value) {
        self.inner.lock().unwrap().insert(image.to_string(), report);
    }

    /// 读取（无缓存返回 None）。
    pub fn get(&self, image: &str) -> Option<Value> {
        self.inner.lock().unwrap().get(image).cloned()
    }

    /// 全部缓存条目数。
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().len()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.inner.lock().unwrap().is_empty()
    }

    /// 清空。
    pub fn clear(&self) {
        self.inner.lock().unwrap().clear();
    }

    /// 事件触发增量再感知：对缓存中所有 image 重新 analyze，更新缓存。
    ///
    /// 返回 (重扫数, 失败数)。analyze 失败时保留旧缓存（不覆盖）。
    pub fn reperceive(&self, analyze_fn: &dyn Fn(&str) -> Result<Value, String>) -> (usize, usize) {
        let images: Vec<String> = self.inner.lock().unwrap().keys().cloned().collect();
        let mut ok = 0;
        let mut fail = 0;
        for img in &images {
            match analyze_fn(img) {
                Ok(report) => {
                    self.put(img, report);
                    ok += 1;
                }
                Err(_) => fail += 1, // 保留旧缓存
            }
        }
        (ok, fail)
    }
}

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

    #[test]
    fn perception_cache_put_get() {
        let c = PerceptionCache::new();
        assert!(c.get("/tmp/a.png").is_none());
        c.put("/tmp/a.png", json!({"elements": []}));
        assert_eq!(c.len(), 1);
        assert!(c.get("/tmp/a.png").is_some());
        c.clear();
        assert_eq!(c.len(), 0);
    }

    #[test]
    fn perception_cache_reperceive_updates() {
        let c = PerceptionCache::new();
        c.put("/tmp/a.png", json!({"version": 1}));
        c.put("/tmp/b.png", json!({"version": 1}));
        // 再感知：a 成功（version 2），b 失败（保留旧）
        let (ok, fail) = c.reperceive(&|img| {
            if img.ends_with("a.png") {
                Ok(json!({"version": 2}))
            } else {
                Err("analyze failed".into())
            }
        });
        assert_eq!(ok, 1);
        assert_eq!(fail, 1);
        assert_eq!(c.get("/tmp/a.png").unwrap()["version"], 2);
        assert_eq!(c.get("/tmp/b.png").unwrap()["version"], 1); // 保留旧
    }
}
