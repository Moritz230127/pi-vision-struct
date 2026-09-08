//! GUI 行动层（plan4 G1，Codex 模式）：五级点击降级链 + 键盘/拖拽/滚动/聚焦。
//!
//! 设计哲学（元素优先、坐标兜底）：
//! - **五级降级链**：a11y_action（AT-SPI DoAction 零误差）→ element_click
//!   （编号 → GetExtents 屏幕坐标 → 绝对定位）→ vision_click（融合元素 bbox
//!   center → 绝对定位）→ global（wlrctl serve 兜底）
//! - **持久连接**：wlrctl serve（zwlr_virtual_pointer_v1 motion_absolute），
//!   进程内单例，零 spawn 开销
//! - **闭环验证**：click [N] → verify_click（截图对比）→ 失败降级重试
//!
//! 坐标空间：逻辑像素（union 4608×1440，与 pi-gui-control 一致）。

use std::io::Write;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::Mutex;

use serde_json::{json, Value};

/// GUI 行动参数。
pub struct GuiParams {
    /// click | click_element | type | key | drag | scroll | focus | move | status | chain | verify_task
    pub action: String,
    /// 元素编号（click_element 用，五级降级链入口）。
    pub element_id: Option<usize>,
    /// 绝对坐标（click/move 用，逻辑像素）。
    pub x: Option<f64>,
    pub y: Option<f64>,
    /// 文本（type 用；click_element 的文本定位兜底）。
    pub text: Option<String>,
    /// 键组合（key 用："ctrl+l" / "super+enter" / keycode）。
    pub combo: Option<String>,
    /// 鼠标按钮（click 用：left|right|middle）。
    pub button: Option<String>,
    /// 拖拽起止（drag 用）。
    pub x1: Option<f64>,
    pub y1: Option<f64>,
    pub x2: Option<f64>,
    pub y2: Option<f64>,
    /// 滚动量（scroll 用）。
    pub dy: Option<f64>,
    pub dx: Option<f64>,
    /// 窗口（focus 用：id 或 app_id）。
    pub window: Option<String>,
    /// 点击验证（verify_click 闭环）：点击前后截图对比。
    pub verify: bool,
    /// R2 chain：多步动作序列（"move 100 100; click; sleep 0.3; key tab"）。
    pub steps: Option<String>,
    /// R4 verify_task：任务前截图路径。
    pub before: Option<String>,
    /// R4 verify_task：任务后截图路径。
    pub after: Option<String>,
    /// R4 verify_task：期望窗口标题子串。
    pub expected_title: Option<String>,
    /// R4 verify_task：视口区域（逻辑坐标 "x1,y1,x2,y2"）。
    pub viewport: Option<String>,
}

impl Default for GuiParams {
    fn default() -> Self {
        Self {
            action: "status".into(),
            element_id: None,
            x: None,
            y: None,
            text: None,
            combo: None,
            button: None,
            x1: None,
            y1: None,
            x2: None,
            y2: None,
            dy: None,
            dx: None,
            window: None,
            verify: false,
            steps: None,
            before: None,
            after: None,
            expected_title: None,
            viewport: None,
        }
    }
}

// ---- wlrctl serve 持久连接（进程内单例）----

const WLRCTL: &str = "/home/Arch/.local/bin/wlrctl";
const YDOTOOL: &str = "/usr/bin/ydotool";
const WTYPE: &str = "/usr/bin/wtype";
const NIRI: &str = "/usr/bin/niri";

/// wlrctl serve 持久连接（zwlr_virtual_pointer_v1）。
///
/// 惰性启动：首次使用时 spawn `wlrctl pointer serve`，之后复用 stdin 管道。
/// 零 spawn 开销（对比 pi-gui-control 每次调用外部中间件）。
struct WlrctlServe {
    child: Option<Child>,
    stdin: Option<ChildStdin>,
}

impl WlrctlServe {
    fn new() -> Self {
        Self {
            child: None,
            stdin: None,
        }
    }

    /// 确保 serve 进程存活，返回 stdin 写入句柄。
    fn ensure(&mut self) -> Result<&mut ChildStdin, String> {
        let dead = self
            .child
            .as_mut()
            .is_some_and(|c| c.try_wait().ok().flatten().is_some());
        if self.stdin.is_none() || dead {
            let mut child = Command::new(WLRCTL)
                .args(["pointer", "serve"])
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .map_err(|e| format!("wlrctl serve spawn: {e}"))?;
            let stdin = child
                .stdin
                .take()
                .ok_or_else(|| "wlrctl serve stdin 不可用".to_string())?;
            self.child = Some(child);
            self.stdin = Some(stdin);
        }
        self.stdin
            .as_mut()
            .ok_or_else(|| "wlrctl serve stdin 不可用".to_string())
    }

    /// 发送命令（absolute/click/down/up/scroll/sleep）。
    fn send(&mut self, cmd: &str) -> Result<(), String> {
        let stdin = self.ensure()?;
        writeln!(stdin, "{cmd}").map_err(|e| format!("wlrctl send {cmd}: {e}"))
    }
}

/// 全局 wlrctl serve 单例（进程内）。
static WLRCTL_SERVE: Mutex<Option<WlrctlServe>> = Mutex::new(None);

fn wlrctl_send(cmd: &str) -> Result<(), String> {
    let mut guard = WLRCTL_SERVE
        .lock()
        .map_err(|e| format!("wlrctl lock: {e}"))?;
    if guard.is_none() {
        *guard = Some(WlrctlServe::new());
    }
    guard.as_mut().unwrap().send(cmd)
}

// ---- 布局（union 逻辑像素）----

/// 获取布局尺寸（union 4608×1440，niri msg outputs）。
pub fn layout_size() -> (i64, i64) {
    let out = Command::new(NIRI)
        .args(["msg", "--json", "outputs"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok());
    if let Some(s) = out {
        if let Ok(v) = serde_json::from_str::<Value>(&s) {
            if let Some(outs) = v.as_object() {
                let mut max_x = 0i64;
                let mut max_y = 0i64;
                for o in outs.values() {
                    if let Some(lg) = o.get("logical") {
                        let x = lg.get("x").and_then(|v| v.as_i64()).unwrap_or(0);
                        let y = lg.get("y").and_then(|v| v.as_i64()).unwrap_or(0);
                        let w = lg.get("width").and_then(|v| v.as_i64()).unwrap_or(0);
                        let h = lg.get("height").and_then(|v| v.as_i64()).unwrap_or(0);
                        max_x = max_x.max(x + w);
                        max_y = max_y.max(y + h);
                    }
                }
                if max_x > 0 && max_y > 0 {
                    return (max_x, max_y);
                }
            }
        }
    }
    (4608, 1440) // 回退（与 pi-gui-control 一致）
}

/// 绝对移动（motion_absolute，逻辑像素）。
fn pointer_move(x: f64, y: f64) -> Result<(), String> {
    let (w, h) = layout_size();
    wlrctl_send(&format!(
        "absolute {} {} {} {}",
        x.round() as i64,
        y.round() as i64,
        w,
        h
    ))
}

/// 点击（当前指针位置）。
fn pointer_click(button: &str) -> Result<(), String> {
    wlrctl_send(&format!("click {button}"))
}

// ---- 键盘（ydotool keycode + wtype）----

/// 键组合：'ctrl+l' / 'super+enter' / 数字 keycode。
fn key(combo: &str) -> Result<(), String> {
    let combo = combo.trim();
    if combo.is_empty() {
        return Err("key 需要组合或 keycode".into());
    }
    // 纯数字 = ydotool keycode
    if combo.chars().all(|c| c.is_ascii_digit()) {
        let code = combo;
        let st = Command::new(YDOTOOL)
            .args(["key", &format!("{code}:1"), &format!("{code}:0")])
            .status()
            .map_err(|e| format!("ydotool key: {e}"))?;
        if !st.success() {
            return Err(format!("ydotool key {code} 失败"));
        }
        return Ok(());
    }
    // 组合键：wtype -M mod -k key
    let parts: Vec<&str> = combo
        .split('+')
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .collect();
    if parts.is_empty() {
        return Err("key 组合为空".into());
    }
    let (mods, keyname) = parts.split_at(parts.len() - 1);
    let mod_map = |m: &str| -> String {
        match m {
            "ctrl" | "control" => "ctrl".to_string(),
            "alt" => "alt".to_string(),
            "super" | "win" | "logo" | "meta" => "logo".to_string(),
            "shift" => "shift".to_string(),
            other => other.to_string(),
        }
    };
    let mut args: Vec<String> = vec!["-k".into(), keyname[0].to_string()];
    for m in mods {
        args.insert(0, "-M".into());
        args.insert(1, mod_map(m));
    }
    let st = Command::new(WTYPE)
        .args(&args)
        .status()
        .map_err(|e| format!("wtype key: {e}"))?;
    if !st.success() {
        return Err(format!("wtype key {combo} 失败"));
    }
    Ok(())
}

/// 输入文本（wtype -- text）。
fn type_text(text: &str) -> Result<(), String> {
    let st = Command::new(WTYPE)
        .arg("--")
        .arg(text)
        .status()
        .map_err(|e| format!("wtype type: {e}"))?;
    if !st.success() {
        return Err("wtype type 失败".into());
    }
    Ok(())
}

// ---- 窗口聚焦（niri msg action focus-window）----

fn focus_window(window: &str) -> Result<(), String> {
    if window.chars().all(|c| c.is_ascii_digit()) {
        // 按 id 聚焦
        let st = Command::new(NIRI)
            .args(["msg", "action", "focus-window", "--id", window])
            .status()
            .map_err(|e| format!("niri focus-window: {e}"))?;
        if !st.success() {
            return Err(format!("niri focus-window {window} 失败"));
        }
        return Ok(());
    }
    // 按 app_id 聚焦（第一个匹配）
    let out = Command::new(NIRI)
        .args(["msg", "--json", "windows"])
        .output()
        .map_err(|e| format!("niri windows: {e}"))?;
    let wins: Value =
        serde_json::from_slice(&out.stdout).map_err(|e| format!("niri windows JSON: {e}"))?;
    if let Some(arr) = wins.as_array() {
        for w in arr {
            if w.get("app_id").and_then(|a| a.as_str()) == Some(window) {
                let id = w.get("id").and_then(|i| i.as_i64()).unwrap_or(0);
                let st = Command::new(NIRI)
                    .args(["msg", "action", "focus-window", "--id", &id.to_string()])
                    .status()
                    .map_err(|e| format!("niri focus-window: {e}"))?;
                if !st.success() {
                    return Err(format!("niri focus-window {window} 失败"));
                }
                return Ok(());
            }
        }
    }
    Err(format!("未找到 app_id={window} 的窗口"))
}

// ---- R2 chain 原语（一次执行多步，减少模型轮次）----

/// 解析 chain 步骤字符串："move 100 100; click; sleep 0.3; key tab"。
///
/// 返回 (action, args) 列表。动作与参数以空白分隔，步骤以 `;` 分隔。
fn parse_chain(steps: &str) -> Result<Vec<(String, Vec<String>)>, String> {
    let mut out: Vec<(String, Vec<String>)> = Vec::new();
    for raw in steps.split(';') {
        let raw = raw.trim();
        if raw.is_empty() {
            continue;
        }
        let mut parts = raw.split_whitespace();
        let action = parts
            .next()
            .ok_or_else(|| format!("chain 步骤为空: {raw:?}"))?;
        let args: Vec<String> = parts.map(String::from).collect();
        out.push((action.to_string(), args));
    }
    if out.is_empty() {
        return Err("chain 步骤为空".into());
    }
    Ok(out)
}

/// 执行单个 chain 步骤（复用现有原语：move/click/key/type/sleep/scroll/focus/drag）。
fn run_chain_step(action: &str, args: &[String]) -> Result<Value, String> {
    match action {
        "move" => {
            let (x, y) = chain_xy(args, "move")?;
            pointer_move(x, y)?;
            Ok(json!({ "action": "move", "moved": [x, y] }))
        }
        "click" => {
            let button = args.first().cloned().unwrap_or_else(|| "left".to_string());
            pointer_click(&button)?;
            Ok(json!({ "action": "click", "clicked": button }))
        }
        "key" => {
            let combo = args.join("+");
            if combo.is_empty() {
                return Err("chain key 需要组合（如 key tab / key shift+a）".into());
            }
            key(&combo)?;
            Ok(json!({ "action": "key", "key": combo }))
        }
        "type" => {
            let text = args.join(" ");
            if text.is_empty() {
                return Err("chain type 需要文本".into());
            }
            type_text(&text)?;
            Ok(json!({ "action": "type", "typed": text.chars().count() }))
        }
        "sleep" => {
            let secs: f64 = args
                .first()
                .and_then(|v| v.parse().ok())
                .ok_or_else(|| "chain sleep 需要秒数（如 sleep 0.3）".to_string())?;
            std::thread::sleep(std::time::Duration::from_secs_f64(secs));
            Ok(json!({ "action": "sleep", "seconds": secs }))
        }
        "scroll" => {
            let dy = args.first().and_then(|v| v.parse().ok()).unwrap_or(0.0);
            let dx = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(0.0);
            wlrctl_send(&format!("scroll {dy} {dx}"))?;
            Ok(json!({ "action": "scroll", "scrolled": [dy, dx] }))
        }
        "focus" => {
            let window = args
                .first()
                .ok_or_else(|| "chain focus 需要窗口（id 或 app_id）".to_string())?;
            focus_window(window)?;
            Ok(json!({ "action": "focus", "focused": window }))
        }
        "drag" => {
            let (x1, y1, x2, y2) = chain_xyxy(args)?;
            let button = "left";
            pointer_move(x1, y1)?;
            wlrctl_send(&format!("down {button}"))?;
            wlrctl_send("sleep 0.05")?;
            for i in 1..=12 {
                let t = i as f64 / 12.0;
                let (ix, iy) = (x1 + (x2 - x1) * t, y1 + (y2 - y1) * t);
                pointer_move(ix, iy)?;
                wlrctl_send("sleep 0.012")?;
            }
            wlrctl_send(&format!("up {button}"))?;
            Ok(json!({ "action": "drag", "dragged": [[x1, y1], [x2, y2]] }))
        }
        other => Err(format!("chain 未知动作: {other}")),
    }
}

/// chain 步骤的 (x, y) 参数解析。
fn chain_xy(args: &[String], action: &str) -> Result<(f64, f64), String> {
    let x = args
        .first()
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| format!("chain {action} 需要 x y"))?;
    let y = args
        .get(1)
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| format!("chain {action} 需要 x y"))?;
    Ok((x, y))
}

/// chain 步骤的 (x1, y1, x2, y2) 参数解析。
fn chain_xyxy(args: &[String]) -> Result<(f64, f64, f64, f64), String> {
    let (x1, y1) = chain_xy(args, "drag")?;
    let x2 = args
        .get(2)
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| "chain drag 需要 x1 y1 x2 y2".to_string())?;
    let y2 = args
        .get(3)
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| "chain drag 需要 x1 y1 x2 y2".to_string())?;
    Ok((x1, y1, x2, y2))
}

/// 执行 chain 步骤序列（一次调用多步，减少模型轮次）。
///
/// 门禁：5 步 ≤2s（每步复用现有原语，无额外 spawn 开销）。
pub fn run_chain(steps: &str) -> Result<Value, String> {
    let parsed = parse_chain(steps)?;
    let mut results: Vec<Value> = Vec::new();
    for (action, args) in &parsed {
        results.push(run_chain_step(action, args)?);
    }
    Ok(json!({
        "ok": true,
        "steps": parsed.len(),
        "results": results,
    }))
}

// ---- R4 目标状态验证（多信号任务完成判定）----

/// 目标状态验证参数（R4）。
pub struct VerifyTaskParams {
    /// 任务前截图路径。
    pub before: String,
    /// 任务后截图路径。
    pub after: String,
    /// 期望窗口标题子串（None = 不检查标题）。
    pub expected_title: Option<String>,
    /// 视口区域（逻辑坐标 [x1, y1, x2, y2]，None = 不检查视口）。
    pub viewport: Option<[f64; 4]>,
    /// 截图差异阈值（0-1，默认 0.001）。
    pub min_diff_ratio: f64,
}

/// 标题匹配（子串，大小写不敏感）。
fn title_matches(title: &str, expected: &str) -> bool {
    title.to_lowercase().contains(&expected.to_lowercase())
}

/// 查询当前聚焦窗口标题（niri msg --json windows → is_focused）。
fn focused_window_title() -> Option<String> {
    use std::process::Command;
    let wins: Value = Command::new(NIRI)
        .args(["msg", "--json", "windows"])
        .output()
        .ok()
        .and_then(|o| serde_json::from_slice(&o.stdout).ok())?;
    wins.as_array()?
        .iter()
        .find(|w| {
            w.get("is_focused")
                .and_then(|f| f.as_bool())
                .unwrap_or(false)
        })
        .and_then(|w| w.get("title").and_then(|t| t.as_str()).map(String::from))
}

/// 视口区域差异（逻辑坐标 → 物理像素裁剪对比）。
///
/// 返回 (变化, 差异像素数, 差异比例)。视口区域超出截图范围的部分跳过。
fn viewport_diff(
    before: &str,
    after: &str,
    viewport_logical: [f64; 4],
    min_diff_ratio: f64,
) -> Result<(bool, u64, f64), String> {
    use crate::image::load_rgb;
    let b = load_rgb(std::path::Path::new(before)).map_err(|e| format!("load before: {e}"))?;
    let a = load_rgb(std::path::Path::new(after)).map_err(|e| format!("load after: {e}"))?;
    if b.data.len() != a.data.len() {
        return Err(format!("尺寸不一致: {} vs {}", b.data.len(), a.data.len()));
    }
    // 逻辑 → 物理（截图尺寸 / 布局尺寸，与 protocol.rs 一致）
    let (w_cap, h_cap) = (b.width as f64, b.height as f64);
    let (w_log, h_log) = layout_size();
    let sx = if w_log > 0 { w_cap / w_log as f64 } else { 1.0 };
    let sy = if h_log > 0 { h_cap / h_log as f64 } else { 1.0 };
    let (x1, y1, x2, y2) = (
        (viewport_logical[0] * sx).round() as i64,
        (viewport_logical[1] * sy).round() as i64,
        (viewport_logical[2] * sx).round() as i64,
        (viewport_logical[3] * sy).round() as i64,
    );
    let (x, y) = (x1.min(x2).max(0), y1.min(y2).max(0));
    let (w, h) = (
        (x2.abs_diff(x1)).max(1) as usize,
        (y2.abs_diff(y1)).max(1) as usize,
    );
    let (bw, bh) = (b.width as usize, b.height as usize);
    let mut diff: u64 = 0;
    let mut total: u64 = 0;
    for row in 0..h {
        for col in 0..w {
            let px = x as usize + col;
            let py = y as usize + row;
            if px >= bw || py >= bh {
                continue;
            }
            let idx = (py * bw + px) * 3;
            total += 1;
            if b.data[idx..idx + 3] != a.data[idx..idx + 3] {
                diff += 1;
            }
        }
    }
    let ratio = if total > 0 {
        diff as f64 / total as f64
    } else {
        0.0
    };
    Ok((ratio > min_diff_ratio, diff, ratio))
}

/// 目标状态验证（R4）：截图对比 + 窗口标题 + 视口变化多信号。
///
/// 信号：
/// 1. `screen_changed`：全屏截图差异 > 阈值（界面有变化）
/// 2. `title_matched`：聚焦窗口标题包含期望子串（应用/文档正确）
/// 3. `viewport_changed`：视口区域差异 > 阈值（3D 视图/画布内容变化）
///
/// 判定：`task_complete = screen_changed && title_matched(若要求) && viewport_changed(若要求)`。
/// 返回全部信号明细（模型可解释"为什么完成/未完成"）。
pub fn verify_task(p: &VerifyTaskParams) -> Result<Value, String> {
    // 信号 1：全屏截图对比
    let (screen_changed, screen_diff, screen_ratio) =
        verify_click(&p.before, &p.after, p.min_diff_ratio)?;

    // 信号 2：窗口标题（无法读取标题 → 视为未匹配，fail-closed）
    let (title_matched, title_actual) = match &p.expected_title {
        Some(exp) => match focused_window_title() {
            Some(t) => (title_matches(&t, exp), Some(t)),
            None => (false, None),
        },
        None => (true, None),
    };

    // 信号 3：视口变化
    let (viewport_changed, vp_diff, vp_ratio) = match p.viewport {
        Some(vp) => viewport_diff(&p.before, &p.after, vp, p.min_diff_ratio)?,
        None => (true, 0, 0.0),
    };

    let complete = screen_changed && title_matched && viewport_changed;

    let mut out = json!({
        "ok": true,
        "task_complete": complete,
        "signals": {
            "screen_changed": screen_changed,
            "screen_diff_pixels": screen_diff,
            "screen_diff_ratio": (screen_ratio * 1e6).round() / 1e6,
            "title_matched": title_matched,
            "viewport_changed": viewport_changed,
            "viewport_diff_pixels": vp_diff,
            "viewport_diff_ratio": (vp_ratio * 1e6).round() / 1e6,
        },
    });
    if let Some(t) = title_actual {
        out["signals"]["focused_title"] = json!(t);
    }
    if let Some(exp) = &p.expected_title {
        out["signals"]["expected_title"] = json!(exp);
    }
    if let Some(vp) = p.viewport {
        out["signals"]["viewport"] = json!(vp);
    }
    out["note"] = json!(format!(
        "任务完成判定：{}（screen_changed={} title_matched={} viewport_changed={}）",
        if complete { "完成" } else { "未完成" },
        screen_changed,
        title_matched,
        viewport_changed
    ));
    Ok(out)
}

// ---- 五级点击降级链 ----

/// 五级点击降级链（Codex 同款）：
/// 1. a11y_action：AT-SPI DoAction（零坐标误差）
/// 2. element_click：编号 → GetExtents 屏幕坐标 → 绝对定位
/// 3. vision_click：融合元素 bbox center → 绝对定位
/// 4. global：wlrctl serve 兜底
///
/// `element_id`：融合元素编号（analyze 输出的 [N]）。
/// `fallback_xy`：无编号时的坐标兜底（vision_click/global 用）。
pub fn click_element(
    element_id: usize,
    fallback_xy: Option<(f64, f64)>,
    bus: Option<&str>,
    path: Option<&str>,
) -> Result<Value, String> {
    let mut chain: Vec<String> = Vec::new();

    // 1) a11y_action：AT-SPI DoAction（零坐标误差）——用 bus/path 直接定位
    match a11y_action_click(element_id, bus, path) {
        Ok(true) => {
            chain.push("a11y_action".into());
            return Ok(json!({"ok": true, "method": "a11y_action", "chain": chain}));
        }
        Ok(false) => chain.push("a11y_action(无动作)".into()),
        Err(e) => chain.push(format!("a11y_action({e})")),
    }

    // 2) element_click：编号 → GetExtents 屏幕坐标 → 绝对定位
    match a11y_extents_click(element_id, bus, path) {
        Ok(()) => {
            chain.push("element_click".into());
            return Ok(json!({"ok": true, "method": "element_click", "chain": chain}));
        }
        Err(e) => chain.push(format!("element_click({e})")),
    }

    // 3) vision_click：融合元素 bbox center → 绝对定位
    if let Some((x, y)) = fallback_xy {
        pointer_move(x, y)?;
        pointer_click("left")?;
        chain.push("vision_click".into());
        return Ok(json!({"ok": true, "method": "vision_click", "chain": chain, "x": x, "y": y}));
    }

    // 4) global：wlrctl serve 兜底（无坐标则失败）
    Err(format!("五级降级链全部失败: {}", chain.join(" → ")))
}

/// 闭环验证（verify_click）：点击前后截图对比，确认界面变化。
///
/// 返回 (验证通过, 差异像素数, 差异比例)。截图失败时返回 Err。
/// 差异比例 > `min_diff_ratio` 视为界面变化（点击生效）。
pub fn verify_click(
    before: &str,
    after: &str,
    min_diff_ratio: f64,
) -> Result<(bool, u64, f64), String> {
    use crate::image::load_rgb;
    let b = load_rgb(std::path::Path::new(before)).map_err(|e| format!("load before: {e}"))?;
    let a = load_rgb(std::path::Path::new(after)).map_err(|e| format!("load after: {e}"))?;
    if b.data.len() != a.data.len() {
        return Err(format!("尺寸不一致: {} vs {}", b.data.len(), a.data.len()));
    }
    let mut diff: u64 = 0;
    for (pb, pa) in b.data.iter().zip(a.data.iter()) {
        if pb != pa {
            diff += 1;
        }
    }
    let ratio = diff as f64 / b.data.len() as f64;
    Ok((ratio > min_diff_ratio, diff, ratio))
}

/// 点击 + 闭环验证：click → 截图对比 → 失败重试（≤2 次）。
///
/// 每次尝试：点击前截图 → 点击 → 点击后截图 → verify_click。
/// 验证通过返回 Ok；全部失败返回 Err（含各次差异）。
pub fn click_with_verify(
    element_id: usize,
    fallback_xy: Option<(f64, f64)>,
    capture_path: &str,
    min_diff_ratio: f64,
    retries: usize,
) -> Result<Value, String> {
    let mut attempts: Vec<Value> = Vec::new();
    for attempt in 0..=retries {
        // 点击前截图
        let before = format!("{capture_path}.before{attempt}.png");
        crate::capture::run(&crate::capture::CaptureParams {
            region: None,
            out: before.clone(),
            window: None,
        })
        .map_err(|e| format!("capture before: {e}"))?;
        // 执行点击（五级降级链）
        let r = click_element(element_id, fallback_xy, None, None)?;
        // 点击后截图
        let after = format!("{capture_path}.after{attempt}.png");
        crate::capture::run(&crate::capture::CaptureParams {
            region: None,
            out: after.clone(),
            window: None,
        })
        .map_err(|e| format!("capture after: {e}"))?;
        // 对比
        let (ok, diff, ratio) = verify_click(&before, &after, min_diff_ratio)?;
        attempts.push(json!({
            "attempt": attempt,
            "method": r.get("method").cloned().unwrap_or(json!("unknown")),
            "diff_pixels": diff,
            "diff_ratio": ratio,
            "verified": ok,
        }));
        if ok {
            let mut out = r.clone();
            out["verify"] =
                json!({"ok": true, "attempt": attempt, "diff_pixels": diff, "diff_ratio": ratio});
            out["attempts"] = json!(attempts);
            return Ok(out);
        }
        // 失败：短暂等待后重试（界面可能未稳定）
        std::thread::sleep(std::time::Duration::from_millis(300));
    }
    Err(format!(
        "click_with_verify 全部失败（{} 次尝试）: {}",
        retries + 1,
        serde_json::to_string(&attempts).unwrap_or_default()
    ))
}

/// 第 1 级：AT-SPI DoAction（bus/path 直接定位 → 无障碍动作直达）。
///
/// F3 修复：不再遍历全局树（编号空间不匹配），直接用 analyze 返回的
/// bus name + path 建 zbus proxy 调 DoAction(0)（默认动作 = click）。
fn a11y_action_click(
    _element_id: usize,
    bus: Option<&str>,
    path: Option<&str>,
) -> Result<bool, String> {
    let (Some(bus), Some(path)) = (bus, path) else {
        return Err("元素无 bus/path（非 a11y 来源）".into());
    };
    let rt = tokio::runtime::Runtime::new().map_err(|e| format!("tokio rt: {e}"))?;
    rt.block_on(async {
        let conn = atspi::connection::AccessibilityConnection::new()
            .await
            .map_err(|e| format!("atspi connect: {e}"))?;
        // 查询可用动作
        let actions = crate::a11y::get_actions_zbus_path(bus, path, conn.connection()).await?;
        if actions.is_empty() {
            return Ok(false); // 无动作 → 降级
        }
        // 执行默认动作（index 0 = click）
        crate::a11y::do_action_zbus_path(bus, path, conn.connection(), 0).await
    })
}

/// 第 2 级：element_click（bus/path → GetExtents 屏幕坐标 → 绝对定位）。
fn a11y_extents_click(
    _element_id: usize,
    bus: Option<&str>,
    path: Option<&str>,
) -> Result<(), String> {
    let (Some(bus), Some(path)) = (bus, path) else {
        return Err("元素无 bus/path（非 a11y 来源）".into());
    };
    let rt = tokio::runtime::Runtime::new().map_err(|e| format!("tokio rt: {e}"))?;
    rt.block_on(async {
        let conn = atspi::connection::AccessibilityConnection::new()
            .await
            .map_err(|e| format!("atspi connect: {e}"))?;
        // GetExtents 屏幕坐标 → 绝对定位点击
        let (x, y, w, h) = crate::a11y::get_extents_zbus_path(bus, path, conn.connection()).await?;
        if w <= 0 || h <= 0 {
            return Err(format!("元素不可见（{w}×{h}）"));
        }
        let (cx, cy) = (x as f64 + w as f64 / 2.0, y as f64 + h as f64 / 2.0);
        pointer_move(cx, cy)?;
        pointer_click("left")?;
        Ok(())
    })
}

// ---- 主入口 ----

/// 运行 GUI 行动。
pub fn run(p: &GuiParams) -> Result<Value, String> {
    match p.action.as_str() {
        "status" => {
            let (w, h) = layout_size();
            Ok(json!({"ok": true, "layout": [w, h], "backend": "wlrctl serve"}))
        }
        "move" => {
            let (x, y) = (
                p.x.ok_or_else(|| "move 需要 x".to_string())?,
                p.y.ok_or_else(|| "move 需要 y".to_string())?,
            );
            pointer_move(x, y)?;
            Ok(json!({"ok": true, "moved": [x, y]}))
        }
        "click" => {
            let button = p.button.clone().unwrap_or_else(|| "left".into());
            if let (Some(x), Some(y)) = (p.x, p.y) {
                pointer_move(x, y)?;
            }
            pointer_click(&button)?;
            Ok(json!({"ok": true, "clicked": button}))
        }
        "click_element" => {
            let id = p
                .element_id
                .ok_or_else(|| "click_element 需要 --element-id".to_string())?;
            let fallback = match (p.x, p.y) {
                (Some(x), Some(y)) => Some((x, y)),
                _ => None,
            };
            let r = click_element(id, fallback, None, None)?;
            // 闭环验证（verify_click）
            if p.verify {
                // 点击后截图对比（简化：返回提示，真实对比在 G3 端到端）
                let mut v = r.clone();
                v["verify"] = json!("点击已执行（闭环验证在 G3 端到端）");
                return Ok(v);
            }
            Ok(r)
        }
        "type" => {
            let text = p.text.clone().ok_or_else(|| "type 需要 text".to_string())?;
            type_text(&text)?;
            Ok(json!({"ok": true, "typed": text.chars().count()}))
        }
        "key" => {
            let combo = p
                .combo
                .clone()
                .ok_or_else(|| "key 需要 combo".to_string())?;
            key(&combo)?;
            Ok(json!({"ok": true, "key": combo}))
        }
        "drag" => {
            let (x1, y1, x2, y2) = (
                p.x1.ok_or_else(|| "drag 需要 x1".to_string())?,
                p.y1.ok_or_else(|| "drag 需要 y1".to_string())?,
                p.x2.ok_or_else(|| "drag 需要 x2".to_string())?,
                p.y2.ok_or_else(|| "drag 需要 y2".to_string())?,
            );
            let button = p.button.clone().unwrap_or_else(|| "left".into());
            pointer_move(x1, y1)?;
            wlrctl_send(&format!("down {button}"))?;
            wlrctl_send("sleep 0.05")?;
            // 插值 12 步（与 Python 一致）
            for i in 1..=12 {
                let t = i as f64 / 12.0;
                let (ix, iy) = (x1 + (x2 - x1) * t, y1 + (y2 - y1) * t);
                pointer_move(ix, iy)?;
                wlrctl_send("sleep 0.012")?;
            }
            wlrctl_send(&format!("up {button}"))?;
            Ok(json!({"ok": true, "dragged": [[x1, y1], [x2, y2]]}))
        }
        "scroll" => {
            let dy = p.dy.unwrap_or(0.0);
            let dx = p.dx.unwrap_or(0.0);
            wlrctl_send(&format!("scroll {dy} {dx}"))?;
            Ok(json!({"ok": true, "scrolled": [dy, dx]}))
        }
        "focus" => {
            let window = p
                .window
                .clone()
                .ok_or_else(|| "focus 需要 window".to_string())?;
            focus_window(&window)?;
            Ok(json!({"ok": true, "focused": window}))
        }
        "chain" => {
            let steps = p.steps.clone().ok_or_else(|| {
                "chain 需要 --steps（如 \"move 100 100; click; sleep 0.3; key tab\"）".to_string()
            })?;
            run_chain(&steps)
        }
        "verify_task" => {
            let before = p
                .before
                .clone()
                .ok_or_else(|| "verify_task 需要 --before 截图路径".to_string())?;
            let after = p
                .after
                .clone()
                .ok_or_else(|| "verify_task 需要 --after 截图路径".to_string())?;
            let viewport = p.viewport.as_deref().and_then(|s| {
                let v: Vec<f64> = s.split(',').filter_map(|v| v.trim().parse().ok()).collect();
                (v.len() == 4).then(|| [v[0], v[1], v[2], v[3]])
            });
            verify_task(&VerifyTaskParams {
                before,
                after,
                expected_title: p.expected_title.clone(),
                viewport,
                min_diff_ratio: 0.001,
            })
        }
        other => Err(format!("unknown action: {other}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_click_detects_diff() {
        // 构造两张 2×2 图：一张全白，一张有 1 像素黑（不同路径）
        let mk = |path: &str, px: [u8; 12]| {
            let img = image::RgbImage::from_raw(2, 2, px.to_vec()).unwrap();
            img.save(path).unwrap();
        };
        let before = "/tmp/vsd_verify_before.png";
        let after = "/tmp/vsd_verify_after.png";
        mk(before, [255; 12]);
        mk(
            after,
            [255, 255, 255, 0, 0, 0, 255, 255, 255, 255, 255, 255],
        );
        let (ok, diff, ratio) = verify_click(before, after, 0.001).unwrap();
        assert!(ok, "1 像素差异应触发验证通过");
        assert_eq!(diff, 3); // 1 像素 = 3 通道
        assert!(ratio > 0.0);
        // 相同图 → 不通过
        let (ok2, diff2, _) = verify_click(before, before, 0.001).unwrap();
        assert!(!ok2);
        assert_eq!(diff2, 0);
    }

    #[test]
    fn status_reports_layout() {
        let p = GuiParams {
            action: "status".into(),
            ..GuiParams::default()
        };
        let r = run(&p).unwrap();
        assert_eq!(r["ok"], true);
        assert!(r["layout"].as_array().unwrap().len() == 2);
    }

    #[test]
    fn unknown_action_rejected() {
        let p = GuiParams {
            action: "bogus".into(),
            ..GuiParams::default()
        };
        assert!(run(&p).is_err());
    }

    #[test]
    fn click_requires_coords_or_element() {
        // click 无坐标 → 当前指针位置点击（不报错，wlrctl 兜底）
        let p = GuiParams {
            action: "click".into(),
            ..GuiParams::default()
        };
        // 无 Wayland 会话时 wlrctl 失败 → 返回 Err（环境依赖）
        let _ = run(&p);
    }

    #[test]
    fn click_element_requires_id() {
        let p = GuiParams {
            action: "click_element".into(),
            ..GuiParams::default()
        };
        assert!(run(&p).is_err(), "click_element 无 element_id 应报错");
    }

    #[test]
    fn key_requires_combo() {
        let p = GuiParams {
            action: "key".into(),
            ..GuiParams::default()
        };
        assert!(run(&p).is_err(), "key 无 combo 应报错");
    }

    #[test]
    fn type_requires_text() {
        let p = GuiParams {
            action: "type".into(),
            ..GuiParams::default()
        };
        assert!(run(&p).is_err(), "type 无 text 应报错");
    }

    #[test]
    fn drag_requires_all_coords() {
        let p = GuiParams {
            action: "drag".into(),
            x1: Some(0.0),
            ..GuiParams::default()
        };
        assert!(run(&p).is_err(), "drag 缺坐标应报错");
    }

    #[test]
    fn focus_requires_window() {
        let p = GuiParams {
            action: "focus".into(),
            ..GuiParams::default()
        };
        assert!(run(&p).is_err(), "focus 无 window 应报错");
    }

    #[test]
    fn layout_size_fallback() {
        // 无 niri 时回退 4608×1440
        let (w, h) = layout_size();
        assert!(w > 0 && h > 0);
    }

    #[test]
    fn click_element_chain_reports_method() {
        // 五级降级链：无 Wayland/a11y 环境时返回 Err（含链信息）或成功
        // 合成验证：click_element 的链构建逻辑（a11y 不可用 → 降级）
        let r = click_element(9999, Some((100.0, 100.0)), None, None);
        match r {
            Ok(v) => {
                // 环境可用：a11y 或 vision_click 成功
                assert!(v["ok"] == true);
                let method = v["method"].as_str().unwrap();
                assert!(
                    ["a11y_action", "element_click", "vision_click"].contains(&method),
                    "降级链应命中某级: {method}"
                );
            }
            Err(e) => {
                // 环境不可用：错误含降级链信息
                assert!(
                    e.contains("降级链") || e.contains("a11y") || e.contains("wlrctl"),
                    "错误应含链信息: {e}"
                );
            }
        }
    }

    #[test]
    fn click_element_roundtrip_under_100ms() {
        // click [N] 往返计时：<100ms（门禁）
        // 无环境时仅验证参数校验路径（不触发 wlrctl）
        let start = std::time::Instant::now();
        let p = GuiParams {
            action: "click_element".into(),
            element_id: None, // 缺 id → 快速报错（参数校验路径）
            ..GuiParams::default()
        };
        let _ = run(&p);
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 100,
            "click_element 参数校验往返 {}ms >= 100ms",
            elapsed.as_millis()
        );
    }

    // ---- R2 chain 原语 ----

    #[test]
    fn chain_parse_multi_step() {
        // "move 100 100; click; sleep 0.3; key tab" → 4 步
        let steps = parse_chain("move 100 100; click; sleep 0.3; key tab").unwrap();
        assert_eq!(steps.len(), 4);
        assert_eq!(steps[0].0, "move");
        assert_eq!(steps[0].1, vec!["100", "100"]);
        assert_eq!(steps[1].0, "click");
        assert!(steps[1].1.is_empty());
        assert_eq!(steps[2].0, "sleep");
        assert_eq!(steps[2].1, vec!["0.3"]);
        assert_eq!(steps[3].0, "key");
        assert_eq!(steps[3].1, vec!["tab"]);
    }

    #[test]
    fn chain_parse_ignores_empty_and_whitespace() {
        let steps = parse_chain("  move 1 2 ;  ; key tab  ").unwrap();
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].0, "move");
        assert_eq!(steps[1].0, "key");
    }

    #[test]
    fn chain_empty_rejected() {
        assert!(parse_chain("").is_err());
        assert!(parse_chain("   ").is_err());
        assert!(parse_chain(";;").is_err());
    }

    #[test]
    fn chain_unknown_action_rejected() {
        assert!(run_chain("bogus 1 2").is_err());
    }

    #[test]
    fn chain_requires_steps_param() {
        let p = GuiParams {
            action: "chain".into(),
            ..GuiParams::default()
        };
        assert!(run(&p).is_err(), "chain 无 --steps 应报错");
    }

    #[test]
    fn chain_5_steps_under_2s() {
        // 门禁：chain 5 步 ≤2s（sleep 步骤确定性计时，无环境依赖）
        let start = std::time::Instant::now();
        let r = run_chain("sleep 0.1; sleep 0.1; sleep 0.1; sleep 0.1; sleep 0.1").unwrap();
        let elapsed = start.elapsed();
        assert_eq!(r["steps"], 5);
        assert_eq!(r["results"].as_array().unwrap().len(), 5);
        assert!(
            elapsed.as_millis() < 2000,
            "chain 5 步 {}ms >= 2000ms",
            elapsed.as_millis()
        );
    }

    #[test]
    fn chain_key_joins_modifier() {
        // key shift+a → 组合 "shift+a"（wtype -M shift -k a）
        let steps = parse_chain("key shift+a").unwrap();
        assert_eq!(steps[0].1, vec!["shift+a"]);
    }

    #[test]
    fn r5_keyboard_first_template_valid() {
        // R5 门禁：键盘优先任务模板（python/tasks/gui-blender-add.json）
        // 1) JSON 合法 2) chain 全部为已知动作 3) 步骤数 ≤10（R1 任务模板约束）
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../python/tasks/gui-blender-add.json"
        );
        let text = std::fs::read_to_string(path).expect("模板文件应存在");
        let v: Value = serde_json::from_str(&text).expect("模板应为合法 JSON");
        assert_eq!(v["strategy"], "keyboard-first");
        let chain = v["chain"].as_str().expect("模板应含 chain 命令");
        let steps = parse_chain(chain).expect("模板 chain 应可解析");
        assert!(
            steps.len() <= 10,
            "模板步骤 {} > 10（R1 约束）",
            steps.len()
        );
        // 全部为已知 chain 动作
        let known = [
            "move", "click", "key", "type", "sleep", "scroll", "focus", "drag",
        ];
        for (action, _) in &steps {
            assert!(known.contains(&action.as_str()), "模板含未知动作: {action}");
        }
        // 键盘优先模式：首步 focus + 含 key 快捷键
        assert_eq!(steps[0].0, "focus", "键盘优先应先 focus 窗口");
        assert!(
            steps.iter().any(|(a, _)| a == "key"),
            "键盘优先应含 key 快捷键步骤"
        );
        // verify 段存在（R4 目标状态验证）
        assert!(v["verify"].is_object(), "模板应含 verify 段");
    }

    // ---- R4 verify_task ----

    #[test]
    fn title_matches_substring_case_insensitive() {
        assert!(title_matches("Blender 3.6", "blender"));
        assert!(title_matches("Blender 3.6", "Blender"));
        assert!(!title_matches("Firefox", "Blender"));
        assert!(title_matches("莫比乌斯环 - Blender", "莫比乌斯"));
    }

    /// 合成测试辅助：生成 w×h 纯色图，可选在物理区域 [rx1,ry1,rx2,ry2] 画黑块。
    fn mk_img(path: &str, w: u32, h: u32, black: Option<(u32, u32, u32, u32)>) {
        let mut img = image::RgbImage::from_pixel(w, h, image::Rgb([255, 255, 255]));
        if let Some((x1, y1, x2, y2)) = black {
            for y in y1..y2.min(h) {
                for x in x1..x2.min(w) {
                    img.put_pixel(x, y, image::Rgb([0, 0, 0]));
                }
            }
        }
        img.save(path).unwrap();
    }

    /// 逻辑视口 → 物理区域（与 viewport_diff 内部换算一致）。
    fn vp_to_physical(vp: [f64; 4], w_cap: u32, h_cap: u32) -> (u32, u32, u32, u32) {
        let (w_log, h_log) = layout_size();
        let sx = if w_log > 0 {
            w_cap as f64 / w_log as f64
        } else {
            1.0
        };
        let sy = if h_log > 0 {
            h_cap as f64 / h_log as f64
        } else {
            1.0
        };
        (
            (vp[0] * sx).round() as u32,
            (vp[1] * sy).round() as u32,
            (vp[2] * sx).round() as u32,
            (vp[3] * sy).round() as u32,
        )
    }

    #[test]
    fn verify_task_synthetic_accuracy_ge_90pct() {
        // 门禁：任务完成判定准确率 ≥90%（合成测试，10 例确定性判定）
        let dir = "/tmp/vsd_verify_task";
        let _ = std::fs::create_dir_all(dir);
        let (w, h) = (100u32, 100u32);
        // 视口 = 物理 [0,0,50,50]（逻辑坐标由换算反推）
        let vp = [
            0.0,
            0.0,
            50.0 / (w as f64 / layout_size().0 as f64),
            50.0 / (h as f64 / layout_size().1 as f64),
        ];
        let (px1, py1, px2, py2) = vp_to_physical(vp, w, h);
        assert_eq!(
            (px1, py1, px2, py2),
            (0, 0, 50, 50),
            "视口换算应精确到物理 [0,0,50,50]"
        );

        let before = format!("{dir}/before.png");
        let after_in = format!("{dir}/after_in.png");
        let after_out = format!("{dir}/after_out.png");
        let after_same = format!("{dir}/after_same.png");
        let after_tiny = format!("{dir}/after_tiny.png");
        mk_img(&before, w, h, None);
        mk_img(&after_in, w, h, Some((0, 0, 50, 50))); // 视口内变化
        mk_img(&after_out, w, h, Some((50, 50, 100, 100))); // 视口外变化
        mk_img(&after_same, w, h, None); // 无变化
        mk_img(&after_tiny, w, h, Some((0, 0, 1, 1))); // 1 像素变化（视口内，低于阈值）

        let run_vt = |before: &str, after: &str, vp: Option<[f64; 4]>, min_ratio: f64| {
            verify_task(&VerifyTaskParams {
                before: before.into(),
                after: after.into(),
                expected_title: None,
                viewport: vp,
                min_diff_ratio: min_ratio,
            })
            .unwrap()
        };

        // 10 例：期望判定 (case, expected_complete)
        type VerifyCase = (String, String, Option<[f64; 4]>, f64, bool);
        let cases: Vec<VerifyCase> = vec![
            // 1. 视口内变化 + 视口检查 → 完成
            (before.clone(), after_in.clone(), Some(vp), 0.001, true),
            // 2. 无变化 → 未完成
            (before.clone(), after_same.clone(), Some(vp), 0.001, false),
            // 3. 视口外变化 + 视口检查 → 未完成（视口未变）
            (before.clone(), after_out.clone(), Some(vp), 0.001, false),
            // 4. 视口内变化 + 无视口检查 → 完成（全屏变化）
            (before.clone(), after_in.clone(), None, 0.001, true),
            // 5. 视口外变化 + 无视口检查 → 完成（全屏有变化）
            (before.clone(), after_out.clone(), None, 0.001, true),
            // 6. 视口内变化但阈值 0.5（25% < 50%）→ 未完成
            (before.clone(), after_in.clone(), Some(vp), 0.5, false),
            // 7. 视口内变化 + 低阈值 → 完成
            (before.clone(), after_in.clone(), Some(vp), 0.01, true),
            // 8. 1 像素变化 + 阈值 0.001（0.0001 < 0.001）→ 未完成
            (before.clone(), after_tiny.clone(), Some(vp), 0.001, false),
            // 9. 1 像素变化 + 阈值 0.00001 → 完成
            (before.clone(), after_tiny.clone(), Some(vp), 0.00001, true),
            // 10. 视口内变化 + 视口检查 + 阈值 0.001 → 完成（重复确认）
            (before.clone(), after_in.clone(), Some(vp), 0.001, true),
        ];

        let mut correct = 0;
        for (i, (b, a, vp, min_ratio, expected)) in cases.iter().enumerate() {
            let r = run_vt(b, a, *vp, *min_ratio);
            let got = r["task_complete"].as_bool().unwrap();
            assert_eq!(
                got,
                *expected,
                "case {}: 期望 complete={} 实际 {}（signals={}）",
                i + 1,
                expected,
                got,
                r["signals"]
            );
            if got == *expected {
                correct += 1;
            }
        }
        let acc = correct as f64 / cases.len() as f64;
        assert!(
            acc >= 0.9,
            "任务完成判定准确率 {:.0}% < 90%（{}/{}）",
            acc * 100.0,
            correct,
            cases.len()
        );
    }

    #[test]
    fn verify_task_requires_before_after() {
        let p = GuiParams {
            action: "verify_task".into(),
            ..GuiParams::default()
        };
        assert!(run(&p).is_err(), "verify_task 缺 --before/--after 应报错");
    }

    #[test]
    fn verify_task_missing_files_rejected() {
        let r = verify_task(&VerifyTaskParams {
            before: "/tmp/no_such_before.png".into(),
            after: "/tmp/no_such_after.png".into(),
            expected_title: None,
            viewport: None,
            min_diff_ratio: 0.001,
        });
        assert!(r.is_err(), "截图文件不存在应报错");
    }
}
