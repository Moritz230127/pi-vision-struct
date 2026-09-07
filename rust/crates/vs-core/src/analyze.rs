//! 任务引擎（对应 Python `vs_analyze.py`，纯 Rust，schema v3 输出）。
//!
//! 任务 = 配置，新任务 = 新配置（零新代码）。配置位于 tasks/<name>.json：
//!
//! ```json
//! {
//!   "steps": [
//!     {"as": "pix",  "cmd": ["vs_pix.py", "--image", "$INPUT"]},
//!     {"as": "ocr",  "cmd": ["vs_ocr.py", "--image", "$INPUT"]},
//!     {"as": "fused","cmd": ["vs_fusion.py", "--reports", "$pix", "$ocr"]}
//!   ],
//!   "report": "fused"
//! }
//! ```
//!
//! 变量：$INPUT/$URL/$DPR/$COMPARE 来自 CLI；$<step> 为前序步骤输出。
//! "if"：变量为空则跳过该步骤。"optional"：失败不中断。
//!
//! 与 Python 版差异：传感器在进程内直接调用（非子进程），输出为 JSON 值
//! （非临时文件路径）——输出结构逐字段一致。

use std::collections::HashMap;
use std::path::Path;

use serde_json::{json, Value};

/// analyze 参数（对应 Python `--task/--input/--url/--compare/--prompt/--dpr`）。
pub struct AnalyzeParams {
    pub task: String,
    pub input: Option<String>,
    pub url: Option<String>,
    pub compare: Option<String>,
    pub prompt: Option<String>,
    pub dpr: f64,
}

/// 任务配置（tasks/<name>.json）。
#[derive(serde::Deserialize)]
struct TaskConfig {
    name: Option<String>,
    steps: Vec<Step>,
}

#[derive(serde::Deserialize)]
struct Step {
    #[serde(rename = "as")]
    as_name: String,
    cmd: Vec<String>,
    #[serde(rename = "if")]
    if_var: Option<String>,
    optional: Option<bool>,
}

/// 解析命令行参数（--flag value 或 --flag=value；多值 flag 收集后续非 flag 参数）。
fn parse_args(cmd: &[String]) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let mut i = 0;
    while i < cmd.len() {
        let c = &cmd[i];
        if let Some(rest) = c.strip_prefix("--") {
            if let Some((k, v)) = rest.split_once('=') {
                map.insert(k.to_string(), v.to_string());
            } else {
                // 收集后续所有非 flag 参数（多值 flag：--reports a b c）
                let mut vals: Vec<String> = Vec::new();
                let mut j = i + 1;
                while j < cmd.len() && !cmd[j].starts_with("--") {
                    vals.push(cmd[j].clone());
                    j += 1;
                }
                map.insert(rest.to_string(), vals.join(" "));
                i = j - 1;
            }
        }
        i += 1;
    }
    map
}

/// 执行一个传感器步骤（进程内调用，替代 Python 子进程）。
fn run_sensor(script: &str, args: &HashMap<String, String>, image: &str) -> Result<Value, String> {
    let img = Path::new(image);
    match script {
        "vs_pix.py" => crate::pix::run(
            img,
            &crate::pix::PixParams {
                colors: args.get("colors").and_then(|v| v.parse().ok()).unwrap_or(8),
                regions: args
                    .get("regions")
                    .map(|v| v.split(',').map(String::from).collect())
                    .unwrap_or_default(),
                compare: args.get("compare").cloned(),
                wcag: Vec::new(),
                diff_threshold: 0,
            },
        ),
        "vs_ocr.py" => crate::ocr::run(
            img,
            &crate::ocr::OcrParams {
                max_items: args
                    .get("max-items")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(60),
                min_conf: args
                    .get("min-conf")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0.5),
            },
        ),
        "vs_dom.py" => crate::dom::run(&crate::dom::DomParams {
            url: args.get("url").cloned().unwrap_or_default(),
            max_elements: args
                .get("max-elements")
                .and_then(|v| v.parse().ok())
                .unwrap_or(200),
            timeout_ms: 30_000,
        }),
        "vs_saliency.py" => crate::saliency::run(
            img,
            &crate::saliency::SaliencyParams {
                top_n: args.get("top-n").and_then(|v| v.parse().ok()).unwrap_or(5),
                min_score: args
                    .get("min-score")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0.3),
            },
        ),
        "vs_ascii.py" => crate::ascii::run(
            img,
            &crate::ascii::AsciiParams {
                cols: args.get("cols").and_then(|v| v.parse().ok()).unwrap_or(64),
                rows: args.get("rows").and_then(|v| v.parse().ok()).unwrap_or(36),
                color: args.contains_key("color"),
                element_aware: false,
                elements: Vec::new(),
            },
        ),
        "vs_scene_stats.py" => crate::scene_stats::run(
            img,
            &crate::scene_stats::SceneStatsParams {
                colors: args.get("colors").and_then(|v| v.parse().ok()).unwrap_or(8),
            },
        ),
        "vs_edge.py" => crate::edge::run(
            img,
            &crate::edge::EdgeParams {
                sigma: args
                    .get("sigma")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(1.0),
                hough_threshold: args
                    .get("hough-threshold")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(40),
                max_lines: args
                    .get("max-lines")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(50),
                max_points: args
                    .get("max-points")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(2000),
                // 默认路径：密度热图（<1K tokens）；probe 时传 density=false 取原始点
                density: true,
            },
        ),
        "vs_depth.py" => crate::depth::run(
            img,
            &crate::depth::DepthParams {
                bins: args.get("bins").and_then(|v| v.parse().ok()).unwrap_or(3),
                region: args.get("region").cloned(),
            },
        ),
        "vs_segment.py" => crate::segment::run(
            img,
            &crate::segment::SegmentParams {
                points: Vec::new(),
                labels: Vec::new(),
                min_area: 0.0,
                max_instances: 10,
            },
        ),
        other => Err(format!("analyze: 未知传感器脚本 {other}")),
    }
}

/// 运行 analyze 任务。
pub fn run(p: &AnalyzeParams) -> Result<Value, String> {
    // 防御：task 名仅允许文件名安全字符
    let valid = !p.task.is_empty()
        && p.task.len() <= 64
        && p.task
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
    if !valid {
        return Err(format!(
            "task 参数应为预置任务配置名，而非自由文本（长度 {}）",
            p.task.len()
        ));
    }

    let tasks_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
        .join("python")
        .join("tasks");
    let cfg_path = tasks_dir.join(format!("{}.json", p.task));
    if !cfg_path.exists() {
        return Err(format!("unknown task: {}", p.task));
    }
    let cfg_text = std::fs::read_to_string(&cfg_path).map_err(|e| format!("read task: {e}"))?;
    let cfg: TaskConfig =
        serde_json::from_str(&cfg_text).map_err(|e| format!("parse task: {e}"))?;

    let mut vars: HashMap<String, String> = HashMap::new();
    vars.insert("INPUT".into(), p.input.clone().unwrap_or_default());
    vars.insert("URL".into(), p.url.clone().unwrap_or_default());
    vars.insert("DPR".into(), p.dpr.to_string());
    vars.insert("COMPARE".into(), p.compare.clone().unwrap_or_default());

    let mut outputs: HashMap<String, Value> = HashMap::new();
    let mut skipped: Vec<String> = Vec::new();

    for step in &cfg.steps {
        // "if" 条件：引用的变量为空则跳过
        if let Some(cond) = &step.if_var {
            let key = cond.trim_start_matches('$');
            if vars.get(key).map(|v| v.is_empty()).unwrap_or(true) {
                skipped.push(step.as_name.clone());
                continue;
            }
        }
        // 解析 cmd：$VAR 替换（先查前序步骤输出，再查 CLI 变量）
        let mut resolved: Vec<String> = Vec::new();
        for c in &step.cmd {
            if let Some(key) = c.strip_prefix('$') {
                if outputs.contains_key(key) {
                    // 前序步骤输出：fusion 场景下用步骤名标记，后续按名取 JSON
                    resolved.push(format!("${key}"));
                } else {
                    let v = vars.get(key).cloned().unwrap_or_default();
                    resolved.push(v);
                }
            } else {
                resolved.push(c.clone());
            }
        }
        if resolved.is_empty() {
            continue;
        }
        let script = resolved[0].clone();
        let args = parse_args(&resolved[1..]);
        let image = vars.get("INPUT").cloned().unwrap_or_default();
        let result = if script == "vs_fusion.py" {
            // 融合：reports 引用前序步骤输出（$<step> 标记）
            let mut report_values: Vec<Value> = Vec::new();
            if let Some(rs) = args.get("reports") {
                for r in rs.split_whitespace() {
                    if let Some(key) = r.strip_prefix('$') {
                        if let Some(v) = outputs.get(key) {
                            report_values.push(v.clone());
                        }
                    } else if let Some(v) = outputs.get(r) {
                        report_values.push(v.clone());
                    }
                }
            }
            if report_values.is_empty() {
                Err("analyze: fusion 缺少前序步骤输出".into())
            } else {
                // 写临时文件（fusion::run 接受路径）
                let dir = std::env::temp_dir();
                let mut paths = Vec::new();
                for (i, v) in report_values.iter().enumerate() {
                    let path = dir.join(format!("vs_analyze_fusion_{}_{}.json", step.as_name, i));
                    std::fs::write(&path, serde_json::to_string(v).unwrap())
                        .map_err(|e| format!("write fusion input: {e}"))?;
                    paths.push(path);
                }
                let r = crate::fusion::run(&paths, args.get("groups").map(String::as_str));
                for path in &paths {
                    let _ = std::fs::remove_file(path);
                }
                r
            }
        } else {
            run_sensor(&script, &args, &image)
        };

        match result {
            Ok(v) => {
                outputs.insert(step.as_name.clone(), v);
            }
            Err(e) => {
                if step.optional.unwrap_or(false) {
                    outputs.insert(step.as_name.clone(), json!({"error": e}));
                } else {
                    return Err(format!("step {} failed: {e}", step.as_name));
                }
            }
        }
    }

    // 合并所有步骤输出为统一融合报告
    let mut merged = json!({
        "schema": "vision-report/v3",
        "task": cfg.name.clone().unwrap_or_else(|| p.task.clone()),
        "sensors": [],
        "source": {},
        "elements": [],
        "anomalies": [],
        "metrics": {},
        "truncated": false,
    });
    let mut sensor_names: Vec<String> = Vec::new();
    let mut truncated = false;
    for (name, report) in &outputs {
        if report.get("error").is_some() || report.get("schema").is_none() {
            continue;
        }
        if let Some(sn) = report.get("sensors").and_then(|v| v.as_array()) {
            for s in sn {
                if let Some(s) = s.as_str() {
                    if !sensor_names.contains(&s.to_string()) {
                        sensor_names.push(s.to_string());
                    }
                }
            }
        }
        if let Some(els) = report.get("elements").and_then(|v| v.as_array()) {
            if let Some(m) = merged["elements"].as_array_mut() {
                m.extend(els.iter().cloned());
            }
        }
        if let Some(ans) = report.get("anomalies").and_then(|v| v.as_array()) {
            if let Some(m) = merged["anomalies"].as_array_mut() {
                m.extend(ans.iter().cloned());
            }
        }
        if let Some(metrics) = report.get("metrics").and_then(|v| v.as_object()) {
            for (k, v) in metrics {
                if !merged["metrics"].as_object().unwrap().contains_key(k) {
                    merged["metrics"][k] = v.clone();
                }
            }
        }
        if let Some(src) = report.get("source").and_then(|v| v.as_object()) {
            for (k, v) in src {
                if !merged["source"].as_object().unwrap().contains_key(k) {
                    merged["source"][k] = v.clone();
                }
            }
        }
        if report
            .get("truncated")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            truncated = true;
        }
        let _ = name;
    }
    merged["sensors"] = json!(sensor_names);
    merged["truncated"] = json!(truncated);
    if !skipped.is_empty() {
        merged["skipped_steps"] = json!(skipped);
    }
    if let Some(prompt) = &p.prompt {
        merged["user_prompt"] = json!(prompt);
        merged["note"] = json!(
            "analyze 仅执行预置数值任务；自由文本问题请用 semantic 动作（VLM 兜底）基于本报告的数值做推理。"
        );
    }
    Ok(merged)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_args_basic() {
        let cmd = vec![
            "--image".into(),
            "x.png".into(),
            "--colors".into(),
            "8".into(),
            "--flag".into(),
        ];
        let m = parse_args(&cmd);
        assert_eq!(m.get("image").unwrap(), "x.png");
        assert_eq!(m.get("colors").unwrap(), "8");
        assert!(m.contains_key("flag"));
    }

    #[test]
    fn parse_args_equals() {
        let cmd = vec!["--max-items=60".into()];
        let m = parse_args(&cmd);
        assert_eq!(m.get("max-items").unwrap(), "60");
    }

    #[test]
    fn invalid_task_name_rejected() {
        let p = AnalyzeParams {
            task: "这是一个自然语言长句，不是任务名".into(),
            input: None,
            url: None,
            compare: None,
            prompt: None,
            dpr: 1.0,
        };
        assert!(run(&p).is_err());
    }

    #[test]
    fn unknown_task_rejected() {
        let p = AnalyzeParams {
            task: "no-such-task".into(),
            input: None,
            url: None,
            compare: None,
            prompt: None,
            dpr: 1.0,
        };
        assert!(run(&p).is_err());
    }
}
