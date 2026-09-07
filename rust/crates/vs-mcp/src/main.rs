//! vs-mcp — pi-vision-struct MCP server（Rust，替代 server.mcp.ts）。
//!
//! 单端口 `vs` 工具（与 TS 版兼容）：`{action, ...参数}` → 传感器报告 JSON。
//! Rust 已实现的 action 直接调用 vs-core；未实现的（wallpaper/env/analyze/
//! cluster/blender_dump/depth/audit3d/zoom/probe/check/setup）回退 Python 脚本。

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use rmcp::handler::server::wrapper::{Json, Parameters};
use rmcp::transport::stdio;
use rmcp::{serve_server, tool, tool_router};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use vs_core::a11y;
use vs_core::ascii;
use vs_core::audit;
use vs_core::blend;
use vs_core::capture;
use vs_core::depth;
use vs_core::detect;
use vs_core::dom;
use vs_core::edge;
use vs_core::fusion;
use vs_core::geometry;
use vs_core::layout;
use vs_core::ocr;
use vs_core::omniparser;
use vs_core::pdf;
use vs_core::pix;
use vs_core::pptx;
use vs_core::rules;
use vs_core::saliency;
use vs_core::scene_stats;
use vs_core::segment;

/// vs 工具参数（单端口：action + 通用参数，与 TS 版 vsParamType 兼容）。
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Default)]
struct VsParams {
    /// 动作名（capture/ocr/dom/... 与 TS 版一致）
    action: String,
    // ---- 通用参数（各 action 按需取用）----
    image: Option<String>,
    out: Option<String>,
    region: Option<String>,
    colors: Option<u32>,
    url: Option<String>,
    file: Option<String>,
    report: Option<String>,
    canvas: Option<String>,
    classes: Option<String>,
    max_items: Option<u32>,
    min_conf: Option<f64>,
    max_elements: Option<u32>,
    app: Option<String>,
    slide: Option<u32>,
    pages: Option<String>,
    top_n: Option<u32>,
    min_score: Option<f64>,
    bins: Option<u32>,
    max_shapes: Option<u32>,
    mode: Option<String>,
    groups: Option<String>,
    blend: Option<String>,
    /// 未 Rust 化 action 的透传参数（Python 回退）
    #[serde(flatten)]
    extra: HashMap<String, Value>,
}

/// 工具处理器。
#[derive(Debug, Clone, Default)]
struct VsToolHandler;

/// action → 传感器报告。
fn dispatch(p: &VsParams) -> Result<Value, String> {
    let action = p.action.as_str();
    match action {
        "capture" => capture::run(&capture::CaptureParams {
            region: parse_region(p.region.as_deref()),
            out: p
                .out
                .clone()
                .unwrap_or_else(|| "/tmp/vs_capture.png".into()),
        }),
        "pixels" => {
            let image = req_path(p, "image")?;
            pix::run(
                &image,
                &pix::PixParams {
                    colors: p.colors.unwrap_or(8),
                    regions: vec![],
                    compare: None,
                    wcag: vec![],
                    diff_threshold: 30,
                },
            )
        }
        "ocr" => {
            let image = req_path(p, "image")?;
            ocr::run(
                &image,
                &ocr::OcrParams {
                    max_items: p.max_items.unwrap_or(100),
                    min_conf: p.min_conf.unwrap_or(0.5),
                },
            )
        }
        "scene_stats" => {
            let image = req_path(p, "image")?;
            scene_stats::run(
                &image,
                &scene_stats::SceneStatsParams {
                    colors: p.colors.unwrap_or(8),
                },
            )
        }
        "dom" => {
            let url = p
                .url
                .clone()
                .ok_or_else(|| "dom 需要 url 参数".to_string())?;
            dom::run(&dom::DomParams {
                url,
                max_elements: p.max_elements.unwrap_or(60),
                timeout_ms: 60_000,
            })
        }
        "pptx" => {
            let file = req_path(p, "file")?;
            pptx::run(&pptx::PptxParams {
                file: file.to_string_lossy().into_owned(),
                slide: p.slide,
            })
        }
        "pdf" => {
            let file = req_path(p, "file")?;
            pdf::run(&pdf::PdfParams {
                file: file.to_string_lossy().into_owned(),
                pages: p.pages.clone(),
            })
        }
        "a11y" => a11y::run(&a11y::A11yParams {
            app: p.app.clone(),
            max_items: p.max_items.unwrap_or(80),
        }),
        "detect" => {
            let image = req_path(p, "image")?;
            let classes = p
                .classes
                .clone()
                .unwrap_or_else(|| "button,text".into())
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            detect::run(
                &image,
                &detect::DetectParams {
                    classes,
                    min_conf: p.min_conf.unwrap_or(0.25),
                },
            )
        }
        "saliency" => {
            let image = req_path(p, "image")?;
            saliency::run(
                &image,
                &saliency::SaliencyParams {
                    top_n: p.top_n.unwrap_or(5),
                    min_score: p.min_score.unwrap_or(0.3),
                },
            )
        }
        "segment" => {
            let image = req_path(p, "image")?;
            segment::run(
                &image,
                &segment::SegmentParams {
                    points: vec![],
                    labels: vec![],
                    min_area: 0.01,
                    max_instances: p.max_items.unwrap_or(10),
                },
            )
        }
        "depth_midas" | "depth" => {
            let image = req_path(p, "image")?;
            depth::run(
                &image,
                &depth::DepthParams {
                    bins: p.bins.unwrap_or(3),
                    region: p.region.clone(),
                },
            )
        }
        "edge" => {
            let image = req_path(p, "image")?;
            edge::run(
                &image,
                &edge::EdgeParams {
                    sigma: 1.0,
                    hough_threshold: 40,
                    max_lines: p.max_items.unwrap_or(50),
                    max_points: 2000,
                },
            )
        }
        "ascii" => {
            let image = req_path(p, "image")?;
            ascii::run(
                &image,
                &ascii::AsciiParams {
                    cols: 64,
                    rows: 36,
                    color: false,
                },
            )
        }
        "geometry" => {
            let image = req_path(p, "image")?;
            geometry::run(
                &image,
                &geometry::GeometryParams {
                    mode: p.mode.clone().unwrap_or_else(|| "polygon".into()),
                    max_shapes: p.max_shapes.unwrap_or(50),
                },
            )
        }
        "fusion" => {
            let reports: Vec<PathBuf> = p
                .extra
                .get("reports")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str())
                        .map(PathBuf::from)
                        .collect()
                })
                .unwrap_or_default();
            if reports.is_empty() {
                return Err("fusion 需要 reports 参数（JSON 数组）".into());
            }
            fusion::run(&reports, p.groups.as_deref())
        }
        "audit" => {
            let report = req_path(p, "report")?;
            audit::run(
                &report,
                &audit::AuditParams {
                    canvas: p.canvas.clone(),
                    overlap_threshold: 0.05,
                    min_overlap_area: 400.0,
                },
            )
        }
        "rules" => {
            let report = req_path(p, "report")?;
            rules::run(
                &report,
                &rules::RulesParams {
                    canvas: p.canvas.clone(),
                    ..rules::RulesParams::default()
                },
            )
        }
        "omniparser" => {
            let image = req_path(p, "image")?;
            omniparser::run(
                &image,
                &omniparser::OmniparserParams {
                    max_items: p.max_items.unwrap_or(100),
                    min_conf: p.min_conf.unwrap_or(0.3),
                },
            )
        }
        "layout" => {
            let image = req_path(p, "image")?;
            layout::run(
                &image,
                &layout::LayoutParams {
                    max_items: p.max_items.unwrap_or(100),
                    min_conf: p.min_conf.unwrap_or(0.5),
                },
            )
        }
        "blend" | "blender_dump" => {
            let file = req_path(p, "blend")?;
            blend::run(&blend::BlendParams {
                file: file.to_string_lossy().into_owned(),
            })
        }
        // ---- Python 回退（未 Rust 化）----
        "wallpaper" | "env" | "analyze" | "cluster" | "audit3d" | "zoom" | "probe" | "check"
        | "setup" => python_fallback(action, p),
        other => Err(format!("未知 action: {other}")),
    }
}

/// 必填路径参数。
fn req_path(p: &VsParams, key: &str) -> Result<PathBuf, String> {
    let v = match key {
        "image" => p.image.clone(),
        "file" => p.file.clone(),
        "report" => p.report.clone(),
        "blend" => p.blend.clone(),
        _ => None,
    };
    v.map(PathBuf::from)
        .ok_or_else(|| format!("缺少参数: {key}"))
}

/// 区域 "x1,y1,x2,y2" → [u32; 4]。
fn parse_region(s: Option<&str>) -> Option<[u32; 4]> {
    let s = s?;
    let parts: Vec<u32> = s.split(',').filter_map(|v| v.trim().parse().ok()).collect();
    if parts.len() == 4 {
        Some([parts[0], parts[1], parts[2], parts[3]])
    } else {
        None
    }
}

/// Python 回退：spawn 对应 vs_*.py 脚本（未 Rust 化的 action）。
fn python_fallback(action: &str, p: &VsParams) -> Result<Value, String> {
    let pkg_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .ok_or_else(|| "无法定位项目根".to_string())?;
    let script = match action {
        "wallpaper" => "vs_wall.py",
        "env" => return Ok(json!({"note": "env 自检请用 /vs check"})),
        "analyze" => "vs_analyze.py",
        "cluster" => "vs_cluster.py",
        "audit3d" => "vs_audit3d.py",
        "zoom" => "vs_protocol.py",
        "probe" => "vs_protocol.py",
        "check" => "setup/vs_setup.py",
        "setup" => "setup/vs_setup.py",
        _ => return Err(format!("未知 action: {action}")),
    };
    let script_path = pkg_root.join("python").join(script);
    if !script_path.exists() {
        return Err(format!("Python 回退脚本不存在: {}", script_path.display()));
    }
    // 参数透传（action 专用 flag）
    let mut args: Vec<String> = vec![script_path.to_string_lossy().into_owned()];
    let push = |args: &mut Vec<String>, key: &str, flag: &str| {
        if let Some(v) = p.extra.get(key).and_then(|v| v.as_str()) {
            args.push(flag.to_string());
            args.push(v.to_string());
        }
    };
    match action {
        "wallpaper" => {
            push(&mut args, "dir", "--dir");
            push(&mut args, "colors", "--colors");
            push(&mut args, "max_files", "--max-files");
        }
        "analyze" => {
            push(&mut args, "task", "--task");
            push(&mut args, "input", "--input");
            push(&mut args, "url", "--url");
        }
        "cluster" => {
            push(&mut args, "dir", "--dir");
            push(&mut args, "files", "--files");
            push(&mut args, "threshold", "--threshold");
        }
        "audit3d" => {
            push(&mut args, "report", "--report");
            push(&mut args, "gap_threshold", "--gap-threshold");
        }
        "zoom" | "probe" => {
            args.insert(0, action.to_string());
            push(&mut args, "image", "--image");
            push(&mut args, "region", "--region");
            push(&mut args, "bbox", "--bbox");
            push(&mut args, "sensor", "--sensor");
        }
        "check" => {
            args.push("--check".into());
        }
        "setup" => {
            if let Some(sa) = p.extra.get("setup_args").and_then(|v| v.as_str()) {
                args.extend(sa.split_whitespace().map(String::from));
            }
        }
        _ => {}
    }
    let out = std::process::Command::new("python3")
        .args(&args)
        .output()
        .map_err(|e| format!("python 回退失败: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "python 回退失败: {}",
            String::from_utf8_lossy(&out.stderr)
                .chars()
                .take(300)
                .collect::<String>()
        ));
    }
    serde_json::from_str(&String::from_utf8_lossy(&out.stdout))
        .map_err(|e| format!("python 输出解析失败: {e}"))
}

#[tool_router(server_handler)]
impl VsToolHandler {
    #[tool]
    async fn vs(&self, Parameters(p): Parameters<VsParams>) -> Result<Json<Value>, String> {
        dispatch(&p).map(Json)
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let handler = VsToolHandler;
    let service = serve_server(handler, stdio()).await?;
    service.waiting().await?;
    Ok(())
}
