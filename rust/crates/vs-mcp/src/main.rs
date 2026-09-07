//! vs-mcp — pi-vision-struct MCP server（Rust，替代 server.mcp.ts）。
//!
//! 细粒度工具：每传感器一个 tool（26 个），全部调用 vs-core 纯 Rust 实现。
//! 仅 check/setup/env 保留 Python 回退（引导安装器无 Rust 对应物）。

use std::path::{Path, PathBuf};

use rmcp::handler::server::wrapper::{Json, Parameters};
use rmcp::transport::stdio;
use rmcp::{serve_server, tool, tool_router};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use vs_core::a11y;
use vs_core::analyze;
use vs_core::ascii;
use vs_core::audit;
use vs_core::audit3d;
use vs_core::blend;
use vs_core::capture;
use vs_core::cluster;
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
use vs_core::protocol;
use vs_core::rules;
use vs_core::saliency;
use vs_core::scene_stats;
use vs_core::segment;
use vs_core::wall;

// ---- 工具参数结构（每传感器最小化）----

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Default)]
struct CaptureParams {
    /// 区域 "x1,y1,x2,y2"（None = 全屏）
    region: Option<String>,
    /// 输出 PNG 路径
    out: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Default)]
struct ImageParams {
    image: String,
    max_items: Option<u32>,
    min_conf: Option<f64>,
    colors: Option<u32>,
    top_n: Option<u32>,
    min_score: Option<f64>,
    bins: Option<u32>,
    region: Option<String>,
    mode: Option<String>,
    max_shapes: Option<u32>,
    classes: Option<String>,
    max_instances: Option<u32>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Default)]
struct FileParams {
    file: String,
    pages: Option<String>,
    slide: Option<u32>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Default)]
struct UrlParams {
    url: String,
    max_elements: Option<u32>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Default)]
struct AppParams {
    app: Option<String>,
    max_items: Option<u32>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Default)]
struct ReportParams {
    report: String,
    canvas: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Default)]
struct FusionParams {
    /// 报告 JSON 路径数组
    reports: Vec<String>,
    groups: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Default)]
struct AnalyzeParams {
    task: String,
    input: Option<String>,
    url: Option<String>,
    compare: Option<String>,
    prompt: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Default)]
struct ProtocolParams {
    /// analyze | zoom | probe
    action: String,
    image: Option<String>,
    region: Option<String>,
    bbox: Option<String>,
    sensor: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Default)]
struct Audit3dParams {
    report: String,
    gap_threshold: Option<f64>,
    method: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Default)]
struct ClusterParams {
    dir: Option<String>,
    files: Option<String>,
    threshold: Option<f64>,
    max_files: Option<u32>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Default)]
struct WallParams {
    dir: String,
    colors: Option<u32>,
    max_files: Option<u32>,
}

/// 工具处理器。
#[derive(Debug, Clone, Default)]
struct VsToolHandler;

// ---- 辅助函数 ----

fn req_image(p: &ImageParams) -> Result<PathBuf, String> {
    if p.image.is_empty() {
        Err("缺少参数: image".into())
    } else {
        Ok(PathBuf::from(&p.image))
    }
}

fn req_file(p: &FileParams) -> Result<PathBuf, String> {
    if p.file.is_empty() {
        Err("缺少参数: file".into())
    } else {
        Ok(PathBuf::from(&p.file))
    }
}

fn parse_region(s: Option<&str>) -> Option<[u32; 4]> {
    let s = s?;
    let parts: Vec<u32> = s.split(',').filter_map(|v| v.trim().parse().ok()).collect();
    if parts.len() == 4 {
        Some([parts[0], parts[1], parts[2], parts[3]])
    } else {
        None
    }
}

/// Python 回退：仅 check/setup/env（引导安装器无 Rust 对应物）。
fn python_fallback(action: &str, extra: &serde_json::Map<String, Value>) -> Result<Value, String> {
    let pkg_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .ok_or_else(|| "无法定位项目根".to_string())?;
    let script = match action {
        "env" => return Ok(json!({"note": "env 自检请用 /vs check"})),
        "check" => "setup/vs_setup.py",
        "setup" => "setup/vs_setup.py",
        _ => return Err(format!("未知 action: {action}")),
    };
    let script_path = pkg_root.join("python").join(script);
    if !script_path.exists() {
        return Err(format!("Python 回退脚本不存在: {}", script_path.display()));
    }
    let mut args: Vec<String> = vec![script_path.to_string_lossy().into_owned()];
    if action == "check" {
        args.push("--check".into());
    } else if let Some(sa) = extra.get("setup_args").and_then(|v| v.as_str()) {
        args.extend(sa.split_whitespace().map(String::from));
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

// ---- 工具定义（26 个细粒度工具）----

#[tool_router(server_handler)]
impl VsToolHandler {
    /// 截屏（xcap，Wayland screencopy）
    #[tool(
        name = "capture",
        description = "截屏并输出 schema v3 envelope（xcap）"
    )]
    async fn capture(
        &self,
        Parameters(p): Parameters<CaptureParams>,
    ) -> Result<Json<Value>, String> {
        capture::run(&capture::CaptureParams {
            region: parse_region(p.region.as_deref()),
            out: p.out.unwrap_or_else(|| "/tmp/vs_capture.png".into()),
        })
        .map(Json)
    }

    /// OCR 文本提取（rapidocr-core）
    #[tool(name = "ocr", description = "OCR 文本提取（rapidocr-core，带坐标框）")]
    async fn ocr(&self, Parameters(p): Parameters<ImageParams>) -> Result<Json<Value>, String> {
        let image = req_image(&p)?;
        ocr::run(
            &image,
            &ocr::OcrParams {
                max_items: p.max_items.unwrap_or(100),
                min_conf: p.min_conf.unwrap_or(0.5),
            },
        )
        .map(Json)
    }

    /// DOM 元素提取（chromiumoxide 驱动 headless Chrome）
    #[tool(name = "dom", description = "DOM 元素提取（headless Chrome）")]
    async fn dom(&self, Parameters(p): Parameters<UrlParams>) -> Result<Json<Value>, String> {
        dom::run(&dom::DomParams {
            url: p.url,
            max_elements: p.max_elements.unwrap_or(60),
            timeout_ms: 60_000,
        })
        .map(Json)
    }

    /// 无障碍树读取（atspi）
    #[tool(name = "a11y", description = "AT-SPI 无障碍树读取")]
    async fn a11y(&self, Parameters(p): Parameters<AppParams>) -> Result<Json<Value>, String> {
        a11y::run(&a11y::A11yParams {
            app: p.app,
            max_items: p.max_items.unwrap_or(80),
        })
        .map(Json)
    }

    /// PDF 解析（pdfium-render）
    #[tool(name = "pdf", description = "PDF 文本/元素解析（pdfium）")]
    async fn pdf(&self, Parameters(p): Parameters<FileParams>) -> Result<Json<Value>, String> {
        let file = req_file(&p)?;
        pdf::run(&pdf::PdfParams {
            file: file.to_string_lossy().into_owned(),
            pages: p.pages,
        })
        .map(Json)
    }

    /// PPTX 解析（OOXML 自研）
    #[tool(name = "pptx", description = "PPTX 幻灯片解析（OOXML）")]
    async fn pptx(&self, Parameters(p): Parameters<FileParams>) -> Result<Json<Value>, String> {
        let file = req_file(&p)?;
        pptx::run(&pptx::PptxParams {
            file: file.to_string_lossy().into_owned(),
            slide: p.slide,
        })
        .map(Json)
    }

    /// .blend 解析（blend-rs）
    #[tool(name = "blend", description = ".blend 3D 场景解析（blend-rs）")]
    async fn blend(&self, Parameters(p): Parameters<FileParams>) -> Result<Json<Value>, String> {
        let file = req_file(&p)?;
        blend::run(&blend::BlendParams {
            file: file.to_string_lossy().into_owned(),
        })
        .map(Json)
    }

    /// 显著性检测（U²-Net）
    #[tool(name = "saliency", description = "显著性检测（U²-Net ONNX）")]
    async fn saliency(
        &self,
        Parameters(p): Parameters<ImageParams>,
    ) -> Result<Json<Value>, String> {
        let image = req_image(&p)?;
        saliency::run(
            &image,
            &saliency::SaliencyParams {
                top_n: p.top_n.unwrap_or(5),
                min_score: p.min_score.unwrap_or(0.3),
            },
        )
        .map(Json)
    }

    /// 深度估计（MiDaS）
    #[tool(name = "depth", description = "深度估计（MiDaS ONNX）")]
    async fn depth(&self, Parameters(p): Parameters<ImageParams>) -> Result<Json<Value>, String> {
        let image = req_image(&p)?;
        depth::run(
            &image,
            &depth::DepthParams {
                bins: p.bins.unwrap_or(3),
                region: p.region.clone(),
            },
        )
        .map(Json)
    }

    /// 前景分割（MobileSAM）
    #[tool(name = "segment", description = "前景分割（MobileSAM ONNX）")]
    async fn segment(&self, Parameters(p): Parameters<ImageParams>) -> Result<Json<Value>, String> {
        let image = req_image(&p)?;
        segment::run(
            &image,
            &segment::SegmentParams {
                points: vec![],
                labels: vec![],
                min_area: 0.01,
                max_instances: p.max_instances.unwrap_or(10),
            },
        )
        .map(Json)
    }

    /// OmniParser（YOLOv9 + Florence-2）
    #[tool(
        name = "omniparser",
        description = "OmniParser 图标检测 + 描述（YOLOv9 + Florence-2）"
    )]
    async fn omniparser(
        &self,
        Parameters(p): Parameters<ImageParams>,
    ) -> Result<Json<Value>, String> {
        let image = req_image(&p)?;
        omniparser::run(
            &image,
            &omniparser::OmniparserParams {
                max_items: p.max_items.unwrap_or(100),
                min_conf: p.min_conf.unwrap_or(0.3),
            },
        )
        .map(Json)
    }

    /// 开放词表检测（OWLv2）
    #[tool(name = "detect", description = "开放词表目标检测（OWLv2 ONNX）")]
    async fn detect(&self, Parameters(p): Parameters<ImageParams>) -> Result<Json<Value>, String> {
        let image = req_image(&p)?;
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
        .map(Json)
    }

    /// 文档版式分析（PP-DocLayoutV3）
    #[tool(name = "layout", description = "文档版式分析（PP-DocLayoutV3）")]
    async fn layout(&self, Parameters(p): Parameters<ImageParams>) -> Result<Json<Value>, String> {
        let image = req_image(&p)?;
        layout::run(
            &image,
            &layout::LayoutParams {
                max_items: p.max_items.unwrap_or(100),
                min_conf: p.min_conf.unwrap_or(0.5),
            },
        )
        .map(Json)
    }

    /// 场景统计（亮度/对比度/主色）
    #[tool(
        name = "scene_stats",
        description = "场景统计（亮度/对比度/主色/直方图）"
    )]
    async fn scene_stats(
        &self,
        Parameters(p): Parameters<ImageParams>,
    ) -> Result<Json<Value>, String> {
        let image = req_image(&p)?;
        scene_stats::run(
            &image,
            &scene_stats::SceneStatsParams {
                colors: p.colors.unwrap_or(8),
            },
        )
        .map(Json)
    }

    /// 像素分析（主色/亮度）
    #[tool(name = "pix", description = "像素分析（主色/亮度/饱和度）")]
    async fn pix(&self, Parameters(p): Parameters<ImageParams>) -> Result<Json<Value>, String> {
        let image = req_image(&p)?;
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
        .map(Json)
    }

    /// 边缘检测（Canny + Hough）
    #[tool(name = "edge", description = "边缘/直线检测（Canny + Hough）")]
    async fn edge(&self, Parameters(p): Parameters<ImageParams>) -> Result<Json<Value>, String> {
        let image = req_image(&p)?;
        edge::run(
            &image,
            &edge::EdgeParams {
                sigma: 1.0,
                hough_threshold: 40,
                max_lines: p.max_items.unwrap_or(50),
                max_points: 2000,
            },
        )
        .map(Json)
    }

    /// ASCII 栅格
    #[tool(name = "ascii", description = "ASCII 字符栅格")]
    async fn ascii(&self, Parameters(p): Parameters<ImageParams>) -> Result<Json<Value>, String> {
        let image = req_image(&p)?;
        ascii::run(
            &image,
            &ascii::AsciiParams {
                cols: 64,
                rows: 36,
                color: false,
            },
        )
        .map(Json)
    }

    /// 几何形状提取（VTracer）
    #[tool(name = "geometry", description = "几何形状提取（VTracer 矢量化）")]
    async fn geometry(
        &self,
        Parameters(p): Parameters<ImageParams>,
    ) -> Result<Json<Value>, String> {
        let image = req_image(&p)?;
        geometry::run(
            &image,
            &geometry::GeometryParams {
                mode: p.mode.clone().unwrap_or_else(|| "polygon".into()),
                max_shapes: p.max_shapes.unwrap_or(50),
            },
        )
        .map(Json)
    }

    /// 多报告融合
    #[tool(name = "fusion", description = "多传感器报告融合（匈牙利匹配）")]
    async fn fusion(&self, Parameters(p): Parameters<FusionParams>) -> Result<Json<Value>, String> {
        if p.reports.is_empty() {
            return Err("fusion 需要 reports 参数（JSON 路径数组）".into());
        }
        let reports: Vec<PathBuf> = p.reports.iter().map(PathBuf::from).collect();
        fusion::run(&reports, p.groups.as_deref()).map(Json)
    }

    /// 规则引擎
    #[tool(name = "rules", description = "视觉规则引擎（对齐/重叠/对比度）")]
    async fn rules(&self, Parameters(p): Parameters<ReportParams>) -> Result<Json<Value>, String> {
        rules::run(
            &PathBuf::from(&p.report),
            &rules::RulesParams {
                canvas: p.canvas.clone(),
                ..rules::RulesParams::default()
            },
        )
        .map(Json)
    }

    /// 审计
    #[tool(name = "audit", description = "视觉审计（重叠/越界/对比度）")]
    async fn audit(&self, Parameters(p): Parameters<ReportParams>) -> Result<Json<Value>, String> {
        audit::run(
            &PathBuf::from(&p.report),
            &audit::AuditParams {
                canvas: p.canvas.clone(),
                overlap_threshold: 0.05,
                min_overlap_area: 400.0,
            },
        )
        .map(Json)
    }

    /// 任务引擎（JSON 配置编排）
    #[tool(name = "analyze", description = "任务引擎：按 JSON 配置编排传感器")]
    async fn analyze(
        &self,
        Parameters(p): Parameters<AnalyzeParams>,
    ) -> Result<Json<Value>, String> {
        analyze::run(&analyze::AnalyzeParams {
            task: p.task,
            input: p.input,
            url: p.url,
            compare: p.compare,
            prompt: p.prompt,
            dpr: 1.0,
        })
        .map(Json)
    }

    /// 多轮反馈协议（analyze/zoom/probe）
    #[tool(name = "protocol", description = "多轮反馈协议（analyze/zoom/probe）")]
    async fn protocol(
        &self,
        Parameters(p): Parameters<ProtocolParams>,
    ) -> Result<Json<Value>, String> {
        let image = p.image.clone().unwrap_or_default();
        protocol::run(&protocol::ProtocolParams {
            action: p.action,
            image,
            region: p.region,
            bbox: p.bbox,
            sensor: p.sensor,
        })
        .map(Json)
    }

    /// 3D 装配审计
    #[tool(name = "audit3d", description = "3D 装配审计（OBB-SAT + 网格距离）")]
    async fn audit3d(
        &self,
        Parameters(p): Parameters<Audit3dParams>,
    ) -> Result<Json<Value>, String> {
        audit3d::run(&audit3d::Audit3dParams {
            report: p.report,
            gap_threshold: p.gap_threshold.unwrap_or(5.0),
            method: p.method.unwrap_or_else(|| "auto".into()),
        })
        .map(Json)
    }

    /// CLIP 离线聚类
    #[tool(name = "cluster", description = "CLIP 感知相似度聚类")]
    async fn cluster(
        &self,
        Parameters(p): Parameters<ClusterParams>,
    ) -> Result<Json<Value>, String> {
        cluster::run(&cluster::ClusterParams {
            dir: p.dir,
            files: p.files,
            threshold: p.threshold.unwrap_or(0.8),
            max_files: p.max_files.unwrap_or(200) as usize,
            exts: vec!["png".into(), "jpg".into(), "jpeg".into(), "webp".into()],
        })
        .map(Json)
    }

    /// 壁纸程序化分类
    #[tool(name = "wall", description = "壁纸批量程序化分类（主色/亮度/饱和度）")]
    async fn wall(&self, Parameters(p): Parameters<WallParams>) -> Result<Json<Value>, String> {
        wall::run(
            &PathBuf::from(&p.dir),
            &wall::WallParams {
                colors: p.colors.unwrap_or(5),
                max_files: p.max_files.unwrap_or(200),
                exts: vec!["png".into(), "jpg".into(), "jpeg".into(), "webp".into()],
            },
        )
        .map(Json)
    }

    /// 环境自检（Python 回退）
    #[tool(name = "check", description = "环境自检（模型/二进制/依赖）")]
    async fn check(
        &self,
        Parameters(p): Parameters<serde_json::Map<String, Value>>,
    ) -> Result<Json<Value>, String> {
        python_fallback("check", &p).map(Json)
    }

    /// 引导安装（Python 回退）
    #[tool(name = "setup", description = "引导安装器（模型下载/目录创建）")]
    async fn setup(
        &self,
        Parameters(p): Parameters<serde_json::Map<String, Value>>,
    ) -> Result<Json<Value>, String> {
        python_fallback("setup", &p).map(Json)
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let handler = VsToolHandler;
    let service = serve_server(handler, stdio()).await?;
    service.waiting().await?;
    Ok(())
}
