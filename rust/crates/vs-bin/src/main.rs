//! vs-bin — V4 单二进制 CLI。
//!
//! 子命令 = action 枚举（25 动作，随阶段逐步填充）。S1-S2 已接入：
//! `ascii` / `scene_stats` / `pix` / `edge`（纯算法传感器）+
//! `fusion` / `rules` / `audit` / `geometry`（融合与规则引擎）+ `schema` 冒烟。

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use serde_json::json;
use vs_core::{
    a11y, analyze, ascii, audit, audit3d, blend, capture, cluster, depth, detect, dom, edge,
    element_fusion, fusion, geometry, gui, layout, ocr, omniparser, pdf, pix, pptx, protocol,
    rules, saliency, scene_stats, segment, setup, wall,
};

/// 数理化视觉套件 V4（全面 Rust 重写）。
#[derive(Parser)]
#[command(name = "vs", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// schema v3 冒烟测试：输出报告信封（验证 vs-core 输出契约）。
    Schema {
        /// 任务名（默认 "smoke"）
        #[arg(long, default_value = "smoke")]
        task: String,
    },
    /// ASCII 栅格（亮度 → 字符密度，可选 8 色标注）。
    Ascii {
        #[arg(long)]
        image: PathBuf,
        #[arg(long, default_value_t = 64)]
        cols: u32,
        #[arg(long, default_value_t = 36)]
        rows: u32,
        #[arg(long)]
        color: bool,
        /// 元素感知模式（plan3 E5a）：元素区域 2× 字符密度，空白区 1×。
        #[arg(long)]
        element_aware: bool,
        /// 元素 bbox 列表（JSON 数组 [[x1,y1,x2,y2],...]，配合 --element-aware）。
        #[arg(long)]
        elements: Option<String>,
    },
    /// 数理化场景统计（MEDIANCUT 直方图 / Otsu 面积比 / 对比度 / 空间布局）。
    #[command(name = "scene_stats")]
    SceneStats {
        #[arg(long)]
        image: PathBuf,
        #[arg(long, default_value_t = 8)]
        colors: u32,
    },
    /// 像素级取色（SLIC 主色直方图 / 区域 / diff / WCAG）。
    Pix {
        #[arg(long)]
        image: PathBuf,
        #[arg(long, default_value_t = 8)]
        colors: u32,
    },
    /// 亚像素边缘检测（Devernay + Hough 直线）。
    Edge {
        #[arg(long)]
        image: PathBuf,
        #[arg(long, default_value_t = 1.0)]
        sigma: f64,
        #[arg(long, default_value_t = 40)]
        hough_threshold: u32,
        #[arg(long, default_value_t = 50)]
        max_lines: u32,
        #[arg(long, default_value_t = 2000)]
        max_points: u32,
        /// 输出 16×16 边缘密度热图（默认路径，<1K tokens）；缺省输出原始 edge_point
        #[arg(long)]
        density: bool,
    },
    /// 融合引擎（D-S 证据 + 匈牙利匹配）。
    Fusion {
        /// 传感器报告 JSON 路径（可多个）
        #[arg(long, required = true, num_args = 1..)]
        reports: Vec<String>,
        /// 匹配组定义 JSON
        #[arg(long)]
        groups: Option<String>,
    },
    /// 元素融合层（plan3 E1）：多传感器 → 一份融合元素清单。
    #[command(name = "fusion_elements")]
    FusionElements {
        /// 传感器报告 JSON 路径（可多个）
        #[arg(long, required = true, num_args = 1..)]
        reports: Vec<String>,
        /// bbox IoU 阈值（默认 0.5）
        #[arg(long, default_value_t = 0.5)]
        min_iou: f64,
        /// 最低置信度（默认 0.0）
        #[arg(long, default_value_t = 0.0)]
        min_conf: f64,
    },
    /// 设计准则规则引擎（对比度/重叠/对齐/间距/安全区）。
    Rules {
        #[arg(long)]
        report: PathBuf,
        /// 画布尺寸 WxH
        #[arg(long)]
        canvas: Option<String>,
    },
    /// 元素级几何/样式审计。
    Audit {
        #[arg(long)]
        report: PathBuf,
        /// 画布尺寸 WxH
        #[arg(long)]
        canvas: Option<String>,
    },
    /// 几何原语（VTracer SVG → 形状）。
    Geometry {
        #[arg(long)]
        image: PathBuf,
        /// polygon | spline
        #[arg(long, default_value = "polygon")]
        mode: String,
        #[arg(long, default_value_t = 50)]
        max_shapes: u32,
    },
    /// 显著性检测（U²-Net ONNX）。
    Saliency {
        #[arg(long)]
        image: PathBuf,
        #[arg(long, default_value_t = 5)]
        top_n: u32,
        #[arg(long, default_value_t = 0.3)]
        min_score: f64,
    },
    /// 深度估计（MiDaS-small ONNX）。
    Depth {
        #[arg(long)]
        image: PathBuf,
        #[arg(long, default_value_t = 3)]
        bins: u32,
        /// x1,y1,x2,y2 区域深度统计
        #[arg(long)]
        region: Option<String>,
    },
    /// 实例分割（MobileSAM）。
    Segment {
        #[arg(long)]
        image: PathBuf,
        #[arg(long, default_value_t = 0.01)]
        min_area_ratio: f64,
        #[arg(long, default_value_t = 10)]
        max_instances: u32,
    },
    /// OCR 文本提取（rapidocr-core）。
    Ocr {
        #[arg(long)]
        image: PathBuf,
        #[arg(long, default_value_t = 100)]
        max_items: u32,
        #[arg(long, default_value_t = 0.5)]
        min_conf: f64,
    },
    /// 截屏（xcap，Wayland screencopy）。
    Capture {
        /// 区域 x1,y1,x2,y2
        #[arg(long)]
        region: Option<String>,
        /// 输出 PNG 路径
        #[arg(long, default_value = "/tmp/vs_capture.png")]
        out: String,
        /// F5：窗口级截图（niri screenshot-window --id）
        #[arg(long)]
        window: Option<String>,
    },
    /// DOM 提取（chromiumoxide 驱动 headless Chrome）。
    Dom {
        #[arg(long)]
        url: String,
        #[arg(long, default_value_t = 60)]
        max_elements: u32,
    },
    /// 无障碍树（atspi）。
    A11y {
        #[arg(long)]
        app: Option<String>,
        #[arg(long, default_value_t = 80)]
        max_items: u32,
    },
    /// PDF 解析（pdfium-render）。
    Pdf {
        #[arg(long)]
        file: PathBuf,
        #[arg(long)]
        pages: Option<String>,
    },
    /// PPTX 解析（OOXML 自研）。
    Pptx {
        #[arg(long)]
        file: PathBuf,
        #[arg(long)]
        slide: Option<u32>,
    },
    /// .blend 解析（blend-rs）。
    Blend {
        #[arg(long)]
        file: PathBuf,
    },
    /// CLIP 离线聚类（open_clip ViT-B-32 ONNX）。
    Cluster {
        #[arg(long)]
        dir: Option<String>,
        #[arg(long)]
        files: Option<String>,
        #[arg(long, default_value_t = 0.75)]
        threshold: f64,
        #[arg(long, default_value_t = 200)]
        max_files: usize,
    },
    /// OmniParser（YOLOv9 + Florence-2）。
    Omniparser {
        #[arg(long)]
        image: PathBuf,
        #[arg(long, default_value_t = 100)]
        max_items: u32,
        #[arg(long, default_value_t = 0.3)]
        min_conf: f64,
    },
    /// 版式分析（PP-DocLayoutV3，Python 回退）。
    Layout {
        #[arg(long)]
        image: PathBuf,
        #[arg(long, default_value_t = 100)]
        max_items: u32,
        #[arg(long, default_value_t = 0.5)]
        min_conf: f64,
    },
    /// 开放词表检测（OWLv2）。
    Detect {
        #[arg(long)]
        image: PathBuf,
        /// 逗号分隔类别
        #[arg(long, default_value = "button,text")]
        classes: String,
        #[arg(long, default_value_t = 0.25)]
        min_conf: f64,
    },
    /// 壁纸批量程序化分类（主色/亮度/饱和度/宽高比）。
    Wall {
        #[arg(long)]
        dir: PathBuf,
        #[arg(long, default_value_t = 5)]
        colors: u32,
        #[arg(long, default_value_t = 200)]
        max_files: u32,
    },
    /// 任务引擎：按 JSON 配置编排传感器与融合算子。
    Analyze(Box<AnalyzeArgs>),
    /// 多轮反馈协议：analyze / zoom / probe / click [N]。
    Protocol(Box<ProtocolArgs>),
    /// GUI 行动层（plan4 G1）：click/type/key/drag/scroll/focus（wlrctl serve）。
    Gui(Box<GuiArgs>),
    Audit3d {
        #[arg(long)]
        report: String,
        #[arg(long, default_value_t = 15.0)]
        gap_threshold: f64,
        #[arg(long, default_value = "auto")]
        method: String,
    },
    /// 只读健康检查（模型存在性 + 二进制完整性 + 环境检测）。
    Check,
    /// 安装引导（模型目录创建 + 下载 URL 提示）。
    Setup,
}

/// Gui 子命令参数（Box 包裹避免 clippy large-size-difference）。
#[derive(clap::Args)]
struct GuiArgs {
    #[arg(long)]
    action: String,
    /// 元素编号（click_element 用，五级降级链入口）。
    #[arg(long)]
    element_id: Option<usize>,
    #[arg(long)]
    x: Option<f64>,
    #[arg(long)]
    y: Option<f64>,
    #[arg(long)]
    text: Option<String>,
    #[arg(long)]
    combo: Option<String>,
    #[arg(long, default_value = "left")]
    button: String,
    #[arg(long)]
    x1: Option<f64>,
    #[arg(long)]
    y1: Option<f64>,
    #[arg(long)]
    x2: Option<f64>,
    #[arg(long)]
    y2: Option<f64>,
    #[arg(long)]
    dy: Option<f64>,
    #[arg(long)]
    dx: Option<f64>,
    #[arg(long)]
    window: Option<String>,
    /// 点击验证（verify_click 闭环）。
    #[arg(long)]
    verify: bool,
    /// R2 chain：多步动作序列（"move 100 100; click; sleep 0.3; key tab"）。
    #[arg(long)]
    steps: Option<String>,
    /// R4 verify_task：任务前截图路径。
    #[arg(long)]
    before: Option<String>,
    /// R4 verify_task：任务后截图路径。
    #[arg(long)]
    after: Option<String>,
    /// R4 verify_task：期望窗口标题子串。
    #[arg(long)]
    expected_title: Option<String>,
    /// R4 verify_task：视口区域（逻辑坐标 "x1,y1,x2,y2"）。
    #[arg(long)]
    viewport: Option<String>,
}

/// Analyze 子命令参数（Box 包裹避免 clippy large-size-difference）。
#[derive(clap::Args)]
struct AnalyzeArgs {
    #[arg(long)]
    task: String,
    #[arg(long)]
    input: Option<String>,
    #[arg(long)]
    url: Option<String>,
    #[arg(long)]
    compare: Option<String>,
    #[arg(long)]
    prompt: Option<String>,
    #[arg(long, default_value_t = 1.0)]
    dpr: f64,
}

/// Protocol 子命令参数（Box 包裹避免 clippy large-size-difference）。
#[derive(clap::Args)]
struct ProtocolArgs {
    #[arg(long)]
    action: String,
    #[arg(long)]
    image: PathBuf,
    #[arg(long)]
    region: Option<String>,
    #[arg(long)]
    bbox: Option<String>,
    #[arg(long)]
    sensor: Option<String>,
    /// probe 确认模式 / click 编号引用：对指定融合元素 id 操作（E6/G1）。
    #[arg(long)]
    confirm: Option<String>,
    /// G3 闭环验证：click 后截图对比（verify_click）。
    #[arg(long)]
    verify_click: bool,
    /// G4 窗口过滤：只分析指定 niri 窗口 id 区域（排除终端/其他窗口内容）。
    #[arg(long)]
    window: Option<String>,
}

fn main() {
    let cli = Cli::parse();
    let out = match cli.command {
        Command::Schema { task } => {
            let env = vs_core::schema::envelope(Some(&task), &["vs-core"], Some("image_px"), None);
            // 冒烟断言：schema 版本与 Python 契约一致
            assert_eq!(env["schema"], json!(vs_core::schema::SCHEMA));
            env
        }
        Command::Ascii {
            image,
            cols,
            rows,
            color,
            element_aware,
            elements,
        } => {
            let parsed = elements
                .as_deref()
                .map(|s| {
                    serde_json::from_str::<Vec<Vec<f64>>>(s)
                        .map(|v| v.into_iter().map(|b| [b[0], b[1], b[2], b[3]]).collect())
                        .unwrap_or_default()
                })
                .unwrap_or_default();
            ascii::run(
                &image,
                &ascii::AsciiParams {
                    cols,
                    rows,
                    color,
                    element_aware,
                    elements: parsed,
                },
            )
            .unwrap_or_else(|e| json!({"error": "vs_ascii failed", "detail": e}))
        }
        Command::SceneStats { image, colors } => {
            scene_stats::run(&image, &scene_stats::SceneStatsParams { colors })
                .unwrap_or_else(|e| json!({"error": "vs_scene_stats failed", "detail": e}))
        }
        Command::Pix { image, colors } => pix::run(
            &image,
            &pix::PixParams {
                colors,
                regions: vec![],
                compare: None,
                wcag: vec![],
                diff_threshold: 30,
            },
        )
        .unwrap_or_else(|e| json!({"error": "vs_pix failed", "detail": e})),
        Command::Edge {
            image,
            sigma,
            hough_threshold,
            max_lines,
            max_points,
            density,
        } => edge::run(
            &image,
            &edge::EdgeParams {
                sigma,
                hough_threshold,
                max_lines,
                max_points,
                density,
            },
        )
        .unwrap_or_else(|e| json!({"error": "vs_edge failed", "detail": e})),
        Command::Fusion { reports, groups } => {
            let paths: Vec<PathBuf> = reports.iter().map(PathBuf::from).collect();
            fusion::run(&paths, groups.as_deref())
                .unwrap_or_else(|e| json!({"error": "vs_fusion failed", "detail": e}))
        }
        Command::FusionElements {
            reports,
            min_iou,
            min_conf,
        } => {
            let vals: Vec<serde_json::Value> = reports
                .iter()
                .map(|p| {
                    std::fs::read_to_string(p)
                        .ok()
                        .and_then(|s| serde_json::from_str(&s).ok())
                        .unwrap_or_else(|| json!({"error": format!("cannot read {p}")}))
                })
                .collect();
            element_fusion::run(&vals, &element_fusion::FusionParams { min_iou, min_conf })
                .unwrap_or_else(|e| json!({"error": "vs_fusion_elements failed", "detail": e}))
        }
        Command::Rules { report, canvas } => rules::run(
            &report,
            &rules::RulesParams {
                canvas,
                ..rules::RulesParams::default()
            },
        )
        .unwrap_or_else(|e| json!({"error": "vs_rules failed", "detail": e})),
        Command::Audit { report, canvas } => audit::run(
            &report,
            &audit::AuditParams {
                canvas,
                overlap_threshold: 0.05,
                min_overlap_area: 400.0,
            },
        )
        .unwrap_or_else(|e| json!({"error": "vs_audit failed", "detail": e})),
        Command::Geometry {
            image,
            mode,
            max_shapes,
        } => geometry::run(&image, &geometry::GeometryParams { mode, max_shapes })
            .unwrap_or_else(|e| json!({"error": "vs_geometry failed", "detail": e})),
        Command::Saliency {
            image,
            top_n,
            min_score,
        } => saliency::run(&image, &saliency::SaliencyParams { top_n, min_score })
            .unwrap_or_else(|e| json!({"error": "vs_saliency failed", "detail": e})),
        Command::Depth {
            image,
            bins,
            region,
        } => depth::run(&image, &depth::DepthParams { bins, region })
            .unwrap_or_else(|e| json!({"error": "vs_depth failed", "detail": e})),
        Command::Segment {
            image,
            min_area_ratio,
            max_instances,
        } => segment::run(
            &image,
            &segment::SegmentParams {
                points: vec![],
                labels: vec![],
                min_area: min_area_ratio,
                max_instances,
            },
        )
        .unwrap_or_else(|e| json!({"error": "vs_segment failed", "detail": e})),
        Command::Ocr {
            image,
            max_items,
            min_conf,
        } => ocr::run(
            &image,
            &ocr::OcrParams {
                max_items,
                min_conf,
            },
        )
        .unwrap_or_else(|e| json!({"error": "vs_ocr failed", "detail": e})),
        Command::Capture {
            region,
            out,
            window,
        } => capture::run(&capture::CaptureParams {
            region: region.as_deref().and_then(|s| {
                let p: Vec<u32> = s.split(',').filter_map(|v| v.trim().parse().ok()).collect();
                (p.len() == 4).then(|| [p[0], p[1], p[2], p[3]])
            }),
            window,
            out,
        })
        .unwrap_or_else(|e| json!({"error": "vs_capture failed", "detail": e})),
        Command::Dom { url, max_elements } => dom::run(&dom::DomParams {
            url,
            max_elements,
            timeout_ms: 60_000,
        })
        .unwrap_or_else(|e| json!({"error": "vs_dom failed", "detail": e})),
        Command::A11y { app, max_items } => a11y::run(&a11y::A11yParams { app, max_items })
            .unwrap_or_else(|e| json!({"error": "vs_a11y failed", "detail": e})),
        Command::Pdf { file, pages } => pdf::run(&pdf::PdfParams {
            file: file.to_string_lossy().into_owned(),
            pages,
        })
        .unwrap_or_else(|e| json!({"error": "vs_pdf failed", "detail": e})),
        Command::Pptx { file, slide } => pptx::run(&pptx::PptxParams {
            file: file.to_string_lossy().into_owned(),
            slide,
        })
        .unwrap_or_else(|e| json!({"error": "vs_pptx failed", "detail": e})),
        Command::Blend { file } => blend::run(&blend::BlendParams {
            file: file.to_string_lossy().into_owned(),
        })
        .unwrap_or_else(|e| json!({"error": "vs_blend failed", "detail": e})),
        Command::Cluster {
            dir,
            files,
            threshold,
            max_files,
        } => cluster::run(&cluster::ClusterParams {
            dir: dir.clone(),
            files: files.clone(),
            threshold,
            max_files,
            ..Default::default()
        })
        .unwrap_or_else(|e| json!({"error": "vs_cluster failed", "detail": e})),
        Command::Omniparser {
            image,
            max_items,
            min_conf,
        } => omniparser::run(
            &image,
            &omniparser::OmniparserParams {
                max_items,
                min_conf,
            },
        )
        .unwrap_or_else(|e| json!({"error": "vs_omniparser failed", "detail": e})),
        Command::Layout {
            image,
            max_items,
            min_conf,
        } => layout::run(
            &image,
            &layout::LayoutParams {
                max_items,
                min_conf,
            },
        )
        .unwrap_or_else(|e| json!({"error": "vs_layout failed", "detail": e})),
        Command::Detect {
            image,
            classes,
            min_conf,
        } => detect::run(
            &image,
            &detect::DetectParams {
                classes: classes
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect(),
                min_conf,
            },
        )
        .unwrap_or_else(|e| json!({"error": "vs_detect failed", "detail": e})),
        Command::Wall {
            dir,
            colors,
            max_files,
        } => wall::run(
            &dir,
            &wall::WallParams {
                colors,
                max_files,
                ..wall::WallParams::default()
            },
        )
        .unwrap_or_else(|e| json!({"error": "vs_wall failed", "detail": e})),
        Command::Analyze(args) => analyze::run(&analyze::AnalyzeParams {
            task: args.task.clone(),
            input: args.input.clone(),
            url: args.url.clone(),
            compare: args.compare.clone(),
            prompt: args.prompt.clone(),
            dpr: args.dpr,
        })
        .unwrap_or_else(|e| json!({"error": "vs_analyze failed", "detail": e})),
        Command::Protocol(args) => protocol::run(&protocol::ProtocolParams {
            action: args.action.clone(),
            image: args.image.to_string_lossy().into_owned(),
            region: args.region.clone(),
            bbox: args.bbox.clone(),
            sensor: args.sensor.clone(),
            confirm: args.confirm.clone(),
            verify_click: args.verify_click,
            window: args.window.clone(),
        })
        .unwrap_or_else(|e| json!({"error": "vs_protocol failed", "detail": e})),
        Command::Gui(args) => gui::run(&gui::GuiParams {
            action: args.action.clone(),
            element_id: args.element_id,
            x: args.x,
            y: args.y,
            text: args.text.clone(),
            combo: args.combo.clone(),
            button: Some(args.button.clone()),
            x1: args.x1,
            y1: args.y1,
            x2: args.x2,
            y2: args.y2,
            dy: args.dy,
            dx: args.dx,
            window: args.window.clone(),
            verify: args.verify,
            steps: args.steps.clone(),
            before: args.before.clone(),
            after: args.after.clone(),
            expected_title: args.expected_title.clone(),
            viewport: args.viewport.clone(),
        })
        .unwrap_or_else(|e| json!({"error": "vs_gui failed", "detail": e})),
        Command::Audit3d {
            report,
            gap_threshold,
            method,
        } => audit3d::run(&audit3d::Audit3dParams {
            report,
            gap_threshold,
            method,
        })
        .unwrap_or_else(|e| json!({"error": "vs_audit3d failed", "detail": e})),
        Command::Check => setup::check(),
        Command::Setup => setup::setup(),
    };
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
