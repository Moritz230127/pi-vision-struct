//! vs-bin — V4 单二进制 CLI。
//!
//! 子命令 = action 枚举（25 动作，随阶段逐步填充）。S1-S2 已接入：
//! `ascii` / `scene_stats` / `pix` / `edge`（纯算法传感器）+
//! `fusion` / `rules` / `audit` / `geometry`（融合与规则引擎）+ `schema` 冒烟。

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use serde_json::json;
use vs_core::{ascii, audit, edge, fusion, geometry, pix, rules, scene_stats};

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
        } => ascii::run(&image, &ascii::AsciiParams { cols, rows, color })
            .unwrap_or_else(|e| json!({"error": "vs_ascii failed", "detail": e})),
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
        } => edge::run(
            &image,
            &edge::EdgeParams {
                sigma,
                hough_threshold,
                max_lines,
                max_points,
            },
        )
        .unwrap_or_else(|e| json!({"error": "vs_edge failed", "detail": e})),
        Command::Fusion { reports, groups } => {
            let paths: Vec<PathBuf> = reports.iter().map(PathBuf::from).collect();
            fusion::run(&paths, groups.as_deref())
                .unwrap_or_else(|e| json!({"error": "vs_fusion failed", "detail": e}))
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
        } => geometry::run(
            &image,
            &geometry::GeometryParams { mode, max_shapes },
        )
        .unwrap_or_else(|e| json!({"error": "vs_geometry failed", "detail": e})),
    };
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
