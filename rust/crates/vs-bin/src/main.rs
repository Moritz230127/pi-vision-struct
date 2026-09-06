//! vs-bin — V4 单二进制 CLI。
//!
//! 子命令 = action 枚举（25 动作，随阶段逐步填充）。S1 已接入：
//! `ascii` / `scene_stats` / `pix` / `edge`（纯算法传感器）+ `schema` 冒烟。

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use serde_json::json;
use vs_core::{ascii, edge, pix, scene_stats};

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
    };
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
