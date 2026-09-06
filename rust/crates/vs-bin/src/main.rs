//! vs-bin — V4 单二进制 CLI。
//!
//! 子命令 = action 枚举（25 动作，S1 起逐步填充）。S0 阶段仅提供
//! `schema` 冒烟子命令（验证 vs-core 输出契约）与版本信息。

use clap::{Parser, Subcommand};
use serde_json::json;

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
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Schema { task } => {
            let env = vs_core::schema::envelope(Some(&task), &["vs-core"], Some("image_px"), None);
            println!("{}", serde_json::to_string_pretty(&env).unwrap());
            // 冒烟断言：schema 版本与 Python 契约一致
            assert_eq!(env["schema"], json!(vs_core::schema::SCHEMA));
        }
    }
}
