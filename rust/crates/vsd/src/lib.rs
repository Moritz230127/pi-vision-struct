//! vsd — V4 常驻视觉 daemon 库（调度器 + 沙箱 + preflight）。
//!
//! 二进制入口在 `main.rs`；本库暴露模块供集成测试与未来 MCP server 复用。

pub mod preflight;
pub mod sandbox;
pub mod scheduler;
