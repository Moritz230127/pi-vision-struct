//! vs-core — 数理化视觉核心算法库（V4 全面 Rust 重写）。
//!
//! 与 Python 版 `vs_schema.py` 输出契约**逐字段兼容**（schema v3）：
//! 所有传感器输出同构 element，融合算子与 DeepSeek 按同一模型消费。
//!
//! 模块划分（对应 plan.md S0-S2）：
//! - [`schema`]：元素模型 / 报告信封 / 视觉原语记法
//! - [`bbox`]：坐标系变换与 bbox 运算
//! - [`color`]：CIE Lab ΔE76 / WCAG 对比度 / hex 互转
//! - [`util`]：Python 兼容的取整等基础工具

pub mod a11y;
pub mod ascii;
pub mod audit;
pub mod bbox;
pub mod blend;
pub mod capture;
pub mod color;
pub mod depth;
pub mod detect;
pub mod dom;
pub mod edge;
pub mod fusion;
pub mod geometry;
pub mod image;
pub mod layout;
pub mod ocr;
pub mod omniparser;
pub mod pdf;
pub mod pix;
pub mod pptx;
pub mod rules;
pub mod saliency;
pub mod scene_stats;
pub mod schema;
pub mod segment;
pub mod setup;
pub mod util;
pub mod wall;
