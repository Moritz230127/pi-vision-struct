//! 色差度量（对应 Python `vs_schema.py` 色差段）。
//!
//! 全部输入为 RGB 通道值（0-255），输出与 Python 版逐字段一致：
//! - `srgb_to_linear` / `relative_luminance` / `contrast_ratio`：WCAG 2.x
//! - `rgb_to_lab` / `delta_e76`：CIE Lab ΔE76（sRGB→XYZ D65）
//! - `hex_to_rgb` / `rgb_to_hex` / `parse_hex`：hex 互转

use crate::util::py_round_i64;

/// sRGB 通道（0-255）→ 线性光强。
pub fn srgb_to_linear(c: f64) -> f64 {
    let c = c / 255.0;
    if c <= 0.03928 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// WCAG 相对亮度（gamma 校正线性光强的加权和）。
pub fn relative_luminance(rgb: [f64; 3]) -> f64 {
    let [r, g, b] = rgb;
    0.2126 * srgb_to_linear(r) + 0.7152 * srgb_to_linear(g) + 0.0722 * srgb_to_linear(b)
}

/// WCAG 2.x 对比度（AA 文本需 ≥4.5，AAA 需 ≥7.0）。
pub fn contrast_ratio(fg: [f64; 3], bg: [f64; 3]) -> f64 {
    let (mut l1, mut l2) = (relative_luminance(fg), relative_luminance(bg));
    if l1 < l2 {
        std::mem::swap(&mut l1, &mut l2);
    }
    (l1 + 0.05) / (l2 + 0.05)
}

/// sRGB → CIE Lab（D65 白点）。
pub fn rgb_to_lab(rgb: [f64; 3]) -> [f64; 3] {
    let [r, g, b] = rgb.map(srgb_to_linear);
    // sRGB → XYZ (D65)
    let x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
    let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
    let z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b;
    let (x, y, z) = (x / 0.95047, y / 1.0, z / 1.08883);

    fn f(t: f64) -> f64 {
        if t > 0.008856 {
            t.powf(1.0 / 3.0)
        } else {
            7.787 * t + 16.0 / 116.0
        }
    }

    let (fx, fy, fz) = (f(x), f(y), f(z));
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

/// CIEDE2000 的经典简化版（ΔE76）：L\*a\*b\* 空间欧氏距离。
///
/// 判读：<1 不可感知；1-2 极微弱；2-10 可感知；10-50 明显；>50 巨大。
pub fn delta_e76(c1: [f64; 3], c2: [f64; 3]) -> f64 {
    let [l1, a1, b1] = rgb_to_lab(c1);
    let [l2, a2, b2] = rgb_to_lab(c2);
    ((l1 - l2).powi(2) + (a1 - a2).powi(2) + (b1 - b2).powi(2)).sqrt()
}

/// `#RRGGBB` / `RRGGBB` → RGB 通道。
///
/// 非法输入 panic（对应 Python `hex_to_rgb` 抛 ValueError）；
/// 安全路径用 [`parse_hex`]。
pub fn hex_to_rgb(h: &str) -> [i64; 3] {
    parse_hex(h).expect("bad hex color")
}

/// RGB 通道 → `#RRGGBB`（大写，通道经 Python 语义取整）。
pub fn rgb_to_hex(rgb: [f64; 3]) -> String {
    format!(
        "#{:02X}{:02X}{:02X}",
        py_round_i64(rgb[0]),
        py_round_i64(rgb[1]),
        py_round_i64(rgb[2])
    )
}

/// 解析 hex 颜色，非法输入返回 `Err`（对应 Python `parse_hex` 的 ValueError）。
pub fn parse_hex(s: &str) -> Result<[i64; 3], String> {
    let h = s.trim_start_matches('#');
    if h.len() != 6 || !h.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!("bad hex color: {s:?}"));
    }
    Ok([
        i64::from_str_radix(&h[0..2], 16).unwrap(),
        i64::from_str_radix(&h[2..4], 16).unwrap(),
        i64::from_str_radix(&h[4..6], 16).unwrap(),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_and_luminance() {
        // 已知值：sRGB 255 → 1.0；0 → 0.0
        assert!((srgb_to_linear(255.0) - 1.0).abs() < 1e-9);
        assert_eq!(srgb_to_linear(0.0), 0.0);
        // 白/黑亮度
        assert!((relative_luminance([255.0, 255.0, 255.0]) - 1.0).abs() < 1e-9);
        assert_eq!(relative_luminance([0.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn contrast() {
        // 黑白对比度 = 21:1
        assert!((contrast_ratio([0.0, 0.0, 0.0], [255.0, 255.0, 255.0]) - 21.0).abs() < 1e-9);
        // 同色 = 1:1
        assert!((contrast_ratio([100.0, 100.0, 100.0], [100.0, 100.0, 100.0]) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn lab_and_delta() {
        // 同色 ΔE = 0
        assert_eq!(delta_e76([10.0, 20.0, 30.0], [10.0, 20.0, 30.0]), 0.0);
        // 黑白 ΔE 应显著（>50）
        assert!(delta_e76([0.0, 0.0, 0.0], [255.0, 255.0, 255.0]) > 50.0);
    }

    #[test]
    fn hex_roundtrip() {
        assert_eq!(hex_to_rgb("#FF8040"), [255, 128, 64]);
        assert_eq!(rgb_to_hex([255.0, 128.0, 64.0]), "#FF8040");
        assert_eq!(parse_hex("ff8040"), Ok([255, 128, 64]));
        assert!(parse_hex("xyz").is_err());
        assert!(parse_hex("#12345").is_err());
    }
}
