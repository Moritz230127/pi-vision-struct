//! ASCII 栅格传感器（对应 Python `vs_ascii.py`，schema v3 输出逐字段一致）。
//!
//! 把图像转成多分辨率 ASCII 文本栅格，供纯文本 LLM "粗看"：
//! 亮度 → 字符密度映射（[`RAMP`]），可选 8 色主色标注（`color`）。

use std::path::Path;

use serde_json::{json, Value};

use crate::image::{load_rgb, RgbImage};

/// 亮度 → 字符（从密到疏），与 Python `RAMP` 一致（10 字符）。
pub const RAMP: &str = "@%#*+=-:. ";

/// ASCII 栅格参数（对应 Python `--cols/--rows/--color`）。
pub struct AsciiParams {
    pub cols: u32,
    pub rows: u32,
    pub color: bool,
}

/// RGB → 亮度（0-255），无 gamma 校正（对应 Python `luminance`）。
fn luminance(r: u8, g: u8, b: u8) -> f64 {
    0.2126 * f64::from(r) + 0.7152 * f64::from(g) + 0.0722 * f64::from(b)
}

/// 区域主色（量化到 8 色），对应 Python `dominant_color`。
fn dominant_color(img: &RgbImage, x0: u32, x1: u32, y0: u32, y1: u32) -> char {
    let mut rs = 0.0f64;
    let mut gs = 0.0f64;
    let mut bs = 0.0f64;
    let mut n = 0.0f64;
    for y in y0..y1 {
        for x in x0..x1 {
            let p = img.pixel(x, y);
            rs += f64::from(p[0]);
            gs += f64::from(p[1]);
            bs += f64::from(p[2]);
            n += 1.0;
        }
    }
    let (r, g, b) = (rs / n, gs / n, bs / n);
    if r > 200.0 && g < 100.0 && b < 100.0 {
        'R'
    } else if g > 200.0 && r < 100.0 && b < 100.0 {
        'G'
    } else if b > 200.0 && r < 100.0 && g < 100.0 {
        'B'
    } else if r > 150.0 && g > 150.0 && b < 100.0 {
        'Y'
    } else if r > 150.0 && g < 100.0 && b > 150.0 {
        'M'
    } else if r < 100.0 && g > 150.0 && b > 150.0 {
        'C'
    } else if r > 200.0 && g > 200.0 && b > 200.0 {
        'W'
    } else {
        'K'
    }
}

/// 图像 → ASCII 栅格行列表（对应 Python `to_ascii_grid`）。
///
/// 分块采样：`y0 = int(ry * h / rows)`（整数截断，与 Python `int()` 一致）。
fn to_ascii_grid(
    img: &RgbImage,
    cols: u32,
    rows: u32,
    use_color: bool,
) -> Result<Vec<String>, String> {
    let (w, h) = (img.width, img.height);
    let mut lines = Vec::with_capacity(rows as usize);
    for ry in 0..rows {
        let y0 = (f64::from(ry) * f64::from(h) / f64::from(rows)) as i64;
        let y1 = (f64::from(ry + 1) * f64::from(h) / f64::from(rows)) as i64;
        let mut line = String::with_capacity(cols as usize);
        for rx in 0..cols {
            let x0 = (f64::from(rx) * f64::from(w) / f64::from(cols)) as i64;
            let x1 = (f64::from(rx + 1) * f64::from(w) / f64::from(cols)) as i64;
            let mut sum = 0.0f64;
            let mut n = 0u64;
            for y in y0..y1 {
                for x in x0..x1 {
                    let p = img.pixel(x as u32, y as u32);
                    sum += luminance(p[0], p[1], p[2]);
                    n += 1;
                }
            }
            if n == 0 {
                return Err(format!("empty block at ({rx},{ry})"));
            }
            let l = sum / n as f64;
            let idx = (((255.0 - l) / 255.0 * 9.0) as i64).clamp(0, 9) as usize;
            let mut ch = RAMP.as_bytes()[idx] as char;
            if use_color {
                let c = dominant_color(img, x0 as u32, x1 as u32, y0 as u32, y1 as u32);
                if c != 'K' {
                    ch = c;
                }
            }
            line.push(ch);
        }
        lines.push(line);
    }
    Ok(lines)
}

/// 运行 ASCII 传感器，输出 schema v3 报告（对应 Python `main()`）。
pub fn run(path: &Path, p: &AsciiParams) -> Result<Value, String> {
    if p.cols == 0 || p.rows == 0 {
        return Err(format!("cols/rows must be >= 1, got {}x{}", p.cols, p.rows));
    }
    let img = load_rgb(path)?;
    let (w, h) = (img.width, img.height);
    let grid = to_ascii_grid(&img, p.cols, p.rows, p.color)?;
    let mut report = crate::schema::envelope(
        Some("ascii"),
        &["ascii"],
        Some("image_px"),
        Some(json!({
            "type": "image",
            "path": path.display().to_string(),
            "size_px": [w, h],
            "cols": p.cols,
            "rows": p.rows,
            "color": p.color,
        })),
    );
    report["ascii"] = json!({
        "cols": p.cols,
        "rows": p.rows,
        "grid": grid,
        "color": p.color,
    });
    report["metrics"] = json!({
        "chars": p.cols * p.rows,
        "ramp": RAMP,
    });
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img2x2(px: [[u8; 3]; 4]) -> RgbImage {
        RgbImage {
            width: 2,
            height: 2,
            data: px.concat(),
        }
    }

    #[test]
    fn solid_white_grid() {
        let img = img2x2([[255, 255, 255]; 4]);
        let g = to_ascii_grid(&img, 2, 2, false).unwrap();
        assert_eq!(g, vec!["@@", "@@"]);
    }

    #[test]
    fn solid_black_grid() {
        let img = img2x2([[0, 0, 0]; 4]);
        let g = to_ascii_grid(&img, 2, 2, false).unwrap();
        assert_eq!(g, vec!["  ", "  "]);
    }

    #[test]
    fn four_color_grid_no_color() {
        // 红/绿/蓝/白 2x2：亮度 → RAMP 索引 7/2/8/0
        let img = img2x2([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]]);
        let g = to_ascii_grid(&img, 2, 2, false).unwrap();
        assert_eq!(g, vec![":#", ".@"]);
    }

    #[test]
    fn four_color_grid_with_color() {
        let img = img2x2([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]]);
        let g = to_ascii_grid(&img, 2, 2, true).unwrap();
        assert_eq!(g, vec!["RG", "BW"]);
    }

    #[test]
    fn dominant_color_rules() {
        let img = img2x2([[255, 255, 0], [255, 0, 255], [0, 255, 255], [128, 128, 128]]);
        assert_eq!(dominant_color(&img, 0, 1, 0, 1), 'Y');
        assert_eq!(dominant_color(&img, 1, 2, 0, 1), 'M');
        assert_eq!(dominant_color(&img, 0, 1, 1, 2), 'C');
        assert_eq!(dominant_color(&img, 1, 2, 1, 2), 'K');
    }
}
