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
    /// 元素感知模式（plan3 E5a，AQuaUI 思想）：
    /// 元素 bbox 覆盖的栅格单元用 2×2 子采样（4 字符），空白区 1 字符。
    pub element_aware: bool,
    /// 元素 bbox 列表 [x1, y1, x2, y2]（融合元素，用于元素感知模式）。
    pub elements: Vec<[f64; 4]>,
}

impl Default for AsciiParams {
    fn default() -> Self {
        Self {
            cols: 64,
            rows: 36,
            color: false,
            element_aware: false,
            elements: Vec::new(),
        }
    }
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

/// 单个栅格单元的字符（亮度 → RAMP，可选颜色标注）。
fn cell_char(img: &RgbImage, x0: i64, x1: i64, y0: i64, y1: i64, use_color: bool) -> char {
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
        return ' ';
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
    ch
}

/// 栅格单元是否与元素 bbox 重叠（用于元素感知模式）。
fn cell_overlaps_bbox(x0: i64, y0: i64, x1: i64, y1: i64, b: &[f64; 4]) -> bool {
    let (bx0, by0, bx1, by1) = (b[0] as i64, b[1] as i64, b[2] as i64, b[3] as i64);
    x0 < bx1 && x1 > bx0 && y0 < by1 && y1 > by0
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

/// 元素感知栅格（plan3 E5a，AQuaUI 思想）：
/// 元素 bbox 覆盖的单元用 2×2 子采样（4 字符，高分辨率），空白区 1 字符（低分辨率）。
/// 行宽可变（元素单元水平展开 2 字符），行数 = rows。
fn to_ascii_grid_element_aware(
    img: &RgbImage,
    cols: u32,
    rows: u32,
    elements: &[[f64; 4]],
    use_color: bool,
) -> Result<Vec<String>, String> {
    let (w, h) = (img.width, img.height);
    let mut lines = Vec::with_capacity(rows as usize);
    for ry in 0..rows {
        let y0 = (f64::from(ry) * f64::from(h) / f64::from(rows)) as i64;
        let y1 = (f64::from(ry + 1) * f64::from(h) / f64::from(rows)) as i64;
        let mut line = String::new();
        for rx in 0..cols {
            let x0 = (f64::from(rx) * f64::from(w) / f64::from(cols)) as i64;
            let x1 = (f64::from(rx + 1) * f64::from(w) / f64::from(cols)) as i64;
            let in_element = elements
                .iter()
                .any(|b| cell_overlaps_bbox(x0, y0, x1, y1, b));
            if in_element {
                // 2×2 子采样：4 字符（水平展开 2，垂直 2）
                let mid_x = (x0 + x1) / 2;
                let mid_y = (y0 + y1) / 2;
                for (sy0, sy1) in [(y0, mid_y), (mid_y, y1)] {
                    for (sx0, sx1) in [(x0, mid_x), (mid_x, x1)] {
                        line.push(cell_char(img, sx0, sx1, sy0, sy1, use_color));
                    }
                }
            } else {
                line.push(cell_char(img, x0, x1, y0, y1, use_color));
            }
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
    let grid = if p.element_aware {
        to_ascii_grid_element_aware(&img, p.cols, p.rows, &p.elements, p.color)?
    } else {
        to_ascii_grid(&img, p.cols, p.rows, p.color)?
    };
    let chars: usize = grid.iter().map(|l| l.chars().count()).sum();
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
            "element_aware": p.element_aware,
        })),
    );
    report["ascii"] = json!({
        "cols": p.cols,
        "rows": p.rows,
        "grid": grid,
        "color": p.color,
        "element_aware": p.element_aware,
    });
    report["metrics"] = json!({
        "chars": chars,
        "ramp": RAMP,
        "element_aware": p.element_aware,
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

    #[test]
    fn element_aware_highlights_element_region() {
        // 4×4 图：左上 2×2 黑（元素区），其余白（空白区）
        let mut px = vec![255u8; 4 * 4 * 3];
        for y in 0..2 {
            for x in 0..2 {
                let i = (y * 4 + x) * 3;
                px[i] = 0;
                px[i + 1] = 0;
                px[i + 2] = 0;
            }
        }
        let img = RgbImage {
            width: 4,
            height: 4,
            data: px,
        };
        // 元素 bbox 覆盖左上 2×2
        let elements = [[0.0f64, 0.0, 2.0, 2.0]];
        let g = to_ascii_grid_element_aware(&img, 2, 2, &elements, false).unwrap();
        // 元素单元（左上）→ 2×2 子采样 4 字符（黑 → ' '）；空白单元 → 1 字符（白 → '@'）
        assert_eq!(g.len(), 2);
        // 行 0：元素单元（4 空格）+ 空白单元（1 '@'）
        assert_eq!(g[0].chars().count(), 5, "元素单元展开 4 字符 + 空白 1 字符");
        assert!(g[0].starts_with("    "), "元素区为 4 个空格（黑）");
        assert!(g[0].ends_with('@'), "空白区为 @（白）");
        // 行 1：无元素单元 → 2 个空白单元各 1 字符
        assert_eq!(g[1].chars().count(), 2, "无元素单元时保持 1 字符/单元");
        assert_eq!(g[1], "@@", "行 1 全白");
    }

    #[test]
    fn element_aware_density_higher_than_plain() {
        // 8×8 图：中心 4×4 黑（元素区），其余白
        let mut px = vec![255u8; 8 * 8 * 3];
        for y in 2..6 {
            for x in 2..6 {
                let i = (y * 8 + x) * 3;
                px[i] = 0;
                px[i + 1] = 0;
                px[i + 2] = 0;
            }
        }
        let img = RgbImage {
            width: 8,
            height: 8,
            data: px,
        };
        let elements = [[2.0f64, 2.0, 6.0, 6.0]];
        let aware = to_ascii_grid_element_aware(&img, 4, 4, &elements, false).unwrap();
        let plain = to_ascii_grid(&img, 4, 4, false).unwrap();
        let aware_chars: usize = aware.iter().map(|l| l.chars().count()).sum();
        let plain_chars: usize = plain.iter().map(|l| l.chars().count()).sum();
        assert!(
            aware_chars > plain_chars,
            "元素感知模式字符密度应更高: {aware_chars} vs {plain_chars}"
        );
    }

    #[test]
    fn element_aware_no_elements_falls_back_to_plain() {
        let img = img2x2([[255, 255, 255]; 4]);
        let g = to_ascii_grid_element_aware(&img, 2, 2, &[], false).unwrap();
        assert_eq!(g, vec!["@@", "@@"], "无元素时与普通模式一致");
    }
}
