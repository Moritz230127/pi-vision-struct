//! 数理化场景统计（对应 Python `vs_scene_stats.py`，schema v3 输出逐字段一致）。
//!
//! 全部输出为带单位的数值：颜色直方图（MEDIANCUT 量化）、Otsu 前景/背景面积比、
//! 亮度对比度统计（WCAG）、空间布局量化（重心/包围盒/覆盖率）。

use std::collections::HashMap;
use std::path::Path;

use serde_json::{json, Value};

use crate::image::{load_rgb, to_gray_u8, RgbImage};
use crate::util::py_round_n;

/// 场景统计参数（对应 Python `--colors`）。
pub struct SceneStatsParams {
    pub colors: u32,
}

// ---------------------------------------------------------------- MEDIANCUT 量化

/// 量化盒：一组 (RGB, count) 颜色。
struct Box {
    colors: Vec<([u8; 3], u32)>,
}

impl Box {
    /// 最大通道跨度（PIL 选盒标准：range 最大的盒优先分裂）。
    fn range(&self) -> u32 {
        let (mut mn, mut mx) = ([255u8; 3], [0u8; 3]);
        for (c, _) in &self.colors {
            for i in 0..3 {
                mn[i] = mn[i].min(c[i]);
                mx[i] = mx[i].max(c[i]);
            }
        }
        (mx[0] - mn[0]).max(mx[1] - mn[1]).max(mx[2] - mn[2]) as u32
    }

    /// 在跨度最大通道的加权中位数处分裂；单色盒返回 `None`。
    fn split(&self) -> Option<(Box, Box)> {
        let (mut mn, mut mx) = ([255u8; 3], [0u8; 3]);
        for (c, _) in &self.colors {
            for i in 0..3 {
                mn[i] = mn[i].min(c[i]);
                mx[i] = mx[i].max(c[i]);
            }
        }
        let ch = (0..3).max_by_key(|&i| (mx[i] - mn[i]) as u32)?;
        if mx[ch] == mn[ch] {
            return None; // 单色盒无法分裂
        }
        let mut sorted = self.colors.clone();
        sorted.sort_by_key(|(c, _)| c[ch]);
        let total: u32 = sorted.iter().map(|(_, n)| n).sum();
        let mut acc = 0u32;
        let mut split_at = sorted.len() / 2;
        for (i, (_, n)) in sorted.iter().enumerate() {
            acc += n;
            if acc * 2 >= total {
                split_at = i + 1;
                break;
            }
        }
        if split_at == 0 || split_at >= sorted.len() {
            return None;
        }
        Some((
            Box {
                colors: sorted[..split_at].to_vec(),
            },
            Box {
                colors: sorted[split_at..].to_vec(),
            },
        ))
    }

    /// 盒的均值色（按 count 加权）+ 总像素数（对应 PIL getcolors 的 count）。
    fn mean(&self) -> (u32, [u8; 3]) {
        let (mut r, mut g, mut b, mut n) = (0u64, 0u64, 0u64, 0u64);
        for (c, cnt) in &self.colors {
            r += u64::from(c[0]) * u64::from(*cnt);
            g += u64::from(c[1]) * u64::from(*cnt);
            b += u64::from(c[2]) * u64::from(*cnt);
            n += u64::from(*cnt);
        }
        if n == 0 {
            return (0, [0, 0, 0]);
        }
        ((n as u32), [(r / n) as u8, (g / n) as u8, (b / n) as u8])
    }
}

/// MEDIANCUT 量化（对应 PIL `Image.quantize(method=MEDIANCUT)`）：
/// 颜色直方图 → 按最大通道跨度反复分裂 → 每盒均值色，按 count 降序。
fn median_cut(img: &RgbImage, n_colors: u32) -> Vec<(u32, [u8; 3])> {
    let mut hist: HashMap<[u8; 3], u32> = HashMap::new();
    for px in img.data.as_chunks::<3>().0 {
        *hist.entry([px[0], px[1], px[2]]).or_insert(0) += 1;
    }
    let mut boxes = vec![Box {
        colors: hist.into_iter().collect(),
    }];
    while boxes.len() < n_colors as usize {
        let Some(idx) = boxes
            .iter()
            .enumerate()
            .max_by_key(|(_, b)| b.range())
            .map(|(i, _)| i)
        else {
            break;
        };
        if let Some((b1, b2)) = boxes[idx].split() {
            boxes[idx] = b1;
            boxes.push(b2);
        } else {
            break; // 无法再分裂（颜色数不足）
        }
    }
    let mut result: Vec<(u32, [u8; 3])> = boxes.iter().map(Box::mean).collect();
    result.sort_by_key(|x| std::cmp::Reverse(x.0)); // count 降序（稳定排序）
    result
}

// ---------------------------------------------------------------- Otsu 阈值

/// skimage `threshold_otsu` 复刻：整数图像用 bincount 分箱
/// （bin_centers = arange(min, max+1)，counts[i] = 值为 i 的像素数），
/// 最大化类间方差，返回首个 argmax 对应的 bin 中心。
fn otsu_threshold(gray: &[u8]) -> f64 {
    if gray.is_empty() {
        return 0.0;
    }
    let min = *gray.iter().min().unwrap() as usize;
    let max = *gray.iter().max().unwrap() as usize;
    let n_bins = max - min + 1;
    let mut counts = vec![0f64; n_bins];
    for &g in gray {
        counts[g as usize - min] += 1.0;
    }
    // 类概率/均值（对应 np.cumsum 系列；bin_centers[i] = min + i）
    let mut weight1 = vec![0f64; n_bins];
    let mut mean1 = vec![0f64; n_bins];
    let mut acc_w = 0.0;
    let mut acc_m = 0.0;
    for i in 0..n_bins {
        let center = (min + i) as f64;
        acc_w += counts[i];
        acc_m += counts[i] * center;
        weight1[i] = acc_w;
        mean1[i] = if acc_w > 0.0 { acc_m / acc_w } else { 0.0 };
    }
    let mut weight2 = vec![0f64; n_bins];
    let mut mean2 = vec![0f64; n_bins];
    acc_w = 0.0;
    acc_m = 0.0;
    for i in (0..n_bins).rev() {
        let center = (min + i) as f64;
        acc_w += counts[i];
        acc_m += counts[i] * center;
        weight2[i] = acc_w;
        mean2[i] = if acc_w > 0.0 { acc_m / acc_w } else { 0.0 };
    }
    // variance12[i] = w1[i] * w2[i+1] * (m1[i] - m2[i+1])^2, i in 0..n_bins-1
    let mut best = (0usize, f64::NEG_INFINITY);
    for i in 0..n_bins - 1 {
        let v = weight1[i] * weight2[i + 1] * (mean1[i] - mean2[i + 1]).powi(2);
        if v > best.1 {
            best = (i, v);
        }
    }
    (min + best.0) as f64 // bin_centers[best.0]
}

// ---------------------------------------------------------------- 统计输出

/// 颜色直方图（MEDIANCUT 量化 → 按面积降序）。
fn color_histogram(img: &RgbImage, n_colors: u32) -> Value {
    let total = f64::from(img.width * img.height);
    let colors: Vec<Value> = median_cut(img, n_colors)
        .iter()
        .map(|(count, rgb)| {
            json!({
                "hex": format!("#{:02X}{:02X}{:02X}", rgb[0], rgb[1], rgb[2]),
                "rgb": [rgb[0], rgb[1], rgb[2]],
                "count": count,
                "percent": py_round_n(f64::from(*count) / total * 100.0, 2),
            })
        })
        .collect();
    json!(colors)
}

/// Otsu 前景/背景面积比。
fn area_ratios(gray: &[u8]) -> Value {
    let total = gray.len() as f64;
    let threshold = otsu_threshold(gray);
    let foreground = gray.iter().filter(|&&g| f64::from(g) < threshold).count() as f64;
    json!({
        "foreground_ratio": py_round_n(foreground / total, 4),
        "background_ratio": py_round_n((total - foreground) / total, 4),
        "threshold": py_round_n(threshold, 1),
        "method": "otsu",
    })
}

/// 亮度对比度统计（min/max/mean/总体 std + WCAG 极值比）。
fn contrast_stats(gray: &[u8]) -> Value {
    let n = gray.len() as f64;
    let arr: Vec<f64> = gray.iter().map(|&g| f64::from(g)).collect();
    let min = arr.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = arr.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean = arr.iter().sum::<f64>() / n;
    let std = (arr.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
    // WCAG 极值对比度（gamma 校正相对亮度）
    let rel = |v: f64| {
        let c = v / 255.0;
        if c <= 0.03928 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };
    let (l1, l2) = (rel(max), rel(min));
    let wcag = (l1.max(l2) + 0.05) / (l1.min(l2) + 0.05);
    json!({
        "luminance_mean": py_round_n(mean, 2),
        "luminance_std": py_round_n(std, 2),
        "luminance_min": py_round_n(min, 2),
        "luminance_max": py_round_n(max, 2),
        "wcag_contrast_ratio": py_round_n(wcag, 2),
    })
}

/// 空间布局量化：亮度加权重心、非黑包围盒、覆盖率、宽高比。
fn spatial_layout(gray: &[u8], w: u32, h: u32) -> Value {
    let total_lum: f64 = gray.iter().map(|&g| f64::from(g)).sum();
    let (center_x, center_y) = if total_lum > 0.0 {
        let (mut sx, mut sy) = (0.0f64, 0.0f64);
        for y in 0..h {
            for x in 0..w {
                let l = f64::from(gray[(y * w + x) as usize]);
                sx += f64::from(x) * l;
                sy += f64::from(y) * l;
            }
        }
        (sx / total_lum, sy / total_lum)
    } else {
        (f64::from(w) / 2.0, f64::from(h) / 2.0)
    };
    // 亮度 > 10 的像素包围盒
    let (mut min_x, mut min_y) = (w, h);
    let (mut max_x, mut max_y) = (0i64, 0i64);
    let mut count = 0u64;
    for y in 0..h {
        for x in 0..w {
            if f64::from(gray[(y * w + x) as usize]) > 10.0 {
                count += 1;
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(i64::from(x));
                max_y = max_y.max(i64::from(y));
            }
        }
    }
    let (spread_x, spread_y, coverage) = if count > 0 {
        (
            (max_x - i64::from(min_x)) as f64,
            (max_y - i64::from(min_y)) as f64,
            count as f64 / f64::from(w * h),
        )
    } else {
        (0.0, 0.0, 0.0)
    };
    json!({
        "center_of_mass_x": py_round_n(center_x / f64::from(w), 4),
        "center_of_mass_y": py_round_n(center_y / f64::from(h), 4),
        "spread_x_normalized": py_round_n(spread_x / f64::from(w), 4),
        "spread_y_normalized": py_round_n(spread_y / f64::from(h), 4),
        "coverage_ratio": py_round_n(coverage, 4),
        "aspect_ratio": py_round_n(f64::from(w) / f64::from(h), 4),
    })
}

/// 运行场景统计传感器，输出 schema v3 报告（对应 Python `main()`）。
pub fn run(path: &Path, p: &SceneStatsParams) -> Result<Value, String> {
    let img = load_rgb(path)?;
    let (w, h) = (img.width, img.height);
    let gray = to_gray_u8(&img);
    let mut report = crate::schema::envelope(
        Some("scene_stats"),
        &["pix", "PIL"],
        Some("image_px"),
        Some(json!({
            "type": "image",
            "path": path.display().to_string(),
            "size_px": [w, h],
        })),
    );
    report["metrics"] = json!({
        "color_histogram": color_histogram(&img, p.colors),
        "area": area_ratios(&gray),
        "contrast": contrast_stats(&gray),
        "spatial": spatial_layout(&gray, w, h),
        "pixel_count": w * h,
        "width": w,
        "height": h,
    });
    report["notation"] = json!(crate::schema::NOTATION_GUIDE);
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(w: u32, h: u32, px: &[u8]) -> RgbImage {
        RgbImage {
            width: w,
            height: h,
            data: px.to_vec(),
        }
    }

    #[test]
    fn median_cut_solid_color() {
        // 纯色图 → 1 个盒
        let im = img(2, 2, &[255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0]);
        let r = median_cut(&im, 8);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0], (4, [255, 0, 0]));
    }

    #[test]
    fn median_cut_two_colors() {
        // 红 2px + 蓝 2px → 2 个盒，count 降序
        let im = img(2, 2, &[255, 0, 0, 255, 0, 0, 0, 0, 255, 0, 0, 255]);
        let r = median_cut(&im, 8);
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].1, [255, 0, 0]);
        assert_eq!(r[1].1, [0, 0, 255]);
    }

    #[test]
    fn otsu_uniform() {
        // 均匀灰度 → 阈值 = 该灰度值
        let gray = vec![128u8; 100];
        assert_eq!(otsu_threshold(&gray), 128.0);
    }

    #[test]
    fn otsu_bimodal() {
        // 双峰 [0]*50 + [255]*50：variance12 对所有 i 恒定，argmax 取首个 → 0（与 skimage 一致）
        let mut gray = vec![0u8; 50];
        gray.extend(vec![255u8; 50]);
        assert_eq!(otsu_threshold(&gray), 0.0);
    }

    #[test]
    fn otsu_gradient() {
        // 渐变图（0..255 均匀）：阈值 = 127（与 skimage 实测一致）
        let gray: Vec<u8> = (0..=255u8).collect();
        assert_eq!(otsu_threshold(&gray), 127.0);
    }

    #[test]
    fn contrast_solid() {
        let gray = vec![128u8; 100];
        let c = contrast_stats(&gray);
        assert_eq!(c["luminance_std"], 0.0);
        assert_eq!(c["wcag_contrast_ratio"], 1.0);
        assert_eq!(c["luminance_min"], 128.0);
        assert_eq!(c["luminance_max"], 128.0);
    }

    #[test]
    fn spatial_solid_black() {
        // 全黑 → 无包围盒，重心 = 中心
        let gray = vec![0u8; 400];
        let s = spatial_layout(&gray, 20, 20);
        assert_eq!(s["coverage_ratio"], 0.0);
        assert_eq!(s["center_of_mass_x"], 0.5);
        assert_eq!(s["center_of_mass_y"], 0.5);
    }
}
