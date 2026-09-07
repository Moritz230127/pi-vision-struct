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

/// 量化盒：一组 (RGB, count) 颜色 + 总像素数。
pub(crate) struct Box {
    pub(crate) colors: Vec<([u8; 3], u32)>,
    pixel_count: u32,
}

impl Box {
    /// RGB 包围盒体积（PIL `compute_box_volume`）：(r+1)(g+1)(b+1)。
    /// 体积 == 1 表示单色盒（不可分裂）。
    fn volume(&self) -> u32 {
        let (mut mn, mut mx) = ([255u8; 3], [0u8; 3]);
        for (c, _) in &self.colors {
            for i in 0..3 {
                mn[i] = mn[i].min(c[i]);
                mx[i] = mx[i].max(c[i]);
            }
        }
        (u32::from(mx[0] - mn[0]) + 1)
            * (u32::from(mx[1] - mn[1]) + 1)
            * (u32::from(mx[2] - mn[2]) + 1)
    }

    /// 分裂（PIL `split` + `splitlists` 精确复刻）：
    /// 轴 = 加权跨度最大（77/150/29 = luma 权重，首个最大）；
    /// 分裂点 = 沿轴累加 count 至 `left*2 > pixelCount`（严格），同色扩展归左；
    /// 右为空时从尾部拉同色到右（主色在末端的情形）。
    fn split(&self) -> Option<(Box, Box)> {
        let (mut mn, mut mx) = ([255u8; 3], [0u8; 3]);
        for (c, _) in &self.colors {
            for i in 0..3 {
                mn[i] = mn[i].min(c[i]);
                mx[i] = mx[i].max(c[i]);
            }
        }
        let f = [
            u32::from(mx[0] - mn[0]) * 77,
            u32::from(mx[1] - mn[1]) * 150,
            u32::from(mx[2] - mn[2]) * 29,
        ];
        // 首个最大（PIL `best < f[i]` 保持首个）
        let axis = (0..3).fold(0, |best, i| if f[i] > f[best] { i } else { best });
        let mut sorted = self.colors.clone();
        sorted.sort_by_key(|(c, _)| std::cmp::Reverse(c[axis])); // PIL 列表为降序
                                                                 // 分裂点：left*2 > pixelCount（严格大于）
        let mut left = 0u32;
        let mut split_idx = sorted.len();
        for (i, (_, n)) in sorted.iter().enumerate() {
            left += n;
            if left * 2 > self.pixel_count {
                split_idx = i + 1;
                break;
            }
        }
        // 同色扩展：splitColourVal 同值像素全部归左
        if split_idx > 0 && split_idx < sorted.len() {
            let split_val = sorted[split_idx - 1].0[axis];
            while split_idx < sorted.len() && sorted[split_idx].0[axis] == split_val {
                split_idx += 1;
            }
        }
        // 尾部回退：右为空时，从尾部拉与尾色相同的像素到右
        if split_idx >= sorted.len() {
            let tail_val = sorted[sorted.len() - 1].0[axis];
            let mut idx = sorted.len();
            while idx > 0 && sorted[idx - 1].0[axis] == tail_val {
                idx -= 1;
            }
            split_idx = idx;
        }
        if split_idx == 0 || split_idx >= sorted.len() {
            return None;
        }
        let (l, r) = sorted.split_at(split_idx);
        let lc: u32 = l.iter().map(|(_, n)| n).sum();
        Some((
            Box {
                colors: l.to_vec(),
                pixel_count: lc,
            },
            Box {
                colors: r.to_vec(),
                pixel_count: self.pixel_count - lc,
            },
        ))
    }

    /// 盒的均值色（按 count 加权）+ 总像素数（对应 PIL getcolors 的 count）。
    ///
    /// 颜色取整为 **round-half-up**（PIL `(int)(.5 + avg/count)`），非整数除法截断。
    pub(crate) fn mean(&self) -> (u32, [u8; 3]) {
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
        let rhu = |v: u64| (v as f64 / n as f64 + 0.5) as u8; // round half up
        ((n as u32), [rhu(r), rhu(g), rhu(b)])
    }
}

/// PIL `ImagingQuantHeap` 精确复刻：按 pixelCount 的 max-heap（父 ≥ 子）。
/// 存储盒索引（1-based，index 0 占位），比较时读 boxes 的 pixel_count。
struct BoxHeap {
    items: Vec<usize>,
}

impl BoxHeap {
    fn new() -> Self {
        Self { items: vec![0] }
    }

    /// PIL `ImagingQuantHeapAdd`：尾部插入，父 < 子时上浮（严格，平局不上浮）。
    fn add(&mut self, idx: usize, boxes: &[Option<Box>]) {
        let count = boxes[idx].as_ref().unwrap().pixel_count;
        self.items.push(idx);
        let mut k = self.items.len() - 1;
        while k > 1 && boxes[self.items[k / 2]].as_ref().unwrap().pixel_count < count {
            self.items[k] = self.items[k / 2];
            k /= 2;
        }
        self.items[k] = idx;
    }

    /// PIL `ImagingQuantHeapRemove`：取根（最大），末元素下沉（选较大子节点，平局取左）。
    fn remove(&mut self, boxes: &[Option<Box>]) -> Option<usize> {
        if self.items.len() <= 1 {
            return None;
        }
        let root = self.items[1];
        let last = self.items.pop().unwrap();
        if self.items.len() > 1 {
            let v = last;
            let vc = boxes[v].as_ref().unwrap().pixel_count;
            self.items[1] = v;
            let mut k = 1;
            loop {
                let mut l = k * 2;
                if l >= self.items.len() {
                    break;
                }
                if l + 1 < self.items.len()
                    && boxes[self.items[l]].as_ref().unwrap().pixel_count
                        < boxes[self.items[l + 1]].as_ref().unwrap().pixel_count
                {
                    l += 1;
                }
                if vc > boxes[self.items[l]].as_ref().unwrap().pixel_count {
                    break;
                }
                self.items[k] = self.items[l];
                k = l;
            }
            self.items[k] = v;
        }
        Some(root)
    }
}

/// MEDIANCUT 量化（PIL `Image.quantize(method=MEDIANCUT)` 精确复刻）：
/// 颜色直方图 → 堆按 pixelCount 最大优先反复分裂（跳过单色盒）→
/// 返回盒数组（内部节点为 `None`）+ 树遍历序（调色板索引 = 左优先前序遍历）。
///
/// 树结构：分裂时父盒标记为内部节点（`None`），左右子各占新索引。
pub(crate) fn median_cut_boxes(img: &RgbImage, n_colors: u32) -> (Vec<Option<Box>>, Vec<usize>) {
    let mut hist: HashMap<[u8; 3], u32> = HashMap::new();
    for px in img.data.as_chunks::<3>().0 {
        *hist.entry([px[0], px[1], px[2]]).or_insert(0) += 1;
    }
    let mut boxes: Vec<Option<Box>> = vec![Some(Box {
        colors: hist.into_iter().collect(),
        pixel_count: img.width * img.height,
    })];
    let mut children: Vec<(usize, usize)> = vec![(0, 0)]; // 内部节点的左右子
    let mut heap = BoxHeap::new();
    heap.add(0, &boxes);
    // 分裂 n_colors-1 次（PIL `while (--nPixels)`）；单色盒移除但不计分裂
    'outer: for _ in 1..n_colors {
        let idx = loop {
            let Some(i) = heap.remove(&boxes) else {
                break 'outer; // 堆空（PIL `goto done`）
            };
            if boxes[i].as_ref().unwrap().volume() > 1 {
                break i;
            }
        };
        if let Some((b1, b2)) = boxes[idx].as_ref().unwrap().split() {
            let l = boxes.len();
            boxes.push(Some(b1));
            children.push((0, 0)); // 占位（叶盒无子）
            let r = boxes.len();
            boxes.push(Some(b2));
            children.push((0, 0));
            boxes[idx] = None; // 内部节点
            children[idx] = (l, r);
            heap.add(l, &boxes);
            heap.add(r, &boxes);
        } else {
            break;
        }
    }
    // 左优先前序遍历叶盒（PIL `annotate_hash_table` 的调色板索引序）
    fn traverse(
        node: usize,
        boxes: &[Option<Box>],
        children: &[(usize, usize)],
        order: &mut Vec<usize>,
    ) {
        if boxes[node].is_some() {
            order.push(node);
        } else {
            let (l, r) = children[node];
            traverse(l, boxes, children, order);
            traverse(r, boxes, children, order);
        }
    }
    let mut order = Vec::new();
    traverse(0, &boxes, &children, &mut order);
    (boxes, order)
}

/// 颜色直方图（PIL quantize 全流程复刻）：
/// 盒树 → 调色板（均值色）→ 每像素最近调色板精化（4×初始距离界）→ count 降序。
fn color_histogram(img: &RgbImage, n_colors: u32) -> Value {
    let (boxes, order) = median_cut_boxes(img, n_colors);
    let n = order.len();
    // 调色板按树遍历序（PIL `annotate_hash_table` 的索引序）；order 均为叶盒（Some）
    let palette: Vec<[u8; 3]> = order
        .iter()
        .map(|&i| boxes[i].as_ref().unwrap().mean().1)
        .collect();
    // 调色板两两平方距离 + 每入口按距离排序（PIL avgDist/avgDistSortKey）
    let dist2 = |a: [u8; 3], b: [u8; 3]| -> u32 {
        let d = |x: u8, y: u8| (i32::from(x) - i32::from(y)).pow(2) as u32;
        d(a[0], b[0]) + d(a[1], b[1]) + d(a[2], b[2])
    };
    let avg_dist: Vec<Vec<u32>> = (0..n)
        .map(|i| (0..n).map(|j| dist2(palette[i], palette[j])).collect())
        .collect();
    let sort_key: Vec<Vec<usize>> = (0..n)
        .map(|i| {
            let mut v: Vec<usize> = (0..n).collect();
            v.sort_by_key(|&j| avg_dist[i][j]);
            v
        })
        .collect();
    // 每 distinct 颜色 → 盒索引（盒树划分颜色，恰属一盒）→ 遍历序调色板索引
    let mut hist: HashMap<[u8; 3], u32> = HashMap::new();
    for px in img.data.as_chunks::<3>().0 {
        *hist.entry([px[0], px[1], px[2]]).or_insert(0) += 1;
    }
    let mut counts = vec![0u32; n];
    for (color, cnt) in &hist {
        let box_idx = boxes
            .iter()
            .position(|b| {
                b.as_ref()
                    .is_some_and(|bb| bb.colors.iter().any(|(c, _)| c == color))
            })
            .expect("color must be in a box");
        let pal_idx = order
            .iter()
            .position(|&i| i == box_idx)
            .expect("box must be in traversal order");
        // 最近调色板精化：初始距离 ×4 为界，搜索更近的调色板色
        let initial = dist2(*color, palette[pal_idx]);
        let mut best = pal_idx;
        let mut best_dist = initial;
        let bound = initial * 4;
        for &j in &sort_key[pal_idx] {
            if avg_dist[pal_idx][j] <= bound {
                let d = dist2(*color, palette[j]);
                if d < best_dist {
                    best_dist = d;
                    best = j;
                }
            } else {
                break;
            }
        }
        counts[best] += cnt;
    }
    // 输出按 count 降序（稳定）；平局保持调色板索引序（PIL getcolors 直方图序 + 稳定排序）
    let total = f64::from(img.width * img.height);
    let mut entries: Vec<(u32, [u8; 3])> = counts
        .iter()
        .enumerate()
        .filter(|(_, c)| **c > 0)
        .map(|(i, c)| (*c, palette[i]))
        .collect();
    entries.sort_by_key(|x| std::cmp::Reverse(x.0));
    let colors: Vec<Value> = entries
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

/// 布局描述（plan3 E5b）：3×3 网格密度 → 区域判定。
///
/// 把图像分成 3×3 网格，计算每格平均亮度（前景密度），
/// 按列/行/中心相对整体密度的比值判定：
///   left_sidebar / right_sidebar / top_toolbar / bottom_statusbar / main_content / uniform。
fn layout_summary(gray: &[u8], w: u32, h: u32) -> Value {
    // 3×3 网格平均亮度
    let mut density = [[0.0f64; 3]; 3];
    for gy in 0..3 {
        for gx in 0..3 {
            let (x0, x1) = (gy * w / 3, (gy + 1) * w / 3);
            let (y0, y1) = (gx * h / 3, (gx + 1) * h / 3);
            let mut sum = 0.0f64;
            let mut n = 0.0f64;
            for y in y0..y1 {
                for x in x0..x1 {
                    sum += f64::from(gray[(y * w + x) as usize]);
                    n += 1.0;
                }
            }
            density[gx as usize][gy as usize] = if n > 0.0 { sum / n } else { 0.0 };
        }
    }
    let overall = density.iter().flatten().sum::<f64>() / 9.0;
    let col = |c: usize| (density[0][c] + density[1][c] + density[2][c]) / 3.0;
    let row = |r: usize| (density[r][0] + density[r][1] + density[r][2]) / 3.0;
    let (left, right) = (col(0), col(2));
    let (top, bottom) = (row(0), row(2));
    let center = density[1][1];
    let mut regions: Vec<&str> = Vec::new();
    if overall > 0.0 {
        if left > overall * 1.3 && left > right {
            regions.push("left_sidebar");
        }
        if right > overall * 1.3 && right > left {
            regions.push("right_sidebar");
        }
        if top > overall * 1.3 {
            regions.push("top_toolbar");
        }
        if bottom > overall * 1.3 {
            regions.push("bottom_statusbar");
        }
        if center > overall * 1.1 {
            regions.push("main_content");
        }
    }
    if regions.is_empty() {
        regions.push("uniform");
    }
    json!(regions)
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
        "layout_summary": layout_summary(&gray, w, h),
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
        let (r, order) = median_cut_boxes(&im, 8);
        assert_eq!(r.len(), 1);
        assert_eq!(order, vec![0]);
        assert_eq!(r[0].as_ref().unwrap().mean(), (4, [255, 0, 0]));
    }

    #[test]
    fn median_cut_two_colors() {
        // 红 2px + 蓝 2px → 2 个盒
        let im = img(2, 2, &[255, 0, 0, 255, 0, 0, 0, 0, 255, 0, 0, 255]);
        let (r, order) = median_cut_boxes(&im, 8);
        assert_eq!(r.len(), 3); // 根 + 2 叶（根为内部节点 None）
        assert_eq!(order.len(), 2);
        assert_eq!(r[order[0]].as_ref().unwrap().mean().1, [255, 0, 0]);
        assert_eq!(r[order[1]].as_ref().unwrap().mean().1, [0, 0, 255]);
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

    #[test]
    fn layout_summary_left_sidebar() {
        // 30×30 图：左 10 列亮（左栏），其余暗（主区）
        let mut gray = vec![0u8; 30 * 30];
        for y in 0..30 {
            for x in 0..10 {
                gray[y * 30 + x] = 200;
            }
        }
        let s = layout_summary(&gray, 30, 30);
        let regions: Vec<&str> = s
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(
            regions.contains(&"left_sidebar"),
            "左栏应被检测: {regions:?}"
        );
    }

    #[test]
    fn layout_summary_top_toolbar() {
        // 30×30 图：上 10 行亮（工具栏），其余暗
        let mut gray = vec![0u8; 30 * 30];
        for y in 0..10 {
            for x in 0..30 {
                gray[y * 30 + x] = 200;
            }
        }
        let s = layout_summary(&gray, 30, 30);
        let regions: Vec<&str> = s
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(
            regions.contains(&"top_toolbar"),
            "顶部工具栏应被检测: {regions:?}"
        );
    }

    #[test]
    fn layout_summary_uniform() {
        // 均匀图 → uniform
        let gray = vec![128u8; 400];
        let s = layout_summary(&gray, 20, 20);
        let regions: Vec<&str> = s
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(regions, vec!["uniform"]);
    }
}
