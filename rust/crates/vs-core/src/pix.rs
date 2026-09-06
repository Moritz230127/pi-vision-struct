//! pix 传感器 — 确定性像素测量（对应 Python `vs_pix.py`，schema v3 输出逐字段一致）。
//!
//! 直方图（SLIC 超像素主色）/ 区域取色 / 亮度饱和度 / 像素级 diff / WCAG 2.x 对比度。
//! 纯像素运算，无模型。报告结构与 `vs_pix.py` 完全一致：
//! `{schema, task, sensors, coordsys, source, elements, anomalies, metrics,
//!   regions?, wcag?}`。

use std::collections::HashMap;
use std::path::Path;

use rayon::prelude::*;
use serde_json::{json, Value};

use crate::color::{contrast_ratio, parse_hex, rgb_to_hex, rgb_to_lab};
use crate::image::{load_rgb, RgbImage};
use crate::util::{py_round_i64, py_round_n};

/// pix 传感器参数（对应 Python `--colors/--regions/--compare/--wcag/--diff-threshold`）。
pub struct PixParams {
    pub colors: u32,
    pub regions: Vec<String>,
    pub compare: Option<String>,
    pub wcag: Vec<String>,
    pub diff_threshold: u32,
}

// ---------------------------------------------------------------- SLIC 超像素

/// skimage `slic` 的忠实移植（skimage 0.26：`_slic.pyx` + `slic_superpixels.py`）：
/// 预处理（float→[0,1] 全局归一化→Lab→×1/compactness）→ regular_grid 网格播种 →
/// 10 轮 k-means（窗口 ±2*step，距离 = d_color² + d_space²/step²）→ 连通性强制。
/// 标签从 1 起，与 Python `start_label=1` 一致。
fn slic(img: &RgbImage, n_segments: u32) -> Vec<u32> {
    let w = img.width as usize;
    let h = img.height as usize;
    let n = w * h;
    let compactness = 10.0f64;

    // 1) 预处理（对应 slic_superpixels.py）：
    //    img_as_float（uint8 → /255）→ 全局 min/max 归一化到 [0,1] → rgb2lab → ×(1/compactness)
    let mut fimg = vec![0.0f64; n * 3];
    let mut imin = f64::INFINITY;
    let mut imax = f64::NEG_INFINITY;
    for (i, &v) in img.data.iter().enumerate() {
        let f = f64::from(v) / 255.0;
        fimg[i] = f;
        imin = imin.min(f);
        imax = imax.max(f);
    }
    for v in fimg.iter_mut() {
        *v -= imin;
    }
    if imax != imin {
        let range = imax - imin;
        for v in fimg.iter_mut() {
            *v /= range;
        }
    }
    // RGB(0-255) → Lab（color.rs 的 D65 矩阵与 skimage rgb2lab 实测误差 <0.005 Lab）
    let mut lab = vec![0.0f64; n * 3];
    for (i, px) in fimg.as_chunks::<3>().0.iter().enumerate() {
        let l = rgb_to_lab([px[0] * 255.0, px[1] * 255.0, px[2] * 255.0]);
        lab[i * 3] = l[0];
        lab[i * 3 + 1] = l[1];
        lab[i * 3 + 2] = l[2];
    }
    let ratio = 1.0 / compactness;
    for v in lab.iter_mut() {
        *v *= ratio;
    }

    // 2) 网格播种（skimage `regular_grid((1, h, w), n_segments)` + `_get_grid_centroids`）
    let (starts, steps) = regular_grid_zyx(h, w, n_segments as usize);
    let step = steps.iter().copied().max().unwrap_or(1) as f64;
    let spatial_weight = 1.0 / (step * step);

    let mut centroids: Vec<[f64; 6]> = Vec::new(); // [z, y, x, L, a, b]
    let mut z = starts[0];
    while z < 1 {
        let mut y = starts[1];
        while y < h {
            let mut x = starts[2];
            while x < w {
                centroids.push([z as f64, y as f64, x as f64, 0.0, 0.0, 0.0]);
                x += steps[2];
            }
            y += steps[1];
        }
        z += steps[0];
    }
    let n_centroids = centroids.len();
    let mut segments = centroids;

    // 3) k-means（对应 `_slic_cython`：max_num_iter=10，窗口 ±2*step，严格 < 比较）
    let mut labels = vec![0u32; n]; // start_label=1 → 0 = 未分配
    let mut distance = vec![f64::INFINITY; n];
    for _ in 0..10 {
        let mut change = false;
        distance.iter_mut().for_each(|d| *d = f64::INFINITY);
        for (k, seg) in segments.iter().enumerate() {
            let (cz, cy, cx) = (seg[0], seg[1], seg[2]);
            // 窗口边界：fmax/fmin 语义（NaN 段 → 全图窗口，NaN 距离永不获胜）
            let z_min = ((cz - 2.0 * steps[0] as f64).max(0.0)) as usize;
            let z_max = ((cz + 2.0 * steps[0] as f64 + 1.0).min(1.0)) as usize;
            let y_min = ((cy - 2.0 * steps[1] as f64).max(0.0)) as usize;
            let y_max = ((cy + 2.0 * steps[1] as f64 + 1.0).min(h as f64)) as usize;
            let x_min = ((cx - 2.0 * steps[2] as f64).max(0.0)) as usize;
            let x_max = ((cx + 2.0 * steps[2] as f64 + 1.0).min(w as f64)) as usize;
            for z in z_min..z_max {
                for y in y_min..y_max {
                    for x in x_min..x_max {
                        let dz = cz - z as f64;
                        let dy = cy - y as f64;
                        let dx = cx - x as f64;
                        let dist_center = (dz * dz + dy * dy + dx * dx) * spatial_weight;
                        let i = (y * w + x) * 3;
                        let dl = lab[i] - segments[k][3];
                        let da = lab[i + 1] - segments[k][4];
                        let db = lab[i + 2] - segments[k][5];
                        let dist_color = dl * dl + da * da + db * db;
                        let d = dist_center + dist_color;
                        let pi = y * w + x;
                        if distance[pi] > d {
                            distance[pi] = d;
                            labels[pi] = (k + 1) as u32;
                            change = true;
                        }
                    }
                }
            }
        }
        if !change {
            break;
        }
        // 重算中心：均值（位置 + Lab 色）
        let mut n_elems = vec![0u64; n_centroids];
        let mut sums = vec![[0.0f64; 6]; n_centroids];
        for (idx, &labv) in labels.iter().enumerate() {
            let k = (labv - 1) as usize;
            let (x, y) = (idx % w, idx / w);
            n_elems[k] += 1;
            sums[k][1] += y as f64;
            sums[k][2] += x as f64;
            sums[k][3] += lab[idx * 3];
            sums[k][4] += lab[idx * 3 + 1];
            sums[k][5] += lab[idx * 3 + 2];
        }
        for k in 0..n_centroids {
            if n_elems[k] > 0 {
                let c = n_elems[k] as f64;
                for v in sums[k].iter_mut() {
                    *v /= c;
                }
                segments[k] = sums[k];
            } else {
                // Cython cdivision：0/0 → NaN，段保持死亡
                segments[k] = [f64::NAN; 6];
            }
        }
    }

    // 4) 连通性强制（对应 `_enforce_label_connectivity_cython`：6-连通 BFS）
    let segment_size = n as f64 / n_centroids as f64;
    let min_size = (0.5 * segment_size) as usize;
    let max_size = (3.0 * segment_size) as usize;

    let mask_label = 0u32;
    let mut connected = vec![0u32; n];
    let mut current_new_label = 1u32;
    let mut coord_list = vec![[0usize; 3]; max_size.max(1)];
    let ddz = [1i64, -1, 0, 0, 0, 0];
    let ddy = [0i64, 0, 1, -1, 0, 0];
    let ddx = [0i64, 0, 0, 0, 1, -1];

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if labels[idx] == mask_label || connected[idx] > mask_label {
                continue;
            }
            let mut adjacent = current_new_label;
            let label = labels[idx];
            connected[idx] = current_new_label;
            let mut current_segment_size = 1usize;
            let mut bfs_visited = 0usize;
            coord_list[0] = [0, y, x];
            while bfs_visited < current_segment_size && current_segment_size < max_size {
                let (cz, cy, cx) = (
                    coord_list[bfs_visited][0] as i64,
                    coord_list[bfs_visited][1] as i64,
                    coord_list[bfs_visited][2] as i64,
                );
                for i in 0..6 {
                    let zz = cz + ddz[i];
                    let yy = cy + ddy[i];
                    let xx = cx + ddx[i];
                    if xx >= 0 && yy >= 0 && zz >= 0 && xx < w as i64 && yy < h as i64 && zz < 1 {
                        let ni = (yy as usize) * w + (xx as usize);
                        if labels[ni] == label && connected[ni] == mask_label {
                            connected[ni] = current_new_label;
                            coord_list[current_segment_size] =
                                [zz as usize, yy as usize, xx as usize];
                            current_segment_size += 1;
                            if current_segment_size >= max_size {
                                break;
                            }
                        } else if connected[ni] > mask_label && connected[ni] != current_new_label {
                            adjacent = connected[ni];
                        }
                    }
                }
                bfs_visited += 1;
            }
            if current_segment_size < min_size {
                for coord in &coord_list[..current_segment_size] {
                    let (_, cy, cx) = (coord[0], coord[1], coord[2]);
                    connected[cy * w + cx] = adjacent;
                }
            } else {
                current_new_label += 1;
            }
        }
    }
    connected
}

/// skimage `regular_grid((1, h, w), n_points)` → (starts, steps)，按 (z, y, x) 序。
fn regular_grid_zyx(h: usize, w: usize, n_points: usize) -> ([usize; 3], [usize; 3]) {
    let ar_shape = [1usize, h, w];
    // unsort_dim_idxs = argsort(argsort(ar_shape))
    let mut order: Vec<usize> = (0..3).collect();
    order.sort_by_key(|&i| ar_shape[i]);
    let mut unsort = [0usize; 3];
    for (rank, &idx) in order.iter().enumerate() {
        unsort[idx] = rank;
    }
    let mut sorted_dims = ar_shape;
    sorted_dims.sort_unstable();

    let n_points = n_points as f64;
    let space_size = (h * w) as f64;
    if space_size <= n_points {
        return ([0; 3], [1; 3]); // 全部 slice(None)
    }
    let mut stepsizes = [(space_size / n_points).powf(1.0 / 3.0); 3];
    if sorted_dims.iter().any(|&d| (d as f64) < stepsizes[0]) {
        let mut space;
        for dim in 0..3 {
            stepsizes[dim] = sorted_dims[dim] as f64;
            space = sorted_dims[dim + 1..].iter().product::<usize>() as f64;
            let exp = 3 - dim - 1;
            if exp > 0 {
                let v = (space / n_points).powf(1.0 / exp as f64);
                for s in stepsizes.iter_mut().skip(dim + 1) {
                    *s = v;
                }
            }
            if (0..3).all(|i| (sorted_dims[i] as f64) >= stepsizes[i]) {
                break;
            }
        }
    }
    // starts = (stepsizes // 2).astype(int)；stepsizes = np.round(stepsizes).astype(int)
    let starts: [usize; 3] = stepsizes.map(|s| (s / 2.0).floor() as usize);
    let steps: [usize; 3] = stepsizes.map(np_round).map(|s| s as usize);
    // 按 unsort_dim_idxs 还原为 (z, y, x) 序
    let mut starts_zyx = [0usize; 3];
    let mut steps_zyx = [1usize; 3];
    for i in 0..3 {
        starts_zyx[unsort[i]] = starts[i];
        steps_zyx[unsort[i]] = steps[i];
    }
    (starts_zyx, steps_zyx)
}

/// numpy `np.round`（四舍六入五成双）。
fn np_round(x: f64) -> f64 {
    let f = x.floor();
    let d = x - f;
    if d < 0.5 {
        f
    } else if d > 0.5 {
        f + 1.0
    } else if f % 2.0 == 0.0 {
        f
    } else {
        f + 1.0
    }
}

// ---------------------------------------------------------------- 主色直方图

/// 每超像素均值色（int 截断，对应 Python `int(region.mean(axis=0))`）→
/// 按均值色聚合 → count 降序（稳定）→ 取前 `colors` 项。
fn dominant_colors(img: &RgbImage, labels: &[u32], colors: u32) -> Value {
    let total = f64::from(img.width * img.height);
    let mut sums: HashMap<u32, [f64; 3]> = HashMap::new();
    let mut counts: HashMap<u32, u64> = HashMap::new();
    for (idx, &lab) in labels.iter().enumerate() {
        let p = &img.data[idx * 3..idx * 3 + 3];
        let e = sums.entry(lab).or_insert([0.0; 3]);
        e[0] += f64::from(p[0]);
        e[1] += f64::from(p[1]);
        e[2] += f64::from(p[2]);
        *counts.entry(lab).or_insert(0) += 1;
    }
    let mut labels_sorted: Vec<u32> = sums.keys().copied().collect();
    labels_sorted.sort_unstable();
    let mut agg: HashMap<[u8; 3], u64> = HashMap::new();
    let mut order: Vec<[u8; 3]> = Vec::new();
    for lab in labels_sorted {
        let c = sums[&lab];
        let cnt = counts[&lab];
        let mean = [
            (c[0] / cnt as f64) as u8,
            (c[1] / cnt as f64) as u8,
            (c[2] / cnt as f64) as u8,
        ];
        let e = agg.entry(mean).or_insert(0);
        if *e == 0 {
            order.push(mean);
        }
        *e += cnt;
    }
    let mut entries: Vec<([u8; 3], u64)> = order.into_iter().map(|c| (c, agg[&c])).collect();
    entries.sort_by_key(|(_, cnt)| std::cmp::Reverse(*cnt)); // count 降序（稳定排序）
    let arr: Vec<Value> = entries
        .iter()
        .take(colors as usize)
        .map(|(rgb, cnt)| {
            json!({
                "hex": format!("#{:02X}{:02X}{:02X}", rgb[0], rgb[1], rgb[2]),
                "rgb": [rgb[0], rgb[1], rgb[2]],
                "pct": py_round_n(*cnt as f64 / total * 100.0, 1),
            })
        })
        .collect();
    json!(arr)
}

// ---------------------------------------------------------------- 亮度/饱和度

/// HSV 均值（对应 PIL `ImageStat.Stat(im.convert("HSV"))`）：
/// V = max(r,g,b)；S = (max-min)*255/max（整数除法，与 PIL 一致）。
/// 返回 round(mean)（Python `round()` 无 ndigits → int）。
fn brightness_saturation(img: &RgbImage) -> (i64, i64) {
    let n = f64::from(img.width * img.height);
    let (sv, ss) = img
        .data
        .par_chunks_exact(3)
        .fold(
            || (0.0f64, 0.0f64),
            |(sv, ss), p| {
                let (r, g, b) = (u32::from(p[0]), u32::from(p[1]), u32::from(p[2]));
                let maxc = r.max(g).max(b);
                let minc = r.min(g).min(b);
                let v = maxc;
                let s = if maxc == minc {
                    0
                } else {
                    (maxc - minc) * 255 / maxc
                };
                (sv + f64::from(v), ss + f64::from(s))
            },
        )
        .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
    (py_round_i64(sv / n), py_round_i64(ss / n))
}

// ---------------------------------------------------------------- 像素级 diff

/// 加载对比图并（必要时）LANCZOS 缩放到目标尺寸（对应 PIL `resize(Image.LANCZOS)`）。
fn load_compare(path: &Path, w: u32, h: u32) -> Result<Vec<u8>, String> {
    let img = image::open(path).map_err(|e| format!("cannot open {path:?}: {e}"))?;
    let rgb = img.to_rgb8();
    if rgb.width() == w && rgb.height() == h {
        Ok(rgb.into_raw())
    } else {
        let resized = image::imageops::resize(&rgb, w, h, image::imageops::FilterType::Lanczos3);
        Ok(resized.into_raw())
    }
}

/// diff 掩码的 8-连通域分析（对应 Python `connected_components`：
/// min_pixels=25, max_comps=20，光栅序 → bbox [x1,y1,x2,y2] + count）。
fn connected_components(mask: &[bool], w: usize, h: usize) -> Vec<(i64, i64, i64, i64, u32)> {
    const MIN_PIXELS: u32 = 25;
    const MAX_COMPS: usize = 20;
    let mut visited = vec![false; w * h];
    let mut comps = Vec::new();
    for start in 0..w * h {
        if !mask[start] || visited[start] {
            continue;
        }
        let mut stack = vec![start];
        visited[start] = true;
        let (mut min_x, mut min_y, mut max_x, mut max_y) = (i64::MAX, i64::MAX, 0i64, 0i64);
        let mut count = 0u32;
        while let Some(cur) = stack.pop() {
            count += 1;
            let (x, y) = (cur % w, cur / w);
            min_x = min_x.min(x as i64);
            min_y = min_y.min(y as i64);
            max_x = max_x.max(x as i64);
            max_y = max_y.max(y as i64);
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                    if nx >= 0 && ny >= 0 && nx < w as i64 && ny < h as i64 {
                        let ni = ny as usize * w + nx as usize;
                        if mask[ni] && !visited[ni] {
                            visited[ni] = true;
                            stack.push(ni);
                        }
                    }
                }
            }
        }
        if count >= MIN_PIXELS {
            comps.push((min_x, min_y, max_x, max_y, count));
            if comps.len() >= MAX_COMPS {
                break;
            }
        }
    }
    comps
}

/// 像素级 diff：逐像素差 → luma（对应 `ImageChops.difference().convert("L")`
/// 的整数 luma `(299R+587G+114B)/1000`）→ 阈值掩码 → 连通域异常。
fn pixel_diff(img: &RgbImage, compare_path: &str, threshold: u32) -> Result<Value, String> {
    let (w, h) = (img.width, img.height);
    let other = load_compare(Path::new(compare_path), w, h)?;
    let n = (w * h) as usize;
    let mask: Vec<bool> = (0..n)
        .into_par_iter()
        .map(|idx| {
            let a = &img.data[idx * 3..idx * 3 + 3];
            let b = &other[idx * 3..idx * 3 + 3];
            let dr = i32::from(a[0]) - i32::from(b[0]);
            let dg = i32::from(a[1]) - i32::from(b[1]);
            let db = i32::from(a[2]) - i32::from(b[2]);
            let luma = (299 * dr.abs() + 587 * dg.abs() + 114 * db.abs()) / 1000;
            luma > threshold as i32
        })
        .collect();
    let comps = connected_components(&mask, w as usize, h as usize);
    let anomalies: Vec<Value> = comps
        .iter()
        .map(|(x1, y1, x2, y2, count)| {
            json!({
                "type": "pixel_diff",
                "bbox": [x1, y1, x2, y2],
                "count": count,
                "threshold": threshold,
            })
        })
        .collect();
    Ok(json!(anomalies))
}

// ---------------------------------------------------------------- 区域/WCAG

/// 解析 `"x1,y1,x2,y2"` 并归一化（对应 Python `parse_region` 的 min/max）。
fn parse_region(s: &str) -> Result<(i64, i64, i64, i64), String> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 4 {
        return Err(format!("bad region: {s:?}"));
    }
    let vals: Result<Vec<i64>, _> = parts.iter().map(|p| p.trim().parse::<i64>()).collect();
    let vals = vals.map_err(|_| format!("bad region: {s:?}"))?;
    Ok((
        vals[0].min(vals[2]),
        vals[1].min(vals[3]),
        vals[0].max(vals[2]),
        vals[1].max(vals[3]),
    ))
}

// ---------------------------------------------------------------- 主入口

/// 运行 pix 传感器，输出 schema v3 报告（与 Python `vs_pix.py` 逐字段一致）。
pub fn run(path: &Path, p: &PixParams) -> Result<Value, String> {
    let img = load_rgb(path)?;
    let (w, h) = (img.width, img.height);
    if w == 0 || h == 0 {
        return Err("empty image".to_string());
    }
    let total = f64::from(w * h);

    // 1) SLIC 超像素 → 主色直方图 + 亮度/饱和度
    let n_segments = ((total / 2000.0) as u32).clamp(10, 200);
    let labels = slic(&img, n_segments);
    let (brightness, saturation) = brightness_saturation(&img);

    let mut report = json!({
        "schema": crate::schema::SCHEMA,
        "task": "pix",
        "sensors": ["pix"],
        "coordsys": "image_px",
        "source": {"type": "image", "path": path.display().to_string(), "size_px": [w, h]},
        "elements": [],
        "anomalies": [],
        "metrics": {
            "dominant_colors": dominant_colors(&img, &labels, p.colors),
            "superpixels": n_segments,
            "brightness": brightness,
            "saturation": saturation,
        },
    });

    // 2) 区域取色（中心像素）→ 同时输出为 v2 元素
    if !p.regions.is_empty() {
        let mut regions = Vec::new();
        let mut elements = Vec::new();
        for (i, s) in p.regions.iter().enumerate() {
            let (x1, y1, x2, y2) = parse_region(s)?;
            let (cx, cy) = ((x1 + x2).div_euclid(2), (y1 + y2).div_euclid(2));
            if cx < 0 || cy < 0 || cx >= i64::from(w) || cy >= i64::from(h) {
                return Err(format!("region sample out of bounds: {s:?}"));
            }
            let sample = img.pixel(cx as u32, cy as u32);
            let hex = format!("#{:02X}{:02X}{:02X}", sample[0], sample[1], sample[2]);
            regions.push(json!({
                "bbox": [x1, y1, x2, y2],
                "hex": hex,
                "rgb": [sample[0], sample[1], sample[2]],
            }));
            elements.push(json!({
                "id": i as i64,
                "type": "region",
                "bbox": [x1, y1, x2, y2],
                "text": Value::Null,
                "conf": 1.0,
                "color": {"fill": hex},
                "font": Value::Null,
                "z": Value::Null,
                "source": ["pix"],
                "coordsys": "image_px",
            }));
        }
        report["regions"] = json!(regions);
        report["elements"] = json!(elements);
    }

    // 3) WCAG 对比度
    if !p.wcag.is_empty() {
        let mut wcag = Vec::new();
        for pair in &p.wcag {
            let mut it = pair.split(',');
            let (fg_s, bg_s) = (it.next(), it.next());
            let (Some(fg_s), Some(bg_s)) = (fg_s, bg_s) else {
                return Err(format!("bad wcag pair: {pair:?}"));
            };
            let fg = parse_hex(fg_s).map_err(|e| format!("bad wcag pair {pair:?}: {e}"))?;
            let bg = parse_hex(bg_s).map_err(|e| format!("bad wcag pair {pair:?}: {e}"))?;
            let fgf = [fg[0] as f64, fg[1] as f64, fg[2] as f64];
            let bgf = [bg[0] as f64, bg[1] as f64, bg[2] as f64];
            let ratio = contrast_ratio(fgf, bgf);
            wcag.push(json!({
                "fg": rgb_to_hex(fgf),
                "bg": rgb_to_hex(bgf),
                "ratio": py_round_n(ratio, 2),
                "passes_aa": ratio >= 4.5,
                "passes_aaa": ratio >= 7.0,
            }));
        }
        report["wcag"] = json!(wcag);
    }

    // 4) 像素级 diff（阈值 + 连通域定位）
    if let Some(cmp) = &p.compare {
        report["anomalies"] = pixel_diff(&img, cmp, p.diff_threshold)?;
    }

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

    fn solid(w: u32, h: u32, rgb: [u8; 3]) -> RgbImage {
        let mut data = Vec::with_capacity((w * h) as usize * 3);
        for _ in 0..(w * h) {
            data.extend_from_slice(&rgb);
        }
        img(w, h, &data)
    }

    fn temp_png(name: &str, im: &RgbImage) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("vs_pix_{}_{}.png", std::process::id(), name));
        let buf = image::RgbImage::from_raw(im.width, im.height, im.data.clone()).unwrap();
        buf.save(&p).unwrap();
        p
    }

    fn params(compare: Option<String>) -> PixParams {
        PixParams {
            colors: 8,
            regions: vec![],
            compare,
            wcag: vec![],
            diff_threshold: 30,
        }
    }

    #[test]
    fn slic_solid_color_single_histogram_entry() {
        // 纯色图 → 1 个直方图条目（与 vs_pix.py 实测一致）
        let im = solid(16, 16, [255, 0, 0]);
        let labels = slic(&im, 10);
        let dc = dominant_colors(&im, &labels, 8);
        let arr = dc.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["hex"], "#FF0000");
        assert_eq!(arr[0]["rgb"], json!([255, 0, 0]));
        assert_eq!(arr[0]["pct"], 100.0);
    }

    #[test]
    fn dominant_colors_two_blocks() {
        // 8x1：左 6 红 + 右 2 蓝 → 红 75%，蓝 25%（与 vs_pix.py 实测一致）
        let mut data = Vec::new();
        for x in 0..8u32 {
            let c = if x < 6 { [255u8, 0, 0] } else { [0u8, 0, 255] };
            data.extend_from_slice(&c);
        }
        let im = img(8, 1, &data);
        let labels = slic(&im, 10);
        let dc = dominant_colors(&im, &labels, 8);
        let arr = dc.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["hex"], "#FF0000");
        assert_eq!(arr[0]["pct"], 75.0);
        assert_eq!(arr[1]["hex"], "#0000FF");
        assert_eq!(arr[1]["pct"], 25.0);
    }

    #[test]
    fn brightness_saturation_values() {
        // 纯红 → V=255, S=255；纯灰 → V=128, S=0；纯黑 → 0,0
        assert_eq!(brightness_saturation(&solid(4, 4, [255, 0, 0])), (255, 255));
        assert_eq!(
            brightness_saturation(&solid(4, 4, [128, 128, 128])),
            (128, 0)
        );
        assert_eq!(brightness_saturation(&solid(4, 4, [0, 0, 0])), (0, 0));
    }

    #[test]
    fn slic_deterministic() {
        let mut data = Vec::new();
        for y in 0..32u32 {
            for x in 0..32u32 {
                data.extend_from_slice(&[
                    ((x * 8) % 256) as u8,
                    ((y * 8) % 256) as u8,
                    (((x + y) * 4) % 256) as u8,
                ]);
            }
        }
        let im = img(32, 32, &data);
        assert_eq!(slic(&im, 10), slic(&im, 10));
    }

    #[test]
    fn report_matches_python_contract() {
        // 与 vs_pix.py 实测输出逐字段一致（solid 8x8 红）
        let im = solid(8, 8, [255, 0, 0]);
        let p = temp_png("solid", &im);
        let report = run(&p, &params(None)).unwrap();
        assert_eq!(report["schema"], "vision-report/v3");
        assert_eq!(report["task"], "pix");
        assert_eq!(report["sensors"], json!(["pix"]));
        assert_eq!(report["coordsys"], "image_px");
        assert_eq!(report["source"]["type"], "image");
        assert_eq!(report["source"]["size_px"], json!([8, 8]));
        assert_eq!(report["elements"], json!([]));
        assert_eq!(report["anomalies"], json!([]));
        assert_eq!(report["metrics"]["superpixels"], 10);
        assert_eq!(report["metrics"]["brightness"], 255);
        assert_eq!(report["metrics"]["saturation"], 255);
        assert_eq!(report["metrics"]["dominant_colors"][0]["pct"], 100.0);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn identical_images_no_diff() {
        // 相同图对比 → 无异常
        let im = solid(20, 20, [200, 100, 50]);
        let p1 = temp_png("id_a", &im);
        let p2 = temp_png("id_b", &im);
        let report = run(&p1, &params(Some(p2.to_str().unwrap().to_string()))).unwrap();
        assert_eq!(report["anomalies"], json!([]));
        let _ = std::fs::remove_file(&p1);
        let _ = std::fs::remove_file(&p2);
    }

    #[test]
    fn diff_detects_block() {
        // 20x20 白图 vs 白图 + (2,2)-(6,6) 5x5 黑块 → 1 个 anomaly（与 vs_pix.py 实测一致）
        let a = solid(20, 20, [255, 255, 255]);
        let mut b = solid(20, 20, [255, 255, 255]);
        for y in 2..7u32 {
            for x in 2..7u32 {
                let i = ((y * 20 + x) * 3) as usize;
                b.data[i] = 0;
                b.data[i + 1] = 0;
                b.data[i + 2] = 0;
            }
        }
        let p1 = temp_png("diff_a", &a);
        let p2 = temp_png("diff_b", &b);
        let report = run(&p1, &params(Some(p2.to_str().unwrap().to_string()))).unwrap();
        let anom = report["anomalies"].as_array().unwrap();
        assert_eq!(anom.len(), 1);
        assert_eq!(anom[0]["type"], "pixel_diff");
        assert_eq!(anom[0]["bbox"], json!([2, 2, 6, 6]));
        assert_eq!(anom[0]["count"], 25);
        assert_eq!(anom[0]["threshold"], 30);
        let _ = std::fs::remove_file(&p1);
        let _ = std::fs::remove_file(&p2);
    }

    #[test]
    fn regions_and_wcag() {
        // 区域取色 + WCAG（与 vs_pix.py 实测一致）
        let im = solid(8, 8, [255, 0, 0]);
        let p = temp_png("rw", &im);
        let mut pr = params(None);
        pr.regions = vec!["2,2,7,7".to_string()];
        pr.wcag = vec!["000000,FFFFFF".to_string()];
        let report = run(&p, &pr).unwrap();
        let regs = report["regions"].as_array().unwrap();
        assert_eq!(regs.len(), 1);
        assert_eq!(regs[0]["bbox"], json!([2, 2, 7, 7]));
        assert_eq!(regs[0]["hex"], "#FF0000");
        assert_eq!(regs[0]["rgb"], json!([255, 0, 0]));
        let el = &report["elements"][0];
        assert_eq!(el["id"], 0);
        assert_eq!(el["type"], "region");
        assert_eq!(el["bbox"], json!([2, 2, 7, 7]));
        assert!(el["text"].is_null());
        assert_eq!(el["conf"], 1.0);
        assert_eq!(el["color"], json!({"fill": "#FF0000"}));
        assert!(el["font"].is_null());
        assert!(el["z"].is_null());
        assert_eq!(el["source"], json!(["pix"]));
        assert_eq!(el["coordsys"], "image_px");
        let w = report["wcag"].as_array().unwrap();
        assert_eq!(w.len(), 1);
        assert_eq!(w[0]["fg"], "#000000");
        assert_eq!(w[0]["bg"], "#FFFFFF");
        assert_eq!(w[0]["ratio"], 21.0);
        assert_eq!(w[0]["passes_aa"], true);
        assert_eq!(w[0]["passes_aaa"], true);
        let _ = std::fs::remove_file(&p);
    }
}
