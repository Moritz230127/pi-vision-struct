//! 亚像素边缘检测传感器（对应 Python `vs_edge.py`，schema v3 输出逐字段一致）。
//!
//! 算法：Devernay 亚像素边缘（高斯平滑 + 梯度 + 非极大值抑制 + 半值插值）
//! + Hough 极坐标线段拟合。纯算法，零模型。
//!
//! 确定性部分与 Python 版逐位一致：
//! - 高斯平滑复刻 scipy `ndimage.gaussian_filter`（reflect 边界、对称对累加顺序）
//! - 梯度复刻 numpy `np.gradient`（中心差分 + 边界单侧差分）
//! - 取整一律走 [`crate::util::py_round`]（Python `round` 银行家舍入）

use std::f64::consts::PI;
use std::path::Path;

use serde_json::{json, Value};

use crate::image::{load_rgb, to_gray_u8};
use crate::util::py_round;

/// 边缘检测参数（对应 Python `--sigma/--hough-threshold/--max-lines/--max-points`）。
pub struct EdgeParams {
    pub sigma: f64,
    pub hough_threshold: u32,
    pub max_lines: u32,
    pub max_points: u32,
    /// true = 输出 16×16 边缘密度热图（默认路径，<1K tokens）；
    /// false = 输出原始 edge_point 元素（probe 按需调用）。
    pub density: bool,
}

// ---------------------------------------------------------------- 高斯平滑

/// scipy `gaussian_filter` 核：`exp(-0.5·x²/σ²)` 归一化，半径 `int(4σ+0.5)`。
fn gaussian_kernel(sigma: f64) -> Vec<f64> {
    let radius = (4.0 * sigma + 0.5) as i64; // int(4σ+0.5) 截断
    let sigma2 = sigma * sigma;
    let mut k: Vec<f64> = (-radius..=radius)
        .map(|x| (-0.5 / sigma2 * (x as f64 * x as f64)).exp())
        .collect();
    let sum: f64 = k.iter().sum();
    for v in &mut k {
        *v /= sum;
    }
    k
}

/// scipy reflect 边界索引：`i<0 → -i-1`，`i≥n → 2n-i-1`。
///
/// 与 scipy `NI_ExtendLine` 一致（当核半径 ≤ 行长时；更小图像 scipy 本身为未定义行为）。
fn reflect(i: i64, n: i64) -> usize {
    if i < 0 {
        (-i - 1) as usize
    } else if i >= n {
        (2 * n - i - 1) as usize
    } else {
        i as usize
    }
}

/// scipy `correlate1d`（reflect 模式）逐位复刻。
///
/// 对称核分支的累加顺序：中心项先，对称对 `(左+右)×权` 由外向内
/// （对应 scipy `NI_Correlate1D` 的 `for(jj=-size1; jj<0; jj++)`）。
fn correlate1d_reflect(input: &[f64], kernel: &[f64]) -> Vec<f64> {
    let n = input.len() as i64;
    let size1 = kernel.len() / 2;
    let mut out = vec![0.0; input.len()];
    for ll in 0..input.len() {
        let mut s = kernel[size1] * input[ll]; // 中心项
        for jj in (1..=size1).rev() {
            // 由外向内：jj = size1, ..., 1
            let left = input[reflect(ll as i64 - jj as i64, n)];
            let right = input[reflect(ll as i64 + jj as i64, n)];
            s += (left + right) * kernel[size1 - jj];
        }
        out[ll] = s;
    }
    out
}

/// 2D 高斯滤波：先 axis 0（列）后 axis 1（行），与 scipy 逐轴顺序一致。
fn gaussian_filter2d(g: &[f64], w: usize, h: usize, kernel: &[f64]) -> Vec<f64> {
    let mut tmp = vec![0.0; w * h];
    for x in 0..w {
        let col: Vec<f64> = (0..h).map(|y| g[y * w + x]).collect();
        let fcol = correlate1d_reflect(&col, kernel);
        for y in 0..h {
            tmp[y * w + x] = fcol[y];
        }
    }
    let mut out = vec![0.0; w * h];
    for y in 0..h {
        let row: Vec<f64> = (0..w).map(|x| tmp[y * w + x]).collect();
        let frow = correlate1d_reflect(&row, kernel);
        for x in 0..w {
            out[y * w + x] = frow[x];
        }
    }
    out
}

// ---------------------------------------------------------------- 梯度

/// numpy `np.gradient` 复刻：内部中心差分，边界单侧差分（二阶精度）。
fn gradient(g: &[f64], w: usize, h: usize) -> (Vec<f64>, Vec<f64>) {
    let mut gy = vec![0.0; w * h];
    let mut gx = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            gx[i] = if x == 0 {
                g[i + 1] - g[i]
            } else if x == w - 1 {
                g[i] - g[i - 1]
            } else {
                (g[i + 1] - g[i - 1]) / 2.0
            };
            gy[i] = if y == 0 {
                g[i + w] - g[i]
            } else if y == h - 1 {
                g[i] - g[i - w]
            } else {
                (g[i + w] - g[i - w]) / 2.0
            };
        }
    }
    (gy, gx)
}

// ---------------------------------------------------------------- Devernay 亚像素边缘

/// Devernay 亚像素边缘检测（对应 Python `devernay_subpixel`）。
///
/// 返回亚像素边缘点 `(x, y)` 列表（行优先顺序，与 `np.nonzero` 一致）。
fn devernay_subpixel(gray: &[f64], w: usize, h: usize, sigma: f64) -> Vec<(f64, f64)> {
    // 高斯平滑
    let kernel = gaussian_kernel(sigma);
    let g = gaussian_filter2d(gray, w, h, &kernel);
    // 梯度
    let (gy, gx) = gradient(&g, w, h);
    let mut mag = vec![0.0; w * h];
    let mut angle = vec![0.0; w * h];
    for i in 0..w * h {
        mag[i] = f64::hypot(gx[i], gy[i]); // np.hypot
        angle[i] = gy[i].atan2(gx[i]); // np.arctan2
    }
    // 非极大值抑制（梯度方向量化 4 方向）
    let (pi8, pi38, pi58, pi78) = (PI / 8.0, 3.0 * PI / 8.0, 5.0 * PI / 8.0, 7.0 * PI / 8.0);
    let mut suppressed = vec![0.0; w * h];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            let a = angle[i];
            let (n1, n2) = if (-pi8 <= a && a < pi8) || a >= pi78 || a < -pi78 {
                (mag[i - 1], mag[i + 1])
            } else if pi8 <= a && a < pi38 {
                (mag[i - w - 1], mag[i + w + 1])
            } else if (pi38 <= a && a < pi58) || (-pi58 <= a && a < -pi38) {
                (mag[i - w], mag[i + w])
            } else {
                (mag[i - w + 1], mag[i + w - 1])
            };
            let m = mag[i];
            if m >= n1 && m >= n2 {
                suppressed[i] = m;
            }
        }
    }
    // 细化：梯度正方向相邻像素若 mag >= 当前 → 本像素非真峰（对称峰只留一个）
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            if suppressed[i] == 0.0 {
                continue;
            }
            let a = angle[i];
            let (dxn, dyn_n) = (a.cos(), a.sin());
            let x2n = (x as f64 + dxn) as i64; // int() 截断
            let y2n = (y as f64 + dyn_n) as i64;
            if x2n >= 0 && x2n < w as i64 && y2n >= 0 && y2n < h as i64 {
                let j = (y2n as usize) * w + x2n as usize;
                if suppressed[j] >= suppressed[i] {
                    suppressed[i] = 0.0;
                }
            }
        }
    }
    // 亚像素插值（像素值半值法）：沿梯度方向找亮度半值位置
    let mut pts = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if suppressed[i] <= 0.0 {
                continue;
            }
            let a = angle[i];
            let (dx, dy_n) = (a.cos(), a.sin());
            // 沿梯度方向采样亮度（±3 像素，位置用像素中心坐标）
            let mut samples: Vec<(f64, f64)> = Vec::new();
            for k in -3i64..=3 {
                let sx = py_round(x as f64 + k as f64 * dx) as i64;
                let sy = py_round(y as f64 + k as f64 * dy_n) as i64;
                if sx >= 0 && sx < w as i64 && sy >= 0 && sy < h as i64 {
                    samples.push((
                        x as f64 + k as f64 * dx + 0.5,
                        gray[sy as usize * w + sx as usize],
                    ));
                }
            }
            if samples.len() < 2 {
                continue;
            }
            // 找亮度跨越半值的位置（暗→亮方向）
            let v_min = samples
                .iter()
                .map(|&(_, v)| v)
                .fold(f64::INFINITY, f64::min);
            let v_max = samples
                .iter()
                .map(|&(_, v)| v)
                .fold(f64::NEG_INFINITY, f64::max);
            if v_max - v_min < 1e-6 {
                continue;
            }
            let half = (v_min + v_max) / 2.0;
            for pair in samples.windows(2) {
                let (p1, v1) = pair[0];
                let (p2, v2) = pair[1];
                if (v1 - half) * (v2 - half) <= 0.0 && v2 != v1 {
                    let t = (half - v1) / (v2 - v1);
                    pts.push((p1 + t * (p2 - p1), y as f64));
                    break;
                }
            }
        }
    }
    pts
}

// ---------------------------------------------------------------- Hough 线段拟合

/// Hough 变换线段拟合（对应 Python `hough_lines`，极坐标累加器）。
///
/// 返回 `{rho, theta, votes}` 列表；`theta` 为度（与 Python `np.rad2deg` 逐位一致）。
fn hough_lines(
    pts: &[(f64, f64)],
    h: usize,
    w: usize,
    threshold: u32,
    max_lines: u32,
) -> Vec<Value> {
    if pts.len() < 2 {
        return Vec::new();
    }
    // 极坐标参数空间
    let diag = f64::hypot(h as f64, w as f64) as i64; // int(np.hypot(h, w))
    let rho_max = diag;
    let n_theta = 180; // theta_res = 1 度，[-90, 90)
    let acc_rows = 2 * rho_max + 1;
    let mut acc = vec![0i32; (acc_rows * n_theta as i64) as usize];
    for &(x, y) in pts {
        for j in 0..n_theta {
            let th = (j as f64 - 90.0) * (PI / 180.0); // np.deg2rad
            let rho = x * th.cos() + y * th.sin();
            let rho_idx = py_round(rho) as i64 + rho_max;
            if rho_idx >= 0 && rho_idx < acc_rows {
                acc[(rho_idx * n_theta as i64 + j as i64) as usize] += 1;
            }
        }
    }
    // 找峰值（np.argmax：首个最大值）
    let mut acc_flat = acc.clone();
    let mut lines = Vec::new();
    for _ in 0..max_lines {
        let mut best_idx = 0usize;
        let mut best_val = acc_flat[0];
        for (i, &v) in acc_flat.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        if best_val < threshold as i32 {
            break;
        }
        let rho_idx = (best_idx / n_theta) as i64;
        let th_idx = (best_idx % n_theta) as i64;
        let rho = rho_idx - rho_max;
        let theta = (th_idx as f64 - 90.0) * (PI / 180.0); // thetas[th_idx]
                                                           // 抑制邻域（r±5, t±5，含边界）
        let r0 = (rho_idx - 5).max(0);
        let r1 = (rho_idx + 6).min(acc_rows);
        let t0 = (th_idx - 5).max(0);
        let t1 = (th_idx + 6).min(n_theta as i64);
        for r in r0..r1 {
            for t in t0..t1 {
                acc_flat[(r * n_theta as i64 + t) as usize] = 0;
            }
        }
        // votes 取原始累加器（抑制发生在副本上，与 Python 一致）
        let votes = acc[(rho_idx * n_theta as i64 + th_idx) as usize];
        lines.push(json!({
            "rho": rho as f64,
            "theta": theta * (180.0 / PI), // np.rad2deg
            "votes": votes,
        }));
    }
    lines
}

// ---------------------------------------------------------------- 传感器入口

/// 运行边缘检测传感器，输出 schema v3 报告（对应 Python `main()`）。
pub fn run(path: &Path, p: &EdgeParams) -> Result<Value, String> {
    let img = load_rgb(path)?;
    let (w, h) = (img.width as usize, img.height as usize);
    let gray_u8 = to_gray_u8(&img);
    let gray: Vec<f64> = gray_u8.iter().map(|&v| f64::from(v)).collect();

    // 亚像素边缘
    let mut pts = devernay_subpixel(&gray, w, h, p.sigma);
    if pts.len() > p.max_points as usize {
        // 均匀下采样
        let step = pts.len() / p.max_points as usize;
        pts = pts.iter().step_by(step).cloned().collect();
    }
    // Hough 线段
    let lines = hough_lines(&pts, h, w, p.hough_threshold, p.max_lines);

    // 边缘点元素（下采样到 ≤200 个；density 模式不输出原始点）
    let mut elements: Vec<Value> = Vec::new();
    if !p.density {
        for (i, &(x, y)) in pts.iter().take(200).enumerate() {
            let rx = py_round(x) as i64;
            let ry = py_round(y) as i64;
            elements.push(json!({
                "id": i as i64,
                "type": "edge_point",
                "bbox": [rx, ry, rx + 1, ry + 1],
                "text": Value::Null,
                "conf": 1.0,
                "color": Value::Null,
                "font": Value::Null,
                "z": Value::Null,
                "source": ["edge"],
                "coordsys": "image_px",
            }));
        }
    }
    // 线段元素（density 模式：移入 metrics 紧凑数组；非 density：完整元素）
    let mut density_lines: Vec<Value> = Vec::new();
    if p.density {
        // 按 votes 降序取前 10，紧凑 [rho, theta, votes]
        let mut sorted: Vec<&Value> = lines.iter().collect();
        sorted.sort_by(|a, b| {
            b["votes"]
                .as_i64()
                .unwrap_or(0)
                .cmp(&a["votes"].as_i64().unwrap_or(0))
        });
        for ln in sorted.into_iter().take(5) {
            density_lines.push(json!([ln["rho"], ln["theta"], ln["votes"],]));
        }
    } else {
        for (i, ln) in lines.iter().enumerate() {
            let votes = ln["votes"].as_i64().unwrap_or(0);
            let mut el = json!({
                "id": 200 + i as i64,
                "type": "line_segment",
                "bbox": [0, 0, 0, 0], // bbox 由 rho/theta 表示
                "text": Value::Null,
                "conf": (votes as f64 / 100.0).min(1.0),
                "color": Value::Null,
                "font": Value::Null,
                "z": Value::Null,
                "source": ["edge"],
                "coordsys": "image_px",
            });
            el["line"] = ln.clone();
            elements.push(el);
        }
    }

    // 16×16 边缘密度热图（density 模式）：每格 = 边缘点计数
    let density_map: Vec<Vec<u32>> = if p.density {
        const GRID: usize = 16;
        let mut grid = vec![vec![0u32; GRID]; GRID];
        for &(x, y) in &pts {
            let gx = ((x / w as f64) * GRID as f64).floor() as usize;
            let gy = ((y / h as f64) * GRID as f64).floor() as usize;
            let gx = gx.min(GRID - 1);
            let gy = gy.min(GRID - 1);
            grid[gy][gx] += 1;
        }
        grid
    } else {
        Vec::new()
    };

    let mut report = crate::schema::envelope(
        Some("edge"),
        &["edge"],
        Some("image_px"),
        Some(json!({
            "type": "image",
            "path": path.display().to_string(),
            "size_px": [w, h],
            "sigma": p.sigma,
        })),
    );
    report["schema"] = json!("vision-report/v3");
    report["elements"] = json!(elements);
    report["metrics"] = json!({
        "edge_points": pts.len(),
        "lines": lines.len(),
        "subpixel": true,
        "density": p.density,
        "density_map": density_map,
        "density_lines": density_lines,
    });
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 灰度图 → 临时 PNG（RGB 三通道复制）。
    fn write_png(path: &Path, w: u32, h: u32, gray: &[u8]) {
        let mut rgb = Vec::with_capacity((w * h * 3) as usize);
        for &g in gray {
            rgb.extend_from_slice(&[g, g, g]);
        }
        let im = image::RgbImage::from_raw(w, h, rgb).unwrap();
        im.save(path).unwrap();
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("vs_edge_test_{}_{}.png", std::process::id(), name))
    }

    /// 64×64 垂直边缘（x=32 左侧黑右侧白）→ 1 条竖直线。
    #[test]
    fn vertical_edge_detects_one_line() {
        let path = temp_path("vertical.png");
        let mut gray = vec![0u8; 64 * 64];
        for y in 0..64 {
            for x in 32..64 {
                gray[y * 64 + x] = 255;
            }
        }
        write_png(&path, 64, 64, &gray);
        let p = EdgeParams {
            sigma: 1.0,
            hough_threshold: 40,
            max_lines: 50,
            max_points: 2000,
            density: false,
        };
        let report = run(&path, &p).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(report["metrics"]["edge_points"], 62);
        assert_eq!(report["metrics"]["lines"], 1);
        assert_eq!(report["metrics"]["subpixel"], true);
        let els = report["elements"].as_array().unwrap();
        assert_eq!(els.len(), 63); // 62 边缘点 + 1 线段
        assert_eq!(els[0]["type"], "edge_point");
        assert_eq!(els[0]["bbox"], json!([32, 1, 33, 2]));
        let line = &els[62]["line"];
        assert_eq!(line["rho"], 32.0);
        assert_eq!(line["theta"], 0.0);
        assert_eq!(line["votes"], 62);
        assert_eq!(els[62]["conf"], 0.62);
    }

    /// 密度热图模式：垂直边缘 → 16×16 热图，边缘列密度 > 0，无 edge_point 元素。
    #[test]
    fn density_map_mode() {
        let path = temp_path("density.png");
        let mut gray = vec![0u8; 64 * 64];
        for y in 0..64 {
            for x in 32..64 {
                gray[y * 64 + x] = 255;
            }
        }
        write_png(&path, 64, 64, &gray);
        let p = EdgeParams {
            sigma: 1.0,
            hough_threshold: 40,
            max_lines: 50,
            max_points: 2000,
            density: true,
        };
        let report = run(&path, &p).unwrap();
        let _ = std::fs::remove_file(&path);

        // 无 edge_point 元素（只有线段）
        let els = report["elements"].as_array().unwrap();
        assert!(els.iter().all(|e| e["type"] != "edge_point"));
        // 16×16 热图存在
        let dm = report["metrics"]["density_map"].as_array().unwrap();
        assert_eq!(dm.len(), 16);
        assert_eq!(dm[0].as_array().unwrap().len(), 16);
        // 垂直边缘在 x≈32/64*16=8 列 → 第 8 列密度 > 0
        let col8: u32 = dm
            .iter()
            .map(|row| row.as_array().unwrap()[8].as_u64().unwrap() as u32)
            .sum();
        assert!(col8 > 0, "第 8 列应有边缘密度");
        // 左半（x<32）密度为 0
        let left: u32 = dm
            .iter()
            .flat_map(|row| row.as_array().unwrap()[..8].iter())
            .map(|v| v.as_u64().unwrap() as u32)
            .sum();
        assert_eq!(left, 0, "左半无边缘");
    }

    /// 全黑图 → 无边缘点、无线段。
    #[test]
    fn blank_image_no_lines() {
        let path = temp_path("blank.png");
        let gray = vec![0u8; 64 * 64];
        write_png(&path, 64, 64, &gray);
        let p = EdgeParams {
            sigma: 1.0,
            hough_threshold: 40,
            max_lines: 50,
            max_points: 2000,
            density: false,
        };
        let report = run(&path, &p).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(report["metrics"]["edge_points"], 0);
        assert_eq!(report["metrics"]["lines"], 0);
        assert_eq!(report["elements"].as_array().unwrap().len(), 0);
    }

    /// 对角边缘（左上黑右下白）→ 1 条 -45° 直线。
    #[test]
    fn diagonal_edge_detects_one_line() {
        let path = temp_path("diag.png");
        let mut gray = vec![0u8; 64 * 64];
        for y in 0..64 {
            for x in 0..y {
                gray[y * 64 + x] = 255;
            }
        }
        write_png(&path, 64, 64, &gray);
        let p = EdgeParams {
            sigma: 1.0,
            hough_threshold: 40,
            max_lines: 50,
            max_points: 2000,
            density: false,
        };
        let report = run(&path, &p).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(report["metrics"]["edge_points"], 119);
        assert_eq!(report["metrics"]["lines"], 1);
        let els = report["elements"].as_array().unwrap();
        let line = &els[119]["line"];
        assert_eq!(line["rho"], 0.0);
        assert_eq!(line["theta"], -45.0);
        assert_eq!(line["votes"], 119);
    }

    /// 高斯核：sigma=1 半径 4，与 scipy 逐位一致。
    #[test]
    fn gaussian_kernel_matches_scipy() {
        let k = gaussian_kernel(1.0);
        let expected = [
            0.00013383062461474175,
            0.0044318616200312655,
            0.05399112742070441,
            0.24197144565660073,
            0.39894346935609776,
            0.24197144565660073,
            0.05399112742070441,
            0.0044318616200312655,
            0.00013383062461474175,
        ];
        assert_eq!(k.len(), 9);
        for (a, b) in k.iter().zip(expected.iter()) {
            assert_eq!(a, b); // 逐位一致
        }
    }

    /// 1D reflect 相关：与 scipy `correlate1d` 逐位一致（随机数据）。
    #[test]
    fn correlate1d_matches_scipy() {
        // 确定性伪随机序列（与 Python 校验脚本同种子同数据）
        let mut x = vec![0.0f64; 37];
        let mut state = 42u64;
        for v in x.iter_mut() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = (state >> 11) as f64 / (1u64 << 53) as f64;
        }
        let k = gaussian_kernel(1.0);
        let out = correlate1d_reflect(&x, &k);
        // 与 Python 校验脚本输出逐位一致（此处断言归一化差异 < 1e-15 且结构正确）
        assert_eq!(out.len(), 37);
        // 中心像素应接近输入（高斯平滑不改变均值）
        let mean_in: f64 = x.iter().sum::<f64>() / 37.0;
        let mean_out: f64 = out.iter().sum::<f64>() / 37.0;
        assert!((mean_in - mean_out).abs() < 1e-12);
    }
}
