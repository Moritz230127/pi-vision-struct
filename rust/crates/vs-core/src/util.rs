//! 基础工具：Python 语义兼容的取整 + PIL 精确 resize 内核。

use crate::image::RgbImage;

/// Python `round(x)` 语义：**银行家舍入**（round half to even）。
///
/// 与 Rust `f64::round()`（half away from zero）不同：
/// `round(2.5) = 2`、`round(3.5) = 4`。schema v3 输出契约要求与
/// Python 版逐字段一致，故所有坐标/颜色取整必须走此函数。
pub fn py_round(v: f64) -> f64 {
    let floor = v.floor();
    let frac = v - floor;
    if frac < 0.5 {
        floor
    } else if frac > 0.5 {
        floor + 1.0
    } else if floor % 2.0 == 0.0 {
        floor
    } else {
        floor + 1.0
    }
}

/// [`py_round`] 的整数版：返回值必为整数值，`as i64` 截断无损。
///
/// 坐标/颜色取整的统一出口，避免调用侧重复 `py_round(v) as i64`。
pub fn py_round_i64(v: f64) -> i64 {
    py_round(v) as i64
}

/// Python `round(x, n)` 语义：对二进制浮点的**精确十进制展开**做 n 位舍入（half-to-even）。
///
/// 不能简单用 `round(x * 10^n) / 10^n`：当 `x * 10^n` 恰好落在 .5 边界时结果可能差 1 ulp
/// （如 `round(2.675, 2)`：精确展开 2.67499... → 2.67，而 `2.675*100 = 267.5` → 2.68）。
///
/// 实现：`x = m × 2^e`（精确分解），`x × 10^n = m × 2^(e+n) × 5^n` 为有理数，
/// 精确比较余数与半值（分母恒为 2 的幂，无浮点误差）。
pub fn py_round_n(v: f64, n: i32) -> f64 {
    if !v.is_finite() {
        return v;
    }
    if n <= 0 {
        return py_round(v);
    }
    let neg = v < 0.0;
    let a = v.abs();
    let bits = a.to_bits();
    // 分解 a = m × 2^e（m 为 53 位尾数，e 为最低有效位指数）
    let (m, e) = if bits >> 52 == 0 {
        (bits & 0xfffffffffffff, -1022 - 52) // 次正规数
    } else {
        (
            (bits & 0xfffffffffffff) | 0x10000000000000,
            ((bits >> 52) & 0x7ff) as i32 - 1023 - 52,
        )
    };
    // a × 10^n = m × 2^(e+n) × 5^n
    let k = e + n;
    let pow5 = 5i128.pow(n as u32);
    let scaled = m as i128 * pow5;
    let r = if k >= 0 {
        // a × 10^n 恰为整数 → a 精确到 ≤n 位小数，无需舍入
        a
    } else if -k >= 128 {
        // den = 2^-k 超出 i128；scaled ≤ 2^53×5^22 < 2^105 < den/2 → 必舍为 0
        0.0
    } else {
        let den = 1i128 << (-k); // 2^-(e+n)
        let q = scaled / den;
        let rem = scaled % den;
        let q = match rem.cmp(&(den / 2)) {
            std::cmp::Ordering::Less => q,
            std::cmp::Ordering::Greater => q + 1,
            std::cmp::Ordering::Equal => {
                if q % 2 == 0 {
                    q
                } else {
                    q + 1 // half-to-even
                }
            }
        };
        q as f64 / 10f64.powi(n)
    };
    if neg {
        -r
    } else {
        r
    }
}

/// PIL `Image.resize(..., Image.LANCZOS)` 精确复刻：两遍分离 + 定点量化。
///
/// 与 PIL `Resample.c` 逐细节一致：
/// - 内核 `lanczos(x) = sinc(x)·sinc(x/3)`（|x|<3，否则 0）
/// - `center = (xx + 0.5)·scale`（无 -0.5）；权重参数 `(x+xmin-center+0.5)/filterscale`
/// - `filterscale = max(scale, 1.0)`；`support = 3.0·filterscale`；窗口四舍五入
/// - 权重归一化 → 定点量化（×2^22，round-half-away）→ INT32 累加（偏置 2^21）→ clip8
/// - 两遍分离：先水平到临时图，再垂直
///
/// 返回 RGB8 交错数据（`out_w × out_h × 3`）。
pub fn resize_lanczos(img: &RgbImage, out_w: u32, out_h: u32) -> Vec<u8> {
    pil_resize(img, out_w, out_h, PilFilter::Lanczos)
}

/// PIL `Image.resize(..., Image.BILINEAR)` 精确复刻（triangle 内核，同两遍分离机制）。
pub fn resize_bilinear(img: &RgbImage, out_w: u32, out_h: u32) -> Vec<u8> {
    pil_resize(img, out_w, out_h, PilFilter::Bilinear)
}

#[derive(Clone, Copy)]
enum PilFilter {
    Lanczos,
    Bilinear,
}

/// PIL 内核：lanczos（3-lobe）与 triangle（双线性）。
fn kernel(f: PilFilter, x: f64) -> f64 {
    match f {
        PilFilter::Lanczos => {
            if x == 0.0 {
                return 1.0;
            }
            if x < -3.0 || x >= 3.0 {
                return 0.0;
            }
            let px = x * std::f64::consts::PI;
            let pxr = px / 3.0;
            (px.sin() / px) * (pxr.sin() / pxr)
        }
        PilFilter::Bilinear => {
            if x < -1.0 || x > 1.0 {
                0.0
            } else {
                1.0 - x.abs()
            }
        }
    }
}

/// PIL `precompute_coeffs`：每输出像素的窗口边界 + 归一化定点权重。
/// 返回 (bounds[out*2], kk[out*ksize], ksize)。
fn precompute_coeffs(
    in_size: usize,
    out_size: usize,
    f: PilFilter,
) -> (Vec<i32>, Vec<i64>, usize) {
    const PRECISION_BITS: i32 = 22;
    let (support_base, _) = match f {
        PilFilter::Lanczos => (3.0, ()),
        PilFilter::Bilinear => (1.0, ()),
    };
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = support_base * filterscale;
    let ksize = (support.ceil() as usize) * 2 + 1;
    let inv_filterscale = 1.0 / filterscale;
    let mut bounds = vec![0i32; out_size * 2];
    let mut kk = vec![0i64; out_size * ksize];
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let mut xmin = (center - support + 0.5) as i64; // C (int) 截断
        if xmin < 0 {
            xmin = 0;
        }
        let mut xmax = (center + support + 0.5) as i64;
        if xmax > in_size as i64 {
            xmax = in_size as i64;
        }
        xmax -= xmin;
        let mut ww = 0.0f64;
        let mut k = vec![0.0f64; xmax as usize];
        for (x, kv) in k.iter_mut().enumerate() {
            let w = kernel(f, (x as f64 + xmin as f64 - center + 0.5) * inv_filterscale);
            *kv = w;
            ww += w;
        }
        if ww != 0.0 {
            for kv in k.iter_mut() {
                *kv /= ww;
            }
        }
        let scale_fixed = (1i64 << PRECISION_BITS) as f64;
        for x in 0..xmax as usize {
            kk[xx * ksize + x] = if k[x] < 0.0 {
                (-0.5 + k[x] * scale_fixed) as i64
            } else {
                (0.5 + k[x] * scale_fixed) as i64
            };
        }
        bounds[xx * 2] = xmin as i32;
        bounds[xx * 2 + 1] = xmax as i32;
    }
    (bounds, kk, ksize)
}

/// PIL `clip8`：`clip8_lookups[in >> 22]`（<0 → 0，>255 → 255）。
fn clip8(v: i64) -> u8 {
    let idx = (v >> 22) as i32;
    if idx <= 0 {
        0
    } else if idx >= 255 {
        255
    } else {
        idx as u8
    }
}

/// PIL 两遍分离 resize（水平 → 垂直），RGB8 交错输出。
fn pil_resize(img: &RgbImage, out_w: u32, out_h: u32, f: PilFilter) -> Vec<u8> {
    const PRECISION_BITS: i32 = 22;
    let (in_w, in_h) = (img.width as usize, img.height as usize);
    let (ow, oh) = (out_w as usize, out_h as usize);
    // 水平 pass → 临时图（ow × in_h）
    let (bounds_h, kk_h, ksize_h) = precompute_coeffs(in_w, ow, f);
    let mut temp = vec![0u8; ow * in_h * 3];
    for yy in 0..in_h {
        let line_in = &img.data[yy * in_w * 3..(yy + 1) * in_w * 3];
        for xx in 0..ow {
            let xmin = bounds_h[xx * 2] as usize;
            let xmax = bounds_h[xx * 2 + 1] as usize;
            let k = &kk_h[xx * ksize_h..(xx + 1) * ksize_h];
            let mut ss = [1i64 << (PRECISION_BITS - 1); 3];
            for x in 0..xmax {
                let px = &line_in[(x + xmin) * 3..(x + xmin) * 3 + 3];
                ss[0] += i64::from(px[0]) * k[x];
                ss[1] += i64::from(px[1]) * k[x];
                ss[2] += i64::from(px[2]) * k[x];
            }
            let o = (yy * ow + xx) * 3;
            temp[o] = clip8(ss[0]);
            temp[o + 1] = clip8(ss[1]);
            temp[o + 2] = clip8(ss[2]);
        }
    }
    // 垂直 pass → 输出（ow × oh）
    let (bounds_v, kk_v, ksize_v) = precompute_coeffs(in_h, oh, f);
    let mut out = vec![0u8; ow * oh * 3];
    for yy in 0..oh {
        let ymin = bounds_v[yy * 2] as usize;
        let ymax = bounds_v[yy * 2 + 1] as usize;
        let k = &kk_v[yy * ksize_v..(yy + 1) * ksize_v];
        for xx in 0..ow {
            let mut ss = [1i64 << (PRECISION_BITS - 1); 3];
            for (y, kv) in k.iter().enumerate().take(ymax) {
                let o = ((y + ymin) * ow + xx) * 3;
                ss[0] += i64::from(temp[o]) * kv;
                ss[1] += i64::from(temp[o + 1]) * kv;
                ss[2] += i64::from(temp[o + 2]) * kv;
            }
            let o = (yy * ow + xx) * 3;
            out[o] = clip8(ss[0]);
            out[o + 1] = clip8(ss[1]);
            out[o + 2] = clip8(ss[2]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::py_round;
    use crate::image::RgbImage;

    /// 5×4 渐变图（与 Python golden 同源）。
    fn grad_5x4() -> RgbImage {
        let mut data = Vec::new();
        for y in 0..4u32 {
            for x in 0..5u32 {
                data.push((x * 50 % 256) as u8);
                data.push((y * 60 % 256) as u8);
                data.push(((x + y) * 30 % 256) as u8);
            }
        }
        RgbImage { width: 5, height: 4, data }
    }

    #[test]
    fn lanczos_matches_pil() {
        use super::resize_lanczos;
        // PIL golden：5×4 → 8×7 LANCZOS（首行 + 末行抽样）
        let out = resize_lanczos(&grad_5x4(), 8, 7);
        // 首行：[(0,0,0),(15,0,7),(54,0,30),(87,0,50),(113,0,66),(146,0,86),(185,0,109),(204,0,120)]
        let row0: Vec<[u8; 3]> = (0..8).map(|x| [out[x * 3], out[x * 3 + 1], out[x * 3 + 2]]).collect();
        let expect0 = [
            [0, 0, 0], [15, 0, 7], [54, 0, 30], [87, 0, 50],
            [113, 0, 66], [146, 0, 86], [185, 0, 109], [204, 0, 120],
        ];
        assert_eq!(row0, expect0, "LANCZOS 首行与 PIL 不一致");
        // 末行（y=6）：[(0,185,90),(15,185,101),...]
        let row6: Vec<[u8; 3]> = (0..8)
            .map(|x| {
                let i = (6 * 8 + x) * 3;
                [out[i], out[i + 1], out[i + 2]]
            })
            .collect();
        let expect6 = [
            [0, 185, 90], [15, 185, 101], [54, 185, 124], [87, 185, 144],
            [113, 185, 160], [146, 185, 180], [185, 185, 203], [204, 185, 214],
        ];
        assert_eq!(row6, expect6, "LANCZOS 末行与 PIL 不一致");
        // 放大 2×：5×4 → 10×8
        let out2 = resize_lanczos(&grad_5x4(), 10, 8);
        let p = |x: u32, y: u32| -> [u8; 3] {
            let i = (y * 10 + x) * 3;
            [out2[i as usize], out2[i as usize + 1], out2[i as usize + 2]]
        };
        assert_eq!(p(0, 0), [0, 0, 0]);
        assert_eq!(p(9, 0), [205, 0, 120]);
        assert_eq!(p(4, 3), [90, 77, 92]);
        assert_eq!(p(4, 4), [90, 103, 106]);
        assert_eq!(p(9, 7), [205, 186, 216]);
    }

    #[test]
    fn bilinear_matches_pil() {
        use super::resize_bilinear;
        // PIL golden：5×4 → 8×7 BILINEAR（首行 + 末行抽样）
        let out = resize_bilinear(&grad_5x4(), 8, 7);
        let row0: Vec<[u8; 3]> = (0..8).map(|x| [out[x * 3], out[x * 3 + 1], out[x * 3 + 2]]).collect();
        let expect0 = [
            [0, 0, 0], [22, 0, 13], [53, 0, 32], [84, 0, 51],
            [116, 0, 69], [147, 0, 88], [178, 0, 107], [200, 0, 120],
        ];
        assert_eq!(row0, expect0, "BILINEAR 首行与 PIL 不一致");
        let row6: Vec<[u8; 3]> = (0..8)
            .map(|x| {
                let i = (6 * 8 + x) * 3;
                [out[i], out[i + 1], out[i + 2]]
            })
            .collect();
        let expect6 = [
            [0, 180, 90], [22, 180, 103], [53, 180, 122], [84, 180, 141],
            [116, 180, 159], [147, 180, 178], [178, 180, 197], [200, 180, 210],
        ];
        assert_eq!(row6, expect6, "BILINEAR 末行与 PIL 不一致");
    }

    #[test]
    fn matches_python_round() {
        // Python: round(0.5)=0, round(1.5)=2, round(2.5)=2, round(3.5)=4
        assert_eq!(py_round(0.5), 0.0);
        assert_eq!(py_round(1.5), 2.0);
        assert_eq!(py_round(2.5), 2.0);
        assert_eq!(py_round(3.5), 4.0);
        // 负数同样 half-to-even
        assert_eq!(py_round(-0.5), 0.0);
        assert_eq!(py_round(-1.5), -2.0);
        assert_eq!(py_round(-2.5), -2.0);
        // 非 .5 边界
        assert_eq!(py_round(2.7), 3.0);
        assert_eq!(py_round(-2.7), -3.0);
        assert_eq!(py_round(2.0), 2.0);
    }

    #[test]
    fn matches_python_round_n() {
        use super::py_round_n;
        // 与 CPython round(x, n) 实测一致（精确十进制展开舍入）
        assert_eq!(py_round_n(2.675, 2), 2.67); // 2.67499... → 2.67（非 2.68）
        assert_eq!(py_round_n(0.125, 2), 0.12); // 精确 .5 → half-to-even
        assert_eq!(py_round_n(1.005, 2), 1.0); // 1.00499... → 1.00
        assert_eq!(py_round_n(0.065, 2), 0.07); // 0.065000000000000002... → 0.07
        assert_eq!(py_round_n(0.045, 2), 0.04); // 0.044999999999999998... → 0.04
        assert_eq!(py_round_n(2.5, 0), 2.0);
        assert_eq!(py_round_n(3.5, 0), 4.0);
        assert_eq!(py_round_n(50.0, 2), 50.0);
        assert_eq!(py_round_n(127.0, 1), 127.0);
        assert_eq!(py_round_n(0.0, 4), 0.0);
        assert_eq!(py_round_n(123.456, 2), 123.46);
        assert_eq!(py_round_n(0.1, 2), 0.1);
        assert_eq!(py_round_n(0.3, 2), 0.3);
        assert_eq!(py_round_n(0.7, 2), 0.7);
        assert_eq!(py_round_n(99.99, 2), 99.99);
        assert_eq!(py_round_n(100.01, 2), 100.01);
        assert_eq!(py_round_n(0.005, 2), 0.01); // 0.005000000000000000104... → 0.01
        assert_eq!(py_round_n(0.015, 2), 0.01); // 0.01499999999999999944... → 0.01
        assert_eq!(py_round_n(0.025, 2), 0.03); // 0.02500000000000000138... → 0.03
        assert_eq!(py_round_n(0.375, 2), 0.38); // 0.375 精确 → half-to-even → 0.38
        assert_eq!(py_round_n(0.625, 2), 0.62); // 0.625 精确 → half-to-even → 0.62
        assert_eq!(py_round_n(2.675, 4), 2.675);
        assert_eq!(py_round_n(-2.675, 2), -2.67);
        assert_eq!(py_round_n(-0.125, 2), -0.12);
    }
}
