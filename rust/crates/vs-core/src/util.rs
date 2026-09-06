//! 基础工具：Python 语义兼容的取整。

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

#[cfg(test)]
mod tests {
    use super::py_round;

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
