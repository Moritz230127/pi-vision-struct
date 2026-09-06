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
}
