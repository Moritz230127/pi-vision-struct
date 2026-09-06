//! 坐标系变换与 bbox 运算（对应 Python `vs_schema.py` 坐标段）。
//!
//! 坐标系（`coordsys` 字段声明）：
//! - `css_px`：CSS 像素（DOM，含 DPR 缩放前）
//! - `device_px`：物理设备像素（grim 截图）
//! - `image_px`：图像文件像素（经缩放后的图片）
//! - `pt`：印刷点（PPTX：1pt = 1/72 英寸）

use crate::util::py_round_i64;

/// CSS 像素（含滚动偏移）→ 设备像素：`device = (css - scroll) × dpr`。
pub fn css_to_device(bbox: &[f64], dpr: f64, scroll: (i64, i64)) -> [i64; 4] {
    let (sx, sy) = scroll;
    [
        py_round_i64((bbox[0] - sx as f64) * dpr),
        py_round_i64((bbox[1] - sy as f64) * dpr),
        py_round_i64((bbox[2] - sx as f64) * dpr),
        py_round_i64((bbox[3] - sy as f64) * dpr),
    ]
}

/// 设备像素 → CSS 像素（含滚动偏移）：`css = device / dpr + scroll`。
pub fn device_to_css(bbox: &[f64], dpr: f64, scroll: (i64, i64)) -> [i64; 4] {
    let (sx, sy) = scroll;
    [
        py_round_i64(bbox[0] / dpr + sx as f64),
        py_round_i64(bbox[1] / dpr + sy as f64),
        py_round_i64(bbox[2] / dpr + sx as f64),
        py_round_i64(bbox[3] / dpr + sy as f64),
    ]
}

/// 按缩放系数 `k` 变换 bbox（图像缩放 / 区域放大回原图）。
pub fn scale_bbox(bbox: &[f64], k: f64) -> [i64; 4] {
    [
        py_round_i64(bbox[0] * k),
        py_round_i64(bbox[1] * k),
        py_round_i64(bbox[2] * k),
        py_round_i64(bbox[3] * k),
    ]
}

/// 裁剪 bbox 到画布内：先取整，再钳制到 `[0, w] × [0, h]`。
pub fn clip_bbox(bbox: &[f64], w: i64, h: i64) -> [i64; 4] {
    [
        py_round_i64(bbox[0]).max(0),
        py_round_i64(bbox[1]).max(0),
        py_round_i64(bbox[2]).min(w),
        py_round_i64(bbox[3]).min(h),
    ]
}

/// bbox 面积（负宽/高按 0 计）。
pub fn bbox_area(bbox: &[f64]) -> f64 {
    (bbox[2] - bbox[0]).max(0.0) * (bbox[3] - bbox[1]).max(0.0)
}

/// 两 bbox 交集；无交集返回 `None`。
pub fn bbox_intersection(a: &[f64], b: &[f64]) -> Option<[i64; 4]> {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);
    if x2 > x1 && y2 > y1 {
        Some([
            py_round_i64(x1),
            py_round_i64(y1),
            py_round_i64(x2),
            py_round_i64(y2),
        ])
    } else {
        None
    }
}

/// 两 bbox 的 IoU（交并比）；无交集或并集为 0 时返回 0.0。
pub fn bbox_iou(a: &[f64], b: &[f64]) -> f64 {
    match bbox_intersection(a, b) {
        None => 0.0,
        Some(inter) => {
            let ia = bbox_area(&[
                inter[0] as f64,
                inter[1] as f64,
                inter[2] as f64,
                inter[3] as f64,
            ]);
            let ua = bbox_area(a) + bbox_area(b) - ia;
            if ua > 0.0 {
                ia / ua
            } else {
                0.0
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transforms() {
        assert_eq!(
            css_to_device(&[10.0, 20.0, 110.0, 220.0], 2.0, (0, 0)),
            [20, 40, 220, 440]
        );
        assert_eq!(
            css_to_device(&[10.0, 20.0, 110.0, 220.0], 2.0, (5, 5)),
            [10, 30, 210, 430]
        );
        assert_eq!(
            device_to_css(&[20.0, 40.0, 220.0, 440.0], 2.0, (0, 0)),
            [10, 20, 110, 220]
        );
        assert_eq!(scale_bbox(&[1.0, 2.0, 3.0, 4.0], 2.5), [2, 5, 8, 10]);
        assert_eq!(
            clip_bbox(&[-5.0, -5.0, 200.0, 200.0], 100, 100),
            [0, 0, 100, 100]
        );
    }

    #[test]
    fn area_intersection_iou() {
        assert_eq!(bbox_area(&[0.0, 0.0, 10.0, 10.0]), 100.0);
        assert_eq!(bbox_area(&[5.0, 5.0, 3.0, 3.0]), 0.0);
        assert_eq!(
            bbox_intersection(&[0.0, 0.0, 10.0, 10.0], &[5.0, 5.0, 15.0, 15.0]),
            Some([5, 5, 10, 10])
        );
        assert_eq!(
            bbox_intersection(&[0.0, 0.0, 1.0, 1.0], &[5.0, 5.0, 6.0, 6.0]),
            None
        );
        // 完全重叠 IoU = 1.0
        assert!((bbox_iou(&[0.0, 0.0, 10.0, 10.0], &[0.0, 0.0, 10.0, 10.0]) - 1.0).abs() < 1e-9);
        // 内嵌：10×10 内含 10×5 → 交集 50，并集 100 → 0.5
        assert!((bbox_iou(&[0.0, 0.0, 10.0, 10.0], &[0.0, 5.0, 10.0, 10.0]) - 0.5).abs() < 1e-9);
        // 侧向半重叠：10×10 与 10×10 错位 5px → 交集 50，并集 150 → 1/3
        assert!(
            (bbox_iou(&[0.0, 0.0, 10.0, 10.0], &[5.0, 0.0, 15.0, 10.0]) - 1.0 / 3.0).abs() < 1e-9
        );
        assert_eq!(bbox_iou(&[0.0, 0.0, 1.0, 1.0], &[5.0, 5.0, 6.0, 6.0]), 0.0);
    }
}
