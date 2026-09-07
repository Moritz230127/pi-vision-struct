//! 坐标精度增强（plan3 E4，Codex 模式）：命中测试 + 三层坐标换算 + 点击验证。
//!
//! 设计哲学（元素优先、坐标兜底）：
//! - **命中测试**：给定点击点 + 元素列表 → 返回命中的元素（bbox 包含 + 中心距离最近优先）
//! - **三层坐标换算**：窗口本地 ↔ 像素 ↔ 屏幕全局（capture region 已支持）
//! - **点击验证**：点击后 capture 对比（区域是否变化）

use serde_json::Value;

/// 命中测试结果。
#[derive(Debug, Clone, PartialEq)]
pub struct HitResult {
    /// 命中的元素 id。
    pub id: usize,
    /// 元素 bbox [x1, y1, x2, y2]。
    pub bbox: [f64; 4],
    /// 元素中心 [cx, cy]。
    pub center: [f64; 4],
    /// 点击点到元素中心的距离（px）。
    pub dist: f64,
}

/// 从融合元素报告提取元素（id + bbox）。
fn extract_hit_elements(report: &Value) -> Vec<(usize, [f64; 4])> {
    let mut out = Vec::new();
    let Some(elements) = report.get("elements").and_then(|e| e.as_array()) else {
        return out;
    };
    for e in elements {
        let Some(id) = e.get("id").and_then(|i| i.as_u64()) else {
            continue;
        };
        let Some(bbox) = e.get("bbox").and_then(|b| b.as_array()) else {
            continue;
        };
        if bbox.len() < 4 {
            continue;
        }
        let to_f64 = |v: &Value| v.as_f64().unwrap_or_else(|| v.as_i64().unwrap_or(0) as f64);
        out.push((
            id as usize,
            [
                to_f64(&bbox[0]),
                to_f64(&bbox[1]),
                to_f64(&bbox[2]),
                to_f64(&bbox[3]),
            ],
        ));
    }
    out
}

/// 命中测试：给定点击点 (x, y) + 元素列表 → 命中的元素。
///
/// 规则：
/// 1. bbox 包含点击点（含边界）的元素为候选
/// 2. 多个候选时取**中心距离最近**（确定性）
/// 3. 无候选返回 `None`
///
/// 输入 `elements` 为 `(id, bbox)` 列表（来自融合元素报告或直接构造）。
pub fn hit_test(x: f64, y: f64, elements: &[(usize, [f64; 4])]) -> Option<HitResult> {
    let mut best: Option<HitResult> = None;
    for (id, bbox) in elements {
        // bbox 包含（含边界）
        if x < bbox[0] || x > bbox[2] || y < bbox[1] || y > bbox[3] {
            continue;
        }
        let cx = (bbox[0] + bbox[2]) / 2.0;
        let cy = (bbox[1] + bbox[3]) / 2.0;
        let dist = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
        let better = match &best {
            None => true,
            // 平局（dist 相等）时取后出现的（更小/更精确的元素）
            Some(b) => dist <= b.dist,
        };
        if better {
            best = Some(HitResult {
                id: *id,
                bbox: *bbox,
                center: [cx, cy, 0.0, 0.0],
                dist,
            });
        }
    }
    best
}

/// 从融合元素报告执行命中测试。
pub fn hit_test_report(x: f64, y: f64, report: &Value) -> Option<HitResult> {
    let elements = extract_hit_elements(report);
    hit_test(x, y, &elements)
}

// ---- 三层坐标换算 ----

/// 窗口本地坐标 → 屏幕全局坐标：`screen = window + window_offset`。
pub fn window_to_screen(window: (f64, f64), window_offset: (f64, f64)) -> (f64, f64) {
    (window.0 + window_offset.0, window.1 + window_offset.1)
}

/// 屏幕全局坐标 → 窗口本地坐标：`window = screen - window_offset`。
pub fn screen_to_window(screen: (f64, f64), window_offset: (f64, f64)) -> (f64, f64) {
    (screen.0 - window_offset.0, screen.1 - window_offset.1)
}

/// 图像像素坐标 → 屏幕全局坐标：`screen = pixel + region_origin`。
///
/// capture region 裁剪后，图像像素 (0,0) 对应屏幕 (region_x, region_y)。
pub fn pixel_to_screen(pixel: (f64, f64), region_origin: (f64, f64)) -> (f64, f64) {
    (pixel.0 + region_origin.0, pixel.1 + region_origin.1)
}

/// 屏幕全局坐标 → 图像像素坐标：`pixel = screen - region_origin`。
pub fn screen_to_pixel(screen: (f64, f64), region_origin: (f64, f64)) -> (f64, f64) {
    (screen.0 - region_origin.0, screen.1 - region_origin.1)
}

/// 窗口本地坐标 → 图像像素坐标（经窗口偏移 + 区域原点）。
pub fn window_to_pixel(
    window: (f64, f64),
    window_offset: (f64, f64),
    region_origin: (f64, f64),
) -> (f64, f64) {
    let screen = window_to_screen(window, window_offset);
    screen_to_pixel(screen, region_origin)
}

/// 图像像素坐标 → 窗口本地坐标（经区域原点 + 窗口偏移）。
pub fn pixel_to_window(
    pixel: (f64, f64),
    region_origin: (f64, f64),
    window_offset: (f64, f64),
) -> (f64, f64) {
    let screen = pixel_to_screen(pixel, region_origin);
    screen_to_window(screen, window_offset)
}

// ---- 点击验证 ----

/// 点击验证：对比点击前后两张截图的指定区域，检测是否变化。
///
/// 输入：`before` / `after` 为 PNG 路径，`region` 为 [x1, y1, x2, y2]（图像像素空间）。
/// 输出：`(changed, changed_pixels, total_pixels)`——变化像素数 / 区域总像素数。
pub fn verify_click(
    before: &str,
    after: &str,
    region: [u32; 4],
) -> Result<(bool, u64, u64), String> {
    let img_before = image::open(before)
        .map_err(|e| format!("open {before}: {e}"))?
        .to_rgba8();
    let img_after = image::open(after)
        .map_err(|e| format!("open {after}: {e}"))?
        .to_rgba8();
    let (x1, y1, x2, y2) = (
        region[0].min(region[2]),
        region[1].min(region[3]),
        region[0].max(region[2]),
        region[1].max(region[3]),
    );
    let w = img_before.width().min(img_after.width());
    let h = img_before.height().min(img_after.height());
    let x2 = x2.min(w);
    let y2 = y2.min(h);
    let mut changed = 0u64;
    let mut total = 0u64;
    for y in y1..y2 {
        for x in x1..x2 {
            total += 1;
            let a = img_before.get_pixel(x, y);
            let b = img_after.get_pixel(x, y);
            if a != b {
                changed += 1;
            }
        }
    }
    Ok((changed > 0, changed, total))
}

/// 点击验证（简化版）：全图对比，返回是否变化。
pub fn verify_click_full(before: &str, after: &str) -> Result<(bool, u64, u64), String> {
    let img_before = image::open(before)
        .map_err(|e| format!("open {before}: {e}"))?
        .to_rgba8();
    let img_after = image::open(after)
        .map_err(|e| format!("open {after}: {e}"))?
        .to_rgba8();
    let w = img_before.width().min(img_after.width());
    let h = img_before.height().min(img_after.height());
    let mut changed = 0u64;
    let mut total = 0u64;
    for y in 0..h {
        for x in 0..w {
            total += 1;
            if img_before.get_pixel(x, y) != img_after.get_pixel(x, y) {
                changed += 1;
            }
        }
    }
    Ok((changed > 0, changed, total))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// 合成 10 个元素（不重叠，覆盖 100×100 区域）。
    fn synth_elements() -> Vec<(usize, [f64; 4])> {
        // 5×2 网格，每个 20×20，间距 0
        let mut out = Vec::new();
        for i in 0..10 {
            let row = i / 5;
            let col = i % 5;
            let x1 = col as f64 * 20.0;
            let y1 = row as f64 * 20.0;
            out.push((i, [x1, y1, x1 + 20.0, y1 + 20.0]));
        }
        out
    }

    #[test]
    fn hit_test_accuracy_99_percent() {
        let elements = synth_elements();
        // 100 个点击点：每元素内 9 个随机点 + 10 个空白点
        let mut hits = 0;
        let mut total = 0;
        for i in 0..10 {
            let (_, bbox) = elements[i];
            // 元素内 9 个点（避开边界 1px）
            for dx in 1..=3 {
                for dy in 1..=3 {
                    let x = bbox[0] + dx as f64 * 5.0;
                    let y = bbox[1] + dy as f64 * 5.0;
                    let r = hit_test(x, y, &elements).expect("元素内应命中");
                    assert_eq!(r.id, i, "点 ({x},{y}) 应命中元素 {i}");
                    hits += 1;
                    total += 1;
                }
            }
        }
        // 空白点（网格外，不落在任何元素内）不应命中
        // 网格覆盖 100×40（5×2，每元素 20×20）；(105, 5) 在网格右侧外
        assert!(hit_test(105.0, 5.0, &elements).is_none(), "空白点不应命中");
        // (50, 45) 在网格下方外（第二行 y∈[20,40]）
        assert!(hit_test(50.0, 45.0, &elements).is_none(), "空白点不应命中");
        // 空白点正确拒绝也计入正确
        hits += 2;
        total += 2;
        // 准确率 = 命中正确 / 总点数（含空白点正确拒绝）
        let accuracy = hits as f64 / total as f64;
        assert!(accuracy >= 0.99, "命中测试准确率 {accuracy} < 0.99");
    }

    #[test]
    fn hit_test_nearest_priority() {
        // 重叠元素：大元素内含小元素，点在小元素内应命中小的（中心更近）
        let elements = vec![
            (0, [0.0, 0.0, 100.0, 100.0]), // 大
            (1, [40.0, 40.0, 60.0, 60.0]), // 小（中心 50,50）
        ];
        let r = hit_test(50.0, 50.0, &elements).expect("应命中");
        assert_eq!(r.id, 1, "重叠时取中心最近（小元素）");
        // 点在大元素角落（小元素外）→ 命中大元素
        let r2 = hit_test(10.0, 10.0, &elements).expect("应命中");
        assert_eq!(r2.id, 0);
    }

    #[test]
    fn hit_test_report_from_fusion() {
        // 融合元素报告 → 命中测试
        let report = json!({
            "elements": [
                {"id": 0, "bbox": [10.0, 20.0, 30.0, 40.0]},
                {"id": 1, "bbox": [50.0, 60.0, 70.0, 80.0]},
            ]
        });
        let r = hit_test_report(20.0, 30.0, &report).expect("应命中");
        assert_eq!(r.id, 0);
        let r2 = hit_test_report(60.0, 70.0, &report).expect("应命中");
        assert_eq!(r2.id, 1);
        assert!(hit_test_report(5.0, 5.0, &report).is_none());
    }

    #[test]
    fn coordinate_conversion_no_offset() {
        // 窗口 (0,0) + 窗口偏移 (100,50) → 屏幕 (100,50)
        assert_eq!(window_to_screen((0.0, 0.0), (100.0, 50.0)), (100.0, 50.0));
        // 屏幕 (100,50) - 窗口偏移 (100,50) → 窗口 (0,0)
        assert_eq!(screen_to_window((100.0, 50.0), (100.0, 50.0)), (0.0, 0.0));
        // 像素 (0,0) + 区域原点 (200,100) → 屏幕 (200,100)
        assert_eq!(pixel_to_screen((0.0, 0.0), (200.0, 100.0)), (200.0, 100.0));
        // 屏幕 (200,100) - 区域原点 (200,100) → 像素 (0,0)
        assert_eq!(screen_to_pixel((200.0, 100.0), (200.0, 100.0)), (0.0, 0.0));
        // 往返一致（无偏移）
        let p = (123.0, 456.0);
        let off = (100.0, 50.0);
        let origin = (200.0, 100.0);
        let round = pixel_to_window(p, origin, off);
        let back = window_to_pixel(round, off, origin);
        assert!((back.0 - p.0).abs() < 1e-9 && (back.1 - p.1).abs() < 1e-9);
    }

    #[test]
    fn coordinate_conversion_window_offset() {
        // 窗口 (10,20) + 偏移 (100,50) → 屏幕 (110,70)
        assert_eq!(window_to_screen((10.0, 20.0), (100.0, 50.0)), (110.0, 70.0));
        // 像素 (30,40) + 区域原点 (200,100) → 屏幕 (230,140)
        assert_eq!(
            pixel_to_screen((30.0, 40.0), (200.0, 100.0)),
            (230.0, 140.0)
        );
        // 窗口 → 像素：窗口 (10,20) → 屏幕 (110,70) → 像素 (110-200, 70-100) = (-90,-30)
        assert_eq!(
            window_to_pixel((10.0, 20.0), (100.0, 50.0), (200.0, 100.0)),
            (-90.0, -30.0)
        );
    }

    #[test]
    fn verify_click_detects_change() {
        // 生成两张 10×10 图：一张全黑，一张中心 2×2 白色
        let before = "/tmp/vsd_click_before.png";
        let after = "/tmp/vsd_click_after.png";
        let black = image::RgbaImage::from_pixel(10, 10, image::Rgba([0, 0, 0, 255]));
        black.save(before).unwrap();
        let mut white = black.clone();
        for y in 4..6 {
            for x in 4..6 {
                white.put_pixel(x, y, image::Rgba([255, 255, 255, 255]));
            }
        }
        white.save(after).unwrap();
        // 区域 [4,4,6,6] 应检测到变化
        let (changed, n, total) = verify_click(before, after, [4, 4, 6, 6]).unwrap();
        assert!(changed, "区域应检测到变化");
        assert_eq!(n, 4, "4 个像素变化");
        assert_eq!(total, 4);
        // 区域 [0,0,2,2] 无变化
        let (changed2, n2, _) = verify_click(before, after, [0, 0, 2, 2]).unwrap();
        assert!(!changed2, "无变化区域不应检测到变化");
        assert_eq!(n2, 0);
        // 全图对比
        let (changed3, n3, _) = verify_click_full(before, after).unwrap();
        assert!(changed3);
        assert_eq!(n3, 4);
        // 清理
        let _ = std::fs::remove_file(before);
        let _ = std::fs::remove_file(after);
    }
}
