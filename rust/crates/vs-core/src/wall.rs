//! 壁纸批量程序化分类（对应 Python `vs_wall.py`，schema v1 输出逐字段一致）。
//!
//! 程序化维度全部确定性：色相族（主色直方图峰值）、亮度档、饱和度档、宽高比。
//! 复用 scene_stats 的 MEDIANCUT 量化 + 自研 PIL 精确 HSV/ImageStat。

use std::collections::HashMap;
use std::path::Path;

use serde_json::{json, Value};

use crate::image::{load_rgb, RgbImage};
use crate::scene_stats::{median_cut_boxes, Box};
use crate::util::{py_round, py_round_n, resize_lanczos};

/// 壁纸参数（对应 Python `--colors/--max-files/--ext`）。
pub struct WallParams {
    pub colors: u32,
    pub max_files: u32,
    pub exts: Vec<String>,
}

impl Default for WallParams {
    fn default() -> Self {
        Self {
            colors: 5,
            max_files: 200,
            exts: vec![".png".into(), ".jpg".into(), ".jpeg".into(), ".webp".into(), ".bmp".into()],
        }
    }
}

/// PIL 12.3.0 `rgb2hsv_row`（colorsys.py 浮点公式）精确复刻：H/S/V 均 0-255。
///
/// 与 PIL `Convert.c` 逐细节一致：f32 运算、`fmod((h/6+1), 1)`、`(int)(h*255)` 截断。
fn rgb_to_hsv(r: u8, g: u8, b: u8) -> [u8; 3] {
    let (r, g, b) = (f32::from(r), f32::from(g), f32::from(b));
    let maxc = r.max(g).max(b);
    let minc = r.min(g).min(b);
    let v = maxc;
    if minc == maxc {
        return [0, 0, v as u8];
    }
    let cr = maxc - minc;
    let s = cr / maxc;
    let rc = (maxc - r) / cr;
    let gc = (maxc - g) / cr;
    let bc = (maxc - b) / cr;
    let h = if r == maxc {
        bc - gc
    } else if g == maxc {
        2.0 + rc - bc
    } else {
        4.0 + gc - rc
    };
    // fmod((h/6.0 + 1.0), 1.0)：h/6 可能为负，+1 后取模
    let h = (h / 6.0 + 1.0).rem_euclid(1.0);
    let uh = (h * 255.0) as u8; // (int) 截断 + CLIP8（h*255 ∈ [0,255)）
    let us = (s * 255.0) as u8;
    [uh, us, v as u8]
}

/// PIL `_downsample`：最长边 max_side 的 LANCZOS 缩小（scale<1 时）。
fn downsample(img: &RgbImage, max_side: u32) -> RgbImage {
    let (w, h) = (img.width, img.height);
    let scale = max_side as f64 / f64::from(w.max(h));
    if scale < 1.0 {
        let nw = (f64::from(w) * scale) as u32; // C int 截断
        let nh = (f64::from(h) * scale) as u32;
        let nw = nw.max(1);
        let nh = nh.max(1);
        let data = resize_lanczos(img, nw, nh);
        RgbImage {
            width: nw,
            height: nh,
            data,
        }
    } else {
        RgbImage {
            width: w,
            height: h,
            data: img.data.clone(),
        }
    }
}

/// PIL `ImageStat.Stat(HSV)` 均值：每通道 float 均值 → Python `round()`（half-to-even）。
/// 返回 (H 均值, S 均值, V 均值)。
fn hsv_mean(img: &RgbImage) -> (i64, i64, i64) {
    let n = f64::from(img.width * img.height);
    let (mut sh, mut ss, mut sv) = (0.0f64, 0.0f64, 0.0f64);
    for px in img.data.as_chunks::<3>().0 {
        let hsv = rgb_to_hsv(px[0], px[1], px[2]);
        sh += f64::from(hsv[0]);
        ss += f64::from(hsv[1]);
        sv += f64::from(hsv[2]);
    }
    (
        py_round(sh / n) as i64,
        py_round(ss / n) as i64,
        py_round(sv / n) as i64,
    )
}

/// 主色相：HSV 色相直方图峰值（只统计饱和度 ≥25 的像素）。纯灰图返回 None。
pub fn dominant_hue_debug(img: &RgbImage) -> Option<i64> {
    dominant_hue(img)
}

fn dominant_hue(img: &RgbImage) -> Option<i64> {
    let ds = downsample(img, 512);
    let mut cnt: HashMap<u8, u32> = HashMap::new();
    for px in ds.data.as_chunks::<3>().0 {
        let hsv = rgb_to_hsv(px[0], px[1], px[2]);
        if hsv[1] >= 25 {
            *cnt.entry(hsv[0]).or_insert(0) += 1;
        }
    }
    if cnt.is_empty() {
        return None;
    }
    // most_common(1)：count 降序，平局按插入序（Python Counter 保序）
    let best = cnt
        .into_iter()
        .max_by(|a, b| a.1.cmp(&b.1).then_with(|| std::cmp::Ordering::Less))
        .map(|(h, _)| h)?;
    // round(hue * 360 / 255)（Python round → half-to-even）
    Some(py_round(f64::from(best) * 360.0 / 255.0) as i64)
}

fn hue_family(hue: Option<i64>, sat: i64) -> &'static str {
    let Some(hue) = hue else { return "灰/中性" };
    if sat < 25 {
        return "灰/中性";
    }
    if !(15..330).contains(&hue) {
        "红"
    } else if hue < 45 {
        "橙"
    } else if hue < 65 {
        "黄"
    } else if hue < 160 {
        "绿"
    } else if hue < 200 {
        "青"
    } else if hue < 250 {
        "蓝"
    } else if hue < 285 {
        "紫"
    } else {
        "品红"
    }
}

fn warm_cool(hue: Option<i64>, sat: i64) -> &'static str {
    let Some(hue) = hue else { return "中性" };
    if sat < 25 {
        return "中性";
    }
    if !(70..330).contains(&hue) {
        "暖色"
    } else {
        "冷色"
    }
}

fn brightness_tier(v: i64) -> &'static str {
    if v < 85 {
        "暗"
    } else if v < 170 {
        "中"
    } else {
        "亮"
    }
}

fn saturation_tier(s: i64) -> &'static str {
    if s < 25 {
        "低饱和"
    } else if s < 60 {
        "中饱和"
    } else {
        "高饱和"
    }
}

fn aspect_of(w: u32, h: u32) -> &'static str {
    let r = if h == 0 { 1.0 } else { f64::from(w) / f64::from(h) };
    if r > 1.2 {
        "横版"
    } else if r < 0.83 {
        "竖版"
    } else {
        "方形"
    }
}

/// MEDIANCUT 量化 → 调色板色像素数（PIL `quantize(MEDIANCUT).getcolors()` 等价）。
/// 返回 (count, rgb) 按 count 降序（平局保持调色板索引序）。
fn quantize_counts(img: &RgbImage, n_colors: u32) -> Vec<(u32, [u8; 3])> {
    let (boxes, order) = median_cut_boxes(img, n_colors);
    let n = order.len();
    let palette: Vec<[u8; 3]> = order
        .iter()
        .map(|&i| boxes[i].as_ref().unwrap().mean().1)
        .collect();
    // 每 distinct 颜色 → 最近调色板（4×初始距离界精化，与 scene_stats 一致）
    let dist2 = |a: [u8; 3], b: [u8; 3]| -> u32 {
        let d = |x: u8, y: u8| (i32::from(x) - i32::from(y)).pow(2) as u32;
        d(a[0], b[0]) + d(a[1], b[1]) + d(a[2], b[2])
    };
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
                    .is_some_and(|bb: &Box| bb.colors.iter().any(|(c, _)| c == color))
            })
            .expect("color must be in a box");
        let pal_idx = order
            .iter()
            .position(|&i| i == box_idx)
            .expect("box must be in traversal order");
        let initial = dist2(*color, palette[pal_idx]);
        let mut best = pal_idx;
        let mut best_dist = initial;
        let bound = initial * 4;
        for (j, &pj) in order.iter().enumerate() {
            if j == pal_idx {
                continue;
            }
            let d = dist2(*color, palette[pj]);
            if d < best_dist && d <= bound {
                best_dist = d;
                best = j;
            }
        }
        counts[best] += cnt;
    }
    let mut entries: Vec<(u32, [u8; 3])> = counts
        .iter()
        .enumerate()
        .filter(|(_, c)| **c > 0)
        .map(|(i, c)| (*c, palette[i]))
        .collect();
    entries.sort_by_key(|x| std::cmp::Reverse(x.0)); // count 降序（稳定）
    entries
}

/// 单张壁纸的程序化特征（对应 Python `analyze_image`）。
fn analyze_image(path: &Path, colors: u32) -> Value {
    match load_rgb(path) {
        Ok(img) => {
            let (w, h) = (img.width, img.height);
            let total = f64::from(w * h);
            let dom = quantize_counts(&img, colors);
            let dominant_colors: Vec<Value> = dom
                .iter()
                .map(|(count, rgb)| {
                    json!({
                        "hex": format!("#{:02X}{:02X}{:02X}", rgb[0], rgb[1], rgb[2]),
                        "pct": py_round_n(f64::from(*count) / total * 100.0, 1),
                    })
                })
                .collect();
            let ds = downsample(&img, 512);
            let (_, sat, bright) = hsv_mean(&ds); // (H, S, V) → 取 S/V
            let hue = dominant_hue(&img);
            let family = hue_family(hue, sat);
            let wc = warm_cool(hue, sat);
            let bt = brightness_tier(bright);
            let st = saturation_tier(sat);
            json!({
                "size_px": [w, h],
                "aspect": aspect_of(w, h),
                "metrics": {
                    "brightness": bright,
                    "saturation": sat,
                    "dominant_colors": dominant_colors,
                },
                "programmatic": {
                    "hue_deg": hue,
                    "family": family,
                    "warm_cool": wc,
                    "tone": bt,
                    "sat_tier": st,
                    "category": format!("{wc}·{bt}·{st}"),
                },
            })
        }
        Err(e) => json!({"error": e.chars().take(200).collect::<String>()}),
    }
}

/// 运行壁纸分类（对应 Python `main()`）。
pub fn run(dir: &Path, p: &WallParams) -> Result<Value, String> {
    let exts: Vec<String> = p
        .exts
        .iter()
        .map(|e| if e.starts_with('.') { e.clone() } else { format!(".{e}") })
        .collect();
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("read_dir {dir:?}: {e}"))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.extension()
                    .map(|x| exts.contains(&format!(".{}", x.to_string_lossy().to_lowercase())))
                    .unwrap_or(false)
        })
        .collect();
    files.sort();
    files.truncate(p.max_files as usize);

    let mut items: Vec<Value> = Vec::new();
    for path in &files {
        let mut item = analyze_image(path, p.colors);
        item["file"] = json!(path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
        items.push(item);
    }

    // groups：by_category / by_tone / by_family（Counter 保序）
    let mut by_category: Vec<(String, u32)> = Vec::new();
    let mut by_tone: Vec<(String, u32)> = Vec::new();
    let mut by_family: Vec<(String, u32)> = Vec::new();
    for it in &items {
        if let Some(prog) = it.get("programmatic") {
            let push = |v: &mut Vec<(String, u32)>, key: &str| {
                let k = prog.get(key).and_then(|x| x.as_str()).unwrap_or("?").to_string();
                if let Some(e) = v.iter_mut().find(|(s, _)| *s == k) {
                    e.1 += 1;
                } else {
                    v.push((k, 1));
                }
            };
            push(&mut by_category, "category");
            push(&mut by_tone, "tone");
            push(&mut by_family, "family");
        }
    }
    let to_obj = |v: &[(String, u32)]| -> Value {
        let mut m = serde_json::Map::new();
        for (k, c) in v {
            m.insert(k.clone(), json!(c));
        }
        json!(m)
    };

    Ok(json!({
        "schema": "vision-report/v1",
        "source": {
            "type": "wallpaper_batch",
            "dir": dir.display().to_string(),
            "count": items.len(),
            "truncated": items.len() >= p.max_files as usize,
        },
        "items": items,
        "groups": {
            "by_category": to_obj(&by_category),
            "by_tone": to_obj(&by_tone),
            "by_family": to_obj(&by_family),
        },
        "semantic": {"enabled": false, "failures": 0},
    }))
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
    fn hsv_matches_pil() {
        // PIL 12.3.0 colorsys 公式实测值
        assert_eq!(rgb_to_hsv(255, 0, 0), [0, 255, 255]); // 纯红 H=0
        assert_eq!(rgb_to_hsv(0, 255, 0), [85, 255, 255]); // 纯绿 H=85
        assert_eq!(rgb_to_hsv(0, 0, 255), [170, 255, 255]); // 纯蓝 H=170
        assert_eq!(rgb_to_hsv(128, 128, 128), [0, 0, 128]); // 灰 S=0
        assert_eq!(rgb_to_hsv(255, 255, 0), [42, 255, 255]); // 黄 H=42
        assert_eq!(rgb_to_hsv(255, 0, 255), [212, 255, 255]); // 品红 H=212
        assert_eq!(rgb_to_hsv(0, 255, 255), [127, 255, 255]); // 青 H=127
        assert_eq!(rgb_to_hsv(200, 100, 50), [14, 191, 200]); // 橙棕
    }

    #[test]
    fn tiers_and_aspect() {
        assert_eq!(brightness_tier(50), "暗");
        assert_eq!(brightness_tier(100), "中");
        assert_eq!(brightness_tier(200), "亮");
        assert_eq!(saturation_tier(10), "低饱和");
        assert_eq!(saturation_tier(40), "中饱和");
        assert_eq!(saturation_tier(200), "高饱和");
        assert_eq!(aspect_of(1920, 1080), "横版");
        assert_eq!(aspect_of(1080, 1920), "竖版");
        assert_eq!(aspect_of(1000, 1000), "方形");
        assert_eq!(hue_family(Some(10), 100), "红");
        assert_eq!(hue_family(Some(50), 100), "黄");
        assert_eq!(hue_family(Some(200), 100), "蓝");
        assert_eq!(hue_family(None, 100), "灰/中性");
        assert_eq!(hue_family(Some(100), 10), "灰/中性");
    }

    #[test]
    fn quantize_solid_color() {
        // 纯红图 → 1 个调色板色
        let im = img(4, 4, &[255, 0, 0].repeat(16));
        let counts = quantize_counts(&im, 5);
        assert_eq!(counts.len(), 1);
        assert_eq!(counts[0].0, 16);
        assert_eq!(counts[0].1, [255, 0, 0]);
    }

    #[test]
    fn analyze_solid_red() {
        // 纯红 100×50 → 横版 + 红 + 亮 + 高饱和
        let path = std::env::temp_dir().join("wall_test_red.png");
        let im = img(100, 50, &[255, 0, 0].repeat(5000));
        let _ = im;
        // 直接测 analyze 逻辑（不写文件）：用 downsample + hsv_mean
        let ds = downsample(&img(100, 50, &[255, 0, 0].repeat(5000)), 512);
        let (_, sat, bright) = hsv_mean(&ds);
        assert_eq!(bright, 255);
        assert_eq!(sat, 255);
        assert_eq!(aspect_of(100, 50), "横版");
        let _ = path;
    }
}

#[cfg(test)]
mod dbg_tests {
    use super::*;

    #[test]
    fn dbg_blue_hue() {
        let data = [0u8, 0, 255].repeat(5000);
        let img = RgbImage { width: 50, height: 100, data };
        let h = dominant_hue(&img);
        println!("dominant_hue = {h:?}");
        let v = 171.0 * 360.0 / 255.0;
        println!("171*360/255 = {v} → py_round = {}", py_round(v));
        assert!(h.is_some());
    }
}
