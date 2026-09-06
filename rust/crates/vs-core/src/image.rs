//! 图像加载与基础像素访问（S1 传感器共享）。
//!
//! 与 Python PIL `Image.open(path).convert("RGB")` 对齐：
//! 统一解码为 RGB8 交错布局（h×w×3），供各传感器消费。

use std::path::Path;

/// 解码后的 RGB 图像（交错布局：每像素 3 字节 R,G,B）。
pub struct RgbImage {
    pub width: u32,
    pub height: u32,
    /// `len = width * height * 3`，行优先。
    pub data: Vec<u8>,
}

impl RgbImage {
    /// 像素 (x, y) 的 RGB 通道。
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 3] {
        let i = ((y * self.width + x) * 3) as usize;
        [self.data[i], self.data[i + 1], self.data[i + 2]]
    }
}

/// 加载图像并统一转为 RGB8（对应 PIL `convert("RGB")`）。
pub fn load_rgb(path: &Path) -> Result<RgbImage, String> {
    let img = image::open(path).map_err(|e| format!("cannot open {path:?}: {e}"))?;
    let rgb = img.to_rgb8();
    let (width, height) = (rgb.width(), rgb.height());
    Ok(RgbImage {
        width,
        height,
        data: rgb.into_raw(),
    })
}

/// 灰度（0-255）——对应 PIL `convert("L")` 的 ITU-R 601-2 luma：
/// `L = (19595R + 38470G + 7471B + 0x8000) >> 16`（带舍入的 65536 除法，与 PIL 一致）。
pub fn to_gray_u8(img: &RgbImage) -> Vec<u8> {
    img.data
        .as_chunks::<3>()
        .0
        .iter()
        .map(|p| {
            let v = (u32::from(p[0]) * 19595
                + u32::from(p[1]) * 38470
                + u32::from(p[2]) * 7471
                + 0x8000)
                >> 16;
            v as u8
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gray_matches_pil_luma() {
        // PIL: convert("L") 对纯色像素的已知值（L24 宏：19595/38470/7471 + 0x8000 >> 16）
        let img = RgbImage {
            width: 1,
            height: 1,
            data: vec![255, 0, 0], // 纯红
        };
        assert_eq!(to_gray_u8(&img), vec![76]); // (255*19595+32768)>>16 = 76.75 → 76
        let img = RgbImage {
            width: 1,
            height: 1,
            data: vec![0, 255, 0], // 纯绿
        };
        assert_eq!(to_gray_u8(&img), vec![150]); // (255*38470+32768)>>16 = 150.17 → 150
        let img = RgbImage {
            width: 1,
            height: 1,
            data: vec![0, 0, 255], // 纯蓝
        };
        assert_eq!(to_gray_u8(&img), vec![29]); // (255*7471+32768)>>16 = 29.57 → 29
    }
}
