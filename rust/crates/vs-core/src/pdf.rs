//! PDF 传感器（对应 Python `vs_pdf.py`：PyMuPDF → pdfium-render，schema v3 输出）。
//!
//! 逐页提取文本块（含精确 bbox，PDF 坐标 pt）。pdfium-render 提供逐字符
//! origin + 字号，按行聚合为文本块（与 PyMuPDF `get_text("blocks")` 同构）。

use serde_json::{json, Value};

/// PDF 参数（对应 Python `--pages`）。
pub struct PdfParams {
    /// PDF 文件路径。
    pub file: String,
    /// 页范围 "1-3" 或 "all"。
    pub pages: Option<String>,
}

/// 尝试加载 pdfium 库（系统库或 libreoffice 内置）。
fn bind_pdfium() -> Result<Box<dyn pdfium_render::prelude::PdfiumLibraryBindings>, String> {
    use pdfium_render::prelude::Pdfium;
    // 1) 系统标准库
    if let Ok(b) = Pdfium::bind_to_system_library() {
        return Ok(b);
    }
    // 2) pypdfium2 自带标准 pdfium（Python 包内）
    for cand in [
        "/home/Arch/.local/lib/python3.11/site-packages/pypdfium2_raw/libpdfium.so",
        "/usr/lib/libreoffice/program/libpdfiumlo.so",
        "/usr/lib64/libreoffice/program/libpdfiumlo.so",
    ] {
        if std::path::Path::new(cand).exists() {
            if let Ok(b) = Pdfium::bind_to_library(cand) {
                return Ok(b);
            }
        }
    }
    Err("pdfium 库不可用（需 libpdfium.so 或 libreoffice libpdfiumlo.so）".into())
}

/// 解析页范围 "1-3" → (lo, hi)（1 基，含端点）。
fn parse_page_range(s: &str, total: usize) -> Result<(usize, usize), String> {
    if s == "all" {
        return Ok((0, total));
    }
    let (lo, hi) = s
        .split_once('-')
        .ok_or_else(|| format!("invalid page range: {s}"))?;
    let lo: usize = lo
        .trim()
        .parse()
        .map_err(|_| format!("invalid page range: {s}"))?;
    let hi: usize = hi
        .trim()
        .parse()
        .map_err(|_| format!("invalid page range: {s}"))?;
    let lo = lo.saturating_sub(1).min(total);
    let hi = hi.min(total);
    if lo >= hi {
        return Err(format!("empty page range: {s}"));
    }
    Ok((lo, hi))
}

/// 解析 PDF 并输出 schema v3 envelope。
pub fn run(p: &PdfParams) -> Result<Value, String> {
    use pdfium_render::prelude::Pdfium;

    let bindings = bind_pdfium()?;
    let pdfium = Pdfium::new(bindings);
    let doc = pdfium
        .load_pdf_from_file(&p.file, None)
        .map_err(|e| format!("pdfium load {}: {e}", p.file))?;

    let total_pages = doc.pages().len() as usize;
    let (lo, hi) = match &p.pages {
        Some(s) => parse_page_range(s, total_pages)?,
        None => (0, total_pages),
    };

    // 页面尺寸（pt，第 0 页）
    let page0 = doc
        .pages()
        .get(0)
        .map_err(|e| format!("pdfium page 0: {e}"))?;
    let page_size_pt = [round1(page0.width().value), round1(page0.height().value)];

    let mut els: Vec<Value> = Vec::new();
    // PyMuPDF 用底部原点（y=0 在底部），pdfium 用顶部原点 → 翻转
    let page_h = page0.height().value;
    for pno in lo..hi {
        let page = doc
            .pages()
            .get(pno as i32)
            .map_err(|e| format!("pdfium page {pno}: {e}"))?;
        let text = page.text().map_err(|e| format!("pdfium text {pno}: {e}"))?;
        let chars = text.chars();
        let n = chars.len();
        // 逐字符收集（origin + 字符 + 字号）
        let mut raw: Vec<(f32, f32, f32, String)> = Vec::new();
        for i in 0..n {
            let ch = chars.get(i).map_err(|e| format!("pdfium char {i}: {e}"))?;
            let (x, y) = ch.origin().map_err(|e| format!("pdfium origin {i}: {e}"))?;
            if let Some(s) = ch.unicode_string() {
                let fs = ch.scaled_font_size().value;
                raw.push((x.value, y.value, fs, s));
            }
        }
        // 按行聚合：y 相近（≤ 2pt）归一行，行内按 x 排序合并
        type Line = (f32, f32, f32, Vec<(f32, String)>);
        let mut lines: Vec<Line> = Vec::new();
        for (x, y, fs, s) in raw {
            if let Some(line) = lines.iter_mut().find(|l| (l.0 - y).abs() <= 2.0) {
                line.1 = line.1.max(y);
                line.2 = line.2.max(fs);
                line.3.push((x, s));
            } else {
                lines.push((y, y, fs, vec![(x, s)]));
            }
        }
        for (y0, _y1, font_size, mut items) in lines {
            items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let text_str: String = items.iter().map(|(_, s)| s.as_str()).collect();
            let text_str = text_str.trim().to_string();
            if text_str.is_empty() {
                continue;
            }
            let x0 = items.first().map(|(x, _)| *x).unwrap_or(0.0);
            let x1 = items.last().map(|(x, _)| *x).unwrap_or(0.0);
            // 行高 = 字号（PyMuPDF blocks 的 y1 含行高）
            let y1 = y0 + font_size;
            // y 翻转：pdfium 顶部原点 → PyMuPDF 底部原点
            let (py0, py1) = (page_h - y1, page_h - y0);
            els.push(json!({
                "id": els.len(),
                "type": "text",
                "bbox": [round1(x0), round1(py0), round1(x1), round1(py1)],
                "text": text_str.chars().take(200).collect::<String>(),
                "conf": Value::Null,
                "color": Value::Null,
                "font": Value::Null,
                "z": Value::Null,
                "source": ["pdf"],
                "coordsys": "pt",
                "page": pno + 1,
                "center": [round1((x0 + x1) / 2.0), round1((py0 + py1) / 2.0)],
                "quad": Value::Null,
            }));
        }
    }

    let mut report = json!({
        "schema": "vision-report/v3",
        "task": "pdf",
        "sensors": ["pdf"],
        "coordsys": "pt",
        "source": {
            "type": "pdf",
            "path": p.file,
            "pages": total_pages,
            "page_size_pt": page_size_pt,
            "render_dir": Value::Null,
            "rendered_pages": [],
        },
        "elements": els,
        "anomalies": [],
        "metrics": {"page_count": total_pages},
        "truncated": false,
    });
    report["elements"] = json!(els);
    Ok(report)
}

/// Python round(x, 1) 语义（四舍五入到 1 位小数，返回 f64 避免 f32 精度噪声）。
fn round1(v: f32) -> f64 {
    ((v as f64) * 10.0).round() / 10.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_page_range_all() {
        assert_eq!(parse_page_range("all", 5).unwrap(), (0, 5));
    }

    #[test]
    fn parse_page_range_subset() {
        assert_eq!(parse_page_range("2-4", 10).unwrap(), (1, 4));
        assert_eq!(parse_page_range("1-3", 2).unwrap(), (0, 2));
    }

    #[test]
    fn parse_page_range_invalid() {
        assert!(parse_page_range("abc", 5).is_err());
        assert!(parse_page_range("3-1", 5).is_err());
    }

    #[test]
    fn round1_semantics() {
        assert_eq!(round1(1.25), 1.3);
        assert_eq!(round1(1.24), 1.2);
        assert_eq!(round1(0.0), 0.0);
    }
}
