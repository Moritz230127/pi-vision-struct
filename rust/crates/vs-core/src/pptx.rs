//! PPTX 传感器（对应 Python `vs_pptx.py`：python-pptx → OOXML 自研解析）。
//!
//! 形状位置（EMU→pt）、填充色、字体、文本、图片引用 —— 全部来自文件本身，
//! 零渲染、零丢失。OOXML 解析：zip crate + roxmltree。

use serde_json::{json, Value};
use std::io::Read;

/// PPTX 参数（对应 Python `--slide`）。
pub struct PptxParams {
    /// PPTX 文件路径。
    pub file: String,
    /// 指定幻灯片（1 基，None = 全部）。
    pub slide: Option<u32>,
}

/// EMU → pt 换算（python-pptx 同款 12700.0）。
const EMU_PT: f64 = 12700.0;

/// 读取 zip 内文件内容。
fn zip_read(zip: &mut zip::ZipArchive<std::fs::File>, name: &str) -> Result<String, String> {
    let mut f = zip
        .by_name(name)
        .map_err(|e| format!("zip read {name}: {e}"))?;
    let mut s = String::new();
    f.read_to_string(&mut s)
        .map_err(|e| format!("read {name}: {e}"))?;
    Ok(s)
}

/// 解析 XML 属性（命名空间感知：取本地名）。返回 owned String。
fn attr<'a, 'input: 'a>(node: roxmltree::Node<'a, 'input>, local: &str) -> Option<String> {
    node.attributes()
        .find(|a| a.name() == local || a.name().ends_with(&format!(":{local}")))
        .map(|a| a.value().to_string())
}

/// 解析子元素（本地名匹配）。
fn child<'a, 'input: 'a>(
    node: roxmltree::Node<'a, 'input>,
    local: &str,
) -> Option<roxmltree::Node<'a, 'input>> {
    node.children()
        .find(|c| c.is_element() && c.tag_name().name() == local)
}

/// 递归查找后代元素（本地名匹配）。
fn find_desc<'a, 'input: 'a>(
    node: roxmltree::Node<'a, 'input>,
    local: &str,
) -> Option<roxmltree::Node<'a, 'input>> {
    if node.tag_name().name() == local {
        return Some(node);
    }
    for c in node.children() {
        if c.is_element() {
            if let Some(found) = find_desc(c, local) {
                return Some(found);
            }
        }
    }
    None
}

/// 解析 EMU 数值（可能带单位后缀如 "12700" 或 "1in"）。
fn emu_val(s: &str) -> Option<f64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    if let Some(v) = s.strip_suffix("mm") {
        // 1mm = 36000 EMU
        return v.trim().parse::<f64>().ok().map(|v| v * 36000.0);
    }
    if let Some(v) = s.strip_suffix("cm") {
        // 1cm = 360000 EMU
        return v.trim().parse::<f64>().ok().map(|v| v * 360000.0);
    }
    if let Some(v) = s.strip_suffix("in") {
        // 1in = 914400 EMU
        return v.trim().parse::<f64>().ok().map(|v| v * 914400.0);
    }
    if let Some(v) = s.strip_suffix("pt") {
        // 1pt = 12700 EMU
        return v.trim().parse::<f64>().ok().map(|v| v * 12700.0);
    }
    s.parse::<f64>().ok()
}

/// 解析填充色（a:solidFill/a:srgbClr val）。
fn fill_color(sp_pr: Option<roxmltree::Node>) -> Option<String> {
    let sp_pr = sp_pr?;
    let solid = find_desc(sp_pr, "solidFill")?;
    let srgb = find_desc(solid, "srgbClr")?;
    let val = attr(srgb, "val")?;
    Some(format!("#{val}"))
}

/// 解析文本（a:txBody → 段落 → runs）。
fn parse_texts(tx_body: Option<roxmltree::Node>) -> Vec<Value> {
    let mut texts = Vec::new();
    let Some(tx_body) = tx_body else { return texts };
    for p in tx_body
        .children()
        .filter(|c| c.is_element() && c.tag_name().name() == "p")
    {
        let mut runs = Vec::new();
        for r in p
            .children()
            .filter(|c| c.is_element() && c.tag_name().name() == "r")
        {
            let r_pr = child(r, "rPr");
            let t = child(r, "t").map(|t| t.text().unwrap_or("").to_string());
            let text = t.unwrap_or_default();
            // 字号（sz 单位 1/100 pt）
            let size_pt = r_pr
                .and_then(|rp| attr(rp, "sz"))
                .and_then(|s| s.parse::<f64>().ok())
                .map(|sz| round2(sz / 100.0));
            let bold = r_pr
                .and_then(|rp| attr(rp, "b"))
                .map(|b| b == "1" || b == "true")
                .unwrap_or(false);
            // 颜色（rPr/solidFill/srgbClr）
            let color = r_pr.and_then(|rp| fill_color(Some(rp)));
            // 字体（latin typeface）
            let font = r_pr
                .and_then(|rp| find_desc(rp, "latin"))
                .and_then(|l| attr(l, "typeface"))
                .map(|s| s.to_string());
            runs.push(json!({
                "text": text.chars().take(200).collect::<String>(),
                "size_pt": size_pt,
                "bold": bold,
                "color": color,
                "font": font,
            }));
        }
        texts.push(json!({"runs": runs}));
    }
    texts
}

/// 递归导出形状（与 python-pptx shape_dump 同构）。
fn shape_dump(node: roxmltree::Node, out: &mut Vec<Value>) {
    let tag = node.tag_name().name();
    // 形状类型映射（python-pptx MSO_SHAPE_TYPE 字符串）
    // sp 带 txBox="1" 时是 TEXT_BOX（python-pptx 识别）
    let is_text_box = tag == "sp"
        && node
            .children()
            .find(|c| c.is_element() && c.tag_name().name().starts_with("nv"))
            .and_then(|n| child(n, "cNvSpPr"))
            .and_then(|n| attr(n, "txBox"))
            .map(|v| v == "1")
            .unwrap_or(false);
    let (shape_type, is_group, is_picture) = match tag {
        "sp" if is_text_box => ("TEXT_BOX (17)", false, false),
        "sp" => ("AUTO_SHAPE (1)", false, false),
        "pic" => ("PICTURE (13)", false, true),
        "grpSp" => ("GROUP (6)", true, false),
        "cxnSp" => ("CONNECTOR (9)", false, false),
        "graphicFrame" => ("GRAPHIC_FRAME (12)", false, false),
        _ => return,
    };

    // id + name（nvSpPr/nvPicPr/nvGrpSpPr → cNvPr）
    let nv = node
        .children()
        .find(|c| c.is_element() && c.tag_name().name().starts_with("nv"))
        .and_then(|n| child(n, "cNvPr"));
    let id = nv.and_then(|n| attr(n, "id")).unwrap_or_else(|| "0".into());
    let name = nv
        .and_then(|n| attr(n, "name"))
        .unwrap_or_default()
        .chars()
        .take(80)
        .collect::<String>();

    // 位置/尺寸（spPr/xfrm: off + ext，EMU→pt）
    let sp_pr = child(node, "spPr");
    let xfrm = sp_pr.and_then(|sp| child(sp, "xfrm"));
    let (pos_pt, size_pt) = match xfrm {
        Some(xf) => {
            let off = child(xf, "off");
            let ext = child(xf, "ext");
            let pos = off.map(|o| {
                let x = attr(o, "x")
                    .and_then(|s| emu_val(&s))
                    .map(|v| round2(v / EMU_PT));
                let y = attr(o, "y")
                    .and_then(|s| emu_val(&s))
                    .map(|v| round2(v / EMU_PT));
                json!([x, y])
            });
            let size = ext.map(|e| {
                let cx = attr(e, "cx")
                    .and_then(|s| emu_val(&s))
                    .map(|v| round2(v / EMU_PT));
                let cy = attr(e, "cy")
                    .and_then(|s| emu_val(&s))
                    .map(|v| round2(v / EMU_PT));
                json!([cx, cy])
            });
            (
                pos.unwrap_or(json!([Value::Null, Value::Null])),
                size.unwrap_or(json!([Value::Null, Value::Null])),
            )
        }
        None => (
            json!([Value::Null, Value::Null]),
            json!([Value::Null, Value::Null]),
        ),
    };

    // 旋转（xfrm/rot，单位 1/60000 度；Python 版 rotation=0 时输出整数 0）
    let rotation = match xfrm
        .and_then(|xf| attr(xf, "rot"))
        .and_then(|r| r.parse::<f64>().ok())
    {
        Some(r) if r != 0.0 => json!(round1(r / 60000.0)),
        _ => json!(0),
    };

    // 文本
    let tx_body = child(node, "txBody");
    let texts = parse_texts(tx_body);

    // 填充
    let fill = fill_color(sp_pr);

    let mut item = json!({
        "id": id,
        "name": name,
        "type": shape_type,
        "pos_pt": pos_pt,
        "size_pt": size_pt,
        "rotation": rotation,
        "texts": texts,
        "fill": fill,
        "image": Value::Null,
        "children": [],
    });

    // 图片信息（pic → blipFill → r:embed → 由调用方解析 rels）
    if is_picture {
        if let Some(blip) = find_desc(node, "blip") {
            let embed = attr(blip, "embed").map(|s| s.to_string());
            item["image"] = json!({"embed": embed});
        }
    }

    // 组子形状
    if is_group {
        let mut children = Vec::new();
        for c in node.children().filter(|c| c.is_element()) {
            let t = c.tag_name().name();
            if t == "sp" || t == "pic" || t == "grpSp" || t == "cxnSp" || t == "graphicFrame" {
                shape_dump(c, &mut children);
            }
        }
        item["children"] = json!(children);
    }
    out.push(item);
}

/// 解析 PPTX 并输出 schema v3 envelope。
pub fn run(p: &PptxParams) -> Result<Value, String> {
    let file = std::fs::File::open(&p.file).map_err(|e| format!("open {}: {e}", p.file))?;
    let mut zip = zip::ZipArchive::new(file).map_err(|e| format!("zip open {}: {e}", p.file))?;

    // 1) presentation.xml → 幻灯片尺寸 + 幻灯片数
    let pres = zip_read(&mut zip, "ppt/presentation.xml")?;
    let pres_doc =
        roxmltree::Document::parse(&pres).map_err(|e| format!("xml parse presentation: {e}"))?;
    let pres_root = pres_doc.root_element();
    let sld_sz = child(pres_root, "sldSz");
    let slide_size_pt = match sld_sz {
        Some(sz) => {
            let cx = attr(sz, "cx")
                .and_then(|s| emu_val(&s))
                .map(|v| round2(v / EMU_PT))
                .unwrap_or(0.0);
            let cy = attr(sz, "cy")
                .and_then(|s| emu_val(&s))
                .map(|v| round2(v / EMU_PT))
                .unwrap_or(0.0);
            json!([cx, cy])
        }
        None => json!([0.0, 0.0]),
    };
    let total_slides = find_desc(pres_root, "sldIdLst")
        .map(|lst| {
            lst.children()
                .filter(|c| c.is_element() && c.tag_name().name() == "sldId")
                .count()
        })
        .unwrap_or(0);

    // 2) 逐幻灯片解析
    let mut slides_out = Vec::new();
    let slide_nums: Vec<usize> = match p.slide {
        Some(n) => vec![n as usize],
        None => (1..=total_slides).collect(),
    };
    for idx in slide_nums {
        let slide_xml = match zip_read(&mut zip, &format!("ppt/slides/slide{idx}.xml")) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let doc = roxmltree::Document::parse(&slide_xml)
            .map_err(|e| format!("xml parse slide{idx}: {e}"))?;
        let root = doc.root_element();
        let mut shapes = Vec::new();
        // 形状在 p:spTree 下（p:sld > p:cSld > p:spTree > 形状）
        let sp_tree = find_desc(root, "spTree");
        let container = sp_tree.unwrap_or(root);
        for c in container.children().filter(|c| c.is_element()) {
            let t = c.tag_name().name();
            if t == "sp" || t == "pic" || t == "grpSp" || t == "cxnSp" || t == "graphicFrame" {
                shape_dump(c, &mut shapes);
            }
        }
        slides_out.push(json!({
            "slide": idx,
            "shapes": shapes,
            "shape_count": shapes.len(),
        }));
    }

    Ok(json!({
        "schema": "vision-report/v3",
        "task": "pptx",
        "sensors": ["pptx"],
        "coordsys": "pt",
        "source": {
            "type": "pptx",
            "path": p.file,
            "slide_size_pt": slide_size_pt,
            "total_slides": total_slides,
        },
        "slides": slides_out,
        "truncated": false,
    }))
}

/// Python round(x, 2) 语义。
fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

/// Python round(x, 1) 语义。
fn round1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emu_conversion() {
        // 1 inch = 914400 EMU = 72 pt
        assert_eq!(emu_val("914400").unwrap(), 914400.0);
        assert_eq!(emu_val("1in").unwrap(), 914400.0);
        assert_eq!(emu_val("1cm").unwrap(), 360000.0);
        assert_eq!(emu_val("1mm").unwrap(), 36000.0);
        assert_eq!(emu_val("72pt").unwrap(), 914400.0);
        assert!(emu_val("").is_none());
    }

    #[test]
    fn round_semantics() {
        assert_eq!(round2(1.234), 1.23);
        assert_eq!(round2(1.235), 1.24);
        assert_eq!(round1(1.25), 1.3);
    }

    #[test]
    fn pptx_params_default() {
        let p = PptxParams {
            file: "/tmp/test.pptx".into(),
            slide: None,
        };
        assert_eq!(p.file, "/tmp/test.pptx");
        assert!(p.slide.is_none());
    }
}
