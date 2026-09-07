//! DOM 传感器（对应 Python `vs_dom.py`：Playwright → chromiumoxide 驱动 headless Chrome）。
//!
//! chromiumoxide（CDP 协议）驱动 headless Chrome → 提取 DOM 元素
//! （tag/bbox/text/computed style）→ schema v3 elements（coordsys=css_px）。

use serde_json::{json, Value};

use crate::schema::envelope;

/// DOM 参数（对应 Python `--url/--max-elements/--timeout-ms`）。
pub struct DomParams {
    pub url: String,
    pub max_elements: u32,
    pub timeout_ms: u64,
}

/// 与 Python `EXTRACT_JS` 等价的提取脚本（返回 elements + dpr + scroll）。
const EXTRACT_JS: &str = r#"
() => {
  const els = document.querySelectorAll('*');
  const out = [];
  const vw = innerWidth, vh = innerHeight;
  for (const el of els) {
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) continue;
    if (r.bottom < 0 || r.right < 0 || r.top > vh || r.left > vw) continue;
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden' || Number(cs.opacity) === 0) continue;
    const tag = el.tagName.toLowerCase();
    if (['script','style','link','meta','title','head','noscript'].includes(tag)) continue;
    const text = (el.innerText || '').trim().replace(/\s+/g, ' ').slice(0, 200);
    const aria = el.getAttribute('aria-label') || '';
    if (!text && !aria && !['img','svg','canvas','video','input','button','a','select','textarea'].includes(tag)) continue;
    out.push({
      tag, role: el.getAttribute('role'),
      text: text || aria,
      bbox: [Math.round(r.left), Math.round(r.top), Math.round(r.right), Math.round(r.bottom)],
      style: {
        color: cs.color, bg: cs.backgroundColor, fontSize: cs.fontSize,
        fontWeight: cs.fontWeight, fontFamily: cs.fontFamily,
        display: cs.display, position: cs.position, zIndex: cs.zIndex,
        overflow: cs.overflow, textAlign: cs.textAlign, lineHeight: cs.lineHeight
      }
    });
  }
  out.sort((a,b) => (b.bbox[2]-b.bbox[0])*(b.bbox[3]-b.bbox[1]) - (a.bbox[2]-a.bbox[0])*(a.bbox[3]-a.bbox[1]));
  return {elements: out, dpr: window.devicePixelRatio, scrollX: window.scrollX, scrollY: window.scrollY};
}
"#;

/// CSS 颜色 → '#RRGGBB'（与 Python `css_to_hex` 一致；透明/未知返回 None）。
fn css_to_hex(color: &str) -> Option<String> {
    let c = color.trim();
    if c.is_empty() {
        return None;
    }
    if let Some(hex) = c.strip_prefix('#') {
        return Some(format!("#{}", hex[..hex.len().min(6)].to_uppercase()));
    }
    // rgb(34, 120, 210) / rgba(...)
    let inner = c
        .strip_prefix("rgb(")
        .or_else(|| c.strip_prefix("rgba("))?
        .strip_suffix(')')?;
    let parts: Vec<&str> = inner.split(',').collect();
    if parts.len() < 3 {
        return None;
    }
    let r = parts[0].trim().parse::<f64>().ok()?.round() as u8;
    let g = parts[1].trim().parse::<f64>().ok()?.round() as u8;
    let b = parts[2].trim().parse::<f64>().ok()?.round() as u8;
    Some(format!("#{r:02X}{g:02X}{b:02X}"))
}

/// 文本类标签（与 Python 一致）。
const TEXT_TAGS: &[&str] = &[
    "p", "span", "h1", "h2", "h3", "h4", "label", "a", "li", "td",
];

/// 驱动 headless Chrome 提取 DOM → schema v3 envelope。
pub fn run(p: &DomParams) -> Result<Value, String> {
    use chromiumoxide::browser::BrowserConfig;
    use chromiumoxide::Browser;

    let rt = tokio::runtime::Runtime::new().map_err(|e| format!("tokio rt: {e}"))?;
    rt.block_on(async {
        let config = BrowserConfig::builder()
            .with_head()
            .build()
            .map_err(|e| format!("browser config: {e}"))?;
        let (mut browser, _handler) = Browser::launch(config)
            .await
            .map_err(|e| format!("chrome launch: {e}"))?;
        let page = browser
            .new_page("about:blank")
            .await
            .map_err(|e| format!("new page: {e}"))?;
        page.goto(&p.url)
            .await
            .map_err(|e| format!("goto {}: {e}", p.url))?;
        page.wait_for_navigation()
            .await
            .map_err(|e| format!("wait nav: {e}"))?;
        // 等待渲染（与 Python wait_for_timeout(800) 对应）
        tokio::time::sleep(std::time::Duration::from_millis(800)).await;
        let data = page
            .evaluate(EXTRACT_JS)
            .await
            .map_err(|e| format!("evaluate: {e}"))?
            .value()
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let _ = browser.close().await;

        let dpr = data["dpr"].as_f64().unwrap_or(1.0);
        let scroll = (
            data["scrollX"].as_i64().unwrap_or(0) as i32,
            data["scrollY"].as_i64().unwrap_or(0) as i32,
        );
        let raw: Vec<Value> = data["elements"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let mut elements: Vec<Value> = Vec::new();
        for (i, e) in raw.iter().take(p.max_elements as usize).enumerate() {
            let tag = e["tag"].as_str().unwrap_or("").to_string();
            let bbox: Vec<i64> = e["bbox"]
                .as_array()
                .map(|a| a.iter().filter_map(|v| v.as_i64()).collect())
                .unwrap_or_default();
            let st = &e["style"];
            let text = e["text"].as_str().unwrap_or("").to_string();
            let el = json!({
                "id": i,
                "type": if TEXT_TAGS.contains(&tag.as_str()) { "text" } else { tag.as_str() },
                "bbox": bbox,
                "text": text,
                "conf": 1.0,
                "color": {
                    "fill": css_to_hex(st["bg"].as_str().unwrap_or("")),
                    "text": css_to_hex(st["color"].as_str().unwrap_or("")),
                },
                "font": {
                    "size_pt": Value::Null,
                    "size_px": st["fontSize"].as_str().unwrap_or(""),
                    "family": st["fontFamily"].as_str().unwrap_or(""),
                    "weight": st["fontWeight"].as_str().unwrap_or(""),
                    "line_height": st["lineHeight"].as_str().unwrap_or(""),
                },
                "z": if st["zIndex"].as_str().unwrap_or("auto") == "auto" { Value::Null } else { st["zIndex"].clone() },
                "source": ["dom"],
                "coordsys": "css_px",
                "style": {
                    "display": st["display"].as_str().unwrap_or(""),
                    "position": st["position"].as_str().unwrap_or(""),
                    "overflow": st["overflow"].as_str().unwrap_or(""),
                    "textAlign": st["textAlign"].as_str().unwrap_or(""),
                },
                "role": e["role"].clone(),
                "bbox_device_px": bbox.iter().map(|v| (*v as f64 * dpr).round() as i64).collect::<Vec<_>>(),
            });
            elements.push(el);
        }

        let mut report = envelope(
            Some("dom"),
            &["dom"],
            Some("css_px"),
            Some(json!({
                "type": "dom",
                "url": p.url,
                "screenshot": Value::Null,
                "viewport_px": [1440, 900],
                "dpr": dpr,
                "scroll": [scroll.0, scroll.1],
            })),
        );
        report["elements"] = Value::Array(elements);
        report["truncated"] = Value::Bool(raw.len() >= p.max_elements as usize);
        Ok(report)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn css_to_hex_rgb() {
        assert_eq!(css_to_hex("rgb(34, 120, 210)"), Some("#2278D2".into()));
        assert_eq!(css_to_hex("rgba(0, 0, 0, 0.5)"), Some("#000000".into()));
        assert_eq!(css_to_hex("#abc"), Some("#ABC".into()));
        assert_eq!(css_to_hex(""), None);
        assert_eq!(css_to_hex("transparent"), None);
    }

    #[test]
    fn dom_params_default() {
        let p = DomParams {
            url: "about:blank".into(),
            max_elements: 60,
            timeout_ms: 20000,
        };
        assert_eq!(p.max_elements, 60);
    }
}
