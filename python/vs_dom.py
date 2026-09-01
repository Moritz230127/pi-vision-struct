#!/usr/bin/env python3
"""vs_dom.py — Playwright + Firefox (WebDriver BiDi) 提取 DOM 结构化信息（schema v2）。

DOM 是布局真值：文本、bbox、computed style（颜色/字号/定位/z-index）全部无损。
输出 v2 元素：coordsys=css_px（视口相对），同时给出 device_px（×DPR，与截图像素对齐）。

用法:
  vs_dom.py --url URL [--max-elements 60] [--out-screenshot PATH] [--timeout-ms 20000]
"""
import argparse
import json
import vs_schema as S
import re
import sys
import urllib.parse

EXTRACT_JS = r"""
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
"""


def css_to_hex(color: str) -> str | None:
    """'rgb(34, 120, 210)' / 'rgba(...)' / '#RRGGBB' → '#2278D2'；透明或未知返回 None。"""
    if not color:
        return None
    color = color.strip()
    if color.startswith("#"):
        return color[:7].upper()
    m = re.match(r"rgba?\(([\d.]+),\s*([\d.]+),\s*([\d.]+)(?:,\s*[\d.]+)?\)", color)
    if not m:
        return None
    try:
        r, g, b = (round(float(v)) for v in m.groups())
    except ValueError as e:
        raise ValueError(f"bad css color: {color!r}") from e
    return f"#{r:02X}{g:02X}{b:02X}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--max-elements", type=int, default=60)
    ap.add_argument("--out-screenshot")
    ap.add_argument("--timeout-ms", type=int, default=20000)
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright  # type: ignore[import-not-found]

        url = args.url
        if url.startswith("data:text/html,"):
            # 数据 URL 必须百分号编码，否则中文/特殊字符导致页面加载失败
            url = "data:text/html;charset=utf-8," + urllib.parse.quote(url[len("data:text/html,"):], safe="")

        with sync_playwright() as p:
            # 注：Playwright 的 Firefox driver 需要官方修补版构建（Juggler 协议），
            # 无法驱动系统 Firefox。D1 已预授权：降级使用 playwright install firefox。
            browser = p.firefox.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            page.goto(url, timeout=args.timeout_ms, wait_until="load")
            page.wait_for_timeout(800)
            data = page.evaluate(EXTRACT_JS)
            shot = None
            if args.out_screenshot:
                page.screenshot(path=args.out_screenshot)
                shot = args.out_screenshot
            browser.close()

        dpr = float(data.get("dpr") or 1.0)
        scroll = (int(data.get("scrollX") or 0), int(data.get("scrollY") or 0))
        elements = []
        for i, e in enumerate(data["elements"][: args.max_elements]):
            bbox_css = e["bbox"]
            st = e["style"]
            elements.append({
                "id": i,
                "type": "text" if e["tag"] in ("p", "span", "h1", "h2", "h3", "h4", "label", "a", "li", "td") else e["tag"],
                "bbox": bbox_css,
                "text": e["text"],
                "conf": 1.0,
                "color": {"fill": css_to_hex(st["bg"]), "text": css_to_hex(st["color"])},
                "font": {"size_pt": None, "size_px": st["fontSize"], "family": st["fontFamily"],
                         "weight": st["fontWeight"], "line_height": st["lineHeight"]},
                "z": st["zIndex"] if st["zIndex"] != "auto" else None,
                "source": ["dom"], "coordsys": "css_px",
                "style": {k: st[k] for k in ("display", "position", "overflow", "textAlign")},
                "role": e.get("role"),
                "bbox_device_px": [round(v * dpr) for v in bbox_css],
            })

        print(S.dump_json({
            "schema": "vision-report/v3",
            "task": "dom",
            "sensors": ["dom"],
            "coordsys": "css_px",
            "source": {"type": "dom", "url": args.url, "screenshot": shot,
                       "viewport_px": [1440, 900], "dpr": dpr, "scroll": list(scroll)},
            "elements": elements,
            "anomalies": [],
            "metrics": {},
            "truncated": len(elements) >= args.max_elements,
        }))
        return 0
    except Exception as e:
        print(json.dumps({"error": "vs_dom failed", "detail": str(e)[:500]}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
