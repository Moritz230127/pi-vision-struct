#!/usr/bin/env python3
"""vs_pdf.py — PDF 文本+版式联合抽取（PyMuPDF，schema v2）。

逐页提取文本块（含精确 bbox，PDF 坐标 pt）+ 可选渲染页面图像
（供 vs_layout 版式分析 / OCR 联动）。

用法:
  vs_pdf.py --file PATH [--pages 1-3|all] [--max-items 500]
            [--render-dir DIR]      # 可选：把每页渲染成 PNG（300dpi）

输出 schema v2 elements: type="text", bbox(pt), text, page；
source 带 page_size_pt 与 render 目录（如渲染）。
"""
import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--pages", default="all", help="如 1-3 或 all")
    ap.add_argument("--max-items", type=int, default=500)
    ap.add_argument("--render-dir")
    args = ap.parse_args()

    try:
        import fitz  # type: ignore[import-not-found]  # PyMuPDF

        doc = fitz.open(args.file)
        total_pages = doc.page_count
        if args.pages == "all":
            page_range = range(total_pages)
        else:
            lo, hi = (int(v) for v in args.pages.split("-"))
            page_range = range(max(0, lo - 1), min(total_pages, hi))

        els = []
        rendered = []
        for pno in page_range:
            page = doc[pno]
            rect = page.rect  # pt
            blocks = page.get_text("blocks")  # (x0,y0,x1,y1,text,block_no,type)
            for b in blocks:
                x0, y0, x1, y1, text, _, btype = b
                text = str(text).strip()
                if not text or btype != 0:
                    continue
                els.append({"id": len(els), "type": "text",
                            "bbox": [round(float(x0), 1), round(float(y0), 1),
                                     round(float(x1), 1), round(float(y1), 1)],
                            "text": text[:200], "conf": None,
                            "color": None, "font": None, "z": None,
                            "source": ["pdf"], "coordsys": "pt",
                            "page": pno + 1,
                            "center": [round((x0 + x1) / 2, 1), round((y0 + y1) / 2, 1)],
                            "quad": None})
                if len(els) >= args.max_items:
                    break
            if args.render_dir:
                out_dir = Path(args.render_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                pix = page.get_pixmap(dpi=150)
                path = out_dir / f"page_{pno + 1:03d}.png"
                pix.save(str(path))
                rendered.append(str(path))

        print(json.dumps({
            "schema": "vision-report/v3", "task": "pdf", "sensors": ["pdf"],
            "coordsys": "pt",
            "source": {"type": "pdf", "path": args.file, "pages": total_pages,
                       "page_size_pt": [round(float(doc[0].rect.width), 1),
                                        round(float(doc[0].rect.height), 1)],
                       "render_dir": args.render_dir or None,
                       "rendered_pages": rendered},
            "elements": els, "anomalies": [], "metrics": {"page_count": total_pages},
            "truncated": len(els) >= args.max_items,
        }, ensure_ascii=False))
        return 0
    except Exception as e:
        print(json.dumps({"error": "vs_pdf failed", "detail": str(e)[:500]}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
