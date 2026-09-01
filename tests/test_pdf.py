#!/usr/bin/env python3
"""test_pdf.py — PDF 抽取自测（PyMuPDF，schema v2）。

运行（pi-vision env）:
  /home/Arch/conda-envs/pi-vision/bin/python tests/test_pdf.py
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / "python"
OUT = ROOT / "tests" / "_out"
OUT.mkdir(parents=True, exist_ok=True)
PYBIN = "/home/Arch/conda-envs/pi-vision/bin/python"

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name} {detail}")


def make_pdf() -> Path:
    import fitz  # type: ignore[import-not-found]

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), "pi-vision-struct PDF test", fontsize=18)
    page.insert_text((72, 170), "Second line numbers 12345", fontsize=12)
    doc.save(str(OUT / "test_pdf.pdf"))
    doc.close()
    return OUT / "test_pdf.pdf"


def test_pdf() -> None:
    print("[vs_pdf] 文本块抽取")
    pdf = make_pdf()
    r = subprocess.run([PYBIN, str(PY / "vs_pdf.py"), "--file", str(pdf),
                        "--render-dir", str(OUT / "pdf_render")],
                       capture_output=True, text=True, timeout=120)
    d = json.loads(r.stdout)
    check("schema v2 + task=pdf", d.get("schema") == "vision-report/v3" and d.get("task") == "pdf")
    check("coordsys=pt", d.get("coordsys") == "pt")
    check("page_size A4 pt", d["source"]["page_size_pt"] == [595.0, 842.0])
    texts = [e["text"] for e in d["elements"]]
    check("英文行提取", any("PDF test" in t for t in texts), str(texts))
    check("数字行提取", any("12345" in t for t in texts))
    check("元素含 bbox+page", all(e.get("bbox") and e.get("page") for e in d["elements"][:3]))
    check("渲染页生成", len(d["source"].get("rendered_pages", [])) == 1)


def main() -> int:
    print(f"pdf self-tests (python {sys.version.split()[0]})\n")
    test_pdf()
    print(f"\n结果: {PASS} 通过 / {FAIL} 失败")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
