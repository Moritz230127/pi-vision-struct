#!/usr/bin/env python3
"""test_ocr_layout.py — OCR 双后端 + 文档版式分析自测。

运行（pi-vision env，首次需代理下载 paddle 模型 ~30MB）:
  /home/Arch/conda-envs/pi-vision/bin/python tests/test_ocr_layout.py
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / "python"
PYBIN = "/home/Arch/conda-envs/pi-vision/bin/python"

PASS = 0
FAIL = 0

FIXTURES = ROOT / "tests" / "bench_fixtures"
TEXT = FIXTURES / "text_lines.png"
PAGE = FIXTURES / "layout_page.png"


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name} {detail}")


def run_tool(args: list[str], timeout: int = 400) -> dict:
    r = subprocess.run([PYBIN, str(PY / args[0]), *args[1:]], capture_output=True,
                       text=True, timeout=timeout)
    if r.returncode != 0:
        raise AssertionError(f"exit {r.returncode}: {r.stdout[:200]} {r.stderr[:200]}")
    try:
        return json.loads(r.stdout)
    except ValueError as e:
        raise AssertionError(f"not json: {r.stdout[:200]}") from e


def test_ocr_rapid_default() -> None:
    print("[vs_ocr] 默认后端 rapidocr")
    d = run_tool(["vs_ocr.py", "--image", str(TEXT)])
    check("schema v2", d.get("schema") == "vision-report/v2")
    check("source 标注 backend=rapidocr", d["source"].get("backend") == "rapidocr")
    check("text_lines 检出 ≥8 元素", len(d.get("elements", [])) >= 8,
          f"got {len(d.get('elements', []))}")
    check("元素含 bbox+quad+center", all(
        e.get("bbox") and e.get("quad") and e.get("center") for e in d.get("elements", [])[:5]))


def test_ocr_paddle_backend() -> None:
    print("[vs_ocr] 后端 paddle（PP-OCRv6 medium）")
    d = run_tool(["vs_ocr.py", "--image", str(TEXT), "--backend", "paddle"], timeout=600)
    check("source 标注 backend=paddle", d["source"].get("backend") == "paddle")
    check("text_lines 检出 ≥8 元素", len(d.get("elements", [])) >= 8,
          f"got {len(d.get('elements', []))}")
    texts = [e["text"] for e in d.get("elements", [])]
    check("识别出中文测试行", any("中文" in t for t in texts), str(texts[:3]))


def test_ocr_real_recall() -> None:
    print("[vs_ocr] 真实截图双后端均可用（测试文件需存在）")
    img = Path("/home/Arch/Pi工作区/测试文件/Firefox截图/2026-08-03_17-57-37.png")
    if not img.exists():
        print("  （跳过：测试文件已清空）")
        return
    a = run_tool(["vs_ocr.py", "--image", str(img), "--max-items", "300"])
    b = run_tool(["vs_ocr.py", "--image", str(img), "--max-items", "300",
                  "--backend", "paddle"], timeout=600)
    na, nb = len(a["elements"]), len(b["elements"])
    # 实测（2026-08-05，不限量）: rapid=99, paddle=92 —— 数量相近，
    # 早期“paddle 96>60”对比受 rapid 的 max-items=60 截断误导。
    check(f"rapid 检出 {na} 元素（≥50）", na >= 50, f"got {na}")
    check(f"paddle 检出 {nb} 元素（≥50）", nb >= 50, f"got {nb}")


def test_layout() -> None:
    print("[vs_layout] 文档版式（PP-DocLayoutV3）")
    d = run_tool(["vs_layout.py", "--image", str(PAGE)])
    els = d.get("elements", [])
    check("schema v2 + task=layout", d.get("schema") == "vision-report/v2" and d.get("task") == "layout")
    check("检出 ≥2 区域", len(els) >= 2, f"got {len(els)}")
    check("区域含 label/bbox/conf", all(e.get("label") and e.get("bbox") and e.get("conf")
                                        for e in els[:5]))
    check("含标题类区域", any("title" in (e.get("label") or "") for e in els), str([e["label"] for e in els]))


def main() -> int:
    print(f"ocr/layout self-tests (python {sys.version.split()[0]})\n")
    test_ocr_rapid_default()
    test_ocr_paddle_backend()
    test_ocr_real_recall()
    test_layout()
    print(f"\n结果: {PASS} 通过 / {FAIL} 失败")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
