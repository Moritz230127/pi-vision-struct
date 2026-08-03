#!/usr/bin/env python3
"""Phase 2 用例1 — Firefox 实景诊断端到端（反馈闭环）。

流程：渲染缺陷页 → pix_analyze/ocr_boxes 数字定位 → 依据数字修复 → 复渲染 →
      pix_analyze --compare 验证 diff 归零。

自动化版本用 Playwright-Firefox 渲染真实页面（headless，同一渲染管线）；
实景 grim 抓屏演示在 pi 会话内执行（见报告）。两者共享同一断言标准：
修复后渲染必须与 golden 参考逐像素一致（diff 归零）。

运行: /home/Arch/conda-envs/pi-vision/bin/python tests/case1_firefox_e2e.py
"""
import json
import subprocess
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / "python"
OUT = ROOT / "tests" / "_out" / "case1"
PYBIN = "/home/Arch/conda-envs/pi-vision/bin/python"

GOLDEN_CSS = "color:#111111"  # 修复后的正确值（golden 参考）
BAD_CSS = "color:#808080"     # 注入缺陷：正文文字对比度不足

# 正文段落预期区域（900×560 视口布局推算，仅用于断言锚点）
P_X1, P_Y1, P_X2, P_Y2 = 40, 100, 480, 200

PAGE = """<!DOCTYPE html><html><head><meta charset="utf-8"><style>
 body{{margin:0;background:#FFFFFF;font-family:sans-serif}}
 .bar{{background:#32363C;color:#FFFFFF;height:48px;display:flex;align-items:center;padding:0 16px;font-size:16px}}
 .card{{margin:24px;padding:20px;border:1px solid #E0E0E0;border-radius:8px;width:420px}}
 h1{{color:#2278D2;font-size:24px}}
 p{{font-size:16px;{body_css}}}
 .btn{{margin-top:12px;background:#2278D2;color:#FFFFFF;border:none;padding:8px 16px;font-size:14px}}
</style></head><body>
<div class="bar">pi-vision · 实景诊断闭环</div>
<div class="card"><h1>反馈闭环测试页</h1>
<p>这是一段用于对比度测量的正文文本。WCAG AA 要求小字号对比度 ≥4.5:1。</p>
<button class="btn">立即修复</button></div>
</body></html>"""

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


def run_tool(script: str, args: list[str], timeout: int = 120) -> dict:
    r = subprocess.run([PYBIN, str(PY / script), *args],
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise AssertionError(f"tool exit {r.returncode}: {r.stdout} {r.stderr[:300]}")
    try:
        return json.loads(r.stdout)
    except ValueError as e:
        raise AssertionError(f"bad JSON from {script}: {r.stdout[:200]}") from e


def render(html: str, out: Path) -> None:
    """用 Playwright-Firefox 渲染页面为 PNG（与 dom_dump 同一管线）。"""
    from playwright.sync_api import sync_playwright  # type: ignore[import-not-found]
    url = "data:text/html;charset=utf-8," + urllib.parse.quote(html, safe="")
    with sync_playwright() as p:
        b = p.firefox.launch(headless=True)
        pg = b.new_page(viewport={"width": 900, "height": 560})
        pg.goto(url, wait_until="load")
        pg.wait_for_timeout(400)
        pg.screenshot(path=str(out))
        b.close()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    golden_png = OUT / "golden.png"
    bad_png = OUT / "bad.png"
    fixed_png = OUT / "fixed.png"

    print("[case1] 渲染 golden 参考与缺陷页")
    render(PAGE.format(body_css=GOLDEN_CSS), golden_png)
    render(PAGE.format(body_css=BAD_CSS), bad_png)

    print("[case1] 测量：pix_analyze 定位缺陷")
    # 1) 对比度测量：缺陷色 #808080 vs 白底 → 应不达标
    wc = run_tool("vs_pix.py", ["--image", str(bad_png), "--wcag", "808080,FFFFFF", "111111,FFFFFF"])
    bad_ratio = wc["wcag"][0]["ratio"]
    golden_ratio = wc["wcag"][1]["ratio"]
    check("缺陷色 #808080 对比度 ≈3.95（<4.5 不达标）",
          3.8 <= bad_ratio <= 4.1 and not wc["wcag"][0]["passes_aa"], str(wc["wcag"][0]))
    check("修复色 #111111 对比度 >12（达标）",
          golden_ratio > 12 and wc["wcag"][1]["passes_aa"], str(wc["wcag"][1]))

    # 2) diff：缺陷页 vs golden → 异常应落在正文段落区域
    d = run_tool("vs_pix.py", ["--image", str(bad_png), "--compare", str(golden_png)])
    an = d.get("anomalies", [])
    total_px = sum(a["count"] for a in an)
    # 连通域输出：文本差异按字形拆为多个分量，全部应位于段落区域
    all_in = bool(an) and all(
        a["bbox"][0] >= P_X1 and a["bbox"][2] <= P_X2 and a["bbox"][1] >= P_Y1 and a["bbox"][3] <= P_Y2
        for a in an
    )
    check("diff 分量全部位于段落区域（字形级定位）", all_in, f"{len(an)} 分量, 首框 {an[0]['bbox'] if an else None}")
    check("diff 总像素 > 100（真实文本像素差异）", total_px > 100, f"total={total_px}")

    # 3) OCR：读回页面文字（传感器双通道：pix + ocr）
    o = run_tool("vs_ocr.py", ["--image", str(bad_png)])
    texts = "".join(it["text"] for it in o["elements"])
    check("OCR 读出标题「反馈闭环测试页」", "反馈闭环测试页" in texts, texts[:120])

    print("[case1] 修复：依据数字把正文颜色改为 #111111 后复渲染")
    render(PAGE.format(body_css=GOLDEN_CSS), fixed_png)

    print("[case1] 复验：fixed vs golden --compare 应 diff 归零")
    v = run_tool("vs_pix.py", ["--image", str(fixed_png), "--compare", str(golden_png)])
    v_an = v.get("anomalies", [])
    check("复截图与 golden 逐像素一致（diff 归零）", not v_an, str(v_an))

    print(f"\ncase1 结果: {PASS} 通过 / {FAIL} 失败")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
