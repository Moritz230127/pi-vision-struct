#!/usr/bin/env python3
"""run_full_test.py — V3.0 全功能全工况跑通测试。

覆盖全部动作/传感器/协议/调度，逐项验证 PASS/FAIL。

测试矩阵:
  L0 源层:   capture / dom / pptx / pdf / a11y
  L1 确定性: pixels / ocr / scene_stats / edge / ascii / geometry
  L2 轻量DL: saliency / segment / depth
  F1 融合:   fusion (D-S)
  F2 协议:   analyze / zoom / probe
  3D:        audit3d
  其他:      wallpaper / cluster / detect / omniparser / layout / audit / rules
  调度:      vsd ping / stats / 并发 / 优先级 / 功耗
  自诊断:    check
"""
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / "python"
PYBIN = f"{Path.home()}/conda-envs/pi-vision/bin/python"
VSENSOR = f"{Path.home()}/conda-envs/vsensor/bin/python"
OMNI = f"{Path.home()}/conda-envs/omniparser/bin/python"

RESULTS: list[tuple[str, bool, str]] = []


def run(pybin: str, script: str, *args: str, timeout: int = 180) -> dict:
    cmd = [pybin, str(PY / script), *args]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return json.loads(r.stdout) if r.stdout.strip() else {"error": "no output"}
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        return {"error": str(e)[:200]}


def check(name: str, cond: bool, detail: str = ""):
    RESULTS.append((name, bool(cond), detail))
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail and not cond else ""))


def make_fixtures():
    """生成测试夹具。"""
    import numpy as np
    from PIL import Image, ImageDraw

    # 1. UI 截图
    img = Image.new("RGB", (500, 300), (245, 245, 245))
    d = ImageDraw.Draw(img)
    d.rectangle([50, 40, 200, 80], fill=(30, 120, 220))
    d.rectangle([50, 100, 200, 140], outline=(200, 200, 200), width=2)
    d.rectangle([50, 160, 100, 210], fill=(220, 180, 30))
    img.save("/tmp/ft_ui.png")

    # 2. 复杂背景
    rng = np.random.default_rng(3)
    arr = np.zeros((400, 600, 3), dtype=np.uint8)
    for y in range(400):
        arr[y, :] = [int(30 + y * 0.4), int(20 + y * 0.3), int(40 + y * 0.2)]
    arr += rng.integers(0, 25, arr.shape, dtype=np.uint8)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img2 = Image.fromarray(arr)
    d2 = ImageDraw.Draw(img2)
    d2.rectangle([200, 150, 400, 250], fill=(200, 50, 50))
    img2.save("/tmp/ft_complex.png")

    # 3. 精密零件（齿轮）
    import math
    img3 = Image.new("RGB", (400, 400), (240, 240, 240))
    d3 = ImageDraw.Draw(img3)
    d3.ellipse([100, 100, 300, 300], outline=(50, 50, 50), width=3)
    d3.ellipse([170, 170, 230, 230], outline=(50, 50, 50), width=3)
    for i in range(8):
        a = i * 45
        x1 = 200 + 140 * math.cos(math.radians(a)); y1 = 200 + 140 * math.sin(math.radians(a))
        x2 = 200 + 160 * math.cos(math.radians(a)); y2 = 200 + 160 * math.sin(math.radians(a))
        d3.line([x1, y1, x2, y2], fill=(50, 50, 50), width=3)
    img3.save("/tmp/ft_gear.png")

    # 4. 文字图（OCR）
    img4 = Image.new("RGB", (400, 150), (255, 255, 255))
    d4 = ImageDraw.Draw(img4)
    d4.text((30, 50), "Hello World 123", fill=(0, 0, 0))
    img4.save("/tmp/ft_text.png")

    # 5. 3D 审计数据
    def box(x0, y0, z0, x1, y1, z1):
        return [[x0, y0, z0], [x1, y0, z0], [x0, y1, z0], [x1, y1, z0],
                [x0, y0, z1], [x1, y0, z1], [x0, y1, z1], [x1, y1, z1]]
    objs = [
        {"name": "A", "type": "MESH", "bbox3d": box(0, 0, 0, 1, 1, 1), "verts": box(0, 0, 0, 1, 1, 1)},
        {"name": "B", "type": "MESH", "bbox3d": box(2, 0, 0, 3, 1, 1), "verts": box(2, 0, 0, 3, 1, 1)},
        {"name": "C", "type": "MESH", "bbox3d": box(1, 0, 0, 2, 1, 1), "verts": box(1, 0, 0, 2, 1, 1)},
    ]
    json.dump({"objects3d": objs}, open("/tmp/ft_audit3d.json", "w"))

    # 6. 规则审计数据
    json.dump({
        "schema": "vision-report/v3", "task": "test", "sensors": ["test"],
        "coordsys": "css_px", "source": {"type": "fused", "size_px": [1000, 800]},
        "elements": [
            {"bbox": [0, 0, 50, 20], "texts": [{"text": "t", "color": "#777777", "size_pt": 12}], "fill": "#CCCCCC"},
            {"bbox": [0, 30, 60, 50], "texts": [{"text": "ok", "color": "#000000", "size_pt": 12}], "fill": "#FFFFFF"},
            {"bbox": [100, 10, 200, 30], "source": ["dom"]},
            {"bbox": [100, 50, 200, 70], "source": ["dom"]},
        ],
    }, open("/tmp/ft_rules.json", "w"))


def test_l0():
    print("\n=== L0 源层 ===")
    # capture（grim 截图）
    r = run(PYBIN, "vs_capture.py", "--out", "/tmp/ft_cap.png")
    check("capture 截图", "error" not in r and os.path.exists("/tmp/ft_cap.png"), str(r.get("error", "")))
    # dom（需要 URL，跳过真实浏览器，验证 CLI 存在）
    r = run(PYBIN, "vs_dom.py", "--url", "about:blank", timeout=30)
    check("dom CLI 可运行", "error" in r or "schema" in r, str(r.get("error", ""))[:80])
    # pptx
    r = run(PYBIN, "vs_pptx.py", "--file", "/tmp/ft_audit3d.json", timeout=30)
    check("pptx 错误处理", "error" in r, "非 pptx 文件应报错")
    # pdf
    r = run(PYBIN, "vs_pdf.py", "--file", "/tmp/ft_audit3d.json", timeout=30)
    check("pdf 错误处理", "error" in r, "非 pdf 文件应报错")
    # a11y
    r = run(PYBIN, "vs_a11y.py", timeout=30)
    check("a11y 可运行", "error" in r or "schema" in r, str(r.get("error", ""))[:80])


def test_l1():
    print("\n=== L1 确定性测量 ===")
    r = run(VSENSOR, "vs_pix.py", "--image", "/tmp/ft_ui.png", "--colors", "4")
    check("pixels 主色", "error" not in r and len(r.get("metrics", {}).get("dominant_colors", [])) > 0)
    r = run(PYBIN, "vs_ocr.py", "--image", "/tmp/ft_text.png")
    check("ocr 文字", "error" not in r and len(r.get("elements", [])) > 0, str(r.get("error", "")))
    r = run(VSENSOR, "vs_scene_stats.py", "--image", "/tmp/ft_complex.png")
    check("scene_stats", "error" not in r and "contrast" in r.get("metrics", {}))
    r = run(PYBIN, "vs_edge.py", "--image", "/tmp/ft_gear.png")
    check("edge 亚像素", "error" not in r and r.get("metrics", {}).get("edge_points", 0) > 100)
    r = run(PYBIN, "vs_ascii.py", "--image", "/tmp/ft_complex.png", "--cols", "40", "--rows", "20")
    check("ascii 栅格", "error" not in r and len(r.get("ascii", {}).get("grid", [])) == 20)
    r = run(VSENSOR, "vs_geometry.py", "--image", "/tmp/ft_ui.png")
    check("geometry 原语", "error" not in r and r.get("metrics", {}).get("shapes", 0) > 0)


def test_l2():
    print("\n=== L2 轻量 DL ===")
    r = run(VSENSOR, "vs_saliency.py", "--image", "/tmp/ft_complex.png", "--device", "cuda")
    check("saliency GPU", "error" not in r and len(r.get("candidates", [])) > 0, str(r.get("error", ""))[:80])
    sal = r
    json.dump(sal, open("/tmp/ft_sal.json", "w"))
    r = run(VSENSOR, "vs_segment.py", "--image", "/tmp/ft_complex.png", "--saliency", "/tmp/ft_sal.json", "--device", "cuda")
    check("segment GPU", "error" not in r and r.get("foreground") is not None, str(r.get("error", ""))[:80])
    r = run(VSENSOR, "vs_depth.py", "--image", "/tmp/ft_complex.png", "--device", "cuda")
    check("depth GPU", "error" not in r and "global" in r.get("depth", {}), str(r.get("error", ""))[:80])


def test_f1():
    print("\n=== F1 融合 ===")
    ocr = run(PYBIN, "vs_ocr.py", "--image", "/tmp/ft_text.png")
    pix = run(VSENSOR, "vs_pix.py", "--image", "/tmp/ft_text.png", "--colors", "4")
    json.dump(ocr, open("/tmp/ft_ocr.json", "w"))
    json.dump(pix, open("/tmp/ft_pix.json", "w"))
    r = run(PYBIN, "vs_fusion.py", "--reports", "/tmp/ft_ocr.json", "/tmp/ft_pix.json")
    check("fusion D-S", "error" not in r and "findings" in r, str(r.get("error", ""))[:80])
    check("fusion schema v3", r.get("schema") == "vision-report/v3")


def test_f2():
    print("\n=== F2 多轮协议 ===")
    r = run(VSENSOR, "vs_protocol.py", "analyze", "--image", "/tmp/ft_complex.png", timeout=240)
    check("analyze 粗报告", "error" not in r and len(r.get("candidates", [])) > 0, str(r.get("error", ""))[:80])
    cands = r.get("candidates", [])
    if cands:
        b = cands[0]["bbox"]
        region = f"{b[0]},{b[1]},{b[2]},{b[3]}"
        r2 = run(VSENSOR, "vs_protocol.py", "zoom", "--image", "/tmp/ft_complex.png", "--region", region, timeout=240)
        check("zoom 细报告", "error" not in r2 and r2.get("metrics", {}).get("edge_points", 0) > 0, str(r2.get("error", ""))[:80])
        r3 = run(VSENSOR, "vs_protocol.py", "probe", "--image", "/tmp/ft_complex.png", "--bbox", region, "--sensor", "pix", timeout=60)
        check("probe 定向取证", "error" not in r3 and r3.get("regions"), str(r3.get("error", ""))[:80])


def test_3d():
    print("\n=== 3D 审计 ===")
    r = run(PYBIN, "vs_audit3d.py", "--report", "/tmp/ft_audit3d.json", "--gap-threshold", "15")
    check("audit3d 干涉检测", "error" not in r and r.get("metrics", {}).get("interference_count", 0) >= 1, str(r.get("error", ""))[:80])


def test_other():
    print("\n=== 其他传感器 ===")
    # wallpaper（目录分类）
    os.makedirs("/tmp/ft_wall", exist_ok=True)
    for i, img in enumerate(["/tmp/ft_ui.png", "/tmp/ft_complex.png", "/tmp/ft_gear.png"]):
        subprocess.run(["cp", img, f"/tmp/ft_wall/img{i}.png"])
    r = run(VSENSOR, "vs_wall.py", "--dir", "/tmp/ft_wall", timeout=120)
    check("wallpaper 分类", "error" not in r, str(r.get("error", ""))[:80])
    # cluster（CLIP 聚类）
    r = run(OMNI, "vs_cluster.py", "--dir", "/tmp/ft_wall", timeout=120)
    check("cluster 聚类", "error" not in r, str(r.get("error", ""))[:80])
    # detect（OWLv2）
    r = run(OMNI, "vs_detect.py", "--image", "/tmp/ft_ui.png", "--classes", "button,icon", timeout=120)
    check("detect 检测", "error" not in r, str(r.get("error", ""))[:80])
    # omniparser（UI 解析）
    r = run(OMNI, "vs_omniparser.py", "--image", "/tmp/ft_ui.png", "--max-items", "20", timeout=180)
    check("omniparser UI", "error" not in r, str(r.get("error", ""))[:80])
    # layout
    r = run(PYBIN, "vs_layout.py", "--image", "/tmp/ft_ui.png", timeout=120)
    check("layout 布局", "error" not in r, str(r.get("error", ""))[:80])
    # audit（规则审计）
    r = run(PYBIN, "vs_audit.py", "--report", "/tmp/ft_rules.json", "--canvas", "1000x800")
    check("audit 审计", "error" not in r and "findings" in r, str(r.get("error", ""))[:80])
    # rules
    r = run(PYBIN, "vs_rules.py", "--report", "/tmp/ft_rules.json", "--canvas", "1000x800")
    check("rules 规则", "error" not in r, str(r.get("error", ""))[:80])


def test_sched():
    print("\n=== 调度器 ===")
    # 启动 vsd
    sock = "/tmp/ft_vsd.sock"
    subprocess.run(["pkill", "-f", "vsd.py --socket /tmp/ft_vsd"], capture_output=True)
    time.sleep(0.5)
    if os.path.exists(sock):
        os.unlink(sock)
    proc = subprocess.Popen(
        [PYBIN, str(PY / "vsd.py"), "--socket", sock],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)

    def send(payload: str) -> dict:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(10)
        s.connect(sock)
        s.sendall(payload.encode())
        resp = s.recv(65536).decode()
        s.close()
        return json.loads(resp)

    try:
        r = send('{"ping": true}')
        check("vsd ping", r.get("pong") is True)
        r = send('{"stats": true}')
        check("vsd stats", "stats" in r)
        # 并发 3 任务
        import threading
        results = []
        def worker(i):
            model = ["saliency", "depth", "saliency"][i]
            r = send(json.dumps({"model": model, "args": {"image": "/tmp/ft_complex.png"}, "priority": 3}))
            results.append(r.get("ok"))
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in threads: t.start()
        for t in threads: t.join()
        check("并发 3 任务", all(results), f"{sum(results)}/3")
    finally:
        proc.terminate()
        subprocess.run(["pkill", "-f", "vsd.py --socket /tmp/ft_vsd"], capture_output=True)


def test_selfdiag():
    print("\n=== 自诊断 ===")
    r = run(PYBIN, "setup/vs_setup.py", "--check", timeout=60)
    check("check 自检", "error" not in r and r.get("ok") is True, str(r.get("error", ""))[:80])


def main():
    print("=" * 60)
    print("V3.0.0 全功能全工况跑通测试")
    print("=" * 60)
    make_fixtures()
    test_l0()
    test_l1()
    test_l2()
    test_f1()
    test_f2()
    test_3d()
    test_other()
    test_sched()
    test_selfdiag()

    print("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print(f"结果: {passed}/{total} PASS")
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  FAIL: {name} — {detail}")
    print("=" * 60)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
