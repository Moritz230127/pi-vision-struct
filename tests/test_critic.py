#!/usr/bin/env python3
"""test_critic.py — VLM-as-critic 闭环自测（本地 HTTP 桩，确定性，不依赖 Ollama）。

覆盖：opt-in 拒绝 / 裁剪与边距 / 裁决并入 / 上限与严重度排序。

运行（conda env）:
  /home/Arch/conda-envs/pi-vision/bin/python tests/test_critic.py
"""
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / "python"
OUT = ROOT / "tests" / "_out" / "critic"
OUT.mkdir(parents=True, exist_ok=True)
PYBIN = "/home/Arch/conda-envs/pi-vision/bin/python"

PASS = 0
FAIL = 0

VERDICT = '{"verdict": "confirmed", "reason": "stub 裁决"}'


class StubHandler(BaseHTTPRequestHandler):
    calls = 0

    def do_POST(self):  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length", 0))
        except (TypeError, ValueError):
            length = 0
        self.rfile.read(length)
        StubHandler.calls += 1
        body = json.dumps({"response": VERDICT, "done_reason": "stop"}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        pass


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name} {detail}")


def run_tool(args: list[str], timeout: int = 120) -> dict:
    r = subprocess.run([PYBIN, str(PY / "vs_critic.py"), *args], capture_output=True,
                       text=True, timeout=timeout)
    if r.returncode != 0:
        raise AssertionError(f"tool exit {r.returncode}: {r.stdout[:300]} {r.stderr[:200]}")
    try:
        return json.loads(r.stdout)
    except ValueError as e:
        raise AssertionError(f"not json: {r.stdout[:300]}") from e


def make_report(elements: list[dict]) -> tuple[dict, Path]:
    report = {
        "schema": "vision-report/v2", "task": "test", "sensors": ["test"],
        "coordsys": "css_px",
        "source": {"type": "fused", "size_px": [400, 300]},
        "elements": elements,
        "findings": [
            {"rule": "text_contrast", "severity": "critical", "bbox": [20, 20, 120, 60],
             "suggested_cause": "低对比", "evidence": {"ratio": 2.0}},
            {"rule": "spacing_anomaly", "severity": "info", "bbox": [200, 20, 300, 60],
             "suggested_cause": "间距离群", "evidence": {"gap": 120.0}},
        ],
    }
    p = OUT / "report.json"
    p.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    im = Image.new("RGB", (400, 300), "#FFFFFF")
    im.save(OUT / "img.png")
    return report, p


def test_optin_refusal() -> None:
    print("[vs_critic] opt-in 拒绝（默认不调 VLM）")
    _, p = make_report([])
    out = run_tool(["--report", str(p), "--image", str(OUT / "img.png")])
    check("critic.enabled = false", out.get("critic", {}).get("enabled") is False)
    check("findings 未被改写", out["findings"][0].get("critic") is None)
    check("无异常", "error" not in out)


def test_crop_and_verdict() -> None:
    print("[vs_critic] 裁剪 + 裁决并入（本地桩）")
    StubHandler.calls = 0
    srv = ThreadingHTTPServer(("127.0.0.1", 0), StubHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _, p = make_report([])
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        out = run_tool(["--report", str(p), "--image", str(OUT / "img.png"),
                        "--enable", "--base-url", base, "--max-tokens", "64"])
    finally:
        srv.shutdown()
    crit = out["findings"][0].get("critic") or {}
    check("裁了 2 个区（桩收到 2 次）", StubHandler.calls == 2, f"calls={StubHandler.calls}")
    check("裁决并入 finding", crit.get("verdict") == "confirmed" and crit.get("reason") == "stub 裁决")
    check("critic 统计", out.get("critic", {}).get("confirmed") == 2)
    check("裁决带模型/耗时", "model" in crit and "ms" in crit)


def test_cap_and_order() -> None:
    print("[vs_critic] --max-critic 上限 + 严重度排序")
    StubHandler.calls = 0
    srv = ThreadingHTTPServer(("127.0.0.1", 0), StubHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _, p = make_report([])
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        out = run_tool(["--report", str(p), "--image", str(OUT / "img.png"),
                        "--enable", "--base-url", base, "--max-critic", "1", "--max-tokens", "64"])
    finally:
        srv.shutdown()
    checked = out.get("critic", {}).get("checked", 0)
    crit_ids = [f["rule"] for f in out["findings"] if f.get("critic")]
    check("只裁 1 个区", checked == 1 and StubHandler.calls == 1, f"checked={checked}")
    check("优先裁 critical（text_contrast）", crit_ids == ["text_contrast"], str(crit_ids))


def test_missing_image() -> None:
    print("[vs_critic] 图片缺失报错")
    _, p = make_report([])
    r = subprocess.run([PYBIN, str(PY / "vs_critic.py"),
                        "--report", str(p), "--image", str(OUT / "nope.png"), "--enable"],
                       capture_output=True, text=True, timeout=60)
    try:
        out = json.loads(r.stdout)
    except ValueError as e:
        raise AssertionError(f"not json: {r.stdout[:200]}") from e
    check("返回 error 且退出码 1", r.returncode == 1 and "error" in out,
          f"rc={r.returncode}")


def main() -> int:
    print(f"critic self-tests (python {sys.version.split()[0]})\n")
    test_optin_refusal()
    test_crop_and_verdict()
    test_cap_and_order()
    test_missing_image()
    print(f"\n结果: {PASS} 通过 / {FAIL} 失败")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
