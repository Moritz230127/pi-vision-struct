#!/usr/bin/env python3
"""test_sandbox.py — 严格本地化沙箱自测（bwrap：零网络 + 只读根）。

验证:
  1. 沙箱内物理断网（连回环都没有）
  2. 沙箱内工具正常工作（ocr / rules / omniparser via unix socket）
  3. 无 bwrap 时优雅退化（直接执行）

运行（omniparser env 有 bwrap 依赖测试）:
  bash python/setup/vs_bwrap.sh /home/Arch/conda-envs/pi-vision/bin/python tests/test_sandbox.py
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / "python"
BWRAP = PY / "setup" / "vs_bwrap.sh"
P1 = "/home/Arch/conda-envs/pi-vision/bin/python"
O1 = "/home/Arch/conda-envs/omniparser/bin/python"

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


def run(args: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def test_egress_blocked() -> None:
    print("[沙箱] 网络物理阻断")
    r = run(["bash", str(BWRAP), "bash", "-c",
             "timeout 5 curl -s https://example.com 2>&1 || true"])
    check("外网 curl 无内容", "html" not in r.stdout.lower() and len(r.stdout.strip()) == 0,
          f"got {r.stdout[:60]!r}")
    r2 = run(["bash", str(BWRAP), "python3", "-c",
              "import socket; s=socket.socket(); s.settimeout(2);\n"
              "s.connect(('127.0.0.1', 11434))"])
    check("回环连接失败（无网络命名空间）", r2.returncode != 0, f"rc={r2.returncode}")


def test_tools_in_sandbox() -> None:
    print("[沙箱] 工具正常运行")
    r = run(["bash", str(BWRAP), P1, str(PY / "vs_ocr.py"),
             "--image", str(ROOT / "tests/bench_fixtures/text_lines.png"),
             "--max-items", "5"])
    d = json.loads(r.stdout)
    check("vs_ocr 沙箱内 ≥5 元素", len(d.get("elements", [])) >= 5,
          f"got {len(d.get('elements', []))}")

    r = run(["bash", str(BWRAP), P1, str(PY / "vs_rules.py"),
             "--report", str(ROOT / "bench/samples/s01_low_contrast.report.json")])
    d = json.loads(r.stdout)
    check("vs_rules 沙箱内 1 finding", len(d.get("findings", [])) == 1)

    # omniparser: unix socket IPC 跨命名空间（daemon 可能未运行 → 自动拉起需模型加载）
    import os
    if os.path.exists(f"{os.path.expanduser('~')}/.cache/omniparser/omniserver.sock"):
        r = run(["bash", str(BWRAP), O1, str(PY / "vs_omniparser.py"),
                 "--image", str(ROOT / "tests/bench_fixtures/layout_page.png"),
                 "--max-items", "3"], timeout=400)
        if r.returncode == 0:
            d = json.loads(r.stdout)
            check("vs_omniparser 沙箱内经 unix socket ≥1 元素",
                  len(d.get("elements", [])) >= 1)
            check("engine 为 daemon", d["source"]["engine"].endswith("daemon"),
                  d["source"]["engine"])
        else:
            check("vs_omniparser 沙箱内可用", False, r.stdout[:120] + r.stderr[:120])
    else:
        print("  （跳过：omniserver 未运行）")


def test_no_bwrap_fallback() -> None:
    print("[沙箱] 无 bwrap 退化（VS_NO_SANDBOX=1 → 直连）")
    env = {"VS_NO_SANDBOX": "1"}
    r = subprocess.run(["bash", str(BWRAP), P1, "-c",
                        "import PIL; print('PIL', PIL.__version__)"],
                       capture_output=True, text=True, timeout=60, env=env)
    check("VS_NO_SANDBOX=1 直连执行", r.returncode == 0 and "PIL" in r.stdout,
          f"rc={r.returncode} {r.stderr[:100]}")


def main() -> int:
    print(f"sandbox self-tests (python {sys.version.split()[0]})\n")
    test_egress_blocked()
    test_tools_in_sandbox()
    test_no_bwrap_fallback()
    print(f"\n结果: {PASS} 通过 / {FAIL} 失败")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
