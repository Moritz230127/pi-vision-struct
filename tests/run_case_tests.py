#!/usr/bin/env python3
"""Phase 2 三用例验收聚合器。

依次运行 case1/case2/case3，汇总退出码。
运行: /home/Arch/conda-envs/pi-vision/bin/python tests/run_case_tests.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYBIN = "/home/Arch/conda-envs/pi-vision/bin/python"
CASES = ["case1_firefox_e2e.py", "case2_pptx_audit.py", "case3_wallpaper.py"]


def main() -> int:
    failed = 0
    for name in CASES:
        print(f"\n{'=' * 60}\n运行 {name}\n{'=' * 60}")
        r = subprocess.run([PYBIN, str(ROOT / name)], capture_output=True, text=True, timeout=600)
        print(r.stdout)
        if r.stderr:
            print(f"[stderr] {r.stderr[:500]}")
        if r.returncode != 0:
            failed += 1
            print(f"✗ {name} 失败 (exit {r.returncode})")
    print(f"\n{'=' * 60}\n聚合结果: {len(CASES) - failed}/{len(CASES)} 用例通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
