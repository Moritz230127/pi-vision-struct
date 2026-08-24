#!/usr/bin/env python3
"""vs_capture.py — 跨平台截图（grim / screencapture / mss），返回结构化元数据。

平台策略:
  Linux+Wayland  grim（原生，首选）
  macOS         screencapture（内置，-x 静默）
  Windows/X11   mss（可选依赖，未装时明确报错）

用法:
  vs_capture.py --out PATH [--region x1,y1,x2,y2] [--timeout 15]
"""
import argparse
import json
import vs_schema as S
import shutil
import subprocess
import sys


def capture_grim(args) -> bool:
    cmd = ["grim"]
    if args.region:
        x1, y1, x2, y2 = (int(v) for v in args.region.split(","))
        cmd += ["-g", f"{min(x1,x2)},{min(y1,y2)},{abs(x2-x1)},{abs(y2-y1)}"]
    cmd.append(args.out)
    try:
        subprocess.run(cmd, check=True, timeout=args.timeout, capture_output=True)
        return True
    except Exception:
        return False


def capture_macos(args) -> bool:
    cmd = ["screencapture", "-x"]
    if args.region:
        x1, y1, x2, y2 = (int(v) for v in args.region.split(","))
        cmd += ["-R", f"{x1},{y1},{abs(x2-x1)},{abs(y2-y1)}"]
    cmd.append(args.out)
    try:
        subprocess.run(cmd, check=True, timeout=args.timeout, capture_output=True)
        return True
    except Exception:
        return False


def capture_mss(args) -> bool:
    try:
        import mss  # type: ignore[import-not-found]
    except ImportError:
        return False
    mon = None
    if args.region:
        x1, y1, x2, y2 = (int(v) for v in args.region.split(","))
        mon = {"left": min(x1, x2), "top": min(y1, y2),
               "width": abs(x2 - x1), "height": abs(y2 - y1)}
    with mss.mss() as sct:
        shot = sct.grab(mon) if mon else sct.grab(sct.monitors[0])
        from PIL import Image  # type: ignore[import-not-found]

        Image.frombytes("RGB", shot.size, shot.rgb).save(args.out)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--region")
    ap.add_argument("--timeout", type=int, default=15)
    args = ap.parse_args()

    detail = ""
    if sys.platform == "darwin":
        ok = capture_macos(args)
        detail = "screencapture"
    elif sys.platform == "win32":
        ok = capture_mss(args)
        detail = "mss"
    else:  # linux
        if shutil.which("grim"):
            ok = capture_grim(args)
            detail = "grim"
        else:
            ok = capture_mss(args)
            detail = "mss"

    if not ok:
        print(S.dump_json({
            "error": "capture failed",
            "detail": f"后端 {detail} 不可用或执行失败。Linux 请装 grim（Wayland）或 mss；"
                      f"macOS 内置 screencapture；Windows 需 `pip install mss`",
        }))
        return 1

    from PIL import Image  # type: ignore[import-not-found]

    im = Image.open(args.out)
    print(S.dump_json({"schema": "vision-report/v1",
                      "source": {"type": "screenshot", "path": args.out,
                                 "size_px": list(im.size), "backend": detail}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
