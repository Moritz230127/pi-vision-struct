#!/usr/bin/env python3
"""vs_capture.py — Wayland 截图（grim），返回结构化元数据。

用法:
  vs_capture.py --out PATH            # 全屏
  vs_capture.py --out PATH --region x1,y1,x2,y2   # 区域（像素）
  vs_capture.py --out PATH --window   # 当前聚焦窗口（grim -l 逻辑模式，可能不可用）

依赖: grim（Wayland 原生）。输出 JSON。
"""
import argparse
import json
import subprocess
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--region")
    ap.add_argument("--timeout", type=int, default=15)
    args = ap.parse_args()

    cmd = ["grim"]
    if args.region:
        x1, y1, x2, y2 = map(int, args.region.split(","))
        cmd += ["-g", f"{min(x1,x2)},{min(y1,y2)},{abs(x2-x1)},{abs(y2-y1)}"]
    cmd.append(args.out)

    try:
        subprocess.run(cmd, check=True, timeout=args.timeout, capture_output=True)
    except Exception as e:
        print(json.dumps({"error": "grim failed", "detail": str(e)[:300]}, ensure_ascii=False))
        return 1

    from PIL import Image
    im = Image.open(args.out)
    print(json.dumps({"schema": "vision-report/v1",
                      "source": {"type": "screenshot", "path": args.out, "size_px": list(im.size)}},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
