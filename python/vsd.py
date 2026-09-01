#!/usr/bin/env python3
"""vsd.py — V3 统一视觉 daemon（模型驻留 + 调度 + unix socket）。

单进程管理 L2 全部 DL 传感器（saliency/segment/depth），经 vsched 调度
（显存预算 + 并发 + LRU + 功耗感知）。unix socket 行分隔 JSON 协议。

协议:
  {"model": "saliency", "args": {"image": "..."}, "priority": 0}
  → {"ok": true, "report": {...}}

用法:
  vsd.py --socket /tmp/vsd.sock [--device cuda]
"""
import argparse
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

import vsched

SOCKET_PATH = f"{Path.home()}/.cache/vsensor/vsd.sock"


def make_infer_fn(script: str, pybin: str):
    """构造推理函数：调用传感器脚本（子进程）。"""
    import subprocess

    def infer(**kwargs) -> dict:
        cmd = [pybin, str(Path(__file__).resolve().parent / script)]
        for k, v in kwargs.items():
            if v is not None and v is not False:
                cmd.append(f"--{k}")
                if v is not True:
                    cmd.append(str(v))
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        try:
            return json.loads(r.stdout)
        except json.JSONDecodeError:
            return {"error": f"{script} 输出非 JSON", "detail": (r.stdout or r.stderr)[:300]}
    return infer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", default=SOCKET_PATH)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--pybin", default=None, help="vsensor env python 路径")
    args = ap.parse_args()

    # vsensor env python
    pybin = args.pybin or f"{Path.home()}/conda-envs/vsensor/bin/python"

    sched = vsched.Scheduler()
    # 注册传感器（推理函数 = 子进程调用）
    sched.register("saliency", make_infer_fn("vs_saliency.py", pybin))
    sched.register("segment", make_infer_fn("vs_segment.py", pybin))
    sched.register("depth", make_infer_fn("vs_depth.py", pybin))

    # unix socket 服务
    os.makedirs(Path(args.socket).parent, exist_ok=True)
    if os.path.exists(args.socket):
        os.unlink(args.socket)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(args.socket)
    server.listen(8)
    print(f"[vsd] listening on {args.socket} (device={args.device})", flush=True)

    def handle(conn):
        try:
            with conn:
                data = conn.recv(65536)
                if not data:
                    return
                try:
                    req = json.loads(data.decode("utf-8"))
                except json.JSONDecodeError:
                    conn.sendall(b'{"ok": false, "error": "invalid JSON"}\n')
                    return
                if req.get("ping"):
                    conn.sendall(b'{"ok": true, "pong": true}\n')
                    return
                if req.get("stats"):
                    conn.sendall((json.dumps({"ok": True, "stats": sched.stats()}) + "\n").encode())
                    return
                model = req.get("model")
                args_dict = req.get("args", {})
                priority = req.get("priority", vsched.PRIORITY["probe"])
                result: dict = {}

                def done(r):
                    nonlocal result
                    result = r

                sched.submit(model, args_dict, priority=priority, callback=done)
                # 等待完成（简单同步等待；生产可改异步）
                deadline = time.time() + 200
                while not result and time.time() < deadline:
                    time.sleep(0.05)
                if not result:
                    result = {"error": "vsd timeout"}
                conn.sendall((json.dumps({"ok": "error" not in result, "report": result}) + "\n").encode())
        except Exception as e:
            try:
                conn.sendall((json.dumps({"ok": False, "error": str(e)[:300]}) + "\n").encode())
            except OSError:
                pass

    import time
    while True:
        conn, _ = server.accept()
        threading.Thread(target=handle, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    sys.exit(main())
