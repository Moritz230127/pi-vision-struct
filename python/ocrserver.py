#!/usr/bin/env python3
"""ocrserver.py — PaddleOCR 常驻服务（本地，模型只加载一次）。

vs_ocr --backend paddle 每次冷载 PP-OCRv6 medium（~7-40s）；本服务把引擎驻留内存，
调用降为纯推理（亚秒级）。仅 unix socket 文件 IPC，无 TCP 端口，与沙箱兼容。

API（unix socket，行分隔 JSON）:
  {"ping": true}                                  → {"ok": true, "pid": N}
  {"image": "/abs/path.png", "min_conf": 0.5}     → {"ok": true, "items": [...]}
    items 与 vs_ocr.paddle_predict 输出同构

socket: ~/.cache/vs-ocr/ocrserver.sock
启动（通常由 vs_ocr --daemon auto 自动拉起，无需手动）:
  setsid -f <pi-vision-python> -u python/ocrserver.py >> ~/.cache/vs-ocr/daemon.log 2>&1
停止: pkill -f 'ocrserve[r].py'
"""
import json
import os
import socketserver
import sys

CACHE_DIR = f"{os.path.expanduser('~')}/.cache/vs-ocr"
SOCKET_PATH = f"{CACHE_DIR}/ocrserver.sock"


class Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        line = self.rfile.readline()
        if not line:
            return
        try:
            req = json.loads(line.decode())
            if req.get("ping"):
                out = {"ok": True, "pid": os.getpid()}
            else:
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from vs_ocr import paddle_predict

                items = paddle_predict(req["image"],
                                       min_conf=float(req.get("min_conf", 0.5)))
                out = {"ok": True, "items": items}
        except Exception as e:
            out = {"ok": False, "error": str(e)[:500]}
        try:
            self.wfile.write((json.dumps(out, ensure_ascii=False) + "\n").encode())
        except OSError:
            pass


class Server(socketserver.ThreadingUnixStreamServer):
    allow_reuse_address = True


def main() -> int:
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)  # 清理残留 socket（崩溃后重启）
    srv = Server(SOCKET_PATH, Handler)
    os.chmod(SOCKET_PATH, 0o600)
    print(f"ocrserver ready on {SOCKET_PATH}", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
