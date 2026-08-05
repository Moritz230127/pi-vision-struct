#!/usr/bin/env bash
# vs_bwrap.sh — 严格本地化沙箱包装（bubblewrap，零网络 + 只读根）。
#
# 用内核命名空间把工具进程物理隔离：
#   --unshare-net  无任何网络（连回环都没有）——进程无法建立任何 TCP/UDP
#   --ro-bind / /  整个根只读（防篡改/防写入系统）
#   --tmpfs        工作目录每次全新（防跨调用残留）
#   可写白名单:   ~/.cache/omniparser（omniserver unix socket + 权重）
#
# 用法: vs_bwrap.sh <命令...>      例如 vs_bwrap.sh <python> script.py args
# 不可用（无 bwrap）时自动退化为直接执行（工具仍可用，但无隔离）。
set -u

BWRAP="$(command -v bwrap 2>/dev/null || true)"
if [ -z "$BWRAP" ] || [ "${VS_NO_SANDBOX:-0}" = "1" ]; then
  exec "$@"
fi

HOME_DIR="$HOME"
OMNI_CACHE="$HOME/.cache/omniparser"
RUNTIME="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

exec "$BWRAP" --unshare-net \
  --ro-bind / / \
  --proc /proc \
  --dev /dev \
  --tmpfs /tmp \
  --tmpfs /run \
  --tmpfs /var/tmp \
  --bind "$OMNI_CACHE" "$OMNI_CACHE" \
  --bind "$RUNTIME" "$RUNTIME" \
  "$@"
