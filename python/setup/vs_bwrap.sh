#!/usr/bin/env bash
# vs_bwrap.sh — 本地化沙箱包装（bubblewrap，零网络 + 系统只读）。
#
# 用内核命名空间把工具进程物理隔离：
#   --unshare-net  无任何网络（连回环都没有）——进程无法建立任何 TCP/UDP
#   --ro-bind / /  系统根只读（防篡改/防写入系统路径）
#   --bind /tmp    主机 /tmp 可见可写（截图/OCR 输入输出均在此，2026-08-09 修复）
#   --bind $HOME   用户主目录可写（vs_capture --out 等输出到工作区必需）
#   --tmpfs /run   每次全新（防跨调用残留）
#   可写白名单:   /tmp、$HOME、~/.cache/omniparser（omniserver unix socket + 权重）
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
  --bind /tmp /tmp \
  --bind /var/tmp /var/tmp \
  --bind "$HOME" "$HOME" \
  --tmpfs /run \
  --bind "$OMNI_CACHE" "$OMNI_CACHE" \
  --bind "$RUNTIME" "$RUNTIME" \
  "$@"
