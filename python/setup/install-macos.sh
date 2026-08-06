#!/usr/bin/env bash
# install-macos.sh — pi-vision-struct 安装器（macOS）
# 用法: bash install-macos.sh [--with-omniparser] [--with-dom]
# 依赖: conda/mamba（建议 brew install --cask miniforge）；截图用内置 screencapture
set -e
cd "$(dirname "$0")/../.."

PYTHON="$(command -v python3 || true)"
if [ -z "$PYTHON" ]; then
  echo "✗ 需要 python3（安装 miniforge 后会自动获得）" >&2
  exit 1
fi

exec "$PYTHON" -u python/setup/vs_setup.py "$@"
