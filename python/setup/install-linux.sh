#!/usr/bin/env bash
# install-linux.sh — pi-vision-struct 安装器（Linux）
# 用法: bash install-linux.sh [--with-omniparser] [--with-dom] [--proxy URL]
# 依赖: conda/mamba（无则自动提示安装 miniforge）
set -e
cd "$(dirname "$0")/../.."

PYTHON="$(command -v python3 || true)"
if [ -z "$PYTHON" ]; then
  echo "✗ 需要 python3（安装 miniforge 后会自动获得）" >&2
  exit 1
fi

exec "$PYTHON" -u python/setup/vs_setup.py "$@"
