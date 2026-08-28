#!/usr/bin/env bash
# depth_wrapper.sh — 深度估算包装器（CPU-only，无 torch）
# 用法: depth_wrapper.sh --image <image>
# 始终使用 pi-vision 环境的 Python（避免 Blender Python 缺少 PIL）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$SCRIPT_DIR/vs_depth.py" "$@"

