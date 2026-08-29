#!/usr/bin/env bash
# depth_geom_wrapper.sh — 真几何深度包装器
# 用法: depth_geom_wrapper.sh <blend> --camera <name> --image <png> --output <json>
# 在 Blender 内运行 vs_depth_geom.py（栅格化真实相机空间深度）；Blender 日志→/dev/null
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BLEND="$1"; shift
OUT=/tmp/depth_geom_out.json
blender --background --python "$SCRIPT_DIR/vs_depth_geom.py" -- "$BLEND" "$@" --output "$OUT" 1>/dev/null 2>/dev/null
cat "$OUT" 2>/dev/null
