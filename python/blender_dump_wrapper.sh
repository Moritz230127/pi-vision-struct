#!/usr/bin/env bash
# blender_dump_wrapper.sh — Blender 场景图导出包装器
# 用法: blender_dump_wrapper.sh <blend_file> [--output out.json]
# 产出 JSON 写入 /tmp/blender_dump_out.json（dispatch 从 stdout 读取）
BLEND="$1"; shift
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT=/tmp/blender_dump_out.json
# Blender 的日志 → /dev/null，JSON → 文件
blender --background --python "$SCRIPT_DIR/vs_blender_dump.py" -- "$BLEND" --output "$OUT" 1>/dev/null 2>/dev/null
# 将 JSON 输出到 stdout 供 dispatch 消费
cat "$OUT" 2>/dev/null

