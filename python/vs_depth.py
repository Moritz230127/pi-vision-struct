#!/usr/bin/env python3
"""vs_depth.py — 深度估计/深度渲染（L1 数理化）。

模式一：从 Blender 直接渲染 Z-pass（精确深度，G6 门禁专用）
  blender --background file.blend --python vs_depth.py -- --blender-zpass --camera NAME --image renders/xxx.png --output depth.json

模式二：单图深度估算（CPU-only，无 torch 依赖，近似分布）
  vs_depth.py --image renders/xxx.png

输出：depth_stats（near/far/median/mean/std/histogram/shape）+ center_depth
全部数值，单位标注（relative 或 relative_to_blender）。
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from PIL import Image

import vs_schema as S


def depth_from_luminance(im: Image.Image) -> np.ndarray:
    """基于亮度梯度的近似深度：暗→近，亮→远（逆深度近似）"""
    gray = np.array(im.convert("L"), dtype=np.float64)
    # 逆深度：255 - luminance，近处暗、远处亮
    depth = 255.0 - gray
    # 归一化到 0-1
    dmin, dmax = depth.min(), depth.max()
    if dmax > dmin:
        depth = (depth - dmin) / (dmax - dmin)
    return depth


def render_blender_zpass(blend_path: str, camera_name: str = "", output_pgm: str = "/tmp/zpass.pgm"):
    """用 Blender 渲染一张深度图（Z-pass），返回 PGM 路径"""
    import subprocess
    # 生成 Blender Python 脚本
    bpy_script = f'''
import bpy
import sys
import mathutils

# 加载 .blend
bpy.ops.wm.open_mainfile(filepath="{blend_path}")

scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = 1024
scene.render.resolution_y = 576
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "OPEN_EXR"

# 设置摄像机
cam_name = "{camera_name}"
cam_obj = None
if cam_name:
    cam_obj = scene.objects.get(cam_name)
if cam_obj:
    scene.camera = cam_obj
elif scene.camera is None and scene.objects:
    for o in scene.objects:
        if o.type == "CAMERA":
            scene.camera = o
            break

# 深度通道
scene.render.image_settings.use_zbuffer = True

# 渲染
output_path = "{output_pgm}"
bpy.ops.render.render(write_still=False)
# 获取深度数据
depth_img = bpy.data.images.get("Render Result")
if depth_img is None:
    depth_img = bpy.data.images.get("render_result")
if depth_img:
    depth_img.save_render(output_path)
else:
    # Fallback：写空文件
    with open(output_path + ".err", "w") as f:
        f.write("Render Result not found")
'''
    bpy_script_path = "/tmp/blender_zpass.py"
    with open(bpy_script_path, "w") as f:
        f.write(bpy_script)

    # 运行 Blender（stderr 静默，避免日志污染）
    result = subprocess.run(
        ["blender", "--background", "--python", bpy_script_path],
        capture_output=True, text=True, timeout=60
    )
    return output_pgm, result.returncode == 0, result.stderr[-500:] if result.stderr else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="输入图片路径（模式二）")
    ap.add_argument("--blender-zpass", action="store_true", help="从 Blender 渲染 Z-pass（精确）")
    ap.add_argument("--blend", help="Blender 文件路径（配合 --blender-zpass）")
    ap.add_argument("--camera", default="", help="摄像机名称（配合 --blender-zpass）")
    ap.add_argument("--output", help="输出 JSON 路径")
    ap.add_argument("--stats-only", action="store_true")
    args = ap.parse_args()

    try:
        report = None

        if args.blender_zpass and args.blend:
            # 模式一：Blender Z-pass（精确）
            pgm_path, ok, err = render_blender_zpass(args.blend, args.camera)
            if ok and os.path.exists(pgm_path):
                # 读取 PGM（Blender 渲染的深度图）
                from PIL import PngImagePlugin
                im_depth = Image.open(pgm_path).convert("L")
                depth_arr = np.array(im_depth, dtype=np.float64)
                # PGM 像素值是 z-buffer 的 16-bit 量化，转换为相对深度
                depth_arr = depth_arr / 255.0
                unit = "relative_to_blender"
            else:
                # Fallback：用图像亮度估算
                if args.image:
                    im = Image.open(args.image).convert("RGB")
                    depth_arr = depth_from_luminance(im)
                    unit = "relative_luminance"
                else:
                    return 1
        elif args.image:
            # 模式二：亮度梯度近似
            im = Image.open(args.image).convert("RGB")
            depth_arr = depth_from_luminance(im)
            unit = "relative_luminance"
        else:
            print(json.dumps({"error": "需要 --image 或 --blender-zpass --blend"}))
            return 1

        h, w = depth_arr.shape
        depth_min = float(depth_arr.min())
        depth_max = float(depth_arr.max())
        depth_mean = float(depth_arr.mean())
        depth_median = float(np.median(depth_arr))
        depth_std = float(depth_arr.std())

        hist, bin_edges = np.histogram(depth_arr.flatten(), bins=10)

        stats = {
            "near": round(depth_min, 6),
            "far": round(depth_max, 6),
            "mean": round(depth_mean, 6),
            "median": round(depth_median, 6),
            "std": round(depth_std, 6),
            "depth_range": round(depth_max - depth_min, 6),
            "histogram": [int(c) for c in hist],
            "bin_edges": [round(v, 6) for v in bin_edges],
            "shape_h": h,
            "shape_w": w,
            "unit": unit,
        }

        report = S.envelope(task="depth", sensors=["luminance" if "luminance" in unit else "Blender_ZPass"],
                            coordsys="image_px",
                            source={"type": "image", "path": args.image or "(blender render)"})
        report["notation"] = S.NOTATION_GUIDE
        report["depth_stats"] = stats
        report["metrics"] = stats

        cy, cx = h // 2, w // 2
        sample_size = min(32, h // 8, w // 8)
        center_patch = depth_arr[cy - sample_size:cy + sample_size,
                                  cx - sample_size:cx + sample_size]
        report["center_depth"] = round(float(center_patch.mean()), 6)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(S.dump_json(report))
        else:
            print(S.dump_json(report))
        return 0
    except Exception as e:
        import traceback
        print(json.dumps({"error": "vs_depth failed", "detail": str(e)[:500],
                          "traceback": traceback.format_exc()[:500]}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
