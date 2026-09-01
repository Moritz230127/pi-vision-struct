#!/usr/bin/env python3
"""vs_depth_geom.py — 真几何深度（在 Blender 解释器内运行）。

通过相机投影将所有 MESH 三角面栅格化到深度缓冲，得到**真实相机空间距离**
（非亮度梯度近似）。输出归一化深度统计：near/far/median/std/center_depth/histogram。

用法（Blender headless）：
    blender --background --python vs_depth_geom.py -- <blend> --camera <name> \
            --image <png_for_size> --resolution 480 --output <json>

若未指定 blend，则对当前已打开场景渲染深度。
深度单位：米（世界坐标相机空间 -Z，即沿视线前方距离）。
"""
import sys
import os
import json
import argparse
import math

import bpy  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]
import mathutils  # type: ignore[import-not-found]


def parse_args():
    argv = sys.argv
    if "--" in argv:
        raw = argv[argv.index("--") + 1:]
    else:
        raw = []
    ap = argparse.ArgumentParser()
    ap.add_argument("blend", nargs="?", default="")
    ap.add_argument("--camera", default="")
    ap.add_argument("--image", default="")  # 用于确定输出尺寸（可选）
    ap.add_argument("--resolution", type=int, default=480)
    ap.add_argument("--output", default="")
    return ap.parse_args(raw)


def load_image_size(path):
    try:
        import struct
        with open(path, "rb") as f:
            head = f.read(26)
        if head[:8] == b"\x89PNG\r\n\x1a\n":
            w = struct.unpack(">I", head[16:20])[0]
            h = struct.unpack(">I", head[20:24])[0]
            return w, h
    except Exception:
        pass
    return None


def rasterize(scene, cam, W, H):
    """栅格化所有可见 MESH 到相机空间深度缓冲。返回 (depth_ndarray, n_pixels)。"""
    cam_inv = cam.matrix_world.inverted()
    dg = bpy.context.evaluated_depsgraph_get()
    proj = cam.calc_matrix_camera(
        dg, x=W, y=H, scale_x=1.0, scale_y=1.0
    )
    # 世界->裁剪 = proj @ cam_inv
    vp = proj @ cam_inv

    depth = np.full((H, W), np.inf, dtype=np.float64)
    meshes_done = 0

    # 预计算像素网格坐标
    ys, xs = np.mgrid[0:H, 0:W]

    def raster_tri(px, py, z, depth):
        # 边界框
        minx = max(0, int(min(px)))
        maxx = min(W - 1, int(max(px)))
        miny = max(0, int(min(py)))
        maxy = min(H - 1, int(max(py)))
        if maxx < minx or maxy < miny:
            return
        # 局部网格
        lx = xs[miny:maxy + 1, minx:maxx + 1].astype(np.float64)
        ly = ys[miny:maxy + 1, minx:maxx + 1].astype(np.float64)
        x0, x1, x2 = px
        y0, y1, y2 = py
        z0, z1, z2 = z
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-9:
            return
        a = ((y1 - y2) * (lx - x2) + (x2 - x1) * (ly - y2)) / denom
        b = ((y2 - y0) * (lx - x2) + (x0 - x2) * (ly - y2)) / denom
        c = 1.0 - a - b
        mask = (a >= 0) & (b >= 0) & (c >= 0)
        if not mask.any():
            return
        zval = a * z0 + b * z1 + c * z2
        sub = depth[miny:maxy + 1, minx:maxx + 1]
        np.minimum.at(sub, np.where(mask), zval[mask])
        depth[miny:maxy + 1, minx:maxx + 1] = sub

    for obj in scene.objects:
        if obj.type != "MESH" or not obj.visible_get():
            continue
        mw = obj.matrix_world
        co = obj.data.vertices
        if len(co) < 3:
            continue
        # 预计算世界坐标顶点 + 相机空间 Z（向量化）
        vw = np.empty((len(co), 3), dtype=np.float64)
        for k, v in enumerate(co):
            vw[k] = (mw @ v.co)
        # 相机空间 Z（前为负）
        zc = np.array(
            [(cam_inv @ mathutils.Vector((vv[0], vv[1], vv[2], 1.0))).z for vv in vw]
        )
        # 三角化所有多边形（向量化索引）
        tri_idx = []
        for poly in obj.data.polygons:
            vi = poly.vertices
            if len(vi) < 3:
                continue
            if len(vi) == 3:
                tri_idx.append((vi[0], vi[1], vi[2]))
            else:
                for k in range(1, len(vi) - 1):
                    tri_idx.append((vi[0], vi[k], vi[k + 1]))
        if not tri_idx:
            continue
        tri_idx = np.array(tri_idx)
        # 批量投影三个顶点到像素坐标
        cw = vw[tri_idx]
        ones = np.ones((len(tri_idx), 3, 1))
        homs = np.concatenate([cw, ones], axis=2)  # (T,3,4)
        vp_np = np.array(vp)  # (4,4)
        clipped = homs @ vp_np  # (T,3,4)
        w = clipped[:, :, 3]
        valid_t = np.all(w > 1e-9, axis=1)  # 三角形三顶点都需在视锥前
        ndc = clipped[:, :, :3] / w[:, :, None]
        px_all = ((ndc[:, :, 0] * 0.5 + 0.5) * W).astype(np.int64)
        py_all = ((1.0 - (ndc[:, :, 1] * 0.5 + 0.5)) * H).astype(np.int64)
        for ti in range(len(tri_idx)):
            if not bool(valid_t[ti]):
                continue
            px = px_all[ti].astype(np.float64)
            py = py_all[ti].astype(np.float64)
            zt = (zc[tri_idx[ti][0]], zc[tri_idx[ti][1]], zc[tri_idx[ti][2]])
            raster_tri(px, py, zt, depth)
        meshes_done += 1
    return depth, meshes_done


def main():
    args = parse_args()
    if args.blend and os.path.exists(args.blend):
        bpy.ops.wm.open_mainfile(filepath=args.blend)
    scene = bpy.context.scene
    # 选相机
    cam = None
    if args.camera:
        cam = scene.objects.get(args.camera)
    if cam is None:
        cam = scene.camera
    if cam is None:
        # 取第一个相机
        for o in scene.objects:
            if o.type == "CAMERA":
                cam = o
                break
    if cam is None:
        print(json.dumps({"error": "vs_depth_geom failed", "detail": "no camera in scene"}))
        sys.exit(1)

    W = args.resolution
    if args.image:
        sz = load_image_size(args.image)
        if sz:
            W = sz[0]
    # 高度按相机宽高比
    if cam.data and cam.data.angle_y:
        aspect = cam.data.angle_x / cam.data.angle_y if cam.data.angle_x else 1.0
    else:
        aspect = 16.0 / 9.0
    H = max(1, int(W / aspect))

    depth, meshes_done = rasterize(scene, cam, W, H)

    valid = depth[np.isfinite(depth)]
    if len(valid) == 0:
        print(json.dumps({"error": "vs_depth_geom failed", "detail": "no geometry projected"}))
        sys.exit(1)

    # 相机空间距离（前方为正）：dist = -z
    dist = -valid
    near = float(dist.min())
    far = float(dist.max())
    median = float(np.median(dist))
    std = float(dist.std())
    center_depth = float(-depth[H // 2, W // 2]) if np.isfinite(depth[H // 2, W // 2]) else median

    # 归一化 0..1
    nz = np.clip(dist, near, far)
    nz = (nz - near) / (far - near + 1e-12)
    hist, edges = np.histogram(nz, bins=10, range=(0.0, 1.0))

    result = {
        "schema": "vision-report/v3",
        "task": "depth",
        "sensors": ["bpy-geometry"],
        "coordsys": "world_m",
        "source": {
            "type": "blender",
            "camera": cam.name,
            "resolution": [W, H],
            "meshes_rasterized": meshes_done,
        },
        "metrics": {
            "near_m": round(near, 6),
            "far_m": round(far, 6),
            "median_m": round(median, 6),
            "std_m": round(std, 6),
            "center_depth_m": round(center_depth, 6),
            "unit": "geometric_camera_distance_m",
            "valid_pixel_ratio": round(len(valid) / (W * H), 4),
        },
        "depth_stats": {
            "near_m": round(near, 6),
            "far_m": round(far, 6),
            "median_m": round(median, 6),
            "std_m": round(std, 6),
            "center_depth_m": round(center_depth, 6),
            "histogram": [int(x) for x in hist],
            "bin_edges": [round(float(e), 3) for e in edges],
            "shape_w": W,
            "shape_h": H,
            "unit": "geometric_camera_distance_m",
        },
        "truncated": False,
        "notation": "primitive notation: reference depth as [depth_m: value] at [point: x,y]; "
                    "geometric camera-space distance (along view axis), not luminance.",
    }

    payload = json.dumps(result, ensure_ascii=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(payload)
    else:
        sys.stdout.write(payload + "\n")
    sys.exit(0)


main()
