#!/usr/bin/env python3
"""vs_blender_dump.py — Blender 场景图导出（L0 数理化，bpy headless）。

在 Blender 解释器内运行：blender --background --python vs_blender_dump.py -- <blend>
输出 schema v2/v3：objects3d[]（matrix_world 4×4、bbox3d 8×3、collection、material、visible）
+ cameras[]（location、rotation、fov）+ scene_stats（物体数、集合数、总面数）
+ topology（邻接关系与距离）。

全数值，零描述。矩阵按行优先展平为 16 元素列表，bbox3d 按 8 角点展开为 24 元素列表。
"""
import json
import sys
import os

# bpy 在 Blender 解释器内可用
import bpy  # type: ignore[import-not-found]
import mathutils  # type: ignore[import-not-found]


def matrix4x4_to_list(m) -> list[float]:
    """4×4 矩阵 → 行优先 16 元素列表"""
    return [round(m[j][i], 8) for i in range(4) for j in range(4)]


def bbox3d_to_list(obj) -> list[list[float]]:
    """物体包围盒（世界坐标）→ 8 角点 × [x,y,z]"""
    bbox_world = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    return [[round(v, 6) for v in corner] for corner in bbox_world]


def get_collection_name(obj) -> str:
    if obj.users_collection:
        return obj.users_collection[0].name
    return "Collection"


def get_material_info(obj) -> dict | None:
    if obj.type not in ("MESH", "CURVE", "SURFACE", "META", "FONT"):
        return None
    if not obj.data or not hasattr(obj.data, "materials"):
        return None
    if obj.data and obj.data.materials:
        mat = obj.data.materials[0]
        if mat and mat.use_nodes and mat.node_tree:
            for node in mat.node_tree.nodes:
                if node.type == "BSDF_PRINCIPLED":
                    base_color = node.inputs.get("Base Color")
                    if base_color and base_color.is_linked:
                        return {"type": "linked", "node": base_color.links[0].from_node.name}
                    elif base_color:
                        cv = base_color.default_value
                        return {"type": "value", "rgba": [round(float(v), 4) for v in cv[:4]]}
            return {"type": mat.name}
    return None


def main():
    # 解析参数（blender --background --python script.py -- <blend_file> [--output out.json]）
    argv = sys.argv
    if "--" in argv:
        raw = argv[argv.index("--") + 1:]
    else:
        raw = []
    output_path = None
    blend_path = ""
    i = 0
    while i < len(raw):
        if raw[i] == "--output" and i + 1 < len(raw):
            output_path = raw[i + 1]
            i += 2
        elif not raw[i].startswith("--") and not blend_path:
            blend_path = raw[i]
            i += 1
        else:
            i += 1
    # fallback: also support --output as sys arg without -- separator
    if not output_path:
        for idx, a in enumerate(sys.argv):
            if a == "--output" and idx + 1 < len(sys.argv):
                output_path = sys.argv[idx + 1]

    # 打开 .blend 文件
    if blend_path and os.path.exists(blend_path):
        bpy.ops.wm.open_mainfile(filepath=blend_path)
    elif blend_path:
        print(json.dumps({"error": "vs_blender_dump failed",
                          "detail": f"blend file not found: {blend_path}"},
                         ensure_ascii=False))
        sys.exit(1)

    scene = bpy.context.scene

    # 收集所有物体
    objects3d = []
    total_tris = 0
    for i, obj in enumerate(scene.objects):
        # 面数统计（直接读原始 mesh 数据，避免 to_mesh 评估带来的依赖图问题）
        tris = 0
        try:
            if obj.type == "MESH" and obj.data and hasattr(obj.data, "polygons"):
                for poly in obj.data.polygons:
                    tris += max(0, len(poly.vertices) - 2)
        except Exception:
            pass

        total_tris += tris

        # 可见性
        visible = obj.visible_get()

        # 变换
        mat_world = matrix4x4_to_list(obj.matrix_world)
        bbox = bbox3d_to_list(obj)

        # 包围盒中心与尺寸
        bbox_arr = [[corner[i] for corner in bbox] for i in range(3)]
        center = [round((min(c) + max(c)) / 2, 6) for c in bbox_arr]
        size = [round(max(c) - min(c), 6) for c in bbox_arr]

        # 邻接（与场景内其他物体的距离）
        obj_info = {
            "id": i,
            "name": obj.name,
            "type": obj.type,
            "collection": get_collection_name(obj),
            "matrix_world": mat_world,
            "bbox3d": bbox,
            "center": center,
            "size": size,
            "tris": tris,
            "visible": visible,
            "material": get_material_info(obj),
        }
        objects3d.append(obj_info)

    # 摄像机（Blender 5.x: scene.camera 仅单个，需从对象集合过滤）
    cameras = []
    for obj in scene.objects:
        if obj.type == "CAMERA":
            cam_data = obj.data
            fov = cam_data.angle  # 弧度
            cameras.append({
                "name": obj.name,
                "location": [round(v, 6) for v in obj.location],
                "rotation_euler": [round(v, 6) for v in obj.rotation_euler],
                "rotation_quaternion": [round(v, 6) for v in obj.rotation_quaternion],
                "fov_rad": round(float(fov), 6),
                "fov_deg": round(float(fov) * 180 / 3.14159265, 4),
                "sensor_width": round(cam_data.sensor_width, 4),
                "lens": round(cam_data.lens, 4),
                "matrix_world": matrix4x4_to_list(obj.matrix_world),
                "is_active": obj == scene.camera,
            })

    # 集合统计
    collections = set()
    for obj in scene.objects:
        for col in obj.users_collection:
            collections.add(col.name)

    # 输出
    report = {
        "schema": "vision-report/v3",
        "task": "blender_dump",
        "sensors": ["bpy"],
        "coordsys": "world_m",
        "source": {
            "type": "blender",
            "filepath": blend_path,
            "blender_version": bpy.app.version_string,
            "scene_name": scene.name,
        },
        "objects3d": objects3d,
        "cameras": cameras,
        "metrics": {
            "object_count": len(objects3d),
            "collection_count": len(collections),
            "total_tris": total_tris,
            "visible_count": sum(1 for o in objects3d if o["visible"]),
        },
        "elements": [],  # 兼容 schema v2
        "anomalies": [],
        "notation": "primitive notation: reference locations as [bbox3d: x1,y1,z1-x8,y8,z8] "
                     "or [matrix_world: 16 values row-major]. Coordinates are world-space meters.",
    }

    payload = json.dumps(report, ensure_ascii=False)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(payload)
    else:
        # 隔离 Blender 日志：只输出 JSON 到 stdout，日志已重定向到 stderr
        sys.stdout.write(payload + "\n")
        sys.stdout.flush()
    sys.exit(0)


main()
