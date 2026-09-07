//! 3D 传感器（对应 Python `vs_blender_dump.py`：bpy → blend-rs 纯 Rust 解析）。
//!
//! 解析 .blend 场景图：objects3d[]（matrix_world 4×4、bbox3d 8×3、tris、
//! 世界坐标顶点）+ cameras[]（location、rotation、fov）+ scene_stats。
//!
//! 边界决策点（plan §4.4 选项 A）：blend-rs 纯 Rust 解析。blend-rs 0.3.0
//! 支持 Blender 3.2（默认 feature）；其他版本需启用对应 feature。

use serde_json::{json, Value};

use blend_rs::blend::traverse::Named;
use blend_rs::blend::{read, PointerLike, Reader};
use blend_rs::blender3_2::{Camera, MVert, Mesh, Object, Scene};

/// 3D 参数。
pub struct BlendParams {
    /// .blend 文件路径。
    pub file: String,
}

/// Blender 对象类型 → Python 字符串（bpy obj.type）。
fn obj_type_str(t: i16) -> &'static str {
    match t {
        0 => "EMPTY",
        1 => "MESH",
        2 => "CURVE",
        3 => "SURFACE",
        4 => "FONT",
        5 => "META",
        6 => "LAMP",
        7 => "LATTICE",
        8 => "ARMATURE",
        9 => "CAMERA",
        10 => "SPEAKER",
        11 => "WAVE",
        12 => "GPENCIL",
        _ => "UNKNOWN",
    }
}

/// 4×4 矩阵列优先展平（Python matrix4x4_to_list：m[j][i] for i,j，round 8）。
fn matrix4x4_to_list(m: &[[f32; 4]; 4]) -> Vec<f64> {
    (0..4)
        .flat_map(|col| (0..4).map(move |row| round_n(m[col][row] as f64, 8)))
        .collect()
}

/// 矩阵 × 向量（4×4 列优先 × 3 分量，w=1）。
fn mat_mul_vec(m: &[[f32; 4]; 4], v: [f32; 3]) -> [f64; 3] {
    let v4 = [v[0] as f64, v[1] as f64, v[2] as f64, 1.0];
    let mut out = [0.0f64; 3];
    for row in 0..3 {
        out[row] = m[0][row] as f64 * v4[0]
            + m[1][row] as f64 * v4[1]
            + m[2][row] as f64 * v4[2]
            + m[3][row] as f64 * v4[3];
    }
    out
}

/// 从 mesh 顶点计算世界坐标 bbox3d（8 角点 × [x,y,z]，round 6）。
fn bbox3d_from_mesh(reader: &Reader, obj: &Object) -> Option<Vec<Vec<f64>>> {
    let mesh = reader
        .deref_single(&obj.data.as_instance_of::<Mesh>())
        .ok()?;
    let verts = reader.deref(&mesh.mvert).ok()?;
    let mut min_v = [f32::MAX; 3];
    let mut max_v = [f32::MIN; 3];
    let mut any = false;
    for v in verts {
        for k in 0..3 {
            min_v[k] = min_v[k].min(v.co[k]);
            max_v[k] = max_v[k].max(v.co[k]);
        }
        any = true;
    }
    if !any {
        return None;
    }
    // 8 角点（局部）→ 世界（Blender bound_box 顺序）
    let corners = [
        [min_v[0], min_v[1], min_v[2]],
        [min_v[0], min_v[1], max_v[2]],
        [min_v[0], max_v[1], max_v[2]],
        [min_v[0], max_v[1], min_v[2]],
        [max_v[0], min_v[1], min_v[2]],
        [max_v[0], min_v[1], max_v[2]],
        [max_v[0], max_v[1], max_v[2]],
        [max_v[0], max_v[1], min_v[2]],
    ];
    Some(
        corners
            .iter()
            .map(|c| {
                let w = mat_mul_vec(&obj.obmat, *c);
                vec![round_n(w[0], 6), round_n(w[1], 6), round_n(w[2], 6)]
            })
            .collect(),
    )
}

/// 世界坐标顶点（下采样 max_verts=4000，round 6）。
fn export_world_verts(reader: &Reader, obj: &Object, max_verts: usize) -> Option<Vec<Vec<f64>>> {
    let mesh = reader
        .deref_single(&obj.data.as_instance_of::<Mesh>())
        .ok()?;
    let verts: Vec<&MVert> = reader.deref(&mesh.mvert).ok()?.collect();
    if verts.is_empty() {
        return None;
    }
    let n = verts.len();
    let step = n.div_ceil(max_verts).max(1);
    let mut pts = Vec::new();
    for k in (0..n).step_by(step) {
        let w = mat_mul_vec(&obj.obmat, verts[k].co);
        pts.push(vec![round_n(w[0], 6), round_n(w[1], 6), round_n(w[2], 6)]);
    }
    Some(pts)
}

/// 面数统计（sum(max(0, totloop - 2))）。
fn count_tris(reader: &Reader, obj: &Object) -> u64 {
    let Ok(mesh) = reader.deref_single(&obj.data.as_instance_of::<Mesh>()) else {
        return 0;
    };
    let Ok(polys) = reader.deref(&mesh.mpoly) else {
        return 0;
    };
    polys.map(|p| (p.totloop as i64 - 2).max(0) as u64).sum()
}

/// 解析 .blend 并输出 schema v3 envelope。
#[allow(clippy::approx_constant)] // Python 版用 3.14159265（非精确 PI），需保持输出一致
pub fn run(p: &BlendParams) -> Result<Value, String> {
    let data = std::fs::read(&p.file).map_err(|e| format!("read {}: {e}", p.file))?;
    let reader = read(&data).map_err(|e| format!("blend parse {}: {e}", p.file))?;

    // 场景名 + active camera
    let scene_name = reader
        .iter::<Scene>()
        .ok()
        .and_then(|mut s| s.next())
        .map(|sc| sc.id.get_name().to_string())
        .unwrap_or_else(|| "Scene".into());

    // 物体
    let mut objects3d = Vec::new();
    let mut cameras = Vec::new();
    let mut total_tris: u64 = 0;
    let mut visible_count: u64 = 0;

    let objs: Vec<&Object> = match reader.iter::<Object>() {
        Ok(it) => it.collect(),
        Err(_) => Vec::new(),
    };
    for (i, obj) in objs.iter().enumerate() {
        let name = obj.id.get_name().to_string();
        let otype = obj_type_str(obj.type_);
        let tris = count_tris(&reader, obj);
        total_tris += tris;
        // 可见性：OB_HIDE = flag bit 0
        let visible = (obj.flag & 0x0001) == 0;
        if visible {
            visible_count += 1;
        }
        let mat_world = matrix4x4_to_list(&obj.obmat);
        // bbox3d：MESH 从顶点计算；非 MESH（EMPTY 等）输出 8 个零点（Python bound_box 同款）
        let bbox = match bbox3d_from_mesh(&reader, obj) {
            Some(b) => Some(b),
            None if otype == "MESH" => None,
            None => Some(vec![vec![0.0, 0.0, 0.0]; 8]),
        };
        let (center, size) = match &bbox {
            Some(b) => {
                let mut c = [0.0f64; 3];
                let mut s = [0.0f64; 3];
                for k in 0..3 {
                    let vals: Vec<f64> = b.iter().map(|corner| corner[k]).collect();
                    let mn = vals.iter().cloned().fold(f64::MAX, f64::min);
                    let mx = vals.iter().cloned().fold(f64::MIN, f64::max);
                    c[k] = round_n((mn + mx) / 2.0, 6);
                    s[k] = round_n(mx - mn, 6);
                }
                (Some(vec![c[0], c[1], c[2]]), Some(vec![s[0], s[1], s[2]]))
            }
            None => (None, None),
        };
        let verts = if otype == "MESH" {
            export_world_verts(&reader, obj, 4000)
        } else {
            None
        };
        // 注意：blend-rs 0.3.0 的 Pointer::is_valid() 用 u8 求和会溢出（8 字节指针），
        // 改用 address()（usize 求和，null 返回 None）。
        let parent = if obj.parent.address().is_some() {
            reader
                .deref_single(&obj.parent.as_instance_of::<Object>())
                .ok()
                .map(|p| p.id.get_name().to_string())
        } else {
            None
        };

        objects3d.push(json!({
            "id": i,
            "name": name,
            "type": otype,
            "collection": "Collection",
            "parent": parent,
            "matrix_world": mat_world,
            "bbox3d": bbox,
            "center": center,
            "size": size,
            "tris": tris,
            "visible": visible,
            "material": Value::Null,
            "verts": verts,
        }));

        // 摄像机
        if obj.type_ == 9 {
            if let Ok(cam) = reader.deref_single(&obj.data.as_instance_of::<Camera>()) {
                let sensor_x = cam.sensor_x as f64;
                let lens = cam.lens as f64;
                let fov = if lens > 0.0 {
                    2.0 * (sensor_x / (2.0 * lens)).atan()
                } else {
                    0.0
                };
                cameras.push(json!({
                    "name": name,
                    "location": obj.loc.iter().map(|v| round_n(*v as f64, 6)).collect::<Vec<_>>(),
                    "rotation_euler": obj.rot.iter().map(|v| round_n(*v as f64, 6)).collect::<Vec<_>>(),
                    "rotation_quaternion": obj.quat.iter().map(|v| round_n(*v as f64, 6)).collect::<Vec<_>>(),
                    "fov_rad": round_n(fov, 6),
                    "fov_deg": round_n(fov * 180.0 / 3.14159265, 4),
                    "sensor_width": round_n(sensor_x, 4),
                    "lens": round_n(lens, 4),
                    "matrix_world": matrix4x4_to_list(&obj.obmat),
                    "is_active": false,
                }));
            }
        }
    }

    Ok(json!({
        "schema": "vision-report/v3",
        "task": "blender_dump",
        "sensors": ["bpy"],
        "coordsys": "world_m",
        "source": {
            "type": "blender",
            "filepath": p.file,
            "blender_version": "3.2 (blend-rs)",
            "scene_name": scene_name,
        },
        "objects3d": objects3d,
        "cameras": cameras,
        "metrics": {
            "object_count": objects3d.len(),
            "collection_count": 1,
            "total_tris": total_tris,
            "visible_count": visible_count,
        },
        "elements": [],
        "anomalies": [],
        "notation": "primitive notation: reference locations as [bbox3d: x1,y1,z1-x8,y8,z8] or [matrix_world: 16 values row-major]. Coordinates are world-space meters.",
    }))
}

/// Python round(x, n) 语义（十进制展开舍入）。
fn round_n(v: f64, n: i32) -> f64 {
    let factor = 10f64.powi(n);
    (v * factor).round() / factor
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn obj_type_mapping() {
        assert_eq!(obj_type_str(1), "MESH");
        assert_eq!(obj_type_str(2), "CURVE");
        assert_eq!(obj_type_str(9), "CAMERA");
        assert_eq!(obj_type_str(0), "EMPTY");
    }

    #[test]
    fn matrix_flatten_order() {
        // 单位矩阵列优先展平 = 对角 1
        let m = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let flat = matrix4x4_to_list(&m);
        assert_eq!(flat.len(), 16);
        assert_eq!(flat[0], 1.0);
        assert_eq!(flat[5], 1.0);
        assert_eq!(flat[10], 1.0);
        assert_eq!(flat[15], 1.0);
        assert_eq!(flat[1], 0.0);
    }

    #[test]
    fn mat_mul_translation() {
        // 平移矩阵：x+10
        let m = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [10.0, 0.0, 0.0, 1.0],
        ];
        let v = mat_mul_vec(&m, [1.0, 2.0, 3.0]);
        assert_eq!(v[0], 11.0);
        assert_eq!(v[1], 2.0);
        assert_eq!(v[2], 3.0);
    }

    #[test]
    fn round_n_semantics() {
        assert_eq!(round_n(1.23456789, 6), 1.234568);
        assert_eq!(round_n(1.23456789, 8), 1.23456789);
        assert_eq!(round_n(2.675, 2), 2.68);
    }
}
