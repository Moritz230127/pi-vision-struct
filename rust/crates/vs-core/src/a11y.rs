//! 无障碍传感器（对应 Python `vs_a11y.py`：AT-SPI 树 → schema v3 元素）。
//!
//! atspi crate（纯 Rust zbus 实现）读取桌面无障碍树：角色/名称/屏幕坐标。
//! 坐标系 screen_px（全屏绝对坐标）。只读保证：仅调用 get_* 只读接口。

use serde_json::{json, Value};

use crate::schema::envelope;

/// 无障碍参数（对应 Python `--app/--max-elements`）。
pub struct A11yParams {
    /// 只导出指定应用名（None = 全部）。
    pub app: Option<String>,
    /// 最大元素数。
    pub max_items: u32,
}

const MAX_DEPTH: u32 = 14;

/// 通过 zbus 直接调用 AT-SPI `GetExtents`（int32 签名）。
///
/// atspi crate 的 `CoordType` 用 `#[repr(u32)]` 序列化为 `u`，但 AT-SPI
/// 期望 `i`（int32）——故绕过 `ComponentProxy` 手动指定签名。
/// 返回 (x, y, w, h)（屏幕坐标）。
///
/// Firefox 等应用的 Component 接口注册不稳定（首次调用可能 UnknownMethod），
/// 故重试 3 次（间隔 100ms）。
pub(crate) async fn get_extents_zbus(
    obj_ref: &atspi::proxy::common::object_ref::ObjectRefOwned,
    conn: &zbus::Connection,
) -> Result<(i32, i32, i32, i32), String> {
    let name = obj_ref.name().ok_or("no bus name")?;
    let path = obj_ref.path();
    let proxy = zbus::Proxy::new(conn, name.clone(), path.clone(), "org.a11y.atspi.Component")
        .await
        .map_err(|e| format!("zbus proxy: {e}"))?;
    let mut last_err = String::new();
    for _ in 0..3 {
        match proxy.call_method("GetExtents", &0u32).await {
            Ok(reply) => {
                return reply
                    .body()
                    .deserialize::<(i32, i32, i32, i32)>()
                    .map_err(|e| format!("deserialize: {e}"));
            }
            Err(e) => {
                last_err = format!("GetExtents: {e}");
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }
        }
    }
    Err(last_err)
}

/// 通过 zbus 直接调用 AT-SPI `DoAction`（org.a11y.atspi.Action 接口）。
///
/// 五级降级链第 1 级（a11y_action）：对无障碍元素执行动作（如 "click"），
/// 零坐标误差（Codex accessibility 动作直达同款）。
/// 返回 Ok(true) 表示动作执行成功。
///
/// `action_index`：动作索引（0 = 默认动作，通常为 "click"）。
pub async fn do_action_zbus(
    obj_ref: &atspi::proxy::common::object_ref::ObjectRefOwned,
    conn: &zbus::Connection,
    action_index: i32,
) -> Result<bool, String> {
    let name = obj_ref.name().ok_or("no bus name")?;
    let path = obj_ref.path();
    let proxy = zbus::Proxy::new(conn, name.clone(), path.clone(), "org.a11y.atspi.Action")
        .await
        .map_err(|e| format!("zbus Action proxy: {e}"))?;
    let mut last_err = String::new();
    for _ in 0..3 {
        match proxy.call_method("DoAction", &action_index).await {
            Ok(reply) => {
                return reply
                    .body()
                    .deserialize::<bool>()
                    .map_err(|e| format!("deserialize DoAction: {e}"));
            }
            Err(e) => {
                last_err = format!("DoAction: {e}");
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }
        }
    }
    Err(last_err)
}

/// 查询元素的可用动作列表（org.a11y.atspi.Action 接口 `GetActions`）。
///
/// 返回动作名列表（如 ["click", "press"]）。空列表 = 元素无动作（降级到坐标点击）。
pub async fn get_actions_zbus(
    obj_ref: &atspi::proxy::common::object_ref::ObjectRefOwned,
    conn: &zbus::Connection,
) -> Result<Vec<String>, String> {
    use atspi::proxy::action::ActionProxy;
    let name: zbus::names::BusName = obj_ref.name().ok_or("no bus name")?.clone().into();
    let path = obj_ref.path();
    let proxy = ActionProxy::builder(conn)
        .destination(name)
        .map_err(|e| format!("ActionProxy destination: {e}"))?
        .path(path)
        .map_err(|e| format!("ActionProxy path: {e}"))?
        .cache_properties(zbus::proxy::CacheProperties::No)
        .build()
        .await
        .map_err(|e| format!("ActionProxy build: {e}"))?;
    let actions = proxy
        .get_actions()
        .await
        .map_err(|e| format!("GetActions: {e}"))?;
    Ok(actions.into_iter().map(|a| a.name).collect())
}

/// 读取 AT-SPI 无障碍树 → schema v3 envelope。
pub fn run(p: &A11yParams) -> Result<Value, String> {
    use atspi::connection::AccessibilityConnection;
    use atspi::proxy::accessible::{AccessibleProxy, ObjectRefExt};

    // 异步运行时（atspi 是 zbus 异步 API）
    let rt = tokio::runtime::Runtime::new().map_err(|e| format!("tokio rt: {e}"))?;
    rt.block_on(async {
        let conn = AccessibilityConnection::new()
            .await
            .map_err(|e| format!("atspi connect: {e}"))?;
        let root = conn
            .root_accessible_on_registry()
            .await
            .map_err(|e| format!("atspi root: {e}"))?;

        let mut elements: Vec<Value> = Vec::new();
        let mut truncated = false;
        let mut next_id: u32 = 0;

        // 递归遍历（与 Python walk 同构：先自身后子节点，深度优先）
        // 递归 async fn 需 Box::pin（Rust 限制）
        fn walk<'a>(
            acc: &'a AccessibleProxy<'_>,
            conn: &'a AccessibilityConnection,
            depth: u32,
            p: &'a A11yParams,
            elements: &'a mut Vec<Value>,
            truncated: &'a mut bool,
            next_id: &'a mut u32,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), String>> + Send + 'a>> {
            Box::pin(async move {
                if *truncated || depth > MAX_DEPTH || *next_id >= p.max_items {
                    *truncated = *truncated || *next_id >= p.max_items;
                    return Ok(());
                }
                // 角色/名称/坐标（只读接口）
                let role_name = acc.get_role_name().await.unwrap_or_default();
                let name = acc.name().await.unwrap_or_default();
                // AccessibleProxy → ObjectRefOwned → zbus 直接调用 GetExtents
                // （atspi crate 的 CoordType 用 repr(u32) 但 AT-SPI 期望 int32，
                //  故绕过 ComponentProxy 手动指定签名）
                let obj_ref: atspi::proxy::common::object_ref::ObjectRefOwned =
                    match acc.try_into() {
                        Ok(o) => o,
                        Err(_) => return Ok(()),
                    };
                // 应用节点（role=application）无 Component 接口 → 不输出元素，
                // 但继续遍历子节点（窗口/按钮/输入框才有 extents）
                let is_app = role_name == "application";
                if !is_app {
                    let ext = match get_extents_zbus(&obj_ref, conn.connection()).await {
                        Ok(e) => e,
                        // 无 Component 接口（部分节点）→ 跳过该节点但继续遍历子节点
                        Err(_) => {
                            let n = acc.child_count().await.unwrap_or(0);
                            for i in 0..n {
                                let child_ref = match acc.get_child_at_index(i).await {
                                    Ok(c) => c,
                                    Err(_) => continue,
                                };
                                let child = match child_ref
                                    .into_accessible_proxy(conn.connection())
                                    .await
                                {
                                    Ok(c) => c,
                                    Err(_) => continue,
                                };
                                walk(
                                    &child, conn, depth + 1, p, elements, truncated, next_id,
                                )
                                .await?;
                            }
                            return Ok(());
                        }
                    };
                    let (x, y, w, h) = ext;
                    let (x2, y2) = (x + w.max(0), y + h.max(0));
                    let visible = w > 0 && h > 0;
                    let el = json!({
                        "id": *next_id,
                        "type": role_name,
                        "bbox": [x, y, x2, y2],
                        "text": if name.is_empty() { Value::Null } else { Value::String(name.chars().take(120).collect()) },
                        "coordsys": "screen_px",
                        // F3：bus name + path，供 click 路径直接定位（不再重新遍历全局树）
                        "bus": obj_ref.name().map(|n| n.to_string()),
                        "path": obj_ref.path().to_string(),
                    });
                    elements.push(el);
                    *next_id += 1;
                    if !visible {
                        return Ok(());
                    }
                }
                let n = acc.child_count().await.unwrap_or(0);
                for i in 0..n {
                    let child_ref = match acc.get_child_at_index(i).await {
                        Ok(c) => c,
                        Err(_) => continue,
                    };
                    let child = match child_ref.into_accessible_proxy(conn.connection()).await {
                        Ok(c) => c,
                        Err(_) => continue,
                    };
                    walk(&child, conn, depth + 1, p, elements, truncated, next_id).await?;
                }
                Ok(())
            })
        }

        // 遍历应用（root 的子节点 = 应用）
        let app_count = root.child_count().await.unwrap_or(0);
        for i in 0..app_count {
            let app_ref = match root.get_child_at_index(i).await {
                Ok(a) => a,
                Err(_) => continue,
            };
            let app = match app_ref.into_accessible_proxy(conn.connection()).await {
                Ok(a) => a,
                Err(_) => continue,
            };
            let app_name = app.name().await.unwrap_or_default();
            if let Some(filter) = &p.app {
                if !app_name.to_lowercase().contains(&filter.to_lowercase()) {
                    continue;
                }
            }
            walk(&app, &conn, 0, p, &mut elements, &mut truncated, &mut next_id).await?;
        }

        let mut report = envelope(
            Some("a11y"),
            &["atspi"],
            Some("screen_px"),
            Some(json!({"type": "a11y", "app": p.app.clone().unwrap_or_else(|| "*".into())})),
        );
        report["elements"] = Value::Array(elements);
        report["truncated"] = Value::Bool(truncated);
        report["metrics"] = json!({"element_count": next_id});
        Ok(report)
    })
}

// ---- F3：bus/path 直接定位（click 路径用，替代全局树遍历）----

/// 通过 bus name + path 直接调用 GetExtents（F3：click 路径直接定位）。
pub async fn get_extents_zbus_path(
    bus: &str,
    path: &str,
    conn: &zbus::Connection,
) -> Result<(i32, i32, i32, i32), String> {
    let proxy = zbus::Proxy::new(conn, bus, path, "org.a11y.atspi.Component")
        .await
        .map_err(|e| format!("zbus proxy: {e}"))?;
    let mut last_err = String::new();
    for _ in 0..3 {
        match proxy.call_method("GetExtents", &0u32).await {
            Ok(reply) => {
                return reply
                    .body()
                    .deserialize::<(i32, i32, i32, i32)>()
                    .map_err(|e| format!("deserialize: {e}"));
            }
            Err(e) => {
                last_err = format!("GetExtents: {e}");
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }
        }
    }
    Err(last_err)
}

/// 通过 bus name + path 直接调用 GetActions（F3）。
pub async fn get_actions_zbus_path(
    bus: &str,
    path: &str,
    conn: &zbus::Connection,
) -> Result<Vec<String>, String> {
    use atspi::proxy::action::ActionProxy;
    let name: zbus::names::BusName = bus
        .try_into()
        .map_err(|e: zbus::names::Error| format!("bus name 解析: {e}"))?;
    let proxy = ActionProxy::builder(conn)
        .destination(name)
        .map_err(|e| format!("ActionProxy destination: {e}"))?
        .path(path)
        .map_err(|e| format!("ActionProxy path: {e}"))?
        .cache_properties(zbus::proxy::CacheProperties::No)
        .build()
        .await
        .map_err(|e| format!("ActionProxy build: {e}"))?;
    let actions = proxy
        .get_actions()
        .await
        .map_err(|e| format!("GetActions: {e}"))?;
    Ok(actions.into_iter().map(|a| a.name).collect())
}

/// 通过 bus name + path 直接调用 DoAction（F3）。
pub async fn do_action_zbus_path(
    bus: &str,
    path: &str,
    conn: &zbus::Connection,
    action_index: i32,
) -> Result<bool, String> {
    let proxy = zbus::Proxy::new(conn, bus, path, "org.a11y.atspi.Action")
        .await
        .map_err(|e| format!("zbus Action proxy: {e}"))?;
    let mut last_err = String::new();
    for _ in 0..3 {
        match proxy.call_method("DoAction", &action_index).await {
            Ok(reply) => {
                return reply
                    .body()
                    .deserialize::<bool>()
                    .map_err(|e| format!("deserialize DoAction: {e}"));
            }
            Err(e) => {
                last_err = format!("DoAction: {e}");
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }
        }
    }
    Err(last_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a11y_params_default() {
        let p = A11yParams {
            app: None,
            max_items: 80,
        };
        assert!(p.app.is_none());
        assert_eq!(p.max_items, 80);
    }
}
