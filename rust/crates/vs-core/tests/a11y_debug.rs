//! a11y 调试探针（临时，验证后删除）。
use atspi::connection::AccessibilityConnection;
use atspi::proxy::accessible::ObjectRefExt;

#[test]
fn a11y_debug_probe() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let conn = AccessibilityConnection::new().await.unwrap();
        println!("unique_name: {:?}", conn.connection().unique_name());
        let root = conn.root_accessible_on_registry().await.unwrap();
        let c = root.get_child_at_index(2).await.unwrap();
        let ap = c.into_accessible_proxy(conn.connection()).await.unwrap();
        let obj_ref: atspi::proxy::common::object_ref::ObjectRefOwned = ap.try_into().unwrap();
        let bus_name = obj_ref.name().unwrap().clone();
        let path = obj_ref.path().clone();
        // 手动构造消息（无 sender）
        let msg = zbus::Message::method_call(&path, "GetExtents")
            .unwrap()
            .interface("org.a11y.atspi.Component")
            .unwrap()
            .destination(bus_name)
            .unwrap()
            .build(&0u32)
            .unwrap();
        println!("msg header: {:?}", msg.header());
        conn.connection().send(&msg).await.unwrap();
        println!("sent (no sender field)");
    });
}
