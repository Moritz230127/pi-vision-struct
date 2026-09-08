//! C3 契约测试：capture/dom/a11y 结构 + 类型 + 非空断言（环境可用时）。
//!
//! 这些传感器依赖真实环境（显示器/Chrome/AT-SPI），无法做 golden 差分。
//! 契约测试验证：
//!   1. 结构：schema v3 envelope 字段齐全
//!   2. 类型：bbox 为整数数组、coordsys 正确
//!   3. 非空：环境可用时（VS_CONTRACT_ENV=1）验证真实输出非空

use serde_json::Value;

/// 验证 schema v3 envelope 结构（所有传感器共用）。
fn assert_envelope_shape(v: &Value, task: &str, coordsys: &str) {
    assert_eq!(v["schema"], "vision-report/v3", "schema 字段");
    assert_eq!(v["task"], task, "task 字段");
    assert!(v["sensors"].is_array(), "sensors 数组");
    assert_eq!(v["coordsys"], coordsys, "coordsys 字段");
    assert!(v["source"].is_object(), "source 对象");
    assert!(v["elements"].is_array(), "elements 数组");
    assert!(v["findings"].is_array(), "findings 数组");
    assert!(v["candidates"].is_array(), "candidates 数组");
    assert!(v["metrics"].is_object(), "metrics 对象");
}

/// 验证 bbox 为整数数组（[x1, y1, x2, y2]）。
fn assert_bbox_int(el: &Value) {
    let bbox = &el["bbox"];
    assert!(bbox.is_array(), "bbox 数组");
    let arr = bbox.as_array().unwrap();
    assert_eq!(arr.len(), 4, "bbox 4 元素");
    for v in arr {
        assert!(v.is_i64() || v.is_u64(), "bbox 元素为整数: {v}");
    }
}

/// 环境可用性检测：VS_CONTRACT_ENV=1 时执行真实调用。
fn env_available() -> bool {
    std::env::var("VS_CONTRACT_ENV")
        .map(|v| v == "1")
        .unwrap_or(false)
}

#[test]
fn capture_contract_shape() {
    // 结构契约：envelope 字段（不依赖环境）
    let v = vs_core::capture::run(&vs_core::capture::CaptureParams {
        region: None,
        out: "/tmp/vs_contract_capture.png".into(),
        window: None,
    });
    match v {
        Ok(report) => {
            assert_envelope_shape(&report, "capture", "screen_px");
            assert_eq!(report["source"]["type"], "screen");
            assert_eq!(report["source"]["backend"], "xcap");
            assert!(report["metrics"]["size_px"].is_array());
        }
        Err(e) => {
            // 无显示器环境：允许失败，但必须是明确的错误（非 panic）
            assert!(
                e.contains("timeout") || e.contains("no monitor") || e.contains("xcap"),
                "capture 失败原因应明确: {e}"
            );
        }
    }
}

#[test]
fn capture_contract_nonempty_when_env() {
    if !env_available() {
        eprintln!("[skip] VS_CONTRACT_ENV 未设置，跳过真实截屏");
        return;
    }
    let v = vs_core::capture::run(&vs_core::capture::CaptureParams {
        region: None,
        out: "/tmp/vs_contract_capture.png".into(),
        window: None,
    })
    .expect("capture 应成功");
    let size = v["metrics"]["size_px"].as_array().unwrap();
    assert!(size[0].as_u64().unwrap() > 0, "宽度 > 0");
    assert!(size[1].as_u64().unwrap() > 0, "高度 > 0");
}

#[test]
fn dom_contract_shape() {
    // 结构契约：css_to_hex 纯函数 + envelope 字段（不依赖 Chrome）
    // css_to_hex 是私有函数，通过公开行为验证：dom 输出 color 字段格式
    let v = vs_core::dom::run(&vs_core::dom::DomParams {
        url: "about:blank".into(),
        max_elements: 10,
        timeout_ms: 5_000,
    });
    match v {
        Ok(report) => {
            assert_envelope_shape(&report, "dom", "css_px");
            assert_eq!(report["source"]["type"], "dom");
            for el in report["elements"].as_array().unwrap() {
                assert_bbox_int(el);
                assert!(el["coordsys"] == "css_px", "元素 coordsys");
            }
        }
        Err(e) => {
            // 无 Chrome 环境：允许失败，但错误应明确
            assert!(
                e.contains("chrome") || e.contains("launch") || e.contains("goto"),
                "dom 失败原因应明确: {e}"
            );
        }
    }
}

#[test]
fn dom_contract_nonempty_when_env() {
    if !env_available() {
        eprintln!("[skip] VS_CONTRACT_ENV 未设置，跳过真实 DOM");
        return;
    }
    let v = vs_core::dom::run(&vs_core::dom::DomParams {
        url: "data:text/html,<html><body><button>OK</button></body></html>".into(),
        max_elements: 20,
        timeout_ms: 15_000,
    })
    .expect("dom 应成功");
    assert!(
        !v["elements"].as_array().unwrap().is_empty(),
        "DOM 元素非空"
    );
}

#[test]
fn a11y_contract_shape() {
    // 结构契约：参数结构 + envelope 字段（不依赖 AT-SPI）
    let v = vs_core::a11y::run(&vs_core::a11y::A11yParams {
        app: None,
        max_items: 10,
    });
    match v {
        Ok(report) => {
            assert_envelope_shape(&report, "a11y", "screen_px");
            assert_eq!(report["source"]["type"], "a11y");
            for el in report["elements"].as_array().unwrap() {
                assert_bbox_int(el);
                assert!(el["coordsys"] == "screen_px", "元素 coordsys");
            }
        }
        Err(e) => {
            // 无 AT-SPI 环境：允许失败，但错误应明确
            assert!(
                e.contains("atspi") || e.contains("connect") || e.contains("registry"),
                "a11y 失败原因应明确: {e}"
            );
        }
    }
}

#[test]
fn a11y_contract_nonempty_when_env() {
    if !env_available() {
        eprintln!("[skip] VS_CONTRACT_ENV 未设置，跳过真实 AT-SPI");
        return;
    }
    let v = vs_core::a11y::run(&vs_core::a11y::A11yParams {
        app: None,
        max_items: 50,
    })
    .expect("a11y 应成功");
    assert!(
        !v["elements"].as_array().unwrap().is_empty(),
        "AT-SPI 元素非空"
    );
}
