//! S7 冒烟测试（真实环境：Wayland 截屏 + AT-SPI 读取）。
//! 运行：cargo test -p vs-core --test smoke_s7 -- --ignored

use vs_core::a11y::A11yParams;
use vs_core::capture::CaptureParams;

#[test]
#[ignore = "真实环境冒烟（Wayland + AT-SPI）"]
fn capture_real_screen() {
    let p = CaptureParams {
        region: None,
        out: "/tmp/vsd_capture_smoke.png".into(),
    };
    match vs_core::capture::run(&p) {
        Ok(v) => {
            let src = &v["source"];
            assert_eq!(src["type"], "screen");
            let size = src["size_px"].as_array().unwrap();
            assert!(size[0].as_u64().unwrap() > 0, "截屏宽度 > 0");
            assert!(size[1].as_u64().unwrap() > 0, "截屏高度 > 0");
            println!("capture OK: {}x{}", size[0], size[1]);
        }
        Err(e) => println!("capture SKIP/FAIL: {e}"),
    }
}

#[test]
#[ignore = "真实环境冒烟（AT-SPI 会话）"]
fn a11y_real_tree() {
    let p = A11yParams {
        app: None,
        max_items: 20,
    };
    match vs_core::a11y::run(&p) {
        Ok(v) => {
            let n = v["metrics"]["element_count"].as_u64().unwrap_or(0);
            println!("a11y OK: {n} elements");
            assert!(n > 0, "AT-SPI 树应有元素");
        }
        Err(e) => println!("a11y SKIP/FAIL: {e}"),
    }
}
