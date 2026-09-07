fn main() {
    match vs_core::capture::run(&vs_core::capture::CaptureParams {
        region: None,
        out: "/tmp/vsd_capture_smoke.png".into(),
    }) {
        Ok(v) => {
            let s = serde_json::to_string_pretty(&v).unwrap();
            println!("capture OK: {}", &s[..s.len().min(500)]);
        }
        Err(e) => println!("capture FAILED: {e}"),
    }
}
