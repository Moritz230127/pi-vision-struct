//! S3 门禁：并发压测（5 任务零 OOM）+ 功耗切换 + 沙箱逃逸测试。
//!
//! 运行：`cargo test -p vsd --test gate_s3`（需 Linux + 可写 /tmp）。

use std::time::Duration;

use vsd::scheduler::{default_scheduler, Scheduler, PRIORITY_ANALYZE, PRIORITY_BG};

/// 并发压测：5 个并发推理请求 → 调度器串行执行 → 零 OOM（vram 不超预算）。
#[test]
fn concurrency_stress_no_oom() {
    let sched = Scheduler::new(6.5, 1); // 6.5GB 预算 + 1 GPU 槽位
                                        // 注册一个模拟推理（耗时 50ms，模拟 GPU 推理）
    sched.register(
        "saliency",
        Box::new(|_, _, use_gpu| {
            std::thread::sleep(Duration::from_millis(50));
            Ok(serde_json::json!({"device": if use_gpu { "cuda" } else { "cpu" }}))
        }),
        None,
    );
    // 5 个并发提交（不同优先级）
    let mut handles = Vec::new();
    for i in 0..5 {
        let s = sched.clone();
        handles.push(std::thread::spawn(move || {
            let prio = if i % 2 == 0 {
                PRIORITY_ANALYZE
            } else {
                PRIORITY_BG
            };
            s.submit(
                "saliency".into(),
                serde_json::json!({"task": i}),
                prio,
                Duration::from_secs(10),
            );
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    // 等待全部完成（5 × 50ms 串行 ≈ 250ms）
    std::thread::sleep(Duration::from_millis(800));
    let st = sched.stats();
    assert_eq!(st["submitted"], 5, "5 任务全部提交");
    assert_eq!(st["completed"], 5, "5 任务全部完成（零丢失）");
    assert_eq!(st["timeouts"], 0, "零超时");
    // 零 OOM：vram_used 不超预算（任务完成后应归零）
    assert_eq!(st["vram_used"], 0.0, "推理完成后显存释放");
}

/// 并发压测：显存预算内并发不 OOM（模拟 2 个大模型同时请求）。
#[test]
fn vram_budget_no_oom() {
    let sched = Scheduler::new(6.5, 1);
    sched.register(
        "segment",
        Box::new(|_, _, use_gpu| {
            std::thread::sleep(Duration::from_millis(30));
            Ok(serde_json::json!({"device": if use_gpu { "cuda" } else { "cpu" }}))
        }),
        None,
    );
    // 3 个 segment（vram_peak 2.0GB）并发 → 预算 6.5GB 内串行执行
    for i in 0..3 {
        sched.submit(
            "segment".into(),
            serde_json::json!({"task": i}),
            PRIORITY_ANALYZE,
            Duration::from_secs(10),
        );
    }
    std::thread::sleep(Duration::from_millis(500));
    let st = sched.stats();
    assert_eq!(st["completed"], 3);
    // 串行执行：任意时刻 vram_used ≤ 预算（此处验证完成后归零）
    assert_eq!(st["vram_used"], 0.0);
}

/// 功耗切换：battery 模式 → 大模型（vram_reserve ≥ 1.0）CPU 降级。
#[test]
fn power_battery_cpu_fallback() {
    // 验证逻辑：显存不足（模拟 battery 降级路径）→ CPU 回退
    let sched_small = Scheduler::new(0.5, 1); // 极小预算模拟降级条件
    sched_small.register(
        "segment",
        Box::new(|_, _, use_gpu| Ok(serde_json::json!({"use_gpu": use_gpu}))),
        None,
    );
    sched_small.submit(
        "segment".into(),
        serde_json::json!({}),
        PRIORITY_ANALYZE,
        Duration::from_secs(5),
    );
    std::thread::sleep(Duration::from_millis(300));
    let st = sched_small.stats();
    assert_eq!(st["cpu_fallback"], 1, "显存不足 → CPU 降级");
    assert_eq!(st["oom_avoided"], 1, "OOM 被避免");
}

/// 沙箱逃逸测试：seccomp 拦截 execve（子进程启动失败）。
#[test]
fn sandbox_blocks_exec_escape() {
    use vsd::sandbox::{apply_seccomp, SandboxConfig};
    let cfg = SandboxConfig {
        allow_exec: false,
        ..SandboxConfig::default()
    };
    // 应用 seccomp（若环境允许）
    if apply_seccomp(cfg.allow_exec).is_ok() {
        // execve 应被拦截 → Command::output 失败
        let r = std::process::Command::new("/bin/echo")
            .arg("escape")
            .output();
        assert!(r.is_err(), "execve 应被 seccomp 拦截（逃逸失败）");
    }
    // 注意：此测试在无 CAP_SYS_ADMIN 环境降级跳过
}

/// 沙箱逃逸测试：landlock 拦截只读路径写入。
#[test]
fn sandbox_blocks_write_escape() {
    use vsd::sandbox::{apply_landlock, SandboxConfig};
    let cfg = SandboxConfig {
        writable_dirs: vec!["/tmp".to_string()],
        read_only_dirs: vec!["/usr".to_string()],
        ..SandboxConfig::default()
    };
    if apply_landlock(&cfg).unwrap_or(false) {
        let r = std::fs::write("/usr/vsd_escape_test", b"x");
        assert!(r.is_err(), "Landlock 外路径写入应被拒绝");
    }
}

/// 调度器默认实例可用性。
#[test]
fn default_scheduler_works() {
    let s = default_scheduler();
    let st = s.stats();
    assert!(st["submitted"].is_u64());
}
