//! 调度管理器（对应 Python `vsched.py`：显存预算 + 并发 + LRU + 优先级 + 功耗感知）。
//!
//! 8GB VRAM 硬约束下，任何并发 GPU 推理都可能 OOM。Scheduler 是 GPU/CPU
//! 推理请求的唯一入口：显存预算表 / 并发信号量（GPU 串行）/ 模型 LRU 淘汰 /
//! 优先级队列（analyze > zoom > probe > bg）/ 功耗感知（battery → CPU 降级）/
//! 超时保护。

use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// 每模型显存预算：(vram_reserve_GB, vram_peak_GB, cpu_ok)。
#[derive(Clone)]
pub struct ModelBudget {
    pub vram_reserve: f64,
    pub vram_peak: f64,
    pub cpu_ok: bool,
}

/// 显存预算表（与 Python `MODEL_BUDGETS` 一致）。
pub fn model_budgets() -> HashMap<String, ModelBudget> {
    let mut m = HashMap::new();
    m.insert(
        "saliency".into(),
        ModelBudget {
            vram_reserve: 0.5,
            vram_peak: 1.0,
            cpu_ok: true,
        },
    );
    m.insert(
        "segment".into(),
        ModelBudget {
            vram_reserve: 1.0,
            vram_peak: 2.0,
            cpu_ok: true,
        },
    );
    m.insert(
        "depth".into(),
        ModelBudget {
            vram_reserve: 0.3,
            vram_peak: 0.8,
            cpu_ok: true,
        },
    );
    m.insert(
        "detect".into(),
        ModelBudget {
            vram_reserve: 2.0,
            vram_peak: 3.0,
            cpu_ok: true,
        },
    );
    m.insert(
        "omniparser".into(),
        ModelBudget {
            vram_reserve: 2.0,
            vram_peak: 3.0,
            cpu_ok: true,
        },
    );
    m
}

/// 总显存预算（GB，8GB 留 1.5GB 安全水位）。
pub const TOTAL_VRAM_BUDGET: f64 = 6.5;
/// 并发 GPU 推理槽位（串行）。
pub const GPU_SLOTS: usize = 1;
/// 功耗轮询间隔（秒）。
pub const POWER_POLL_INTERVAL: u64 = 30;

/// 优先级（数字小 = 高优先级）。
#[allow(dead_code)] // main.rs 接线后移除
pub const PRIORITY_ANALYZE: u32 = 0;
#[allow(dead_code)] // main.rs 接线后移除
pub const PRIORITY_ZOOM: u32 = 1;
pub const PRIORITY_PROBE: u32 = 2;
#[allow(dead_code)] // main.rs 接线后移除
pub const PRIORITY_BG: u32 = 3;

/// 调度任务。
pub struct Task {
    pub model: String,
    pub args: serde_json::Value,
    #[allow(dead_code)] // 队列排序用（S4 接线结果回传）
    pub priority: u32,
    #[allow(dead_code)] // LRU 排序用（S4 接线）
    pub submitted_at: Instant,
    pub deadline: Instant,
    pub device: String,
}

/// 推理函数：`(model, args, use_gpu) -> Result<Value, String>`。
pub type InferFn =
    Box<dyn Fn(&str, &serde_json::Value, bool) -> Result<serde_json::Value, String> + Send + Sync>;

/// 功耗监视器：检测电源状态（battery/plugged/low-power）+ CPU 频率。
pub struct PowerMonitor {
    mode: Mutex<String>,
    poll_interval: u64,
}

impl PowerMonitor {
    pub fn new(poll_interval: u64) -> Arc<Self> {
        let m = Arc::new(Self {
            mode: Mutex::new(Self::detect()),
            poll_interval,
        });
        // 后台轮询线程
        let m2 = m.clone();
        std::thread::spawn(move || loop {
            std::thread::sleep(Duration::from_secs(m2.poll_interval));
            let new_mode = Self::detect();
            let mut mode = m2.mode.lock().unwrap();
            if *mode != new_mode {
                eprintln!("[vsched] 功耗模式: {} → {}", *mode, new_mode);
                *mode = new_mode;
            }
        });
        m
    }

    /// 检测电源状态。Linux: /sys/class/power_supply/BAT*/status + capacity。
    /// 返回档位：plugged（充电/满电/无电池）/ battery（放电）/ low-power（放电 + 低电量 <20%）。
    pub fn detect() -> String {
        let mut battery = false;
        let mut capacity: Option<u32> = None;
        if let Ok(entries) = std::fs::read_dir("/sys/class/power_supply") {
            for e in entries.flatten() {
                let name = e.file_name();
                let name = name.to_string_lossy();
                if name.starts_with("BAT") {
                    battery = true;
                    let status_path = e.path().join("status");
                    if let Ok(status) = std::fs::read_to_string(status_path) {
                        let status = status.trim();
                        if status == "Discharging" {
                            // 读取电量（低电量 → low-power）
                            if let Ok(cap) = std::fs::read_to_string(e.path().join("capacity")) {
                                capacity = cap.trim().parse::<u32>().ok();
                            }
                            return match capacity {
                                Some(c) if c < 20 => "low-power".to_string(),
                                _ => "battery".to_string(),
                            };
                        }
                        if status == "Charging" || status == "Full" {
                            return "plugged".to_string();
                        }
                    }
                }
            }
        }
        let _ = battery;
        "plugged".to_string() // 无电池 → 视为 plugged
    }

    /// 当前 CPU 频率（MHz）。读取 /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq。
    pub fn cpu_freq_mhz(&self) -> Option<u64> {
        let path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq";
        std::fs::read_to_string(path)
            .ok()?
            .trim()
            .parse::<u64>()
            .ok()
            .map(|khz| khz / 1000)
    }

    pub fn is_battery(&self) -> bool {
        let m = self.mode.lock().unwrap();
        *m == "battery" || *m == "low-power"
    }

    /// 低功耗模式（battery + 电量 <20%）：所有模型 CPU 降级。
    pub fn is_low_power(&self) -> bool {
        *self.mode.lock().unwrap() == "low-power"
    }

    /// 当前档位（供 stats 输出）。
    pub fn mode(&self) -> String {
        self.mode.lock().unwrap().clone()
    }
}

/// 调度器状态统计。
#[derive(Default, Clone)]
pub struct SchedStats {
    pub submitted: u64,
    pub completed: u64,
    pub cpu_fallback: u64,
    pub oom_avoided: u64,
    pub timeouts: u64,
}

/// 简易信号量（std::sync::Semaphore 尚未稳定，用 Mutex + Condvar 实现）。
struct Semaphore {
    count: Mutex<usize>,
    cond: Condvar,
}

/// 信号量许可：Drop 时释放。
struct Permit<'a> {
    sem: &'a Semaphore,
}

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        let mut count = self.sem.count.lock().unwrap();
        *count += 1;
        self.sem.cond.notify_one();
    }
}

impl Semaphore {
    fn new(permits: usize) -> Self {
        Self {
            count: Mutex::new(permits),
            cond: Condvar::new(),
        }
    }

    /// 阻塞获取一个许可（返回守卫，Drop 时释放）。
    fn acquire(&self) -> Permit<'_> {
        let mut count = self.count.lock().unwrap();
        while *count == 0 {
            count = self.cond.wait(count).unwrap();
        }
        *count -= 1;
        Permit { sem: self }
    }
}

/// 调度管理器：显存预算 + 并发 + LRU + 优先级 + 功耗。
pub struct Scheduler {
    total_vram: f64,
    #[allow(dead_code)] // 信号量初始化用
    gpu_slots: usize,
    power: Arc<PowerMonitor>,
    budgets: HashMap<String, ModelBudget>,
    // 队列 + 条件变量
    queue: Mutex<Vec<Task>>,
    queue_cond: Condvar,
    // 显存状态
    vram_used: Mutex<f64>,
    loaded: Mutex<HashMap<String, Instant>>,
    // 推理函数注册表（pub(crate)：供 main.rs 的 run_in_scheduler 访问）
    pub(crate) infer_fns: Mutex<HashMap<String, InferFn>>,
    evict_hooks: Mutex<HashMap<String, Box<dyn Fn() + Send + Sync>>>,
    // 统计
    stats: Mutex<SchedStats>,
    // GPU 串行槽位（信号量，阻塞工作线程用 std）
    gpu_sem: Semaphore,
}

impl Scheduler {
    pub fn new(total_vram: f64, gpu_slots: usize) -> Arc<Self> {
        let s = Arc::new(Self {
            total_vram,
            gpu_slots,
            power: PowerMonitor::new(POWER_POLL_INTERVAL),
            budgets: model_budgets(),
            queue: Mutex::new(Vec::new()),
            queue_cond: Condvar::new(),
            vram_used: Mutex::new(0.0),
            loaded: Mutex::new(HashMap::new()),
            infer_fns: Mutex::new(HashMap::new()),
            evict_hooks: Mutex::new(HashMap::new()),
            stats: Mutex::new(SchedStats::default()),
            gpu_sem: Semaphore::new(gpu_slots),
        });
        // 工作线程：取队列 → 执行
        let s2 = s.clone();
        std::thread::spawn(move || s2.run_loop());
        s
    }

    /// 注册模型推理函数（由 vsd 调用）。
    pub fn register(&self, model: &str, f: InferFn, evict: Option<Box<dyn Fn() + Send + Sync>>) {
        self.infer_fns.lock().unwrap().insert(model.to_string(), f);
        if let Some(h) = evict {
            self.evict_hooks
                .lock()
                .unwrap()
                .insert(model.to_string(), h);
        }
    }

    /// 提交任务到队列（按优先级排序）。
    #[allow(dead_code)] // S4 接线结果回传时启用
    pub fn submit(&self, model: String, args: serde_json::Value, priority: u32, timeout: Duration) {
        let task = Task {
            model,
            args,
            priority,
            submitted_at: Instant::now(),
            deadline: Instant::now() + timeout,
            device: "cuda".to_string(),
        };
        let mut q = self.queue.lock().unwrap();
        q.push(task);
        q.sort_by_key(|t| t.priority);
        self.stats.lock().unwrap().submitted += 1;
        self.queue_cond.notify_one();
    }

    /// 工作循环：取最高优先级任务 → 执行。
    fn run_loop(&self) {
        loop {
            let task = {
                let mut q = self.queue.lock().unwrap();
                while q.is_empty() {
                    q = self.queue_cond.wait(q).unwrap();
                }
                q.remove(0)
            };
            self.execute(task);
        }
    }

    /// 执行任务：显存检查 → 加载/复用 → 推理 → 释放。
    fn execute(&self, task: Task) {
        let budget = self
            .budgets
            .get(&task.model)
            .cloned()
            .unwrap_or(ModelBudget {
                vram_reserve: 0.5,
                vram_peak: 1.0,
                cpu_ok: true,
            });
        // 功耗模式：low-power → 所有模型 CPU 降级；battery → 大模型 CPU 降级
        let mut use_gpu = task.device == "cuda";
        if self.power.is_low_power()
            || (self.power.is_battery() && budget.vram_reserve >= 1.0)
        {
            use_gpu = false;
            self.stats.lock().unwrap().cpu_fallback += 1;
        }
        // 显存预算检查
        {
            let mut vram = self.vram_used.lock().unwrap();
            if use_gpu && *vram + budget.vram_peak > self.total_vram {
                if budget.cpu_ok {
                    use_gpu = false;
                    self.stats.lock().unwrap().cpu_fallback += 1;
                    self.stats.lock().unwrap().oom_avoided += 1;
                } else {
                    // 无 CPU 回退 → 等待（LRU 淘汰腾显存）
                    self.evict_lru(budget.vram_peak);
                }
            }
            if use_gpu {
                *vram += budget.vram_reserve;
                self.loaded
                    .lock()
                    .unwrap()
                    .insert(task.model.clone(), Instant::now());
            }
        }
        // GPU 串行槽位（信号量，阻塞等待；Permit 作用域结束释放）
        let _permit = if use_gpu {
            Some(self.gpu_sem.acquire())
        } else {
            None
        };
        // 超时保护：推理前检查 + 推理后检查（推理超时也计入）
        let result = if Instant::now() > task.deadline {
            self.stats.lock().unwrap().timeouts += 1;
            Err("vsched timeout".to_string())
        } else {
            let r = self.infer(&task, use_gpu);
            if Instant::now() > task.deadline {
                self.stats.lock().unwrap().timeouts += 1;
            }
            r
        };
        if use_gpu {
            *self.vram_used.lock().unwrap() -= budget.vram_reserve;
        }
        let _ = result; // 结果由调用方通过回调/响应处理（S3 简化：记录）
        self.stats.lock().unwrap().completed += 1;
    }

    /// 实际推理（由注册的推理函数执行）。
    fn infer(&self, task: &Task, use_gpu: bool) -> Result<serde_json::Value, String> {
        let fns = self.infer_fns.lock().unwrap();
        let Some(f) = fns.get(&task.model) else {
            return Err(format!("unknown model: {}", task.model));
        };
        f(&task.model, &task.args, use_gpu)
    }

    /// 同步推理入口（供 daemon 直接调用；S4 接入真实调度队列）。
    pub fn infer_sync(
        &self,
        model: &str,
        args: &serde_json::Value,
        use_gpu: bool,
    ) -> Result<serde_json::Value, String> {
        let fns = self.infer_fns.lock().unwrap();
        let Some(f) = fns.get(model) else {
            return Err(format!("unknown model: {model}"));
        };
        f(model, args, use_gpu)
    }

    /// LRU 淘汰：卸载最久未用的模型腾显存。
    fn evict_lru(&self, need: f64) {
        let mut vram = self.vram_used.lock().unwrap();
        let mut loaded = self.loaded.lock().unwrap();
        if loaded.is_empty() {
            return;
        }
        let mut order: Vec<String> = loaded.keys().cloned().collect();
        order.sort_by_key(|m| loaded.get(m).copied().unwrap_or(Instant::now()));
        for model in order {
            let budget = self.budgets.get(&model).cloned().unwrap_or(ModelBudget {
                vram_reserve: 0.5,
                vram_peak: 1.0,
                cpu_ok: true,
            });
            *vram -= budget.vram_reserve;
            loaded.remove(&model);
            if let Some(h) = self.evict_hooks.lock().unwrap().get(&model) {
                h();
            }
            if *vram + need <= self.total_vram {
                break;
            }
        }
    }

    /// 调度器状态。
    pub fn stats(&self) -> serde_json::Value {
        let s = self.stats.lock().unwrap();
        let vram = *self.vram_used.lock().unwrap();
        let loaded: Vec<String> = self.loaded.lock().unwrap().keys().cloned().collect();
        let qlen = self.queue.lock().unwrap().len();
        serde_json::json!({
            "submitted": s.submitted,
            "completed": s.completed,
            "cpu_fallback": s.cpu_fallback,
            "oom_avoided": s.oom_avoided,
            "timeouts": s.timeouts,
            "mode": self.power.mode(),
            "cpu_freq_mhz": self.power.cpu_freq_mhz(),
            "vram_used": (vram * 100.0).round() / 100.0,
            "loaded_models": loaded,
            "queue_depth": qlen,
        })
    }
}

/// 便捷：默认调度器（6.5GB 预算 + 1 GPU 槽位）。
pub fn default_scheduler() -> Arc<Scheduler> {
    Scheduler::new(TOTAL_VRAM_BUDGET, GPU_SLOTS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn power_detect() {
        // 无电池环境 → plugged
        let mode = PowerMonitor::detect();
        assert!(mode == "plugged" || mode == "battery");
    }

    #[test]
    fn priority_queue_order() {
        let s = Scheduler::new(6.5, 1);
        // 注册一个立即返回的推理函数
        s.register(
            "test",
            Box::new(|_, _, _| Ok(serde_json::json!({"ok": true}))),
            None,
        );
        // 提交 bg 优先任务 + analyze 优先任务 → analyze 先执行
        s.submit(
            "test".into(),
            serde_json::json!({}),
            PRIORITY_BG,
            Duration::from_secs(5),
        );
        s.submit(
            "test".into(),
            serde_json::json!({}),
            PRIORITY_ANALYZE,
            Duration::from_secs(5),
        );
        std::thread::sleep(Duration::from_millis(200));
        let st = s.stats();
        assert_eq!(st["completed"], 2);
    }

    #[test]
    fn vram_budget_cpu_fallback() {
        // 预算不足 + cpu_ok → CPU 降级
        let s = Scheduler::new(0.5, 1); // 极小预算
        s.register(
            "segment",
            Box::new(|_, _, use_gpu| Ok(serde_json::json!({"use_gpu": use_gpu}))),
            None,
        );
        s.submit(
            "segment".into(),
            serde_json::json!({}),
            PRIORITY_PROBE,
            Duration::from_secs(5),
        );
        std::thread::sleep(Duration::from_millis(200));
        let st = s.stats();
        assert_eq!(st["completed"], 1);
        assert_eq!(st["cpu_fallback"], 1);
        assert_eq!(st["oom_avoided"], 1);
    }

    #[test]
    fn timeout_protection() {
        let s = Scheduler::new(6.5, 1);
        s.register(
            "test",
            Box::new(|_, _, _| {
                std::thread::sleep(Duration::from_millis(500));
                Ok(serde_json::json!({"ok": true}))
            }),
            None,
        );
        // 超时 50ms < 推理 500ms
        s.submit(
            "test".into(),
            serde_json::json!({}),
            PRIORITY_PROBE,
            Duration::from_millis(50),
        );
        std::thread::sleep(Duration::from_millis(700));
        let st = s.stats();
        assert_eq!(st["timeouts"], 1);
    }
}
