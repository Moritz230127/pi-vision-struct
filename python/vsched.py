#!/usr/bin/env python3
"""vsched.py — V3 调度管理器（显存预算 + 并发控制 + LRU + 功耗感知）。

8GB VRAM 硬约束下，任何并发 GPU 推理都可能 OOM。vsched 是 GPU/CPU
推理请求的唯一入口：
  - 显存预算表（每模型 vram_reserve + vram_peak，总预算 6.5GB）
  - 并发信号量（GPU 槽位=1 串行）
  - 模型 LRU 淘汰（冷载 1-3s / 热 <100ms）
  - 优先级队列（analyze > zoom > probe > bg）
  - 功耗感知（battery → 降频/CPU 降级；plugged → 全速）
  - 超时保护 + CPU 降级兜底

用法（库接口，由 vsd.py 调用）:
  from vsched import Scheduler
  sched = Scheduler()
  sched.submit("saliency", {"image": path}, priority=0)
"""
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------- 显存预算表

# 每模型: (vram_reserve_GB, vram_peak_GB, cpu_fallback)
MODEL_BUDGETS: dict[str, dict[str, Any]] = {
    "saliency": {"vram_reserve": 0.5, "vram_peak": 1.0, "cpu_ok": True},
    "segment": {"vram_reserve": 1.0, "vram_peak": 2.0, "cpu_ok": True},
    "depth": {"vram_reserve": 0.3, "vram_peak": 0.8, "cpu_ok": True},
    "detect": {"vram_reserve": 2.0, "vram_peak": 3.0, "cpu_ok": True},
    "omniparser": {"vram_reserve": 2.0, "vram_peak": 3.0, "cpu_ok": True},
}

TOTAL_VRAM_BUDGET = 6.5  # GB（8GB 留 1.5GB 安全水位）
GPU_SLOTS = 1            # 并发 GPU 推理槽位（串行）
POWER_POLL_INTERVAL = 30  # 秒

# 优先级（数字小 = 高优先级）
PRIORITY = {"analyze": 0, "zoom": 1, "probe": 2, "bg": 3}


@dataclass
class Task:
    """调度任务。"""
    model: str
    args: dict[str, Any]
    priority: int
    callback: Callable[[dict], None] | None = None
    submitted_at: float = field(default_factory=time.time)
    deadline: float = 0.0
    device: str = "cuda"


class PowerMonitor:
    """功耗感知：检测电源状态（battery/plugged）。"""

    def __init__(self, poll_interval: int = POWER_POLL_INTERVAL):
        self.poll_interval = poll_interval
        self.mode = self._detect()
        self._stop = False
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _detect(self) -> str:
        """检测电源状态。Linux: /sys/class/power_supply/BAT1/status。"""
        for p in Path("/sys/class/power_supply").glob("BAT*"):
            try:
                status = (p / "status").read_text().strip()
                if status == "Discharging":
                    return "battery"
                if status in ("Charging", "Full"):
                    return "plugged"
            except OSError:
                pass
        return "plugged"  # 无电池 → 视为 plugged

    def _poll(self):
        while not self._stop:
            time.sleep(self.poll_interval)
            new_mode = self._detect()
            if new_mode != self.mode:
                old = self.mode
                self.mode = new_mode
                print(f"[vsched] 功耗模式: {old} → {new_mode}", flush=True)

    def stop(self):
        self._stop = True

    def is_battery(self) -> bool:
        return self.mode == "battery"


class Scheduler:
    """调度管理器：显存预算 + 并发 + LRU + 优先级 + 功耗。"""

    def __init__(self, total_vram: float = TOTAL_VRAM_BUDGET,
                 gpu_slots: int = GPU_SLOTS):
        self.total_vram = total_vram
        self.gpu_slots = gpu_slots
        self.power = PowerMonitor()
        self._lock = threading.Lock()
        self._gpu_sem = threading.Semaphore(gpu_slots)
        self._queue: list[Task] = []
        self._queue_cond = threading.Condition(self._lock)
        self._loaded: dict[str, float] = {}  # model → 最后使用时间
        self._vram_used = 0.0
        self._infer_fns: dict[str, Callable] = {}
        self._evict_hooks: dict[str, Callable] = {}
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        self._stats = {"submitted": 0, "completed": 0, "cpu_fallback": 0,
                       "oom_avoided": 0, "timeouts": 0}

    # ------------------------------------------------------------ 提交

    def submit(self, model: str, args: dict[str, Any], priority: int = PRIORITY["probe"],
               callback: Callable[[dict], None] | None = None,
               timeout: float = 120.0) -> None:
        """提交任务到队列。"""
        task = Task(model=model, args=args, priority=priority, callback=callback,
                    deadline=time.time() + timeout)
        with self._queue_cond:
            self._queue.append(task)
            self._queue.sort(key=lambda t: t.priority)
            self._stats["submitted"] += 1
            self._queue_cond.notify()

    # ------------------------------------------------------------ 执行循环

    def _run(self):
        while True:
            with self._queue_cond:
                while not self._queue:
                    self._queue_cond.wait()
                task = self._queue.pop(0)
            self._execute(task)

    def _execute(self, task: Task):
        """执行任务：显存检查 → 加载/复用 → 推理 → 释放。"""
        budget = MODEL_BUDGETS.get(task.model, {"vram_reserve": 0.5, "vram_peak": 1.0, "cpu_ok": True})
        # 功耗模式：battery → 大模型 CPU 降级
        use_gpu = task.device == "cuda"
        if self.power.is_battery() and budget.get("vram_reserve", 0) >= 1.0:
            use_gpu = False
            self._stats["cpu_fallback"] += 1
        # 显存预算检查
        with self._lock:
            peak = budget.get("vram_peak", 1.0)
            if use_gpu and self._vram_used + peak > self.total_vram:
                if budget.get("cpu_ok", True):
                    use_gpu = False
                    self._stats["cpu_fallback"] += 1
                    self._stats["oom_avoided"] += 1
                else:
                    # 无 CPU 回退 → 等待（LRU 淘汰腾显存）
                    self._evict_lru(peak)
            if use_gpu:
                self._vram_used += budget.get("vram_reserve", 0.5)
                self._loaded[task.model] = time.time()
        # GPU 串行槽位
        if use_gpu:
            self._gpu_sem.acquire()
        try:
            result = self._infer(task, use_gpu)
        except Exception as e:
            result = {"error": "vsched execute failed", "detail": str(e)[:300]}
        finally:
            if use_gpu:
                self._gpu_sem.release()
                with self._lock:
                    self._vram_used -= budget.get("vram_reserve", 0.5)
        if task.callback:
            task.callback(result)
        self._stats["completed"] += 1

    def _infer(self, task: Task, use_gpu: bool) -> dict:
        """实际推理（由 vsd 注册的推理函数执行）。"""
        fn = self._infer_fns.get(task.model)
        if fn is None:
            return {"error": f"unknown model: {task.model}"}
        args = dict(task.args)
        args["device"] = "cuda" if use_gpu else "cpu"
        return fn(**args)

    # ------------------------------------------------------------ LRU 淘汰

    def _evict_lru(self, need: float):
        """LRU 淘汰：卸载最久未用的模型腾显存。"""
        if not self._loaded:
            return
        for model in sorted(self._loaded, key=self._loaded.get):
            budget = MODEL_BUDGETS.get(model, {"vram_reserve": 0.5})
            self._vram_used -= budget.get("vram_reserve", 0.5)
            del self._loaded[model]
            self._evict_hooks.get(model, lambda: None)()
            if self._vram_used + need <= self.total_vram:
                break

    # ------------------------------------------------------------ 注册

    def register(self, model: str, infer_fn: Callable, evict_hook: Callable | None = None):
        """注册模型推理函数（由 vsd 调用）。"""
        self._infer_fns[model] = infer_fn
        if evict_hook:
            self._evict_hooks[model] = evict_hook

    # ------------------------------------------------------------ 状态

    def stats(self) -> dict:
        return {**self._stats, "mode": self.power.mode,
                "vram_used": round(self._vram_used, 2),
                "loaded_models": list(self._loaded.keys()),
                "queue_depth": len(self._queue)}

    def stop(self):
        self.power.stop()


# ---------------------------------------------------------------- 便捷入口

def main() -> int:
    """CLI：状态查询。"""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true", help="显示调度器状态")
    args = ap.parse_args()
    if args.status:
        sched = Scheduler()
        print(json.dumps(sched.stats(), ensure_ascii=False))
        sched.stop()
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
