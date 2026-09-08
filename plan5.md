# plan5.md — GUI 控制性能修复（对标 Codex：从 45s/轮 到 1s/轮）

> 定位：plan4 G3 实测失败后的性能修复计划。Blender 莫比乌斯环测试 10 分钟未完成
> （Codex 约 5 秒），根因是**架构选错主通道**——我们用 DL 传感器（像素推理 15s）
> 做感知，Codex 用 AX 树（文本结构 ~100ms）。
>
> 目标：每轮感知 ≤1s（当前 15s）、行动 ≤0.1s（当前 15s）、单任务 ≤10 轮。

---

## 1. 实测数据（2026-09-08，4K 全屏 5760×1800）

### 1.1 单轮成本对比

| 环节 | 我们 | Codex | 差距 |
| --- | --- | --- | --- |
| capture | 1.1s（grim 全屏） | ~50ms（窗口级） | 22× |
| ocr | 7.7s（rapidocr 4K） | —（不用 OCR） | ∞ |
| detect | 4.0s（OWLv2 960²） | —（不用检测） | ∞ |
| omniparser | 0.9s（YOLOv9+Florence-2） | —（不用） | ∞ |
| scene_stats | 0.6s | — | ∞ |
| edge | 0.8s | — | ∞ |
| **analyze 合计** | **~15s** | **~100ms**（AX 树） | **150×** |
| 行动（click） | **15s**（需 analyze 拿坐标） | **~50ms**（AX 动作直达） | **300×** |
| 验证 | 1.1s（截图对比） | ~100ms（AX 快照） | 10× |
| **每轮总成本** | **~45s** | **~1-2s** | **~30×** |

### 1.2 关键实测发现

- **a11y 树完全不可用**：Blender/Firefox 均返回 0 元素
  （`a11y` 无 --app 时 apps=[]，AT-SPI 总线在运行但树读不到）
- **type 键盘到达**：kitty 终端 type "TEST123" 前后差异 147M——键盘注入基本可用
- **Shift+A 在 Blender 无效**：gui key / ydotool keycode 均失败（组合键问题）
- **坐标换算已修**：物理 5760×1800 → 逻辑 4608×1440（÷1.25）
- **终端污染已修**：analyze --window 过滤

---

## 2. 根因分析（4 层）

### 根因 1：感知架构选错主通道（150×，最根本）

```text
Codex:  AX 树 = 文本（"按钮[3] 名称=发送 位置=..."）→ 零 DL 推理 ~100ms
我们:   capture → ocr(7.7s) + detect(4s) + omniparser(0.9s) → 融合 → 15s
```

AX 树是应用主动暴露的结构（零推理），我们用 DL 从像素猜结构。**a11y.rs 已实现但 analyze 默认走 DL 传感器**——架构上选错了主通道。

### 根因 2：行动架构差距（300×）

Codex 的 AX 动作直达（~50ms）；我们的 vision_click 需先 analyze（15s）拿坐标。**AT-SPI DoAction 已实现（G1）但真实应用不可用**（树读不到）。

### 根因 3：模型差距（不可改变，可缓解）

GPT-6 Astra 多模态 + 快推理；DeepSeek-V4-Flash 纯文本 + 慢推理。缓解：每轮 ≤1K tokens + 减少轮次。

### 根因 4：试错放大（10 分钟放大器）

Shift+A 无效试 3-4 种方式、坐标偏位重试、终端污染决策错误——每轮 45s × 3-4 次试错 = 3 分钟。

---

## 3. 修复方案（按优先级）

### F1 · 打通 a11y 树（P0，感知 150× 的前提）

**目标**：a11y 树在真实应用可用（Blender/Firefox 返回元素）。

- [ ] 诊断 AT-SPI 树读不到的原因：
  - atspi crate 的 `get_desktop` / `get_apps` 调用链（可能 bus name 解析错误）
  - 对比 gui_a11y.py（Python 版能读到 Firefox——实测 apps 列表有 Firefox）
  - 用 busctl 直接调 AT-SPI 接口验证（GetChildren / GetApplication）
- [ ] 修复 a11y.rs 的树读取（对齐 Python 版 gui_a11y.py 的调用方式）
- [ ] 门禁：`vs-bin a11y --app firefox` 返回 ≥1 元素；`--app blender` 返回 ≥1 元素

### F2 · a11y 树优先的 analyze（P0，感知 15s → 0.1s）

**目标**：analyze 默认走 AT-SPI（~100ms），DL 传感器仅兜底。

- [ ] analyze 流程改为：a11y 树（主）→ 融合元素（DL 兜底，仅当 a11y 空）
- [ ] a11y 元素并入统一编号（element_fusion merge_a11y 已有）
- [ ] 门禁：analyze 在 a11y 可用时 ≤1s；a11y 空时降级 DL（≤15s）

### F3 · AT-SPI 动作直达（P0，行动 15s → 0.05s）

**目标**：click [N] 优先 DoAction（~50ms），vision_click 仅兜底。

- [ ] 打通 DoAction 在真实应用的调用（F1 修复后）
- [ ] click [N] 流程：a11y 动作（主）→ element_click（GetExtents）→ vision_click（兜底）
- [ ] 门禁：真实应用 click [N] 用 a11y 动作 ≤0.1s

### F4 · 键盘组合键修复（P1，消除试错）

**目标**：Shift+A 等组合键在 Blender 可用。

- [ ] 诊断 wtype/ydotool 组合键问题（shift 修饰键时序）
- [ ] 测试：wtype -M shift -k a / ydotool key 42:1 30:1 42:0 30:0
- [ ] 门禁：Blender 中 Shift+A 打开 Add 菜单（截图验证）

### F5 · 窗口级 capture（P1，capture 1.1s → 0.1s）

**目标**：只截目标窗口（niri screenshot-window）。

- [ ] capture 支持 --window ID（niri msg action screenshot-window）
- [ ] 门禁：窗口级 capture ≤0.2s

### F6 · 增量感知（P1，每轮 15s → 1s）

**目标**：事件驱动 + 缓存，只重扫变化区域。

- [ ] PerceptionCache 接入 analyze（G3 已实现缓存，接入默认路径）
- [ ] 门禁：连续 analyze 同窗口 ≤1s（缓存命中）

### F7 · 上下文精简（P2，决策加速）

**目标**：每轮 ≤1K tokens（当前 2.5K）。

- [ ] analyze 元素 top-10 截断 + 字段精简（当前 top-15）
- [ ] 门禁：analyze 输出 ≤1K tokens

---

## 4. 门禁标准（对标 Codex）

| 指标 | 当前 | 目标 |
| --- | --- | --- |
| 感知（analyze） | 15s | **≤1s**（a11y 主通道） |
| 行动（click [N]） | 15s | **≤0.1s**（AX 动作直达） |
| 每轮总成本 | 45s | **≤2s** |
| 单任务轮次 | 10+（试错） | **≤5**（无试错） |
| 上下文 | 2.5K tokens | **≤1K** |
| 真实任务 | Blender 莫比乌斯环 10min 未完成 | **≤30s** 完成 |

---

## 5. 里程碑

| 里程碑 | 内容 | 预计 |
| --- | --- | --- |
| M1 | F1+F2 完成（a11y 树打通 + 主通道切换） | 感知 15s → 1s |
| M2 | F3+F4 完成（动作直达 + 键盘修复） | 行动 15s → 0.1s |
| M3 | F5-F7 完成（窗口 capture + 增量 + 精简） | 每轮 ≤2s |
| M4 | Blender 莫比乌斯环端到端验证 | ≤30s 完成 |

---

## 6. 风险与缓解

| 风险 | 等级 | 缓解 |
| --- | --- | --- |
| AT-SPI 树在部分应用仍不可用 | 中 | DL 传感器兜底（现有路径） |
| DoAction 对部分应用无效 | 中 | element_click（GetExtents）兜底 |
| wtype 组合键无法修复 | 中 | ydotool keycode 时序 + 鼠标路径兜底 |
| a11y 树 token 大（1200 节点） | 低 | A11y-Compressor 压缩（业界方案） |
| 工作量（~1,500 行 Rust） | 高 | 分阶段 Gate-Based |
