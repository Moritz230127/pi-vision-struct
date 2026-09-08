# plan4.md — pi-version-and-gui：深度融合 GUI 控制（对标 Ch

  atGPT-6 Astra + Codex Computer Use）

> 定位：V4 第二阶段工程。在 plan3（感知层 E1-E6）之上，实现**感知-行动闭环**的
> GUI 控制能力——效率对标 ChatGPT-6 Astra + Codex Computer Use，精度对标
> Codex Windows/macOS，最终以 `pi install npm:pi-version-and-gui` 发布。
>
> 核心架构（用户确认）：**深度融合** = 感知/行动/协议/事件共享同一数据模型
> （element_index），单进程零 IPC。这是 Codex 高效的本质（共
  享数据模型 + 闭环），
> 单进程是超越点。

---

## 1. 技术基础盘点（实测确认，2026-09-08）

| 能力 | 现状 | 来源 |
| --- | --- | --- |
| 绝对定位 | wlrctl v0.2.2（zwlr_virtual_pointer_v1 motion_absolute）+ serve 持久连接 | pi-gui-control 已验证 ✅ |
| 窗口管理 | niri msg（windows/outputs/focused-window JSON + action + **event-stream**） | 系统原生 ✅ |
| AX 树 | atspi crate 0.30 + **GetExtents 屏幕坐标**（a11y.rs 已实现，Firefox 重试 3 次） | vs-core ✅ |
| **AX 动作** | **atspi DoAction 未实现**（五级降级链第 1 级前提） | ⚠️ 需新增 |
| 感知融合 | element_fusion（编号）+ hit_test（命中 100%）+ capture + verify_click | plan3 E1-E6 ✅ |
| 分层协议 | analyze → zoom → probe（编号引用协议） | plan3 E3 ✅ |
| 统一调度 | vsd daemon（tokio + 持久连接 + 沙箱） | S3 ✅ |
| 键盘 | ydotool keycode（niri 兼容，enter=28 已验证）+ wtype | pi-gui-control ✅ |
| 事件流 | **niri event-stream 确认可用**（Workspaces/Windows changed 事件） | 实测 ✅ |

**结论**：所有底层能力已就绪，缺的是**行动层（gui.rs）+ AX
  动作（DoAction）+ 事件驱动（events.rs）+ npm 布局**。

---

## 2. 架构：感知-行动闭环（单进程，Codex 同款 + 超越）

```text
pi-version-and-gui（npm 包，monorepo 单包）
├── extensions/index.ts        # TS 薄壳（exec vs-bin，~50 行，零业务逻辑）
├── skills/gui-control/        # skill（编号引用协议教学：click [N]）
├── scripts/install.js         # postinstall：GitHub Releases 下载二进制
├── bin/                       # 3 个 Rust 二进制（Releases 下载，不打包）
└── models/                    # 模型清单（下载 URL，不打包）

vsd daemon（单进程，tokio 统一调度）
├── 感知层（已有 E1-E6）
│   ├── element_fusion → 编号 [0]...[N]
│   ├── a11y → AT-SPI 树 + GetExtents 屏幕坐标
│   └── capture / hit_test / verify_click
├── 行动层（新增 gui.rs）
│   └── 五级降级链：a11y_action → element_click → vision_click → global
├── 协议层（扩展 protocol.rs）
│   └── analyze → zoom → probe → click [N] → verify
└── 事件层（新增 events.rs）
    └── niri event-stream → 变化即感知（增量，节流 100ms）
```

---

## 3. 核心设计（对标 Codex 的每个机制）

### 3.1 五级点击降级链（Codex 同款，解决"不准"）

| 级别 | 方法 | 精度 | 适用 | 状态 |
| --- | --- | --- | --- | --- |
| 1 | **a11y_action**：AT-SPI `DoAction` 动作直达 | 零坐标误差 | 有 AX 树 + 动作的元素 | ⚠️ 需实现 |
| 2 | **element_click**：编号 → GetExtents 屏幕坐标 → 绝对定位 | ±0px | AX 树有位置无动作 | ✅ 基础已有 |
| 3 | **vision_click**：融合元素 bbox center → 绝对定位 | ±2px | 无 AX 树（游戏/Canvas） | ✅ 基础已有 |
| 4 | **global**：wlrctl serve 兜底 | ±2px | 全部失败时 | ✅ 已有 |

**每级失败自动降级**（Codex 多级降级链）。

### 3.2 element_index 统一引用（Codex 核心）

- 融合元素 + a11y 元素**统一编号** `[0]...[N]`
- DeepSeek 输出 `click [3]`，**不输出坐标**
- 运行时：编号 → 查表 → 优先 a11y 动作，其次 GetExtents 坐标
- **绕过了坐标换算**（Codex 洞察："AX 树不仅用于看，更用于点"）

### 3.3 闭环验证（Codex 基石）

```text
click [3] → 执行 → verify_click（截图对比）→ 成功？→ 再感知
                              ↓ 失败
                   降级下一级 → 重试（≤2 次）
```

### 3.4 事件驱动（超越 Codex）

- niri `event-stream` 监听窗口/焦点变化 → **变化即感知**（增量重扫）
- Codex 是"行动后快照"，我们做到"变化即感知"——少一轮
- 节流 100ms 防事件风暴

### 3.5 坐标换算（Codex 三层）

```text
AT-SPI GetExtents（屏幕坐标）──┐
capture 像素（scale 换算）─────┼→ 逻辑坐标（union 4608×1440）
niri 窗口信息（tile 启发式）───┘
```

### 3.6 键盘优先（pi-gui-control 已有）

- focus + key combos（ctrl+l, super+enter）
- ydotool keycode（niri 兼容，enter=28 已验证）

---

## 4. npm 布局（pi install npm:pi-version-and-gui）

### 4.1 关键约束：二进制 151MB（vs-bin 57 + vsd 34 + vs-mcp 60）

**方案：GitHub Releases + postinstall 下载**（npm 包 <1MB）：

```text
pi-version-and-gui/                    # npm 包根（= 扩展根）
├── package.json        # name: pi-version-and-gui（npm
  已验证可用，404=未占用）
│                       # pi: { extensions: ["./extensions"
  ], skills: ["./skills"] }
│                       # scripts: { postinstall: "node scripts/install.js" }
├── extensions/
│   └── index.ts        # TS 薄壳（exec vs-bin，零业务逻辑）
├── skills/
│   └── gui-control/    # skill：教模型用编号引用协议（click [N]）
├── scripts/
│   └── install.js      # postinstall：从 GitHub Release
  s 下载二进制 + 模型清单
├── bin/                # 下载的二进制（vs-bin/vsd/vs-mcp）
├── models/             # 模型清单（下载 URL，不打包）
└── README.md
```

### 4.2 安装流程

`pi install npm:pi-version-and-gui` → npm 装 TS 壳（<1MB）
  → postinstall 从 GitHub
Releases 下载 3 个二进制（~60MB 压缩）→ 模型按需下载（~/.ca
  che/vsensor，~2GB）。

### 4.3 布局决策

- **monorepo 单包**：视觉 + GUI 一个包（深度融合的物理体现）
- **二进制不打包**：npm 包保持轻量，二进制走 Releases
- **模型不打包**：~/.cache/vsensor 按需下载
- **TS 壳保留**：pi 平台约束（ExtensionAPI 是 TS 原生），~50 行 exec 胶水

---

## 5. 阶段划分（Gate-Based，每阶段独立交付）

### G1 · 行动层（gui.rs + 五级降级链）

**目标**：从"能感知"到"能行动"。

- [ ] `gui.rs`：五级点击降级链（a11y_action → element_click
   → vision_click → global）
- [ ] `a11y.rs` 扩展：AT-SPI `DoAction`（动作直达）
- [ ] `element_fusion.rs` 扩展：a11y 元素并入统一编号
- [ ] `protocol.rs` 扩展：`click [N]` 动作 + verify 闭环
- [ ] vs-bin 子命令 `gui`（click/type/key/drag/scroll/focus）
- **门禁**：合成测试五级降级链全通；a11y 动作命中率 100%；click [N] 往返 <100ms

### G2 · 事件驱动（events.rs + 增量再感知）

**目标**：从"行动后快照"到"变化即感知"。

- [ ] `events.rs`：niri event-stream 监听（tokio）+ 节流 100ms
- [ ] 增量再感知：窗口/焦点变化 → 只重扫变化区域
- [ ] vsd 集成：事件循环 + 感知缓存
- **门禁**：事件流测试（窗口变化 → 触发再感知）；节流生效

### G3 · 端到端闭环（真实 GUI 控制）

**目标**：真实应用验证"编号引用 → 定位 → 点击 → 验证"。

- [ ] 真实 GUI 测试：浏览器/文件管理器/终端（analyze → click [N] → verify）
- [ ] 效率基准：每轮 token 数 + 往返延迟（对标 Astra+Codex）
- [ ] 精度基准：命中率 + 坐标误差（对标 Codex Win/mac）
- **门禁**：真实应用点击命中率 ≥ 99%；单任务 ≤5 轮；每轮 ≤2K tokens

### G4 · npm 发布布局（不发布，仅布局）

**目标**：`pi install npm:pi-version-and-gui` 一键安装。

- [ ] monorepo 重组：pi-version-and-gui 包结构（extensions/s
  kills/scripts/bin/models）
- [ ] `install.js`：GitHub Releases 下载 + 模型清单
- [ ] 重命名：pi-vision-struct → pi-version-and-gui（扩展目
  录 + package.json + 知识库）
- [ ] skill：gui-control（编号引用协议教学）
- **门禁**：本地 `npm pack` + 模拟安装成功；二进制下载脚本可执行

### G5 · 全面验证（收尾）

- [ ] 全量回归（全部测试全绿 + clippy 0 + 零 Python 回退）
- [ ] 效率/精度基准卡（对标 Codex 数据）
- [ ] 文档同步（README / SKILL.md / 知识库）
- **门禁**：全绿 + kb-audit 通过

---

## 6. 门禁标准（对标验证矩阵）

| 标准 | 门禁 |
| --- | --- |
| 效率 ≥ Astra+Codex | 每轮 ≤2K tokens；单任务 ≤5 轮；click [N] 往返 <100ms |
| 精度 ≥ Codex Win/mac | a11y 动作命中率 100%；GetExtents 坐标 ±0px；五级降级链全通 |
| 深度融合 | 感知/行动共享 element_index；单进程零 IPC；事件驱动 |
| npm 发布 | `pi install npm:pi-version-and-gui` 一键安装；postinstall 下载二进制 |
| 回归 | 全量测试全绿 + clippy 0 + 零 Python 回退 |

---

## 7. 里程碑

| 里程碑 | 内容 | 预计 |
| --- | --- | --- |
| M1 | G1 完成（行动层 + 五级降级链） | 第一批可交付（能行动） |
| M2 | G2 完成（事件驱动） | 变化即感知 |
| M3 | G3 完成（端到端闭环） | 真实 GUI 控制达标 |
| M4 | G4-G5 完成（npm 布局 + 验证） | 待定（不发布） |

---

## 8. 风险与缓解

| 风险 | 等级 | 缓解 |
| --- | --- | --- |
| AT-SPI 树残缺（游戏/Canvas） | 中 | 降级到 vision_click（融合元素） |
| GetExtents 对部分应用不可靠 | 中 | 重试 3 次（已有）+ 降级 |
| atspi DoAction 实现复杂 | 中 | 先实现 element_click（GetExtents），DoAction 后补 |
| wlrctl serve 需 Wayland 会话 | 低 | vsd 检测 + 报错 |
| 虚拟指针被应用忽略（少数） | 低 | 键盘优先 + focus 兜底 |
| niri event-stream 事件风暴 | 低 | 节流 100ms 合并 |
| npm 二进制下载失败 | 低 | install.js 重试 + 手动下载提示 |
| 工作量（~2,500 行 Rust） | 高 | 分阶段 Gate-Based，每阶段独立交付 |
