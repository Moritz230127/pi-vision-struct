# ⛔ 项目终止维护 / End of Life

**本仓库已于 2026-09-13 正式封库，不再维护与更新。**

## 中文

### 决定

`pi-version-and-gui` 项目自即日起停止一切维护与更新，仓库归档为**只读**，
不再接受 issue 与 PR。这是本项目的最后一次公告。

### 原因

本项目的核心前提是「纯文本模型需要外部视觉通道」——把图片转化为坐标 / hex /
矩阵 / 统计等精确数理数据，供文本模型做数值推理与 GUI 控制。

DeepSeek-V4.1 已原生支持多模态输入，该前提不再成立，本项目随之失去存在意义。
与其继续维护一套上游已经内建的旁路方案，不如就此封存。

### 仍然可用

- 全部代码、文档、提交历史与 **MIT 许可**保留，可自由使用、fork、二次开发。
- GitHub Release 中的预编译二进制 `vs-bin` / `vsd` / `vs-mcp`（v4.0.0）**仍可下载**，
  已有的 npm 安装（postinstall 从 Release 拉取二进制）**不会中断**。
- 模型缓存 `~/.cache/vsensor/` 与本地数据不受影响。

### 不再提供

- 功能更新、安全修复、依赖升级与兼容性适配。
- issue / pull request 响应（仓库归档后无法新建）。
- 请注意：AT-SPI / Wayland / niri 等环境变更可能导致工具失效，请自行评估后使用。

### 最后版本状态

| 项目 | 值 |
| --- | --- |
| 最后版本 | v4.0.3（npm `pi-version-and-gui@4.0.3`） |
| 架构 | 全 Rust（vs-core 37 模块 / vs-bin 30+ 子命令 / vsd / vs-mcp 28 工具）+ TS 薄壳 |
| 验证 | 241 测试全绿、clippy 0 警告、12 项差分门禁 |
| 平台 | 仅 Linux x86_64（niri Wayland） |
| 许可 | MIT |

---

## English

### Decision

The `pi-version-and-gui` project has stopped all maintenance and updates as of
2026-09-13. The repository is now **archived (read-only)** and no longer accepts
issues or pull requests. This is the project's final announcement.

### Why

The project's premise was that text-only models need an external vision channel —
converting images into precise numeric data (coordinates / hex / matrices /
statistics) for numeric reasoning and GUI control.

DeepSeek-V4.1 now supports multimodal input natively, so that premise no longer
holds and the project has lost its purpose. Rather than keep maintaining a bypass
that upstream has since built in, it is sealed here.

### What still works

- All source, docs, commit history and the **MIT license** remain available for
  use, forking and further development.
- Prebuilt binaries `vs-bin` / `vsd` / `vs-mcp` in the v4.0.0 Release **remain
  downloadable**, so existing npm installs (postinstall fetches them from
  Releases) **keep working**.
- The model cache at `~/.cache/vsensor/` and local data are unaffected.

### No longer provided

- Feature updates, security fixes, dependency upgrades, compatibility work.
- Issue / pull request responses (the archived repo cannot accept new ones).
- Note: changes around AT-SPI / Wayland / niri may break the tooling — evaluate
  before relying on it.

### Final state

| Item | Value |
| --- | --- |
| Last version | v4.0.3 (npm `pi-version-and-gui@4.0.3`) |
| Architecture | All-Rust (vs-core 37 modules / vs-bin 30+ subcommands / vsd / vs-mcp 28 tools) + thin TS shell |
| Verification | 241 tests green, clippy 0 warnings, 12 differential gates |
| Platform | Linux x86_64 only (niri Wayland) |
| License | MIT |

---

**感谢所有使用、测试与反馈过这个项目的人。**

**Thanks to everyone who used, tested and gave feedback on this project.**
