# Changelog

## 2.2.2 — 2026-08-29（双宿主重构：pi-coding-agent + Claude Code/MCP）

### ① 重架构为双宿主一等扩展

- 抽离框架无关核心 `extensions/pi-vision-core.ts`（python 解析 / preflight / bwrap 沙箱 / Act 参数表 / dispatch），作为**单一事实源**。
- `extensions/index.ts` 仅保留 pi 适配器壳（`pi.registerTool` / `pi.registerCommand`），行为等价，对外契约不变。
- 新增 `extensions/server.mcp.ts`：MCP stdio server，暴露单端口 `vs` 工具，供 **Claude Code** 经 `claude mcp add` 接入。
- 两个宿主共用同一份 25 个 action 与同一份 Python 传感器（**零改动**）；`vsParamType` / `VS_DESCRIPTION` 由核心导出，杜绝双份 schema。

### ② 新增 check / setup 动作（双宿主对等）

- `check`：只读环境自检；`setup`：安装（接 `setup_args`，如 `--with-omniparser`）。
- Claude Code 侧等价于 pi 的 `/vs check` / `/vs setup`；两者结构一致。

### ③ 工程化

- `package.json`：`type: module`、`@modelcontextprotocol/sdk` 进 dependencies、`typebox` 由 peer→dependency；新增 `build` / `mcp` / `mcp:dev` 脚本。
- 新增 `tsconfig.mcp.json`（仅编译 `extensions/`，不被 pi 框架误读）；新增 `docs/claude-code.md`。
- `skills/vision-situation/SKILL.md` 改为主机无关（补 check/setup，修正 action 计数与 chart→chart_data）。

> **非破坏性**：pi 侧入口、default export 签名、命令/工具注册方式全部保留；Python 传感器一行未改。

## 2.2.0 — 2026-08-29（真几何深度 + VLM 语义兜底 + 依赖健壮性）

### ① 真几何深度（depth_geom）

- 新增 `vs_depth_geom.py` + `depth_geom_wrapper.sh`：在 Blender 内将所有 MESH 三角面栅格化到
  相机空间深度缓冲，得到真实几何距离（米），非亮度梯度近似。
- 向量化投影 + 扫描线光栅化（numpy）：480p 约 42s、1080p 约 57s（主要为 Blender 启动）。
- 涡扇实测：near=0.75m / far=2.05m / median≈1.9m，与引擎真实尺寸吻合。
- `depth` 动作现路由 Blender 几何深度（主用）；原亮度梯度 `vs_depth.py` 保留为无 Blend 时的回退。

### ② 可选 VLM 语义兜底（semantic）

- 新增 `vs_semantic_v2.py` + `semantic` 动作：不进入数理化主链，明确标记 `semantic_fallback: true`。
- 走本地 Ollama（127.0.0.1 硬编码，无外发），输出带 note 提示与数值报告区分。
- 仅当用户明确需要语义理解时调用（如这图讲了什么故事）。
- 图像自动下采样至 1024 边，避免超出 VLM 上下文（4K 图原会触发 400 错误）。

### ③ Python 依赖健壮性

- `vs_chart.py` 移除对 `vs_vlm` 的硬 import，改为内联 Ollama 网关（与 semantic 同源），
  Ollama 不可用时返回结构化错误 + 修复提示，不再崩溃。
- `depth` 动作不再依赖 torch（几何深度走 Blender，亮度回退走 PIL）；omniparser env 仍承载 detect/cluster/omniparser。
- 全部脚本 `py_compile` 通过；客户视角 21 例（含 depth_geom + semantic）零报错一次过。

## 2.1.0 — 2026-08-28（audit3d 最大精度重构）

### 问题

- 涡扇场景图 AABB 审计误报 1042 干涉：嵌套/同心结构（转子在静子内、机匣包裹压气机）的
  轴对齐包围盒天然重叠，但实际网格间存在装配间隙，并非真实穿透。

### 修复（最大精度）

- `vs_blender_dump.py`：导出真实世界坐标点云 `verts`（MESH 多边形顶点 + CURVE 样条离散点，
  每物体 ≤4000 下采样）；新增 `parent` 字段；保留 `collection` 层级。
- `vs_audit3d.py` 重写精度模型（三级递进，对每对 MESH/CURVE）：
  1. **AABB 预筛**：已分离的成对物体直接跳过（不计任何计数）
  2. **OBB-SAT**：用 `bbox3d` 8 角点直接构造有向包围盒，分离轴定理（15 轴）判定，
     旋转感知，比 AABB 紧致，消除旋转假干涉
  3. **网格级 surface-to-surface 距离**（核心）：`scipy.cKDTree` 双向最近邻求最小表面间距。
     环形间隙 → 正间隙判 `clearance`；真实接触 → 距离≈0 判 `interference`。
  - 新增 `--method`：`auto`(默认 OBB+Mesh) / `obb` / `mesh` / `aabb`(回退)
  - 新增 `verified_clearance` 指标：经网格级确认存在真实正间隙的成对数量
- 依赖：pi-vision env 新增 `scipy`（cKDTree 必需）；requirements.txt 更新。

### 验证（turbofan_v4.blend，172 MESH + 8 CURVE，16110 对）

- AABB 档：interference=1042，clearance=0
- OBB 档：interference=952，clearance=128
- **auto 档（最大精度）：interference=4，gap_warn=126，clearance=1056**
- 剩余 4 真干涉均经 KDTree 验证表面距离 0.0mm（spinner↔tip_cap、nacelle↔bypass_duct、
  core_casing↔bypass_duct、elbow↔elbow），为结构件真实接触，非误报。
- 客户视角 19/19 零报错一次通过；fusion 8/8。

## 1.0.0 — 2026-08-25（正式发布：Linux）

### 定位

- 平台收窄为 **Linux x86_64 only**：package.json `os:["linux"]`；macOS/Windows 安装器指引从 README 移除
- README 全面重写：详细架构设计（分层通道/单端口/自稳定层/沙箱模型/坐标系）+
  手把手安装教程（七步、每步带验证命令、不依赖一键脚本）

### 发布门禁证据

- **安装教程成功路径 Docker 全程实测**：纯净 ubuntu:24.04 按教程步骤 0→1→3 逐字执行——
  Miniforge 124MB 安装 → mamba 2.5.0 → pi-vision env → requirements.txt → 核心依赖导入 OK；
  vs_setup --check 无 git 环境按预期降级。加上此前失败路径双场景，安装体验两端均已验证
- 功能面：G1 静态全绿；G2 fusion 8/8；G3 客户视角 19 例 ONE_SHOT_PASS；注入 ~713 tokens
- 能力面（v0.4.0）：OWLv2 detect / chart 图表转数据 / l2_model 可配置

## 0.4.0 — 2026-08-25（能力补全：U1 L2可配置 + U2 detect + U3 chart + P3 发布验证）

### U2 · 自然图像检测传感器 vs_detect.py

- OWLv2 zero-shot（google/owlv2-base-patch16-ensemble，omniparser env）：任意文本类别 → bbox+conf
- 选型实录：ultralytics YOLO-World 8.4.x/8.3.x 实测 set_classes 嵌入异常
  （开放词分数≈0.01 噪声，bus.jpg wheel 仅 0.01；OWLv2 同图 wheel 0.358），故换轨
- 元素带 primitive 记法；权重首用 ~1.2GB 后离线

### U3 · 图表理解传感器 vs_chart.py

- 图表→结构化数据专用管线（title/x_axis/y_axis/series[{name,points}]/notes），
  走 vs_vlm 网关，max_tokens=4096 并剥离思考型 <think> 块后解析
- 合成柱状图实测：标题+四季度数值全部正确提取；理念=提取与推理分离

### U1 · L2 模型可配置

- config `pi-vision-struct.json` 新增 `l2_model` 键：semantic/critic/chart 默认档位
  一处切换（解析顺序 config > 内置默认）；--model 显式覆盖仍有效

### P3 · 发布验证（Docker 裸机最坏路径）

- ubuntu:24.04（无 python3）：安装器输出 "✗ 需要 python3（安装 miniforge 后会自动获得）"
- python:3.12-slim（无 conda）：结构化 JSON 指引 miniforge 安装 URL
- 两场景均无堆栈崩溃，exit 语义正确 —— 新鲜机器失败模式全部可操作

### 单端口注册

- vs 工具 action 枚举 18→20（detect/chart）；注入 713 tokens（仍为原 1359 的约一半）

## 0.3.0 — 2026-08-25（单端口架构：4 工具 → 1 个 `vs`）

### 架构（系统工程优化）

- vs_measure/vs_struct/vs_fuse/vs_cluster 四工具合并为单一 `vs` 工具（18 action 枚举分发，
  内部 ROUTE 路由表复用原三张分发表 + cluster 收编为标准 Act）；行为零变更
- 上下文注入实测 1359 → 704 tokens（-48%）
- 描述语言统一为中文紧凑风格；主描述压缩为路由指南（细节在 SKILL.md）
- 语言决策记录：否决 Rust/Go 重写（热点在原生后端，子进程模型下宿主语言性能无关），防重复讨论

### 发布工程

- vs_setup.py --check 新增 a11y 能力探测（at-spi2-core/python-gobject；缺失仅警告并给出发行版安装提示，不阻塞）
- README 追加 v0.3.0 章节（只增不改，保护未提交的用户改动）
- SKILL.md 重写为单端口用法（决策表/action 参考/示例全部对齐）

## 0.2.2 — 2026-08-25（自稳定 P0：preflight 层 + git 完整性）

### 调用前预检（self-stabilization core）

- index.ts 新增 preflight 层：L1 解释器可执行（--version）→ L2 核心依赖可导入（import vs_schema）
- 缓存策略：ok 进程内终身；fail 保留 30s 后允许重试（瞬态故障可自愈重试）
- 失败时返回结构化 `PREFLIGHT` 错误 + 可操作修复指引，不再让坏环境产生难懂的下游堆栈
- 自诊断豁免：inline (-c) 健康检查与 vs_setup.py（/vs setup|check）永不被预检阻断 —— 保证自诊断永不被自诊断阻断
- 预检探针级验证：好环境 L1/L2 双通过；坏解释器故障注入正确拒绝（ENOENT）

### /vs check 增强

- 新增 git 完整性段：当前 tag/commit（describe --tags --dirty）+ 未提交改动计数（排除 .bak）
- 商业封装门禁的一部分：工作树偏离已知 tag 时用户可见

### 配套自愈能力（已有，本轮验证确认）

- ocrserver 崩溃/重启后的残留 socket 自动清理；vs_ocr --daemon auto 自动重新拉起（实测通过）

## 0.2.1 — 2026-08-24（坐标原语记法 + a11y 传感器 + 证据锚定）

### 坐标原语记法（DeepSeek Thinking-with-Visual-Primitives 式）

- `vs_schema`：新增 `bbox_primitive()` / `point_primitive()` / `NOTATION_GUIDE`
- `vs_crosscheck` / `vs_audit`：全部 anomaly 挂 `primitive` 字段（如 `[bbox: 10, 10, 90, 50]`）；
  报告顶层加 `notation` 引导下游模型用坐标原语推理，禁用相对方位描述

### 新 L0 传感器：vs_a11y（Linux AT-SPI）

- 原生应用无障碍树 → schema v2 元素（角色/名称/screen_px bbox/文本），与 dom/pptx 同层的桌面真值源
- 走宿主系统 python3（需 at-spi2-core + python-gobject；conda env 无 gi）
- `--list` 列应用；`--app` 过滤；`--max-elements` 上限；`--with-text` 文本类角色抓内容
- 只读保证：仅调用 get_* 接口；gi 缺失时输出可操作安装指引而非堆栈
- index.ts：vs_struct 新增 a11y action（bin=python3，sandbox=false 需会话 DBus）

### 证据锚定（文本版 SoM）

- `vs_crosscheck` 异常 evidence 增上游元素 id：`dom_element_id` / `ocr_element_id` / `a_id` / `b_id`

### 文档

- SKILL.md：新增高分辨率 zoom-in 工作流（整图粗扫 → region 重检 + 原语记法引用）及 a11y 决策表行

### 架构优化（效率/质量/紧凑）

- `vs_schema.anomaly()`：crosscheck/audit 全部异常项收敛到唯一构造出口（rules 为独立 finding 契约不合并）
- 序列化单通道：9 个传感器脚本的输出统一走 `S.dump_json`（错误路径保持不变）
- 新增 `vs_vlm.py` 共享网关：semantic/critic 的 Ollama 调用合并为唯一出口（含图像预处理），消除重复 HTTP 拼装
- 新增 `ocrserver.py`：PaddleOCR 常驻服务（unix socket 行 JSON，omniserver 同模式）；vs_ocr --daemon auto|always|never 自动拉起/强制/禁用，实测冷载 7-40s → 调用 2.8s
- index.ts：超时分级常量化 `T.fast…T.modelXL`，替换全部 17 处魔数

### 验证

- py_compile 全部通过；test_fusion.py 8/8；crosscheck 端到端确认 notation/primitive/evidence id 落盘；a11y 实机冒烟通过

## 0.2.0 — 2026-08-03（分层无损 v2 + Phase 2.1–2.4）

### 架构 v2（vision-report/schema v2）

- 统一元素模型 `{id,type,bbox[x1,y1,x2,y2],text,conf,color,font,z,source[],coordsys}`；坐标系 css_px/device_px/image_px/pt + 转换
- 分层无损通道：L0 源码（DOM/PPTX）→ L1 确定性测量（颜色/OCR/diff/WCAG）→ L2 语义（opt-in）

### 新传感器/算子

- `vs_schema`：元素模型、CIELAB ΔE76、WCAG gamma 对比度、bbox 几何
- `vs_crosscheck`：颜色漂移（密集采样最小 ΔE）、OCR↔DOM 文本互验、元素重叠
- `vs_audit`：重叠/出界/对比度审计
- `vs_analyze`：配置驱动任务引擎（3 内置任务）+ `vs_omniparser`（OmniParser V2，CPU，图标级元素 + Florence-2 语义）
- `vs_rules`：确定性设计准则规则引擎（对比度/重叠/对齐漂移/间距一致/安全区；仅评估设计元素，OCR 不误报）
- `vs_cluster`：CLIP 离线相似图聚类（确定性）
- `vs_critic`：VLM-as-critic 闭环（裁剪可疑区 → 本地 qwen3-vl 复核 → 裁决并入 evidence；opt-in）

### 验证

- 基准套件 `bench/`：颜色 ΔE=0.0、OCR CER=0.0、diff 定位 recall=1.0、漂移检出 ΔE=76.3
- 验收抽样集 `bench/samples`（12 确定性样本）：规则臂一致率 100%、critic 臂 75%（分歧已实证解释）
- 全量回归：17+8+3+19+7+10 = 64 断言全绿
- 真实数据：Firefox 截图（OCR 60 项 / OmniParser 101 元素）、46 张壁纸分类与聚类

### 分发

- 引导安装器 `/vs setup`（conda env 检测/创建、依赖、自测；--check/--dry-run/--with-omniparser/--with-dom）
- docs/omniparser-setup.md（OmniParser 环境维护手册：版本钉死、HF 缓存坑、config 补丁）

## 0.1.0 — 2026-08-02（v1 传感器）

- 基础工具：vs_env_check、screen_capture（grim）、pix_analyze、ocr_boxes（RapidOCR PP-OCRv6）、dom_dump（Playwright Firefox）、pptx_dump、wallpaper_classify、semantic_tag（L2 opt-in）
- 自测 17/17；真实场景验证（Firefox 截图、文件截图、46 张壁纸）

### 待发布前补

- repository 字段（git 仓库 URL，需创建仓库后填入）
