# Changelog

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
