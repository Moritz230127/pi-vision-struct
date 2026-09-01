# Claude Code 接入（MCP）

`pi-vision-struct` 同时支持两个宿主：

- **pi-coding-agent** —— 通过 `extensions/index.ts`（ExtensionAPI），由 `package.json` 的 `pi.extensions` 自动加载
- **Claude Code** —— 通过 **MCP**（Model Context Protocol），本文件说明

两个宿主共用同一份框架无关核心 `extensions/pi-vision-core.ts`，**任何 Python 传感器一行不改**。

## 原理

核心 `dispatch()` 返回 `{content:[{type:"text",text}]}`，本就是 MCP 的原生返回格式。
`extensions/server.mcp.ts` 把它包成一个 MCP server（stdio 传输），暴露单端口工具 `vs`，其 `action` 枚举与参数 schema 与 pi 完全一致（由核心 `vsParamType` 单一来源派生）。

## 接入步骤

### 1. 安装依赖并构建

```bash
cd pi-vision-struct
npm install
npm run build        # → dist/extensions/{pi-vision-core,index,server.mcp}.js
```

> 仅 MCP 侧需要 `@modelcontextprotocol/sdk`；pi 侧不依赖它。

### 2. 注册 MCP server 到 Claude Code

```bash
# 方式 A（推荐，全局/项目级持久化）
claude mcp add pi-vision-struct -- node /abs/path/to/pi-vision-struct/dist/extensions/server.mcp.js

# 方式 B（开发期免编译，用 tsx 直接跑 TS 源码）
claude mcp add pi-vision-struct -- npx tsx /abs/path/to/pi-vision-struct/extensions/server.mcp.ts
```

路径务必用**绝对路径**（stdio server 依赖进程 cwd 解析 `python/` 目录，核心用 `import.meta.url` 自行推导，无需硬编码）。

### 3. 验证

```bash
claude mcp list              # 确认 pi-vision-struct 已连接（state: connected）
claude mcp get pi-vision-struct   # tools/list 应返回 vs，action 含 v3.0.0 动作集（含 saliency/segment/fusion/zoom/probe 等）
```

之后在 Claude Code 内直接调用，例如：

```
vs({ action: "env" })
vs({ action: "pixels", image: "/tmp/shot.png", colors: 5 })
vs({ action: "ocr", image: "/tmp/shot.png", max_items: 20 })
vs({ action: "check" })       # 环境自检（与 pi 的 /vs check 等价）
```

错误 action（如 `vs({action:"nope"})`）会以 `isError:true` 返回结构化错误，Claude Code 可据此纠正。

## Python 环境

MCP server 用与 pi 完全相同的 python 解析顺序：

1. 环境变量 `PI_VISION_PYTHON`（pi-vision env）/ `PI_VISION_OMNI_PYTHON`（omniparser env）
2. 配置文件 `~/.config/pi-vision-struct.json`（键 `pi_vision_python` / `omniparser_python`）
3. 候选路径：`~/conda-envs`、`~/miniforge3`、`~/miniconda3`、`~/mambaforge`
4. 回退 `python3`

未安装依赖时，调任意动作会触发 `PREFLIGHT` 错误并提示 `/vs setup`——
在 Claude Code 下等价于执行 `vs({ action: "setup", setup_args: "--with-omniparser" })`。

## 沙箱与豁免（同 pi）

- Linux + 存在 `bwrap` + 未设 `VS_NO_SANDBOX=1` → 默认在 bwrap 沙箱（`--unshare-net` 零网络 + 只读根）运行
- 豁免（不进沙箱）：`dom`（加载用户 URL）、`a11y`（会话 DBus）、`semantic`/`chart_data`（本机 Ollama 127.0.0.1）、`check`/`setup`（需网络安装）
- 其他平台自动降级为无沙箱

## 注意

- Claude Code 侧 `dom`/`a11y`/`semantic`/`setup` 需要网络或桌面会话 DBus，与 pi 同约束
- 全部输出为 schema v3 JSON（数字/坐标/hex），可直接数值推理，零主观描述
