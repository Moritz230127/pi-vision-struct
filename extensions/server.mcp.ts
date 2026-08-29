/**
 * pi-vision-struct — Claude Code 适配器壳（MCP stdio server）
 *
 * 框架无关核心在 `./pi-vision-core`；本文件把核心接到 MCP（Model Context Protocol）
 * 上，使 Claude Code 通过 `claude mcp add` 即可使用全部动作（双宿主之一）。
 *
 * 设计选择：用低层 `Server` + 手写 ListTools/CallTool 处理器，直接以 JSON Schema
 * 暴露工具（MCP 线缆协议的 tool.inputSchema 本就是 JSON Schema），避免引入 Zod 依赖，
 * 并复用核心的 `toMcpJsonSchema(vsParamType)` 单一来源。
 *
 * pi-coding-agent 宿主见同目录 `index.ts`，共用同一核心。
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
	CallToolRequestSchema,
	ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import {
	dispatch,
	buildRoute,
	vsParamType,
	toMcpJsonSchema,
	VS_DESCRIPTION,
	VERSION,
} from "./pi-vision-core.js";

const ROUTE = buildRoute();
const inputSchema = toMcpJsonSchema(vsParamType);

const server = new Server(
	{ name: "pi-vision-struct", version: VERSION },
	{ capabilities: { tools: {} } },
);

// ---- tools/list：声明单端口 vs 工具 ----
server.setRequestHandler(ListToolsRequestSchema, async () => ({
	tools: [
		{
			name: "vs",
			title: "Vision Suite 视觉套件（单端口 25 动作）",
			description: VS_DESCRIPTION,
			inputSchema,
		},
	],
}));

// ---- tools/call：分发到核心 dispatch ----
server.setRequestHandler(CallToolRequestSchema, async (request) => {
	const params = (request.params.arguments ?? {}) as Record<string, unknown>;
	const r = await dispatch(ROUTE, String(params.action), params);
	const text = r.content[0]?.text ?? "";
	const isError = text.trimStart().startsWith('{"error"');
	return {
		content: r.content,
		isError,
	};
});

const transport = new StdioServerTransport();
await server.connect(transport);

// 进程退出清理（stdin 关闭等）
for (const sig of ["SIGINT", "SIGTERM"] as const) {
	process.on(sig, async () => {
		await server.close().catch(() => {});
		process.exit(0);
	});
}
