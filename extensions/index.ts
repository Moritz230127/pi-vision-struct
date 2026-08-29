/**
 * pi-vision-struct — pi-coding-agent 适配器壳（单端口 vs 工具 + /vs 命令）
 *
 * 框架无关核心在 `./pi-vision-core`；本文件只负责把核心接到
 * `@earendil-works/pi-coding-agent` 的 ExtensionAPI 上（双宿主之一）。
 *
 * Claude Code 宿主见同目录 `server.mcp.ts`（MCP），共用同一核心。
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import {
	dispatch,
	buildRoute,
	vsParamType,
	VS_DESCRIPTION,
	gitIntegrity,
	runPython,
	PKG_ROOT,
	DEFAULT_PYTHON,
} from "./pi-vision-core.js";

export default function visionStructExtension(pi: ExtensionAPI) {
	const pythonBin = process.env.PI_VISION_PYTHON || DEFAULT_PYTHON;

	pi.registerCommand("vs", {
		description:
			"pi-vision-struct: 状态自检 / 引导安装。用法: /vs 或 /vs check（只读自检）、/vs setup（核心安装+自测）、/vs setup --with-omniparser（含 OmniParser env）、/vs setup --with-dom（含 playwright firefox）",
		handler: async (args, ctx) => {
			const rest = (args ?? "").trim();
			if (rest.startsWith("setup")) {
				const setupArgs =
					rest === "setup"
						? ["--dry-run"]
						: rest.replace(/^setup\s*/, "").split(/\s+/).filter(Boolean);
				ctx.ui.notify(
					`/vs setup 开始（参数: ${setupArgs.join(" ") || "默认核心"}，安装需数分钟，期间请勿关闭）…`,
					"info",
				);
				const out = await runPython(
					pythonBin,
					["-u", `${PKG_ROOT}python/setup/vs_setup.py`, ...setupArgs],
					1200000,
				);
				try {
					const j = JSON.parse(out) as {
						ok?: boolean;
						steps?: { name: string; status: string }[];
						next?: string[];
					};
					const fails = (j.steps ?? [])
						.flatMap((s) => (s.status === "ok" ? [] : [`✗ ${s.name}`]))
						.join("\n");
					const next = j.next ? `\n${j.next.join("\n")}` : "";
					const line = j.ok
						? `✓ 全部就绪${next}`
						: `✗ 失败步骤:\n${fails || "未知"}\n修复后重跑 /vs setup`;
					ctx.ui.notify(
						`pi-vision-struct /vs setup\n${line}`,
						j.ok ? "info" : "error",
					);
				} catch {
					ctx.ui.notify(
						`pi-vision-struct /vs setup\n${out.slice(0, 500)}`,
						"error",
					);
				}
				return;
			}
			const out = await runPython(
				pythonBin,
				["-u", `${PKG_ROOT}python/setup/vs_setup.py`, "--check"],
				30000,
			);
			try {
				const j = JSON.parse(out) as {
					envs?: Record<string, { exists: boolean; complete: boolean }>;
				};
				const es = j.envs ?? {};
				const line = Object.entries(es)
					.map(([k, v]) => {
						let state: string;
						if (!v.exists) {
							state = "未创建";
						} else if (v.complete) {
							state = "完整";
						} else {
							state = "缺失依赖";
						}
						return `${k}: ${state}`;
					})
					.join(" | ");
				const gitLine = await gitIntegrity();
				ctx.ui.notify(
					`pi-vision-struct 环境自检\n${line}\n${gitLine}\n用法: /vs setup 安装`,
					"info",
				);
			} catch {
				ctx.ui.notify(`pi-vision-struct\n${out.slice(0, 500)}`, "error");
			}
		},
	});

	// ---- 单端口：全部视觉能力收敛为一个 `vs` 工具，最小化上下文注入 ----
	const ROUTE = buildRoute();

	pi.registerTool({
		name: "vs",
		label: "Vision Suite 视觉套件（单端口 25 动作）",
		description: VS_DESCRIPTION,
		parameters: vsParamType, // ← 共用核心 schema（单一来源）
		async execute(_id, params) {
			return dispatch(ROUTE, String(params.action), params);
		},
	});
}
