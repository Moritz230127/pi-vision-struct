/**
 * pi-vision-struct — 结构化视觉扩展（分组 CLI 风格，4 工具）
 *
 * 为纯文本模型（DeepSeek）提供像素级精确的视觉通道：
 * 工具输出结构化 JSON（数字/坐标/hex），而非自然语言转述。
 *
 * 分层无损通道（vision-report/v2）：
 *   L0 源码    → vs_struct  (dom / pptx / omniparser)
 *   L1 测量    → vs_measure (capture / pixels / ocr / wallpaper / semantic / env)
 *   融合/准则  → vs_fuse    (analyze / crosscheck / audit / rules / critic)
 *   感知聚类   → vs_cluster (CLIP 相似图分组)
 *
 * 设计决策：15 个细粒度工具 → 4 个分组工具（action 枚举）。
 *   - 每轮工具 schema token 约降 40%（共享参数去重）
 *   - 模型路由选择面变小（准确率提升）
 *   - action 用 enum 校验，保留参数校验的大部分收益
 *   - 完整命令参考在技能 skills/vision-situation（按需加载，不进 AGENTS.md）
 *
 * 全部只读、本地、无网络外发。Python 脚本惰性启动（调用时才拉起进程）。
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { execFile } from "node:child_process";
import { fileURLToPath } from "node:url";

const DEFAULT_PYTHON = "/home/Arch/conda-envs/pi-vision/bin/python";
const OMNI_PYTHON = "/home/Arch/conda-envs/omniparser/bin/python";
const PKG_ROOT = fileURLToPath(new URL("..", import.meta.url));
const PY = (name: string) => `${PKG_ROOT}python/${name}`;

function runPython(
	pythonBin: string,
	args: string[],
	timeoutMs: number,
): Promise<string> {
	return new Promise((resolve) => {
		execFile(
			pythonBin,
			args,
			{ timeout: timeoutMs, maxBuffer: 16 * 1024 * 1024 },
			(err, stdout, stderr) => {
				if (err) {
					resolve(
						JSON.stringify({
							error: "python failed",
							code: err.code ?? null,
							detail: String(stderr || err.message || err).slice(0, 500),
						}),
					);
				} else {
					resolve(stdout.trim());
				}
			},
		);
	});
}

interface Act {
	script?: string; // python 脚本（缺省 = 内联）
	bin?: string; // python 解释器（缺省 = pi-vision env）
	timeout: number;
	build: (p: Record<string, string>) => string[]; // 参数 → CLI 参数
	inline?: string; // 内联 python 脚本
}

function num(p: Record<string, string>, k: string): string[] {
	const v = p[k];
	return v === undefined ? [] : [String(v)];
}

function flag(p: Record<string, string>, k: string, flagName: string): string[] {
	const v = p[k];
	return v === undefined ? [] : [flagName, String(v)];
}

function list(p: Record<string, string>, k: string, flagName: string): string[] {
	const v = p[k];
	if (v === undefined || v === "") return [];
	return [flagName, ...v.split(",").filter(Boolean)];
}

function on(p: Record<string, string>, k: string, flagName: string): string[] {
	return p[k] === "true" ? [flagName] : [];
}

const MEASURE: Record<string, Act> = {
	capture: {
		script: "vs_capture.py",
		timeout: 20000,
		build: (p) => [
			...flag(p, "out", "--out"),
			...flag(p, "region", "--region"),
		],
	},
	pixels: {
		script: "vs_pix.py",
		timeout: 30000,
		build: (p) => [
			...flag(p, "image", "--image"),
			...list(p, "regions", "--regions"),
			...flag(p, "compare", "--compare"),
			...flag(p, "colors", "--colors"),
			...list(p, "wcag", "--wcag"),
			...flag(p, "threshold", "--diff-threshold"),
		],
	},
	ocr: {
		script: "vs_ocr.py",
		timeout: 60000,
		build: (p) => [
			...flag(p, "image", "--image"),
			...flag(p, "region", "--region"),
			...flag(p, "upscale", "--upscale"),
			...flag(p, "max_items", "--max-items"),
			...flag(p, "min_conf", "--min-conf"),
		],
	},
	wallpaper: {
		script: "vs_wall.py",
		timeout: 300000,
		build: (p) => [
			...flag(p, "dir", "--dir"),
			...flag(p, "colors", "--colors"),
			...flag(p, "max_files", "--max-files"),
			...list(p, "ext", "--ext"),
			...on(p, "semantic", "--semantic"),
			...flag(p, "semantic_max", "--semantic-max"),
		],
	},
	semantic: {
		script: "vs_semantic.py",
		timeout: 360000,
		build: (p) => [
			...flag(p, "image", "--image"),
			...on(p, "enable", "--enable"),
			...flag(p, "prompt", "--prompt"),
			...flag(p, "max_tokens", "--max-tokens"),
		],
	},
	env: {
		timeout: 30000,
		inline: [
			"import json,sys",
			"mods=['onnxruntime','rapidocr','pptx','playwright','PIL']",
			"res={'python':sys.version.split()[0]}",
			"for m in mods:",
			"    try:",
			"        mod=__import__(m); res[m]=getattr(mod,'__version__','ok')",
			"    except Exception as e: res[m]='MISSING:'+str(e)[:60]",
			"print(json.dumps(res,ensure_ascii=False))",
		].join("\n"),
		build: () => [],
	},
};

const STRUCT: Record<string, Act> = {
	dom: {
		script: "vs_dom.py",
		timeout: 60000,
		build: (p) => [
			...flag(p, "url", "--url"),
			...flag(p, "max_elements", "--max-elements"),
			...flag(p, "screenshot", "--out-screenshot"),
		],
	},
	pptx: {
		script: "vs_pptx.py",
		timeout: 30000,
		build: (p) => [
			...flag(p, "file", "--file"),
			...flag(p, "max_shapes", "--max-shapes"),
			...flag(p, "slide", "--slide"),
		],
	},
	omniparser: {
		script: "vs_omniparser.py",
		bin: OMNI_PYTHON,
		timeout: 600000,
		build: (p) => [
			...flag(p, "image", "--image"),
			...flag(p, "max_items", "--max-items"),
			...on(p, "no_ocr", "--no-ocr"),
		],
	},
};

const FUSE: Record<string, Act> = {
	analyze: {
		script: "vs_analyze.py",
		timeout: 300000,
		build: (p) => [
			...flag(p, "task", "--task"),
			...flag(p, "input", "--input"),
			...flag(p, "url", "--url"),
			...flag(p, "dpr", "--dpr"),
		],
	},
	crosscheck: {
		script: "vs_crosscheck.py",
		timeout: 60000,
		build: (p) => [
			...flag(p, "image", "--image"),
			...flag(p, "dom", "--dom"),
			...flag(p, "ocr", "--ocr"),
			...flag(p, "dpr", "--dpr"),
			...flag(p, "color_threshold", "--color-threshold"),
		],
	},
	audit: {
		script: "vs_audit.py",
		timeout: 30000,
		build: (p) => [
			...flag(p, "report", "--report"),
			...flag(p, "canvas", "--canvas"),
			...flag(p, "overlap_threshold", "--overlap-threshold"),
		],
	},
	rules: {
		script: "vs_rules.py",
		timeout: 60000,
		build: (p) => [
			...flag(p, "report", "--report"),
			...flag(p, "canvas", "--canvas"),
			...flag(p, "align_tol", "--align-tol"),
			...flag(p, "margin", "--margin"),
		],
	},
	critic: {
		script: "vs_critic.py",
		timeout: 900000,
		build: (p) => [
			...flag(p, "report", "--report"),
			...flag(p, "image", "--image"),
			...on(p, "enable", "--enable"),
			...flag(p, "max_critic", "--max-critic"),
			...flag(p, "margin", "--margin"),
		],
	},
};

async function dispatch(
	acts: Record<string, Act>,
	action: string,
	params: Record<string, unknown>,
): Promise<{
	content: { type: "text"; text: string }[];
	details: Record<string, never>;
}> {
	const act = acts[action];
	if (!act) {
		return Promise.resolve({
			content: [
				{
					type: "text",
					text: JSON.stringify({
						error: `未知 action: ${action}`,
						detail: `可用: ${Object.keys(acts).join(", ")}`,
					}),
				},
			],
			details: {},
		});
	}
	const p = Object.fromEntries(
		Object.entries(params).map(([k, v]) => [k, String(v)]),
	);
	const out = act.inline
		? await runPython(act.bin ?? DEFAULT_PYTHON, ["-c", act.inline], act.timeout)
		: await runPython(
				act.bin ?? DEFAULT_PYTHON,
				[PY(act.script ?? ""), ...act.build(p)],
				act.timeout,
			);
	return { content: [{ type: "text", text: out }], details: {} };
}

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
				ctx.ui.notify(
					`pi-vision-struct 环境自检\n${line}\n用法: /vs setup 安装`,
					"info",
				);
			} catch {
				ctx.ui.notify(`pi-vision-struct\n${out.slice(0, 500)}`, "error");
			}
		},
	});

	pi.registerTool({
		name: "vs_measure",
		label: "Vision Measure (capture/pixels/ocr/wallpaper/semantic/env)",
		description:
			"确定性测量与感知传感器（本地，无网络）。action 枚举：capture=Wayland 截屏(region 可选)；pixels=主色直方图/区域取色/双图 diff 异常定位/WCAG 对比度(传 image, 可选 regions/compare/colors/wcag/threshold)；ocr=RapidOCR 文本+精确 bbox(传 image, 可选 region/upscale/max_items/min_conf)；wallpaper=壁纸批量程序化分类(传 dir, 可选 colors/max_files/ext/semantic)；semantic=本地 qwen3-vl L2 语义标签(传 image, opt-in enable)；env=conda 环境自检。输出 schema v2 JSON。完整命令参考见技能 vision-situation。",
		parameters: Type.Object({
			action: Type.Union([
				Type.Literal("capture"),
				Type.Literal("pixels"),
				Type.Literal("ocr"),
				Type.Literal("wallpaper"),
				Type.Literal("semantic"),
				Type.Literal("env"),
			]),
			image: Type.Optional(Type.String({ description: "图片路径" })),
			out: Type.Optional(Type.String({ description: "截屏输出 PNG 路径（capture）" })),
			region: Type.Optional(Type.String({ description: "区域 x1,y1,x2,y2（capture/ocr）" })),
			compare: Type.Optional(Type.String({ description: "diff 对比图路径（pixels）" })),
			colors: Type.Optional(Type.Number({ description: "主色数量（pixels/wallpaper）" })),
			wcag: Type.Optional(Type.String({ description: "对比度对 前景hex,背景hex（pixels，逗号分隔多对）" })),
			threshold: Type.Optional(Type.Number({ description: "diff 阈值（pixels，默认 30）" })),
			dir: Type.Optional(Type.String({ description: "壁纸目录（wallpaper）" })),
			max_files: Type.Optional(Type.Number({ description: "最多处理（wallpaper）" })),
			ext: Type.Optional(Type.String({ description: "扩展名列表（wallpaper，逗号分隔）" })),
			semantic: Type.Optional(Type.Boolean({ description: "opt-in 语义标签（wallpaper）" })),
			semantic_max: Type.Optional(Type.Number({ description: "语义标注上限（wallpaper）" })),
			upscale: Type.Optional(Type.Number({ description: "OCR 放大倍数（默认 2）" })),
			max_items: Type.Optional(Type.Number({ description: "OCR 最多条数（默认 100）" })),
			min_conf: Type.Optional(Type.Number({ description: "OCR 最低置信度（默认 0.5）" })),
			enable: Type.Optional(Type.Boolean({ description: "opt-in 开启 L2 语义（semantic）" })),
			prompt: Type.Optional(Type.String({ description: "语义提示词（semantic，可选）" })),
			max_tokens: Type.Optional(Type.Number({ description: "语义最大 token（semantic）" })),
		}),
		async execute(_id, params) {
			return dispatch(MEASURE, String(params.action), params);
		},
	});

	pi.registerTool({
		name: "vs_struct",label: "Structure (dom/pptx/omniparser)",
		description:
			"L0 源码结构化 + DL 图标感知。action 枚举：dom=Playwright Firefox 加载 URL 导出 DOM+computed style(传 url, 可选 max_elements/screenshot)；pptx=python-pptx 导出形状/填充/字体/坐标 pt(传 file, 可选 max_shapes/slide)；omniparser=OmniParser V2 任意截图图标级元素+语义描述(传 image, 可选 max_items/no_ocr；CPU 首载 10-20s)。输出 schema v2 JSON。完整命令参考见技能 vision-situation。",
		parameters: Type.Object({
			action: Type.Union([
				Type.Literal("dom"),
				Type.Literal("pptx"),
				Type.Literal("omniparser"),
			]),
			url: Type.Optional(Type.String({ description: "要分析的 URL（dom）" })),
			max_elements: Type.Optional(Type.Number({ description: "最多元素（dom，默认 60）" })),
			screenshot: Type.Optional(Type.String({ description: "DOM 会话截图输出路径（dom）" })),
			file: Type.Optional(Type.String({ description: "pptx 文件路径（pptx）" })),
			max_shapes: Type.Optional(Type.Number({ description: "最多形状（pptx，默认 200）" })),
			slide: Type.Optional(Type.Number({ description: "只导出第 N 张（pptx）" })),
			image: Type.Optional(Type.String({ description: "图片路径（omniparser）" })),
			max_items: Type.Optional(Type.Number({ description: "最多元素（omniparser，默认 60）" })),
			no_ocr: Type.Optional(Type.Boolean({ description: "跳过 OCR 仅图标（omniparser）" })),
		}),
		async execute(_id, params) {
			return dispatch(STRUCT, String(params.action), params);
		},
	});

	pi.registerTool({
		name: "vs_fuse",label: "Fusion & Rules (analyze/crosscheck/audit/rules/critic)",
		description:
			"确定性融合/审计/准则/复核（本地）。action 枚举：analyze=配置驱动任务引擎(传 task，可选 input/url/dpr；内置 diagnose-screenshot/audit-pptx/classify-images)；crosscheck=DOM↔OCR↔像素三方互验(传 image，可选 dom/ocr 报告 JSON/dpr/color_threshold)；audit=重叠/出界/对比度审计(传 report JSON，可选 canvas/overlap_threshold)；rules=设计准则规则引擎(传 report JSON，可选 canvas/align_tol/margin)；critic=VLM 复核裁剪区(传 report+image，opt-in enable，可选 max_critic/margin；全局属性缺陷出界/安全区在裁剪视图会误判)。输出 schema v2。完整命令参考见技能 vision-situation。",
		parameters: Type.Object({
			action: Type.Union([
				Type.Literal("analyze"),
				Type.Literal("crosscheck"),
				Type.Literal("audit"),
				Type.Literal("rules"),
				Type.Literal("critic"),
			]),
			task: Type.Optional(Type.String({ description: "任务名（analyze）" })),
			input: Type.Optional(Type.String({ description: "输入 图片/pptx/目录（analyze）" })),
			url: Type.Optional(Type.String({ description: "URL（analyze 的 DOM 源）" })),
			dpr: Type.Optional(Type.Number({ description: "DPR（默认 1.0）" })),
			image: Type.Optional(Type.String({ description: "图片路径（crosscheck/critic）" })),
			dom: Type.Optional(Type.String({ description: "dom_dump 输出 JSON（crosscheck）" })),
			ocr: Type.Optional(Type.String({ description: "ocr 输出 JSON（crosscheck）" })),
			color_threshold: Type.Optional(Type.Number({ description: "ΔE 阈值（crosscheck，默认 5）" })),
			report: Type.Optional(Type.String({ description: "报告 JSON 路径（audit/rules/critic）" })),
			canvas: Type.Optional(Type.String({ description: "画布 WxH（audit/rules）" })),
			overlap_threshold: Type.Optional(Type.Number({ description: "IoU 阈值（audit，默认 0.05）" })),
			align_tol: Type.Optional(Type.Number({ description: "对齐容差 px（rules，默认 4）" })),
			margin: Type.Optional(Type.Number({ description: "安全区/裁剪边距 px（rules/critic）" })),
			enable: Type.Optional(Type.Boolean({ description: "opt-in 开启 VLM 复核（critic）" })),
			max_critic: Type.Optional(Type.Number({ description: "裁剪上限（critic，默认 8）" })),
		}),
		async execute(_id, params) {
			return dispatch(FUSE, String(params.action), params);
		},
	});

	pi.registerTool({
		name: "vs_cluster",label: "CLIP Similarity Clustering",
		description:
			"CLIP (ViT-B-32, CPU, offline) 相似图聚类：感知相似度矩阵 + 阈值贪心分组（确定性，同输入同输出）。传 dir 或 files(逗号分隔)，可选 threshold(默认 0.75，越大分组越细)/max_files(默认 200)。输出 clusters[]（代表图+成员相似度）+ top_pairs[]。首次运行下载模型 ~350MB（需代理，之后离线）。运行于 omniparser env。",
		parameters: Type.Object({
			dir: Type.Optional(Type.String({ description: "图片目录" })),
			files: Type.Optional(Type.String({ description: "逗号分隔文件列表（与 dir 二选一）" })),
			threshold: Type.Optional(Type.Number({ description: "余弦相似度阈值（默认 0.75）" })),
			max_files: Type.Optional(Type.Number({ description: "最多处理（默认 200）" })),
		}),
		async execute(_id, params) {
			const p = Object.fromEntries(
				Object.entries(params).map(([k, v]) => [k, String(v)]),
			);
			const args = [PY("vs_cluster.py")];
			if (p.dir) args.push("--dir", p.dir);
			if (p.files) args.push("--files", p.files);
			if (p.threshold) args.push("--threshold", p.threshold);
			if (p.max_files) args.push("--max-files", p.max_files);
			const out = await runPython(OMNI_PYTHON, args, 600000);
			return { content: [{ type: "text", text: out }], details: {} };
		},
	});
}
