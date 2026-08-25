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
import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";

const PKG_ROOT = fileURLToPath(new URL("..", import.meta.url));
const PY = (name: string) => `${PKG_ROOT}python/${name}`;

// ---- 跨平台 python 解析：env > 配置文件 > 平台候选路径 > PATH ----
const HOME = homedir();
const isWin = process.platform === "win32";
const BIN = isWin ? "Scripts" : "bin";
const EXE = isWin ? ".exe" : "";
const CONFIG_DIR = isWin
	? process.env.APPDATA ?? `${HOME}/AppData/Roaming`
	: process.env.XDG_CONFIG_HOME ?? `${HOME}/.config`;
const CONFIG_FILE = `${CONFIG_DIR}/pi-vision-struct.json`;

function readConfig(): Record<string, string> {
	try {
		if (!existsSync(CONFIG_FILE)) return {};
		const cfg = JSON.parse(readFileSync(CONFIG_FILE, "utf-8"));
		return typeof cfg === "object" && cfg ? cfg : {};
	} catch {
		return {};
	}
}

function candidatePython(envName: string): string[] {
	// 安装器 vs_setup.py 会写配置文件；此处为无配置时的候选路径
	const cands: string[] = [
		`${HOME}/conda-envs/${envName}/${BIN}/python${EXE}`,
		`${HOME}/miniforge3/envs/${envName}/${BIN}/python${EXE}`,
		`${HOME}/miniconda3/envs/${envName}/${BIN}/python${EXE}`,
		`${HOME}/mambaforge/envs/${envName}/${BIN}/python${EXE}`,
	];
	return cands;
}

function resolvePython(envName: string, envVar: string): string {
	if (process.env[envVar]) return process.env[envVar]!;
	const cfg = readConfig();
	const key = envName === "pi-vision" ? "pi_vision_python" : "omniparser_python";
	if (cfg[key] && existsSync(cfg[key]!)) return cfg[key]!;
	for (const c of candidatePython(envName)) {
		if (existsSync(c)) return c;
	}
	return isWin ? "python" : "python3";
}

const DEFAULT_PYTHON = resolvePython("pi-vision", "PI_VISION_PYTHON");
const OMNI_PYTHON = resolvePython("omniparser", "PI_VISION_OMNI_PYTHON");
// Linux 且 bwrap 存在且未禁用 → 沙箱；其他平台自动降级为无沙箱
const SANDBOX_ENABLED =
	process.platform === "linux" &&
	process.env.VS_NO_SANDBOX !== "1" &&
	existsSync(PY("setup/vs_bwrap.sh"));
const BWRAP = PY("setup/vs_bwrap.sh"); // 严格本地化沙箱（零网络 + 只读根）

// 失败日志（本地、有上限）：工具报错时记录 {ts, tool, args, error}。
// 用途：真实失败分布 → 数据驱动的下一轮优化决策。目录 ~/.cache/vs-failures/
// （fs/os 已在顶部导入，这里只补 append/mkdir/rename/stat）
import {
	appendFileSync,
	mkdirSync,
	renameSync,
	statSync,
} from "node:fs";

const FAIL_DIR = `${homedir()}/.cache/vs-failures`;
const FAIL_FILE = `${FAIL_DIR}/failures.jsonl`;
const FAIL_MAX_BYTES = 512 * 1024; // 0.5MB 上限，超限轮转（保留最近 150 条）

function logFailure(tool: string, args: string[], detail: string): void {
	try {
		mkdirSync(FAIL_DIR, { recursive: true });
		const entry =
			JSON.stringify({
				ts: Date.now(),
				tool,
				args: args.slice(0, 12),
				detail: detail.slice(0, 300),
			}) + "\n";
		appendFileSync(FAIL_FILE, entry);
		try {
			if (statSync(FAIL_FILE).size > FAIL_MAX_BYTES) {
				const keep = readFileSync(FAIL_FILE, "utf-8")
					.split("\n")
					.slice(-150)
					.join("\n");
				appendFileSync(`${FAIL_FILE}.tmp`, keep);
				renameSync(`${FAIL_FILE}.tmp`, FAIL_FILE);
			}
		} catch {
			// 轮转失败不影响主路径
		}
	} catch {
		// 日志失败不影响工具本身
	}
}

// ---- 自稳定 P0：调用前预检（preflight）----
// L1 解释器可执行 → L2 核心依赖（vs_schema）可导入。
// 缓存策略：ok 进程内终身；fail 保留 30s 后允许重试（瞬态故障可自愈重试）。
// 豁免：inline/-c 健康检查与 /vs setup|check 诊断路径 —— 保证自诊断永不被自诊断阻断。
const PREFLIGHT_FAIL_TTL = 30_000;
const preflightCache = new Map<
	string,
	{ ok: boolean; detail?: string; at: number }
>();

function execFileNative(
	bin: string,
	args: string[],
	timeoutMs: number,
	cwd?: string,
): Promise<void> {
	return new Promise((resolve, reject) => {
		execFile(bin, args, { timeout: timeoutMs, cwd }, (err) =>
			err ? reject(err) : resolve(),
		);
	});
}

async function preflight(
	bin: string,
	pythonDir: string,
): Promise<{ ok: boolean; detail?: string }> {
	const cached = preflightCache.get(bin);
	if (cached) {
		if (cached.ok) return { ok: true };
		if (Date.now() - cached.at < PREFLIGHT_FAIL_TTL)
			return { ok: false, detail: cached.detail };
	}
	try {
		await execFileNative(bin, ["--version"], 10_000);
	} catch (e) {
		const r = {
			ok: false as const,
			detail: `解释器不可用: ${bin} (${String((e as Error).message).slice(0, 120)})。修复: /vs setup 或检查 PI_VISION_PYTHON/PATH`,
		};
		preflightCache.set(bin, { ...r, at: Date.now() });
		return r;
	}
	try {
		await execFileNative(bin, ["-c", "import vs_schema"], 15_000, pythonDir);
	} catch {
		const r = {
			ok: false as const,
			detail: `核心依赖导入失败 (import vs_schema @ ${pythonDir})。修复: /vs setup`,
		};
		preflightCache.set(bin, { ...r, at: Date.now() });
		return r;
	}
	preflightCache.set(bin, { ok: true, at: Date.now() });
	return { ok: true };
}

function runPython(
	pythonBin: string,
	args: string[],
	timeoutMs: number,
	sandbox = true,
): Promise<string> {
	return (async () => {
		// inline (-c) 为健康/环境自检路径；vs_setup.py 为诊断/修复自身，均豁免预检 ——
		// 保证自诊断永不被自诊断阻断
		const isSelfDiag = args.some((a) => String(a).includes("vs_setup.py"));
		if (args[0] !== "-c" && !isSelfDiag) {
			const pf = await preflight(pythonBin, `${PKG_ROOT}python`);
			if (!pf.ok) {
				logFailure("preflight", args.slice(0, 4), pf.detail ?? "failed");
				return JSON.stringify({
					error: "preflight failed",
					code: "PREFLIGHT",
					detail: pf.detail,
				});
			}
		}
		return await new Promise<string>((resolve) => {
		// 沙箱: bwrap 内核隔离（--unshare-net 零网络、--ro-bind / 只读根）。
		// 豁免（sandbox=false）: dom（本职加载用户 URL）/ critic / semantic
		// （依赖宿主 Ollama 的 127.0.0.1，硬编码无用户可控目标，已审计）。
		// 非 Linux / 无 bwrap 平台自动无沙箱（配置或 VS_NO_SANDBOX 可禁用）。
		const useSandbox = SANDBOX_ENABLED && sandbox;
		const cmd = useSandbox ? BWRAP : pythonBin;
		const cmdArgs = useSandbox ? [pythonBin, ...args] : args;
		const tool = args[0] ? args[0].split("/").pop() ?? args[0] : pythonBin;
		execFile(
			cmd,
			cmdArgs,
			{ timeout: timeoutMs, maxBuffer: 16 * 1024 * 1024 },
			(err, stdout, stderr) => {
				if (err) {
					logFailure(tool, args, String(stderr || err.message || err));
					resolve(
						JSON.stringify({
							error: "python failed",
							code: err.code ?? null,
							detail: String(stderr || err.message || err).slice(0, 500),
						}),
					);
				} else {
					const out = stdout.trim();
					if (out.startsWith("{") && out.includes('"error"')) {
						logFailure(tool, args, out.slice(0, 300));
					}
					resolve(out);
					}
				},
			);
			});
		})();
}

interface Act {
	script?: string; // python 脚本（缺省 = 内联）
	bin?: string; // python 解释器（缺省 = pi-vision env）
	timeout: number;
	build: (p: Record<string, string>) => string[]; // 参数 → CLI 参数
	inline?: string; // 内联 python 脚本
	sandbox?: boolean; // 默认 true（bwrap 隔离）; false = 豁免（见上）
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

// 超时分级常量（ms）：按操作类型命名，禁止散落魔数
const T = {
	fast: 20_000, // 截屏 / 环境自检
	short: 30_000, // pptx / a11y 轻结构
	normal: 60_000, // dom / ocr / 融合单步
	heavy: 120_000, // pdf
	doc: 300_000, // 整页管线 / 版式模型
	vlm: 360_000, // critic 多区域 VLM 复核
	model: 600_000, // omniparser 冷启动
	modelXL: 900_000, // CLIP 批量聚类
};

const MEASURE: Record<string, Act> = {
	capture: {
		script: "vs_capture.py",
		timeout: T.fast,
		build: (p) => [
			...flag(p, "out", "--out"),
			...flag(p, "region", "--region"),
		],
	},
	pixels: {
		script: "vs_pix.py",
		timeout: T.short,
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
		timeout: T.normal,
		build: (p) => [
			...flag(p, "image", "--image"),
			...flag(p, "region", "--region"),
			...flag(p, "upscale", "--upscale"),
			...flag(p, "max_items", "--max-items"),
			...flag(p, "min_conf", "--min-conf"),
			...flag(p, "backend", "--backend"),
			...flag(p, "preprocess", "--preprocess"),
			...flag(p, "daemon", "--daemon"),
		],
	},
	wallpaper: {
		script: "vs_wall.py",
		timeout: T.doc,
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
		sandbox: false, // 依赖宿主 Ollama 127.0.0.1（硬编码目标）
		timeout: T.vlm,
		build: (p) => [
			...flag(p, "image", "--image"),
			...on(p, "enable", "--enable"),
			...flag(p, "prompt", "--prompt"),
			...flag(p, "max_tokens", "--max-tokens"),
		],
	},
	env: {
		timeout: T.short,
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
		sandbox: false, // 本职加载用户指定 URL，需要网络
		timeout: T.normal,
		build: (p) => [
			...flag(p, "url", "--url"),
			...flag(p, "max_elements", "--max-elements"),
			...flag(p, "screenshot", "--out-screenshot"),
		],
	},
	pptx: {
		script: "vs_pptx.py",
		timeout: T.short,
		build: (p) => [
			...flag(p, "file", "--file"),
			...flag(p, "max_shapes", "--max-shapes"),
			...flag(p, "slide", "--slide"),
		],
	},
	omniparser: {
		script: "vs_omniparser.py",
		bin: OMNI_PYTHON,
		timeout: T.model,
		build: (p) => [
			...flag(p, "image", "--image"),
			...flag(p, "max_items", "--max-items"),
			...on(p, "no_ocr", "--no-ocr"),
		],
	},
	layout: {
		script: "vs_layout.py",
		timeout: T.doc,
		build: (p) => [
			...flag(p, "image", "--image"),
			...flag(p, "max_items", "--max-items"),
			...flag(p, "min_conf", "--min-conf"),
		],
	},
	pdf: {
		script: "vs_pdf.py",
		timeout: T.heavy,
		build: (p) => [
			...flag(p, "file", "--file"),
			...flag(p, "pages", "--pages"),
			...flag(p, "max_items", "--max-items"),
			...flag(p, "render_dir", "--render-dir"),
		],
	},
	a11y: {
		script: "vs_a11y.py",
		bin: "python3", // 需要宿主 python-gobject（AT-SPI）；conda env 无 gi
		sandbox: false, // 需访问会话 DBus 无障碍总线；脚本本身只读
		timeout: T.fast,
		build: (p) => [
			...on(p, "list", "--list"),
			...flag(p, "app", "--app"),
			...flag(p, "max_elements", "--max-elements"),
			...on(p, "with_text", "--with-text"),
		],
	},
};

const FUSE: Record<string, Act> = {
	analyze: {
		script: "vs_analyze.py",
		timeout: T.doc,
		build: (p) => [
			...flag(p, "task", "--task"),
			...flag(p, "input", "--input"),
			...flag(p, "url", "--url"),
			...flag(p, "dpr", "--dpr"),
			...flag(p, "compare", "--compare"),
		],
	},
	crosscheck: {
		script: "vs_crosscheck.py",
		timeout: T.normal,
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
		timeout: T.short,
		build: (p) => [
			...flag(p, "report", "--report"),
			...flag(p, "canvas", "--canvas"),
			...flag(p, "overlap_threshold", "--overlap-threshold"),
		],
	},
	rules: {
		script: "vs_rules.py",
		timeout: T.normal,
		build: (p) => [
			...flag(p, "report", "--report"),
			...flag(p, "canvas", "--canvas"),
			...flag(p, "align_tol", "--align-tol"),
			...flag(p, "margin", "--margin"),
		],
	},
	critic: {
		script: "vs_critic.py",
		sandbox: false, // 依赖宿主 Ollama 127.0.0.1（硬编码目标）
		timeout: T.modelXL,
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
		? await runPython(act.bin ?? DEFAULT_PYTHON, ["-c", act.inline], act.timeout,
						   act.sandbox ?? true)
		: await runPython(
				act.bin ?? DEFAULT_PYTHON,
				[PY(act.script ?? ""), ...act.build(p)],
				act.timeout,
				act.sandbox ?? true,
			);
	return { content: [{ type: "text", text: out }], details: {} };
}

export default function visionStructExtension(pi: ExtensionAPI) {
	const pythonBin = process.env.PI_VISION_PYTHON || DEFAULT_PYTHON;

	async function gitIntegrity(): Promise<string> {
	try {
		const desc = await new Promise<string>((res) =>
			execFile(
				"git",
				["--no-pager", "describe", "--tags", "--always", "--dirty"],
				{ timeout: 5000, cwd: PKG_ROOT },
				(e, o) => (e ? res("?") : res(String(o).trim())),
			),
		);
		const dirty = await new Promise<number>((res) =>
			execFile(
				"git",
				["status", "--porcelain"],
				{ timeout: 5000, cwd: PKG_ROOT },
				(e, o) => {
					if (e) return res(-1);
					res(
						String(o)
							.split("\n")
							.filter((l) => l.trim() && !l.includes(".bak")).length,
						);
				},
			),
		);
		const state =
			dirty < 0 ? "状态未知" : dirty === 0 ? "工作树干净" : `${dirty} 处未提交改动`;
		return `git: ${desc} (${state})`;
	} catch {
		return "git: 不可用";
	}
}

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

	pi.registerTool({
		name: "vs_measure",
		label: "Vision Measure (capture/pixels/ocr/wallpaper/semantic/env)",
		description:
			"像素级测量与感知传感器（本地，无网络）。用于一切需要从图片取数值/坐标/文字的任务——模型不能直接看图，任何视觉推理必须先经此工具获取结构化数据。actions: capture=Wayland 截屏(必填 out)；pixels=主色直方图/区域取色/diff 异常定位/WCAG 对比度(必填 image)；ocr=文字+精确 4 点 bbox(必填 image，小字加 upscale)；wallpaper=壁纸批量程序化分类(必填 dir)；semantic=L2 语义标签(opt-in enable)；env=环境自检。区分：整页多传感器融合用 vs_fuse analyze；图标级 UI 元素用 vs_struct omniparser；本工具是单图测量/文字/颜色。",
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
			backend: Type.Optional(
				Type.String({
					description:
						"OCR 后端：rapidocr（默认，快 ~2-5s，召回中）/ paddle（PP-OCRv6 medium，高召回但慢 ~7-40s）",
				}),
			),
			preprocess: Type.Optional(
				Type.String({
					description:
						"预处理：none（默认）/ contrast（自动对比度拉伸，低对比度文字用）",
				}),
			),
			enable: Type.Optional(Type.Boolean({ description: "opt-in 开启 L2 语义（semantic）" })),
			prompt: Type.Optional(Type.String({ description: "语义提示词（semantic，可选）" })),
			max_tokens: Type.Optional(Type.Number({ description: "语义最大 token（semantic）" })),
		}),
		async execute(_id, params) {
			return dispatch(MEASURE, String(params.action), params);
		},
	});

	pi.registerTool({
		name: "vs_struct",label: "Structure (dom/pptx/omniparser/layout)",
		description:
			"L0 源码结构化 + DL 感知。actions: dom=网页布局无损真值(必填 url，DOM+computed style 比截图更准)；pptx=PPTX 结构(必填 file，pt 坐标/填充 hex/字体)；omniparser=任意截图图标级 UI 元素+语义描述(必填 image，无 DOM 时的替代；CPU 首载 10-20s，单图 30-60s)；layout=文档版式分析 PP-DocLayoutV3(必填 image，标题/正文/图表/表格区域；首次下模型 ~30MB 需代理；与 omniparser 互补：本工具面向文档而非 UI)；pdf=PDF 文本块抽取(必填 file，pt 坐标，可选 pages/render_dir 渲染页面供版式分析)；a11y=桌面应用无障碍树(原生应用的 L0 真值：角色/名称/屏幕坐标；可选 app 过滤/list 列应用/with_text 抓文本；走系统 python3 需 python-gobject)。区分：只要文字/颜色用 vs_measure；本工具提供元素结构与源码层证据，输出可传给 vs_fuse 做审计/规则。",
		parameters: Type.Object({
			action: Type.Union([
				Type.Literal("dom"),
				Type.Literal("pptx"),
				Type.Literal("omniparser"),
				Type.Literal("layout"),
				Type.Literal("pdf"),
				Type.Literal("a11y"),
			]),
			url: Type.Optional(Type.String({ description: "要分析的 URL（dom）" })),
			max_elements: Type.Optional(Type.Number({ description: "最多元素（dom，默认 60）" })),
			screenshot: Type.Optional(Type.String({ description: "DOM 会话截图输出路径（dom）" })),
			file: Type.Optional(Type.String({ description: "pptx 文件路径（pptx）" })),
			max_shapes: Type.Optional(Type.Number({ description: "最多形状（pptx，默认 200）" })),
			slide: Type.Optional(Type.Number({ description: "只导出第 N 张（pptx）" })),
			image: Type.Optional(Type.String({ description: "图片路径（omniparser/layout）" })),
			max_items: Type.Optional(Type.Number({ description: "最多元素（omniparser/layout，默认 60）" })),
			no_ocr: Type.Optional(Type.Boolean({ description: "跳过 OCR 仅图标（omniparser）" })),
			min_conf: Type.Optional(Type.Number({ description: "最小置信度（layout，默认 0.3）" })),
			pages: Type.Optional(Type.String({ description: "PDF 页范围 1-3 或 all（pdf）" })),
			render_dir: Type.Optional(Type.String({ description: "渲染页面输出目录（pdf，可选）" })),
			app: Type.Optional(Type.String({ description: "应用名过滤（a11y）" })),
			with_text: Type.Optional(Type.Boolean({ description: "文本类角色额外抓内容（a11y）" })),
		}),
		async execute(_id, params) {
			return dispatch(STRUCT, String(params.action), params);
		},
	});

	pi.registerTool({
		name: "vs_fuse",label: "Fusion & Rules (analyze/crosscheck/audit/rules/critic)",
		description:
			"确定性融合/审计/准则/复核（本地）。把多个测量/结构报告合并判定，或对已有报告做规则审计。actions: analyze=配置驱动整页管线(必填 task；diagnose-screenshot 是整页分析首选)；crosscheck=DOM↔OCR↔像素三方互验(必填 image，可选 dom/ocr 报告)；audit=重叠/出界/对比度审计(必填 report)；rules=设计准则引擎(必填 report；R1对比度/R2重叠/R3对齐/R4间距/R5安全区，仅评估设计元素)；critic=VLM 复核裁剪区(必填 report+image，opt-in enable；出界/安全区等全局属性缺陷在裁剪视图会误判)。注意：audit/rules/critic 需先把前置工具输出存为报告 JSON 文件再传入。",
		parameters: Type.Object({
			action: Type.Union([
				Type.Literal("analyze"),
				Type.Literal("crosscheck"),
				Type.Literal("audit"),
				Type.Literal("rules"),
				Type.Literal("critic"),
			]),
			task: Type.Optional(Type.String({ description: "任务名（analyze）" })),
			compare: Type.Optional(Type.String({ description: "对比图路径（analyze diff-screenshots 任务）" })),
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
			"CLIP (ViT-B-32, CPU, offline) 相似图聚类：感知相似度矩阵 + 阈值贪心分组（确定性，同输入同输出）。传 dir 或 files(逗号分隔)，可选 threshold(默认 0.75，越大分组越细)/max_files(默认 200)。输出 clusters[]（代表图+成员相似度）+ top_pairs[]。首次运行下载模型 ~350MB（需代理，之后离线）。运行于 omniparser env。用于图片集合的相似分组（壁纸/截图/照片），单张分析不要用它。",
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
