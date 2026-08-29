/**
 * pi-vision-core — 数理化视觉扩展的**框架无关核心**（双宿主单一事实源）
 *
 * 本模块不依赖任何宿主框架（pi-coding-agent / MCP 均不 import）。
 * 所有真实业务逻辑（python 解析 / preflight / bwrap 沙箱 / Act 参数表 / dispatch）
 * 都在这里；pi 与 Claude Code 各写一个薄适配器壳（index.ts / server.mcp.ts）即可接入。
 *
 * 设计：22 个细粒度能力 + check/setup → 单端口 `vs`（action 枚举）；全部输出可复算。
 */

import { Type } from "typebox";
import { execFile } from "node:child_process";
import { fileURLToPath } from "node:url";
import {
	existsSync,
	readFileSync,
	appendFileSync,
	mkdirSync,
	renameSync,
	statSync,
} from "node:fs";
import { homedir } from "node:os";

const PKG_ROOT = fileURLToPath(new URL("..", import.meta.url));
const PY = (name: string) => `${PKG_ROOT}python/${name}`;

// ---- 版本（避免硬编码漂移）----
export const VERSION: string = (() => {
	try {
		const pkg = JSON.parse(
			readFileSync(fileURLToPath(new URL("../package.json", import.meta.url)), "utf-8"),
		);
		return String(pkg.version ?? "2.2.2");
	} catch {
		return "2.2.2";
	}
})();

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
		const isBash = pythonBin === "bash" || pythonBin.endsWith("/bash");
		if (args[0] !== "-c" && !isSelfDiag && !isBash) {
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
		// 豁免（sandbox=false）: dom（本职加载用户 URL）/ a11y（会话 DBus）/
		//   semantic（宿主 Ollama）/ check|setup（需网络安装或读环境）
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
	vlm: 360_000, // 已废弃（原 critic 复核），保留键名兼容旧配置
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
	chart_data: {
		script: "vs_chart.py",
		sandbox: false,
		timeout: T.vlm,
		build: (p) => [
			...flag(p, "image", "--image"),
			...flag(p, "region", "--region"),
			...flag(p, "prompt", "--prompt"),
			...on(p, "enable", "--enable"),
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
	scene_stats: {
		script: "vs_scene_stats.py",
		timeout: T.short,
		build: (p) => [
			...flag(p, "image", "--image"),
			...flag(p, "region", "--region"),
			...flag(p, "colors", "--colors"),
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
	detect: {
		script: "vs_detect.py",
		bin: OMNI_PYTHON,
		timeout: T.doc,
		build: (p) => [
			...flag(p, "image", "--image"),
			...flag(p, "classes", "--classes"),
			...flag(p, "threshold", "--threshold"),
			...flag(p, "max_items", "--max-items"),
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
			...flag(p, "prompt", "--prompt"),
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
	audit3d: {
		script: "vs_audit3d.py",
		timeout: T.normal,
		build: (p) => [
			...flag(p, "report", "--report"),
			...flag(p, "gap_threshold", "--gap-threshold"),
			...on(p, "method", "--method"),
		],
	},
	// 可选 VLM 语义兜底：不进入数值主链；sandbox=false（宿主 Ollama 127.0.0.1 硬编码）
	semantic: {
		script: "vs_semantic_v2.py",
		sandbox: false,
		timeout: T.vlm,
		build: (p) => [
			...flag(p, "image", "--image"),
			...flag(p, "prompt", "--prompt"),
			...flag(p, "model", "--model"),
		],
	},
	blender_dump: {
		script: "blender_dump_wrapper.sh",
		bin: "bash",
		timeout: T.heavy,
		build: (p) => [
			...flag(p, "blend", "--blend"),
		],
	},
	depth: {
		script: "depth_geom_wrapper.sh",
		bin: "bash",
		timeout: T.heavy,
		build: (p) => [
			...flag(p, "blend", "--blend"),
			...flag(p, "image", "--image"),
			...flag(p, "camera", "--camera"),
		],
	},
};

// ---- 自诊断动作（check / setup）：双宿主均可用，沙箱豁免 ----
const SELFDIAG: Record<string, Act> = {
	check: {
		bin: DEFAULT_PYTHON,
		sandbox: false,
		timeout: 30_000,
		build: () => ["-u", `${PKG_ROOT}python/setup/vs_setup.py`, "--check"],
	},
	setup: {
		bin: DEFAULT_PYTHON,
		sandbox: false, // 需网络安装依赖
		timeout: 1_200_000,
		build: (p) => {
			const rest = (p.setup_args ?? "").trim();
			const setupArgs =
				rest === "" ? ["--dry-run"] : rest.split(/\s+/).filter(Boolean);
			return ["-u", `${PKG_ROOT}python/setup/vs_setup.py`, ...setupArgs];
		},
	},
};

/**
 * 组装单端口路由表（双宿主共用）。22 个动作 + cluster + check/setup。
 */
export function buildRoute(): Record<string, Act> {
	const ROUTE: Record<string, Act> = {};
	for (const table of [MEASURE, STRUCT, FUSE, SELFDIAG])
		for (const [k, v] of Object.entries(table)) ROUTE[k] = v;
	ROUTE.cluster = {
		script: "vs_cluster.py",
		bin: OMNI_PYTHON,
		timeout: T.modelXL,
		build: (p) => [
			...flag(p, "dir", "--dir"),
			...flag(p, "files", "--files"),
			...flag(p, "threshold", "--threshold"),
			...flag(p, "max_files", "--max-files"),
		],
	};
	return ROUTE;
}

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

// ---------- 导出（双宿主共用）----------

export { PKG_ROOT, PY, DEFAULT_PYTHON, OMNI_PYTHON, SANDBOX_ENABLED, BWRAP };
export { runPython, gitIntegrity, preflight, resolvePython };
export { dispatch };
export { T };
export type { Act };

/**
 * 单端口 `vs` 工具的参数 schema（Typebox）—— pi 与 Claude Code 共用，单一来源。
 * 追加 check/setup 动作与 setup_args；其余字段逐字保留自原 index.ts。
 */
export const vsParamType = Type.Object({
	action: Type.Union([
		Type.Literal("capture"), Type.Literal("pixels"), Type.Literal("ocr"),
		Type.Literal("wallpaper"), Type.Literal("scene_stats"), Type.Literal("env"),
		Type.Literal("dom"), Type.Literal("pptx"), Type.Literal("omniparser"),
		Type.Literal("layout"), Type.Literal("pdf"), Type.Literal("a11y"),
		Type.Literal("analyze"), Type.Literal("crosscheck"), Type.Literal("audit"),
		Type.Literal("rules"), Type.Literal("cluster"),
		Type.Literal("detect"), Type.Literal("chart_data"),
		Type.Literal("blender_dump"), Type.Literal("depth"), Type.Literal("audit3d"),
		Type.Literal("semantic"),
		Type.Literal("check"), Type.Literal("setup"),
	]),
	out: Type.Optional(Type.String({ description: "截屏输出路径（capture）" })),
	image: Type.Optional(Type.String({ description: "图片路径（pixels/ocr等多数动作）" })),
	region: Type.Optional(Type.String({ description: "区域 x1,y1,x2,y2（capture/ocr/pixels）" })),
	colors: Type.Optional(Type.Number({ description: "主色数（pixels/wallpaper）" })),
	compare: Type.Optional(Type.String({ description: "对比图（pixels diff）" })),
	wcag: Type.Optional(Type.String({ description: "前景hex,背景hex…（pixels WCAG）" })),
	threshold: Type.Optional(Type.Number({ description: "diff阈值30/聚类0.75" })),
	dir: Type.Optional(Type.String({ description: "图片目录（wallpaper/cluster）" })),
	ext: Type.Optional(Type.String({ description: "扩展名列表（wallpaper）" })),
	max_files: Type.Optional(Type.Number({ description: "最多文件（wallpaper/cluster，默认200）" })),
	prompt: Type.Optional(Type.String({ description: "提示词（semantic/chart_data 语义兜底）" })),
	model: Type.Optional(Type.String({ description: "VLM 模型（semantic 兜底，默认取配置 l2_model）" })),
	enable: Type.Optional(Type.Boolean({ description: "开启 L2/VLM（wallpaper/chart_data语义）" })),
	upscale: Type.Optional(Type.Number({ description: "OCR 放大倍数（默认2）" })),
	max_items: Type.Optional(Type.Number({ description: "最多条目（ocr/omniparser/layout/pdf）" })),
	min_conf: Type.Optional(Type.Number({ description: "最低置信度（ocr/layout）" })),
	backend: Type.Optional(Type.String({ description: "OCR 后端 rapidocr|paddle（ocr）" })),
	preprocess: Type.Optional(Type.String({ description: "none|contrast（ocr）" })),
	daemon: Type.Optional(Type.String({ description: "auto|always|never（ocr paddle 常驻）" })),
	blend: Type.Optional(Type.String({ description: "Blender 文件路径（blender_dump/depth）" })),
	camera: Type.Optional(Type.String({ description: "摄像机名称（depth blender-zpass）" })),
	output: Type.Optional(Type.String({ description: "输出 JSON 路径（blender_dump/depth）" })),
	gap_threshold: Type.Optional(Type.Number({ description: "间隙阈值 mm（audit3d，默认15）" })),
	method: Type.Optional(Type.String({ description: "精度档（audit3d）：auto=OBB-SAT+网格KDTree最大精度 / obb=仅有向包围盒分离轴 / mesh=仅点云距离 / aabb=原AABB回退" })),
	url: Type.Optional(Type.String({ description: "URL（dom；analyze 可选）" })),
	max_elements: Type.Optional(Type.Number({ description: "最多元素（dom/a11y，默认60/80）" })),
	screenshot: Type.Optional(Type.String({ description: "DOM 会话截图输出（dom）" })),
	file: Type.Optional(Type.String({ description: "pptx/pdf 文件路径" })),
	max_shapes: Type.Optional(Type.Number({ description: "最多形状（pptx，默认200）" })),
	slide: Type.Optional(Type.Number({ description: "仅导出第 N 张（pptx）" })),
	no_ocr: Type.Optional(Type.Boolean({ description: "跳过 OCR 仅图标（omniparser）" })),
	pages: Type.Optional(Type.String({ description: "PDF 页范围 1-3/all（pdf）" })),
	render_dir: Type.Optional(Type.String({ description: "渲染输出目录（pdf）" })),
	app: Type.Optional(Type.String({ description: "应用名过滤（a11y）" })),
	list: Type.Optional(Type.Boolean({ description: "仅列应用（a11y）" })),
	with_text: Type.Optional(Type.Boolean({ description: "抓文本内容（a11y）" })),
	task: Type.Optional(Type.String({ description: "预置任务配置名（analyze）：diagnose-screenshot / audit-pptx / classify-images / diff-screenshots。非自由文本" })),
	input: Type.Optional(Type.String({ description: "输入 图片/pptx/目录（analyze）" })),
	setup_args: Type.Optional(Type.String({ description: "setup 动作参数：留空=--dry-run；可传 --with-omniparser / --with-dom 等（setup 自带 vs_setup.py 预检豁免）" })),
	dpr: Type.Optional(Type.Number({ description: "DPR（默认1.0）" })),
	dom: Type.Optional(Type.String({ description: "dom 报告 JSON（crosscheck）" })),
	ocr: Type.Optional(Type.String({ description: "ocr 报告 JSON（crosscheck）" })),
	color_threshold: Type.Optional(Type.Number({ description: "ΔE 阈值（crosscheck，默认5）" })),
	report: Type.Optional(Type.String({ description: "报告 JSON（audit/rules）" })),
	canvas: Type.Optional(Type.String({ description: "画布 WxH（audit/rules）" })),
	overlap_threshold: Type.Optional(Type.Number({ description: "IoU 阈值（audit，默认0.05）" })),
	align_tol: Type.Optional(Type.Number({ description: "对齐容差 px（rules，默认4）" })),
	margin: Type.Optional(Type.Number({ description: "边距 px（rules）" })),
	files: Type.Optional(Type.String({ description: "逗号分隔文件列表，与dir二选一（cluster）" })),
	classes: Type.Optional(Type.String({ description: "逗号分隔类别（detect 开放词表）" })),
});

/**
 * Typebox schema → MCP JSON Schema 子集（供 server.mcp.ts 使用）。
 * Typebox 把可选属性标 `optional:true`，借此收集 required；去掉 `$id`、补 additionalProperties。
 */
export function toMcpJsonSchema(t: ReturnType<typeof Type.Object>): {
	type: "object";
	properties: Record<string, unknown>;
	required: string[];
	additionalProperties: boolean;
} {
	// Typebox already builds correct JSON Schema with a top-level .required array
	// listing only truly required fields (e.g. ["action"]). Optional properties
	// simply omit that key. We just pass the schema through, stripping the
	// internal $id that Typebox adds and setting additionalProperties: false.
	const tAny = t as any;
	const props = tAny.properties ?? {};
	const clean: Record<string, unknown> = {};
	for (const [k, v] of Object.entries(props)) {
		const { $id, ...rest } = v as Record<string, unknown>;
		clean[k] = rest;
	}
	return {
		type: "object",
		properties: clean,
		required: tAny.required ?? [],
		additionalProperties: false,
	};
}

/**
 * 工具描述文案（双宿主共用，单一来源）；补 check/setup 一行。
 */
export const VS_DESCRIPTION =
	"数理化视觉套件（v2.2.2）25动作。全部输出为精确数值/坐标/hex/矩阵，零主观描述。\n"
	+ "测量：capture截图(out)/pixels取色(image,region,colors)/ocr文字坐标(image)/wallpaper(dir)/scene_stats数理统计(image)/env。\n"
	+ "结构：dom(url)/pptx(file)/omniparser(image)/layout(image)/pdf(file)/a11y(app,list)/detect(image,classes,阈值)。\n"
	+ "三维：blender_dump(blend)→场景图4×4矩阵+8点bbox3d/depth(blend+image)→深度矩阵→mm统计/audit3d(report,阈值mm)→间隙/干涉。\n"
	+ "融合：analyze(task)/crosscheck互验(image)/audit(report)/rules(report)。CLIP聚类(dir)。\n"
	+ "自检：check（只读环境自检）/setup（安装，setup_args 传 --with-omniparser 等）。\n"
	+ "原则：先capture后分析；所有输出带单位；数值推理由DeepSeek执行。";
