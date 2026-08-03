/**
 * pi-vision-struct — 结构化视觉工具
 *
 * 为纯文本模型（DeepSeek）提供像素级精确的视觉通道：
 * 工具输出结构化 JSON（数字/坐标/hex），而非自然语言转述。
 *
 * 分层无损通道（vision-report/v1）：
 *   L0 源码：dom_dump（DOM+computed style）、pptx_dump（PPTX XML）
 *   L1 测量：pix_analyze（直方图/区域/diff/WCAG）、ocr_boxes（带坐标框）
 *   传感器：screen_capture（grim）
 *
 * 全部只读、本地、无网络外发。Python 脚本运行于 conda env `pi-vision`
 * （默认 /home/Arch/conda-envs/pi-vision/bin/python，可用 PI_VISION_PYTHON 覆盖）。
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
						: rest
								.replace(/^setup\s*/, "")
								.split(/\s+/)
								.filter(Boolean);
				// 首次安装前先展示计划；已有环境时直接执行
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
						envs?: Record<string, { exists: boolean; complete: boolean }>;
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
		name: "vs_env_check",
		label: "Vision Struct Env Check",
		description:
			"Report the pi-vision conda environment status: python version and availability of onnxruntime, rapidocr, pptx, playwright, numpy, PIL. Returns structured JSON.",
		parameters: Type.Object({}),
		async execute() {
			const script = [
				"import json,sys",
				"mods=['onnxruntime','rapidocr','pptx','playwright','PIL']",
				"res={'python':sys.version.split()[0]}",
				"for m in mods:",
				"    try:",
				"        mod=__import__(m); res[m]=getattr(mod,'__version__','ok')",
				"    except Exception as e: res[m]='MISSING:'+str(e)[:60]",
				"print(json.dumps(res,ensure_ascii=False))",
			].join("\n");
			const out = await runPython(pythonBin, ["-c", script], 30000);
			return { content: [{ type: "text", text: out }], details: {} };
		},
	});

	pi.registerTool({
		name: "screen_capture",
		label: "Screen Capture (grim)",
		description:
			"Capture the screen (or a region) on Wayland via grim and return the image path and dimensions. Use for diagnosing the user's actual screen / Firefox / desktop issues.",
		parameters: Type.Object({
			out: Type.String({
				description: "输出 PNG 路径（建议 /tmp/vs_cap.png）",
			}),
			region: Type.Optional(
				Type.String({ description: "区域 x1,y1,x2,y2（可选，缺省全屏）" }),
			),
		}),
		async execute(_id, params) {
			const args = [PY("vs_capture.py"), "--out", params.out];
			if (params.region) args.push("--region", params.region);
			const out = await runPython(pythonBin, args, 20000);
			return { content: [{ type: "text", text: out }], details: {} };
		},
	});

	pi.registerTool({
		name: "pix_analyze",
		label: "Pixel Analysis",
		description:
			"Deterministic pixel measurements (no model): dominant color histogram (hex+%), region colors, brightness/saturation, pixel-level diff anomaly localization between two images, WCAG 2.x contrast ratios. Returns structured JSON with exact numbers DeepSeek can reason over.",
		parameters: Type.Object({
			image: Type.String({ description: "图片路径" }),
			regions: Type.Optional(
				Type.Array(
					Type.String({ description: '区域 "x1,y1,x2,y2"（中心像素取色）' }),
				),
			),
			compare: Type.Optional(
				Type.String({ description: "对比图路径，做像素级差异定位" }),
			),
			colors: Type.Optional(Type.Number({ description: "主色数量（默认 8）" })),
			wcag: Type.Optional(
				Type.Array(Type.String({ description: '"前景hex,背景hex" 对比度对' })),
			),
			threshold: Type.Optional(
				Type.Number({ description: "diff 阈值（默认 30）" }),
			),
		}),
		async execute(_id, params) {
			const args = [PY("vs_pix.py"), "--image", params.image];
			const regions = params.regions ?? [];
			if (regions.length) args.push("--regions", ...regions);
			if (params.compare) args.push("--compare", params.compare);
			if (params.colors) args.push("--colors", String(params.colors));
			const wcag = params.wcag ?? [];
			if (wcag.length) args.push("--wcag", ...wcag);
			if (params.threshold)
				args.push("--diff-threshold", String(params.threshold));
			const out = await runPython(pythonBin, args, 30000);
			return { content: [{ type: "text", text: out }], details: {} };
		},
	});

	pi.registerTool({
		name: "ocr_boxes",
		label: "OCR with Boxes (RapidOCR)",
		description:
			"Extract text with exact bounding boxes and confidence via local RapidOCR (PP-OCRv6). Supports region crop + upscale for small text. Returns structured JSON: text, conf, box (4 points), center.",
		parameters: Type.Object({
			image: Type.String({ description: "图片路径" }),
			region: Type.Optional(
				Type.String({ description: '裁剪区域 "x1,y1,x2,y2"（可选）' }),
			),
			upscale: Type.Optional(
				Type.Number({ description: "放大倍数（默认 2，小字号用）" }),
			),
			max_items: Type.Optional(
				Type.Number({ description: "最多返回条数（默认 100）" }),
			),
			min_conf: Type.Optional(
				Type.Number({ description: "最低置信度（默认 0.5）" }),
			),
		}),
		async execute(_id, params) {
			const args = [PY("vs_ocr.py"), "--image", params.image];
			if (params.region) args.push("--region", params.region);
			if (params.upscale) args.push("--upscale", String(params.upscale));
			if (params.max_items) args.push("--max-items", String(params.max_items));
			if (params.min_conf) args.push("--min-conf", String(params.min_conf));
			const out = await runPython(pythonBin, args, 60000);
			return { content: [{ type: "text", text: out }], details: {} };
		},
	});

	pi.registerTool({
		name: "dom_dump",
		label: "DOM Structured Dump",
		description:
			"Load a URL in a controlled Firefox (Playwright, WebDriver BiDi) and extract visible elements: tag, role, text, bbox, computed style (color, bg, font-size, z-index, position, overflow...). DOM is the lossless layout truth. Optionally save a screenshot of the session.",
		parameters: Type.Object({
			url: Type.String({ description: "要分析的 URL" }),
			max_elements: Type.Optional(
				Type.Number({ description: "最多元素（默认 60）" }),
			),
			screenshot: Type.Optional(
				Type.String({ description: "会话截图输出路径（可选）" }),
			),
		}),
		async execute(_id, params) {
			const args = [PY("vs_dom.py"), "--url", params.url];
			if (params.max_elements)
				args.push("--max-elements", String(params.max_elements));
			if (params.screenshot) args.push("--out-screenshot", params.screenshot);
			const out = await runPython(pythonBin, args, 60000);
			return { content: [{ type: "text", text: out }], details: {} };
		},
	});

	pi.registerTool({
		name: "pptx_dump",
		label: "PPTX Structure Export",
		description:
			"Losslessly export PPTX structure via python-pptx: per-slide shapes with position/size in pt, fill color hex, font size/color, text, images. DeepSeek can compute overlaps, alignment, contrast from these exact numbers.",
		parameters: Type.Object({
			file: Type.String({ description: "pptx 文件路径" }),
			max_shapes: Type.Optional(
				Type.Number({ description: "最多形状（默认 200）" }),
			),
			slide: Type.Optional(
				Type.Number({ description: "只导出第 N 张幻灯片（可选）" }),
			),
		}),
		async execute(_id, params) {
			const args = [PY("vs_pptx.py"), "--file", params.file];
			if (params.max_shapes)
				args.push("--max-shapes", String(params.max_shapes));
			if (params.slide) args.push("--slide", String(params.slide));
			const out = await runPython(pythonBin, args, 30000);
			return { content: [{ type: "text", text: out }], details: {} };
		},
	});

	pi.registerTool({
		name: "wallpaper_classify",
		label: "Wallpaper Batch Classify",
		description:
			"Batch-classify wallpaper images deterministically: dominant colors (hex+%), brightness, saturation, dominant hue family (red/orange/yellow/green/cyan/blue/purple/neutral), tone (dark/mid/bright), aspect ratio, and a combined category. Optionally attach L2 semantic tags (style/theme) via local Ollama qwen3-vl — opt-in (--semantic), localhost only.",
		parameters: Type.Object({
			dir: Type.String({ description: "壁纸目录路径" }),
			colors: Type.Optional(Type.Number({ description: "主色数量（默认 5）" })),
			max_files: Type.Optional(
				Type.Number({ description: "最多处理文件数（默认 200）" }),
			),
			ext: Type.Optional(
				Type.Array(
					Type.String({
						description: "文件扩展名（如 png jpg，默认常见图片格式）",
					}),
				),
			),
			semantic: Type.Optional(
				Type.Boolean({
					description:
						"opt-in: 附加 L2 语义标签（风格/主题，本地 qwen3-vl:8b）",
				}),
			),
			semantic_max: Type.Optional(
				Type.Number({ description: "语义标注最多文件数（默认 10）" }),
			),
		}),
		async execute(_id, params) {
			const args = [PY("vs_wall.py"), "--dir", params.dir];
			if (params.colors) args.push("--colors", String(params.colors));
			if (params.max_files) args.push("--max-files", String(params.max_files));
			const ext = params.ext ?? [];
			if (ext.length) args.push("--ext", ...ext);
			if (params.semantic) args.push("--semantic");
			if (params.semantic_max)
				args.push("--semantic-max", String(params.semantic_max));
			const out = await runPython(pythonBin, args, 300000);
			return { content: [{ type: "text", text: out }], details: {} };
		},
	});

	pi.registerTool({
		name: "semantic_tag",
		label: "Semantic Tag (L2, opt-in)",
		description:
			"L2 semantic labeling via local Ollama qwen3-vl:8b (localhost:11434, no network egress). Use only for non-measurable attributes (style/theme/type). OPT-IN by default: pass enable=true or set PI_VISION_SEMANTIC=1, otherwise it refuses and returns enabled:false. VRAM-safe: image downscaled to 768px, num_ctx=8192, max 300 tokens.",
		parameters: Type.Object({
			image: Type.String({ description: "图片路径" }),
			enable: Type.Optional(
				Type.Boolean({ description: "显式开启 L2 语义（默认关闭）" }),
			),
			prompt: Type.Optional(
				Type.String({ description: "自定义语义标签提示词（可选）" }),
			),
			max_tokens: Type.Optional(
				Type.Number({
					description: "最大生成 token 数（默认 2048，思考型模型需要）",
				}),
			),
		}),
		async execute(_id, params) {
			const args = [PY("vs_semantic.py"), "--image", params.image];
			if (params.enable) args.push("--enable");
			if (params.prompt) args.push("--prompt", params.prompt);
			if (params.max_tokens)
				args.push("--max-tokens", String(params.max_tokens));
			const out = await runPython(pythonBin, args, 360000);
			return { content: [{ type: "text", text: out }], details: {} };
		},
	});

	pi.registerTool({
		name: "vs_crosscheck",
		label: "Cross-check Fusion (DOM vs OCR vs Pixels)",
		description:
			"Multi-sensor cross-verification: compares DOM declared colors against measured pixel colors (CIELAB ΔE), DOM text against OCR text, and detects element overlaps. Automatically surfaces rendering anomalies with evidence pairs. Requires a screenshot image plus optional dom/ocr report JSON paths (from dom_dump/ocr_boxes tool results saved to files).",
		parameters: Type.Object({
			image: Type.String({ description: "截图图片路径（device 像素）" }),
			dom: Type.Optional(
				Type.String({ description: "dom_dump 输出 JSON 文件路径" }),
			),
			ocr: Type.Optional(
				Type.String({ description: "ocr_boxes 输出 JSON 文件路径" }),
			),
			dpr: Type.Optional(Type.Number({ description: "DPR（默认 1.0）" })),
			color_threshold: Type.Optional(
				Type.Number({ description: "ΔE 阈值（默认 5.0）" }),
			),
		}),
		async execute(_id, params) {
			const args = [PY("vs_crosscheck.py"), "--image", params.image];
			if (params.dom) args.push("--dom", params.dom);
			if (params.ocr) args.push("--ocr", params.ocr);
			if (params.dpr) args.push("--dpr", String(params.dpr));
			if (params.color_threshold)
				args.push("--color-threshold", String(params.color_threshold));
			const out = await runPython(pythonBin, args, 60000);
			return { content: [{ type: "text", text: out }], details: {} };
		},
	});

	pi.registerTool({
		name: "vs_audit",
		label: "Element Audit (overlap / off-canvas / contrast)",
		description:
			"Deterministic geometric/style audit of any element list (DOM elements or PPTX shapes): pairwise overlap, off-canvas, WCAG contrast failures. Takes a report JSON file path (from pptx_dump or dom_dump).",
		parameters: Type.Object({
			report: Type.String({
				description: "pptx_dump / dom_dump 输出 JSON 文件路径",
			}),
			canvas: Type.Optional(
				Type.String({ description: "画布尺寸 WxH（如 720x540，pptx 用 pt）" }),
			),
			overlap_threshold: Type.Optional(
				Type.Number({ description: "IoU 阈值（默认 0.05）" }),
			),
		}),
		async execute(_id, params) {
			const args = [PY("vs_audit.py"), "--report", params.report];
			if (params.canvas) args.push("--canvas", params.canvas);
			if (params.overlap_threshold)
				args.push("--overlap-threshold", String(params.overlap_threshold));
			const out = await runPython(pythonBin, args, 30000);
			return { content: [{ type: "text", text: out }], details: {} };
		},
	});

	pi.registerTool({
		name: "vs_analyze",
		label: "Task Engine (config-driven pipeline)",
		description:
			"Config-driven vision pipeline: runs sensors + fusion operators per a task config (tasks/<name>.json) and returns the fused report. New tasks = new configs, no code. Built-in tasks: diagnose-screenshot, audit-pptx, classify-images.",
		parameters: Type.Object({
			task: Type.String({ description: "任务名（如 diagnose-screenshot）" }),
			input: Type.Optional(
				Type.String({ description: "输入（图片路径 / pptx / 目录）" }),
			),
			url: Type.Optional(
				Type.String({
					description: "可选 URL（diagnose-screenshot 的 DOM 源）",
				}),
			),
			dpr: Type.Optional(Type.Number({ description: "DPR（默认 1.0）" })),
		}),
		async execute(_id, params) {
			const args = [PY("vs_analyze.py"), "--task", params.task];
			if (params.input) args.push("--input", params.input);
			if (params.url) args.push("--url", params.url);
			if (params.dpr) args.push("--dpr", String(params.dpr));
			const out = await runPython(pythonBin, args, 300000);
			return { content: [{ type: "text", text: out }], details: {} };
		},
	});

	pi.registerTool({
		name: "vs_omniparser",
		label: "OmniParser UI Elements (icons + text)",
		description:
			"OmniParser V2 (Microsoft): structure ANY screenshot into icon-level UI elements with semantic captions (Florence-2) + OCR text, no DOM needed. Runs CPU-only in the omniparser conda env; first call loads models (~10-20s), then ~30-60s per image. Output: schema v2 elements (type: icon|text, bbox, text, interactivity).",
		parameters: Type.Object({
			image: Type.String({ description: "图片路径（任意截图）" }),
			max_items: Type.Optional(
				Type.Number({ description: "最多元素（默认 60）" }),
			),
			no_ocr: Type.Optional(
				Type.Boolean({ description: "跳过 OCR 仅图标（默认 false）" }),
			),
		}),
		async execute(_id, params) {
			const args = [PY("vs_omniparser.py"), "--image", params.image];
			if (params.max_items) args.push("--max-items", String(params.max_items));
			if (params.no_ocr) args.push("--no-ocr");
			const out = await runPython(OMNI_PYTHON, args, 600000);
			return { content: [{ type: "text", text: out }], details: {} };
		},
	});

	pi.registerTool({
		name: "vs_rules",
		label: "Design Rule Engine (contrast/alignment/spacing/safe-area)",
		description:
			"Deterministic design-guideline rule engine over a schema-v2 report (DOM / PPTX / OmniParser / fusion output). Rules: R1 text_contrast (WCAG AA 4.5:1 / 3:1 large), R2 overlap, R3 alignment_drift (edge/center cluster near-miss), R4 spacing_anomaly (outlier gaps > k x median), R5 safe_area (off-canvas / edge text). Every finding carries evidence (values + thresholds) and suggested_cause; metrics include design_score. Input: a report JSON file path.",
		parameters: Type.Object({
			report: Type.String({
				description:
					"schema-v2 报告 JSON 路径（vs_analyze / vs_dom / pptx_dump / vs_omniparser 输出）",
			}),
			canvas: Type.Optional(
				Type.String({
					description: "画布 WxH（可选，默认从报告 source 推导）",
				}),
			),
			align_tol: Type.Optional(
				Type.Number({ description: "对齐聚类容差 px（默认 4）" }),
			),
			margin: Type.Optional(
				Type.Number({ description: "安全区边缘阈值 px（默认 2）" }),
			),
		}),
		async execute(_id, params) {
			const args = [PY("vs_rules.py"), "--report", params.report];
			if (params.canvas) args.push("--canvas", params.canvas);
			if (params.align_tol) args.push("--align-tol", String(params.align_tol));
			if (params.margin) args.push("--margin", String(params.margin));
			const out = await runPython(pythonBin, args, 60000);
			return { content: [{ type: "text", text: out }], details: {} };
		},
	});

	pi.registerTool({
		name: "vs_cluster",
		label: "CLIP Offline Clustering (similar images)",
		description:
			"CLIP (ViT-B-32, CPU, offline) perceptual similarity clustering of images (wallpapers / screenshots / photos). Greedy single-link grouping at a cosine threshold (deterministic; same input => same output). Output: schema v2 report with clusters[] (representative, members with sim_to_rep), top_pairs[] and metrics. Runs in the omniparser conda env; first call downloads the model weights (~350MB, needs proxy), afterwards fully offline.",
		parameters: Type.Object({
			dir: Type.Optional(Type.String({ description: "图片目录（递归枚举）" })),
			files: Type.Optional(
				Type.String({ description: "逗号分隔的文件列表（与 dir 二选一）" }),
			),
			threshold: Type.Optional(
				Type.Number({
					description: "余弦相似度阈值（默认 0.75，越大分组越细）",
				}),
			),
			max_files: Type.Optional(
				Type.Number({ description: "最多处理（默认 200）" }),
			),
		}),
		async execute(_id, params) {
			const args = [PY("vs_cluster.py")];
			if (params.dir) args.push("--dir", params.dir);
			if (params.files) args.push("--files", params.files);
			if (params.threshold) args.push("--threshold", String(params.threshold));
			if (params.max_files) args.push("--max-files", String(params.max_files));
			const out = await runPython(OMNI_PYTHON, args, 600000);
			return { content: [{ type: "text", text: out }], details: {} };
		},
	});

	pi.registerTool({
		name: "vs_critic",
		label: "VLM-as-Critic (qwen3-vl review of findings)",
		description:
			"VLM-as-critic loop: crops each suspicious region of a rules/audit report (by severity, capped) and asks local qwen3-vl:8b (Ollama, localhost only) to confirm/reject the finding. Verdicts merge into findings as critic evidence. OPT-IN (L2 semantic cost): pass enable=true or set PI_VISION_CRITIC=1, otherwise returns the report unchanged with critic.enabled=false. Note: global-property defects (off-canvas/safe-area) are misjudged in crop view (context lost) - exempt those or interpret with care.",
		parameters: Type.Object({
			report: Type.String({
				description: "规则/审计报告 JSON 路径（vs_rules / vs_audit 输出）",
			}),
			image: Type.String({ description: "原图路径（裁剪可疑区）" }),
			enable: Type.Optional(
				Type.Boolean({ description: "显式开启 VLM 复核（默认 opt-in 关闭）" }),
			),
			max_critic: Type.Optional(
				Type.Number({
					description: "裁剪上限（默认 8，按 critical>warn>info）",
				}),
			),
			margin: Type.Optional(
				Type.Number({ description: "裁剪边距 px（默认 4）" }),
			),
		}),
		async execute(_id, params) {
			const args = [
				PY("vs_critic.py"),
				"--report",
				params.report,
				"--image",
				params.image,
			];
			if (params.enable) args.push("--enable");
			if (params.max_critic)
				args.push("--max-critic", String(params.max_critic));
			if (params.margin) args.push("--margin", String(params.margin));
			const out = await runPython(pythonBin, args, 900000);
			return { content: [{ type: "text", text: out }], details: {} };
		},
	});
}
