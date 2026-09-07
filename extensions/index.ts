/**
 * pi-vision-struct — pi-coding-agent 适配器壳（V4.1：exec vs-bin 二进制，零业务逻辑）
 *
 * 全部计算在 Rust 二进制 `vs-bin`；本文件只做：
 * spawn 二进制 + 参数透传 + 输出回传。工具名/参数与 V3 完全兼容。
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { execFile } from "node:child_process";
import { fileURLToPath } from "node:url";

const PKG_ROOT = fileURLToPath(new URL("..", import.meta.url));
const VS_BIN = process.env.VS_BIN || `${PKG_ROOT}rust/target/release/vs-bin`;

// action → vs-bin 子命令（全 Rust，零 Python 回退）
const RUST_ACTIONS: Record<string, string> = {
	capture: "capture",
	pixels: "pix",
	ocr: "ocr",
	scene_stats: "scene_stats",
	dom: "dom",
	pptx: "pptx",
	pdf: "pdf",
	a11y: "a11y",
	detect: "detect",
	saliency: "saliency",
	segment: "segment",
	depth_midas: "depth",
	edge: "edge",
	ascii: "ascii",
	geometry: "geometry",
	fusion: "fusion",
	audit: "audit",
	rules: "rules",
	omniparser: "omniparser",
	layout: "layout",
	blend: "blend",
	blender_dump: "blend",
	wallpaper: "wall",
	analyze: "analyze",
	cluster: "cluster",
	audit3d: "audit3d",
	zoom: "protocol",
	probe: "protocol",
	check: "check",
	setup: "setup",
};

// 参数名 → vs-bin flag（仅透传，无逻辑）
const FLAGS: Record<string, string> = {
	image: "--image",
	region: "--region",
	colors: "--colors",
	url: "--url",
	file: "--file",
	report: "--report",
	canvas: "--canvas",
	classes: "--classes",
	max_items: "--max-items",
	min_conf: "--min-conf",
	max_elements: "--max-elements",
	app: "--app",
	slide: "--slide",
	pages: "--pages",
	top_n: "--top-n",
	min_score: "--min-score",
	bins: "--bins",
	max_shapes: "--max-shapes",
	mode: "--mode",
	groups: "--groups",
	out: "--out",
	blend: "--file",
};

function run(bin: string, args: string[], timeoutMs: number): Promise<string> {
	return new Promise((resolve) => {
		execFile(bin, args, { timeout: timeoutMs, maxBuffer: 16 * 1024 * 1024 }, (err, stdout, stderr) => {
			if (err) {
				resolve(JSON.stringify({ error: "vs failed", detail: String(stderr || err.message).slice(0, 500) }));
			} else {
				resolve(stdout.trim());
			}
		});
	});
}

async function dispatch(action: string, params: Record<string, unknown>): Promise<string> {
	const sub = RUST_ACTIONS[action];
	if (sub) {
		const args = [sub];
		for (const [k, v] of Object.entries(params)) {
			if (FLAGS[k] && v !== undefined && v !== "") args.push(FLAGS[k], String(v));
		}
		return run(VS_BIN, args, 600_000);
	}
	return JSON.stringify({ error: `未知 action: ${action}` });
}

export default function visionStructExtension(pi: ExtensionAPI) {
	pi.registerTool({
		name: "vs",
		label: "Vision Suite 视觉套件（单端口 25 动作）",
		description:
			"数理化视觉套件（V4 Rust）。全部输出为精确数值/坐标/hex/矩阵，零主观描述。\n" +
			"测量：capture截图(out)/pixels取色(image,region,colors)/ocr文字坐标(image)/wallpaper(dir)/scene_stats数理统计(image)/env。\n" +
			"结构：dom(url)/pptx(file)/omniparser(image)/layout(image)/pdf(file)/a11y(app,list)/detect(image,classes,阈值)。\n" +
			"三维：blender_dump(blend)/depth(blend+image)/audit3d(report,阈值mm)。\n" +
			"融合：analyze(task)/audit(report)/rules(report)。CLIP聚类(dir)。\n" +
			"自检：check（只读环境自检）/setup（安装，setup_args 传 --with-omniparser 等）。",
		parameters: {
			type: "object",
			properties: {
				action: { type: "string", enum: Object.keys(RUST_ACTIONS) },
				image: { type: "string" },
				region: { type: "string" },
				colors: { type: "number" },
				url: { type: "string" },
				file: { type: "string" },
				report: { type: "string" },
				canvas: { type: "string" },
				classes: { type: "string" },
				max_items: { type: "number" },
				min_conf: { type: "number" },
				max_elements: { type: "number" },
				app: { type: "string" },
				slide: { type: "number" },
				pages: { type: "string" },
				top_n: { type: "number" },
				min_score: { type: "number" },
				bins: { type: "number" },
				max_shapes: { type: "number" },
				mode: { type: "string" },
				groups: { type: "string" },
				out: { type: "string" },
				blend: { type: "string" },
				setup_args: { type: "string" },
			},
			required: ["action"],
		},
		async execute(_id, params) {
			const text = await dispatch(String(params.action), params);
			return { content: [{ type: "text", text }], details: {} };
		},
	});
}
