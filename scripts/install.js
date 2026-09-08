#!/usr/bin/env node
// ============================================================================
// pi-version-and-gui postinstall 安装器
// ============================================================================
// 职责（两阶段）：
//   1. 二进制：从 GitHub Release 下载 3 个 Rust 二进制（vs-bin/vsd/vs-mcp）
//      → <pkg_root>/rust/target/release/（与 extensions/index.ts 的 VS_BIN 路径一致）
//   2. 模型：从 modelscope 下载 rapidocr 3 个 ONNX（OCR 传感器必需）
//      → ~/.cache/vsensor/rapidocr/（Rust 二进制硬编码读此路径）
//
// 无外部依赖（Node 内置 https），首次尝试 + 3 次重试，校验大小，仅 Linux x86_64。
//
// ⚠️ 二进制问题说明（重要）：
//   - 二进制 151MB 不打包进 npm（npm 包保持轻量 248KB），从 GitHub Release 下载
//   - 若 GitHub 不可达（网络限制），手动下载：
//       curl -L -o <pkg_root>/rust/target/release/vs-bin \
//         https://github.com/Moritz230127/pi-vision-struct/releases/download/v4.0.0/vs-bin
//   - npm 11 的 install-scripts 安全策略可能阻止 postinstall 自动执行：
//       npm install-scripts approve pi-version-and-gui
//     或手动运行：node scripts/install.js
//
// ⚠️ 模型问题说明（重要）：
//   - 模型 4.1GB 不打包进 npm（太大），存 ~/.cache/vsensor/（独立于工作区）
//   - rapidocr 3 个 ONNX 本脚本自动下载（modelscope 直连）
//   - 其余模型（u2net/midas/mobile_sam/florence2/owlv2/doclayout/clip）需 PyTorch
//     导出或 HuggingFace 下载，见 README「模型清单」——本脚本只提示不下载
//   - 模型缺失时对应传感器报错，运行 `vs check` 查看缺失清单
// ============================================================================

import { chmodSync, createWriteStream, existsSync, mkdirSync, renameSync, rmSync, statSync } from "node:fs";
import { get } from "node:https";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { homedir } from "node:os";

const PKG_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const RELEASE_DIR = join(PKG_ROOT, "rust", "target", "release");

const OWNER = "Moritz230127";
const REPO = "pi-vision-struct";
const TAG = "v4.0.0";

// Release v4.0.0 资产大小（字节），与本地 cargo build 产物一致（已实测核对）
const BINARIES = [
  {
    name: "vs-bin",
    size: 60563792,
    url: `https://github.com/${OWNER}/${REPO}/releases/download/${TAG}/vs-bin`,
  },
  {
    name: "vsd",
    size: 35591064,
    url: `https://github.com/${OWNER}/${REPO}/releases/download/${TAG}/vsd`,
  },
  {
    name: "vs-mcp",
    size: 63204136,
    url: `https://github.com/${OWNER}/${REPO}/releases/download/${TAG}/vs-mcp`,
  },
];

// rapidocr 模型（modelscope 直连，OCR 传感器必需）
// 目标目录 ~/.cache/vsensor/rapidocr/（Rust 二进制硬编码读此路径）
const MODEL_DIR = join(homedir(), ".cache", "vsensor", "rapidocr");
const MODELS = [
  {
    name: "PP-OCRv6_det_small.onnx",
    url: "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.0/onnx/PP-OCRv6/det/PP-OCRv6_det_small.onnx",
    size: 0, // 0 = 不校验大小（modelscope 不保证 Content-Length 稳定）
  },
  {
    name: "PP-OCRv6_rec_small.onnx",
    url: "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.0/onnx/PP-OCRv6/rec/PP-OCRv6_rec_small.onnx",
    size: 0,
  },
  {
    name: "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
    url: "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.0/onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx",
    size: 0,
  },
];

const MAX_ATTEMPTS = 4; // 首次尝试 + 3 次重试
const RETRY_DELAY_MS = 1000;
const MAX_REDIRECTS = 5; // GitHub 会 302 到 release-assets.githubusercontent.com
const REQUEST_TIMEOUT_MS = 30_000;

function log(...args) {
  console.log("[install.js]", ...args);
}

function platformSupported() {
  return process.platform === "linux" && process.arch === "x64";
}

// 下载单个文件到 destPath（先写 .tmp 再原子 rename），校验字节数与预期大小一致
// expectedSize=0 时跳过大小校验（modelscope 不保证 Content-Length 稳定）
function downloadFile(url, destPath, expectedSize, redirectsLeft = MAX_REDIRECTS) {
  return new Promise((resolvePromise, rejectPromise) => {
    const req = get(url, (res) => {
      // 跟随重定向（GitHub Release → release-assets）
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        res.resume();
        if (redirectsLeft <= 0) {
          rejectPromise(new Error(`重定向次数过多: ${url}`));
          return;
        }
        resolvePromise(downloadFile(new URL(res.headers.location, url).href, destPath, expectedSize, redirectsLeft - 1));
        return;
      }
      if (res.statusCode !== 200) {
        res.resume();
        rejectPromise(new Error(`HTTP ${res.statusCode}: ${url}`));
        return;
      }
      const declared = Number(res.headers["content-length"]);
      if (expectedSize > 0 && Number.isFinite(declared) && declared > 0 && declared !== expectedSize) {
        res.resume();
        rejectPromise(new Error(`Content-Length ${declared} 与预期 ${expectedSize} 不符: ${url}`));
        return;
      }
      const tmp = `${destPath}.tmp`;
      const out = createWriteStream(tmp);
      let received = 0;
      res.on("data", (chunk) => {
        received += chunk.length;
      });
      res.pipe(out);
      out.on("finish", () => {
        out.close(() => {
          if (expectedSize > 0 && received !== expectedSize) {
            rmSync(tmp, { force: true });
            rejectPromise(new Error(`大小不符: 收到 ${received} 字节, 预期 ${expectedSize} 字节`));
            return;
          }
          renameSync(tmp, destPath);
          chmodSync(destPath, 0o755);
          resolvePromise(received);
        });
      });
      out.on("error", (err) => {
        rmSync(tmp, { force: true });
        rejectPromise(err);
      });
      res.on("error", (err) => {
        rmSync(tmp, { force: true });
        rejectPromise(err);
      });
    });
    req.setTimeout(REQUEST_TIMEOUT_MS, () => {
      req.destroy(new Error(`下载超时 (${REQUEST_TIMEOUT_MS / 1000}s): ${url}`));
    });
    req.on("error", (err) => {
      rmSync(`${destPath}.tmp`, { force: true });
      rejectPromise(err);
    });
  });
}

// 下载一组文件（幂等：已存在且大小正确则跳过）
async function downloadAll(items, destDir, label) {
  mkdirSync(destDir, { recursive: true });
  let failed = false;
  for (const item of items) {
    const dest = join(destDir, item.name);
    if (existsSync(dest) && (item.size === 0 || statSync(dest).size === item.size)) {
      log(`${label} ${item.name}: 已存在，跳过下载`);
      continue;
    }
    let ok = false;
    for (let attempt = 1; attempt <= MAX_ATTEMPTS && !ok; attempt++) {
      try {
        log(`${label} ${item.name}: 下载中（尝试 ${attempt}/${MAX_ATTEMPTS}）${item.url}`);
        const received = await downloadFile(item.url, dest, item.size);
        log(`${label} ${item.name}: 完成，${received} 字节`);
        ok = true;
      } catch (err) {
        log(`${label} ${item.name}: 尝试 ${attempt}/${MAX_ATTEMPTS} 失败: ${err.message}`);
        if (attempt < MAX_ATTEMPTS) {
          await new Promise((r) => setTimeout(r, RETRY_DELAY_MS * attempt));
        }
      }
    }
    if (!ok) {
      failed = true;
      log(`✗ ${label} ${item.name} 下载失败。请手动下载并放到 ${dest}:`);
      log(`  curl -L -o ${dest} ${item.url}`);
    }
  }
  return !failed;
}

async function main() {
  if (!platformSupported()) {
    log(`跳过: 仅支持 Linux x86_64（当前 ${process.platform}/${process.arch}），请手动获取二进制`);
    return;
  }

  // 阶段 1：二进制（GitHub Release）
  const binOk = await downloadAll(BINARIES, RELEASE_DIR, "二进制");

  // 阶段 2：rapidocr 模型（modelscope，OCR 必需）
  const modelOk = await downloadAll(MODELS, MODEL_DIR, "模型");

  if (!binOk) {
    log("安装失败: 二进制下载未完成");
    process.exitCode = 1;
  } else if (!modelOk) {
    log("二进制就绪，但部分模型下载失败（OCR 传感器可能不可用）");
    log("提示: 其余模型（u2net/midas/mobile_sam/florence2/owlv2 等）见 README 模型清单");
    process.exitCode = 1;
  } else {
    log("完成: 3 个二进制 + rapidocr 模型已就绪");
    log("提示: 其余模型（u2net/midas/mobile_sam/florence2/owlv2/doclayout/clip）");
    log("      需 PyTorch 导出或 HuggingFace 下载，见 README「模型清单」");
    log("      运行 `vs check` 查看缺失清单");
  }
}

main();
