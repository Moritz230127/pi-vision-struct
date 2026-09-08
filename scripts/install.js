#!/usr/bin/env node
// postinstall: 从 GitHub Release v4.0.0 下载 3 个 Rust 二进制到 <pkg_root>/rust/target/release/
// （与 extensions/index.ts 的 VS_BIN 路径一致）。
// 无外部依赖（Node 内置 https），首次尝试 + 3 次重试，校验大小，仅 Linux x86_64。

import { chmodSync, createWriteStream, existsSync, mkdirSync, renameSync, rmSync, statSync } from "node:fs";
import { get } from "node:https";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const PKG_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const RELEASE_DIR = join(PKG_ROOT, "rust", "target", "release");

const OWNER = "Moritz230127";
const REPO = "pi-vision-struct";
const TAG = "v4.0.0";

// Release v4.0.0 资产大小（字节），与本地 cargo build 产物一致（已实测核对）
const BINARIES = [
  { name: "vs-bin", size: 60563792 },
  { name: "vsd", size: 35591064 },
  { name: "vs-mcp", size: 63204136 },
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
      if (Number.isFinite(declared) && declared > 0 && declared !== expectedSize) {
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
          if (received !== expectedSize) {
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

async function main() {
  if (!platformSupported()) {
    log(`跳过: 仅支持 Linux x86_64（当前 ${process.platform}/${process.arch}），请手动获取二进制`);
    return;
  }
  mkdirSync(RELEASE_DIR, { recursive: true });

  let failed = false;
  for (const bin of BINARIES) {
    const dest = join(RELEASE_DIR, bin.name);
    const url = `https://github.com/${OWNER}/${REPO}/releases/download/${TAG}/${bin.name}`;

    if (existsSync(dest) && statSync(dest).size === bin.size) {
      log(`${bin.name}: 已存在且大小正确（${bin.size} 字节），跳过下载`);
      continue;
    }

    let ok = false;
    for (let attempt = 1; attempt <= MAX_ATTEMPTS && !ok; attempt++) {
      try {
        log(`${bin.name}: 下载中（尝试 ${attempt}/${MAX_ATTEMPTS}）${url}`);
        const received = await downloadFile(url, dest, bin.size);
        log(`${bin.name}: 完成，${received} 字节`);
        ok = true;
      } catch (err) {
        log(`${bin.name}: 尝试 ${attempt}/${MAX_ATTEMPTS} 失败: ${err.message}`);
        if (attempt < MAX_ATTEMPTS) {
          await new Promise((r) => setTimeout(r, RETRY_DELAY_MS * attempt));
        }
      }
    }
    if (!ok) {
      failed = true;
      log(`✗ ${bin.name} 下载失败。请手动下载并放到 ${dest}:`);
      log(`  curl -L -o ${dest} ${url}`);
    }
  }

  if (failed) {
    log("安装失败: 二进制下载未完成");
    process.exitCode = 1;
  } else {
    log("完成: 3 个二进制已就绪");
  }
}

main();
