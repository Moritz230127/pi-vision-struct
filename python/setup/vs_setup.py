#!/usr/bin/env python3
"""vs_setup.py — pi-vision-struct 引导安装器（Phase 2.4，纯 stdlib）。

在干净机器上完成：conda env 检测/创建 → 核心依赖 → （可选）playwright firefox
→ （可选 --with-omniparser）OmniParser env + 锁定依赖 → 自测。

模式：
  默认      完整核心安装（pi-vision env + requirements + 自测）
  --check   只读健康检查（不改变任何东西）
  --dry-run 打印将执行的命令，不执行
  --with-omniparser 附加 OmniParser env 安装（模型下载与 config 补丁见 docs/omniparser-setup.md）
  --with-dom 附加 playwright firefox 下载（dom_dump 需要）
  --proxy http://127.0.0.1:10808  下载走代理（可选）
  --envs-dir /path   覆盖 conda envs 目录（默认 $HOME/conda-envs，不可写时自动尝试）

输出：JSON {ok, steps:[...], envs:{...}, tests:{...}}
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

IS_WIN = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"
BIN_DIR = "Scripts" if IS_WIN else "bin"
PY_EXE = "python.exe" if IS_WIN else "python"

HOME = Path.home()
PKGS = HOME / "conda-pkgs"
ROOT = Path(__file__).resolve().parent.parent.parent  # 包根（pi-vision-struct/）
REQ = ROOT / "python" / "requirements.txt"
REQ_OMNI = ROOT / "python" / "requirements-omniparser.txt"
SELF_TESTS = ROOT / "tests" / "run_self_tests.py"

# 平台默认 envs 目录：Linux 沿用 ~/conda-envs；macOS/Windows 用 conda 自带 envs
DEFAULT_ENVS = HOME / "conda-envs"
if IS_MAC:
    DEFAULT_ENVS = HOME / "miniforge3" / "envs"
if IS_WIN:
    DEFAULT_ENVS = Path(os.environ.get("USERPROFILE", str(HOME))) / "miniforge3" / "envs"

CONFIG_DIR = Path(os.environ.get("APPDATA", str(HOME / ".config"))) if IS_WIN else HOME / ".config"
CONFIG_FILE = CONFIG_DIR / "pi-vision-struct.json"


def _conda_bin(name: str) -> str:
    return f"{name}.bat" if IS_WIN else name


MAMBA_LOCATIONS = [
    "/opt/miniforge/condabin/mamba", "/opt/miniconda3/condabin/mamba",
    str(HOME / "miniforge3" / "condabin" / "mamba"),
    str(HOME / "miniconda3" / "condabin" / "mamba"),
    str(HOME / "mambaforge" / "condabin" / "mamba"),
]
CONDA_LOCATIONS = [p.replace("mamba", "conda") for p in MAMBA_LOCATIONS]
if IS_WIN:
    CONDA_LOCATIONS += [
        str(HOME / "miniforge3" / "condabin" / "conda.bat"),
        str(HOME / "miniconda3" / "condabin" / "conda.bat"),
    ]


def find_conda() -> str:
    for cmd in (_conda_bin("mamba"), _conda_bin("conda")):
        p = shutil.which(cmd)
        if p:
            return p
    for loc in MAMBA_LOCATIONS + CONDA_LOCATIONS:
        if os.path.isfile(loc):
            return loc
    raise RuntimeError("未找到 mamba/conda（安装 miniforge: https://github.com/conda-forge/miniforge）")


def env_python(envs: Path, name: str) -> Path:
    return envs / name / BIN_DIR / PY_EXE


def detect_capture() -> str:
    """探测可用截图后端：grim(linux-wayland) > screencapture(mac) > mss"""
    if IS_MAC and shutil.which("screencapture"):
        return "screencapture"
    if not IS_WIN and shutil.which("grim"):
        return "grim"
    try:
        import mss  # type: ignore[import-not-found]

        return "mss"
    except ImportError:
        return "none"


def write_config(pi_py: Path, omni_py: Path, envs: Path) -> None:
    """写入跨平台配置（扩展启动时读取）。"""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = {
        "pi_vision_python": str(pi_py),
        "omniparser_python": str(omni_py),
        "envs_dir": str(envs),
        "sandbox_enabled": (not IS_WIN and shutil.which("bwrap") is not None),
        "capture_backend": detect_capture(),
        "installed_by": "vs_setup.py",
    }
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  配置已写入: {CONFIG_FILE}")


def base_env(envs_dir: Path) -> dict:
    """envs 目录不可写（如 /opt/miniforge 属 root）时，改用用户目录。"""
    env = dict(os.environ)
    env["CONDA_ENVS_PATH"] = str(envs_dir)
    env["CONDA_PKGS_DIRS"] = str(PKGS)
    return env


def run(cmd: list[str], env: dict, dry: bool, timeout: int = 1800) -> tuple[int, str]:
    if dry:
        print(f"  [DRY] {' '.join(cmd)}")
        return 0, ""
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stderr or r.stdout)[-1200:]
    except subprocess.TimeoutExpired as te:
        return -1, f"超时 {te}"
    except OSError as oe:
        return -1, f"执行失败 {oe}"


def step_result(name: str, ok: bool, detail: str = "") -> dict:
    return {"name": name, "status": "ok" if ok else "fail", "detail": detail[:500]}


def pip_env(env: dict, proxy: str | None) -> dict:
    if proxy:
        env = dict(env)
        env["HTTPS_PROXY"] = proxy
        env["HTTP_PROXY"] = proxy
    return env


def check_env(py: Path, mod: str) -> bool:
    if not py.exists():
        return False
    r = subprocess.run([str(py), "-c", f"import {mod}"], capture_output=True, text=True, timeout=60)
    return r.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--with-omniparser", action="store_true")
    ap.add_argument("--with-dom", action="store_true")
    ap.add_argument("--proxy")
    ap.add_argument("--envs-dir", default=str(DEFAULT_ENVS))
    ap.add_argument("--python", default="3.12")
    args = ap.parse_args()

    steps: list[dict] = []
    try:
        conda = find_conda()
    except RuntimeError as e:
        print(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False))
        return 1

    envs = Path(args.envs_dir)
    py = env_python(envs, "pi-vision")
    omni_py = env_python(envs, "omniparser")
    env = base_env(envs)

    if args.check:
        # ---- 只读深度健康检查（env + 模型 + 沙箱 + Ollama + daemon）----
        core_ok = all(check_env(py, m) for m in ("PIL", "numpy", "onnxruntime", "rapidocr", "pptx"))
        omni_ok = all(check_env(omni_py, m) for m in ("torch", "transformers", "ultralytics"))

        def exists(p: Path) -> bool:
            return p.exists()

        omni_home = Path.home() / ".cache" / "omniparser"
        models = {
            "omniparser_weights": exists(omni_home / "weights" / "icon_detect_v3" / "model.pt")
            and exists(omni_home / "weights" / "icon_caption_florence"),
            "florence_hf_cache": exists(Path.home() / ".cache" / "huggingface" / "hub"
                                         / "models--microsoft--Florence-2-base"),
            "clip_hf_cache": exists(Path.home() / ".cache" / "huggingface" / "hub"
                                     / "models--laion--CLIP-ViT-B-32-laion2B-s34B-b79K"),
        }
        daemon_sock = exists(omni_home / "omniserver.sock")
        bwrap_ok = shutil.which("bwrap") is not None
        ollama_ok = False
        try:
            import urllib.request

            with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2) as r:
                ollama_ok = "qwen3-vl:8b" in r.read().decode("utf-8", "ignore")
        except Exception:
            ollama_ok = False
        report = {
            "ok": True, "mode": "check",
            "conda": conda, "envs_dir": str(envs),
            "envs": {
                "pi-vision": {"exists": py.exists(), "complete": core_ok},
                "omniparser": {"exists": omni_py.exists(), "complete": omni_ok},
            },
            "models": models,
            "sandbox": {"bwrap": bwrap_ok},
            "ollama": {"qwen3_vl": ollama_ok},
            "daemon": {"socket": daemon_sock},
            "steps": [{"name": "health_check", "status": "ok",
                        "detail": f"core={'完整' if core_ok else '缺失'} omni={'完整' if omni_ok else '缺失'}"}],
        }
        print(json.dumps(report, ensure_ascii=False))
        return 0

    # ---- 核心：pi-vision env ----
    if not py.exists():
        steps.append(step_result("create env pi-vision", False))
        rc, err = run([conda, "create", "-y", "-n", "pi-vision", f"python={args.python}"],
                      env, args.dry_run)
        steps[-1] = step_result("create env pi-vision", rc == 0, err)
        if rc != 0:
            print(json.dumps({"ok": False, "conda": conda, "steps": steps}, ensure_ascii=False))
            return 1
    else:
        steps.append(step_result("env pi-vision 已存在", True, str(py)))

    rc, err = run([str(py), "-m", "pip", "install", "-q", "-r", str(REQ)],
                  pip_env(env, args.proxy), args.dry_run, timeout=3600)
    steps.append(step_result("install core deps", rc == 0, err))

    if args.with_dom:
        rc, err = run([str(py), "-m", "playwright", "install", "firefox"],
                      pip_env(env, args.proxy), args.dry_run, timeout=3600)
        steps.append(step_result("playwright firefox", rc == 0, err))

    # ---- 自测 ----
    if not args.dry_run and SELF_TESTS.exists() and py.exists():
        rc, err = run([str(py), "-u", str(SELF_TESTS)], env, False, timeout=1200)
        ok = rc == 0 and "0 失败" in err
        steps.append(step_result("self-tests (17)", ok, err[-300:]))
    else:
        steps.append(step_result("self-tests (17)", True, "dry-run 跳过"))

    # ---- 可选：OmniParser env ----
    if args.with_omniparser:
        if not omni_py.exists():
            rc, err = run([conda, "create", "-y", "-n", "omniparser", "python=3.12"],
                          env, args.dry_run)
            steps.append(step_result("create env omniparser", rc == 0, err))
        else:
            steps.append(step_result("env omniparser 已存在", True))
        rc, err = run([str(omni_py), "-m", "pip", "install", "-q", "torch", "torchvision",
                       "--index-url", "https://download.pytorch.org/whl/cpu"],
                      pip_env(env, args.proxy), args.dry_run, timeout=3600)
        steps.append(step_result("install torch (cpu)", rc == 0, err))
        rc, err = run([str(omni_py), "-m", "pip", "install", "-q", "-r", str(REQ_OMNI)],
                      pip_env(env, args.proxy), args.dry_run, timeout=3600)
        steps.append(step_result("install omniparser deps", rc == 0, err))
        if not args.dry_run:
            steps.append(step_result(
                "omniparser 模型与补丁", False,
                "需手动：运行 python/setup/repair_omniparser.sh 下载权重（~/.cache/omniparser），见 docs/omniparser-setup.md"))

    ok = all(s["status"] == "ok" for s in steps)
    if ok and not args.dry_run:
        write_config(py, omni_py, envs)
    print(json.dumps({
        "ok": ok, "mode": "setup",
        "conda": conda, "envs_dir": str(envs), "dry_run": args.dry_run,
        "config_file": str(CONFIG_FILE) if ok and not args.dry_run else None,
        "steps": steps,
        "next": ["pi install npm:pi-vision-struct 后重启 pi，工具即可用"]
        if ok else ["修复失败步骤后重跑 /vs setup"],
    }, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
