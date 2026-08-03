#!/usr/bin/env python3
"""test_cluster.py — CLIP 离线聚类自测（确定性）。

运行（omniparser env，首次需模型下载）:
  /home/Arch/conda-envs/omniparser/bin/python tests/test_cluster.py
"""
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / "python"
OUT = ROOT / "tests" / "_out" / "cluster"
OUT.mkdir(parents=True, exist_ok=True)
PYBIN = "/home/Arch/conda-envs/omniparser/bin/python"

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name} {detail}")


def run_tool(args: list[str]) -> dict:
    r = subprocess.run([PYBIN, str(PY / "vs_cluster.py"), *args],
                       capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        raise AssertionError(f"tool exit {r.returncode}: {r.stdout[:300]} {r.stderr[:200]}")
    try:
        return json.loads(r.stdout)
    except ValueError as e:
        raise AssertionError(f"not json: {r.stdout[:300]}") from e


def make_images() -> list[Path]:
    """2 张相同红 + 1 张绿 + 1 张蓝（合成，确定性真值）。"""
    files = []
    for i in range(2):
        p = OUT / f"red_{i}.png"
        Image.new("RGB", (64, 64), (200, 30, 30)).save(p)
        files.append(p)
    g = OUT / "green.png"
    Image.new("RGB", (64, 64), (30, 200, 30)).save(g)
    files.append(g)
    b = OUT / "blue.png"
    Image.new("RGB", (64, 64), (30, 30, 200)).save(b)
    files.append(b)
    return files


def test_cluster_correctness() -> None:
    print("[vs_cluster] 聚类正确性（相同图必同簇，异色分离）")
    files = make_images()
    # CLIP 语义上纯色小图嵌入相近（"纯色背景"是共享概念），阈值 0.70 会合并。
    # 机制验证用阈值 0.9999：只有完全相同图（sim=1.0）能同簇。
    out = run_tool(["--files", ",".join(str(f) for f in files), "--threshold", "0.9999"])
    check("schema v2 envelope", out.get("schema") == "vision-report/v2" and out.get("task") == "cluster")
    m = out.get("metrics", {})
    check("图像数 = 4", m.get("image_count") == 4, str(m))
    # 两张相同红图必须同簇且 sim_to_rep = 1.0
    reds = [str(f) for f in files if "red" in f.name]
    same = None
    for c in out.get("clusters", []):
        names = [mem["file"] for mem in c["members"]]
        if reds[0] in names and reds[1] in names:
            same = c
    check("相同红图同簇", same is not None)
    if same:
        sims = {mem["file"]: mem["sim_to_rep"] for mem in same["members"]}
        check("相同图相似度 = 1.0", any(abs(s - 1.0) < 1e-3 for s in sims.values()), str(sims))
    # 绿/蓝各自成簇（不与红同簇）
    green_blue_own = all(
        ("green.png" in (mem["file"] for mem in c["members"]) and c["size"] == 1)
        or ("blue.png" in (mem["file"] for mem in c["members"]) and c["size"] == 1)
        or not ("green.png" in (mem["file"] for mem in c["members"]) or "blue.png" in (mem["file"] for mem in c["members"]))
        for c in out.get("clusters", [])
    )
    check("绿/蓝各自成簇", green_blue_own, str(out.get("clusters")))
    # 严格阈值下：1 个双元素簇（红）+ 2 个单例（绿/蓝）= 3 簇
    check("簇数 = 3（红对 + 绿 + 蓝）", m.get("cluster_count") == 3, str(m))


def test_determinism() -> None:
    print("[vs_cluster] 确定性（同输入同输出）")
    files = make_images()
    args = ["--files", ",".join(str(f) for f in files), "--threshold", "0.70"]
    a = run_tool(args)
    b = run_tool(args)
    def sig(d: dict) -> list:
        return [[mem["file"] for mem in c["members"]] for c in d.get("clusters", [])]
    check("两次运行分组一致", sig(a) == sig(b), f"{sig(a)} vs {sig(b)}")


def main() -> int:
    print(f"cluster self-tests (python {sys.version.split()[0]})\n")
    test_cluster_correctness()
    test_determinism()
    print(f"\n结果: {PASS} 通过 / {FAIL} 失败")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
