#!/usr/bin/env python3
"""vs_cluster.py — CLIP 离线聚类（Phase 2.3，独立 CLI）。

对图片目录/列表做感知相似度分组：
  1. open_clip ViT-B-32（CPU）编码 → L2 归一化嵌入
  2. 余弦相似度矩阵（归一化向量的点积）
  3. 贪心单链分组：按文件名顺序，与已有组代表相似度均值 > 阈值则并入，否则新建组
     （确定性：无随机初始化；同输入同输出）
  4. 每图给出 top-2 相似图作为证据

输出: schema v2 envelope（task=cluster）+ clusters[] + metrics{image_count,
      cluster_count, threshold} + top_pairs[]。

用法:
  vs_cluster.py --dir <图片目录> [--threshold 0.75] [--model ViT-B-32]
                [--pretrained laion2b_s34b_b79k] [--max-files 200]
  vs_cluster.py --files a.png,b.jpg,c.png [...]
首次运行会下载模型权重（约 350MB，需代理）；之后完全离线。
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any

import vs_schema as S


def collect_images(paths: list[str], exts: set[str]) -> list[str]:
    out = []
    for p in paths:
        pth = Path(p)
        if pth.is_dir():
            for f in sorted(pth.iterdir()):
                if f.is_file() and f.suffix.lower() in exts:
                    out.append(str(f))
        elif pth.is_file() and pth.suffix.lower() in exts:
            out.append(str(pth))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--files")
    ap.add_argument("--threshold", type=float, default=0.75)
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--max-files", type=int, default=200)
    ap.add_argument("--ext", action="append", default=[])
    args = ap.parse_args()

    try:
        exts = set((e if e.startswith(".") else f".{e}").lower() for e in (args.ext or [])) or \
            {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
        if args.dir:
            images = collect_images([args.dir], exts)
        elif args.files:
            images = collect_images(args.files.split(","), exts)
        else:
            raise ValueError("需要 --dir 或 --files")
        images = images[: args.max_files]
        if not images:
            print(json.dumps({"error": "vs_cluster failed", "detail": "无匹配图片"}, ensure_ascii=False))
            return 1

        import torch  # type: ignore[import-not-found]
        import open_clip  # type: ignore[import-not-found]

        torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
        model, _, preprocess = open_clip.create_model_and_transforms(
            args.model, pretrained=args.pretrained)
        model.eval()
        tokenizer = open_clip.get_tokenizer(args.model)

        from PIL import Image  # type: ignore[import-not-found]

        embs = []
        for img in images:
            try:
                im = Image.open(img).convert("RGB")
            except Exception:
                im = None
            if im is None:
                continue
            with torch.no_grad():
                v = model.encode_image(preprocess(im).unsqueeze(0).float())
                v = torch.nn.functional.normalize(v, dim=-1)
            embs.append((img, v[0].numpy()))
        if len(embs) < 2:
            print(json.dumps({"error": "vs_cluster failed",
                              "detail": f"可编码图片仅 {len(embs)} 张（至少 2 张才能聚类）"},
                             ensure_ascii=False))
            return 1

        names = [e[0] for e in embs]
        import numpy as np  # type: ignore[import-not-found]

        M = np.stack([e[1] for e in embs])  # (n, d) 已归一化
        sim = M @ M.T

        # 贪心单链分组（确定性：按文件名顺序处理）
        groups: list[list[int]] = []
        for i in range(len(names)):
            best_g, best_sim = -1, -1.0
            for gi, g in enumerate(groups):
                s = float(np.mean(sim[i, g]))
                if s > best_sim:
                    best_sim, best_g = s, gi
            if best_g >= 0 and best_sim > args.threshold:
                groups[best_g].append(i)
            else:
                groups.append([i])

        clusters = []
        for gi, g in enumerate(groups):
            rep = g[0]
            repsim = float(sim[rep, rep])
            members = []
            for i in g:
                members.append({"file": names[i], "sim_to_rep": round(float(sim[i, rep]), 4)})
            clusters.append({
                "id": gi, "size": len(g),
                "representative": names[rep],
                "mean_sim": round(float(np.mean([sim[i, j] for i in g for j in g if i != j] or [1.0])), 4),
                "members": sorted(members, key=lambda m: -m["sim_to_rep"]),
            })

        top_pairs = []
        for i in range(len(names)):
            row = [(j, float(sim[i, j])) for j in range(len(names)) if j != i]
            row.sort(key=lambda t: -t[1])
            for j, s in row[:2]:
                top_pairs.append({"a": names[i], "b": names[j], "sim": round(s, 4)})

        report = S.envelope(task="cluster", sensors=["clip"], coordsys=None,
                            source={"type": "images", "count": len(names),
                                    "model": args.model, "pretrained": args.pretrained,
                                    "threshold": args.threshold})
        report["clusters"] = clusters
        report["top_pairs"] = top_pairs
        report["metrics"] = {
            "image_count": len(names), "cluster_count": len(clusters),
            "singleton_count": sum(1 for g in groups if len(g) == 1),
            "threshold": args.threshold,
            "max_cluster_size": max(len(g) for g in groups),
        }
        print(S.dump_json(report))
        return 0
    except Exception as e:
        print(json.dumps({"error": "vs_cluster failed", "detail": str(e)[:500]}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
