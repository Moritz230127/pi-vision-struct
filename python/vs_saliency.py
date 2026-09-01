#!/usr/bin/env python3
"""vs_saliency.py — V3 显著性传感器（U²-Net ONNX，GPU/CPU）。

U²-Net 显著性检测 → 显著性图 → top-N 候选区域 bbox + score。
候选区域是 zoom-in 协议的入口（破解"复杂背景不知道看哪里"死锁）。

依赖: vsensor env（onnxruntime + U²-Net ONNX ~/.cache/vsensor/u2net.onnx）
用法:
  vs_saliency.py --image PATH [--top-n 5] [--min-score 0.3]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import vs_schema as S

WEIGHTS = f"{Path.home()}/.cache/vsensor/u2net.onnx"

# CUDA 运行时库路径（torch 自带 nvidia cudnn/cublas，onnxruntime 需要显式找到）
_NVIDIA_LIBS = f"{Path.home()}/conda-envs/vsensor/lib/python3.12/site-packages/nvidia"
for _sub in ("cudnn/lib", "cublas/lib", "cufft/lib", "cusparse/lib"):
    _p = f"{_NVIDIA_LIBS}/{_sub}"
    if os.path.isdir(_p) and _p not in os.environ.get("LD_LIBRARY_PATH", ""):
        os.environ["LD_LIBRARY_PATH"] = _p + ":" + os.environ.get("LD_LIBRARY_PATH", "")

# 显式预加载 CUDA 库（LD_LIBRARY_PATH 对已启动进程的 dlopen 无效，需 ctypes 预加载）
import ctypes
for _lib in ("libcudnn.so.9", "libcublas.so.12", "libcufft.so.11", "libcusparse.so.12"):
    for _sub in ("cudnn/lib", "cublas/lib", "cufft/lib", "cusparse/lib"):
        _cand = f"{_NVIDIA_LIBS}/{_sub}/{_lib}"
        if os.path.exists(_cand):
            try:
                ctypes.CDLL(_cand)
            except OSError:
                pass
            break


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--min-score", type=float, default=0.3)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    try:
        import onnxruntime as ort
        import numpy as np
        from PIL import Image

        # 设备选择
        if args.device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        try:
            sess = ort.InferenceSession(WEIGHTS, providers=providers)
        except Exception:
            sess = ort.InferenceSession(WEIGHTS, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name

        im = Image.open(args.image).convert("RGB")
        w, h = im.size
        # 缩放到 320×320（U²-Net 标准输入）
        im_resized = im.resize((320, 320), Image.LANCZOS)
        arr = np.array(im_resized, dtype=np.float32) / 255.0
        # 归一化（ImageNet 均值/方差）
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        tensor = arr.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

        out = sess.run(None, {input_name: tensor})
        sal = out[0]
        if sal.ndim == 4:
            sal = sal[0]
        if sal.shape[0] == 1:
            sal = sal[0]
        sal_map = np.clip(sal, 0, 1).astype(np.float32)

        # 上采样回原图尺寸
        sal_img = Image.fromarray((sal_map * 255).astype(np.uint8))
        sal_full = np.array(sal_img.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0

        # 连通域提取候选区域
        from scipy import ndimage
        binary = sal_full > args.min_score
        labeled, n = ndimage.label(binary)
        candidates = []
        for i in range(1, n + 1):
            ys, xs = np.nonzero(labeled == i)
            if len(ys) < 50:  # 去噪
                continue
            score = float(sal_full[ys, xs].mean())
            candidates.append({
                "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                "score": round(score, 3),
                "pixels": int(len(ys)),
            })
        candidates.sort(key=lambda c: -c["score"])
        candidates = candidates[: args.top_n]

        elements = []
        for i, c in enumerate(candidates):
            elements.append(S.element(i, "candidate", c["bbox"],
                                      conf=c["score"], source=["saliency"], coordsys="image_px"))
            elements[-1]["saliency"] = c

        report = S.envelope(task="saliency", sensors=["saliency"], coordsys="image_px",
                            source={"type": "image", "path": args.image,
                                    "size_px": [w, h], "device": args.device})
        report["schema"] = "vision-report/v3"
        report["elements"] = elements
        report["candidates"] = candidates
        report["metrics"] = {
            "candidates": len(candidates),
            "top_score": candidates[0]["score"] if candidates else 0.0,
            "device": args.device,
        }
        print(S.dump_json(report))
        return 0
    except Exception as e:
        print(json.dumps({"error": "vs_saliency failed", "detail": str(e)[:500]},
                         ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
