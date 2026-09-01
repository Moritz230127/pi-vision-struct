#!/usr/bin/env python3
"""vs_depth.py — V3 单目深度传感器（MiDaS-small ONNX）。

MiDaS-small（21M 参数）→ 逆深度图 → 相对深度统计（近/中/远分布）。
精密零件空间关系：哪个部件在前、哪个在后、深度梯度。

依赖: vsensor env（onnxruntime + MiDaS ONNX ~/.cache/vsensor/midas_v21_small_256.onnx）
用法:
  vs_depth.py --image PATH [--bins 3] [--region x1,y1,x2,y2]
"""
import argparse
import ctypes
import json
import os
import sys
from pathlib import Path

import vs_schema as S

WEIGHTS = f"{Path.home()}/.cache/vsensor/midas_v21_small_256.onnx"

# CUDA 运行时库预加载（onnxruntime CUDA provider 需要 cudnn/cublas）
_NVIDIA_LIBS = f"{Path.home()}/conda-envs/vsensor/lib/python3.12/site-packages/nvidia"
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
    ap.add_argument("--bins", type=int, default=3, help="深度分位数（近/中/远）")
    ap.add_argument("--region", help="x1,y1,x2,y2 区域深度统计")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    try:
        import onnxruntime as ort
        import numpy as np
        from PIL import Image

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
        # MiDaS 归一化（ImageNet）
        im_resized = im.resize((256, 256), Image.LANCZOS)
        arr = np.array(im_resized, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        tensor = arr.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

        out = sess.run(None, {input_name: tensor})
        depth = out[0][0]  # (256, 256) 逆深度

        # 上采样回原图
        depth_img = Image.fromarray(((depth - depth.min()) / (depth.max() - depth.min() + 1e-9) * 255).astype(np.uint8))
        depth_full = np.array(depth_img.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0

        # 全局深度统计
        def depth_stats(d: np.ndarray) -> dict:
            d = d.astype(np.float32)
            qs = np.quantile(d, [0.25, 0.5, 0.75])
            return {
                "min": round(float(d.min()), 3),
                "max": round(float(d.max()), 3),
                "mean": round(float(d.mean()), 3),
                "q25": round(float(qs[0]), 3),
                "median": round(float(qs[1]), 3),
                "q75": round(float(qs[2]), 3),
            }

        global_stats = depth_stats(depth_full)

        # 深度分箱（近/中/远）
        bins = []
        for i in range(args.bins):
            lo = i / args.bins
            hi = (i + 1) / args.bins
            mask = (depth_full >= lo) & (depth_full < hi)
            bins.append({
                "bin": i, "range": [round(lo, 2), round(hi, 2)],
                "area_ratio": round(float(mask.mean()), 4),
            })

        # 区域深度
        region_stats = None
        if args.region:
            x1, y1, x2, y2 = (int(v) for v in args.region.split(","))
            region = depth_full[y1:y2, x1:x2]
            if region.size > 0:
                region_stats = depth_stats(region)

        report = S.envelope(task="depth", sensors=["depth"], coordsys="image_px",
                            source={"type": "image", "path": args.image,
                                    "size_px": [w, h], "device": args.device})
        report["schema"] = "vision-report/v3"
        report["depth"] = {"global": global_stats, "bins": bins,
                           "region": region_stats, "inverse": True}
        report["metrics"] = {
            "bins": args.bins,
            "depth_range": [round(float(depth_full.min()), 3), round(float(depth_full.max()), 3)],
            "device": args.device,
        }
        print(S.dump_json(report))
        return 0
    except Exception as e:
        print(json.dumps({"error": "vs_depth failed", "detail": str(e)[:500]},
                         ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
