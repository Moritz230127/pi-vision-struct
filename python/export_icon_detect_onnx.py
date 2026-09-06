#!/usr/bin/env python3
"""export_icon_detect_onnx.py — OmniParser YOLOv9 icon_detect ONNX 导出（S6 前置）。

icon_detect_v3/model.pt 是 TorchScript 格式（OmniParser 自定义 YOLOv9Detector）。
导出原始输出（未解码）：输入 (1,3,640,640) → 3 层 logits（stride 8/16/32）。
解码（sigmoid + anchor + NMS）在推理侧做（与 YOLOv9Detector._decode 一致）。

用法: omniparser/bin/python python/export_icon_detect_onnx.py
输出: ~/.cache/vsensor/icon_detect.onnx
"""
import sys
from pathlib import Path

import torch

WEIGHTS = Path.home() / ".cache" / "omniparser" / "weights" / "icon_detect_v3" / "model.pt"
OUT = Path.home() / ".cache" / "vsensor" / "icon_detect.onnx"


def main() -> int:
    print(f"[export] 加载 TorchScript from {WEIGHTS} ...", flush=True)
    model = torch.jit.load(str(WEIGHTS), map_location="cpu").eval()

    # 测试输入
    dummy = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        out = model(dummy)
    if isinstance(out, (list, tuple)):
        shapes = [tuple(o.shape) for o in out]
    else:
        shapes = [tuple(out.shape)]
    print(f"[export] 原始输出: {shapes}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy,),
        str(OUT),
        input_names=["images"],
        output_names=[f"out{i}" for i in range(len(shapes))],
        opset_version=14,
        dynamo=False,
    )
    print(f"[export] → {OUT}", flush=True)

    # 验证：ONNX vs TorchScript 输出一致
    import onnxruntime as ort
    import numpy as np

    sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"images": dummy.numpy()})
    with torch.no_grad():
        pt_out = model(dummy)
    if not isinstance(pt_out, (list, tuple)):
        pt_out = [pt_out]
    for i, (o, p) in enumerate(zip(onnx_out, pt_out)):
        diff = np.abs(o - p.numpy()).max()
        print(f"[export] out{i}: ONNX {o.shape} vs PT {tuple(p.shape)} max_diff={diff:.6f}", flush=True)
        assert diff < 1e-4, f"out{i} 不一致: {diff}"
    print("[export] 导出完成 ✓", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
