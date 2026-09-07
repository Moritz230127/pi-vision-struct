#!/usr/bin/env python3
"""export_clip_onnx.py — open_clip ViT-B-32（laion2b_s34b_b79k）图像编码器 → ONNX。

权重: ~/.cache/clip/ViT-B-32.pt（open_clip 缓存，laion2b_s34b_b79k）
输出: ~/.cache/vsensor/clip_vitb32.onnx（pixel_values (1,3,224,224) → image_features (1,512)）

预处理（open_clip 标准）: Resize(224, BICUBIC) + CenterCrop(224) +
Normalize(mean=[0.48145466,0.4578275,0.40821073], std=[0.26862954,0.26130258,0.27577711])
"""
import sys
from pathlib import Path

import torch

HOME = Path.home()
OUT = HOME / ".cache" / "vsensor" / "clip_vitb32.onnx"


def main() -> int:
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k")
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    try:
        torch.onnx.export(
            model,
            dummy,
            str(OUT),
            input_names=["pixel_values"],
            output_names=["image_features"],
            dynamic_axes={"pixel_values": {0: "batch"}, "image_features": {0: "batch"}},
            opset_version=17,
        )
    except Exception as e:
        print(f"导出失败: {e}", file=sys.stderr)
        return 1
    print(f"导出成功: {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
