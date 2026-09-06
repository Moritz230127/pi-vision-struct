#!/usr/bin/env python3
"""vs_segment_decoder.py — MobileSAM mask decoder 助手（Rust segment 传感器回退）。

Rust 侧计算 image encoder embeddings（ONNX），本脚本加载 embeddings +
mobile_sam 模型，执行 prompt encoder + mask decoder + 后处理，输出 masks。

协议（行 JSON）:
  输入: {"embeddings": "/tmp/emb.npy", "image": "path.png",
         "points": [[x, y], ...], "labels": [1, ...],
         "boxes": [[x1,y1,x2,y2], ...]}
  输出: {"ok": true, "masks": "/tmp/masks.npy", "scores": [0.9, ...]}
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

WEIGHTS = f"{Path.home()}/.cache/vsensor/mobile_sam.pt"


def main() -> int:
    req = json.loads(sys.stdin.read())
    try:
        from mobile_sam import sam_model_registry  # type: ignore[import-not-found]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = sam_model_registry["vit_t"](checkpoint=WEIGHTS)
        model.to(device)
        model.eval()

        # 加载 Rust 计算的 embeddings (1, 256, 64, 64)
        embeddings = torch.from_numpy(np.load(req["embeddings"])).to(device)
        image_pe = model.prompt_encoder.get_dense_pe()

        # prompt 编码：对每个点/框单独 predict（对应 vs_segment.py 逐 prompt 循环）
        points = req.get("points") or []
        labels = req.get("labels") or []
        boxes = req.get("boxes") or []
        # 坐标缩放到 1024 空间（对应 SamPredictor.apply_coords）
        from PIL import Image
        im = Image.open(req["image"]).convert("RGB")
        ow, oh = im.size
        scale = 1024.0 / max(oh, ow)
        newh = int(oh * scale + 0.5)
        neww = int(ow * scale + 0.5)
        all_masks = []
        all_scores = []
        for pi in range(max(len(points), len(boxes))):
            if points:
                raw = points[pi]
                sx = raw[0] * (neww / ow)
                sy = raw[1] * (newh / oh)
                pt = torch.tensor([[[sx, sy]]], dtype=torch.float32, device=device)
                lb = torch.tensor([[labels[pi] if pi < len(labels) else 1]], dtype=torch.int64, device=device)
                sparse, dense = model.prompt_encoder(points=(pt, lb), boxes=None, masks=None)
            elif boxes:
                raw = boxes[pi]
                bx = torch.tensor([[[
                    raw[0] * (neww / ow), raw[1] * (newh / oh),
                    raw[2] * (neww / ow), raw[3] * (newh / oh),
                ]]], dtype=torch.float32, device=device)
                sparse, dense = model.prompt_encoder(points=None, boxes=bx, masks=None)
            else:
                # 无提示 → 全图中心点
                pt = torch.tensor([[[512.0, 512.0]]], dtype=torch.float32, device=device)
                lb = torch.tensor([[1]], dtype=torch.int64, device=device)
                sparse, dense = model.prompt_encoder(points=(pt, lb), boxes=None, masks=None)
            # mask 解码（multimask_output=False → 单 mask）
            with torch.no_grad():
                masks, iou_pred = model.mask_decoder(
                    embeddings, image_pe, sparse, dense, multimask_output=False
                )
                # 后处理到原图尺寸（input_size = resize 后尺寸，非 1024×1024）
                w, h = im.size
                input_size = (newh, neww)
                original_size = (h, w)
                masks = model.postprocess_masks(masks, input_size, original_size)
                masks = (masks > 0.0).cpu().numpy()  # (1, 1, H, W) bool
                all_masks.append(masks[0, 0])
                all_scores.append(float(iou_pred.cpu().numpy().flatten()[0]))

        out_path = req.get("out_masks", "/tmp/vsd_masks.npy")
        np.save(out_path, np.stack(all_masks))
        print(json.dumps({"ok": True, "masks": out_path, "scores": all_scores}))
        return 0
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(json.dumps({"ok": False, "error": str(e)[:500]}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
