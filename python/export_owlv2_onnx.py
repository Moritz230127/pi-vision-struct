#!/usr/bin/env python3
"""export_owlv2_onnx.py — OWLv2 拆分 ONNX 导出（detect Plan B）。

导出 3 个 ONNX 到 ~/.cache/vsensor/owlv2/：
  text_encoder.onnx    input_ids (1,16) → text_embeds (1,768)
  vision_encoder.onnx   pixel_values (1,3,960,960) → image_feats (1,3600,768)
  detection_head.onnx   image_feats + query_embeds + query_mask
                       → pred_logits (1,3600,Q+1) + pred_boxes (1,3600,4)

验证：ONNX vs PyTorch 输出差 < 1e-3。
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

OUT_DIR = Path.home() / ".cache" / "vsensor" / "owlv2"
MODEL_ID = "google/owlv2-base-patch16-ensemble"


def main() -> int:
    from transformers import Owlv2ForObjectDetection

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = Owlv2ForObjectDetection.from_pretrained(MODEL_ID)
    model.eval()
    cfg = model.config

    # ---- 1) text_encoder: input_ids → text_embeds ----
    class TextEncoder(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.text_model = m.owlv2.text_model
            self.text_projection = m.owlv2.text_projection

        def forward(self, input_ids):
            out = self.text_model(input_ids=input_ids)
            # pooler_output（EOS token 过 pooler）→ (batch, hidden)
            return self.text_projection(out[1])

    te = TextEncoder(model)
    input_ids = torch.randint(1, 100, (1, 16), dtype=torch.long)
    with torch.no_grad():
        ref_te = te(input_ids)
    torch.onnx.export(
        te, (input_ids,), str(OUT_DIR / "text_encoder.onnx"),
        input_names=["input_ids"], output_names=["text_embeds"],
        dynamic_axes={"input_ids": {0: "batch"}, "text_embeds": {0: "batch"}},
        opset_version=17,
    )
    print(f"text_encoder.onnx: ref {tuple(ref_te.shape)}")

    # ---- 2) vision_encoder: pixel_values → image_feats ----
    class VisionEncoder(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.vision_model = m.owlv2.vision_model
            self.layer_norm = m.layer_norm

        def forward(self, pixel_values):
            out = self.vision_model(pixel_values=pixel_values)
            last_hidden = out[0]
            image_embeds = self.vision_model.post_layernorm(last_hidden)
            class_token_out = torch.broadcast_to(
                image_embeds[:, :1, :], image_embeds[:, :-1].shape
            )
            image_embeds = image_embeds[:, 1:, :] * class_token_out
            image_embeds = self.layer_norm(image_embeds)
            # (batch, seq-1, hidden) → (batch, num_patches, hidden)
            # 960/16 = 60 patches（固定，tracing 时 shape 是 symbolic）
            b, s, d = image_embeds.shape
            ph = pw = 60
            return image_embeds.reshape(b, ph, pw, d).reshape(b, ph * pw, d)

    ve = VisionEncoder(model)
    pixel_values = torch.randn(1, 3, 960, 960)
    with torch.no_grad():
        ref_ve = ve(pixel_values)
    torch.onnx.export(
        ve, (pixel_values,), str(OUT_DIR / "vision_encoder.onnx"),
        input_names=["pixel_values"], output_names=["image_feats"],
        dynamic_axes={"pixel_values": {0: "batch"}, "image_feats": {0: "batch"}},
        opset_version=17,
    )
    print(f"vision_encoder.onnx: ref {tuple(ref_ve.shape)}")

    # ---- 3) detection_head: image_feats + query_embeds + query_mask → logits + boxes ----
    class DetectionHead(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.class_head = m.class_head
            self.box_head = m.box_head
            self.box_bias = m.box_bias
            self.sigmoid = m.sigmoid

        def forward(self, image_feats, query_embeds, query_mask):
            pred_logits, _ = self.class_head(image_feats, query_embeds, query_mask)
            pred_boxes = self.box_head(image_feats)
            pred_boxes = pred_boxes + self.box_bias
            pred_boxes = self.sigmoid(pred_boxes)
            return pred_logits, pred_boxes

    dh = DetectionHead(model)
    nq = 2  # 测试用 2 个类别
    image_feats = torch.randn(1, 3600, 768)
    query_embeds = torch.randn(1, nq, 512)  # text_projection 输出 512 维
    query_mask = torch.ones(1, nq, dtype=torch.long)
    with torch.no_grad():
        ref_logits, ref_boxes = dh(image_feats, query_embeds, query_mask)
    torch.onnx.export(
        dh, (image_feats, query_embeds, query_mask),
        str(OUT_DIR / "detection_head.onnx"),
        input_names=["image_feats", "query_embeds", "query_mask"],
        output_names=["pred_logits", "pred_boxes"],
        dynamic_axes={
            "image_feats": {0: "batch"},
            "query_embeds": {0: "batch", 1: "num_queries"},
            "query_mask": {0: "batch", 1: "num_queries"},
            "pred_logits": {0: "batch", 2: "num_queries"},
            "pred_boxes": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"detection_head.onnx: ref logits {tuple(ref_logits.shape)} boxes {tuple(ref_boxes.shape)}")

    # ---- 验证：ONNX vs PyTorch ----
    import onnxruntime as ort

    checks = [
        ("text_encoder.onnx", {"input_ids": input_ids.numpy()}, [ref_te.numpy()]),
        ("vision_encoder.onnx", {"pixel_values": pixel_values.numpy()}, [ref_ve.numpy()]),
        (
            "detection_head.onnx",
            {
                "image_feats": image_feats.numpy(),
                "query_embeds": query_embeds.numpy(),
                "query_mask": query_mask.numpy(),
            },
            [ref_logits.numpy(), ref_boxes.numpy()],
        ),
    ]
    for name, inputs, refs in checks:
        sess = ort.InferenceSession(str(OUT_DIR / name), providers=["CPUExecutionProvider"])
        outs = sess.run(None, inputs)
        for i, (o, r) in enumerate(zip(outs, refs)):
            diff = float(np.abs(o - r).max())
            status = "OK" if diff < 1e-3 else "FAIL"
            print(f"  {name} out{i}: max_diff={diff:.2e} {status}")
            if diff >= 1e-3:
                return 1
    print("导出完成:", OUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
