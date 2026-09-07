#!/usr/bin/env python3
"""export_mobile_sam_decoder_onnx.py — MobileSAM prompt encoder + mask decoder ONNX 导出。

把 segment 传感器的 Python 回退（vs_segment_decoder.py）替换为纯 Rust：
  1. prompt_encoder.onnx : points (1,1,2) + labels (1,1) → sparse (1,1,256) + dense (1,256,64,64)
  2. mask_decoder.onnx    : image_embeddings (1,256,64,64) + image_pe (1,256,64,64)
                          + sparse (1,1,256) + dense (1,256,64,64)
                          → masks (1,1,256,256) + iou_pred (1,1)
  3. postprocess_masks 在 Rust 侧实现（resize + pad 裁剪，SAM 语义）

用法: vsensor/bin/python python/export_mobile_sam_decoder_onnx.py
输出: ~/.cache/vsensor/mobile_sam_decoder.onnx + mobile_sam_prompt_encoder.onnx
"""
import os
from pathlib import Path

import torch

WEIGHTS = Path.home() / ".cache" / "vsensor" / "mobile_sam.pt"
OUT_DIR = Path.home() / ".cache" / "vsensor"


def main() -> int:
    from mobile_sam import sam_model_registry

    model = sam_model_registry["vit_t"](checkpoint=str(WEIGHTS))
    model.eval()
    pe = model.prompt_encoder
    md = model.mask_decoder
    print("[export] MobileSAM 加载完成", flush=True)

    # ---- 1) prompt_encoder: points + labels → sparse + dense + image_pe ----
    class PromptEncoder(torch.nn.Module):
        def __init__(self, pe):
            super().__init__()
            self.pe = pe

        def forward(self, points, labels):
            # 复刻 _embed_points（显式 Where 避免 masked assignment 广播问题）
            points = points + 0.5
            # pad=True（boxes is None）
            padding_point = torch.zeros((points.shape[0], 1, 2), dtype=points.dtype)
            padding_label = -torch.ones((labels.shape[0], 1), dtype=labels.dtype)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
            point_embedding = self.pe.pe_layer.forward_with_coords(points, self.pe.input_image_size)
            # 显式 Where（cond reshape 到 (B,N,1) 正确广播）
            cond = (labels == -1).unsqueeze(-1)
            point_embedding = torch.where(cond, torch.zeros_like(point_embedding), point_embedding)
            point_embedding = point_embedding + torch.where(
                cond, self.pe.not_a_point_embed.weight.squeeze(0).unsqueeze(0).unsqueeze(0),
                torch.zeros_like(point_embedding))
            point_embedding = point_embedding + torch.where(
                (labels == 0).unsqueeze(-1),
                self.pe.point_embeddings[0].weight.squeeze(0).unsqueeze(0).unsqueeze(0),
                torch.zeros_like(point_embedding))
            point_embedding = point_embedding + torch.where(
                (labels == 1).unsqueeze(-1),
                self.pe.point_embeddings[1].weight.squeeze(0).unsqueeze(0).unsqueeze(0),
                torch.zeros_like(point_embedding))
            # dense：no_mask_embed
            dense = self.pe.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                points.shape[0], -1, self.pe.image_embedding_size[0], self.pe.image_embedding_size[1])
            # image_pe：get_dense_pe() = pe_layer(image_embedding_size)
            image_pe = self.pe.pe_layer(self.pe.image_embedding_size).unsqueeze(0)
            return point_embedding, dense, image_pe

    pe_export = PromptEncoder(pe)
    dummy_pts = torch.tensor([[[512.0, 512.0]]], dtype=torch.float32)
    dummy_lbl = torch.tensor([[1]], dtype=torch.int64)
    with torch.no_grad():
        s, d, ip = pe_export(dummy_pts, dummy_lbl)
    print(f"[export] prompt_encoder 输出: sparse {tuple(s.shape)} dense {tuple(d.shape)} image_pe {tuple(ip.shape)}", flush=True)
    torch.onnx.export(
        pe_export, (dummy_pts, dummy_lbl), str(OUT_DIR / "mobile_sam_prompt_encoder.onnx"),
        input_names=["points", "labels"],
        output_names=["sparse_embeddings", "dense_embeddings", "image_pe"],
        dynamic_axes={
            "points": {0: "batch", 1: "num_points"},
            "labels": {0: "batch", 1: "num_points"},
            "sparse_embeddings": {0: "batch", 1: "num_points"},
        },
        opset_version=17,
    )
    print("[export] mobile_sam_prompt_encoder.onnx ✓", flush=True)

    # ---- 2) mask_decoder: embeddings + image_pe + sparse + dense → masks + iou ----
    class MaskDecoder(torch.nn.Module):
        def __init__(self, md):
            super().__init__()
            self.md = md

        def forward(self, image_embeddings, image_pe, sparse_embeddings, dense_embeddings):
            masks, iou_pred = self.md(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            return masks, iou_pred

    md_export = MaskDecoder(md)
    dummy_emb = torch.randn(1, 256, 64, 64)
    dummy_pe = torch.randn(1, 256, 64, 64)
    dummy_sparse = torch.randn(1, 2, 256)  # 1 真实点 + 1 padding（prompt_encoder pad=True）
    dummy_dense = torch.randn(1, 256, 64, 64)
    with torch.no_grad():
        m, i = md_export(dummy_emb, dummy_pe, dummy_sparse, dummy_dense)
    print(f"[export] mask_decoder 输出: masks {tuple(m.shape)} iou {tuple(i.shape)}", flush=True)
    torch.onnx.export(
        md_export, (dummy_emb, dummy_pe, dummy_sparse, dummy_dense),
        str(OUT_DIR / "mobile_sam_mask_decoder.onnx"),
        input_names=["image_embeddings", "image_pe", "sparse_embeddings", "dense_embeddings"],
        output_names=["masks", "iou_pred"],
        opset_version=17,
    )
    print("[export] mobile_sam_mask_decoder.onnx ✓", flush=True)

    # ---- 3) 验证：ONNX 与 torch 输出一致 ----
    import onnxruntime as ort
    import numpy as np

    sess_pe = ort.InferenceSession(str(OUT_DIR / "mobile_sam_prompt_encoder.onnx"), providers=["CPUExecutionProvider"])
    sess_md = ort.InferenceSession(str(OUT_DIR / "mobile_sam_mask_decoder.onnx"), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        s_t, d_t, ip_t = pe_export(dummy_pts, dummy_lbl)
        m_t, i_t = md_export(dummy_emb, ip_t, s_t, d_t)
    s_o, d_o, ip_o = sess_pe.run(None, {"points": dummy_pts.numpy(), "labels": dummy_lbl.numpy()})
    m_o, i_o = sess_md.run(None, {
        "image_embeddings": dummy_emb.numpy(),
        "image_pe": ip_o,
        "sparse_embeddings": s_o,
        "dense_embeddings": d_o,
    })
    print(f"[verify] prompt_encoder max_err={np.abs(s_o - s_t.detach().numpy()).max():.2e} / {np.abs(d_o - d_t.detach().numpy()).max():.2e} / {np.abs(ip_o - ip_t.detach().numpy()).max():.2e}")
    print(f"[verify] mask_decoder max_err={np.abs(m_o - m_t.detach().numpy()).max():.2e} / {np.abs(i_o - i_t.detach().numpy()).max():.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
