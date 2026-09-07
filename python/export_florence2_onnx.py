#!/usr/bin/env python3
"""export_florence2_onnx.py — Florence-2 拆分 ONNX 导出（S6 前置）。

把 OmniParser 的 Florence-2 caption 模型拆分为 4 个 ONNX（自研生成循环需要）：
  1. vision_encoder.onnx : pixel_values (1,3,64,64) → image_features (1,N,768)
     （DaViT forward_features_unpool + image_pos_embed，图标裁剪 64×64）
  2. embed_tokens.onnx   : input_ids (1,seq) → embeddings (1,seq,768)
  3. language_encoder.onnx: inputs_embeds (1,seq,768) → encoder_outputs (1,seq,768)
  4. decoder.onnx        : decoder_input_ids (1,seq) + encoder_outputs (1,enc,768)
                          → logits (1,seq,vocab)

用法: omniparser/bin/python python/export_florence2_onnx.py
输出: ~/.cache/vsensor/florence2/*.onnx
"""
import json
import os
import sys
from pathlib import Path

import torch

OMNI_DIR = Path.home() / ".cache" / "omniparser"
WEIGHTS = OMNI_DIR / "weights" / "icon_caption_florence"
OUT_DIR = Path.home() / ".cache" / "vsensor" / "florence2"

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_MODULES_CACHE"] = str(OMNI_DIR / "transformers_modules")


def main() -> int:
    from transformers import AutoModelForCausalLM

    print(f"[export] 加载 Florence-2 from {WEIGHTS} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(WEIGHTS), torch_dtype=torch.float32, trust_remote_code=True
    )
    model.eval()
    cfg = model.config
    d_model = cfg.text_config.d_model
    vocab = cfg.vocab_size
    print(f"[export] d_model={d_model} vocab={vocab} "
          f"enc={cfg.text_config.encoder_layers} dec={cfg.text_config.decoder_layers}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    opset = 17

    # ---- 1) vision_encoder: pixel_values → image_features（完整 _encode_image 复刻）----
    class VisionEncoder(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.vision_tower = m.vision_tower
            self.image_pos_embed = m.image_pos_embed
            self.visual_temporal_embed = m.visual_temporal_embed
            self.image_projection = m.image_projection
            self.image_proj_norm = m.image_proj_norm
            self.image_feature_source = m.image_feature_source

        def forward(self, pixel_values):
            batch_size = pixel_values.shape[0]
            T = 1
            x = self.vision_tower.forward_features_unpool(pixel_values)
            if self.image_pos_embed is not None:
                x = x.view(batch_size * T, -1, x.shape[-1])
                num_tokens = x.shape[-2]
                h = w = int(num_tokens ** 0.5)
                x = x.view(batch_size * T, h, w, x.shape[-1])
                pos_embed = self.image_pos_embed(x)
                x = x + pos_embed
                x = x.view(batch_size, T * h * w, x.shape[-1])
            if self.visual_temporal_embed is not None:
                vt = self.visual_temporal_embed(
                    x.view(batch_size, T, -1, x.shape[-1])[:, :, 0]
                )
                x = x.view(batch_size, T, -1, x.shape[-1]) + vt.view(1, T, 1, x.shape[-1])
            x_feat_dict = {}
            x_feat_dict['spatial_avg_pool'] = x.view(batch_size, T, -1, x.shape[-1]).mean(dim=2)
            x_feat_dict['temporal_avg_pool'] = x.view(batch_size, T, -1, x.shape[-1]).mean(dim=1)
            x = x.view(batch_size, T, -1, x.shape[-1])[:, -1]
            x_feat_dict['last_frame'] = x
            new_x = [x_feat_dict[s] for s in self.image_feature_source]
            x = torch.cat(new_x, dim=1)
            x = x @ self.image_projection
            x = self.image_proj_norm(x)
            return x

    ve = VisionEncoder(model)
    # 测试 768×768 输入（Florence2ImageProcessor size=768，DaViT patch 32 → 576 tokens）
    dummy_px = torch.randn(1, 3, 768, 768)
    with torch.no_grad():
        feat = ve(dummy_px)
    print(f"[export] vision_encoder 输出: {tuple(feat.shape)}", flush=True)
    torch.onnx.export(
        ve, (dummy_px,), str(OUT_DIR / "vision_encoder.onnx"),
        input_names=["pixel_values"], output_names=["image_features"],
        dynamic_axes={"pixel_values": {0: "batch"}, "image_features": {0: "batch"}},
        opset_version=opset,
    )
    print("[export] vision_encoder.onnx ✓", flush=True)

    # ---- 2) embed_tokens: input_ids → embeddings ----
    class EmbedTokens(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.embed = m.language_model.get_input_embeddings()

        def forward(self, input_ids):
            return self.embed(input_ids)

    et = EmbedTokens(model)
    dummy_ids = torch.tensor([[2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        emb = et(dummy_ids)
    print(f"[export] embed_tokens 输出: {tuple(emb.shape)}", flush=True)
    torch.onnx.export(
        et, (dummy_ids,), str(OUT_DIR / "embed_tokens.onnx"),
        input_names=["input_ids"], output_names=["embeddings"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "embeddings": {0: "batch", 1: "seq"}},
        opset_version=opset,
    )
    print("[export] embed_tokens.onnx ✓", flush=True)

    # ---- 3) language_encoder: inputs_embeds → encoder_outputs ----
    class LanguageEncoder(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.encoder = m.language_model.model.encoder

        def forward(self, inputs_embeds):
            out = self.encoder(inputs_embeds=inputs_embeds)
            return out.last_hidden_state

    le = LanguageEncoder(model)
    dummy_emb = torch.randn(1, 13, d_model)
    with torch.no_grad():
        enc_out = le(dummy_emb)
    print(f"[export] language_encoder 输出: {tuple(enc_out.shape)}", flush=True)
    torch.onnx.export(
        le, (dummy_emb,), str(OUT_DIR / "language_encoder.onnx"),
        input_names=["inputs_embeds"], output_names=["encoder_outputs"],
        dynamic_axes={
            "inputs_embeds": {0: "batch", 1: "seq"},
            "encoder_outputs": {0: "batch", 1: "seq"},
        },
        opset_version=opset,
    )
    print("[export] language_encoder.onnx ✓", flush=True)

    # ---- 4) decoder: decoder_input_ids + encoder_outputs → logits ----
    # 固定 decoder seq=20（padding + attention_mask），避免 ONNX 动态 seq 的
    # MatMul buffer 复用问题；生成时用累积 ids + padding。
    MAX_DEC_SEQ = 20

    class Decoder(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.decoder = m.language_model.model.decoder
            self.lm_head = m.language_model.lm_head

        def forward(self, decoder_input_ids, decoder_attention_mask, encoder_outputs):
            out = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_outputs,
            )
            return self.lm_head(out.last_hidden_state)

    dec = Decoder(model)
    dummy_dec_ids = torch.zeros(1, MAX_DEC_SEQ, dtype=torch.long)
    dummy_dec_ids[0, 0] = 2
    dummy_dec_mask = torch.zeros(1, MAX_DEC_SEQ, dtype=torch.long)
    dummy_dec_mask[0, 0] = 1
    with torch.no_grad():
        logits = dec(dummy_dec_ids, dummy_dec_mask, enc_out)
    print(f"[export] decoder 输出: {tuple(logits.shape)}", flush=True)
    torch.onnx.export(
        dec, (dummy_dec_ids, dummy_dec_mask, enc_out), str(OUT_DIR / "decoder.onnx"),
        input_names=["decoder_input_ids", "decoder_attention_mask", "encoder_outputs"],
        output_names=["logits"],
        dynamic_axes={
            "decoder_input_ids": {0: "batch"},
            "decoder_attention_mask": {0: "batch"},
            "encoder_outputs": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=opset,
    )
    print("[export] decoder.onnx ✓", flush=True)

    # ---- 数值验证：ONNX vs PyTorch ----
    import onnxruntime as ort

    sess_v = ort.InferenceSession(str(OUT_DIR / "vision_encoder.onnx"), providers=["CPUExecutionProvider"])
    sess_e = ort.InferenceSession(str(OUT_DIR / "embed_tokens.onnx"), providers=["CPUExecutionProvider"])
    sess_l = ort.InferenceSession(str(OUT_DIR / "language_encoder.onnx"), providers=["CPUExecutionProvider"])
    sess_d = ort.InferenceSession(str(OUT_DIR / "decoder.onnx"), providers=["CPUExecutionProvider"])

    with torch.no_grad():
        # vision
        onnx_feat = sess_v.run(None, {"pixel_values": dummy_px.numpy()})[0]
        torch_feat = ve(dummy_px).numpy()
        v_err = float(abs(onnx_feat - torch_feat).max())
        print(f"[verify] vision_encoder max_err={v_err:.2e}", flush=True)
        # embed
        onnx_emb = sess_e.run(None, {"input_ids": dummy_ids.numpy()})[0]
        torch_emb = et(dummy_ids).numpy()
        e_err = float(abs(onnx_emb - torch_emb).max())
        print(f"[verify] embed_tokens max_err={e_err:.2e}", flush=True)
        # language encoder
        onnx_enc = sess_l.run(None, {"inputs_embeds": dummy_emb.numpy()})[0]
        torch_enc = le(dummy_emb).numpy()
        l_err = float(abs(onnx_enc - torch_enc).max())
        print(f"[verify] language_encoder max_err={l_err:.2e}", flush=True)
        # decoder（固定 seq=20 + mask）
        onnx_logits = sess_d.run(
            None,
            {"decoder_input_ids": dummy_dec_ids.numpy(),
             "decoder_attention_mask": dummy_dec_mask.numpy(),
             "encoder_outputs": enc_out.numpy()},
        )[0]
        torch_logits = dec(dummy_dec_ids, dummy_dec_mask, enc_out).numpy()
        d_err = float(abs(onnx_logits - torch_logits).max())
        print(f"[verify] decoder max_err={d_err:.2e}", flush=True)

    ok = all(e < 1e-3 for e in (v_err, e_err, l_err, d_err))
    print(f"[export] {'全部通过' if ok else '存在超差'}（阈值 1e-3）", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
