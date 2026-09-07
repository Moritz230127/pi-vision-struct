#!/usr/bin/env python3
"""生成 rapidocr-core db_postprocess 差分测试 fixture。

用 Python rapidocr 的 det 模型（onnxruntime 直连）+ DetPreProcess + DBPostProcess 生成：
  rust/fixtures/db_postprocess/<name>/pred.npy   （det 原始输出 4D f32）
  rust/fixtures/db_postprocess/<name>/expected.json（boxes/scores/dest_shape/tolerances）

参数与 Rust 测试 DbPostProcessConfig 一致：
  thresh=0.3, box_thresh=0.5, max_candidates=1000, unclip_ratio=1.6, min_size=3
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/Arch/conda-envs/pi-vision/lib/python3.12/site-packages")

import onnxruntime as ort  # noqa: E402
from rapidocr.ch_ppocr_det.utils import DBPostProcess, DetPreProcess  # noqa: E402

ROOT = Path("/home/Arch/Pi工作区/pi-vision-struct-v4")
OUT = ROOT / "rust" / "fixtures" / "db_postprocess"
SAMPLES = ROOT / "bench" / "samples"
DET_MODEL = Path.home() / ".cache" / "vsensor" / "rapidocr" / "PP-OCRv6_det_small.onnx"

# 与 Rust 测试一致的 DB 参数
DB_CFG = dict(thresh=0.3, box_thresh=0.5, max_candidates=1000, unclip_ratio=1.6)


def gen(name: str, img_path: Path) -> None:
    import cv2

    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"cannot read {img_path}")
    h, w = img.shape[:2]

    # 预处理（与 TextDetector 内部一致）
    pre = DetPreProcess(limit_side_len=736, limit_type="min", mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    tensor = pre(img)
    if tensor is None:
        raise RuntimeError("preprocess failed")

    # det 推理
    sess = ort.InferenceSession(str(DET_MODEL), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    pred = sess.run(None, {input_name: tensor})[0]
    print(f"[{name}] pred shape={pred.shape}")

    # DB postprocess（Python 基线）
    db = DBPostProcess(**DB_CFG)
    boxes, scores = db(pred, (h, w))
    print(f"[{name}] boxes={len(boxes)} scores={[round(s, 4) for s in scores]}")

    # 保存
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "pred.npy", pred.astype(np.float32))
    expected = {
        "dest_shape": [h, w],
        "boxes": [[[float(p[0]), float(p[1])] for p in b] for b in boxes],
        "scores": [float(s) for s in scores],
        "tolerances": {
            "max_mean_center_delta": 2.0,
            "max_mean_score_delta": 0.15,
            "max_mean_corner_delta": 3.0,
            "max_mean_size_delta": 4.0,
        },
    }
    (d / "expected.json").write_text(json.dumps(expected, indent=2))
    print(f"[{name}] saved -> {d}")


if __name__ == "__main__":
    for name in ["s03_alignment", "s04_spacing"]:
        gen(name, SAMPLES / f"{name}.png")
