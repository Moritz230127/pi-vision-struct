#!/usr/bin/env python3
"""export_doclayout_onnx.py — PP-DocLayoutV3 → ONNX（PirTranslatedLayer 路径）。

paddle2onnx 2.1.0 不支持 PIR 静态图，改用 PirTranslatedLayer（paddle.jit.load）：
1. paddle.jit.load 加载 PIR 推理模型
2. 包装 forward（3 输入：im_shape + pixel_values + scale_factor）
3. dygraph2onnx 导出

输出: ~/.cache/vsensor/doclayout.onnx
"""
import sys
from pathlib import Path

import paddle

HOME = Path.home()
MODEL_DIR = HOME / ".paddlex" / "official_models" / "PP-DocLayoutV3"
OUT = HOME / ".cache" / "vsensor" / "doclayout.onnx"


class DocLayoutWrapper(paddle.nn.Layer):
    """包装 PirTranslatedLayer：forward 接受 (im_shape, pixel_values, scale_factor)。"""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, im_shape, pixel_values, scale_factor):
        return self.inner(im_shape, pixel_values, scale_factor)


def main() -> int:
    inner = paddle.jit.load(str(MODEL_DIR / "inference"))
    model = DocLayoutWrapper(inner)
    model.eval()

    # 冒烟：3 输入推理
    x = paddle.randn([1, 3, 800, 800])
    shape = paddle.to_tensor([[800.0, 800.0]])
    scale = paddle.to_tensor([[1.0, 1.0]])
    try:
        out = model(shape, x, scale)
        print("冒烟推理 OK:", type(out))
    except Exception as e:
        print(f"冒烟失败: {e}", file=sys.stderr)
        return 1

    spec = [
        paddle.static.InputSpec(shape=[1, 2], dtype="float32", name="im_shape"),
        paddle.static.InputSpec(shape=[1, 3, 800, 800], dtype="float32", name="pixel_values"),
        paddle.static.InputSpec(shape=[1, 2], dtype="float32", name="scale_factor"),
    ]
    import paddle2onnx

    paddle2onnx.dygraph2onnx(
        model, save_file=str(OUT), input_spec=spec, opset_version=17)
    print(f"导出成功: {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
