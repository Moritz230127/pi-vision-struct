#!/usr/bin/env python3
"""gen_schema_golden.py — 生成 vs_schema.py 的 golden 输出（S0 差分测试基线）。

用法: python3 rust/tests/gen_schema_golden.py > rust/tests/schema_golden.json
输出: 与 Rust vs-core::schema/bbox/color 逐字段对比的基准 JSON。
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
import vs_schema as S  # noqa: E402


def main() -> None:
    golden = {
        # schema 模块
        "SCHEMA": S.SCHEMA,
        "element": S.element(1, "text", [1.0, 2.0, 3.0, 4.0], text="hi", conf=0.9,
                             source=["ocr"], coordsys="image_px"),
        "element_min": S.element(0, "icon", [0, 0, 10, 10]),
        "envelope": S.envelope(task="test", sensors=["pix"], coordsys="css_px"),
        "envelope_full": S.envelope(task="t", sensors=["a", "b"], coordsys="pt",
                                    source={"type": "file", "path": "/tmp/x.png"}),
        "bbox_primitive": S.bbox_primitive([1.0, 2.5, 3.0, 4.5]),
        "point_primitive": S.point_primitive(1.5, 2.5),
        "anomaly": S.anomaly("overlap", [0.0, 0.0, 10.0, 10.0],
                             {"iou": 0.5}, 0.8, "too close"),
        "anomaly_min": S.anomaly("gap", [1.0, 1.0, 2.0, 2.0], {}),
        # bbox 模块
        "css_to_device": S.css_to_device([10.0, 20.0, 110.0, 220.0], 2.0),
        "css_to_device_scroll": S.css_to_device([10.0, 20.0, 110.0, 220.0], 2.0, (5, 5)),
        "device_to_css": S.device_to_css([20.0, 40.0, 220.0, 440.0], 2.0),
        "scale_bbox": S.scale_bbox([1.0, 2.0, 3.0, 4.0], 2.5),
        "clip_bbox": S.clip_bbox([-5.0, -5.0, 200.0, 200.0], 100, 100),
        "bbox_area": S.bbox_area([0.0, 0.0, 10.0, 10.0]),
        "bbox_area_neg": S.bbox_area([5.0, 5.0, 3.0, 3.0]),
        "bbox_intersection": S.bbox_intersection([0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]),
        "bbox_intersection_none": S.bbox_intersection([0.0, 0.0, 1.0, 1.0], [5.0, 5.0, 6.0, 6.0]),
        "bbox_iou_full": S.bbox_iou([0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]),
        "bbox_iou_half": S.bbox_iou([0.0, 0.0, 10.0, 10.0], [0.0, 5.0, 10.0, 10.0]),
        "bbox_iou_none": S.bbox_iou([0.0, 0.0, 1.0, 1.0], [5.0, 5.0, 6.0, 6.0]),
        # color 模块
        "srgb_to_linear_0": S.srgb_to_linear(0.0),
        "srgb_to_linear_255": S.srgb_to_linear(255.0),
        "srgb_to_linear_128": S.srgb_to_linear(128.0),
        "relative_luminance_white": S.relative_luminance((255.0, 255.0, 255.0)),
        "relative_luminance_black": S.relative_luminance((0.0, 0.0, 0.0)),
        "contrast_ratio_bw": S.contrast_ratio((0.0, 0.0, 0.0), (255.0, 255.0, 255.0)),
        "contrast_ratio_same": S.contrast_ratio((100.0, 100.0, 100.0), (100.0, 100.0, 100.0)),
        "rgb_to_lab_white": S.rgb_to_lab((255.0, 255.0, 255.0)),
        "rgb_to_lab_red": S.rgb_to_lab((255.0, 0.0, 0.0)),
        "delta_e76_same": S.delta_e76((10.0, 20.0, 30.0), (10.0, 20.0, 30.0)),
        "delta_e76_bw": S.delta_e76((0.0, 0.0, 0.0), (255.0, 255.0, 255.0)),
        "hex_to_rgb": S.hex_to_rgb("#FF8040"),
        "rgb_to_hex": S.rgb_to_hex((255.0, 128.0, 64.0)),
        "rgb_to_hex_round": S.rgb_to_hex((254.5, 127.5, 63.5)),
        "parse_hex": S.parse_hex("ff8040"),
    }
    print(json.dumps(golden, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
