#!/usr/bin/env python3
"""vs_layout.py — 文档版式分析（PP-DocLayoutV3，paddle 全家桶，schema v2）。

把文档/论文/网页截图结构化为语义区域：标题、正文、图表、表格、公式等。
与 OmniParser（UI 图标）互补：本工具面向"文档"而非"界面"。

后端: paddlex PP-DocLayoutV3（本地 CPU，首次自动下载模型 ~30MB，需代理）。
已知坑: paddlepaddle oneDNN bug → enable_mkldnn=False。

用法:
  vs_layout.py --image PATH [--max-items 100] [--min-conf 0.3]

输出 schema v2 elements: type="layout", label=区域类别, bbox, conf。
"""
import argparse
import json
import vs_schema as S
import os
import sys

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")  # 跳过连通性检查


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--max-items", type=int, default=100)
    ap.add_argument("--min-conf", type=float, default=0.3)
    args = ap.parse_args()

    try:
        from PIL import Image  # type: ignore[import-not-found]

        im = Image.open(args.image)
        w, h = im.size

        import paddlex as pdx  # type: ignore[import-not-found]

        model = pdx.create_model("PP-DocLayoutV3", enable_mkldnn=False)
        out = list(model.predict(args.image))
        first = out[0] if out else {}
        boxes = first.get("boxes", []) or []

        els = []
        for b in boxes:
            conf = float(b.get("score", 0.0))
            if conf < args.min_conf:
                continue
            label = str(b.get("label", "unknown"))
            coord = b.get("coordinate") or []
            if len(coord) == 4:
                bbox = [int(v) for v in coord]
            else:
                xs = [int(p[0]) for p in coord] if coord else [0]
                ys = [int(p[1]) for p in coord] if coord else [0]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
            els.append({"id": len(els), "type": "layout", "label": label,
                        "bbox": bbox, "conf": round(conf, 3),
                        "text": None, "color": None, "font": None, "z": None,
                        "source": ["layout"], "coordsys": "image_px",
                        "center": [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2],
                        "quad": None})
            if len(els) >= args.max_items:
                break

        print(S.dump_json({"schema": "vision-report/v2", "task": "layout",
                          "sensors": ["layout"], "coordsys": "image_px",
                          "source": {"type": "image", "path": args.image,
                                     "size_px": [w, h], "engine": "PP-DocLayoutV3"},
                          "elements": els, "anomalies": [], "metrics": {},
                          "truncated": len(els) >= args.max_items}))
        return 0
    except Exception as e:
        print(json.dumps({"error": "vs_layout failed", "detail": str(e)[:500]}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
