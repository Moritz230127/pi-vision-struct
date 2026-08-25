#!/usr/bin/env python3
"""vs_chart.py — 图表理解传感器：图表图像 → 结构化数据（L2，走 vs_vlm 网关）。

针对差距分析中的 ChartQA 短板：不问"图表说明什么"（那是推理），只做
"图表里有什么数据"（这是提取）——标题/轴/系列/数据点转成可计算的 JSON，
交给纯文本模型在其最强项（数值推理）上继续工作。

用法:
  vs_chart.py --image IMG [--region x1,y1,x2,y2] [--enable] [--prompt 自定义]
输出:
  schema v2 + chart{title,x_axis,y_axis,series[{name,points[[x,y]...]}],notes}

模型: config `l2_model`（默认 qwen3-vl:8b），--model 可覆盖。opt-in: --enable。
"""
import argparse
import base64
import io
import json
import sys
from typing import Any

import vs_schema as S

CHART_PROMPT = """你是图表数据提取器。从图中提取结构化数据，只输出一个 JSON 对象，不要任何其他文字：
{"title": "图表标题或null",
 "x_axis": {"label": "x轴标签或null", "type": "category|numeric|time"},
 "y_axis": {"label": "y轴标签或null"},
 "series": [{"name": "系列名", "points": [["x值", y数值], ...]}],
 "notes": ["图例、单位、数据来源等补充信息"]}
规则：数值一律转成数字类型；类别轴 x 用原文本；读不准的点给最接近的估计；
series 必须覆盖图中全部系列；不确定的内容放 notes，不要编造。"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--region", help="x1,y1,x2,y2 —— 只理解图表区域")
    ap.add_argument("--enable", action="store_true", help="显式开启（VLM 有思考成本）")
    ap.add_argument("--prompt", default=CHART_PROMPT)
    ap.add_argument("--model", default=None, help="默认取 config l2_model")
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    report = S.envelope(task="chart", sensors=["vlm"], coordsys="image_px",
                        source={"type": "image", "path": args.image})

    if not (args.enable or os_environ_critic()):
        report["chart"] = {"enabled": False,
                           "reason": "opt-in: 传 --enable 或设 PI_VISION_CRITIC=1"}
        print(S.dump_json(report))
        return 0

    try:
        import vs_vlm

        im_b64 = vs_vlm.image_to_b64(args.image, max_side=1536)
        prompt = args.prompt
        if args.region:
            x1, y1, x2, y2 = (int(v) for v in args.region.split(","))
            prompt += f"\n（只提取该区域内的图表：x1={x1},y1={y1},x2={x2},y2={y2}）"
        r = vs_vlm.generate(prompt, im_b64, model=args.model,
                             max_tokens=4096, timeout=args.timeout)
        if not r["ok"]:
            report["error"] = r["error"]
            print(S.dump_json(report))
            return 1
        parsed = _try_parse_json(r["text"])
        report["chart"] = {
            "enabled": True, "model": r.get("model"),
            "data": parsed if parsed is not None else None,
            "raw": None if parsed is not None else r["text"][:2000],
            "parse_ok": parsed is not None,
        }
        if parsed and isinstance(parsed, dict):
            for i, srs in enumerate(parsed.get("series") or []):
                pts = srs.get("points") or []
                xs = [p[0] for p in pts if isinstance(p, (list, tuple)) and len(p) == 2]
                report["elements"].append(S.element(
                    i, "chart_series",
                    [0, 0, 0, 0], text=str(srs.get("name")),
                    conf=None,
                    coordsys="image_px",
                ))
                report["elements"][-1]["point_count"] = len(pts)
                report["elements"][-1]["sample"] = xs[:6]
        report["notation"] = S.NOTATION_GUIDE
        report["metrics"] = {"series_count": len(report["elements"]),
                             "parse_ok": report["chart"]["parse_ok"]}
        print(S.dump_json(report))
        return 0
    except Exception as e:
        print(json.dumps({"error": "vs_chart failed", "detail": str(e)[:500]},
                         ensure_ascii=False))
        return 1


def os_environ_critic() -> bool:
    import os
    return os.environ.get("PI_VISION_CRITIC") == "1"


def _try_parse_json(text: str) -> Any:
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)  # 思考型变体剥离
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


if __name__ == "__main__":
    sys.exit(main())
