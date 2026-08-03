#!/usr/bin/env python3
"""vs_analyze.py — 任务引擎：按 JSON 配置编排传感器与融合算子（schema v2）。

任务 = 配置，新任务 = 新配置（零新代码）。配置位于 tasks/<name>.json：

{
  "steps": [
    {"as": "pix",  "cmd": ["vs_pix.py", "--image", "$INPUT"]},
    {"as": "ocr",  "cmd": ["vs_ocr.py", "--image", "$INPUT"]},
    {"as": "dom",  "cmd": ["vs_dom.py", "--url", "$URL"], "if": "$URL", "optional": true},
    {"as": "fused","cmd": ["vs_crosscheck.py", "--image", "$INPUT", "--dom", "$dom",
                            "--ocr", "$ocr", "--dpr", "$DPR"]}
  ],
  "report": "fused"
}

变量：$INPUT/$URL/$DPR 来自 CLI；$<step> 为前序步骤输出的临时 JSON 路径。
"if"：变量为空则跳过该步骤。"optional"：失败不中断。
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import vs_schema as S

ROOT = Path(__file__).resolve().parent
PYBIN = "/home/Arch/conda-envs/pi-vision/bin/python"


def run(cmd: list[str], timeout: int) -> tuple[int, str]:
    full: list[str] = [PYBIN]
    head = cmd[0] if cmd else ""
    if head.endswith(".py") and not head.startswith("/"):
        full.append(str(ROOT / head))
    else:
        full.append(head)
    full.extend(cmd[1:])
    r = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout or r.stderr).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--input")
    ap.add_argument("--url")
    ap.add_argument("--dpr", type=float, default=1.0)
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    try:
        cfg_path = ROOT / "tasks" / f"{args.task}.json"
        if not cfg_path.exists():
            print(json.dumps({"error": f"unknown task: {args.task}"}))
            return 1
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

        vars_map: dict[str, str] = {
            "INPUT": args.input or "", "URL": args.url or "", "DPR": str(args.dpr),
        }
        outputs: dict[str, str] = {}
        skipped: list[str] = []
        tmpdir = tempfile.mkdtemp(prefix="vs_analyze_")

        for step in cfg["steps"]:
            name = step["as"]
            # "if" 条件：引用的变量为空则跳过
            if step.get("if"):
                cond_var = step["if"]
                if not vars_map.get(cond_var.lstrip("$"), ""):
                    skipped.append(name)
                    continue
            cmd: list[str] = []
            for c in [str(x) for x in step.get("cmd", [])]:
                if c.startswith("$"):
                    key = c[1:]
                    cmd.append(vars_map.get(key) or outputs.get(key) or "")
                else:
                    cmd.append(c)
            try:
                code, out_text = run(cmd, args.timeout)
            except Exception as e:
                code, out_text = 1, str(e)
            path = os.path.join(tmpdir, f"{name}.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write(out_text if out_text else json.dumps({"error": "no output"}))
            outputs[name] = path
            if code != 0 and not step.get("optional"):
                print(out_text or json.dumps({"error": f"step {name} failed", "code": code}))
                return code

        # 合并所有步骤输出为统一融合报告（sensor 数据 + 融合异常全部可见）
        merged: dict[str, Any] = {
            "schema": "vision-report/v2", "task": cfg.get("name") or args.task,
            "sensors": [], "source": {}, "elements": [], "anomalies": [],
            "metrics": {}, "truncated": False,
        }
        sensor_names = []
        for name in outputs:
            try:
                step_report = S.load_json(outputs[name])
            except ValueError:
                continue
            if "error" in step_report or step_report.get("schema") is None:
                continue
            sn = step_report.get("sensors") or [name]
            merged["sensors"].extend(s for s in sn if s not in merged["sensors"])
            sensor_names.append(name)
            merged["elements"].extend(step_report.get("elements") or [])
            merged["anomalies"].extend(step_report.get("anomalies") or [])
            for k, v in (step_report.get("metrics") or {}).items():
                merged["metrics"].setdefault(k, v)
            src = step_report.get("source") or {}
            if src:
                merged["source"].update({kk: vv for kk, vv in src.items() if kk not in merged["source"]})
        merged["truncated"] = any(
            (S.load_json(outputs[n]) if n in outputs else {}).get("truncated") for n in outputs
        )
        if skipped:
            merged["skipped_steps"] = skipped
        print(S.dump_json(merged))
        return 0
    except Exception as e:
        print(json.dumps({"error": "vs_analyze failed", "detail": str(e)[:500]}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
