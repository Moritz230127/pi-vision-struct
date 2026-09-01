#!/usr/bin/env python3
"""vs_analyze.py — 任务引擎：按 JSON 配置编排传感器与融合算子（schema v2）。

任务 = 配置，新任务 = 新配置（零新代码）。配置位于 tasks/<name>.json：

{
  "steps": [
    {"as": "pix",  "cmd": ["vs_pix.py", "--image", "$INPUT"]},
    {"as": "ocr",  "cmd": ["vs_ocr.py", "--image", "$INPUT"]},
    {"as": "dom",  "cmd": ["vs_dom.py", "--url", "$URL"], "if": "$URL", "optional": true},
    {"as": "fused","cmd": ["vs_fusion.py", "--reports", "$pix", "$ocr"]}
  ],
  "report": "fused"
}

变量：$INPUT/$URL/$DPR 来自 CLI；$<step> 为前序步骤输出的临时 JSON 路径。
"if"：变量为空则跳过该步骤。"optional"：失败不中断。
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import vs_schema as S

ROOT = Path(__file__).resolve().parent
PYBIN = "/home/Arch/conda-envs/pi-vision/bin/python"

# 合法 task 名：仅允许文件名安全字符（防止把自然语言当成文件路径）
_TASK_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_\-]{0,63}$")


def _valid_task_name(task: str) -> bool:
    return bool(_TASK_NAME_RE.match(task))


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
    ap.add_argument("--compare", help="diff 任务的对比图路径（$COMPARE）")
    ap.add_argument("--prompt", help="自由文本指令（随报告带出，供下游语义兜底/人工参考）")
    ap.add_argument("--dpr", type=float, default=1.0)
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    # 防御：task 应是配置名（文件名安全字符）。若传了自然语言长句，明确报错并列出可用任务。
    if not _valid_task_name(args.task):
        avail = sorted(p.name[:-5] for p in (ROOT / "tasks").glob("*.json"))
        hint = (
            f"task 参数应为预置任务配置名，而非自由文本。"
            f"检测到传入值形如自然语言指令（长度 {len(args.task)}）。\n"
            f"可用任务: {', '.join(avail)}\n"
            f"若需自由文本视觉分析，请用 semantic 动作（VLM 语义兜底）。"
        )
        print(json.dumps({"error": "invalid task name", "detail": hint[:600]},
                         ensure_ascii=False))
        return 1

    try:
        cfg_path = ROOT / "tasks" / f"{args.task}.json"
        if not cfg_path.exists():
            print(json.dumps({"error": f"unknown task: {args.task}"}))
            return 1
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

        vars_map: dict[str, str] = {
            "INPUT": args.input or "", "URL": args.url or "", "DPR": str(args.dpr),
            "COMPARE": args.compare or "",
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
            "schema": "vision-report/v3", "task": cfg.get("name") or args.task,
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
        if args.prompt:
            merged["user_prompt"] = args.prompt
            merged["note"] = (
                "analyze 仅执行预置数值任务；自由文本问题请用 semantic 动作（VLM 兜底）"
                "基于本报告的数值做推理。"
            )
        print(S.dump_json(merged))
        return 0
    except Exception as e:
        print(json.dumps({"error": "vs_analyze failed", "detail": str(e)[:500]}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
