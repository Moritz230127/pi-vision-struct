#!/usr/bin/env python3
"""run_acceptance.py — Phase 2.3 验收：规则+VLM 结论 vs 人工真值一致率。

流程:
  1. 构建/检查抽样集（bench/samples，gen_samples.py 产物）
  2. 对每个样本跑 vs_rules → findings（保存到 samples/_rules/）
  3. 与人工真值比对：
     - 规则召回（每条规则的命中样本数/期望样本数）
     - 样本一致率（期望全部命中 且 无非预期 warn/critical）
     - 干净样本误报（FP 计数；info 级豁免，仅 warn/critical 计 FP）
  4. （可选 --enable-critic）缺陷样本裁剪区给 qwen3-vl 复核：
     - critic 召回（期望 finding 被 confirmed 的比例）
     - critic 分歧（rejected 的期望 finding）
  5. 输出 bench/acceptance_report.json + 表格

验收判据（PLAN2）: 规则+VLM 结论与人工标注一致率 ≥ 原生多模态基准。
原生多模态臂需免费 Gemini/GLM 密钥（用户侧），此处列明为残余差距。

用法:
  /home/Arch/conda-envs/pi-vision/bin/python bench/run_acceptance.py [--enable-critic]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / "python"
SAMPLES = Path(__file__).resolve().parent / "samples"
RULES_OUT = SAMPLES / "_rules"
PYBIN = "/home/Arch/conda-envs/pi-vision/bin/python"
CRITIC_PYBIN = PYBIN  # critic 复用 pi-vision env（PIL + urllib，无 DL 依赖）

SEV_OK = {"critical", "warn"}  # warn/critical 同桶；info 为咨询级
SEV_INFO = {"info"}


def run_tool(pybin: str, script: str, args: list[str], timeout: int = 180) -> dict:
    r = subprocess.run([pybin, str(PY / script), *args], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise AssertionError(f"{script} exit {r.returncode}: {r.stdout[:300]} {r.stderr[:200]}")
    try:
        return json.loads(r.stdout)
    except ValueError as e:
        raise AssertionError(f"{script} 非 JSON 输出: {r.stdout[:300]}") from e


def severity_bucket(f: dict) -> str:
    return "warn_critical" if f.get("severity") in SEV_OK else "info"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--enable-critic", action="store_true", help="运行 VLM critic 臂（慢：每个裁剪区 1-2.5 分钟）")
    ap.add_argument("--critic-samples", default="s01_low_contrast,s02_overlap,s03_alignment,s04_spacing,s05_offcanvas,s06_edge_text")
    args = ap.parse_args()

    try:
        from gen_samples import build  # type: ignore[import-not-found]

        build(SAMPLES)
        RULES_OUT.mkdir(parents=True, exist_ok=True)

        rows = []
        rule_hits: dict[str, list[bool]] = {}
        fp_clean: list[tuple[str, str, str]] = []

        for sid_full in sorted(p.stem for p in SAMPLES.glob("*.report.json")):
            sid = sid_full[: -len(".report")]
            gt = json.loads((SAMPLES / f"{sid}.gt.json").read_text(encoding="utf-8"))
            expected = gt.get("expected", [])
            out = run_tool(PYBIN, "vs_rules.py", ["--report", str(SAMPLES / f"{sid}.report.json")])
            findings = out.get("findings", [])
            (RULES_OUT / f"{sid}.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")

            # 期望规则 vs 实际（规则 + 严重度桶匹配：critical/warn 同桶，info 独立）
            matched, missing = [], []
            for exp in expected:
                exp_bucket = severity_bucket({"severity": exp["severity"]})
                hit = any(f["rule"] == exp["rule"] and severity_bucket(f) == exp_bucket
                          for f in findings)
                if hit:
                    matched.append(exp["rule"])
                else:
                    missing.append(exp["rule"])
                rule_hits.setdefault(exp["rule"], []).append(hit)
            # 非预期 warn/critical（info 豁免）
            exp_rules = {e["rule"] for e in expected}
            unexpected = [f for f in findings
                          if f["rule"] not in exp_rules and severity_bucket(f) == "warn_critical"]
            for f in unexpected:
                fp_clean.append((sid, f["rule"], str(f.get("evidence"))[:60]))
            ok = (not missing) and (not unexpected)
            rows.append({"sample": sid, "clean": gt["clean"], "expected": expected,
                         "matched": matched, "missing": missing,
                         "unexpected": [f["rule"] for f in unexpected],
                         "finding_count": len(findings), "ok": ok})

        # ---- 规则侧指标 ----
        n = len(rows)
        agreement = sum(1 for r in rows if r["ok"])
        rule_recall = {rule: (sum(hits), len(hits)) for rule, hits in rule_hits.items()}
        clean_rows = [r for r in rows if r["clean"]]
        defect_rows = [r for r in rows if not r["clean"]]

        # ---- critic 臂 ----
        critic = {"enabled": args.enable_critic}
        if args.enable_critic:
            critic_checked, critic_confirmed, critic_rejected = 0, 0, 0
            critic_details = []
            critic_samples = [s for s in args.critic_samples.split(",") if s]
            for sid in critic_samples:
                rpt = RULES_OUT / f"{sid}.json"
                if not rpt.exists():
                    continue
                cr = run_tool(CRITIC_PYBIN, "vs_critic.py",
                              ["--report", str(rpt), "--image", str(SAMPLES / f"{sid}.png"),
                               "--enable", "--max-critic", "4", "--max-tokens", "2048"],
                              timeout=1500)
                for f in cr.get("findings", []):
                    c = f.get("critic") or {}
                    if c.get("ok"):
                        critic_checked += 1
                        v = c.get("verdict", "uncertain")
                        critic_confirmed += int(v == "confirmed")
                        critic_rejected += int(v == "rejected")
                        critic_details.append({
                            "sample": sid, "rule": f.get("rule"),
                            "verdict": v, "reason": c.get("reason", "")[:80],
                        })
            critic.update({
                "checked_findings": critic_checked,
                "confirmed": critic_confirmed, "rejected": critic_rejected,
                "critic_recall": round(critic_confirmed / max(1, critic_checked), 3),
                "details": critic_details,
            })

        # ---- 报告 ----
        report = {
            "schema": "acceptance/v1",
            "task": "aesthetic_judgment_2.3",
            "generated": "deterministic-samples",
            "metrics": {
                "sample_count": n, "defect_count": len(defect_rows), "clean_count": len(clean_rows),
                "sample_agreement": round(agreement / n, 3),
                "defect_recall": round(sum(1 for r in defect_rows if r["ok"]) / max(1, len(defect_rows)), 3),
                "clean_precision": round(sum(1 for r in clean_rows if r["ok"]) / max(1, len(clean_rows)), 3),
                "rule_recall": {k: round(h / t, 3) for k, (h, t) in rule_recall.items()},
                "fp_clean_warn_critical": len(fp_clean),
            },
            "fp_details": fp_clean,
            "critic": critic,
            "residual_gaps": [
                "原生多模态基线（Gemini/GLM 免费层）需用户提供密钥后执行同套抽样集对比，当前报告该臂未运行",
                "critic 仅复核规则已发现的 finding，不主动发现新缺陷（无开放集标注能力）",
                "info 级 finding 不计入样本失败（对齐/间距为咨询级，避免把自然排版差异当缺陷）",
                "抽样集为确定性合成样本；真实截图美学判断需人工标注集（待建）",
                "critic 对全局属性缺陷（出界/安全区）在裁剪视图下会误拒（实测 s05）：裁剪丢失画布上下文，此类 finding 应豁免 VLM 复核或携带画布边界",
                "critic 能识别规则误报（实测 s02 重叠但无内容遮挡被拒）：闭环对精度有增益，overlap 阈值可按此调优",
            ],
        }
        (Path(__file__).resolve().parent / "acceptance_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        # ---- 打印 ----
        print(f"{'样本':22} {'clean':5} {'匹配':30} {'缺失':18} {'意外':10} 状态")
        for r in rows:
            print(f"{r['sample']:22} {str(r['clean']):5} {str(r['matched']):30} "
                  f"{str(r['missing']):18} {str(r['unexpected']):10} {'PASS' if r['ok'] else 'FAIL'}")
        m = report["metrics"]
        print(f"\n样本一致率      : {m['sample_agreement']:.1%} ({agreement}/{n})")
        print(f"缺陷样本召回    : {m['defect_recall']:.1%}  | 干净样本精度: {m['clean_precision']:.1%}")
        print(f"规则召回        : {m['rule_recall']}")
        print(f"干净样本 FP(w/c): {m['fp_clean_warn_critical']}")
        if args.enable_critic:
            print(f"critic 裁决      : checked={critic['checked_findings']} confirmed={critic['confirmed']} "
                  f"rejected={critic['rejected']} recall={critic['critic_recall']:.1%}")
        print("\n残余差距:")
        for g in report["residual_gaps"]:
            print(f"  - {g}")
        print(f"\n报告 -> bench/acceptance_report.json")
        return 0
    except Exception as e:
        print(json.dumps({"error": "run_acceptance failed", "detail": str(e)[:500]}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
