#!/usr/bin/env python3
"""vs_critic.py — VLM-as-critic 闭环（Phase 2.3，schema v2）。

流程：取规则/审计报告的 findings → 按严重度排序裁剪可疑区（bbox+margin）→
本地 qwen3-vl:8b（Ollama，localhost 仅本机）逐区裁决 → 裁决作为 critic 证据
并入每条 finding，输出增强后的报告 + critic 统计。

原则：
  - 默认 opt-in：--enable 或 PI_VISION_CRITIC=1 才调用 VLM（L2 语义有思考成本）
  - 裁决只是证据（confirmed/rejected/uncertain），不覆盖确定性结论
  - --max-critic 控制裁剪上限（默认 8），按 critical > warn > info 排序
  - 只连 localhost:11434，无网络外发

用法:
  vs_critic.py --report rules_report.json --image shot.png --enable
               [--max-critic 8] [--margin 4] [--model qwen3-vl:8b]
               [--base-url http://localhost:11434] [--keep-crops]
"""
import argparse
import base64
import io
import json
import vs_schema as S
import os
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from PIL import Image  # type: ignore[import-not-found]

from vs_semantic import _try_parse_json  # 复用 JSON 提取（剥 markdown 围栏等）

SEVERITY_RANK = {"critical": 0, "warn": 1, "info": 2}

CRITIC_PROMPT = (
    "你是 UI/视觉质量复核员。图中红框/标注区域疑似存在如下问题：\n"
    "问题类型: {rule}\n"
    "描述: {desc}\n"
    "证据: {evidence}\n"
    "请判断该区域是否**确实**存在此问题。只输出 JSON，不要输出其他内容：\n"
    '{{"verdict":"confirmed|rejected|uncertain","reason":"一句话中文理由"}}\n'
    "confirmed=问题确实存在；rejected=区域内容正常或问题不存在；uncertain=无法判断。"
)


def prepare_crops(
    report: dict,
    image_path: str,
    max_critic: int,
    margin: int,
) -> tuple[list[dict], tuple[int, int] | None, str | None]:
    """按严重度排序裁剪可疑区。返回 (crops, image_size, error)。"""
    findings = report.get("findings") or report.get("anomalies") or []
    ranked = sorted(
        (f for f in findings if f.get("bbox")),
        key=lambda f: (SEVERITY_RANK.get(f.get("severity", "info"), 9)),
    )[:max_critic]
    if not ranked:
        return [], None, None
    try:
        im = Image.open(image_path).convert("RGB")
    except Exception as e:
        return [], None, f"图片无法打开: {e}"
    w, h = im.size
    crops = []
    for idx, f in enumerate(ranked):
        try:
            x1, y1, x2, y2 = (int(v) for v in f["bbox"][:4])
        except (TypeError, ValueError):
            continue
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        region = im.crop((x1, y1, x2, y2))
        buf = io.BytesIO()
        region.save(buf, format="PNG")
        crops.append({
            "index": idx, "finding": f, "region": [x1, y1, x2, y2],
            "png_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
        })
    return crops, (w, h), None


def critic_judge(
    crop: dict,
    model: str,
    base_url: str,
    max_tokens: int,
    timeout: int,
) -> dict:
    """调用本地 Ollama 裁决单区域。永不抛异常。"""
    prompt = CRITIC_PROMPT.format(
        rule=crop["finding"].get("rule", "unknown"),
        desc=crop["finding"].get("suggested_cause", ""),
        evidence=json.dumps(crop["finding"].get("evidence", {}), ensure_ascii=False)[:400],
    )
    t0 = time.time()
    try:
        import vs_vlm
        r = vs_vlm.generate(prompt, crop["png_b64"], model=model,
                            base_url=base_url, max_tokens=max_tokens, timeout=timeout)
        if not r["ok"]:
            return {"ok": False, "verdict": "uncertain", "error": r["error"],
                    "ms": int((time.time() - t0) * 1000)}
        text = r["text"]
        parsed = _try_parse_json(text) or {}
        verdict = str(parsed.get("verdict", "uncertain"))
        if verdict not in ("confirmed", "rejected", "uncertain"):
            verdict = "uncertain"
        return {
            "ok": True, "verdict": verdict,
            "reason": str(parsed.get("reason", ""))[:200],
            "model": model, "ms": int((time.time() - t0) * 1000),
        }
    except Exception as e:
        return {"ok": False, "verdict": "uncertain", "error": str(e)[:300], "ms": int((time.time() - t0) * 1000)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--enable", action="store_true", help="显式开启 VLM 复核（默认 opt-in 关闭）")
    ap.add_argument("--max-critic", type=int, default=8)
    ap.add_argument("--margin", type=int, default=4)
    ap.add_argument("--model", default="qwen3-vl:8b")
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--keep-crops", action="store_true")
    args = ap.parse_args()

    try:
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    except Exception as e:
        print(json.dumps({"error": "vs_critic failed", "detail": f"报告读取失败: {e}"}, ensure_ascii=False))
        return 1

    if not (args.enable or os.environ.get("PI_VISION_CRITIC") == "1"):
        report["critic"] = {
            "enabled": False,
            "reason": "opt-in: 传 --enable 或设 PI_VISION_CRITIC=1（VLM 复核有思考成本）",
        }
        print(S.dump_json(report))
        return 0

    crops, image_size, err = prepare_crops(report, args.image, args.max_critic, args.margin)
    if err:
        print(json.dumps({"error": "vs_critic failed", "detail": err}, ensure_ascii=False))
        return 1

    critic_stats = {
        "enabled": True, "checked": 0,
        "confirmed": 0, "rejected": 0, "uncertain": 0,
        "model": args.model, "image_size": image_size, "margin": args.margin,
    }
    crop_dir = None
    if args.keep_crops and crops:
        crop_dir = Path(tempfile.mkdtemp(prefix="vs_critic_"))
    for c in crops:
        verdict = critic_judge(c, args.model, args.base_url, args.max_tokens, args.timeout)
        c["finding"]["critic"] = verdict
        critic_stats["checked"] += 1
        critic_stats[verdict["verdict"]] = critic_stats.get(verdict["verdict"], 0) + 1
        if args.keep_crops and crop_dir:
            try:
                import base64 as _b64
                img = Image.open(io.BytesIO(_b64.b64decode(c["png_b64"])))
                img.save(crop_dir / f"crop_{c['index']}_{verdict['verdict']}.png")
            except Exception:
                pass
    report["critic"] = critic_stats
    if crop_dir:
        report["critic"]["crop_dir"] = str(crop_dir)
    print(S.dump_json(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
