#!/usr/bin/env python3
"""test_fusion.py — V3 融合引擎单元测试（S1 门禁印证）。

覆盖：
  S1-1 D-S 组合数学正确性（手算期望值）
  S1-2 匈牙利匹配全局最优（无重复）
  S1-3 冲突处理（K>0.7 → needs_review）
  S1-5 确定性（同输入两次运行输出一致）
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import vs_fusion as F


def test_ds_combine_math():
    """S1-1: 已知 mass 输入 → 手算期望输出一致。"""
    m1 = {F.CONSISTENT: 0.6, F.CONFLICT: 0.1, F.UNCERTAIN: 0.3}
    m2 = {F.CONSISTENT: 0.7, F.CONFLICT: 0.0, F.UNCERTAIN: 0.3}
    r = F.ds_combine(m1, m2)
    # 手算: K=0.07; m_cons=0.81/0.93=0.871; m_conf=0.03/0.93=0.0323; m_unc=0.09/0.93=0.0968
    assert abs(r[F.CONSISTENT] - 0.871) < 0.01
    assert abs(r[F.CONFLICT] - 0.0323) < 0.01
    assert abs(r[F.UNCERTAIN] - 0.0968) < 0.01


def test_ds_combine_identity():
    """全不确定与任何 mass 组合 → 原 mass（单位元）。"""
    m = {F.CONSISTENT: 0.8, F.CONFLICT: 0.1, F.UNCERTAIN: 0.1}
    u = {F.CONSISTENT: 0.0, F.CONFLICT: 0.0, F.UNCERTAIN: 1.0}
    r = F.ds_combine(m, u)
    assert abs(r[F.CONSISTENT] - 0.8) < 1e-9
    assert abs(r[F.CONFLICT] - 0.1) < 1e-9
    assert abs(r[F.UNCERTAIN] - 0.1) < 1e-9


def test_ds_combine_total_conflict():
    """完全冲突 → 全不确定（防除零）。"""
    m1 = {F.CONSISTENT: 1.0, F.CONFLICT: 0.0, F.UNCERTAIN: 0.0}
    m2 = {F.CONSISTENT: 0.0, F.CONFLICT: 1.0, F.UNCERTAIN: 0.0}
    r = F.ds_combine(m1, m2)
    assert r[F.UNCERTAIN] == 1.0


def test_hungarian_no_duplicate():
    """S1-2: 合成 10 对 bbox → 全局最优匹配（无重复）。"""
    boxes_a = [[i * 100, 0, i * 100 + 50, 50] for i in range(10)]
    boxes_b = [[i * 100 + 2, 2, i * 100 + 52, 52] for i in range(10)]
    pairs = F.hungarian_match(boxes_a, boxes_b)
    assert len(pairs) == 10
    ia = [p[0] for p in pairs]
    ib = [p[1] for p in pairs]
    assert len(set(ia)) == 10 and len(set(ib)) == 10
    assert all(p[2] > 0.8 for p in pairs)


def test_hungarian_crossing():
    """交叉 bbox：匈牙利应给出全局最优而非贪心。"""
    # A0 与 B1 更近，A1 与 B0 更近（交叉）
    boxes_a = [[0, 0, 10, 10], [100, 0, 110, 10]]
    boxes_b = [[100, 0, 110, 10], [0, 0, 10, 10]]
    pairs = F.hungarian_match(boxes_a, boxes_b)
    # 全局最优：A0-B1, A1-B0（IoU=1.0）
    assert (0, 1) in [(p[0], p[1]) for p in pairs]
    assert (1, 0) in [(p[0], p[1]) for p in pairs]


def test_conflict_needs_review():
    """S1-3: K>0.7 → needs_review 而非误报。"""
    m3 = {F.CONSISTENT: 0.9, F.CONFLICT: 0.1, F.UNCERTAIN: 0.0}
    m4 = {F.CONSISTENT: 0.1, F.CONFLICT: 0.9, F.UNCERTAIN: 0.0}
    k = F._conflict_coef([m3, m4])
    assert k > 0.7
    bel, pla = F.belief_plausibility(F.ds_combine(m3, m4))
    v = F.decide(bel, pla, k)
    assert v == "needs_review"


def test_verdict_confirmed():
    """高一致证据 → confirmed。"""
    m = F.ds_combine_all([
        {F.CONSISTENT: 0.8, F.CONFLICT: 0.0, F.UNCERTAIN: 0.2},
        {F.CONSISTENT: 0.7, F.CONFLICT: 0.0, F.UNCERTAIN: 0.3},
    ])
    bel, pla = F.belief_plausibility(m)
    v = F.decide(bel, pla, 0.0)
    assert v == "confirmed"


def test_determinism():
    """S1-5: 同输入两次运行输出一致。"""
    els = [
        {"id": 0, "type": "text", "bbox": [0, 0, 50, 50], "text": "hello",
         "conf": 0.9, "source": ["ocr"]},
        {"id": 1, "type": "text", "bbox": [2, 2, 52, 52], "text": "hello",
         "conf": 0.8, "source": ["dom"]},
    ]
    f1 = F.fuse_elements(els)
    f2 = F.fuse_elements(els)
    assert json.dumps(f1, sort_keys=True) == json.dumps(f2, sort_keys=True)


def test_mass_factories():
    """mass 映射表：各传感器工厂输出合法。"""
    assert abs(F.mass_from_ocr(0.9)[F.CONSISTENT] - 0.72) < 1e-9
    assert F.mass_from_color_delta(0.0)[F.CONSISTENT] == 1.0
    assert F.mass_from_color_delta(100.0)[F.CONSISTENT] < 0.01
    assert F.mass_from_detect(0.5)[F.CONSISTENT] == 0.5
    assert F.mass_from_segment(0.5)[F.CONSISTENT] == 1.0
    assert F.mass_from_saliency(0.8)[F.CONSISTENT] == 0.8
    assert F.mass_from_depth(0.0)[F.CONSISTENT] == 1.0


def test_text_sim():
    """文本相似度：精确/包含/编辑距离。"""
    assert F._text_sim("hello", "hello") == 1.0
    assert F._text_sim("hello", "hello world") == 0.7
    assert F._text_sim("hello", "hallo") > 0.5
    assert F._text_sim("", "x") == 0.0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
