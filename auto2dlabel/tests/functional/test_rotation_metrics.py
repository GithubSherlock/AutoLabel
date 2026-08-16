"""rotate_iou（旋转四边形 IoU）合成数值测试 —— shapely，无权重依赖。

覆盖：相同四边形 / 相离 / 半重叠数值断言 / 旋转 vs 轴对齐 / 对称性 /
退化与短输入容错。
"""

from __future__ import annotations

import math

import pytest

from auto2dlabel.benchmarks.common import rotate_iou

# 10×10 轴对齐正方形（顺时针点序）
_SQ = [0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]


def test_identical_quads() -> None:
    assert rotate_iou(_SQ, _SQ) == pytest.approx(1.0, abs=1e-9)


def test_disjoint_quads() -> None:
    far = [100.0, 100.0, 110.0, 100.0, 110.0, 110.0, 100.0, 110.0]
    assert rotate_iou(_SQ, far) == 0.0


def test_half_overlap_numeric() -> None:
    """两个 10×10 正方形沿 x 偏移 5 → 交集 50，并集 150 → IoU 1/3。"""
    shifted = [5.0, 0.0, 15.0, 0.0, 15.0, 10.0, 5.0, 10.0]
    assert rotate_iou(_SQ, shifted) == pytest.approx(1.0 / 3.0, abs=1e-9)


def test_rotated_vs_axis_aligned_same_center() -> None:
    """同中心旋转 45° 正方形 vs 轴对齐：IoU ∈ (0, 1) 且对称。"""
    s, c = 10.0, 5.0  # 边长与中心
    a = math.sqrt(2) / 2 * (s / 2)
    rotated = [
        c - a, c, c, c - a, c + a, c, c, c + a,  # 45° 菱形（对角线沿轴）
    ]
    iou = rotate_iou(_SQ, rotated)
    assert 0.0 < iou < 1.0
    assert rotate_iou(rotated, _SQ) == pytest.approx(iou, abs=1e-9)
    # 45° 菱形（内接圆）vs 轴对齐正方形：交 = 正方形∩菱形 = 正方形（菱形含于正方形）
    # 菱形顶点 (±a,0)/(0,±a)，a=3.535；正方形 (±5,±5) 包含菱形 → IoU = 菱形面积/正方形面积
    diamond = [
        5.0 - a, 5.0, 5.0, 5.0 - a, 5.0 + a, 5.0, 5.0, 5.0 + a,
    ]
    assert rotate_iou(_SQ, diamond) == pytest.approx((2 * a ** 2) / 100.0, abs=1e-6)


def test_degenerate_inputs() -> None:
    """短输入 / 全同点退化多边形 → 0.0 容错。"""
    assert rotate_iou([0.0, 0.0], _SQ) == 0.0
    degenerate = [1.0, 1.0] * 4
    assert rotate_iou(degenerate, degenerate) == 0.0
