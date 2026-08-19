"""tracker_time_benchmark 合成数据生成器测试 —— 确定性 + 场景语义 + 计时形状。

计时本身不做数值断言（噪声），只验证返回结构正确——
纯 tracker 耗时数值以 test-v0.4.md 实测表为准。
"""

from __future__ import annotations

import pytest

from auto2dlabel.benchmarks.tracker_time_benchmark import (
    measure_update_time,
    synthetic_frames,
)
from auto2dlabel.schema.annotation import Bbox


def _snap(frames: list[list[Bbox]]) -> list[list[tuple[float, float, float]]]:
    """帧序列快照（x, y, conf 三元组，逐位可比较）。"""
    return [[(b.x, b.y, b.confidence) for b in frame] for frame in frames]


def test_synthetic_frames_deterministic_same_seed() -> None:
    """固定 seed 两次生成逐位一致（基准可复现）。"""
    assert _snap(synthetic_frames(20, 10, seed=0)) == \
        _snap(synthetic_frames(20, 10, seed=0))


def test_synthetic_frames_different_seed_differs() -> None:
    """不同 seed 序列不同（随机游走确实生效，非空转）。"""
    assert _snap(synthetic_frames(20, 10, seed=0)) != \
        _snap(synthetic_frames(20, 10, seed=1))


def test_synthetic_frames_scenario_semantics() -> None:
    """场景语义：10% 消失产生缺失帧、每帧框数 ≤ 目标数、置信度在 [0.4, 1.0]。"""
    frames = synthetic_frames(50, 20, seed=3)
    assert all(len(f) <= 50 for f in frames)
    assert any(len(f) < 50 for f in frames), "10% 消失率应产生至少一个缺失帧"
    assert all(0.4 <= b.confidence <= 1.0 for f in frames for b in f)


def test_measure_update_time_counts_and_shape() -> None:
    """计时返回结构正确（数值本身不做断言，抗噪声）。"""
    frames = synthetic_frames(10, 5, seed=0)
    total, ms_per_frame, n_total = measure_update_time(frames, repeats=1)
    assert total >= 0
    assert ms_per_frame == pytest.approx(total / 5 * 1000)
    assert n_total == sum(len(f) for f in frames)
