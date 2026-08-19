"""ByteTrack 后处理器单测 —— 关联/生命周期/运动属性（纯数值，无模型依赖）。

覆盖 v0.4 Phase 1 验收要点：
- 线性运动同 ID + 卡尔曼速度收敛（Auto3dLabel v0.2 运动属性）
- 新目标新 ID、低分救援、未匹配低分无 ID、丢帧恢复、超 buffer 换 ID
- 轨迹历史（可视化折线）与空帧鲁棒性
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from auto2dlabel.models.tracking import ByteTracker
from auto2dlabel.schema.annotation import Bbox


def _det(
    cx: float, cy: float, w: float = 50.0, h: float = 50.0, conf: float = 0.9,
) -> Bbox:
    """以中心点构造检测框（测试坐标约定）。"""
    return Bbox(
        x=cx - w / 2, y=cy - h / 2, width=w, height=h,
        label="person", confidence=conf,
    )


# ============================================================
# 关联与 ID 语义
# ============================================================

def test_linear_motion_same_id_and_velocity() -> None:
    """匀速直线运动：15 帧同 ID，卡尔曼速度收敛到 (5, 0) 像素/帧。

    步长取 5px（相邻帧 IoU=0.9，远离 0.8 第一段阈值浮点边界）。
    """
    tracker = ByteTracker()
    bboxes = [_det(125.0 + 5.0 * i, 100.0) for i in range(15)]

    ids = []
    for b in bboxes:
        tracker.update([b])
        ids.append(b.track_id)

    assert len(set(ids)) == 1, f"匀速直线运动应全程同 ID，实际: {ids}"
    assert ids[0] is not None

    vel = tracker.velocity(ids[0])
    assert vel is not None
    vx, vy = vel
    assert vx == pytest.approx(5.0, abs=1.0), f"vx 应收敛到 5，实际 {vx}"
    assert vy == pytest.approx(0.0, abs=0.5), f"vy 应接近 0，实际 {vy}"


def test_track_id_assigned_immediately() -> None:
    """设计取舍：未匹配高分检测立即创建轨迹并输出 ID（标注不丢检测优先）。"""
    tracker = ByteTracker()
    det = _det(100.0, 100.0)
    tracker.update([det])
    assert det.track_id == 0, "首帧高分检测应立即拿到 ID"


def test_new_target_gets_new_id() -> None:
    """第二帧出现的新目标获得新 ID，不与既有轨迹混用。"""
    tracker = ByteTracker()
    first = _det(100.0, 100.0)
    tracker.update([first])

    second = _det(300.0, 100.0)
    tracker.update([first, second])

    assert first.track_id is not None
    assert second.track_id is not None
    assert second.track_id != first.track_id


def test_low_score_rescue_keeps_id() -> None:
    """BYTE 第二段：目标连续帧后降为低分框（0.3），仍被救援关联同 ID。"""
    tracker = ByteTracker()
    for _ in range(3):
        tracker.update([_det(100.0, 100.0, conf=0.9)])

    low = _det(100.0, 100.0, conf=0.3)
    tracker.update([low])
    assert low.track_id == 0, "低分框应与既有轨迹救援匹配（IoU 0.5 门控）"


def test_unmatched_low_det_has_no_track_id() -> None:
    """孤立低分框：不创建轨迹、track_id 保持 None。"""
    tracker = ByteTracker()
    low = _det(100.0, 100.0, conf=0.3)
    tracker.update([low])
    assert low.track_id is None
    assert tracker.tracks() == []


# ============================================================
# 生命周期
# ============================================================

def test_occlusion_recovery_keeps_id() -> None:
    """静止目标丢 2 帧后重现：轨迹仍存活，恢复同 ID。"""
    tracker = ByteTracker()
    tracker.update([_det(100.0, 100.0)])
    tracker.update([_det(100.0, 100.0)])
    tracker.update([_det(100.0, 100.0)])
    tracker.update([])
    tracker.update([])

    reappeared = _det(100.0, 100.0)
    tracker.update([reappeared])
    assert reappeared.track_id == 0, "buffer 内丢帧应恢复同 ID"


def test_buffer_expiry_gets_new_id() -> None:
    """丢失超过 track_buffer 帧：轨迹删除，重现时换新 ID。"""
    tracker = ByteTracker(track_buffer=2)
    tracker.update([_det(100.0, 100.0)])
    for _ in range(3):  # time_since_update 3 > buffer 2 → 删除
        tracker.update([])

    reappeared = _det(100.0, 100.0)
    tracker.update([reappeared])
    assert reappeared.track_id == 1, f"超 buffer 应换新 ID，实际 {reappeared.track_id}"


def test_empty_frames_no_crash() -> None:
    """连续空帧不崩溃，无轨迹存活。"""
    tracker = ByteTracker()
    for _ in range(3):
        tracker.update([])
    assert tracker.tracks() == []


# ============================================================
# 轨迹历史与查询
# ============================================================

def test_trajectory_history() -> None:
    """轨迹中心点历史帧序完整（可视化折线数据源）。"""
    tracker = ByteTracker()
    for i in range(5):
        tracker.update([_det(100.0, 100.0)])

    centers = tracker.trajectory(0)
    assert centers is not None
    assert len(centers) == 5
    assert all(c == pytest.approx((100.0, 100.0)) for c in centers)


def test_trajectory_and_velocity_unknown_id_none() -> None:
    """未知 ID 查询返回 None（调用方按需跳过）。"""
    tracker = ByteTracker()
    tracker.update([_det(100.0, 100.0)])
    assert tracker.trajectory(99) is None
    assert tracker.velocity(99) is None


def test_active_track_ids() -> None:
    """active_track_ids 只含本帧命中的轨迹。"""
    tracker = ByteTracker()
    tracker.update([_det(100.0, 100.0), _det(300.0, 100.0)])
    tracker.update([_det(100.0, 100.0)])  # id 1 本帧未命中

    assert tracker.active_track_ids == {0}


# ============================================================
# 确定性（property 测试）：同输入必同输出
# ============================================================

def _scripted_frames() -> list[list[Bbox]]:
    """手工混合场景帧序列：两段关联 + 遮挡恢复 + 孤立低分 + 新目标。"""
    return [
        [_det(100.0, 100.0, conf=0.9), _det(300.0, 100.0, conf=0.7)],
        [_det(105.0, 100.0, conf=0.9), _det(305.0, 100.0, conf=0.35)],  # 低分救援
        [_det(110.0, 100.0, conf=0.9)],  # 目标 2 消失
        [_det(115.0, 100.0, conf=0.9), _det(305.0, 100.0, conf=0.8),
         _det(500.0, 500.0, conf=0.3)],  # 恢复 + 孤立低分（无 ID）
        [_det(120.0, 100.0, conf=0.9), _det(310.0, 100.0, conf=0.8),
         _det(500.0, 500.0, conf=0.9)],  # 新目标新轨迹
    ]


def _run_snapshot(
    frames_fn: Callable[[], list[list[Bbox]]],
) -> list[tuple[object, ...]]:
    """跑一遍 tracker（每次重建帧对象），收集逐帧输出 + 轨迹状态快照。"""
    tracker = ByteTracker()
    snap: list[tuple[object, ...]] = []
    for boxes in frames_fn():
        tracker.update(boxes)
        snap.append(tuple(
            (b.track_id, b.x, b.y, b.width, b.height, b.confidence)
            for b in boxes
        ))
    for tr in tracker.tracks():
        snap.append((tr.track_id, tr.hits, tr.time_since_update,
                     tr.velocity(), tuple(tr.trajectory())))
    return snap


def _assert_snapshots_equal(
    first: list[tuple[object, ...]], second: list[tuple[object, ...]],
) -> None:
    """逐项对比并定位首个不一致（失败信息可读）。"""
    assert len(first) == len(second)
    for i, (a, b) in enumerate(zip(first, second)):
        assert a == b, f"快照第 {i} 项不一致:\n{a}\n!=\n{b}"


def test_deterministic_same_input_same_output() -> None:
    """确定性 property：同一帧序列独立跑两次，逐帧输出与轨迹状态逐位相等。

    ByteTracker 无随机性（numpy 确定性运算 + scipy 确定性匈牙利匹配）；
    本测试防未来引入随机 tie-breaking / set 迭代序依赖 / 非确定排序。
    """
    _assert_snapshots_equal(
        _run_snapshot(_scripted_frames), _run_snapshot(_scripted_frames))


def test_deterministic_dense_synthetic_scenario() -> None:
    """密集合成场景（随机游走 30 目标 × 50 帧，固定 seed）同输入必同输出。"""
    from auto2dlabel.benchmarks.tracker_time_benchmark import synthetic_frames

    _assert_snapshots_equal(
        _run_snapshot(lambda: synthetic_frames(30, 50, seed=5)),
        _run_snapshot(lambda: synthetic_frames(30, 50, seed=5)))
