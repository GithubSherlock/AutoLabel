"""test_track3d：合成序列跟踪（3 目标匀速 → ID 一致性 / 新建 / 删除 / 速度收敛）。"""

from __future__ import annotations

from auto3dlabel.schema.box3d import Box3D
from auto3dlabel.tools.track3d import Track3D, Tracker3D


def _box(x: float, z: float, label: str = "Car", conf: float = 0.9) -> Box3D:
    return Box3D(label=label, cx=x, cy=0.0, cz=z, h=1.5, w=1.6, l=3.9, yaw_bev=0.0, confidence=conf)


def _sequence(
    starts: list[tuple[float, float, float, float]],
    n_frames: int = 10,
) -> list[list[Box3D]]:
    """每帧框列表：起点 (x, z) + 每帧速度 (vx, vz) 匀速移动。"""
    seq: list[list[Box3D]] = []
    for t in range(n_frames):
        seq.append(
            [_box(x + vx * t * 0.1, z + vz * t * 0.1) for x, z, vx, vz in starts]
        )
    return seq


def test_three_targets_ten_frames_id_consistency() -> None:
    """3 目标 10 帧匀速分离 → 每帧 update 返回的 track_id 稳定一致。"""
    seq = _sequence([(0.0, 10.0, 2.0, 0.0), (0.0, 30.0, -1.0, 0.5), (50.0, 5.0, 0.0, 2.0)])
    tracker = Tracker3D()
    id_sets = [set(tracker.update(frame)) for frame in seq]
    assert len(id_sets[0]) == 3
    assert all(s == id_sets[0] for s in id_sets)
    assert len(tracker.tracks()) == 3


def test_new_target_gets_new_id() -> None:
    """第 5 帧新目标出现 → 新 track_id（不与既有冲突），旧目标 id 不变。"""
    tracker = Tracker3D()
    ids_before: set[int] = set()
    for t in range(10):
        boxes = [_box(2.0 * t * 0.1, 10.0)]
        if t >= 5:
            boxes.append(_box(30.0, 40.0))
        ids = tracker.update(boxes)
        if t < 5:
            ids_before.add(ids[0])
        else:
            assert ids[0] in ids_before  # 旧目标 id 不变
            assert ids[1] not in ids_before  # 新目标新 id
    assert len(tracker.tracks()) == 2


def test_missed_track_deleted_after_max_age() -> None:
    """目标消失 ≥max_age 帧 → 轨迹删除；框再出现 → 全新 id。"""
    tracker = Tracker3D(max_age=3)
    first_id = tracker.update([_box(0.0, 10.0)])[0]
    for _ in range(3):  # 连续 3 帧无框 = max_age
        assert tracker.update([]) == []
        assert first_id in {t.track_id for t in tracker.tracks()}
    assert tracker.update([]) == []  # 第 4 帧删除
    assert first_id not in {t.track_id for t in tracker.tracks()}
    new_id = tracker.update([_box(0.0, 10.0)])[0]
    assert new_id != first_id


def test_velocity_kalman_converges() -> None:
    """匀速 vx=2m/s 单目标 → 末帧 Kalman 速度 ≈ (2, 0)（收敛后误差 <0.3）。"""
    tracker = Tracker3D(use_kalman=True, dt=0.1)
    for t in range(10):
        tracker.update([_box(2.0 * t * 0.1, 5.0)])
    track = tracker.tracks()[0]
    vx, vz = track.velocities[-1]
    assert abs(vx - 2.0) < 0.3, track.velocities
    assert abs(vz) < 0.1


def test_velocity_difference_fallback() -> None:
    """use_kalman=False → 差分平滑 fallback：匀速 → 速度收敛到真值。"""
    tracker = Tracker3D(use_kalman=False, dt=0.1)
    for t in range(10):
        tracker.update([_box(1.0 * t * 0.1, 5.0)])
    vx, _ = tracker.tracks()[0].velocities[-1]
    assert abs(vx - 1.0) < 1e-6


def test_iou_threshold_blocks_spurious_match() -> None:
    """IoU 低于阈值 → 不匹配：新框得新 id（而非错配老 id），老轨迹记 missed。"""
    tracker = Tracker3D(iou_threshold=0.3, max_age=3)
    old_id = tracker.update([_box(0.0, 10.0)])[0]
    # 目标瞬移到 100m 外（IoU=0）：不匹配 → 新 id；老轨迹在容忍期内保留
    new_id = tracker.update([_box(100.0, 100.0)])[0]
    assert new_id != old_id
    assert old_id in {t.track_id for t in tracker.tracks()}


def test_track3d_dataclass_basics() -> None:
    """Track3D 数据容器：history/velocities 追加与 age。"""
    track = Track3D(track_id=7, label="Car")
    track.append(_box(0.0, 0.0), None)
    track.append(_box(0.1, 0.0), (1.0, 0.0))
    assert track.age == 2 and track.velocities == [(1.0, 0.0)]
    assert track.history[0].cx == 0.0


def test_update_empty_boxes_ok() -> None:
    """空帧（无检测）→ 返回空且不抛（missed 计数走 max_age 路径）。"""
    tracker = Tracker3D()
    assert tracker.update([]) == []
    assert tracker.update([_box(1.0, 2.0)]) == [1]
    assert tracker.update([]) == []
