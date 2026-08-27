"""test_batch3d：LiDAR 引擎批量推理契约（Fake 注入，零 mmdet3d）。

覆盖：suggest_batch_size 纯函数、FakeDetector3D.detect_batch 记录与过滤、
annotate_frames_lidar_batch 逐帧后处理（类别过滤 / fit_points 不串帧 /
parity 单帧==批量 / BEV 逐帧落盘）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from auto3dlabel.models.detection3d import suggest_batch_size
from auto3dlabel.tests.helpers.fakes import FakeDetector3D, det3d
from auto3dlabel.tests.helpers.synth import write_frame
from auto3dlabel.tools.pipeline import annotate_frame, annotate_frames_lidar_batch

# 相机系车框（velo_car 点云经 Tr_velo_to_cam 的期望位置，同 test_pipeline_fake）
_CAR_BBOX = (1.0, 1.5, 10.0, 4.0, 1.2, 2.0, 0.0)  # [x,y,z,l,h,w,ry] 底面中心


def _velo_car(n: int = 500, x0: float = 8.0, y0: float = -2.0) -> np.ndarray:
    """velodyne 系车侧矩形点云（4×2m，z∈[-1.5,-0.3]，同 test_pipeline_fake）。"""
    rng = np.random.default_rng(7)
    pts = np.stack(
        [
            rng.uniform(x0, x0 + 4.0, n),
            rng.uniform(y0, y0 + 2.0, n),
            rng.uniform(-1.5, -0.3, n),
        ],
        axis=1,
    )
    return np.hstack([pts, np.zeros((n, 1), dtype=np.float32)]).astype(np.float32)


def _car_frame(root: Any, frame_id: str, points: np.ndarray | None = None) -> Any:
    return write_frame(
        root, frame_id=frame_id, points=points if points is not None else _velo_car()
    )


# ── suggest_batch_size 纯函数 ──────────────────────────────────

def test_suggest_batch_size_math() -> None:
    """批大小 = 空闲 × 0.85 ÷ 单帧峰值（floor）：20GiB 空闲 / 4.5GiB 单帧 → 3。"""
    assert suggest_batch_size(int(4.5e9), int(20e9)) == 3


def test_suggest_batch_size_clamps() -> None:
    """非正输入 → 1；超大空闲 → 钳 max_batch；单帧峰值超空闲 → 1。"""
    assert suggest_batch_size(0, int(20e9)) == 1
    assert suggest_batch_size(-1, int(20e9)) == 1
    assert suggest_batch_size(int(1e9), 0) == 1
    assert suggest_batch_size(int(1e6), int(1e12), max_batch=16) == 16
    assert suggest_batch_size(int(1e12), int(1e9)) == 1


# ── FakeDetector3D 批量接口 ───────────────────────────────────

def test_fake_detect_batch_records_and_filters(tmp_path: Any) -> None:
    """整批一次调用记录 (frame_ids, conf)，逐帧 conf 过滤，不混用单帧接口。"""
    frames = [_car_frame(tmp_path, "000000"), _car_frame(tmp_path, "000001")]
    fake = FakeDetector3D([
        det3d(label="Car", confidence=0.9, bbox=_CAR_BBOX),
        det3d(label="Car", confidence=0.2, bbox=_CAR_BBOX),  # < 阈值滤除
    ])
    out = fake.detect_batch(frames, conf_threshold=0.3)
    assert fake.batch_calls == [(["000000", "000001"], 0.3)]
    assert fake.calls == []
    assert all(len(r) == 1 for r in out)


# ── annotate_frames_lidar_batch 管线 ──────────────────────────

def test_batch_two_frames_independent(tmp_path: Any) -> None:
    """两帧一次 forward → 每帧独立 FrameResult；fit_points 各自点云计数（不串帧）。"""
    frame_a = _car_frame(tmp_path, "000000", _velo_car(500))
    frame_b = _car_frame(tmp_path, "000001", np.zeros((1, 4), dtype=np.float32))
    fake = FakeDetector3D([det3d(label="Car", confidence=0.9, bbox=_CAR_BBOX)])
    results = annotate_frames_lidar_batch(
        [frame_a, frame_b], ["car"], fake, confidence=0.3, viz=False
    )
    assert [r.frame_id for r in results] == ["000000", "000001"]
    assert fake.batch_calls == [(["000000", "000001"], 0.3)]
    assert len(results[0].boxes3d) == 1 and results[0].boxes3d[0].fit_points > 100
    assert results[1].boxes3d[0].fit_points == 0  # 空点云帧 → 0 点（LiDAR 观测性红线）


def test_batch_prompt_filter(tmp_path: Any) -> None:
    """prompts 类别过滤逐帧生效（同单帧语义：只保留 Car）。"""
    frames = [_car_frame(tmp_path, "000000"), _car_frame(tmp_path, "000001")]
    fake = FakeDetector3D([
        det3d(label="Car", confidence=0.9, bbox=_CAR_BBOX),
        det3d(label="Pedestrian", confidence=0.9, bbox=_CAR_BBOX),
    ])
    results = annotate_frames_lidar_batch(frames, ["car"], fake, confidence=0.3, viz=False)
    assert all(len(r.boxes3d) == 1 and r.boxes3d[0].label == "Car" for r in results)


def test_batch_parity_single_frame(tmp_path: Any) -> None:
    """parity 铁律：批量(1 帧) == 单帧（后处理同源 _lidar_frame_result）。"""
    frame = _car_frame(tmp_path, "000000")
    single = annotate_frame(
        frame, ["car"], det3d=FakeDetector3D([det3d(confidence=0.9, bbox=_CAR_BBOX)]),
        confidence=0.3, viz=False,
    )
    batch = annotate_frames_lidar_batch(
        [frame], ["car"], FakeDetector3D([det3d(confidence=0.9, bbox=_CAR_BBOX)]),
        confidence=0.3, viz=False,
    )
    assert len(batch) == 1
    assert len(single.boxes3d) == len(batch[0].boxes3d) == 1
    a, b = single.boxes3d[0], batch[0].boxes3d[0]
    assert a.label == b.label and a.confidence == b.confidence
    assert a.fit_points == b.fit_points
    assert (a.cx, a.cy, a.cz, a.l, a.h, a.w, a.yaw_bev) == (
        b.cx, b.cy, b.cz, b.l, b.h, b.w, b.yaw_bev,
    )


def test_batch_viz_outputs(tmp_path: Any) -> None:
    """批量 BEV 逐帧落盘（同单帧路径）。"""
    out_dir = tmp_path / "out"
    frames = [_car_frame(tmp_path, "000000"), _car_frame(tmp_path, "000001")]
    fake = FakeDetector3D([det3d(label="Car", confidence=0.9, bbox=_CAR_BBOX)])
    results = annotate_frames_lidar_batch(
        frames, ["car"], fake, confidence=0.3, viz=True, out_dir=out_dir
    )
    for r in results:
        assert r.bev_path and Path(r.bev_path).is_file()
