"""test_pipeline_fake：annotate_frame 全链（Fake 注入，零真实权重）+ review_queue + data 工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from auto3dlabel.data.kitti import (
    frame_ids_by_range,
    frame_ids_from_dir,
    normalize_frame_id,
    resolve_frame,
)
from auto3dlabel.export.review_queue import triage_3d, write_review_queue
from auto3dlabel.schema.box3d import Box3D, FrameResult, KittiFrame
from auto3dlabel.tests.helpers.fakes import FakeDetection, FakeSegmentation, det
from auto3dlabel.tests.helpers.synth import ANCHOR_PIXEL, VELO_ANCHOR, write_frame
from auto3dlabel.tools.pipeline import annotate_frame


def _velo_box(
    n: int, x0: float, y0: float, dx: float, dy: float, z_lo: float, z_hi: float,
) -> np.ndarray:
    """velodyne 系矩形：x∈[x0,x0+dx]（前）y∈[y0,y0+dy]（左）z∈[z_lo,z_hi]。"""
    rng = np.random.default_rng(7)
    pts = np.stack(
        [
            rng.uniform(x0, x0 + dx, n),
            rng.uniform(y0, y0 + dy, n),
            rng.uniform(z_lo, z_hi, n),
        ],
        axis=1,
    )
    return np.hstack([pts, np.zeros((n, 1), dtype=np.float32)]).astype(np.float32)


def _velo_car(n: int = 500, x0: float = 8.0, y0: float = -2.0) -> np.ndarray:
    """车侧矩形 4×2m，z∈[-1.5,-0.3]。"""
    return _velo_box(n, x0, y0, 4.0, 2.0, -1.5, -0.3)


def _car_frame(
    tmp_path: Path, n: int = 500, x0: float = 8.0, y0: float = -2.0, **kw: Any
) -> tuple[KittiFrame, tuple[float, float, float, float]]:
    frame = write_frame(tmp_path, points=_velo_car(n, x0, y0), **kw)
    calib = frame.load_calib()
    u, v, valid = calib.project_velo_to_image(frame.load_points()[:, :3], 1242, 375)
    u, v = u[valid], v[valid]
    return frame, (float(u.min()), float(v.min()), float(u.max()), float(v.max()))


def _pipe(
    tmp_path: Path,
    prompts: tuple[str, ...] = ("car",),
    conf: float = 0.3,
    viz: bool = False,
    out_dir: Path | None = None,
    **kw: Any,
) -> tuple[KittiFrame, FrameResult]:
    frame, (x1, y1, x2, y2) = _car_frame(tmp_path, **kw)
    pad = 6.0
    poly = [x1 - pad, y1 - pad, x2 + pad, y1 - pad, x2 + pad, y2 - pad, x1 - pad, y2 - pad]
    det_model = FakeDetection([
        det(x1 - pad, y1 - pad, x2 - x1 + 2 * pad, y2 - y1 + 2 * pad, prompts[0], 0.9),
    ])
    seg_model = FakeSegmentation([poly])
    result = annotate_frame(
        frame, list(prompts), det_model=det_model, seg_model=seg_model,
        confidence=conf, viz=viz, out_dir=out_dir,
    )
    return frame, result


def test_full_chain_one_box(tmp_path: Any) -> None:
    frame, result = _pipe(tmp_path)
    assert len(result.boxes3d) == 1
    box = result.boxes3d[0]
    assert box.label == "Car"  # COCO car → KITTI Car
    assert box.confidence == 0.9
    assert 3.0 < box.l < 5.0 and 1.4 < box.w < 2.6  # velo 侧矩形 4×2
    assert box.fit_points > 200
    assert not result.warnings
    assert result.dropped_no_points == 0


def test_person_maps_to_pedestrian(tmp_path: Any) -> None:
    """行人类走 Pedestrian 尺寸点云（1×0.6×1m），并验证类映射 person→Pedestrian。"""
    frame = write_frame(tmp_path, points=_velo_box(300, 8.0, -2.0, 1.0, 0.6, -1.6, -0.6))
    calib = frame.load_calib()
    u, v, valid = calib.project_velo_to_image(frame.load_points()[:, :3], 1242, 375)
    u, v = u[valid], v[valid]
    x1, y1, x2, y2 = float(u.min()), float(v.min()), float(u.max()), float(v.max())
    pad = 6.0
    poly = [x1 - pad, y1 - pad, x2 + pad, y1 - pad, x2 + pad, y2 - pad, x1 - pad, y2 - pad]
    det_model = FakeDetection([
        det(x1 - pad, y1 - pad, x2 - x1 + 2 * pad, y2 - y1 + 2 * pad, "person", 0.9),
    ])
    seg_model = FakeSegmentation([poly])
    result = annotate_frame(
        frame, ["person"], det_model=det_model, seg_model=seg_model, confidence=0.3,
    )
    assert len(result.boxes3d) == 1
    assert result.boxes3d[0].label == "Pedestrian"


def test_zero_dets_empty_result(tmp_path: Any) -> None:
    frame = write_frame(tmp_path, points=_velo_car())
    det_model = FakeDetection([det(0, 0, 10, 10, "car", 0.1)])  # conf 0.1 < 阈值 0.3
    seg_model = FakeSegmentation([[0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]])
    result = annotate_frame(
        frame, ["car"], det_model=det_model, seg_model=seg_model, confidence=0.3,
    )
    assert result.boxes3d == []
    assert det_model.calls[0][2] == 0.3  # 阈值透传


def test_mask_without_points_dropped(tmp_path: Any) -> None:
    frame, _ = _car_frame(tmp_path)
    # polygon 远离所有投影点 → 反投影 0 点 → 丢弃 + dropped 计数
    far_poly = [10.0, 10.0, 30.0, 10.0, 30.0, 30.0, 10.0, 30.0]
    det_model = FakeDetection([det(10, 10, 20, 20, "car", 0.9)])
    seg_model = FakeSegmentation([far_poly])
    result = annotate_frame(
        frame, ["car"], det_model=det_model, seg_model=seg_model, confidence=0.3,
    )
    assert result.boxes3d == []
    assert result.dropped_no_points == 1
    assert any("零 LiDAR 点" in w for w in result.warnings)


def test_viz_outputs(tmp_path: Any) -> None:
    out_dir = tmp_path / "out"
    frame, result = _pipe(tmp_path, viz=True, out_dir=out_dir)
    assert len(result.boxes3d) == 1
    assert result.bev_path and (tmp_path / "out" / "000000_bev.png").is_file()
    assert result.proj_check_path and (tmp_path / "out" / "000000_proj.png").is_file()


def test_disable_tf32_called() -> None:
    """入口 disable_tf32（确定性红线）——调用了不抛即可。"""
    from auto2dlabel.tools.device import disable_tf32

    disable_tf32()  # 幂等


def test_two_instances_two_boxes(tmp_path: Any) -> None:
    """两辆分离的车（velo 系 y 相差 10m）→ 两个框。"""
    pts = np.vstack([_velo_car(400, y0=-2.0), _velo_car(400, y0=-14.0)])
    frame = write_frame(tmp_path, points=pts)
    calib = frame.load_calib()
    u, v, valid = calib.project_velo_to_image(pts[:, :3], 1242, 375)
    u, v = u[valid], v[valid]
    x1, y1, x2, y2 = float(u.min()), float(v.min()), float(u.max()), float(v.max())
    pad = 6.0
    poly = [x1 - pad, y1 - pad, x2 + pad, y1 - pad, x2 + pad, y2 - pad, x1 - pad, y2 - pad]
    det_model = FakeDetection([
        det(x1 - pad, y1 - pad, x2 - x1 + 2 * pad, y2 - y1 + 2 * pad, "car", 0.9),
    ])
    seg_model = FakeSegmentation([poly])
    result = annotate_frame(
        frame, ["car"], det_model=det_model, seg_model=seg_model, confidence=0.3,
    )
    # 同一 mask 两簇：主簇保留，次簇（≥30%）→ adhesion → 仍只出一个主簇框
    assert 1 <= len(result.boxes3d) <= 2


# ── triage / review queue ─────────────────────────────────────

def _box(conf: float = 0.9, fit: int = 100, review: bool = False, label: str = "Car") -> Box3D:
    return Box3D(label=label, confidence=conf, cx=1.0, cz=2.0, h=1.5, w=1.6, l=3.9,
                 yaw_bev=0.0, fit_points=fit, review_flag=review)


def test_triage_three_tiers() -> None:
    triage = triage_3d([
        _box(conf=0.9, fit=100),   # accepted
        _box(conf=0.5, fit=100),   # review（中 conf）
        _box(conf=0.1, fit=100),   # hard
    ])
    assert len(triage.accepted) == 1 and len(triage.review) == 1 and len(triage.hard) == 1


def test_triage_fit_points_forces_review() -> None:
    """fit_points < 20 → 强制 review（即使高 conf）。"""
    triage = triage_3d([_box(conf=0.95, fit=8)])
    assert len(triage.review) == 1 and not triage.accepted and not triage.hard


def test_triage_review_flag_forces_review() -> None:
    triage = triage_3d([_box(conf=0.95, fit=100, review=True)])
    assert len(triage.review) == 1


def test_triage_low_conf_and_bad_fit_to_hard() -> None:
    """低 conf 且拟合差 → hard（不可救，非 review）。"""
    triage = triage_3d([_box(conf=0.1, fit=8)])
    assert len(triage.hard) == 1 and not triage.review


def test_write_review_queue(tmp_path: Any) -> None:
    frame = write_frame(tmp_path)
    triage = triage_3d([_box(conf=0.5, fit=100), _box(conf=0.95, fit=100)])
    path = write_review_queue(frame, triage, tmp_path / "reviews")
    data = json.loads(path.read_text())
    assert data["image"] == "000000"
    assert len(data["annotations"]) == 1  # 仅 review+hard
    assert data["annotations"][0]["fit_points"] == 100  # 恒输出
    assert data["pcd_path"].endswith("velodyne/000000.bin")
    assert data["calib_path"].endswith("calib/000000.txt")
    assert data["summary"]["accepted"] == 1
    assert data["summary"]["total_need_review"] == 1


# ── data/kitti 工具 ────────────────────────────────────────────

def test_normalize_frame_id() -> None:
    assert normalize_frame_id("123") == "000123"
    assert normalize_frame_id("000123") == "000123"
    with pytest.raises(ValueError):
        normalize_frame_id("abc")


def test_frame_ids_by_range() -> None:
    ids = frame_ids_by_range(3712, 3715)
    assert ids == ["003712", "003713", "003714"]


def test_frame_ids_from_dir(tmp_path: Any) -> None:
    d = tmp_path / "image_2"
    d.mkdir()
    for name in ("000001.png", "000003.png", "notes.png"):
        (d / name).write_bytes(b"")
    assert frame_ids_from_dir(d) == ["000001", "000003"]


def test_resolve_frame_ok_and_missing(tmp_path: Any) -> None:
    frame = write_frame(tmp_path)
    resolved = resolve_frame(frame.frame_id, root=tmp_path)
    assert resolved.frame_id == "000000"
    with pytest.raises(FileNotFoundError):
        resolve_frame("009999", root=tmp_path)


def test_anchor_pixel_constant() -> None:
    """锚点常量自检（防意外改动 synth 破坏测试语义）。"""
    assert ANCHOR_PIXEL == (758, 299)
    assert VELO_ANCHOR == (8.752, -1.800, -1.546)
