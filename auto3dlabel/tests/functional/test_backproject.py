"""test_backproject：mask 栅格化 + 语义点云反投影（mask 不用 bbox 红线）。"""

from __future__ import annotations

from typing import Any

import numpy as np

from auto3dlabel.tests.helpers import synth
from auto3dlabel.tools.backproject import backproject_semantics, rasterize_masks


def test_rasterize_two_instances() -> None:
    """两个不重叠矩形 polygon → label 图两个 ID + 背景 -1。"""
    polys = [
        [[10.0, 10.0, 20.0, 10.0, 20.0, 20.0, 10.0, 20.0]],
        [[30.0, 30.0, 40.0, 30.0, 40.0, 40.0, 30.0, 40.0]],
    ]
    label_map = rasterize_masks(polys, 100, 100)
    assert label_map[15, 15] == 0
    assert label_map[35, 35] == 1
    assert label_map[50, 50] == -1


def test_rasterize_later_polygon_wins() -> None:
    """重叠 polygon：后画覆盖（实例 ID 语义）。"""
    polys = [
        [[0.0, 0.0, 50.0, 0.0, 50.0, 50.0, 0.0, 50.0]],
        [[25.0, 25.0, 50.0, 25.0, 50.0, 50.0, 25.0, 50.0]],
    ]
    label_map = rasterize_masks(polys, 100, 100)
    assert label_map[30, 30] == 1
    assert label_map[10, 10] == 0


def test_backproject_hits_and_drops(tmp_path: Any) -> None:
    """合成帧：一个多边形罩住锚点投影像素 → 命中实例 0；实例 1 空 → dropped=1。"""
    frame = synth.write_frame(tmp_path, points=np.array([[*synth.VELO_ANCHOR, 0.0]]))
    # 锚点像素 (758, 299) 的小邻域矩形 + 一个远离的矩形（无点）
    polys = [
        [[748.0, 289.0, 768.0, 289.0, 768.0, 309.0, 748.0, 309.0]],
        [[10.0, 10.0, 20.0, 10.0, 20.0, 20.0, 10.0, 20.0]],
    ]
    result = backproject_semantics(frame, polys)
    assert len(result.points_cam) == 1
    assert result.instance_ids.tolist() == [0]
    assert result.dropped == 1
    assert np.allclose(result.points_cam[0], synth.ANCHOR_CAM, atol=1e-6)


def test_backproject_no_mask_hit(tmp_path: Any) -> None:
    """多边形全部罩不住点 → 全丢（宁缺勿假）。"""
    frame = synth.write_frame(tmp_path, points=np.array([[*synth.VELO_ANCHOR, 0.0]]))
    polys = [[[10.0, 10.0, 20.0, 10.0, 20.0, 20.0, 10.0, 20.0]]]
    result = backproject_semantics(frame, polys)
    assert len(result.points_cam) == 0
    assert result.dropped == 1


def test_backproject_empty_polygons(tmp_path: Any) -> None:
    frame = synth.write_frame(tmp_path, points=np.array([[*synth.VELO_ANCHOR, 0.0]]))
    result = backproject_semantics(frame, [])
    assert len(result.points_cam) == 0
    assert result.dropped == 0


def test_points_of_empty_instance() -> None:
    """points_of 未命中实例 → (0,3)。"""
    from auto3dlabel.tools.backproject import BackprojectResult

    r = BackprojectResult(
        points_cam=np.zeros((0, 3)), instance_ids=np.zeros(0, dtype=np.int32), dropped=0
    )
    assert r.points_of(0).shape == (0, 3)


def test_backproject_invalid_points_excluded(tmp_path: Any) -> None:
    """相机后方点（z<0）不参与查表（valid 掩码）。"""
    pts = np.array([[*synth.VELO_ANCHOR, 0.0], [0.0, 0.0, -100.0, 0.0]])
    frame = synth.write_frame(tmp_path, points=pts)
    polys = [[[748.0, 289.0, 768.0, 289.0, 768.0, 309.0, 748.0, 309.0]]]
    result = backproject_semantics(frame, polys)
    assert len(result.points_cam) == 1
