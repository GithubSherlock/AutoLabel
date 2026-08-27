"""test_cluster：DBSCAN 主簇保留 + 粘连嫌疑 + 体素降采样路径。"""

from __future__ import annotations

import numpy as np

from auto3dlabel.configs.kitti import VOXEL_MAX_POINTS
from auto3dlabel.tests.helpers.synth import rect_points
from auto3dlabel.tools.cluster import _voxel_downsample, cluster_instance


def test_too_few_points_returns_none() -> None:
    pts = rect_points(10.0, 20.0, 4.0, 2.0, 0.0, n=10)
    assert cluster_instance(pts, "Car") is None


def test_single_cluster_kept() -> None:
    pts = rect_points(10.0, 20.0, 4.0, 2.0, 0.0, n=400)
    result = cluster_instance(pts, "Car")
    assert result is not None
    assert len(result.points_cam) > 300  # 几乎全部保留
    assert not result.adhesion_flag
    # 中心恢复
    assert abs(float(result.points_cam[:, 0].mean()) - 10.0) < 0.1
    assert abs(float(result.points_cam[:, 2].mean()) - 20.0) < 0.1


def test_background_cluster_removed() -> None:
    """主簇 + 远处小背景簇（光线穿透模拟）→ 背景丢弃。"""
    main = rect_points(10.0, 20.0, 4.0, 2.0, 0.0, n=400)
    bg = rect_points(10.0, 40.0, 1.0, 1.0, 0.0, n=30)
    result = cluster_instance(np.vstack([main, bg]), "Car")
    assert result is not None
    assert len(result.points_cam) >= 350  # 主簇保留
    assert float(result.points_cam[:, 2].max()) < 25.0  # 背景簇已剔除


def test_adhesion_flag_when_second_cluster_large() -> None:
    """次大簇 ≥ 主簇 30% → adhesion_flag（两实例同 mask 嫌疑）。"""
    a = rect_points(10.0, 20.0, 4.0, 2.0, 0.0, n=200)
    b = rect_points(10.0, 26.0, 3.0, 1.5, 0.0, n=80)  # 40% → adhesion
    result = cluster_instance(np.vstack([a, b]), "Car")
    assert result is not None and result.adhesion_flag
    assert len(result.points_cam) >= 190  # 主簇仍是 a


def test_voxel_downsample_mapping() -> None:
    """降采样：inv 映射长度 = 代表点数，聚类标签回射到原始点。"""
    pts = rect_points(10.0, 20.0, 4.0, 2.0, 0.0, n=VOXEL_MAX_POINTS + 100)
    result = cluster_instance(pts, "Car")
    assert result is not None
    assert len(result.points_cam) > 20000 * 0.8  # 大体量保留
    assert result.points_cam.shape[1] == 3


def test_voxel_helpers() -> None:
    pts = np.array([[0.0, 0.0], [0.05, 0.05], [1.0, 1.0]])
    reps, inv = _voxel_downsample(pts)
    assert len(reps) == 2
    assert len(inv) == 3
    assert inv[0] == inv[1] and inv[2] != inv[0]


def test_unknown_label_default_eps() -> None:
    pts = rect_points(5.0, 8.0, 2.0, 1.0, 0.0, n=200)
    assert cluster_instance(pts, "Mystery") is not None
