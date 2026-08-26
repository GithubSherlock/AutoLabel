"""test_nuscenes_data：四元数转换 + NusBox + dataroot 守卫 + 假 nusc 表枚举/GT（零真实数据）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from auto3dlabel.data.nuscenes import (
    dataroot_exists,
    gt_boxes_of_sample,
    load_nuscenes,
    samples_of_scene,
)
from auto3dlabel.schema.nuscenes_box import NusBox
from auto3dlabel.tools.geometry import quat_to_yaw, yaw_to_quat


def test_yaw_quat_roundtrip_and_anchors() -> None:
    """yaw→quat→yaw 往返 + 锚点值（0/π/2/-π/2）。"""
    for yaw in (0.0, np.pi / 2, -np.pi / 2, 2.4):
        assert abs(quat_to_yaw(yaw_to_quat(yaw)) - yaw) < 1e-9
    w, x, y, z = yaw_to_quat(np.pi / 2)
    assert abs(w - np.cos(np.pi / 4)) < 1e-9 and x == 0.0 and y == 0.0
    assert abs(z - np.sin(np.pi / 4)) < 1e-9


def test_nusbox_to_dict_velocity_optional() -> None:
    """to_dict：官方提交字段结构；velocity None 不输出。"""
    box = NusBox(
        label="car", confidence=0.8, translation=(1.0, 2.0, 0.5),
        size=(2.0, 4.0, 1.5), quaternion=(1.0, 0.0, 0.0, 0.0), track_id="inst-1",
    )
    d = box.to_dict()
    assert d == {
        "detection_name": "car",
        "detection_score": 0.8,
        "translation": [1.0, 2.0, 0.5],
        "size": [2.0, 4.0, 1.5],
        "rotation": [1.0, 0.0, 0.0, 0.0],
    }
    box.velocity = (3.0, 0.0)
    assert box.to_dict()["velocity"] == [3.0, 0.0]


def test_dataroot_exists(tmp_path: Path) -> None:
    """dataroot 校验：v1.0-mini 表 + LIDAR_TOP samples 齐备才算存在。"""
    (tmp_path / "v1.0-mini").mkdir()
    assert not dataroot_exists(tmp_path)
    (tmp_path / "samples" / "LIDAR_TOP").mkdir(parents=True)
    assert dataroot_exists(tmp_path)


def test_load_nuscenes_missing_dataroot_raises(tmp_path: Path) -> None:
    """dataroot 不存在 → FileNotFoundError（devkit 已装时走数据校验路径）。"""
    with pytest.raises(FileNotFoundError, match="dataroot"):
        load_nuscenes(root=tmp_path / "nope")


def test_load_nuscenes_devkit_missing_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """devkit 未装 → ImportError 带 --no-deps 指引（守卫路径）。"""
    monkeypatch.setitem(sys.modules, "nuscenes.nuscenes", None)
    with pytest.raises(ImportError, match="nuscenes-devkit"):
        load_nuscenes(root=tmp_path)


class _FakeNusc:
    """假 devkit 表：1 场景 2 sample + 3 标注（含 1 条 num_lidar_pts=0 剔除）。"""

    def __init__(self) -> None:
        self.scene = [{"name": "scene-A", "first_sample_token": "s0"}]
        self._samples = {
            "s0": {"token": "s0", "next": "s1"},
            "s1": {"token": "s1", "next": ""},
        }
        self.sample_annotation = [
            {
                "sample_token": "s0", "category_name": "vehicle.car",
                "translation": [1.0, 2.0, 3.0], "size": [2.0, 4.0, 1.5],
                "rotation": [1.0, 0.0, 0.0, 0.0], "instance_token": "i1",
                "num_lidar_pts": 100,
            },
            {
                "sample_token": "s0", "category_name": "vehicle.bus",
                "translation": [5.0, 0.0, 0.0], "size": [2.5, 10.0, 3.0],
                "rotation": [1.0, 0.0, 0.0, 0.0], "instance_token": "i2",
                "num_lidar_pts": 0,
            },
            {
                "sample_token": "s1", "category_name": "human.pedestrian",
                "translation": [0.0, 0.0, 0.0], "size": [0.6, 0.7, 1.8],
                "rotation": [1.0, 0.0, 0.0, 0.0], "instance_token": "i3",
                "num_lidar_pts": 30,
            },
        ]

    def get(self, table: str, token: str) -> dict:
        assert table == "sample"
        return self._samples[token]


def test_samples_of_scene_chain_and_missing() -> None:
    """first_sample_token 链序枚举；场景不存在 → KeyError。"""
    nusc = _FakeNusc()
    samples = samples_of_scene(nusc, "scene-A")
    assert [s["token"] for s in samples] == ["s0", "s1"]
    with pytest.raises(KeyError):
        samples_of_scene(nusc, "scene-X")


def test_gt_boxes_of_sample_filters_and_maps() -> None:
    """GT 直映射（类名剥前缀）+ num_lidar_pts=0 剔除 + instance_token 穿透。"""
    nusc = _FakeNusc()
    boxes = gt_boxes_of_sample(nusc, "s0")
    assert len(boxes) == 1  # bus 被剔除
    assert boxes[0].label == "car"
    assert boxes[0].translation == (1.0, 2.0, 3.0)
    assert boxes[0].track_id == "i1"
    assert gt_boxes_of_sample(nusc, "s1")[0].label == "pedestrian"
