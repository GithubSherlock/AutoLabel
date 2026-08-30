"""test_nuscenes_data：四元数转换 + NusBox + dataroot 守卫 + 假 nusc 表枚举/GT
+ 相机系→全局转换（P6b boxes_cam_to_global，零真实数据）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from auto3dlabel.data.nuscenes import (
    QX_HALF_PI,
    boxes_cam_to_global,
    dataroot_exists,
    gt_boxes_of_sample,
    lidar_sample_data,
    load_nuscenes,
    quat_mul,
    rot_matrix,
    samples_of_scene,
    sensor_sample_data,
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
                "sample_token": "s0", "category_name": "vehicle.bus.rigid",
                "translation": [5.0, 0.0, 0.0], "size": [2.5, 10.0, 3.0],
                "rotation": [1.0, 0.0, 0.0, 0.0], "instance_token": "i2",
                "num_lidar_pts": 0,
            },
            {
                "sample_token": "s1", "category_name": "human.pedestrian.adult",
                "translation": [0.0, 0.0, 0.0], "size": [0.6, 0.7, 1.8],
                "rotation": [1.0, 0.0, 0.0, 0.0], "instance_token": "i3",
                "num_lidar_pts": 30,
            },
            {
                "sample_token": "s1", "category_name": "animal",
                "translation": [9.0, 9.0, 0.0], "size": [0.5, 1.0, 0.8],
                "rotation": [1.0, 0.0, 0.0, 0.0], "instance_token": "i4",
                "num_lidar_pts": 10,
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
    """GT 官方 category 映射（子类聚合）+ num_lidar_pts=0 剔除 + instance_token 穿透。"""
    nusc = _FakeNusc()
    boxes = gt_boxes_of_sample(nusc, "s0")
    assert len(boxes) == 1  # bus.rigid 被 num_lidar_pts=0 剔除
    assert boxes[0].label == "car"
    assert boxes[0].translation == (1.0, 2.0, 3.0)
    assert boxes[0].track_id == "i1"
    s1 = gt_boxes_of_sample(nusc, "s1")
    assert [b.label for b in s1] == ["pedestrian"]  # adult 子类映射 + animal 忽略类跳过


class _FakeSampleNusc:
    """假 sample_data 表（sensor_sample_data 用）：LIDAR_TOP + CAM_FRONT 两通道。"""

    def __init__(self) -> None:
        self._data = {
            "sd-lidar": {
                "filename": "samples/LIDAR_TOP/x.pcd",
                "calibrated_sensor_token": "cs1",
                "ego_pose_token": "ep1",
            },
            "sd-camf": {
                "filename": "samples/CAM_FRONT/y.jpg",
                "calibrated_sensor_token": "cs2",
                "ego_pose_token": "ep1",
            },
        }

    def get(self, table: str, token: str) -> dict:
        assert table == "sample_data"
        return self._data[token]


def test_sensor_sample_data_generalized() -> None:
    """sensor_sample_data 泛化：指定通道取对应记录；lidar_sample_data 薄壳向后兼容。"""
    nusc = _FakeSampleNusc()
    sample = {"data": {"LIDAR_TOP": "sd-lidar", "CAM_FRONT": "sd-camf"}}
    assert sensor_sample_data(sample, nusc, "CAM_FRONT")["filename"].endswith("y.jpg")
    assert lidar_sample_data(sample, nusc)["filename"].endswith("x.pcd")


IDENTITY_POSE = {"rotation": [1.0, 0.0, 0.0, 0.0], "translation": [0.0, 0.0, 0.0]}
NUS_CLASSES = ["car", "truck", "trailer", "bus", "construction_vehicle",
               "bicycle", "motorcycle", "pedestrian", "traffic_cone", "barrier"]
HALF_SQRT2 = 0.7071067811865476


def test_boxes_cam_to_global_identity() -> None:
    """恒等位姿：几何中心/dims 直传 + q_local = q2⊗q1（yaw=0 → (√2/2,√2/2,0,0)）。

    v0.15 output_to_nusc_box 定式锚点：q2 = 绕相机 x 转 π/2、q1 = 绕 z 转 yaw、
    velocity = 相机 x-z 平面 [vx,0,vz] 随位姿旋转 → 全局 x-y。
    """
    boxes = np.asarray([[1.0, 2.0, 3.0, 1.6, 4.0, 1.5, 0.0, 2.0, 0.5]])
    scores = np.asarray([0.9])
    labels = np.asarray([0])
    out = boxes_cam_to_global(
        boxes, scores, labels, NUS_CLASSES, IDENTITY_POSE, IDENTITY_POSE
    )
    assert len(out) == 1
    box = out[0]
    assert box.label == "car" and box.confidence == pytest.approx(0.9)
    assert box.translation == (1.0, 2.0, 3.0)  # 几何中心直传
    assert box.size == (1.6, 4.0, 1.5)  # (w,l,h) 直传——旧 converter 语义
    assert box.quaternion == pytest.approx((HALF_SQRT2, HALF_SQRT2, 0.0, 0.0))
    # [vx,0,vz] 随位姿旋转 → 全局 x-y 平面 (vx,vy)（vz 成全局 z 分量，官方提交丢弃）
    assert box.velocity == pytest.approx((2.0, 0.0))


def test_boxes_cam_to_global_yaw_pi_half_anchor() -> None:
    """yaw=π/2 锚点：q_local = (1/2,1/2,-1/2,1/2)；车头（局部 +x）→ 相机 +z。

    R(q2⊗q1) = R_x(π/2)R_z(π/2) = [[0,-1,0],[0,0,-1],[1,0,0]]（手算）——
    box 局部长度轴 +x → (0,0,1) = 相机光轴（前向）✓。
    """
    boxes = np.asarray([[0.0, 0.0, 10.0, 1.6, 4.0, 1.5, np.pi / 2, 0.0, 0.0]])
    out = boxes_cam_to_global(
        boxes, np.asarray([0.9]), np.asarray([0]), NUS_CLASSES,
        IDENTITY_POSE, IDENTITY_POSE,
    )
    q = out[0].quaternion
    assert q == pytest.approx((0.5, 0.5, -0.5, 0.5))
    forward_glob = rot_matrix(q) @ np.array([1.0, 0.0, 0.0])
    assert forward_glob == pytest.approx((0.0, 0.0, 1.0))  # 相机 +z（光轴）


def test_boxes_cam_to_global_known_calib() -> None:
    """已知 calib 位姿：calib 绕 z 转 90° + 平移 [10,0,0]，ego 恒等。

    几何中心 (1,2,3) → R_z90·c + t = (8,1,3)；速度 [2,0,0.5] → (0,2,0.5)→(0,2)；
    姿态 rot_matrix(quat) == R_calib @ rot_matrix(q_local)。
    """
    z90 = yaw_to_quat(np.pi / 2)  # (√2/2,0,0,√2/2)
    calib = {"rotation": list(z90), "translation": [10.0, 0.0, 0.0]}
    boxes = np.asarray([[1.0, 2.0, 3.0, 1.6, 4.0, 1.5, 0.0, 2.0, 0.5]])
    out = boxes_cam_to_global(
        boxes, np.asarray([0.9]), np.asarray([0]), NUS_CLASSES, IDENTITY_POSE, calib
    )
    box = out[0]
    assert box.translation == pytest.approx((8.0, 1.0, 3.0))
    assert box.velocity == pytest.approx((0.0, 2.0))
    # 姿态一致性：rot_matrix(quat) == R_calib @ R(q_local)，q_local 用模块函数复算
    q_local = quat_mul(QX_HALF_PI, yaw_to_quat(0.0))  # 本框 yaw=0
    expected = rot_matrix(z90) @ rot_matrix(q_local)
    np.testing.assert_allclose(rot_matrix(box.quaternion), expected, atol=1e-9)


def test_boxes_cam_to_global_label_drop_and_no_velocity() -> None:
    """标签越界丢弃 + stderr 诊断；7 列（无速度）→ velocity None。"""
    boxes = np.asarray([[0.0, 0.0, 5.0, 1.0, 3.0, 1.5, 0.0]])  # 无速度列
    scores = np.asarray([0.8, 0.7])
    labels = np.asarray([2, 42])
    out = boxes_cam_to_global(
        boxes[:1], scores[:1], labels[:1], NUS_CLASSES, IDENTITY_POSE, IDENTITY_POSE
    )
    assert len(out) == 1
    assert out[0].label == "trailer" and out[0].velocity is None
    out2 = boxes_cam_to_global(
        np.repeat(boxes, 2, axis=0), scores, labels, NUS_CLASSES,
        IDENTITY_POSE, IDENTITY_POSE,
    )
    assert len(out2) == 1  # idx=42 越界丢弃（10 类表）
