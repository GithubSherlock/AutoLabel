"""test_nuscenes_data：四元数转换 + NusBox + dataroot 守卫 + 假 nusc 表枚举/GT
+ 相机系→全局转换（P6b boxes_cam_to_global，零真实数据）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from auto3dlabel.configs.nuscenes import NUSCENES_CAMERAS
from auto3dlabel.data.nuscenes import (
    QX_HALF_PI,
    box3d_dict_to_nusbox,
    boxes_cam_to_global,
    cameras_of_sample,
    dataroot_exists,
    ego_translation_of_sample,
    gt_boxes_of_sample,
    lidar_sample_data,
    load_lidar_points,
    load_nuscenes,
    nusbox_to_box3d_dict,
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


# ── P2：点云加载 / ego 位姿 / 6 相机 / NusBox↔渲染 dict 往返 ────

class _FakeP2Nusc:
    """假 devkit 表（P2 函数用）：sample_data（LIDAR_TOP + 6 相机）+ ego_pose。"""

    def __init__(self) -> None:
        self._sd = {
            "sd-lidar": {
                "filename": "samples/LIDAR_TOP/n008-0.bin",
                "calibrated_sensor_token": "cs1",
                "ego_pose_token": "ep1",
            },
        }
        for i, name in enumerate(NUSCENES_CAMERAS):
            self._sd[f"sd-{name}"] = {
                "token": f"t{i}",
                "filename": f"samples/{name}/cam{i}.jpg",
            }
        self._ego = {
            "ep1": {"translation": [10.0, 20.0, 0.5], "rotation": [1.0, 0.0, 0.0, 0.0]}
        }

    def get(self, table: str, token: str) -> dict:
        if table == "sample_data":
            return self._sd[token]
        assert table == "ego_pose"
        return self._ego[token]


def test_load_lidar_points_reads_file_and_drops_nan(tmp_path: Path) -> None:
    """(N,5) 直读 + NaN 行剔除（文件名经 sample_data 表映射）。"""
    pts = np.array(
        [[1.0, 2.0, 3.0, 0.5, 1.0], [np.nan, 0.0, 0.0, 0.0, 0.0],
         [4.0, 5.0, 6.0, 0.8, 0.0]],
        dtype=np.float32,
    )
    (tmp_path / "samples" / "LIDAR_TOP").mkdir(parents=True)
    pts.tofile(tmp_path / "samples" / "LIDAR_TOP" / "n008-0.bin")
    nusc = _FakeP2Nusc()
    sample = {"data": {"LIDAR_TOP": "sd-lidar"}}
    out = load_lidar_points(sample, nusc, tmp_path)
    assert out.shape == (2, 5) and out.dtype == np.float32
    np.testing.assert_allclose(out, pts[[0, 2]], atol=1e-6)


def test_ego_translation_of_sample() -> None:
    """LIDAR_TOP 记录的 ego_pose → 全局位置 (x,y,z)。"""
    nusc = _FakeP2Nusc()
    sample = {"data": {"LIDAR_TOP": "sd-lidar"}}
    assert ego_translation_of_sample(sample, nusc) == (10.0, 20.0, 0.5)


def test_cameras_of_sample_six_views() -> None:
    """6 相机标准序（NUSCENES_CAMERAS）+ filename/token 透出。"""
    nusc = _FakeP2Nusc()
    sample = {
        "data": {name: f"sd-{name}" for name in NUSCENES_CAMERAS}
        | {"LIDAR_TOP": "sd-lidar"}
    }
    cams = cameras_of_sample(sample, nusc)
    assert [c["name"] for c in cams] == list(NUSCENES_CAMERAS)
    assert all(c["filename"].startswith(f"samples/{c['name']}/") for c in cams)
    assert cams[0]["token"] == "t0"


def test_nusbox_to_box3d_dict_identity_anchor() -> None:
    """恒等姿态锚点：translation (1,2,0.5) size (2,4,1.5) → (−ty,−tz,tx)/(h,w,l)/ry=−π/2。"""
    box = NusBox(
        label="car", confidence=0.8, translation=(1.0, 2.0, 0.5),
        size=(2.0, 4.0, 1.5), quaternion=(1.0, 0.0, 0.0, 0.0),
        velocity=(3.0, -1.0), track_id="inst-9",
    )
    d = nusbox_to_box3d_dict(box, fit_points=12)
    assert d["label"] == "car" and d["confidence"] == 0.8
    assert (d["cx"], d["cy"], d["cz"]) == (-2.0, -0.5, 1.0)
    assert (d["h"], d["w"], d["l"]) == (1.5, 2.0, 4.0)
    assert abs(d["rotation_y"] + np.pi / 2) < 1e-9
    assert d["fit_points"] == 12
    assert d["velocity"] == [3.0, -1.0] and d["track_id"] == "inst-9"


def test_nusbox_to_box3d_dict_yaw_anchor() -> None:
    """yaw_g=0.3 → rotation_y = −0.3 − π/2（yaw_bev=−yaw_g 代入唯一转换点）。"""
    box = NusBox(
        label="car", translation=(0.0, 0.0, 0.0), size=(2.0, 4.0, 1.5),
        quaternion=yaw_to_quat(0.3),
    )
    d = nusbox_to_box3d_dict(box)
    assert abs(d["rotation_y"] - (-0.3 - np.pi / 2)) < 1e-9


def test_nusbox_dict_roundtrip() -> None:
    """渲染 dict → NusBox 逆变换全字段往返（yaw 经 quat_to_yaw 比较）。"""
    box = NusBox(
        label="truck", confidence=0.6, translation=(7.5, -2.0, 1.2),
        size=(2.5, 8.0, 3.0), quaternion=yaw_to_quat(1.1),
        velocity=(0.0, 4.0), track_id="t-1",
    )
    back = box3d_dict_to_nusbox(nusbox_to_box3d_dict(box))
    assert back.label == box.label
    assert back.confidence == pytest.approx(box.confidence)
    assert back.translation == pytest.approx(box.translation)
    assert back.size == pytest.approx(box.size)
    assert abs(quat_to_yaw(back.quaternion) - quat_to_yaw(box.quaternion)) < 1e-9
    assert back.velocity == pytest.approx(box.velocity)
    assert back.track_id == "t-1"


def test_nusbox_dict_roundtrip_optional_keys_and_edit() -> None:
    """velocity/track_id 缺省 → 往返 None；前端编辑 rotation_y 后正确回写全局 yaw。"""
    d = nusbox_to_box3d_dict(
        NusBox(
            label="pedestrian", translation=(1.0, 0.0, 0.0), size=(0.6, 0.7, 1.8),
            quaternion=(1.0, 0.0, 0.0, 0.0),
        )
    )
    assert "velocity" not in d and "track_id" not in d
    d["rotation_y"] = 0.0  # 模拟前端编辑：ry=0 → 车头 cam +x = 全局 −y（yaw_g=−π/2）
    back = box3d_dict_to_nusbox(d)
    assert abs(quat_to_yaw(back.quaternion) + np.pi / 2) < 1e-9
    assert back.velocity is None and back.track_id is None


def test_nusbox_dict_roundtrip_with_ego_offset() -> None:
    """ego 偏移：center = M@(t−t_ego) → 逆变换 t = Mᵀ@c + t_ego 全字段往返。"""
    ego = (3.0, 1.0, 0.5)
    box = NusBox(
        label="bus", confidence=0.9, translation=(7.5, -2.0, 1.2),
        size=(2.5, 10.0, 3.0), quaternion=yaw_to_quat(-0.8),
        velocity=(5.0, 0.0), track_id="b-1",
    )
    d = nusbox_to_box3d_dict(box, ego_translation=ego)
    # 锚点：t−ego = (4.5,−3,0.7) → M@ = (3,−0.7,4.5)
    assert (d["cx"], d["cy"], d["cz"]) == (3.0, -0.7, 4.5)
    back = box3d_dict_to_nusbox(d, ego_translation=ego)
    assert back.translation == pytest.approx(box.translation)
    assert back.size == pytest.approx(box.size)
    assert abs(quat_to_yaw(back.quaternion) - quat_to_yaw(box.quaternion)) < 1e-9
    assert back.velocity == pytest.approx(box.velocity)
    assert back.track_id == "b-1"
