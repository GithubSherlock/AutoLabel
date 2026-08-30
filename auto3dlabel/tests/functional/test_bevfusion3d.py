"""test_bevfusion3d：BEVFusion 封装（v0.3 P3；零真实权重铁律 + 守卫路径双态兼容）。

mmdet3d 未装状态下全绿是设计目标（ImportError 守卫在 _load）；已装环境
守卫测试自动跳过（skipif）——双态质量门均归零。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from auto3dlabel.configs.kitti import MMDET3D_CONFIG_DIR, WEIGHTS_DIR
from auto3dlabel.configs.model_catalog import BEVFUSION_NAMES
from auto3dlabel.configs.nuscenes import NUSCENES_CAMERAS
from auto3dlabel.models.bevfusion3d import (
    BevFusionDetector,
    _normalize_bevfusion_boxes,
    _transpose_sparse_conv_weights,
    build_bevfusion_data,
    create_bevfusion_detector,
)

MMDET3D_INSTALLED = importlib.util.find_spec("mmdet3d") is not None


class _FakeNusc:
    """FakeNusc 桩：get('sample_data'|'calibrated_sensor') 行为对齐 devkit 表。"""

    def __init__(
        self,
        cam_translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        cam_rotation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        lidar_translation: tuple[float, float, float] = (1.0, 0.0, 0.0),
        lidar_rotation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        timestamp: int = 1531883530447123,
    ) -> None:
        self._sd: dict[str, dict] = {
            "LIDAR_TOP": {
                "filename": "samples/LIDAR_TOP/lidar.pcd.bin",
                "calibrated_sensor_token": "lidar_calib",
                "timestamp": timestamp,
            }
        }
        self._calib: dict[str, dict] = {
            "lidar_calib": {
                "translation": list(lidar_translation),
                "rotation": list(lidar_rotation),
            }
        }
        for cam in NUSCENES_CAMERAS:
            self._sd[cam] = {
                "filename": f"samples/{cam}/img.jpg",
                "calibrated_sensor_token": f"{cam}_calib",
                "timestamp": timestamp,
            }
            self._calib[f"{cam}_calib"] = {
                "translation": list(cam_translation),
                "rotation": list(cam_rotation),
                "camera_intrinsic": [
                    [721.5, 0.0, 609.6],
                    [0.0, 721.5, 172.9],
                    [0.0, 0.0, 1.0],
                ],
            }

    def get(self, table: str, token: str) -> dict:
        if table == "sample_data":
            return self._sd[token]
        if table == "calibrated_sensor":
            return self._calib[token]
        raise KeyError(table)


SAMPLE: dict = {
    "data": {cam: cam for cam in NUSCENES_CAMERAS} | {"LIDAR_TOP": "LIDAR_TOP"}
}


def test_build_bevfusion_data_contract(tmp_path: Path) -> None:
    """data_ dict 契约：6 相机键集 + cam2img 3x3 / lidar2cam 4x4 数值 + timestamp 秒。"""
    nusc = _FakeNusc()
    data = build_bevfusion_data(nusc, SAMPLE, tmp_path)
    assert data["box_type_3d"] is None  # 纯函数零 mmdet3d 依赖，由 _load 提供
    assert data["box_mode_3d"] is None
    assert set(data["images"]) == set(NUSCENES_CAMERAS)
    assert data["lidar_points"]["lidar_path"].endswith(".pcd.bin")
    # timestamp 微秒 → 秒（pkl infos 约定）
    assert data["timestamp"] == pytest.approx(1531883530.447123)
    for cam, info in data["images"].items():
        assert info["img_path"] == str(tmp_path / f"samples/{cam}/img.jpg")
        assert info["cam2img"].shape == (3, 3)
        assert info["lidar2cam"].shape == (4, 4)
        # 相机位姿恒等、LIDAR 平移 (1,0,0) → lidar2cam = inv(I) @ T_lidar
        # （LIDAR 原点在相机系 +1m 处，平移为正）
        expected = np.eye(4)
        expected[:3, 3] = [1.0, 0.0, 0.0]
        np.testing.assert_allclose(info["lidar2cam"], expected, atol=1e-9)


def test_build_bevfusion_data_rotated_calib(tmp_path: Path) -> None:
    """标定链红线：相机绕 z 轴 +90° → lidar2cam = Rz(-90°)（位姿链数值锁定）。"""
    s2 = 0.5**0.5  # sin/cos 45°
    nusc = _FakeNusc(
        cam_rotation=(s2, 0.0, 0.0, s2),  # Rz(+90°)
        lidar_translation=(0.0, 0.0, 0.0),
    )
    data = build_bevfusion_data(nusc, SAMPLE, tmp_path)
    expected = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    for info in data["images"].values():
        np.testing.assert_allclose(info["lidar2cam"], expected, atol=1e-9)


def test_normalize_bevfusion_boxes_identity() -> None:
    """decode 输出即契约 [x,y,z,l,w,h,yaw,vx,vy]（实证锁定回归：曾误 swap → car 全灭）。"""
    boxes = np.asarray(
        [[1.0, 2.0, 3.0, 3.9, 1.6, 1.5, 0.1, 0.2, -0.3]], dtype=np.float32
    )
    norm = _normalize_bevfusion_boxes(boxes)
    assert norm.shape == (1, 9)
    np.testing.assert_allclose(norm[0], [1.0, 2.0, 3.0, 3.9, 1.6, 1.5, 0.1, 0.2, -0.3])
    # 7 值（无速度）同样恒等；非法输入原样返回
    np.testing.assert_allclose(
        _normalize_bevfusion_boxes(boxes[:, :7])[0],
        [1.0, 2.0, 3.0, 3.9, 1.6, 1.5, 0.1],
    )
    passthrough = np.zeros((2, 3), dtype=np.float32)
    assert _normalize_bevfusion_boxes(passthrough) is passthrough


def test_transpose_sparse_conv_weights() -> None:
    """稀疏卷积权重换序（零权重回归）：prefix 5D 键 permute(1,2,3,4,0)，其余键原样。"""
    w = torch.arange(2 * 3 * 3 * 3 * 4, dtype=torch.float32).reshape(2, 3, 3, 3, 4)
    state = {
        "pts_middle_encoder.conv_input.0.weight": w,
        "pts_middle_encoder.conv_input.0.bias": torch.zeros(16),  # 1D 不换
        "img_backbone.layer.0.weight": torch.zeros(2, 3, 3, 3, 4),  # 非 prefix 不动
    }
    out = _transpose_sparse_conv_weights(state)
    assert set(out) == {"pts_middle_encoder.conv_input.0.weight"}
    torch.testing.assert_close(
        out["pts_middle_encoder.conv_input.0.weight"], w.permute(1, 2, 3, 4, 0)
    )


def test_create_bevfusion_detector_routing() -> None:
    """工厂路由：None/未知名/LiDAR 引擎名 → None；bevfusion_nus → 正确路径。"""
    assert create_bevfusion_detector(None) is None
    assert create_bevfusion_detector("pointpillars_nus") is None  # LiDAR 名不误路由
    assert create_bevfusion_detector("unknown3d") is None
    det = create_bevfusion_detector("bevfusion_nus")
    assert isinstance(det, BevFusionDetector)
    entry = BEVFUSION_NAMES["bevfusion_nus"]
    assert det._config_path == str(MMDET3D_CONFIG_DIR / entry["config"])
    assert det._checkpoint_path == str(WEIGHTS_DIR / entry["weights_dir"] / entry["checkpoint"])


@pytest.mark.skipif(MMDET3D_INSTALLED, reason="mmdet3d 已装，守卫路径跳过")
def test_bevfusion_load_guard_import() -> None:
    """mmdet3d 未装 → _load 抛 ImportError（守卫路径；已装环境自动跳过）。"""
    det = BevFusionDetector("missing_config.py", "missing.pth")
    with pytest.raises(ImportError):
        det._load()
