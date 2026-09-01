"""test_nuscenes_pipeline：三协议分派（detect_points / detect_sample 4 参 / 3 参）
+ queue_one_sample 注入 + generate_review_queue resume 幂等（Fake 零真实权重）。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import auto3dlabel.tools.nuscenes_pipeline as nsp
from auto3dlabel.configs.nuscenes import NUSCENES_CAMERAS
from auto3dlabel.schema.nuscenes_box import NusBox
from auto3dlabel.tools.nuscenes_pipeline import (
    generate_review_queue,
    predict_sample_nus,
    queue_one_sample,
)

NUS_CLASSES = ["car", "truck", "pedestrian"]


class _FakeNusc:
    """假 devkit 表：sample_data（LIDAR_TOP + 6 相机）+ 恒等 ego_pose/calibrated_sensor。"""

    version = "v1.0-mini"
    dataroot = "/tmp/fake-root"

    def __init__(self) -> None:
        self._sd = {"sd-lidar": {
            "filename": "samples/LIDAR_TOP/n.bin",
            "calibrated_sensor_token": "cs1",
            "ego_pose_token": "ep1",
        }}
        for i, name in enumerate(NUSCENES_CAMERAS):
            self._sd[f"sd-{name}"] = {"filename": f"samples/{name}/c{i}.jpg", "token": f"t{i}"}
        self._pose = {"rotation": [1.0, 0.0, 0.0, 0.0], "translation": [0.0, 0.0, 0.0]}

    def get(self, table: str, token: str) -> dict:
        if table == "sample_data":
            return self._sd[token]
        return self._pose  # ego_pose / calibrated_sensor 恒等


def _sample() -> dict:
    return {"token": "s0", "data": {
        "LIDAR_TOP": "sd-lidar", **{n: f"sd-{n}" for n in NUSCENES_CAMERAS},
    }}


class _FakeLidarDet:
    """detect_points 协议：返回 1 个传感器系框 (1,2,0,l,w,h,0)。"""

    class_names = tuple(NUS_CLASSES)

    def __init__(self, boxes: np.ndarray | None = None) -> None:
        self._boxes = boxes if boxes is not None else np.asarray(
            [[1.0, 2.0, 0.0, 1.6, 4.0, 1.5, 0.0, 0.0, 0.0]]
        )
        self.last_pts: np.ndarray | None = None

    def detect_points(
        self, pts: np.ndarray, conf: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.last_pts = pts
        return self._boxes, np.asarray([0.8]), np.asarray([0])


class _FakeFusionDet:
    """detect_sample 4 位置参协议（bevfusion 形）：同 9 值传感器系契约。"""

    class_names = tuple(NUS_CLASSES)

    def __init__(self) -> None:
        self.last_dataroot: Path | None = None

    def detect_sample(
        self, sample: dict, nusc: object, dataroot: Path, conf_threshold: float | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.last_dataroot = dataroot
        return (
            np.asarray([[3.0, 0.0, 0.0, 1.6, 4.0, 1.5, 0.0]]),
            np.asarray([0.7]),
            np.asarray([1]),
        )


class _FakeMonoDet:
    """detect_sample 3 位置参协议（fcos3d 形）：直接返回全局 NusBox。"""

    class_names = tuple(NUS_CLASSES)

    def detect_sample(
        self, sample: dict, nusc: object, conf_threshold: float | None = None
    ) -> list[NusBox]:
        return [NusBox(label="pedestrian", confidence=0.6, translation=(9.0, 1.0, 0.0),
                       size=(0.6, 0.7, 1.8), quaternion=(1.0, 0.0, 0.0, 0.0))]


def _write_lidar_bin(tmp_path: Path, fake: _FakeNusc) -> Path:
    bin_path = tmp_path / fake._sd["sd-lidar"]["filename"]
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    np.zeros((8, 5), dtype=np.float32).tofile(bin_path)
    return tmp_path


def test_predict_sample_nus_lidar_engine(tmp_path: Path) -> None:
    """detect_points → load_lidar_points 喂点云 → 传感器系 9 值 → 恒等位姿全局 NusBox。"""
    fake = _FakeNusc()
    dataroot = _write_lidar_bin(tmp_path, fake)
    det = _FakeLidarDet()
    out = predict_sample_nus(det, _sample(), fake, dataroot, 0.3, NUS_CLASSES)
    assert det.last_pts is not None and det.last_pts.shape == (8, 5)  # 点云已读
    assert len(out) == 1
    box = out[0]
    assert box.label == "car" and box.confidence == pytest.approx(0.8)
    assert box.translation == pytest.approx((1.0, 2.0, 0.0))  # 恒等位姿直映射
    assert box.size == pytest.approx((4.0, 1.6, 1.5))  # (w,l,h)：输入 (l,w,h) 重排


def test_predict_sample_nus_fusion_dispatch(tmp_path: Path) -> None:
    """detect_sample 4 位置参 → 融合分支（dataroot 透传 + 同补偿链）。"""
    fake = _FakeNusc()
    dataroot = _write_lidar_bin(tmp_path, fake)
    det = _FakeFusionDet()
    out = predict_sample_nus(det, _sample(), fake, dataroot, 0.3, NUS_CLASSES)
    assert det.last_dataroot == dataroot
    assert len(out) == 1
    assert out[0].label == "truck" and out[0].translation == pytest.approx((3.0, 0.0, 0.0))


def test_predict_sample_nus_mono_passthrough() -> None:
    """detect_sample 3 位置参 → 单目分支：NusBox 直返（不再过补偿链）。"""
    out = predict_sample_nus(
        _FakeMonoDet(), _sample(), _FakeNusc(), Path("/tmp/x"), 0.3, NUS_CLASSES
    )
    assert len(out) == 1
    assert out[0].label == "pedestrian"
    assert out[0].translation == pytest.approx((9.0, 1.0, 0.0))


def test_predict_sample_nus_unknown_protocol_raises() -> None:
    """无 detect_points/detect_sample → TypeError。"""
    with pytest.raises(TypeError, match="协议"):
        predict_sample_nus(object(), _sample(), _FakeNusc(), Path("/tmp/x"), 0.3, NUS_CLASSES)


def test_queue_one_sample_writes_file(tmp_path: Path) -> None:
    """predict_fn 注入 → 队列文件写出（不依赖真实模型）。"""
    fake = _FakeNusc()
    dataroot = _write_lidar_bin(tmp_path, fake)

    def predict_fn(_sample: dict) -> list[NusBox]:
        return [NusBox(
            label="car", confidence=0.5, translation=(40.0, 0.0, 0.0),
            size=(2.0, 4.0, 1.5), quaternion=(1.0, 0.0, 0.0, 0.0),
        )]

    path = queue_one_sample(predict_fn, _sample(), "scene-A", fake, dataroot, tmp_path / "q")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["dataset"] == "nuscenes"
    assert data["scene_name"] == "scene-A"
    assert data["summary"]["review_count"] == 1  # conf 0.5 + fit 0 → review


def test_generate_review_queue_resume_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """resume 幂等：首跑写 2 个文件，二跑全跳过（written 空）。"""
    fake = _FakeNusc()
    _write_lidar_bin(tmp_path, fake)
    fake.dataroot = str(tmp_path)  # generate_review_queue 取 nusc.dataroot（非参数）
    sample_data = {
        "LIDAR_TOP": "sd-lidar", **{n: f"sd-{n}" for n in NUSCENES_CAMERAS},
    }
    samples = [
        {"token": "s0", "data": sample_data},
        {"token": "s1", "data": sample_data},
    ]
    det = _FakeLidarDet(boxes=np.zeros((0, 9)))  # 零预测 → 空队列文件
    monkeypatch.setattr(nsp, "load_nuscenes", lambda root=None, version="v1.0-mini": fake)
    monkeypatch.setattr(nsp, "val_scene_names", lambda version="v1.0-mini": ["scene-A"])
    monkeypatch.setattr(nsp, "samples_of_scene", lambda nusc, scene: samples)

    out = tmp_path / "reviews"
    written = generate_review_queue(det, out, conf=0.3)
    assert set(written) == {"s0", "s1"}
    assert (out / "s0_review.json").is_file() and (out / "s1_review.json").is_file()
    assert generate_review_queue(det, out, conf=0.3) == {}  # 二跑全跳过
