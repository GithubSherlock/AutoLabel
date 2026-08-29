"""test_detection3d：LiDAR 3D 检测器封装（零真实权重铁律 + 守卫路径双态兼容）。

mmdet3d 未装状态下全绿是设计目标（ImportError 守卫在 _load）；已装环境
守卫测试自动跳过（skipif）——双态质量门均归零。
"""

from __future__ import annotations

import importlib.util
from typing import Any

import pytest

from auto3dlabel.configs.kitti import (
    DETECTOR3D_NAMES,
    MMDET3D_CONFIG_DIR,
    WEIGHTS_DIR,
    prompts_to_kitti_labels,
)
from auto3dlabel.models.detection3d import (
    Det3DResult,
    Mmdet3dDetector,
    _init_model_trusted,
    create_detector3d,
)

MMDET3D_INSTALLED = importlib.util.find_spec("mmdet3d") is not None


def test_det3dresult_to_box3d() -> None:
    """mmdet3d 底面中心 7 值 → Box3D 体积中心语义（cy = y - h/2、yaw 唯一转换点、字段透传）。"""
    r = Det3DResult(
        label="Car",
        confidence=0.85,
        bbox=[8.0, -0.9, 18.0, 3.9, 1.5, 1.6, 0.0],
        x1=10.0,
        y1=20.0,
        x2=30.0,
        y2=40.0,
    )
    box = r.to_box3d()
    assert box.label == "Car"
    assert box.confidence == 0.85
    assert box.cx == 8.0 and box.cz == 18.0
    assert box.cy == pytest.approx(-0.9 - 0.75)  # 底面 y → 体积中心
    assert (box.h, box.w, box.l) == (1.5, 1.6, 3.9)
    assert box.rotation_y == pytest.approx(0.0)  # ry=0 → yaw_bev=π/2 → ry 往返 0
    assert (box.x1, box.y1, box.x2, box.y2) == (10.0, 20.0, 30.0, 40.0)


def test_create_detector3d_routing() -> None:
    """工厂路由：None/2D 名/未知名 → None；3D 名 → Mmdet3dDetector（构造零加载）。"""
    assert create_detector3d(None) is None
    assert create_detector3d("kitti_finetune") is None  # 2D 名不误路由
    assert create_detector3d("unknown3d") is None
    det = create_detector3d("pointpillars_kitti")
    assert isinstance(det, Mmdet3dDetector)
    entry = DETECTOR3D_NAMES["pointpillars_kitti"]
    assert det._checkpoint_path == str(WEIGHTS_DIR / entry["weights_dir"] / entry["checkpoint"])
    assert det._config_path == str(MMDET3D_CONFIG_DIR / entry["config"])
    # 零加载：config/权重文件不存在也不抛（存在性延迟到 _load，缺权重时抛给调用方兜底）
    assert isinstance(create_detector3d("centerpoint_nus"), Mmdet3dDetector)
    # v0.3 P2：pvrcnn_kitti 路由（照 centerpoint_nus 既有断言模式）
    det_pv = create_detector3d("pvrcnn_kitti")
    assert isinstance(det_pv, Mmdet3dDetector)
    entry_pv = DETECTOR3D_NAMES["pvrcnn_kitti"]
    assert det_pv._checkpoint_path == str(
        WEIGHTS_DIR / entry_pv["weights_dir"] / entry_pv["checkpoint"]
    )
    assert det_pv._config_path == str(MMDET3D_CONFIG_DIR / entry_pv["config"])


@pytest.mark.skipif(MMDET3D_INSTALLED, reason="mmdet3d 已装，守卫路径跳过")
def test_detector3d_load_guard_import() -> None:
    """mmdet3d 未装 → _load 抛 ImportError（守卫路径；已装环境自动跳过）。"""
    det = Mmdet3dDetector("missing_config.py", "missing.pth")
    with pytest.raises(ImportError):
        det._load()


@pytest.mark.skipif(not MMDET3D_INSTALLED, reason="需 mmdet3d 注入 fake init_model")
def test_init_model_trusted_scoped_patch(monkeypatch: Any) -> None:
    """v0.3 P2 回归：torch 2.6+ weights_only 兼容 patch 的作用域契约。

    patch 仅覆盖 init_model 调用期（调用期内 torch.load 被替换、
    weights_only 默认 False），返回/异常两条路径都恢复原函数——
    进程内其他 checkpoint 加载路径零影响。零真实权重（fake init_model）。
    """
    import torch

    orig_load = torch.load
    calls: list[str] = []

    def _fake_init_model(
        config: Any = None, checkpoint: Any = None, device: str = "cuda:0"
    ) -> str:
        calls.append(device)
        assert torch.load is not orig_load  # 调用期内已 patch
        return "fake-model"

    monkeypatch.setattr("mmdet3d.apis.init_model", _fake_init_model)
    assert _init_model_trusted("cfg.py", "ckpt.pth", "cuda:0") == "fake-model"
    assert calls == ["cuda:0"]
    assert torch.load is orig_load  # 正常路径恢复

    def _boom(config: Any = None, checkpoint: Any = None, device: str = "cuda:0") -> str:
        raise RuntimeError("boom")

    monkeypatch.setattr("mmdet3d.apis.init_model", _boom)
    with pytest.raises(RuntimeError):
        _init_model_trusted("cfg.py", "ckpt.pth", "cpu")
    assert torch.load is orig_load  # 异常路径也恢复


def test_prompts_to_kitti_labels() -> None:
    """prompts（COCO 名）→ LiDAR 引擎保留的 KITTI 类集合；无交集 → None（保留全部）。"""
    assert prompts_to_kitti_labels(["car"]) == {"Car"}
    assert prompts_to_kitti_labels(["person", "bicycle"]) == {"Pedestrian", "Cyclist"}
    assert prompts_to_kitti_labels(["truck"]) is None  # 3-class 模型无 Truck
    assert prompts_to_kitti_labels(["dog", "cat"]) is None
    assert prompts_to_kitti_labels([]) is None
