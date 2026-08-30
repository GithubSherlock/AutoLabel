"""test_mono3d：单目 3D 检测器封装（v0.3 P3 / P6b；零真实权重铁律 + 守卫路径双态兼容）。

mmdet3d 未装状态下全绿是设计目标（ImportError 守卫在 _load）；已装环境
守卫测试自动跳过（skipif）——双态质量门均归零。
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from auto3dlabel.configs.kitti import MMDET3D_CONFIG_DIR, WEIGHTS_DIR
from auto3dlabel.configs.model_catalog import FCOS3D_NAMES, MONO3D_NAMES
from auto3dlabel.models.detection3d import MMDET3D_KITTI_CLASSES
from auto3dlabel.models.mono3d import (
    Fcos3dNuScenesDetector,
    Mono3dDetector,
    _fcos3d_reorder_dims_yaw,
    _mono_output_to_det3d,
    build_mono_data,
    create_fcos3d_detector,
    create_mono3d_detector,
)

MMDET3D_INSTALLED = importlib.util.find_spec("mmdet3d") is not None


def test_build_mono_data_contract() -> None:
    """data_ dict 契约：images.CAM2 {img_path, cam2img 3x4 float32} + box 语义透传。

    box_type_3d 必须是**类**（pgd head 以 img_meta['box_type_3d'](...) 构造框，
    字符串会 TypeError）——纯函数零 mmdet3d 依赖，默认 None，传什么透传什么。
    """
    cam2img = np.arange(12, dtype=np.float64).reshape(3, 4)
    data = build_mono_data("/tmp/x.png", cam2img)
    assert data["box_type_3d"] is None  # 由调用方 _load 提供 get_box_type 类
    assert data["box_mode_3d"] is None
    img = data["images"]["CAM2"]
    assert img["img_path"] == "/tmp/x.png"
    assert img["cam2img"].shape == (3, 4)
    assert img["cam2img"].dtype == np.float32
    np.testing.assert_allclose(img["cam2img"], cam2img)

    # 类哨兵透传（模拟 get_box_type 返回的 CameraInstance3DBoxes）
    class FakeBoxCls:  # noqa: D401
        pass

    sentinel = FakeBoxCls
    data2 = build_mono_data("/tmp/x.png", cam2img, box_type_3d=sentinel, box_mode_3d=1)
    assert data2["box_type_3d"] is sentinel
    assert data2["box_mode_3d"] == 1


def test_mono_output_to_det3d_lw_order() -> None:
    """解包 L-W 顺序锁死：7 值 [x,y,z,l,h,w,ry] 原样透传（相机系底面中心）。"""
    boxes = np.asarray([[8.0, -0.9, 18.0, 3.9, 1.5, 1.6, 0.1]], dtype=np.float32)
    scores = np.asarray([0.85], dtype=np.float32)
    labels = np.asarray([2], dtype=np.int64)
    dets = _mono_output_to_det3d(boxes, scores, labels, MMDET3D_KITTI_CLASSES)
    assert len(dets) == 1
    assert dets[0].label == "Car"  # 默认 KITTI 类序 idx2
    assert dets[0].confidence == pytest.approx(0.85)
    assert dets[0].bbox == pytest.approx([8.0, -0.9, 18.0, 3.9, 1.5, 1.6, 0.1])  # l/h/w 顺序

    # to_box3d 复用断言：cy = y - h/2（底面 → 体积中心）、尺寸字段对准
    box = dets[0].to_box3d()
    assert box.cy == pytest.approx(-0.9 - 0.75)
    assert (box.h, box.w, box.l) == pytest.approx((1.5, 1.6, 3.9))
    assert box.cx == 8.0 and box.cz == 18.0


def test_create_mono3d_detector_routing() -> None:
    """工厂路由：None/未知名 → None；pgd_kitti → Mono3dDetector（构造零加载）。"""
    assert create_mono3d_detector(None) is None
    assert create_mono3d_detector("pointpillars_kitti") is None  # LiDAR 名不误路由
    assert create_mono3d_detector("unknown3d") is None
    det = create_mono3d_detector("pgd_kitti")
    assert isinstance(det, Mono3dDetector)
    entry = MONO3D_NAMES["pgd_kitti"]
    assert det._config_path == str(MMDET3D_CONFIG_DIR / entry["config"])
    assert det._checkpoint_path == str(WEIGHTS_DIR / entry["weights_dir"] / entry["checkpoint"])


@pytest.mark.skipif(MMDET3D_INSTALLED, reason="mmdet3d 已装，守卫路径跳过")
def test_mono3d_load_guard_import() -> None:
    """mmdet3d 未装 → _load 抛 ImportError（守卫路径；已装环境自动跳过）。"""
    det = Mono3dDetector("missing_config.py", "missing.pth")
    with pytest.raises(ImportError):
        det._load()


def test_build_mono_data_camera_key() -> None:
    """P6b：camera 参数泛化——nuScenes 用 CAM_FRONT 单键（LoadImageFromFileMono3D
    单键 images 分支）；默认 CAM2 向后兼容（KITTI pgd）。"""
    cam2img = np.arange(9, dtype=np.float64).reshape(3, 3)
    data = build_mono_data("/tmp/n.jpg", cam2img, camera="CAM_FRONT")
    assert set(data["images"]) == {"CAM_FRONT"}
    assert data["images"]["CAM_FRONT"]["img_path"] == "/tmp/n.jpg"
    np.testing.assert_allclose(data["images"]["CAM_FRONT"]["cam2img"], cam2img)
    # 默认键回归
    assert set(build_mono_data("/tmp/k.png", cam2img)["images"]) == {"CAM2"}


def test_fcos3d_reorder_dims_yaw() -> None:
    """P6b 实测定序锁定：head 输出 [d0,d1,d2,yaw,vx,vz]=(l,h,w)+负 yaw
    → 归一 [w,l,h,+yaw,vx,vz]（w=d2、l=d0、h=d1、yaw 取负、速度直传）。

    实测锚点（Mini 单样本探针）：pred car dims (4.44,1.51,1.84) 按 (l,h,w)
    解读匹配 GT wlh (2.00,4.73,1.48)；yaw=-GT 相机系 yaw。4 组合评测中
    仅「dims 重排 + yaw 负号」mAP 非零（0.8 / car 8.0），其余全 0。
    """
    tail = np.asarray([[4.44, 1.51, 1.84, 5.051, 0.05, 4.49]])
    out = _fcos3d_reorder_dims_yaw(tail)
    assert out.shape == tail.shape
    np.testing.assert_allclose(out[0, :3], [1.84, 4.44, 1.51])  # (w,l,h)
    assert out[0, 3] == pytest.approx(-5.051)  # yaw 负号
    np.testing.assert_allclose(out[0, 4:], [0.05, 4.49])  # 速度直传
    # 输入不被就地修改（copy 语义）
    assert tail[0, 3] == 5.051 and tail[0, 0] == 4.44
    # 空批量零异常
    assert _fcos3d_reorder_dims_yaw(np.zeros((0, 6))).shape == (0, 6)


def test_create_fcos3d_detector_routing() -> None:
    """P6b 工厂路由：None/未知名/LiDAR 名 → None；fcos3d_nus → Fcos3dNuScenesDetector。"""
    assert create_fcos3d_detector(None) is None
    assert create_fcos3d_detector("unknown3d") is None
    assert create_fcos3d_detector("pgd_kitti") is None  # KITTI 单目名不误路由
    assert create_fcos3d_detector("pointpillars_nus") is None  # LiDAR 名不误路由
    det = create_fcos3d_detector("fcos3d_nus")
    assert isinstance(det, Fcos3dNuScenesDetector)
    entry = FCOS3D_NAMES["fcos3d_nus"]
    assert det._config_path == str(MMDET3D_CONFIG_DIR / entry["config"])
    assert det._checkpoint_path == str(
        WEIGHTS_DIR / entry["weights_dir"] / entry["checkpoint"]
    )
    assert det._class_names == ()  # 构造零加载：类序延迟到 _load（mmdet3d 序 10 类）


@pytest.mark.skipif(MMDET3D_INSTALLED, reason="mmdet3d 已装，守卫路径跳过")
def test_fcos3d_load_guard_import() -> None:
    """P6b：mmdet3d 未装 → _load 抛 ImportError（共享 _load_mono_engine 守卫）。"""
    det = Fcos3dNuScenesDetector("missing_config.py", "missing.pth")
    with pytest.raises(ImportError):
        det._load()
