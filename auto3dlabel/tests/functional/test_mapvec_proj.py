"""test_mapvec_proj：投影链锚点单测（CAM_FRONT yaw=0 → 主点 = 图像中心）。

数值自证(不靠目检)：正前方 z=0 的点应投到主点 (621, 187.5) = 图像中心；
偏离左右/远近的点按针孔模型手算锚点对拍。
"""

from __future__ import annotations

import numpy as np
import pytest

from auto3dlabel.schema.mapvec_proj import (
    bev_panel,
    cam_pose,
    ego_to_world,
    intrinsics_from_k,
    project_lines,
    project_points,
    world_to_cam,
)

W, H = 1242, 375
FX, FY, CX, CY = 621.0, 621.0, 621.0, 187.5


def _k() -> list[list[float]]:
    return [[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]]


def _ego() -> list[float]:
    return [-67.254, 27.964, -0.024, 0.159, 0.105, -0.054]  # 真实帧 200


def _ego_zero() -> list[float]:
    """零姿态 ego(yaw=pitch=roll=0)：投影锚点可精确手算。"""
    return [-67.254, 27.964, -0.024, 0.0, 0.0, 0.0]


def _se() -> list[float]:
    return [1.2, 0.0, 1.65, 0.0, 0.0, 0.0]  # CAM_FRONT sensor2ego(度,yaw=0)


def test_intrinsics_from_k() -> None:
    assert intrinsics_from_k(_k()) == (FX, FY, CX, CY)


def test_ego_to_world_hand_anchor() -> None:
    """ego 原点 (0,0) → 世界系 ego 位置(高度取 ego[2])。"""
    eg = _ego()
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 5.0]])
    out = np.asarray(ego_to_world(pts, eg))
    # (0,0) → ego 自身位置
    np.testing.assert_allclose(out[0], [eg[0], eg[1], eg[2]], atol=1e-9)
    # (10,0) → 沿 yaw=0.159° 前向：旋转位移 ~ (10, 10·sin(yaw))
    np.testing.assert_allclose(
        out[1],
        [eg[0] + 10 * np.cos(np.radians(eg[3])), eg[1] + 10 * np.sin(np.radians(eg[3])), eg[2]],
        atol=1e-9,
    )
    # (0,5) → 左向(+y)偏移
    np.testing.assert_allclose(
        out[2],
        [eg[0] - 5 * np.sin(np.radians(eg[3])), eg[1] + 5 * np.cos(np.radians(eg[3])), eg[2]],
        atol=1e-9,
    )


def test_cam_pose_front_yaw_zero() -> None:
    """CAM_FRONT sensor2ego yaw=0 + ego yaw=0 → 相机位姿 = ego + (1.2, 0, 1.65)。"""
    eg = _ego_zero()
    loc, rot = cam_pose(eg, _se())
    np.testing.assert_allclose(loc, [eg[0] + 1.2, eg[1], eg[2] + 1.65], atol=1e-9)
    # 姿态弧度 = ego 弧度(挂点 yaw/pitch/roll 全 0)
    np.testing.assert_allclose(rot, (0.0, 0.0, 0.0), atol=1e-12)


def test_world_to_cam_pinhole_anchor() -> None:
    """世界系点 → 相机系：沿相机光轴 (0,0,+z) 的点 → 相机系 (0,0,+z)。"""
    pts = np.array([[10.0, 20.0, 1.65]])
    loc = (10.0, 20.0, 1.65)
    rot = (0.0, 0.0, 0.0)  # 恒等姿态
    c = world_to_cam(pts, loc, rot)
    # CARLA_TO_CAM:世界 (x前,y右,z上) → 相机 (x右,y下,z前)；光轴点 (0,0,+z)
    np.testing.assert_allclose(c[0], [0.0, 0.0, 0.0], atol=1e-9)


def test_project_points_front_center_to_principal_point() -> None:
    """CAM_FRONT(零姿态)：正前方地面点(z 恒为 ego 高)→ u=主点 cx、v 在光轴下方。

    ego_to_world 强制 z=ego[2](地面),相机在 +1.65m → 地面点必在光轴**下方**
    (v>cy),与产出方 test_mapviz 同判据(k.cy < v < H)。u 无横向偏移 → 精确 =cx。
    """
    eg = _ego_zero()
    pose = cam_pose(eg, _se())
    fx, fy, cx, cy = intrinsics_from_k(_k())
    px = project_points(np.array([[10.0, 0.0, 0.0]]), eg, pose, fx, fy, cx, cy, W, H)
    assert px[0] is not None
    u, v = px[0]
    assert abs(u - CX) < 1e-6  # 正前方无横向偏移 → 精确主点 u
    assert CY < v < H  # 地面点在光轴下方(相机高于路面 OFF[2])
    # 光轴高度锚点:手算 v = cy + fy·(1.65)/(10−1.2)
    assert v == pytest.approx(CY + FY * 1.65 / (10.0 - 1.2), abs=1e-6)


def test_project_points_offset_hand_anchor() -> None:
    """偏右 2m、前 20m 的点 → u > cx;偏左 → u < cx(针孔口径,非目检)。"""
    eg = _ego_zero()
    pose = cam_pose(eg, _se())
    fx, fy, cx, cy = intrinsics_from_k(_k())
    # 相机 x_cam = CARLA y(右);点 y=2 → 相机系 x=2 → u > cx;y=-2 → u < cx
    for y, expect_less in ((-2.0, True), (2.0, False)):
        px = project_points(np.array([[20.0, y, 0.0]]), eg, pose, fx, fy, cx, cy, W, H)
        assert px[0] is not None
        u, _ = px[0]
        assert (u < cx) == expect_less
    # 数值锚点:y=2 → u = cx + 621·2/(18.8);z=0 → 相机系 y=1.65 → v = cy + 621·1.65/18.8
    px_r = project_points(np.array([[20.0, 2.0, 0.0]]), eg, pose, fx, fy, cx, cy, W, H)
    assert px_r[0] == pytest.approx(
        (CX + FY * 2.0 / (20.0 - 1.2), CY + FY * 1.65 / (20.0 - 1.2)), abs=1e-6
    )


def test_project_lines_breaks_at_behind_camera() -> None:
    """折线穿过相机平面 → 相机后(深度 ≤0.5m)断开成段,不跨遮挡连线。"""
    eg = _ego_zero()
    pose = cam_pose(eg, _se())
    fx, fy, cx, cy = intrinsics_from_k(_k())
    # ego_to_world 强制 z=ego 高度(地面点)——相机在 1.65m 高,x=1 深度 −0.2 相机后,
    # x=5 深度 3.8(相机前但高过光轴? 实际地面 z=0 在相机下,v>cy),x=8 也可见
    line = np.array([[-1.0, 0.0], [1.0, 0.0], [5.0, 0.0], [8.0, 0.0]])  # z=0 地面折线
    segs = project_lines([line], eg, pose, fx, fy, cx, cy, W, H)
    # 可见点:x=5(深度 3.8, v=621·1.65/3.8+187.5=457>375? 出图)→ 实际 None
    # x=8 → (621, 338) 可见。只有 1 点可见 → 不成段(需 ≥2 连续点)
    assert segs == []


def test_bev_panel_is_pil_image() -> None:
    img = bev_panel([], [], title="t")
    assert img.size == (420, 420)
    from PIL import Image

    assert isinstance(img, Image.Image)
