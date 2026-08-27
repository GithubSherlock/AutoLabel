"""test_calib：KITTI 标定解析 + 投影链数值验证（真实 calib 数值硬编码，离线可跑）。"""

from __future__ import annotations

import numpy as np
import pytest

from auto3dlabel.schema.calib import KittiCalib
from auto3dlabel.tests.helpers import synth


def _calib() -> KittiCalib:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "calib.txt"
        synth.write_calib(p)
        return KittiCalib.from_file(p)


def test_parse_shapes() -> None:
    c = _calib()
    assert c.P2.shape == (3, 4)
    assert c.R0_rect.shape == (3, 3)
    assert c.Tr_velo_to_cam.shape == (3, 4)
    assert c.Tr_imu_to_velo.shape == (3, 4)


def test_parse_p2_focal() -> None:
    c = _calib()
    assert abs(c.P2[0, 0] - 707.0493) < 1e-3
    assert abs(c.P2[1, 1] - 707.0493) < 1e-3
    assert abs(c.P2[0, 2] - 604.0814) < 1e-3  # cx
    assert abs(c.P2[1, 2] - 180.5066) < 1e-3  # cy


def test_missing_required_lines_raises() -> None:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "bad.txt"
        p.write_text("P0: 1 0 0 0 0 1 0 0 0 0 1 0\nR0_rect: 1 0 0 0 1 0 0 0 1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="缺少必需行"):
            KittiCalib.from_file(p)


def test_velo_to_cam_anchor() -> None:
    """真实数值锚点：velodyne (8.752,-1.800,-1.546) → 相机系（已双重验证）。"""
    c = _calib()
    cam = c.velo_to_cam(np.array([synth.VELO_ANCHOR]))
    assert np.allclose(cam[0], synth.ANCHOR_CAM, atol=1e-6)


def test_velo_to_cam_ignores_intensity() -> None:
    """(N,4) 输入自动取前 3 列。"""
    c = _calib()
    pts4 = np.array([[8.752, -1.800, -1.546, 0.42]])
    cam = c.velo_to_cam(pts4)
    assert np.allclose(cam[0], synth.ANCHOR_CAM, atol=1e-6)


def test_project_anchor_pixel() -> None:
    """投影链核心锚点：velodyne → (758, 299)。"""
    c = _calib()
    u, v, valid = c.project_velo_to_image(np.array([synth.VELO_ANCHOR]), synth.IMG_W, synth.IMG_H)
    assert valid[0] and u[0] == synth.ANCHOR_PIXEL[0] and v[0] == synth.ANCHOR_PIXEL[1]


def test_project_cam_to_image_matches_velo() -> None:
    """两条路径（velo→pixel 与 velo→cam→pixel）一致。"""
    c = _calib()
    u1, v1, _ = c.project_velo_to_image(np.array([synth.VELO_ANCHOR]), synth.IMG_W, synth.IMG_H)
    cam = c.velo_to_cam(np.array([synth.VELO_ANCHOR]))
    u2, v2, _ = c.project_cam_to_image(cam, synth.IMG_W, synth.IMG_H)
    assert (u1 == u2).all() and (v1 == v2).all()


def test_behind_camera_invalid() -> None:
    """z ≤ 0 的点（相机后方）invalid。"""
    c = _calib()
    behind = np.array([[0.0, 0.0, -1.0]])
    u, v, valid = c.project_cam_to_image(behind, synth.IMG_W, synth.IMG_H)
    assert not valid[0]
    assert u[0] == 0 and v[0] == 0  # 哨兵


def test_out_of_image_invalid() -> None:
    """投影到图外的点 invalid + u/v 置 0。"""
    c = _calib()
    far_side = np.array([[100.0, 0.0, 1.0]])  # 大 x 小 z → 超出宽度
    _, _, valid = c.project_cam_to_image(far_side, synth.IMG_W, synth.IMG_H)
    assert not valid[0]


def test_vectorized_batch() -> None:
    c = _calib()
    pts = np.array([synth.VELO_ANCHOR, synth.VELO_ANCHOR, [0.0, 0.0, -5.0]])
    u, v, valid = c.project_velo_to_image(pts, synth.IMG_W, synth.IMG_H)
    assert valid.sum() == 2
    assert (u[:2] == synth.ANCHOR_PIXEL[0]).all()
