"""test_mapvec_crosscheck：复制版 vs AutoDriveData 原版交叉验证（同一输入 → 同一输出）。

红线「AutoLabel 不 import AutoDriveData」指运行期依赖,不是禁止测试对照。此测试
仅在本机有 AutoDriveData 时跑(importorskip 跳过),锁定复制代码不漂移:
- chamfer_ap.py 原版 → chamfer_distance/chamfer_cost_matrix/match_greedy 逐元素一致
- mapviz.py + geometry.py 原版 → ego_to_world/cam_pose/world_to_cam 数值一致
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

AUTODRIVE_DATA = Path("/root/autodl-tmp/Documents/Projects/AutoDriveData")
pytestmark = pytest.mark.skipif(
    not AUTODRIVE_DATA.is_dir(), reason="本机无 AutoDriveData 兄弟仓库,交叉验证跳过"
)
sys.path.insert(0, str(AUTODRIVE_DATA))

import autodrivedata.chamfer_ap as orig_ca  # noqa: E402
import autodrivedata.geometry as orig_geo  # noqa: E402
import autodrivedata.mapviz as orig_mv  # noqa: E402

from auto3dlabel.schema import mapvec_proj as my_proj  # noqa: E402
from auto3dlabel.tools import mapvec_compare as my_cmp  # noqa: E402

rng = np.random.default_rng(0)


def _lines(n: int, max_len: int = 20) -> list[np.ndarray]:
    out = []
    for _ in range(n):
        m = int(rng.integers(2, max_len + 1))
        out.append(rng.uniform(-15, 15, (m, 2)))
    return out


def test_chamfer_fns_match_original() -> None:
    for _ in range(5):
        preds, gts = _lines(6), _lines(5)
        c_my = my_cmp.chamfer_cost_matrix(preds, gts)
        c_orig = orig_ca.chamfer_cost_matrix(preds, gts)
        np.testing.assert_allclose(c_my, c_orig, atol=1e-6)
        for p in preds:
            for g in gts:
                assert abs(my_cmp.chamfer_distance(p, g) - orig_ca.chamfer_distance(p, g)) < 1e-6
        for thr in (0.5, 1.0, 1.5):
            assert my_cmp.match_greedy(preds, gts, thr, c_my) == orig_ca.match_greedy(
                preds, gts, thr, c_orig
            )


def test_ego_to_world_matches_original() -> None:
    ego = [-67.254, 27.964, -0.024, 0.159, 0.105, -0.054]  # 真实帧 200 ego2global
    pts = np.column_stack([np.linspace(-15, 15, 20), np.linspace(-30, 30, 20)])
    mine = np.asarray(my_proj.ego_to_world(pts, ego))
    orig = np.asarray(orig_mv.ego_to_world(pts, ego))
    np.testing.assert_allclose(mine, orig, atol=1e-9)


def test_cam_pose_matches_original() -> None:
    ego = [-67.254, 27.964, -0.024, 0.159, 0.105, -0.054]
    se = [1.2, 0.0, 1.65, 0.0, 0.0, 0.0]  # CAM_FRONT sensor2ego
    mine = my_proj.cam_pose(ego, se)
    orig = orig_mv.cam_pose(ego, se)
    for a, b in zip(mine, orig):
        np.testing.assert_allclose(a, b, atol=1e-12)


def test_world_to_cam_matches_original() -> None:
    """复制版(几何链 + CARLA_TO_CAM)vs 原版 autodrivedata.world_to_cam。"""
    pts = rng.uniform(-100, 100, (50, 3))
    loc = (10.0, 20.0, 1.65)
    rot = (math.radians(0.1), math.radians(0.16), math.radians(-0.05))
    mine = my_proj.world_to_cam(pts, loc, rot)
    orig = orig_geo.world_to_cam(pts, loc, rot)
    np.testing.assert_allclose(mine, orig, atol=1e-9)


def test_carla_rotation_matrix_matches_original() -> None:
    rot = (math.radians(1.1), math.radians(0.16), math.radians(-0.5))
    np.testing.assert_allclose(
        my_proj.carla_rotation_matrix(rot), orig_geo.carla_rotation_matrix(rot), atol=1e-12
    )
