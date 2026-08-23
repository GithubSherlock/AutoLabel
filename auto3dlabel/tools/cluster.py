"""DBSCAN 语义点聚类（BEV xz 平面）+ 粘连嫌疑标记。

背景点剔除关键：2D mask 像素对应一条光线上的多个深度点（前景 + 身后背景），
DBSCAN 在 BEV 把前景主簇与背景簇分开——主簇保留，其余丢弃。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN

from auto3dlabel.configs.kitti import (
    ADAPTIVE_EPS,
    CLUSTER_MAX_POINTS,
    DBSCAN_MIN_SAMPLES,
    VOXEL_MAX_POINTS,
    VOXEL_SIZE,
)


@dataclass
class ClusterResult:
    """主簇点（原始分辨率）+ 粘连嫌疑标记（多实例遮挡合并的 HITL 兜底）。"""

    points_cam: np.ndarray  # (M,3) 主簇相机系点
    adhesion_flag: bool = False  # 次大簇与主簇体量接近 / 主簇点数超上限


def _voxel_downsample(pts_bev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N,2) BEV 点 → (体素代表点, inv 映射到原始点下标)。"""
    vox = np.floor(pts_bev / VOXEL_SIZE).astype(np.int64)
    keys = vox[:, 0] * (2**32) + vox[:, 1]
    uniq_keys, inv = np.unique(keys, return_inverse=True)
    reps = np.array(
        [pts_bev[inv == i].mean(axis=0) for i in range(len(uniq_keys))], dtype=np.float64
    )
    return reps, inv


def cluster_instance(points_cam: np.ndarray, label: str) -> ClusterResult | None:
    """实例语义点 → 主簇（DBSCAN BEV xz）。

    返回 None：点数不足 DBSCAN_MIN_SAMPLES（调用方计 dropped/丢弃）；
    大簇（> VOXEL_MAX_POINTS）先 0.1m 体素降采样，标签经 inv 映射回原始点。
    """
    if len(points_cam) < DBSCAN_MIN_SAMPLES:
        return None
    pts_bev = points_cam[:, [0, 2]]
    if len(pts_bev) > VOXEL_MAX_POINTS:
        reps, inv = _voxel_downsample(pts_bev)
        cluster_pts, back_map = reps, inv
    else:
        cluster_pts, back_map = pts_bev, None

    eps = ADAPTIVE_EPS.get(label, 1.0)
    labels = DBSCAN(eps=eps, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(cluster_pts)
    valid = labels >= 0
    if not valid.any():
        return None
    sizes = np.bincount(labels[valid])
    main = int(np.argmax(sizes))
    main_mask = labels == main
    # 粘连嫌疑：次大簇 ≥ 主簇 30%（两实例同 mask 且空间分开）或主簇超点数上限
    adhesion = False
    if len(sizes) > 1:
        second = int(np.sort(sizes)[-2])
        adhesion = bool(second >= 0.3 * sizes[main])
    if int(sizes[main]) > CLUSTER_MAX_POINTS:
        adhesion = True

    if back_map is not None:
        main_mask = main_mask[back_map]
    return ClusterResult(points_cam=points_cam[main_mask], adhesion_flag=adhesion)
