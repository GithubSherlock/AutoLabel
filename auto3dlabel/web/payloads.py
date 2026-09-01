"""Web 3D 复核 frame-data payload 纯函数（v0.3 P4：四视图真 3D 复核）。

契约：点云经 velo_to_cam 转**相机系**（x 右 y 下 z 前）交付，逐框角点
corners_cam 8x3 同系——yaw/尺寸/标定数学全部留在 Python 侧，前端零 calib
依赖，仅换轴渲染 (x, z, -y)（three 装配见 static/viewer3d.js）。
下采样 seed 确定性：同帧同 seed → 同一批点（审核一致性，不随请求变化）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from auto3dlabel.data.nuscenes import load_lidar_file
from auto3dlabel.schema.box3d import Box3D, load_points_bin
from auto3dlabel.schema.calib import KittiCalib
from auto3dlabel.tools.geometry import global_to_cam_like

# 单帧点云交付上限（KITTI 帧 ~12 万点；渲染帧率与 JSON 体积折中）
MAX_POINTS = 100_000
_DOWNSAMPLE_SEED = 0


def downsample_points(
    pts: np.ndarray, max_points: int = MAX_POINTS, seed: int = _DOWNSAMPLE_SEED
) -> np.ndarray:
    """(N,C) 点随机均匀下采样到 max_points（seed 确定性；不超限原样返回同一对象）。"""
    if len(pts) <= max_points:
        return pts
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(pts), size=max_points, replace=False))
    return pts[idx]


def validate_queue_name(name: str) -> bool:
    """队列文件名白名单（从 server.py 抽公共：后缀白名单 + 路径遍历拒绝，行为零变）。"""
    return (
        name.endswith(("_review.json", "_reviewed.json"))
        and "/" not in name
        and "\\" not in name
    )


def _objects_from_annotations(annotations: list[Any]) -> list[dict[str, Any]]:
    """annotations（Box3D dict 列表）→ objects（坏框跳过；KITTI/nus 共用）。

    index = 原 annotations 下标（前端表格/保存回写对齐，坏框跳过不错位）。
    """
    objects: list[dict[str, Any]] = []
    for i, b in enumerate(annotations):
        if not isinstance(b, dict):
            continue
        try:
            box = Box3D.from_dict(b)
        except (TypeError, ValueError):
            continue  # 坏框跳过（宁缺勿假）
        objects.append({
            "index": i,
            "label": box.label,
            "confidence": box.confidence,
            "fit_points": box.fit_points,
            "corners": box.corners_cam().tolist(),
        })
    return objects


def _nuscenes_payload(data: dict[str, Any]) -> dict[str, Any] | None:
    """nuScenes 队列（dataset=="nuscenes"）→ 渲染 payload：cam_like 点云 + 6 相机路径。

    points = global_to_cam_like(pcd xyz, ego_translation)（队列 JSON 自足，零 devkit）；
    cameras 的 image_path 为 dataroot 绝对路径（review-image 端点读）。
    """
    pcd_path = str(data.get("pcd_path") or "")
    try:
        pts_glob = load_lidar_file(Path(pcd_path))
    except (OSError, ValueError):
        return None
    ego_raw = data.get("ego_translation", (0.0, 0.0, 0.0))
    ego = (float(ego_raw[0]), float(ego_raw[1]), float(ego_raw[2]))
    pts_cam = global_to_cam_like(pts_glob[:, :3], ego)
    dataroot = str(data.get("dataroot") or "")
    cameras = [
        {
            "name": c.get("name", ""),
            "image_path": str(Path(dataroot) / c["filename"]),
        }
        for c in data.get("cameras", [])
        if isinstance(c, dict) and c.get("filename")
    ]
    return {
        "image": data.get("image", ""),
        "image_path": "",
        "bev_path": "",
        "points": downsample_points(pts_cam).tolist(),
        "objects": _objects_from_annotations(data.get("annotations", [])),
        "cameras": cameras,
        "dataset": "nuscenes",
    }


def frame_payload(name: str, review_dir: Path) -> dict[str, Any] | None:
    """队列文件名 → 四视图渲染 payload；损坏/缺 pcd/缺 calib → None（端点转 400）。

    KITTI 输出键：image/image_path/bev_path（透传）、points（相机系 (N,3) list）、
    objects（label/confidence/fit_points/corners 8x3 list；坏框跳过）。
    nuScenes（dataset=="nuscenes"）：cam_like 点云 + cameras 6 相机（见 _nuscenes_payload）。
    """
    src = review_dir / name
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("dataset") == "nuscenes":
        return _nuscenes_payload(data)
    pcd_path = str(data.get("pcd_path") or "")
    calib_path = str(data.get("calib_path") or "")
    try:
        pts_velo = load_points_bin(Path(pcd_path))
        pts_cam = KittiCalib.from_file(Path(calib_path)).velo_to_cam(pts_velo)
    except (OSError, ValueError):
        return None
    return {
        "image": data.get("image", ""),
        "image_path": data.get("image_path", ""),
        "bev_path": data.get("bev_path", ""),
        "points": downsample_points(pts_cam).tolist(),
        "objects": _objects_from_annotations(data.get("annotations", [])),
    }
