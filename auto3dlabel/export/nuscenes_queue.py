"""nuScenes 复核队列（v0.4 P2）：NusBox 全局系 → cam_like 渲染 dict → triage_3d 复用。

队列文件 = KITTI 协议扩展（Web 四端点零协议改动）：dataset:"nuscenes" +
scene_name/sample_token/version/dataroot/ego_translation/cameras（6 相机）+
annotations 为 cam_like 渲染 dict（Box3D 键 + velocity/track_id 透传顶层键）。

坐标系（红线）：annotations/points 均为 ego 局部「相机式」帧（x 右/y 下/z 前，
tools/geometry.global_to_cam_like 唯一转换点）——前端（viewer3d/logic3d/P1 手柄）
零改动；保存时 data/nuscenes.box3d_dict_to_nusbox 转回全局系。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from auto3dlabel.configs.nuscenes import NUSCENES_MIN_FIT_POINTS
from auto3dlabel.data.nuscenes import (
    cameras_of_sample,
    lidar_sample_data,
    nusbox_to_box3d_dict,
)
from auto3dlabel.export.review_queue import Triage3D, triage_3d
from auto3dlabel.schema.box3d import Box3D
from auto3dlabel.schema.nuscenes_box import NusBox
from auto3dlabel.tools.geometry import global_to_cam_like, points_in_box


def _render_boxes(
    boxes: list[NusBox], points_global: np.ndarray, ego_translation: tuple[float, float, float]
) -> tuple[list[Box3D], list[dict]]:
    """NusBox → (渲染 Box3D 列表, 渲染 dict 列表)：fit_points = 框内点实测（LiDAR 观测性）。"""
    pts_cam = global_to_cam_like(points_global, ego_translation)
    boxes3d: list[Box3D] = []
    dicts: list[dict] = []
    for nb in boxes:
        d = nusbox_to_box3d_dict(nb, ego_translation)
        b = Box3D.from_dict(d)
        b.fit_points = points_in_box(pts_cam, b)
        boxes3d.append(b)
        d["fit_points"] = b.fit_points
        dicts.append(d)
    return boxes3d, dicts


def triage_nus(
    boxes: list[NusBox],
    points_global: np.ndarray,
    ego_translation: tuple[float, float, float],
    tau_high: float = 0.7,
    tau_low: float = 0.3,
) -> Triage3D:
    """NusBox → cam_like 渲染 Box3D（fit_points 实测）→ 复用 triage_3d 三档。

    min_fit_points 用 NUSCENES_MIN_FIT_POINTS=10（32 线密度低于 KITTI 64 线）。
    """
    boxes3d, _dicts = _render_boxes(boxes, points_global, ego_translation)
    return triage_3d(
        boxes3d, tau_high=tau_high, tau_low=tau_low,
        min_fit_points=NUSCENES_MIN_FIT_POINTS,
    )


def build_nuscenes_review_queue(
    scene_name: str,
    sample: dict,
    pred_boxes: list[NusBox],
    points_global: np.ndarray,
    ego_translation: tuple[float, float, float],
    nusc: Any,
    dataroot: Path,
    out_dir: Path,
    tau_high: float = 0.7,
    tau_low: float = 0.3,
) -> Path:
    """sample → {sample_token}_review.json（仅 review+hard 两档；纯函数，Fake 可测）。

    payloads/server 消费键（P2-3 契约）：dataset/version/dataroot/scene_name/sample_token/
    pcd_path（绝对路径，payloads 直读免 devkit）/ego_translation/cameras/annotations。
    """
    boxes3d, dicts = _render_boxes(pred_boxes, points_global, ego_translation)
    triage = triage_3d(
        boxes3d, tau_high=tau_high, tau_low=tau_low,
        min_fit_points=NUSCENES_MIN_FIT_POINTS,
    )
    review_ids = {id(b) for b in triage.review + triage.hard}
    annotations = [dicts[i] for i, b in enumerate(boxes3d) if id(b) in review_ids]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = {
        "dataset": "nuscenes",
        "version": getattr(nusc, "version", ""),
        "dataroot": str(dataroot),
        "scene_name": scene_name,
        "sample_token": sample["token"],
        "image": sample["token"],  # 前端标题（无单帧主图，6 相机走 cameras）
        "pcd_path": str(dataroot / lidar_sample_data(sample, nusc)["filename"]),
        "ego_translation": list(ego_translation),
        "cameras": cameras_of_sample(sample, nusc),
        "image_size": [1600, 900],  # nuScenes 相机分辨率（前端展示用）
        "bev_path": "",
        "description": "nuScenes 3D 复核队列 — 中置信度/拟合质量不足样本需人工确认",
        "annotations": annotations,
        "summary": {
            "accepted": len(triage.accepted),
            "review_count": len(triage.review),
            "hard_count": len(triage.hard),
            "total_need_review": len(triage.review) + len(triage.hard),
        },
    }
    path = out / f"{sample['token']}_review.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
