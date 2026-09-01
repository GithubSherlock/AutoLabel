"""nuScenes 标注管线（v0.4 P2）：三引擎协议统一 → 全局 NusBox + 复核队列生成。

predict_sample_nus 分派（零坐标系差异泄漏到队列层）：
- LiDAR 引擎（detect_points，centerpoint/pointpillars）：点云直推 → 传感器系 9 值
  → boxes_sensor_to_global（两级补偿链，公共）
- 融合引擎（detect_sample 4 位置参，bevfusion）：同 9 值契约 → 同补偿链
- 单目引擎（detect_sample 3 位置参，fcos3d）：直接返回全局 NusBox
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from auto3dlabel.configs.kitti import DEFAULT_CONF
from auto3dlabel.data.nuscenes import (
    boxes_sensor_to_global,
    ego_translation_of_sample,
    lidar_sample_data,
    load_lidar_points,
    load_nuscenes,
    samples_of_scene,
    val_scene_names,
)
from auto3dlabel.export.nuscenes_queue import build_nuscenes_review_queue
from auto3dlabel.schema.nuscenes_box import NusBox


def _sensor_outputs_to_global(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    sample: dict,
    nusc: Any,
    class_names: list[str],
) -> list[NusBox]:
    """传感器系 9 值输出 → 全局 NusBox（LIDAR_TOP 的 ego/calib 查表 + 公共补偿链）。"""
    lidar_data = lidar_sample_data(sample, nusc)
    ego = nusc.get("ego_pose", lidar_data["ego_pose_token"])
    calib = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
    return boxes_sensor_to_global(boxes, scores, labels, class_names, ego, calib)


def predict_sample_nus(
    det: Any,
    sample: dict,
    nusc: Any,
    dataroot: Path,
    conf: float,
    class_names: list[str],
) -> list[NusBox]:
    """三引擎协议统一：nuScenes sample → 全局系 NusBox 列表（≥ conf 阈值）。

    detect_sample 按位置参数个数分派：≥4 = 融合（sample, nusc, dataroot, conf_threshold）
    / ==3 = 单目（sample, nusc, conf_threshold，直返 NusBox）。
    """
    if hasattr(det, "detect_points"):
        pts = load_lidar_points(sample, nusc, dataroot)
        boxes, scores, labels = det.detect_points(pts, conf)
        if len(boxes) == 0:
            return []
        return _sensor_outputs_to_global(boxes, scores, labels, sample, nusc, class_names)
    if hasattr(det, "detect_sample"):
        sig = inspect.signature(det.detect_sample)
        positional = [
            p
            for p in sig.parameters.values()
            if p.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if len(positional) >= 4:  # 融合：sample, nusc, dataroot, conf_threshold
            boxes, scores, labels = det.detect_sample(
                sample, nusc, dataroot, conf_threshold=conf
            )
            if len(boxes) == 0:
                return []
            return _sensor_outputs_to_global(
                boxes, scores, labels, sample, nusc, class_names
            )
        return list(det.detect_sample(sample, nusc, conf_threshold=conf))  # 单目
    raise TypeError(f"引擎无 detect_points/detect_sample 协议: {type(det).__name__}")


def queue_one_sample(
    predict_fn: Callable[[dict], list[NusBox]],
    sample: dict,
    scene_name: str,
    nusc: Any,
    dataroot: Path,
    out_dir: Path,
    tau_high: float = 0.7,
    tau_low: float = 0.3,
) -> Path:
    """单 sample → {sample_token}_review.json（predict_fn 注入 → Fake 零权重可测）。"""
    pred = predict_fn(sample)
    points = load_lidar_points(sample, nusc, dataroot)
    ego = ego_translation_of_sample(sample, nusc)
    return build_nuscenes_review_queue(
        scene_name=scene_name,
        sample=sample,
        pred_boxes=pred,
        points_global=points,
        ego_translation=ego,
        nusc=nusc,
        dataroot=dataroot,
        out_dir=out_dir,
        tau_high=tau_high,
        tau_low=tau_low,
    )


def generate_review_queue(
    det: Any,
    out_dir: Path,
    conf: float = DEFAULT_CONF,
    dataroot: Path | None = None,
    version: str = "v1.0-mini",
    tau_high: float = 0.7,
    tau_low: float = 0.3,
) -> dict[str, Path]:
    """val 场景全量 → 复核队列（resume 幂等：已存在 {sample_token}_review.json 跳过）。"""
    nusc = load_nuscenes(root=dataroot, version=version)
    dataroot_p = Path(nusc.dataroot)
    class_names = list(det.class_names)  # 触发懒加载——config 真实类序
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    predict = partial(
        predict_sample_nus, det, nusc=nusc, dataroot=dataroot_p,
        conf=conf, class_names=class_names,
    )
    written: dict[str, Path] = {}
    for scene_name in val_scene_names(version):
        for sample in samples_of_scene(nusc, scene_name):
            path = out / f"{sample['token']}_review.json"
            if path.is_file():
                continue  # resume：已存在跳过（幂等）
            written[sample["token"]] = queue_one_sample(
                predict, sample, scene_name, nusc, dataroot_p, out,
                tau_high=tau_high, tau_low=tau_low,
            )
    return written
