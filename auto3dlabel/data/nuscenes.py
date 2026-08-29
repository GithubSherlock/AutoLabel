"""nuScenes 数据访问层（v0.2 P3）：dataroot 校验 + devkit 懒加载守卫 + val 枚举 + GT → NusBox
+ 传感器系→全局系补偿链（v0.3 P3 从 smoke_nuscenes 抽公共，smoke_bevfusion 复用）。

数据源：/autodl-pub/data/nuScenes/Fulldatasetv1.0/Mini/v1.0-mini.tgz 解压后的
标准 dataroot（见 configs/nuscenes.DEFAULT_NUSCENES_ROOT）。
GT 走 devkit 原始表（sample_annotation 数值 dict），不构造 devkit Box（免 pyquaternion 路径开销）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from auto3dlabel.configs.nuscenes import DEFAULT_NUSCENES_ROOT, NUSCENES_CATEGORY_MAP
from auto3dlabel.schema.nuscenes_box import NusBox
from auto3dlabel.tools.geometry import yaw_to_quat


def dataroot_exists(root: Path | None = None) -> bool:
    """标准 dataroot 校验（v1.0-mini 表 + LIDAR_TOP samples 齐备）。"""
    root = root or DEFAULT_NUSCENES_ROOT
    return (root / "v1.0-mini").is_dir() and (root / "samples" / "LIDAR_TOP").is_dir()


def load_nuscenes(root: Path | None = None, version: str = "v1.0-mini") -> Any:
    """devkit NuScenes 实例（懒加载；devkit 未装 → ImportError 带安装指引）。

    红线：补装必须 --no-deps（防 numpy 升 2.2.6 破坏 ABI，同 mmcv 纪律）。
    """
    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError as e:
        raise ImportError(
            "nuscenes-devkit 未安装（P3 依赖）: "
            "pip install nuscenes-devkit --no-deps + pyquaternion"
        ) from e
    root = root or DEFAULT_NUSCENES_ROOT
    if not dataroot_exists(root):
        raise FileNotFoundError(
            f"nuScenes dataroot 不存在: {root}（先解压 "
            "/autodl-pub/data/nuScenes/Fulldatasetv1.0/Mini/v1.0-mini.tgz 到此路径）"
        )
    return NuScenes(version=version, dataroot=str(root), verbose=False)


def val_scene_names(version: str = "v1.0-mini") -> list[str]:
    """devkit 官方 split 的 val 场景名（不硬编码场景列表）。

    devkit 版本键：v1.0-trainval → "val"；v1.0-mini → "mini_val"（键名映射在此登记）。
    """
    split_key = {"v1.0-mini": "mini_val", "v1.0-trainval": "val"}.get(version, "val")
    try:
        from nuscenes.utils.splits import create_splits_scenes
    except ImportError as e:
        raise ImportError("nuscenes-devkit 未安装: pip install nuscenes-devkit --no-deps") from e
    splits: Any = create_splits_scenes()  # devkit 无类型注解，索引推断为 slice
    if split_key not in splits:
        raise KeyError(f"split 无键 {split_key}（可用: {sorted(splits)}）")
    return list(splits[split_key])


def samples_of_scene(nusc: Any, scene_name: str) -> list[dict]:
    """场景 → sample 记录列表（按 first_sample_token 链序）。

    nuScenes 的 sample 表即 LIDAR_TOP keyframe 全集（每 sample 恰一个 LIDAR_TOP）。
    """
    scene = next((s for s in nusc.scene if s["name"] == scene_name), None)
    if scene is None:
        raise KeyError(f"场景不存在: {scene_name}")
    samples: list[dict] = []
    token: str = scene["first_sample_token"]
    while token:
        sample = nusc.get("sample", token)
        samples.append(sample)
        token = sample["next"]
    return samples


def gt_boxes_of_sample(nusc: Any, sample_token: str) -> list[NusBox]:
    """sample → GT NusBox 列表（原始表数值直映射，类名走官方 category 映射表）。"""
    boxes: list[NusBox] = []
    for ann in nusc.sample_annotation:
        if ann["sample_token"] != sample_token:
            continue
        if "num_lidar_pts" in ann and ann["num_lidar_pts"] <= 0:
            continue  # 无 LiDAR 观测的标注不参与（简化口径，如实记录）
        name = NUSCENES_CATEGORY_MAP.get(ann["category_name"])
        if name is None:
            continue  # 官方忽略类（animal/debris/emergency 等）不参与评测
        boxes.append(
            NusBox(
                label=name,
                confidence=1.0,
                translation=tuple(ann["translation"]),
                size=tuple(ann["size"]),
                quaternion=tuple(ann["rotation"]),
                track_id=ann["instance_token"],
            )
        )
    return boxes


def lidar_sample_data(sample: dict, nusc: Any) -> dict:
    """sample → LIDAR_TOP 的 sample_data 记录（filename/ego_pose/calibrated_sensor/timestamp）。

    ego_pose 与 LIDAR 位姿都挂在 LIDAR_TOP 的 sample_data 记录上（sample 表无此字段）——
    补偿链与 BEVFusion data_ 构建都从这里取。
    """
    record = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    assert isinstance(record, dict)
    return record


def rot_matrix(q: tuple[float, float, float, float]) -> np.ndarray:
    """四元数 (w,x,y,z) → 3x3 旋转矩阵（Hamilton 约定，nus devkit 同）。

    v0.3 P3 起公共：boxes_sensor_to_global 补偿链 + bevfusion3d 标定位姿共用。
    """
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_mul(
    q1: tuple[float, float, float, float], q2: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """Hamilton 积 q1 ⊗ q2（nus devkit Quaternion 同约定）。"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def boxes_sensor_to_global(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    class_names: list[str] | tuple[str, ...],
    ego: dict,
    calib: dict,
) -> list[NusBox]:
    """传感器系 9 值输出 → 全局系 NusBox（两级补偿链，v0.3 P3 抽公共）。

    nus 模型 9 值输出 [x,y,z,l,w,h,yaw,vx,vy]（传感器系中心 + 速度）——
    实测校准：2021-08 旧权重 L-W 顺序（与 1.4.0 DeltaXYZWLHRBBoxCoder 的
    W-L 定义相反，跨场景 81.5% car 匹配实证）；yaw 为标准语义（0=+x 前）。
    补偿：传感器系 → calib 位姿 → ego 地面系 → ego_pose → 全局系。
    纯函数：ego/calib 为 devkit ego_pose/calibrated_sensor 记录 dict
    （{'rotation': [w,x,y,z], 'translation': [x,y,z]}），由调用方查表传入；
    标签越界丢弃并每调用一次向 stderr 诊断（类表与 head 类数不符时）。
    """
    r_ego = rot_matrix(tuple(ego["rotation"]))
    t_ego = np.asarray(ego["translation"], dtype=np.float64)
    q_ego = tuple(ego["rotation"])
    r_calib = rot_matrix(tuple(calib["rotation"]))
    t_calib = np.asarray(calib["translation"], dtype=np.float64)
    q_calib = tuple(calib["rotation"])
    out: list[NusBox] = []
    warned = False
    for i in range(len(boxes)):
        idx = int(labels[i])
        if idx < 0 or idx >= len(class_names):
            if not warned:  # 诊断：类表与模型 head 类数不符（每调用一次）
                print(
                    f"[诊断] 标签越界跳过 idx={idx} len(class_names)={len(class_names)}",
                    file=sys.stderr,
                )
                warned = True
            continue
        x, y, z, l, w, h, yaw = (float(v) for v in boxes[i, :7])
        center = r_ego @ (r_calib @ np.array([x, y, z]) + t_calib) + t_ego
        quat = quat_mul(q_ego, quat_mul(q_calib, yaw_to_quat(yaw)))
        velocity: tuple[float, float] | None = None
        if boxes.shape[1] >= 9:
            v_glob = r_ego @ (r_calib @ np.array([float(boxes[i, 7]), float(boxes[i, 8]), 0.0]))
            velocity = (float(v_glob[0]), float(v_glob[1]))
        out.append(
            NusBox(
                label=class_names[idx],
                confidence=float(scores[i]),
                translation=(float(center[0]), float(center[1]), float(center[2])),
                size=(w, l, h),
                quaternion=quat,
                velocity=velocity,
            )
        )
    return out
