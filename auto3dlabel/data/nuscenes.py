"""nuScenes 数据访问层（v0.2 P3）：dataroot 校验 + devkit 懒加载守卫 + val 枚举 + GT → NusBox。

数据源：/autodl-pub/data/nuScenes/Fulldatasetv1.0/Mini/v1.0-mini.tgz 解压后的
标准 dataroot（见 configs/nuscenes.DEFAULT_NUSCENES_ROOT）。
GT 走 devkit 原始表（sample_annotation 数值 dict），不构造 devkit Box（免 pyquaternion 路径开销）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from auto3dlabel.configs.nuscenes import DEFAULT_NUSCENES_ROOT, NUSCENES_CATEGORY_MAP
from auto3dlabel.schema.nuscenes_box import NusBox


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
