"""nuScenes 路径/类映射/评测口径单一事实源（同 configs/kitti.py 纪律，不得散副本）。

数据源（v0.2 P3，2026-08-27 实测定案）：
- AutoDL 公共数据盘 `/autodl-pub/data/nuScenes/Fulldatasetv1.0/Mini/v1.0-mini.tgz`（3.9G 单卷）
- 解压即官方标准 dataroot（samples/sweeps/maps/v1.0-mini 表），devkit 直用
- 完整 Trainval 为 10 卷 blob tgz（294G）→ 用 Mini（10 场景）做 P3 简化评测；
  与 zoo 数字（官方 val 全量）对表时如实标注口径差异
"""

from __future__ import annotations

import os
from pathlib import Path

# 解压后的标准 dataroot（v1.0-mini）：环境变量 NUSCENES_ROOT 可覆盖
DEFAULT_NUSCENES_ROOT = Path(
    os.environ.get("NUSCENES_ROOT", "/root/autodl-tmp/Documents/datasets/nuscenes_mini")
)

# 官方 3D detection 任务 10 类（devkit NUSCENES_DETECTION_CLASSES 同序）
# mmdet3d nus-3d 模型输出名与此完全一致 → 映射恒等（仍走显式表，防顺序漂移）
NUSCENES_CLASSES = [
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
]

# mmdet3d 类名 → 官方检测类名（恒等映射的显式形式；未知类丢弃不导出）
MMDET3D_TO_NUSCENES = {c: c for c in NUSCENES_CLASSES}

# 官方 GT category_name → 检测类名（devkit 23 类映射表；None = 官方忽略类不参与评测）。
# pedestrian/bus 的 category 带第三段子类——split('.')[-1] 会错成 adult/rigid（2026-08-27
# Mini 冒烟实证 pedestrian/bus GT 全丢），必须走显式全表
NUSCENES_CATEGORY_MAP: dict[str, str | None] = {
    "animal": None,
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.personal_mobility": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.stroller": "pedestrian",
    "human.pedestrian.wheelchair": "pedestrian",
    "movable_object.barrier": "barrier",
    "movable_object.debris": None,
    "movable_object.pushable_pullable": None,
    "movable_object.trafficcone": "traffic_cone",
    "static_object.bicycle_rack": None,
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.emergency.ambulance": None,
    "vehicle.emergency.police": None,
    "vehicle.motorcycle": "motorcycle",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}

# 评测口径：全局系（x 前 y 左 z 上）旋转矩形 x-y 平面 IoU + 按距离分桶（简化版
# 不实现官方 TP 指标，如实记录口径差异，见 benchmarks/nuscenes_benchmark.py）
NUSCENES_IOU_THRESHOLD = 0.5
DISTANCE_BINS = [(0, 25), (25, 50)]  # 简化分桶：近/中（官方 0-50m 四桶的降级）

# nuScenes 6 相机通道（devkit 标准序，BEVFusion 多视角输入）
NUSCENES_CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

# BEVFUSION_NAMES / FCOS3D_NAMES（nuScenes 融合/单目路由表）已收拢至
# configs/model_catalog.py（单一事实源，v0.3 P6），勿在别处散副本
