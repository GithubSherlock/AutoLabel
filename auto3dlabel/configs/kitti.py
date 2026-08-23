"""KITTI 路径/映射/阈值单一事实源（auto3dlabel）。

其他模块一律引用此处常量，不得散副本（同 auto2dlabel task_plan.DEFAULT_MODEL 纪律）。
"""

from __future__ import annotations

import os
from pathlib import Path

# 数据根：环境变量 KITTI_OBJECT_ROOT > 默认 AutoDL 数据集盘路径
DEFAULT_KITTI_ROOT = Path(
    os.environ.get("KITTI_OBJECT_ROOT", "/root/autodl-tmp/Documents/datasets/KITTI/object")
)

# COCO 类名（2D 检测器输出）→ KITTI 类名（导出/评测用）
# 评测只算 Car/Pedestrian/Cyclist 三类（KITTI 官方 3D 评测口径）；Truck/Tram 导出但不评测
COCO_TO_KITTI = {
    "car": "Car",
    "person": "Pedestrian",
    "bicycle": "Cyclist",
    "truck": "Truck",
    "train": "Tram",
    "tram": "Tram",
    "motorcycle": "Cyclist",
}
KITTI_EVAL_CLASSES = ["Car", "Pedestrian", "Cyclist"]

# 反向映射（KITTI 类名 → COCO 类名）：agent 质量评估 evaluate_detections 的
# prompt 覆盖检查按 COCO prompt 匹配（2D 语义），3D 框 label 是 KITTI 名
# （Pedestrian 不含 person 子串会误报缺类）——工具返回 dict 的 name 键携带 COCO 名
KITTI_TO_COCO = {
    "Car": "car",
    "Pedestrian": "person",
    "Cyclist": "bicycle",
    "Truck": "truck",
    "Tram": "train",
}

# 聚类 eps（BEV 平面，按类别尺度差异；垂直方向不参与聚类）
ADAPTIVE_EPS = {
    "Car": 1.0,
    "Pedestrian": 0.5,
    "Cyclist": 0.7,
    "Truck": 1.5,
    "Tram": 2.0,
}
DBSCAN_MIN_SAMPLES = 15

# 体素降采样（语义点 > 阈值时先降采样再聚类，防 DBSCAN 慢）
VOXEL_SIZE = 0.1
VOXEL_MAX_POINTS = 20000

# 拟合置信度（宁缺勿假红线）：低于门槛强制 review 档，供 HITL 分流
MIN_FIT_POINTS = 20
MIN_FIT_POINTS_ABS = 5  # 低于此值直接丢弃（退化）

# 粘连嫌疑：簇点数/BEV 长边超类上限 → review_flag（多实例遮挡合并的 HITL 兜底）
CLUSTER_MAX_POINTS = 8000
MAX_SIZE_FACTOR = 1.5
CLASS_MAX_DIMS = {  # (l, w) 鸟瞰长宽上限（米），略大于 KITTI 常见尺寸
    "Car": (7.0, 3.2),
    "Pedestrian": (1.6, 1.6),
    "Cyclist": (3.0, 2.0),
    "Truck": (14.0, 4.0),
    "Tram": (20.0, 3.6),
}

# 高度分位（相机 y 向，剔除地面/顶部噪点）
Z_PERCENT_LOW = 5
Z_PERCENT_HIGH = 95

# KITTI 官方 trainval split：train 000000-003711 / val 003712-007480（本地无 split 文件按区间推导）
VAL_FRAME_RANGE = (3712, 7481)

# 管线默认参数（标注档，宁多勿漏）
DEFAULT_CONF = 0.3
DEFAULT_IOU = 0.5
DEFAULT_DET_MODEL = "IDEA-Research/grounding-dino-tiny"
DEFAULT_SEG_MODEL = "sam2_l.pt"
