"""nuScenes 简化 3D 评测（v0.2 P3，口径如实记录，不冒充官方 TP 指标）。

与官方 nuScenes-eval 的差异（记录在案）：
- 匹配 = 全局系 x-y 旋转矩形 IoU（shapely），忽略 z/高度——官方为平移距离阈值 TP 定义
- AP = 召回 0.1:0.1:1 共 10 点插值平均（官方同 10 点口径）
- 距离分桶简化为 0-25m / 25-50m（官方 0-50m 四桶）
- 数据 = v1.0-mini val 2 场景（官方 val 150 场景）→ zoo 数字对照仅供参考
"""

from __future__ import annotations

from typing import Any

import numpy as np
from shapely.geometry import Polygon

from auto3dlabel.configs.nuscenes import DISTANCE_BINS, NUSCENES_CLASSES, NUSCENES_IOU_THRESHOLD
from auto3dlabel.schema.nuscenes_box import NusBox
from auto3dlabel.tools.geometry import quat_to_yaw


def _xy_polygon(box: NusBox) -> Polygon:
    """全局系 x-y 旋转矩形（车头 yaw = quat_to_yaw，尺寸 (w, l)）。"""
    cx, cy = box.translation[0], box.translation[1]
    w, l = box.size[0], box.size[1]
    yaw = quat_to_yaw(box.quaternion)
    dx, dy = np.cos(yaw), np.sin(yaw)  # 车头
    px, py = -dy, dx  # 侧向（左）
    corners = np.array(
        [
            [cx + dx * l / 2 + px * w / 2, cy + dy * l / 2 + py * w / 2],
            [cx + dx * l / 2 - px * w / 2, cy + dy * l / 2 - py * w / 2],
            [cx - dx * l / 2 - px * w / 2, cy - dy * l / 2 - py * w / 2],
            [cx - dx * l / 2 + px * w / 2, cy - dy * l / 2 + py * w / 2],
        ],
        dtype=np.float64,
    )
    return Polygon(corners)


def nus_iou(a: NusBox, b: NusBox) -> float:
    """两 NusBox 全局系 x-y 旋转矩形 IoU（退化/无交返回 0）。"""
    pa, pb = _xy_polygon(a), _xy_polygon(b)
    if pa.is_empty or pb.is_empty or pa.area <= 0 or pb.area <= 0:
        return 0.0
    inter = pa.intersection(pb).area
    union = pa.area + pb.area - inter
    return float(inter / union) if union > 0 else 0.0


def _distance(box: NusBox, ego: tuple[float, float] = (0.0, 0.0)) -> float:
    """全局系 x-y 平面到自车位置的距离（官方 distance 口径；ego 缺省 = 原点）。

    注意：nus 全局坐标是 city 系（目标距原点常 >500m）——相对全局原点分桶无意义，
    必须传 ego 位置（run_nuscenes_benchmark 的 egos 参数）。
    """
    return float(np.hypot(box.translation[0] - ego[0], box.translation[1] - ego[1]))


def nuscenes_ap(
    gt: list[NusBox],
    pred: list[NusBox],
    class_name: str,
    iou_threshold: float = NUSCENES_IOU_THRESHOLD,
) -> dict[str, Any]:
    """单类 AP（贪心 conf 降序 + GT 唯一匹配 + 召回 10 点插值）。

    同 kitti_official_ap.official_ap 结构，但输入/输出是 NusBox 列表（单 token 聚合后）。
    """
    gt_boxes = [b for b in gt if b.label == class_name]
    pred_boxes = sorted(
        (b for b in pred if b.label == class_name),
        key=lambda b: b.confidence,
        reverse=True,
    )
    tp_list: list[float] = []
    fp_list: list[float] = []
    gt_matched = [False] * len(gt_boxes)
    for pb in pred_boxes:
        best_iou, best_idx = 0.0, -1
        for j, gb in enumerate(gt_boxes):
            if not gt_matched[j]:
                iou = nus_iou(pb, gb)
                if iou > best_iou:
                    best_iou, best_idx = iou, j
        if best_iou >= iou_threshold and best_idx >= 0:
            tp_list.append(1.0)
            fp_list.append(0.0)
            gt_matched[best_idx] = True
        else:
            tp_list.append(0.0)
            fp_list.append(1.0)

    if not tp_list:
        return {"ap": 0.0, "gt_count": len(gt_boxes), "pred_count": 0}

    order = np.argsort([b.confidence for b in pred_boxes])[::-1]
    tp = np.cumsum(np.array(tp_list)[order])
    fp = np.cumsum(np.array(fp_list)[order])
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / max(len(gt_boxes), 1)
    # 官方 10 召回点 0.1:0.1:1 取其后最大 precision 平均；保留 4 位小数（同 kitti 口径）
    recall_points = np.arange(0.1, 1.01, 0.1)
    ap = float(np.mean([np.max(precision[recall >= t], initial=0.0) for t in recall_points]))
    return {
        "ap": round(ap, 4),
        "gt_count": len(gt_boxes),
        "pred_count": int(tp[-1] + fp[-1]),
    }


def run_nuscenes_benchmark(
    gt: dict[str, list[NusBox]],
    pred: dict[str, list[NusBox]],
    class_names: list[str] | None = None,
    egos: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """{sample_token: [NusBox]} → {overall: {class: {ap}}, distance: {bin: {class: {ap}}}, mAP}。

    egos: {sample_token: (ego_x, ego_y)}——距离分桶相对**自车**（官方 distance 口径；
    缺省相对全局原点，仅在坐标近原点的合成测试下等价）。overall 不受 egos 影响。
    简化口径（见模块 docstring）：与官方 val 全量数字不可直接对等，记录对照仅供参考。
    """
    classes = class_names or NUSCENES_CLASSES
    all_gt = [b for boxes in gt.values() for b in boxes]
    all_pred = [b for boxes in pred.values() for b in boxes]

    overall: dict[str, dict[str, Any]] = {}
    for cls in classes:
        overall[cls] = nuscenes_ap(all_gt, all_pred, cls)

    by_bin: dict[str, dict[str, dict[str, Any]]] = {}
    for lo, hi in DISTANCE_BINS:
        gt_bin: list[NusBox] = []
        pred_bin: list[NusBox] = []
        for token, boxes in gt.items():
            ego = egos[token] if egos and token in egos else (0.0, 0.0)
            gt_bin += [b for b in boxes if lo <= _distance(b, ego) < hi]
            pred_bin += [
                b for b in pred.get(token, []) if lo <= _distance(b, ego) < hi
            ]
        by_bin[f"{lo}-{hi}m"] = {
            cls: nuscenes_ap(gt_bin, pred_bin, cls) for cls in classes
        }

    map_ap = round(float(np.mean([overall[c]["ap"] for c in classes])), 4)
    return {"overall": overall, "distance": by_bin, "mAP": map_ap}
