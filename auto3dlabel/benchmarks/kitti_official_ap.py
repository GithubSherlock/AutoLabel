"""KITTI 官方 40-point AP 评测（与模型 zoo 数字对表口径）。

与现有 evaluate_per_class（11-point + 全类 IoU 0.5）的差异：
- AP 采样 = 0:0.025:1 共 41 个 recall 点（KITTI 官方 40 区间口径）
- 每类 IoU 阈值 = KITTI_OFFICIAL_IOU（Car 0.7 / Pedestrian 0.5 / Cyclist 0.5）
- 匹配逻辑同 evaluate_per_class：贪心 conf 降序 + GT 唯一匹配 + 难度分层 GT × 全量预测
双口径并存：本模块与 kitti3d_benchmark.py 分别出表，验收按官方口径。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from auto3dlabel.benchmarks.load_gt3d import gt_frame, pred_frame
from auto3dlabel.configs.kitti import AP_NUM_POINTS, KITTI_EVAL_CLASSES, KITTI_OFFICIAL_IOU
from auto3dlabel.schema.box3d import KittiFrame
from auto3dlabel.tools.geometry import iou3d_list

# 官方 clean_data 的预测 2D 高度豁免阈值（easy 40 / moderate 25 / hard 25 px）
_MIN_HEIGHT = {"easy": 40.0, "moderate": 25.0, "hard": 25.0}
_DC_IOU = 0.7  # 官方 DontCare 2D 豁免 IoU（image_box_overlap criterion=-1）


def _image_iou(a: list[float], b: list[float]) -> float:
    """2D IoU（官方 image_box_overlap criterion=-1：交 / 并）。"""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    if iw <= 0:
        return 0.0
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if ih <= 0:
        return 0.0
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def official_ap(
    gt: dict[str, dict[str, Any]],
    pred: dict[str, list[dict[str, Any]]],
    class_name: str,
    iou_threshold: float,
    difficulty: str,
    iou_fn: Callable[[list[float], list[float]], float] = iou3d_list,
) -> dict[str, Any]:
    """官方 40-point AP（贪心 conf 降序 + 官方 FP 豁免机制）。

    官方口径同款（mmdet3d kitti_utils/eval.py clean_data + fused_compute_statistics）：
    - 预测 2D 高度 < MIN_HEIGHT[difficulty] → ignored_dt（不算 FP/TP）
    - 预测 2D 框与 DontCare 2D 框 IoU > 0.7 → 豁免（不算 FP）
    - AP40 = 41 点（0:0.025:1）中后 40 点（recall 0.025..1.0）最大 precision 平均
    - GT 侧难度过滤由 _filter_difficulty 完成（官方 ignored_gt 同判据）

    Args:
        gt: {frame_id: {"objects": [{"name", "bbox": 7 值, "bbox2d", "difficulty"}],
            "dontcares": [[x1,y1,x2,y2], ...]}}
        pred: {frame_id: [{"name", "bbox": 7 值, "bbox2d", "conf"}]}
        difficulty: easy/moderate/hard（决定 MIN_HEIGHT）
        iou_threshold: 该类官方 IoU 阈值（KITTI_OFFICIAL_IOU）。
    Returns:
        {"ap", "precision", "recall", "gt_count", "pred_count"}（ap 为 0-1 小数）
    """
    tp_list: list[float] = []
    fp_list: list[float] = []
    scores: list[float] = []
    n_gt = 0

    for img_id, gt_data in gt.items():
        gt_boxes = [
            o["bbox"] for o in gt_data["objects"] if o["name"] == class_name
        ]
        dontcares: list[list[float]] = gt_data.get("dontcares", [])
        n_gt += len(gt_boxes)
        gt_matched = [False] * len(gt_boxes)

        pred_boxes = sorted(
            (p for p in pred.get(img_id, []) if p["name"] == class_name),
            key=lambda p: p["conf"],
            reverse=True,
        )
        for pb in pred_boxes:
            b2d = pb.get("bbox2d", [0.0, 0.0, 0.0, 0.0])
            if b2d[3] - b2d[1] < _MIN_HEIGHT[difficulty]:
                continue  # ignored_dt：过小预测（官方不罚）
            if any(_image_iou(b2d, dc) > _DC_IOU for dc in dontcares):
                continue  # DontCare 2D 豁免（官方不罚）
            best_iou, best_idx = 0.0, -1
            for j, gb in enumerate(gt_boxes):
                if not gt_matched[j]:
                    iou = iou_fn(pb["bbox"], gb)
                    if iou > best_iou:
                        best_iou, best_idx = iou, j
            if best_iou >= iou_threshold and best_idx >= 0:
                tp_list.append(1.0)
                fp_list.append(0.0)
                gt_matched[best_idx] = True
            else:
                tp_list.append(0.0)
                fp_list.append(1.0)
            scores.append(pb["conf"])

    if not scores:
        return {"precision": 0.0, "recall": 0.0, "ap": 0.0, "gt_count": n_gt, "pred_count": 0}

    order = np.argsort(scores)[::-1]
    tp = np.cumsum(np.array(tp_list)[order])
    fp = np.cumsum(np.array(fp_list)[order])
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / max(n_gt, 1)
    # KITTI 官方 AP40：41 点中后 40 个 recall 采样（0.025..1.0），
    # 每点取其后最大 precision（单调化），再平均
    recall_points = np.linspace(0, 1, AP_NUM_POINTS + 1)[1:]
    ap = float(np.mean([np.max(precision[recall >= t], initial=0.0) for t in recall_points]))
    ap = round(ap, 4)
    return {
        "ap": ap,
        "precision": float(precision[-1]),
        "recall": float(recall[-1]),
        "gt_count": n_gt,
        "pred_count": int(tp[-1] + fp[-1]),
    }


def _filter_difficulty(
    gt: dict[str, dict[str, Any]], difficulty: str
) -> dict[str, dict[str, Any]]:
    """GT 按难度分层（同 kitti3d_benchmark._evaluate_layer 口径；预测全量；
    dontcares 透传——官方豁免判定与难度无关）。"""
    return {
        k: {
            "objects": [o for o in v["objects"] if o.get("difficulty") == difficulty],
            "dontcares": v.get("dontcares", []),
        }
        for k, v in gt.items()
    }


def run_kitti_official(
    frames: list[KittiFrame],
    predictions: dict[str, list[Any]],
    difficulties: list[str] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """帧列表 + {frame_id: [Box3D]} → 官方口径 {difficulty: {class: {ap, ...}}}。

    ap 为 0-1 小数（zoo 对照时 ×100）；IoU 阈值取 KITTI_OFFICIAL_IOU 每类官方值。
    """
    difficulties = difficulties or ["easy", "moderate", "hard"]
    gt: dict[str, dict[str, Any]] = {f.frame_id: gt_frame(f) for f in frames}
    pred: dict[str, list[dict[str, Any]]] = {
        fid: pred_frame(boxes) for fid, boxes in predictions.items()
    }
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for diff in difficulties:
        gt_f = _filter_difficulty(gt, diff)
        result[diff] = {}
        for cls in KITTI_EVAL_CLASSES:
            result[diff][cls] = official_ap(
                gt_f, pred, cls, iou_threshold=KITTI_OFFICIAL_IOU[cls], difficulty=diff
            )
    return result
