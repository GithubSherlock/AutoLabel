"""KITTI 官方 40-point AP 评测（mmdet3d kitti_utils/eval.py 精确移植，numba 免编译版）。

与现有 evaluate_per_class（11-point + 全类 IoU 0.5）的差异：
- 难度累积分层：hard = 全部 GT；moderate = easy+moderate；easy = 仅 easy
  （官方 clean_data 在完整 GT 池上打 ignore 标记，不预筛难度段）
- ignored GT（更难难度对象 + Van/Person_sitting 别名）留在匹配池**吸收**预测 → 豁免不罚 FP
- 预测 2D 高度 < MIN_HEIGHT → ignored_dt（不算 TP/FP）；DontCare 豁免判据 =
  (预测 ∩ DontCare)/预测面积 > IoU 阈值（官方 image_box_overlap criterion=0，非并集 IoU）
- 阈值采样（官方 get_thresholds）：仅 TP 分数按 recall 1/40 步进抽 ≤41 个阈值，
  每个阈值独立重匹配计数；低于最低阈值的预测全部豁免
- 匹配规则 per-GT 贪心（GT 行序）：阈值收集 pass 按最高分（compute_fp=False），
  计数 pass 按最大重叠、无有效匹配时 ignored 检测可占位（compute_fp=True）
- AP40 = 41 点（0:0.025:1）中后 40 点单调化 precision 平均；阈值数 < 40 时缺失点
  precision = 0（官方同款稀疏惩罚：小样本集 AP 偏低是官方行为，非缺陷）

移植依据：mmdet3d 1.4.0 evaluation/functional/kitti_utils/eval.py
（clean_data / compute_statistics_jit / get_thresholds / get_mAP40 逐行对译）。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from auto3dlabel.benchmarks.load_gt3d import gt_frame_official, pred_frame
from auto3dlabel.configs.kitti import KITTI_EVAL_CLASSES, KITTI_OFFICIAL_IOU
from auto3dlabel.schema.box3d import KittiFrame
from auto3dlabel.tools.geometry import iou3d_list

# 官方 clean_data 常量（难度下标 0/1/2 = easy/moderate/hard，官方 eval.py 原值）
_MIN_HEIGHT = (40.0, 25.0, 25.0)
_MAX_OCCLUSION = (0, 1, 2)
_MAX_TRUNCATION = (0.15, 0.3, 0.5)
_DIFFICULTY = {"easy": 0, "moderate": 1, "hard": 2}
# 官方 valid_class==0：别名类吸收匹配 → 豁免（clean_data 原表）
_ALIAS = {"Car": "Van", "Pedestrian": "Person_sitting"}
_N_SAMPLE_PTS = 41  # 官方 N_SAMPLE_PTS（AP40 = 41 点中后 40 点平均）


def _gt_flags(objects: list[dict[str, Any]], class_name: str, diff_idx: int) -> list[int]:
    """clean_data GT 段：0 有效 / 1 ignored（更难难度或别名）/ -1 其他类。

    ignore = occ > MAX_OCCLUSION 或 trunc > MAX_TRUNCATION 或 2D 高 <= MIN_HEIGHT
    （高度边界为 <=，与官方逐字一致）。
    """
    flags: list[int] = []
    for o in objects:
        name = o["name"].lower()
        if name == class_name.lower():
            valid = 1
        elif _ALIAS.get(class_name, "").lower() == name:
            valid = 0
        else:
            valid = -1
        b2d = o["bbox2d"]
        ignore = (
            o["occluded"] > _MAX_OCCLUSION[diff_idx]
            or o["truncated"] > _MAX_TRUNCATION[diff_idx]
            or (b2d[3] - b2d[1]) <= _MIN_HEIGHT[diff_idx]
        )
        if valid == 1 and not ignore:
            flags.append(0)
        elif valid == 0 or (ignore and valid == 1):
            flags.append(1)
        else:
            flags.append(-1)
    return flags


def _dt_flags(preds: list[dict[str, Any]], class_name: str, diff_idx: int) -> list[int]:
    """clean_data DT 段：0 有效 / 1 高度豁免（2D 高 < MIN_HEIGHT，先于类判定）/ -1 其他类。"""
    flags: list[int] = []
    for p in preds:
        b2d = p.get("bbox2d", [0.0, 0.0, 0.0, 0.0])
        height = abs(b2d[3] - b2d[1])
        if height < _MIN_HEIGHT[diff_idx]:
            flags.append(1)
        elif p["name"].lower() == class_name.lower():
            flags.append(0)
        else:
            flags.append(-1)
    return flags


def _iou_matrix(
    gt_rows: list[dict[str, Any]],
    dt_rows: list[dict[str, Any]],
    iou_fn: Callable[[list[float], list[float]], float],
) -> np.ndarray:
    """(n_dt, n_gt) 3D IoU 矩阵——每帧计算一次（全行池），三难度 × 三类复用。"""
    n_dt, n_gt = len(dt_rows), len(gt_rows)
    m = np.zeros((n_dt, n_gt), dtype=np.float64)
    for i, gb in enumerate(gt_rows):
        for j, db in enumerate(dt_rows):
            m[j, i] = iou_fn(db["bbox"], gb["bbox"])
    return m


def _pass_a(
    overlaps: np.ndarray,
    gt_flags: list[int],
    dt_flags: list[int],
    dt_scores: list[float],
    iou_thr: float,
) -> list[float]:
    """compute_fp=False（thresh=0）：per-GT 最高分匹配 → 有效 TP 分数（阈值采样用）。

    ignored 匹配（GT 或 DT 任一侧 flagged 1）照常占用检测，但不产出 TP 分数。
    """
    n_dt, n_gt = overlaps.shape
    assigned = [False] * n_dt
    tp_scores: list[float] = []
    for i in range(n_gt):
        if gt_flags[i] == -1:
            continue
        det_idx, best = -1, -1e9
        for j in range(n_dt):
            if dt_flags[j] == -1 or assigned[j]:
                continue
            if overlaps[j, i] > iou_thr and dt_scores[j] > best:
                det_idx, best = j, dt_scores[j]
        if det_idx < 0:
            continue
        assigned[det_idx] = True
        if gt_flags[i] != 1 and dt_flags[det_idx] != 1:
            tp_scores.append(dt_scores[det_idx])
    return tp_scores


def _dt_dc_overlap(dt_b2d: list[float], dc: list[float]) -> float:
    """官方 image_box_overlap criterion=0：交 / 预测面积（DontCare 豁免判据）。"""
    iw = min(dt_b2d[2], dc[2]) - max(dt_b2d[0], dc[0])
    if iw <= 0:
        return 0.0
    ih = min(dt_b2d[3], dc[3]) - max(dt_b2d[1], dc[1])
    if ih <= 0:
        return 0.0
    area_dt = (dt_b2d[2] - dt_b2d[0]) * (dt_b2d[3] - dt_b2d[1])
    return iw * ih / area_dt if area_dt > 0 else 0.0


def _pass_b(
    overlaps: np.ndarray,
    gt_flags: list[int],
    dt_flags: list[int],
    dt_scores: list[float],
    dt_b2d: list[list[float]],
    dc_boxes: list[list[float]],
    iou_thr: float,
    score_thr: float,
) -> tuple[int, int, int]:
    """compute_fp=True：单阈值计数 → (tp, fp, fn)（compute_statistics_jit 逐行移植）。

    匹配：per-GT 贪心，有效检测按最大重叠；无有效匹配时 ignored 检测可占位
    （占位后被任何后续有效检测覆盖）；FP = 未分配有效检测 - DontCare 覆盖。
    """
    n_dt, n_gt = overlaps.shape
    assigned = [False] * n_dt
    below = [s < score_thr for s in dt_scores]
    tp = fp = fn = 0
    no_detection = -1e9
    for i in range(n_gt):
        if gt_flags[i] == -1:
            continue
        det_idx = -1
        valid_detection = no_detection
        max_overlap = 0.0
        assigned_ignored = False
        for j in range(n_dt):
            if dt_flags[j] == -1 or assigned[j] or below[j]:
                continue
            ov = overlaps[j, i]
            if ov > iou_thr and (ov > max_overlap or assigned_ignored) and dt_flags[j] == 0:
                max_overlap = ov
                det_idx = j
                valid_detection = 1.0
                assigned_ignored = False
            elif ov > iou_thr and valid_detection == no_detection and dt_flags[j] == 1:
                det_idx = j
                valid_detection = 1.0
                assigned_ignored = True
        if valid_detection == no_detection:
            if gt_flags[i] == 0:
                fn += 1
        elif gt_flags[i] == 1 or dt_flags[det_idx] == 1:
            assigned[det_idx] = True
        else:
            tp += 1
            assigned[det_idx] = True
    for j in range(n_dt):
        if not (assigned[j] or dt_flags[j] == -1 or dt_flags[j] == 1 or below[j]):
            fp += 1
    # DontCare 豁免（官方 criterion=0：交/预测面积 > IoU 阈值）
    for dc in dc_boxes:
        for j in range(n_dt):
            if assigned[j] or dt_flags[j] == -1 or dt_flags[j] == 1 or below[j]:
                continue
            if _dt_dc_overlap(dt_b2d[j], dc) > iou_thr:
                assigned[j] = True
                fp -= 1
    return tp, fp, fn


def _get_thresholds(scores_desc: list[float], num_gt: int) -> list[float]:
    """官方 get_thresholds：TP 分数按 recall 1/40 步进抽 ≤41 个阈值（降序）。"""
    thresholds: list[float] = []
    if not scores_desc or num_gt <= 0:
        return thresholds
    current_recall = 0.0
    step = 1.0 / (_N_SAMPLE_PTS - 1)
    n = len(scores_desc)
    for i, score in enumerate(scores_desc):
        l_recall = (i + 1) / num_gt
        r_recall = (i + 2) / num_gt if i < n - 1 else l_recall
        if (r_recall - current_recall) < (current_recall - l_recall) and i < n - 1:
            continue
        thresholds.append(score)
        current_recall += step
        if len(thresholds) >= _N_SAMPLE_PTS:
            break
    return thresholds


def official_ap(
    gt: dict[str, dict[str, Any]],
    pred: dict[str, list[dict[str, Any]]],
    class_name: str,
    iou_threshold: float,
    difficulty: str,
    iou_fn: Callable[[list[float], list[float]], float] = iou3d_list,
    iou_matrices: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """官方 40-point AP（单类单难度；难度累积分层 + ignored 豁免 + 阈值采样）。

    Args:
        gt: {frame_id: {"objects": [...], "dontcares": [...]}}（gt_frame_official 产物）
        pred: {frame_id: [{"name", "bbox": 7 值, "bbox2d", "conf"}]}（pred_frame 产物）
        difficulty: easy/moderate/hard（官方下标 0/1/2，MIN_HEIGHT 依此取）
        iou_threshold: 该类官方 IoU 阈值（KITTI_OFFICIAL_IOU）。
        iou_matrices: {frame_id: (n_dt, n_gt)} 预计算矩阵（run_kitti_official 复用，
            None 时逐帧现算——行序必须与 objects/pred 一致）。
    Returns:
        {"ap", "precision", "recall", "gt_count", "pred_count"}（ap 为 0-1 小数；
        precision/recall 为单调化后最高阈值点值；gt_count 只计有效 GT）
    """
    diff_idx = _DIFFICULTY[difficulty]
    n_valid_gt = 0
    n_preds = 0
    tp_scores_all: list[float] = []
    frame_entries: list[
        tuple[list[int], list[int], list[float], list[list[float]], list[list[float]], np.ndarray]
    ] = []
    for fid, g in gt.items():
        gt_rows = g["objects"]
        dt_rows = pred.get(fid, [])
        gt_flags = _gt_flags(gt_rows, class_name, diff_idx)
        dt_flags = _dt_flags(dt_rows, class_name, diff_idx)
        n_valid_gt += gt_flags.count(0)
        n_preds += len(dt_rows)
        overlaps = (
            iou_matrices[fid] if iou_matrices is not None and fid in iou_matrices
            else _iou_matrix(gt_rows, dt_rows, iou_fn)
        )
        dt_scores = [p["conf"] for p in dt_rows]
        dt_b2d = [p.get("bbox2d", [0.0, 0.0, 0.0, 0.0]) for p in dt_rows]
        frame_entries.append(
            (gt_flags, dt_flags, dt_scores, dt_b2d, g.get("dontcares", []), overlaps)
        )
        tp_scores_all.extend(_pass_a(overlaps, gt_flags, dt_flags, dt_scores, iou_threshold))

    thresholds = _get_thresholds(sorted(tp_scores_all, reverse=True), n_valid_gt)
    pr = np.zeros((len(thresholds), 3))  # 每行 = (tp, fp, fn)
    for gt_flags, dt_flags, dt_scores, dt_b2d, dc_boxes, overlaps in frame_entries:
        for t, thr in enumerate(thresholds):
            tp, fp, fn = _pass_b(
                overlaps, gt_flags, dt_flags, dt_scores, dt_b2d, dc_boxes, iou_threshold, thr
            )
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn

    if len(thresholds) == 0:
        return {
            "ap": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "gt_count": n_valid_gt,
            "pred_count": n_preds,
        }

    recall = pr[:, 0] / np.maximum(pr[:, 0] + pr[:, 2], 1e-12)
    precision = pr[:, 0] / np.maximum(pr[:, 0] + pr[:, 1], 1e-12)
    # 官方单调化（max over trailing）；缺失点保持 0（稀疏惩罚）
    for i in range(len(thresholds)):
        precision[i] = float(precision[i:].max())
        recall[i] = float(recall[i:].max())
    padded = np.zeros(_N_SAMPLE_PTS)
    padded[: len(thresholds)] = precision
    ap = round(float(padded[1:].mean()), 4)  # 官方 get_mAP40：41 点中后 40 点
    return {
        "ap": ap,
        "precision": float(precision[0]),
        "recall": float(recall[0]),
        "gt_count": n_valid_gt,
        "pred_count": n_preds,
    }


def run_kitti_official(
    frames: list[KittiFrame],
    predictions: dict[str, list[Any]],
    difficulties: list[str] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """帧列表 + {frame_id: [Box3D]} → 官方口径 {difficulty: {class: {ap, ...}}}。

    ap 为 0-1 小数（zoo 对照时 ×100）；IoU 阈值取 KITTI_OFFICIAL_IOU 每类官方值。
    IoU 矩阵每帧预计算一次，9 组（难度 × 类）复用。
    """
    difficulties = difficulties or ["easy", "moderate", "hard"]
    gt: dict[str, dict[str, Any]] = {f.frame_id: gt_frame_official(f) for f in frames}
    pred: dict[str, list[dict[str, Any]]] = {
        fid: pred_frame(boxes) for fid, boxes in predictions.items()
    }
    matrices: dict[str, np.ndarray] = {
        fid: _iou_matrix(g["objects"], pred.get(fid, []), iou3d_list)
        for fid, g in gt.items()
    }
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for diff in difficulties:
        result[diff] = {}
        for cls in KITTI_EVAL_CLASSES:
            result[diff][cls] = official_ap(
                gt, pred, cls, iou_threshold=KITTI_OFFICIAL_IOU[cls], difficulty=diff,
                iou_matrices=matrices,
            )
    return result
