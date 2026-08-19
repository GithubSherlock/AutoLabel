"""Tracking 评测 —— CLEAR MOT 指标（MOTA/IDF1/IDSW/MT/ML）+ 检测口径同报。

遵循 v0.4 Phase 1 红线：评测必须同时报 MOTA/IDF1 与检测 recall/IDSW，
避免「跟踪器背检测的锅」——recall 低是检测问题，IDSW 高才是跟踪问题。

指标口径（对齐 MOTChallenge devkit / TrackEval）：
- 逐帧贪心 IoU 匹配（≥iou_threshold，每框至多匹配一次）→ TP/FP/FN
- MOTA = 1 - (FN + FP + IDSW) / num_gt
- IDSW：某 gt_id 当前帧匹配到的 pred_id 与上一匹配帧不同 → +1（累计）
- IDF1：gt_id × pred_id 全局二分匹配，权重 = 二者帧级共现帧数，
  匈牙利最大化总权重 → IDTP；IDF1 = 2·IDTP / (2·IDTP + IDFN + IDFP)
- MT/ML：轨迹被匹配帧数占自身长度 ≥0.8 / ≤0.2 的 GT 轨迹数
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrackBox:
    """单帧单框（MOT txt 行 / 评测中间表示）。"""

    frame: int
    track_id: int
    x: float
    y: float
    w: float
    h: float
    conf: float = 1.0

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)


@dataclass
class TrackingMetrics:
    """CLEAR MOT 跟踪指标 + 检测口径（红线同报）。"""

    mota: float
    idf1: float
    idsw: int
    mt: int
    ml: int
    fp: int
    fn: int
    num_gt: int
    num_pred: int
    idtp: int
    num_gt_tracks: int
    recall: float
    precision: float

    def summary_line(self) -> str:
        return (
            f"MOTA={self.mota:.4f}  IDF1={self.idf1:.4f}  IDSW={self.idsw}  "
            f"MT={self.mt}/{self.num_gt_tracks}  ML={self.ml}/{self.num_gt_tracks}  "
            f"FP={self.fp}  FN={self.fn}  "
            f"recall={self.recall:.4f}  precision={self.precision:.4f}"
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "mota": round(self.mota, 4), "idf1": round(self.idf1, 4),
            "idsw": float(self.idsw), "mt": float(self.mt), "ml": float(self.ml),
            "fp": float(self.fp), "fn": float(self.fn),
            "num_gt": float(self.num_gt), "num_pred": float(self.num_pred),
            "idtp": float(self.idtp), "num_gt_tracks": float(self.num_gt_tracks),
            "recall": round(self.recall, 4), "precision": round(self.precision, 4),
        }


def load_track_boxes(
    path: Path, keep_classes: set[int] | None = None,
) -> dict[int, list[TrackBox]]:
    """解析 MOT GT/预测 txt（frame,id,x,y,w,h,conf,class,visibility）→ {frame: [TrackBox]}。

    conf=0（MOT 约定非活跃标注）与 keep_classes 之外的类别跳过。
    """
    boxes: dict[int, list[TrackBox]] = {}
    for line in path.read_text().strip().splitlines():
        parts = line.split(",")
        if len(parts) < 8:
            continue
        frame = int(float(parts[0]))
        cls_id = int(float(parts[7]))
        conf = float(parts[6])
        if conf <= 0:
            continue
        if keep_classes is not None and cls_id not in keep_classes:
            continue
        x, y, w, h = (float(parts[2]), float(parts[3]),
                      float(parts[4]), float(parts[5]))
        boxes.setdefault(frame, []).append(
            TrackBox(frame=frame, track_id=int(float(parts[1])),
                     x=x, y=y, w=w, h=h, conf=conf))
    return boxes


def _iou(a: TrackBox, b: TrackBox) -> float:
    ax1, ay1, ax2, ay2 = a.xyxy
    bx1, by1, bx2, by2 = b.xyxy
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _match_frame(
    gt_boxes: list[TrackBox],
    pred_boxes: list[TrackBox],
    iou_threshold: float,
) -> tuple[list[tuple[int, int]], int, int]:
    """单帧贪心 IoU 匹配（按 IoU 降序取合法对）。

    Returns:
        (matched: [(gt_idx, pred_idx)], num_unmatched_gt, num_unmatched_pred)
    """
    if not gt_boxes or not pred_boxes:
        return [], len(gt_boxes), len(pred_boxes)

    import numpy as np

    ious = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, g in enumerate(gt_boxes):
        for j, p in enumerate(pred_boxes):
            ious[i, j] = _iou(g, p)
    order = np.unravel_index(np.argsort(-ious.ravel()), ious.shape)

    matched: list[tuple[int, int]] = []
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    for i, j in zip(*order):
        if ious[i, j] < iou_threshold:
            break  # 降序，后面全部不合格
        if i in used_gt or j in used_pred:
            continue
        matched.append((int(i), int(j)))
        used_gt.add(int(i))
        used_pred.add(int(j))
    return matched, len(gt_boxes) - len(matched), len(pred_boxes) - len(matched)


def evaluate_tracking(
    gt: dict[int, list[TrackBox]],
    pred: dict[int, list[TrackBox]],
    iou_threshold: float = 0.5,
) -> TrackingMetrics:
    """CLEAR MOT 评测：MOTA/IDF1/IDSW/MT/ML + 检测 recall/precision 同报。"""
    frames = sorted(set(gt) | set(pred))

    fn = fp = 0
    idsw = 0
    last_match: dict[int, int] = {}  # gt_id -> 上一匹配帧的 pred_id
    links: dict[tuple[int, int], int] = {}  # (gt_id, pred_id) -> 共现帧数
    gt_frames: dict[int, int] = {}  # gt_id -> 出现帧数
    gt_matched_frames: dict[int, int] = {}
    pred_frames: dict[int, int] = {}

    for frame in frames:
        gs = gt.get(frame, [])
        ps = pred.get(frame, [])
        matched, u_gt, u_pred = _match_frame(gs, ps, iou_threshold)
        fn += u_gt
        fp += u_pred

        for gi, pj in matched:
            g_id, p_id = gs[gi].track_id, ps[pj].track_id
            prev = last_match.get(g_id)
            if prev is not None and prev != p_id:
                idsw += 1
            last_match[g_id] = p_id
            links[(g_id, p_id)] = links.get((g_id, p_id), 0) + 1
            gt_matched_frames[g_id] = gt_matched_frames.get(g_id, 0) + 1
        for g in gs:
            gt_frames[g.track_id] = gt_frames.get(g.track_id, 0) + 1
        for p in ps:
            pred_frames[p.track_id] = pred_frames.get(p.track_id, 0) + 1

    num_gt = sum(gt_frames.values())
    num_pred = sum(pred_frames.values())

    # IDF1：全局二分匹配（权重 = 共现帧数）
    idtp = 0
    if links:
        from scipy.optimize import linear_sum_assignment

        gt_ids = sorted({g for g, _ in links})
        pred_ids = sorted({p for _, p in links})
        import numpy as np

        cost = np.zeros((len(gt_ids), len(pred_ids)))
        for (gid, pid), n in links.items():
            cost[gt_ids.index(gid), pred_ids.index(pid)] = -n
        rows, cols = linear_sum_assignment(cost)
        idtp = int(sum(-cost[r, c] for r, c in zip(rows, cols)))

    idfn = num_gt - idtp
    idfp = num_pred - idtp
    idf1 = 2 * idtp / (2 * idtp + idfn + idfp) if (2 * idtp + idfn + idfp) > 0 else 0.0

    # MT/ML：轨迹被匹配帧数占自身长度
    mt = sum(
        1 for gid, n in gt_frames.items()
        if gt_matched_frames.get(gid, 0) / n >= 0.8
    )
    ml = sum(
        1 for gid, n in gt_frames.items()
        if gt_matched_frames.get(gid, 0) / n <= 0.2
    )

    tp = num_gt - fn
    recall = tp / num_gt if num_gt else 0.0
    precision = tp / num_pred if num_pred else 0.0
    mota = 1 - (fn + fp + idsw) / num_gt if num_gt else 0.0

    return TrackingMetrics(
        mota=mota, idf1=idf1, idsw=idsw, mt=mt, ml=ml,
        fp=fp, fn=fn, num_gt=num_gt, num_pred=num_pred, idtp=idtp,
        num_gt_tracks=len(gt_frames), recall=recall, precision=precision,
    )
