"""跟踪模型层 —— 跟踪算法本体（ByteTrack / BoT-SORT）+ ReID 特征模型。

算法本体（tracking-by-detection 纯后处理，检测器无关；numpy + scipy，
零权重依赖、轻量、实时）：
- ByteTrack (ECCV 2022)：8 维卡尔曼滤波（cx, cy, a, h 及各自速度）恒定速度模型
  + BYTE 两段关联（高分框先匹配 IoU 0.8，低分框与剩余轨迹二次匹配 IoU 0.5——
  低分框不丢弃，直接缓解 mot 域差距（v0.3 实测 recall 0.1 量级）下
  「正确检测多为低分框」的问题）
- BoT-SORT (arXiv:2206.14651) 精度档：第一段（高分）关联 cost =
  λ·(1−cos) + (1−λ)·(1−IoU)（λ=0.98 外观主导），cosine 门控拒绝外观不一致的
  匹配，任一侧缺特征自动回退纯 IoU；轨迹外观特征 EMA 维护（α=0.9，L2 重归一化）；
  ECC 相机运动补偿（轨迹预测框按帧间相机运动 warp 后参与第一段 IoU，
  下采样估参 + 合理性三查，异常回退恒等）

与 ByteTrack 原版差异（2026-08-19 立项讨论定案，标注场景取舍）：
- 未匹配高分检测**立即**创建轨迹并输出 ID（原版需 min_hits=3 帧激活才输出）——
  标注工具「不丢检测」优先，短轨迹也拿到 ID

ReID 特征模型（BoT-SORT 外观关联用）：CLIP / SigLIP 图像编码器
（transformers 懒加载，与 classification.py 同模式：HF_HOME=weights/hf、
构造零加载），只取图像侧特征 get_image_features（无文本编码），输出
L2 归一化向量。特征由调用方经 extract_frame_features 预提取后按帧注入
tracker（update 的 features 参数，与 bboxes 逐索引对齐）。

用法:
    tracker = ByteTracker()
    for bboxes in per_frame_dets:   # 帧序输入
        tracker.update(bboxes)      # 就地为每个 Bbox 写 track_id（None=未关联）
    tracker.trajectory(track_id)    # 轨迹中心点序列（可视化）
    tracker.velocity(track_id)      # (vx, vy) 像素/帧（Auto3dLabel v0.2 运动属性）
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import scipy.linalg

from auto2dlabel.models import Image, os
from auto2dlabel.models.model_catalog import REID_MODELS, WEIGHTS_DIR
from auto2dlabel.schema.annotation import Bbox
from auto2dlabel.tools.device import get_device

# ============================================================
# 跟踪算法本体：卡尔曼滤波 + 轨迹 + 关联（检测器无关，零权重）
# ============================================================

class KalmanFilter:
    """8 维恒定速度卡尔曼滤波器（cx, cy, a, h 及各自速度）。

    与 ByteTrack 官方实现同参：位置噪声权重 1/20、速度噪声权重 1/160。
    """

    def __init__(self) -> None:
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(
        self, measurement: np.ndarray[Any, Any],
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """从首次观测初始化状态均值与协方差。"""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        return mean, np.diag(np.square(std))

    def predict(
        self, mean: np.ndarray[Any, Any], covariance: np.ndarray[Any, Any],
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """恒定速度运动模型预测。"""
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(mean, self._motion_mat.T)
        covariance = (
            np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T))
            + motion_cov
        )
        return mean, covariance

    def project(
        self, mean: np.ndarray[Any, Any], covariance: np.ndarray[Any, Any],
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """状态空间 → 观测空间投影。"""
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot(
            (self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(
        self,
        mean: np.ndarray[Any, Any],
        covariance: np.ndarray[Any, Any],
        measurement: np.ndarray[Any, Any],
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """卡尔曼更新（Cholesky 解算增益）。"""
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False,
        ).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance


class Tracklet:
    """单条轨迹：卡尔曼状态 + 生命周期 + 轨迹历史（中心点折线）。"""

    def __init__(self, track_id: int, bbox: Bbox, kf: KalmanFilter | None = None) -> None:
        self.track_id = track_id
        self.label = bbox.label  # 首帧类别（轨迹身份参考）
        self.kf = kf or KalmanFilter()
        measurement = self._measure(bbox)
        self.mean, self.covariance = self.kf.initiate(measurement)
        self.hits = 0  # 累计命中帧数
        self.time_since_update = 0  # 连续丢失帧数（> track_buffer 删除）
        self._centers: list[tuple[float, float]] = []
        # L2 归一化 ReID 外观特征（EMA 维护；None=外观关联未启用）。
        # 只存 Tracklet、不进 Bbox —— track_id 穿透点红线
        self.feature: np.ndarray[Any, Any] | None = None

    @staticmethod
    def _measure(bbox: Bbox) -> np.ndarray[Any, Any]:
        """[cx, cy, a, h] 观测向量（ByteTrack 约定）。"""
        cx = bbox.x + bbox.width / 2
        cy = bbox.y + bbox.height / 2
        a = bbox.width / max(bbox.height, 1e-6)
        return np.array([cx, cy, a, bbox.height], dtype=np.float64)

    def predict(self) -> None:
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)

    def update(self, bbox: Bbox) -> None:
        """命中：卡尔曼更新 + 生命周期刷新 + 轨迹点追加。"""
        self.mean, self.covariance = self.kf.update(
            self.mean, self.covariance, self._measure(bbox))
        self.hits += 1
        self.time_since_update = 0
        center = (float(self.mean[0]), float(self.mean[1]))
        if len(self._centers) >= 60:
            self._centers.pop(0)
        self._centers.append(center)

    def mark_missed(self) -> None:
        self.time_since_update += 1

    def update_feature(
        self, feature: np.ndarray[Any, Any], alpha: float = 0.9,
    ) -> None:
        """命中时外观特征更新：首次赋值（拷贝），此后 EMA + L2 重归一化。"""
        if self.feature is None:
            self.feature = feature.copy()
        else:
            self.feature = ema_feature(self.feature, feature, alpha)

    def state(self) -> tuple[float, float, float, float]:
        """当前状态框 [x1, y1, x2, y2]（IoU 用）。"""
        cx, cy, a, h = self.mean[:4]
        w = a * h
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def velocity(self) -> tuple[float, float]:
        """(vx, vy) 像素/帧，卡尔曼状态直接读取（Auto3dLabel v0.2 运动属性）。"""
        return (float(self.mean[4]), float(self.mean[5]))

    def trajectory(self) -> list[tuple[float, float]]:
        """轨迹中心点历史（帧序），可视化轨迹线用。"""
        return list(self._centers)


def _iou_matrix(
    dets: list[Bbox],
    tracklets: list[Tracklet],
    warp: np.ndarray[Any, Any] | None = None,
) -> np.ndarray[Any, Any]:
    """检测 × 轨迹 IoU 矩阵（轨迹取预测后状态框，可选经仿射 warp 补偿相机运动）。

    warp 为 None 时与 ByteTrack 语义逐位一致。
    """
    det_boxes = np.array([d.xyxy for d in dets], dtype=np.float64)
    tr_boxes = np.array([t.state() for t in tracklets], dtype=np.float64)
    if warp is not None:
        tr_boxes = _warp_boxes(tr_boxes, warp)
    x1 = np.maximum(det_boxes[:, None, 0], tr_boxes[None, :, 0])
    y1 = np.maximum(det_boxes[:, None, 1], tr_boxes[None, :, 1])
    x2 = np.minimum(det_boxes[:, None, 2], tr_boxes[None, :, 2])
    y2 = np.minimum(det_boxes[:, None, 3], tr_boxes[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_d = (det_boxes[:, 2] - det_boxes[:, 0]) * (det_boxes[:, 3] - det_boxes[:, 1])
    area_t = (tr_boxes[:, 2] - tr_boxes[:, 0]) * (tr_boxes[:, 3] - tr_boxes[:, 1])
    union = area_d[:, None] + area_t[None, :] - inter
    out: np.ndarray[Any, Any] = inter / np.maximum(union, 1e-6)
    return out


def _associate(
    dets: list[Bbox], tracklets: list[Tracklet], iou_threshold: float,
) -> tuple[list[tuple[Bbox, Tracklet]], list[Bbox], list[Tracklet]]:
    """IoU 门控 + 匈牙利匹配（cost = 1 - IoU，低于阈值置 1 不可匹配）。

    Returns:
        (匹配对, 未匹配检测, 未匹配轨迹)。
    """
    if not dets or not tracklets:
        return [], list(dets), list(tracklets)

    iou = _iou_matrix(dets, tracklets)
    cost = np.where(iou >= iou_threshold, 1.0 - iou, 1.0)

    from scipy.optimize import linear_sum_assignment

    row, col = linear_sum_assignment(cost)
    matches: list[tuple[Bbox, Tracklet]] = []
    matched_det_idx: set[int] = set()
    matched_tr_idx: set[int] = set()
    for i, j in zip(row, col):
        if cost[i, j] < 1.0:
            matches.append((dets[i], tracklets[j]))
            matched_det_idx.add(i)
            matched_tr_idx.add(j)
    unmatched_dets = [d for k, d in enumerate(dets) if k not in matched_det_idx]
    unmatched_tr = [t for k, t in enumerate(tracklets) if k not in matched_tr_idx]
    return matches, unmatched_dets, unmatched_tr


# ============================================================
# BoT-SORT 纯函数组（EMA / 仿射 / 相似度 / 融合关联 / ECC）
# ============================================================

def ema_feature(
    prev: np.ndarray[Any, Any], new: np.ndarray[Any, Any], alpha: float,
) -> np.ndarray[Any, Any]:
    """外观特征 EMA（BoT-SORT：f = α·f + (1−α)·f_new）+ L2 重归一化。

    零向量结果原样返回（防除零，退化可接受——后续仍可被新特征覆盖）。
    """
    merged = alpha * prev + (1.0 - alpha) * new
    norm = float(np.linalg.norm(merged))
    return merged if norm < 1e-12 else merged / norm


def _warp_boxes(
    boxes: np.ndarray[Any, Any], warp: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """(n,4) x1y1x2y2 经 2×3 仿射矩阵 warp（tl/br 两点齐次变换，官方同款）。"""
    pts = np.concatenate([boxes[:, :2], boxes[:, 2:]], axis=0)  # (2n, 2)
    ones = np.ones((pts.shape[0], 1), dtype=pts.dtype)
    warped = np.concatenate([pts, ones], axis=1) @ warp.T  # (2n, 2)
    stacked: np.ndarray[Any, Any] = np.stack(
        [warped[: len(boxes)], warped[len(boxes):]], axis=1)  # (n, 2, 2) tl/br
    return stacked.reshape(-1, 4)


def _similarity_matrix(
    det_feats: list[np.ndarray[Any, Any] | None], tracklets: list[Tracklet],
) -> np.ndarray[Any, Any]:
    """检测 × 轨迹余弦相似度；任一侧缺特征 → NaN（该对走纯 IoU 通道）。"""
    n_d, n_t = len(det_feats), len(tracklets)
    out = np.full((n_d, n_t), np.nan, dtype=np.float64)
    d_idx = [i for i, f in enumerate(det_feats) if f is not None]
    t_idx = [i for i, t in enumerate(tracklets) if t.feature is not None]
    if not d_idx or not t_idx:
        return out
    d_rows: list[np.ndarray[Any, Any]] = []
    for i in d_idx:
        f = det_feats[i]
        assert f is not None  # d_idx 已过滤 None
        d_rows.append(f)
    t_rows: list[np.ndarray[Any, Any]] = []
    for i in t_idx:
        f = tracklets[i].feature
        assert f is not None  # t_idx 已过滤 None
        t_rows.append(f)
    d_mat = np.stack(d_rows)  # (nd, D)
    t_mat = np.stack(t_rows)  # (nt, D)
    out[np.ix_(d_idx, t_idx)] = d_mat @ t_mat.T
    return out


def _associate_high(
    dets: list[Bbox],
    det_feats: list[np.ndarray[Any, Any] | None],
    tracklets: list[Tracklet],
    warp: np.ndarray[Any, Any] | None,
    match_thresh: float,
    lambda_: float,
    appearance_thresh: float,
) -> tuple[
    list[tuple[Bbox, np.ndarray[Any, Any] | None, Tracklet]],
    list[tuple[Bbox, np.ndarray[Any, Any] | None]],
    list[Tracklet],
]:
    """BoT-SORT 第一段（高分）关联：cost = λ·(1−cos) + (1−λ)·(1−IoU)。

    通道语义：
    - 双侧都有特征且 cosine ≥ appearance_thresh、IoU ≥ match_thresh → 融合 cost
    - 任一侧缺特征且 IoU ≥ match_thresh → 纯 IoU cost（回退 ByteTrack 语义）
    - 其余（IoU 不足或 cosine 门控拒绝）→ cost=1.0 不可匹配
    IoU 计算在 warp 补偿后的轨迹预测框上进行（warp=None 即无补偿）。

    Returns:
        (匹配三元组(检测, 特征, 轨迹), 未匹配检测(带特征), 未匹配轨迹)。
    """
    if not dets or not tracklets:
        return [], list(zip(dets, det_feats)), list(tracklets)

    iou = _iou_matrix(dets, tracklets, warp)
    sim = _similarity_matrix(det_feats, tracklets)
    cost = np.full_like(iou, 1.0)
    iou_ok = iou >= match_thresh
    reid_ok = ~np.isnan(sim) & (sim >= appearance_thresh)
    cost[reid_ok] = lambda_ * (1.0 - sim[reid_ok]) + (1.0 - lambda_) * (1.0 - iou[reid_ok])
    io_only = iou_ok & np.isnan(sim)
    cost[io_only] = 1.0 - iou[io_only]

    from scipy.optimize import linear_sum_assignment

    row, col = linear_sum_assignment(cost)
    matches: list[tuple[Bbox, np.ndarray[Any, Any] | None, Tracklet]] = []
    matched_det_idx: set[int] = set()
    matched_tr_idx: set[int] = set()
    for i, j in zip(row, col):
        if cost[i, j] < 1.0:
            matches.append((dets[i], det_feats[i], tracklets[j]))
            matched_det_idx.add(i)
            matched_tr_idx.add(j)
    unmatched_dets = [
        (d, det_feats[k]) for k, d in enumerate(dets) if k not in matched_det_idx
    ]
    unmatched_tr = [t for k, t in enumerate(tracklets) if k not in matched_tr_idx]
    return matches, unmatched_dets, unmatched_tr


def _estimate_ecc(
    prev_gray: np.ndarray[Any, Any], curr_gray: np.ndarray[Any, Any], mode: str,
) -> np.ndarray[Any, Any] | None:
    """相邻帧 ECC 全局运动估计 → 2×3 warp（prev 坐标 → curr 坐标）。

    仅支持平移/欧氏/仿射三种模式；输入退化（恒定图等 cv2 报错）返回 None。
    """
    import cv2

    warp_modes = {
        "translation": cv2.MOTION_TRANSLATION,
        "euclidean": cv2.MOTION_EUCLIDEAN,
        "affine": cv2.MOTION_AFFINE,
    }
    if mode not in warp_modes:
        raise ValueError(f"未知 ECC 模式: {mode}（可选: {', '.join(warp_modes)}）")
    warp_init = np.eye(2, 3, dtype=np.float32)
    try:
        # motionType 用位置参数（OpenCV 4.x 关键字名，跨版本兼容）
        _, warp_est = cv2.findTransformECC(
            prev_gray, curr_gray, warp_init, warp_modes[mode],
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4),
        )
    except cv2.error:
        return None  # 恒定图/无梯度退化输入
    return np.asarray(warp_est, dtype=np.float64)


def _rescale_warp(
    warp: np.ndarray[Any, Any], sx: float, sy: float,
) -> np.ndarray[Any, Any]:
    """下采样空间 warp → 原图像素空间 warp（x 缩放 sx、y 缩放 sy）。

    推导（x_s = D⁻¹·x，D=diag(sx,sy)）：x' = D·R·D⁻¹·x + D·t，
    即 R00/R11 不变、R01×sx/sy、R10×sy/sx、t 逐分量×缩放。
    """
    r = warp[:, :2].copy()
    t = warp[:, 2].copy()
    r[0, 1] *= sx / sy
    r[1, 0] *= sy / sx
    t[0] *= sx
    t[1] *= sy
    return np.concatenate([r, t.reshape(2, 1)], axis=1)


def _warp_sane(warp: np.ndarray[Any, Any], width: int, height: int) -> bool:
    """ECC warp 合理性三查（防局部最优/病态解污染关联）：
    线性部缩放异常（|det−1|>0.15）/ 条件数过大 / 平移超过图像短边 30%。
    """
    r = warp[:, :2]
    t = warp[:, 2]
    if abs(float(np.linalg.det(r)) - 1.0) > 0.15:
        return False
    if float(np.linalg.cond(r)) > 1e3:
        return False
    if math.hypot(float(t[0]), float(t[1])) > 0.3 * min(width, height):
        return False
    return True


class ByteTracker:
    """ByteTrack 后处理器主类。

    Args:
        match_thresh: 第一段（高分）关联 IoU 阈值。
        match_low_thresh: 第二段（低分救援）关联 IoU 阈值。
        track_high_thresh: 高分检测池置信度下限。
        track_low_thresh: 参与低分救援的置信度下限（低于此值的检测直接丢弃）。
        new_track_thresh: 未匹配检测创建新轨迹的置信度下限。
        track_buffer: 连续丢失多少帧后删除轨迹。
    """

    def __init__(
        self,
        match_thresh: float = 0.8,
        match_low_thresh: float = 0.5,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.5,
        track_buffer: int = 30,
    ) -> None:
        self.match_thresh = match_thresh
        self.match_low_thresh = match_low_thresh
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.track_buffer = track_buffer
        self._tracklets: list[Tracklet] = []
        self._next_id = 0

    def update(self, bboxes: Sequence[Bbox]) -> list[Bbox]:
        """处理一帧检测：BYTE 两段关联，就地为每个 bbox 写 track_id。

        返回输入顺序的 bbox 列表（同对象引用）；未关联（低分未命中）
        的 bbox track_id 保持 None。
        """
        dets = list(bboxes)
        for tr in self._tracklets:
            tr.predict()

        high = [d for d in dets if d.confidence >= self.track_high_thresh]
        low = [
            d for d in dets
            if self.track_low_thresh <= d.confidence < self.track_high_thresh
        ]

        # 第一段：高分 vs 全部轨迹
        matches, unmatched_high, unmatched_tr = _associate(
            high, self._tracklets, self.match_thresh)
        # 第二段：低分 vs 剩余轨迹（救援失联轨迹）
        matches2, _, unmatched_tr2 = _associate(
            low, unmatched_tr, self.match_low_thresh)

        for det, tr in matches + matches2:
            tr.update(det)
            det.track_id = tr.track_id
        for tr in unmatched_tr2:
            tr.mark_missed()
        for det in unmatched_high:
            if det.confidence >= self.new_track_thresh:
                self._new_track(det)

        self._tracklets = [
            tr for tr in self._tracklets if tr.time_since_update <= self.track_buffer
        ]
        return dets

    def _new_track(self, bbox: Bbox) -> None:
        """为未匹配高分检测创建新轨迹并立即赋 ID（标注不丢检测优先）。"""
        tracklet = Tracklet(self._next_id, bbox)
        self._next_id += 1
        tracklet.hits = 1
        tracklet.update(bbox)
        bbox.track_id = tracklet.track_id
        self._tracklets.append(tracklet)

    def tracks(self) -> list[Tracklet]:
        """当前存活轨迹（含丢失未超时的）。"""
        return list(self._tracklets)

    def trajectory(self, track_id: int) -> list[tuple[float, float]] | None:
        """按 ID 查轨迹中心点历史；未知 ID 返回 None。"""
        for tr in self._tracklets:
            if tr.track_id == track_id:
                return tr.trajectory()
        return None

    def velocity(self, track_id: int) -> tuple[float, float] | None:
        """按 ID 查 (vx, vy) 像素/帧速度；未知 ID 返回 None。"""
        for tr in self._tracklets:
            if tr.track_id == track_id:
                return tr.velocity()
        return None

    @property
    def active_track_ids(self) -> set[int]:
        """本帧命中的轨迹 ID 集合。"""
        return {tr.track_id for tr in self._tracklets if tr.time_since_update == 0}


class BotSORTTracker(ByteTracker):
    """BoT-SORT 精度档（ByteTrack + ReID 外观关联 + ECC 相机运动补偿）。

    与 ByteTrack 差异（论文: BoT-SORT, arXiv:2206.14651）：
    - 第一段（高分）关联 cost = λ·(1−cos) + (1−λ)·(1−IoU)（λ=0.98 外观主导），
      cosine 门控（appearance_thresh）拒绝外观不一致的匹配；
      任一侧缺特征自动回退纯 IoU（ByteTrack 语义）
    - 轨迹外观特征 EMA 维护（alpha=0.9）；低分救援不更新特征
    - ECC 全局运动补偿：轨迹预测框按帧间相机运动 warp 后参与第一段 IoU
      （下采样估参 + 合理性三查，异常回退恒等）

    特征由调用方预提取注入（update 的 features 参数，与 bboxes 逐索引对齐）；
    image 参数仅 ECC 用（BGR ndarray）。两者均可为 None——
    features=None 且 use_cmc=False 时与 ByteTracker 逐位等价（单测锚点）。
    """

    def __init__(
        self,
        match_thresh: float = 0.8,
        match_low_thresh: float = 0.5,
        track_high_thresh: float = 0.6,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.7,
        track_buffer: int = 30,
        lambda_: float = 0.98,
        appearance_thresh: float = 0.25,
        ema_alpha: float = 0.9,
        use_cmc: bool = True,
        cmc_model: str = "euclidean",
        cmc_max_side: int = 640,
    ) -> None:
        super().__init__(
            match_thresh, match_low_thresh, track_high_thresh,
            track_low_thresh, new_track_thresh, track_buffer,
        )
        self.lambda_ = lambda_
        self.appearance_thresh = appearance_thresh
        self.ema_alpha = ema_alpha
        self.use_cmc = use_cmc
        self.cmc_model = cmc_model
        self.cmc_max_side = cmc_max_side
        self._prev_gray: np.ndarray[Any, Any] | None = None

    def update(
        self,
        bboxes: Sequence[Bbox],
        image: np.ndarray[Any, Any] | None = None,
        features: Sequence[np.ndarray[Any, Any] | None] | None = None,
    ) -> list[Bbox]:
        """处理一帧检测（BoT-SORT）：ECC 补偿 + ReID 融合关联 + 特征 EMA。

        Args:
            bboxes: 本帧检测（就地为每个 bbox 写 track_id，None=未关联）。
            image: BGR ndarray（cv2 约定），仅 ECC 用；None=跳过本帧补偿。
            features: 与 bboxes 逐索引对齐的 L2 归一化特征（未提取项 None）。
        """
        dets = list(bboxes)
        feats_aligned: list[np.ndarray[Any, Any] | None]
        if features is None:
            feats_aligned = [None] * len(dets)
        else:
            feats_aligned = list(features)

        warp = self._cmc_estimate(image)
        for tr in self._tracklets:
            tr.predict()

        high: list[Bbox] = []
        high_feats: list[np.ndarray[Any, Any] | None] = []
        low: list[Bbox] = []
        for det, feat in zip(dets, feats_aligned):
            if det.confidence >= self.track_high_thresh:
                high.append(det)
                high_feats.append(feat)
            elif det.confidence >= self.track_low_thresh:
                low.append(det)

        # 第一段：高分融合关联（ReID 门控 + warp 补偿 IoU）
        matches, unmatched_high, unmatched_tr = _associate_high(
            high, high_feats, self._tracklets, warp,
            self.match_thresh, self.lambda_, self.appearance_thresh)
        # 第二段：低分 vs 剩余轨迹，纯 IoU（低分框无特征语义，救援不更新特征）
        matches2, _, unmatched_tr2 = _associate(
            low, unmatched_tr, self.match_low_thresh)

        for det, feat, tr in matches:
            tr.update(det)
            det.track_id = tr.track_id
            if feat is not None:
                tr.update_feature(feat, self.ema_alpha)
        for det, tr in matches2:
            tr.update(det)
            det.track_id = tr.track_id
        for tr in unmatched_tr2:
            tr.mark_missed()
        for det, feat in unmatched_high:
            if det.confidence >= self.new_track_thresh:
                self._new_track(det, feat)

        self._tracklets = [
            tr for tr in self._tracklets if tr.time_since_update <= self.track_buffer
        ]
        return dets

    def _new_track(
        self, bbox: Bbox, feature: np.ndarray[Any, Any] | None = None,
    ) -> None:
        """与 ByteTrack 同体，追加首帧外观特征初始化。"""
        tracklet = Tracklet(self._next_id, bbox)
        self._next_id += 1
        tracklet.hits = 1
        tracklet.update(bbox)
        if feature is not None:
            tracklet.update_feature(feature)
        bbox.track_id = tracklet.track_id
        self._tracklets.append(tracklet)

    def _cmc_estimate(
        self, image: np.ndarray[Any, Any] | None,
    ) -> np.ndarray[Any, Any] | None:
        """本帧 vs 上帧 ECC → 2×3 warp（原图像素空间）；首帧/异常/关断 → None。

        下采样至 cmc_max_side 估参（CPU 可行），回原图空间后过合理性三查；
        全部退化路径返回 None = 无补偿（恒等）。
        """
        if not self.use_cmc or image is None:
            return None
        import cv2

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        gray_small = gray
        sx = sy = 1.0
        scale = min(1.0, self.cmc_max_side / max(w, h))
        if scale < 1.0:
            rw = max(1, int(round(w * scale)))
            rh = max(1, int(round(h * scale)))
            gray_small = cv2.resize(gray, (rw, rh), interpolation=cv2.INTER_AREA)
            sx, sy = w / rw, h / rh
        if self._prev_gray is None:
            self._prev_gray = gray_small
            return None
        warp_small = _estimate_ecc(self._prev_gray, gray_small, self.cmc_model)
        self._prev_gray = gray_small
        if warp_small is None:
            return None
        warp = _rescale_warp(warp_small, sx, sy)
        return warp if _warp_sane(warp, w, h) else None


# ============================================================
# ReID 特征模型（BoT-SORT 外观关联用，CLIP / SigLIP 图像编码器）
# ============================================================

class ReIDModel(Protocol):
    """ReID 特征提取接口（Protocol，允许 duck typing 注入假模型）。"""

    def extract(
        self, images: Sequence[Image.Image],
    ) -> list[np.ndarray[Any, Any]]:
        """PIL 裁剪图列表 → L2 归一化特征向量列表（与输入顺序对齐）。"""
        ...


def _image_embeds(output: Any) -> Any:
    """`get_image_features` 输出兼容提取（transformers <5 返回 Tensor，5.x 返回输出对象）。"""
    import torch

    if isinstance(output, torch.Tensor):
        return output
    pooled = getattr(output, "pooler_output", None)  # CLIP: BaseModelOutputWithPooling
    if pooled is not None:
        return pooled
    return output.image_embeds  # SigLIP: SiglipImageModelOutput


def l2_normalize(features: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """行级 L2 归一化（零向量行原样保留，不抛异常——纯函数单测锚点）。"""
    norms: np.ndarray[Any, Any] = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0  # 零向量行防除零（结果仍为零向量）
    out: np.ndarray[Any, Any] = features / norms  # __truediv__ stub 返回 Any
    return out


def crop_bbox(image: Image.Image, bbox: Bbox, pad: int = 0) -> Image.Image:
    """按 bbox 裁剪子图（越界 clamp + floor/ceil 取整；全越界退化 1×1）。

    PIL Image.crop 要求整数坐标且左闭右开；浮点 xyxy 先扩展 pad 再夹取。
    """
    w, h = image.size
    x1, y1, x2, y2 = bbox.xyxy
    x1 = max(0, math.floor(x1 - pad))
    y1 = max(0, math.floor(y1 - pad))
    x2 = min(w, math.ceil(x2 + pad))
    y2 = min(h, math.ceil(y2 + pad))
    if x2 <= x1 or y2 <= y1:
        # 全越界：退化为边界 1×1（crop 不抛异常）
        x1, y1 = min(max(x1, 0), w - 1), min(max(y1, 0), h - 1)
        x2, y2 = x1 + 1, y1 + 1
    return image.crop((int(x1), int(y1), int(x2), int(y2)))


def extract_frame_features(
    model: ReIDModel,
    image: Image.Image,
    bboxes: Sequence[Bbox],
    min_conf: float = 0.0,
    pad: int = 0,
) -> list[np.ndarray[Any, Any] | None]:
    """单帧 bbox → 特征（与 bboxes 逐索引对齐；conf < min_conf 项为 None）。

    调用方传 min_conf=tracker.track_high_thresh：仅高分段参与外观关联
    （BoT-SORT 官方语义），并省去低分框的推理量。
    """
    if not bboxes:
        return []
    idxs = [i for i, b in enumerate(bboxes) if b.confidence >= min_conf]
    feats = model.extract([crop_bbox(image, bboxes[i], pad=pad) for i in idxs])
    out: list[np.ndarray[Any, Any] | None] = [None] * len(bboxes)
    for i, feat in zip(idxs, feats):
        out[i] = feat
    return out


class ClipReIDModel:
    """CLIP 图像编码器 ReID 特征（openai/clip-vit-base-patch32）。

    复用分类层的 CLIP 懒加载模式；权重下载到 auto2dlabel/weights/hf/
    （HF_HOME 统一管理，与 ClipModel 共享同一份缓存）。
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device or get_device()
        self._model: Any = None  # transformers 为可选依赖，类型按 Any 处理
        self._processor: Any = None

    def _load(self) -> Any:
        """加载模型与处理器（幂等）。transformers 为可选依赖，类型按 Any 处理。"""
        if self._model is not None:
            return self._model
        os.environ.setdefault("HF_HOME", str(WEIGHTS_DIR / "hf"))
        try:
            from transformers import CLIPModel, CLIPProcessor  # pyright: ignore[reportMissingImports]  # isort: skip
        except ImportError:
            raise ImportError("transformers 未安装，请运行: pip install transformers")

        self._model = CLIPModel.from_pretrained(self._model_name)
        self._processor = CLIPProcessor.from_pretrained(self._model_name)
        self._model.to(self._device)
        self._model.eval()
        return self._model

    def extract(
        self, images: Sequence[Image.Image],
    ) -> list[np.ndarray[Any, Any]]:
        """PIL 裁剪图列表 → L2 归一化图像特征列表（与输入顺序对齐）。"""
        if not images:
            return []
        model = self._load()
        processor = self._processor
        assert processor is not None  # _load 已初始化 processor

        import torch

        inputs = processor(images=list(images), return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = _image_embeds(model.get_image_features(**inputs))
        return list(l2_normalize(feats.detach().cpu().numpy()))


class SigLIPReIDModel:
    """SigLIP 图像编码器 ReID 特征（google/siglip-base-patch16-224）。

    与 ClipReIDModel 同构（get_image_features 输出 L2 归一化向量）。
    """

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device or get_device()
        self._model: Any = None  # transformers 为可选依赖，类型按 Any 处理
        self._processor: Any = None

    def _load(self) -> Any:
        """加载模型与处理器（幂等）。transformers 为可选依赖，类型按 Any 处理。"""
        if self._model is not None:
            return self._model
        os.environ.setdefault("HF_HOME", str(WEIGHTS_DIR / "hf"))
        try:
            from transformers import AutoProcessor, SiglipModel  # pyright: ignore[reportMissingImports]  # isort: skip
        except ImportError:
            raise ImportError("transformers 未安装，请运行: pip install transformers")

        self._model = SiglipModel.from_pretrained(self._model_name)
        self._processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            self._model_name)
        self._model.to(self._device)
        self._model.eval()
        return self._model

    def extract(
        self, images: Sequence[Image.Image],
    ) -> list[np.ndarray[Any, Any]]:
        """PIL 裁剪图列表 → L2 归一化图像特征列表（与输入顺序对齐）。"""
        if not images:
            return []
        model = self._load()
        processor = self._processor
        assert processor is not None  # _load 已初始化 processor

        import torch

        inputs = processor(images=list(images), return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = _image_embeds(model.get_image_features(**inputs))
        return list(l2_normalize(feats.detach().cpu().numpy()))


def create_reid_model(
    model_name: str = "openai/clip-vit-base-patch32",
) -> ReIDModel:
    """工厂函数：按模型名路由（siglip / clip，大小写不敏感）。"""
    lower = model_name.lower()
    if "siglip" in lower:
        return SigLIPReIDModel(model_name=model_name)
    if "clip" in lower:
        return ClipReIDModel(model_name=model_name)
    raise ValueError(
        f"无法识别的 ReID 模型: '{model_name}'。\n"
        f"可用: {', '.join(REID_MODELS)}"
    )
