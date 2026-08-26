"""3D 侧轻量多目标跟踪（v0.2 P2）：BEV IoU 匈牙利匹配 + 恒速 Kalman 速度估计。

纯 numpy/scipy（线性代数部分自写 4×4 矩阵运算，不引外部状态库）；
相机系 BEV 平面 (x, z) 上跟踪（坐标系红线：与 Box3D 一致，不混 velodyne 系）。

速度输出 = Kalman 恒速状态 vx/vz（观测差分平滑为 fallback）；
nuScenes 全局系使用时由调用方做 ego 补偿（P3）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from auto3dlabel.schema.box3d import Box3D
from auto3dlabel.tools.geometry import bev_iou

# 恒速 Kalman：state [x, z, vx, vz]；F=恒速转移（dt 由构造参数决定），H 观测位置
_H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
_Q = np.diag([0.05, 0.05, 0.5, 0.5])  # 过程噪声（v 分量大 → 允许机动、快速学速度）
_R = np.eye(2) * 0.25  # 观测噪声（中心点 ~0.5m 量级）


@dataclass
class Track3D:
    """单目标轨迹：track_id + 类别 + 框历史 + 速度历史（相机系 BEV）。"""

    track_id: int
    label: str
    history: list[Box3D] = field(default_factory=list)
    velocities: list[tuple[float, float]] = field(default_factory=list)  # (vx, vz)

    @property
    def age(self) -> int:
        return len(self.history)

    def append(self, box: Box3D, velocity: tuple[float, float] | None) -> None:
        self.history.append(box)
        if velocity is not None:
            self.velocities.append(velocity)


class Tracker3D:
    """BEV IoU 匈牙利跟踪器（恒速 Kalman 速度估计 + 差分平滑 fallback）。"""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 3,
        use_kalman: bool = True,
        dt: float = 0.1,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.use_kalman = use_kalman
        self.dt = dt
        self._next_id = 1
        self._tracks: dict[int, Track3D] = {}
        self._missed: dict[int, int] = {}  # track_id → 连续未匹配帧数
        self._kf: dict[int, tuple[np.ndarray, np.ndarray]] = {}  # (x, P) 恒速状态

    def tracks(self) -> list[Track3D]:
        """当前活跃轨迹（含连续未匹配但未超 max_age 的）。"""
        return list(self._tracks.values())

    def update(self, boxes: list[Box3D]) -> list[int]:
        """一帧更新；返回与 boxes 对齐的 track_id（新目标分配新 id）。

        匹配：BEV IoU 匈牙利（≤0 → 视为不匹配）；匹配 track 更新 Kalman 与历史；
        未匹配 track 记 missed，≥max_age 删除；未匹配 box 新建 track。
        """
        ids = sorted(self._tracks)
        n_t, n_b = len(ids), len(boxes)
        cost = np.full((n_t, n_b), 0.0)
        for i, tid in enumerate(ids):
            last = self._tracks[tid].history[-1]
            for j, box in enumerate(boxes):
                iou = bev_iou(last, box)
                cost[i, j] = 1.0 - iou if iou > 0 else 1.0

        matched: dict[int, int] = {}  # box_idx → track_id
        if n_t and n_b:
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols, strict=True):
                if 1.0 - cost[i, j] >= self.iou_threshold:
                    matched[int(j)] = ids[i]

        used_boxes: set[int] = set()
        for j, tid in matched.items():
            box = boxes[j]
            used_boxes.add(j)
            velocity = self._observe(tid, box)
            self._tracks[tid].append(box, velocity)
            self._missed[tid] = 0

        for tid in ids:
            if tid not in matched.values():
                self._missed[tid] = self._missed.get(tid, 0) + 1
                if self._missed[tid] > self.max_age:
                    del self._tracks[tid]
                    self._kf.pop(tid, None)
                    self._missed.pop(tid, None)

        track_ids: list[int] = []
        for j, box in enumerate(boxes):
            if j in matched:
                track_ids.append(matched[j])
                continue
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = Track3D(track_id=tid, label=box.label, history=[box])
            self._missed[tid] = 0
            if self.use_kalman:
                x0 = np.array([box.cx, box.cz, 0.0, 0.0])
                self._kf[tid] = (x0, np.eye(4))
            track_ids.append(tid)
        return track_ids

    def _observe(self, tid: int, box: Box3D) -> tuple[float, float] | None:
        """位置观测 → Kalman 更新；返回速度 (vx, vz)（首帧观测无速度 → None）。

        差分平滑 fallback（use_kalman=False）：v = 0.5·v_prev + 0.5·Δx/dt。
        """
        history = self._tracks[tid].history
        if not history:
            return None
        if self.use_kalman:
            if len(history) == 1:
                # 第 2 帧观测：差分初始化速度（避免 Kalman 从 0 慢爬），再进入滤波
                prev = history[-1]
                vx = (box.cx - prev.cx) / self.dt
                vz = (box.cz - prev.cz) / self.dt
                self._kf[tid] = (np.array([box.cx, box.cz, vx, vz]), np.eye(4))
                return vx, vz
            dt = self.dt
            f = np.array(
                [
                    [1.0, 0.0, dt, 0.0],
                    [0.0, 1.0, 0.0, dt],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            x, P = self._kf[tid]
            x = f @ x
            P = f @ P @ f.T + _Q
            z = np.array([box.cx, box.cz])
            y = z - _H @ x
            S = _H @ P @ _H.T + _R
            K = P @ _H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (np.eye(4) - K @ _H) @ P
            self._kf[tid] = (x, P)
            return float(x[2]), float(x[3])
        prev = history[-1]
        vx = (box.cx - prev.cx) / self.dt
        vz = (box.cz - prev.cz) / self.dt
        if self._tracks[tid].velocities:
            old_vx, old_vz = self._tracks[tid].velocities[-1]
            vx, vz = 0.5 * old_vx + 0.5 * vx, 0.5 * old_vz + 0.5 * vz
        return vx, vz
