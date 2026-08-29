"""3D bbox 数据模型（穿透点纪律：yaw/尺寸字段每增消费点必在此登记，漏一处静默丢角）。

穿透点清单：
1. Box3D.to_dict（Web 队列 / agent 工具返回值 / benchmark 中间格式）
2. export/kitti_label.line_from_box3d（rotation_y 唯一转换点在此消费）
3. benchmarks/load_gt3d + kitti3d_benchmark（GT/pred 7 值 dict 化）
4. web/server.save_review（3D 字段直通保留）
5. export/review_queue.triage_3d（confidence + fit_points 双键分流）
6. tools/fit.fit_box3d（单一产出源）
7. track_id（v0.2 P2）：tools/track3d.Tracker3D.update 回写 + to_dict 非 None 输出
   + cli --track3d 序列导出（KITTI label 15 字段不含，nuScenes instance_token 走 P3）
8. web/payloads.frame_payload（v0.3 P4）：corners_cam 8x3 相机系角点 + load_points_bin
   点云 → 四视图渲染数据（前端零 calib 依赖，yaw 数学留在 Python 侧）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from auto3dlabel.schema.calib import KittiCalib
from auto3dlabel.tools import geometry


@dataclass
class Box3D:
    """相机系（rectified cam0：x 右 y 下 z 前）3D 框。

    角度内部唯一表示 yaw_bev：车头方向相对 +z 轴、向 +x 为正，[-π, π]；
    KITTI rotation_y 仅在导出/评测 dict 化时经 property 转换（红线：不得混用）。
    """

    label: str
    confidence: float = 1.0
    cx: float = 0.0
    cy: float = 0.0
    cz: float = 0.0
    h: float = 0.0
    w: float = 0.0
    l: float = 0.0
    yaw_bev: float = 0.0
    fit_points: int = 0  # 拟合点数（置信度/HITL 分流键，恒输出）
    truncated: float = 0.0
    occluded: int = 0
    alpha: float = 0.0  # v0.1 由 2D 检测不可得，恒 0.0（评测不计 alpha）
    x1: float = 0.0  # 2D bbox（导出字段 4-7，检测框透传）
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    review_flag: bool = False  # fit_points 低/粘连嫌疑 → HITL 强制 review
    edited_by_human: bool | None = None
    track_id: int | None = None  # v0.2 P2 多帧跟踪 id（Tracker3D.update 回写；None = 单帧模式）

    @property
    def rotation_y(self) -> float:
        """KITTI camera 系 rotation_y（绕 y 轴，从 +x 向 +z 为正），wrap [-π, π]。

        唯一转换点：ry = yaw_bev - π/2（见 tools/geometry.yaw_to_rotation_y 推导与测试）。
        """
        return geometry.yaw_to_rotation_y(self.yaw_bev)

    @classmethod
    def from_gt_row(
        cls, label: str, h: float, w: float, l: float, x: float, y: float, z: float, ry: float
    ) -> Box3D:
        """KITTI label_2 15 字段行 → Box3D（GT 消费，评测用）。

        红线：label_2 的 y 是**物体底部中心**（地面，相机系 y 向下），物体向上延伸——
        此处 cy = y - h/2 归一到内部「体积中心」语义；导出（export/kitti_label.line_from_box3d）
        对称地 y = cy + h/2 回写。
        """
        return cls(
            label=label, cx=x, cy=y - h / 2, cz=z, h=h, w=w, l=l,
            yaw_bev=geometry.rotation_y_to_yaw(ry), confidence=1.0,
        )

    def corners_bev(self) -> np.ndarray:
        """(4,2) 鸟瞰角点（x, z 平面），顺序：右前/右后/左后/左前。

        车头方向向量为 (sin yaw, cos yaw)，右向垂直向量为 (cos yaw, -sin yaw)。
        """
        dx, dz = np.sin(self.yaw_bev), np.cos(self.yaw_bev)  # 车头单位向量
        px, pz = dz, -dx  # 垂直向量（右）
        hl, hw = self.l / 2, self.w / 2
        return np.array(
            [
                [self.cx + dx * hl + px * hw, self.cz + dz * hl + pz * hw],
                [self.cx - dx * hl + px * hw, self.cz - dz * hl + pz * hw],
                [self.cx - dx * hl - px * hw, self.cz - dz * hl - pz * hw],
                [self.cx + dx * hl - px * hw, self.cz + dz * hl - pz * hw],
            ],
            dtype=np.float64,
        )

    def corners_cam(self) -> np.ndarray:
        """(8,3) 相机系 8 角点，列序 (x, y, z)（BEV 消费者取 [:, [0, 2]]）。"""
        bev = self.corners_bev()  # (4,2) (x,z)
        y_low, y_high = self.cy - self.h / 2, self.cy + self.h / 2
        low = np.hstack([bev[:, :1], np.full((4, 1), y_low), bev[:, 1:]])
        high = np.hstack([bev[:, :1], np.full((4, 1), y_high), bev[:, 1:]])
        return np.vstack([low, high])

    def to_dict(self) -> dict:
        """Web 队列/agent 工具/评测中间格式。

        键命名 label/confidence（兼容 evaluate_detections 的 SAHI dict 分支）；fit_points 恒输出；
        edited_by_human 仅 True 时输出（同 auto2dlabel Bbox.to_dict 防膨胀纪律）。
        """
        d: dict = {
            "label": self.label,
            "confidence": self.confidence,
            "cx": self.cx,
            "cy": self.cy,
            "cz": self.cz,
            "h": self.h,
            "w": self.w,
            "l": self.l,
            "rotation_y": self.rotation_y,
            "fit_points": self.fit_points,
            "truncated": self.truncated,
            "occluded": self.occluded,
            "alpha": self.alpha,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "review_flag": self.review_flag,
        }
        if self.edited_by_human:
            d["edited_by_human"] = True
        if self.track_id is not None:
            d["track_id"] = self.track_id
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Box3D:
        """Web save 重建（3D 字段直通，不经 auto2dlabel _bbox_from_dict）。"""
        return cls(
            label=str(d.get("label", "")),
            confidence=float(d.get("confidence", 1.0)),
            cx=float(d.get("cx", 0.0)),
            cy=float(d.get("cy", 0.0)),
            cz=float(d.get("cz", 0.0)),
            h=float(d.get("h", 0.0)),
            w=float(d.get("w", 0.0)),
            l=float(d.get("l", 0.0)),
            yaw_bev=geometry.rotation_y_to_yaw(float(d.get("rotation_y", 0.0))),
            fit_points=int(d.get("fit_points", 0)),
            truncated=float(d.get("truncated", 0.0)),
            occluded=int(d.get("occluded", 0)),
            alpha=float(d.get("alpha", 0.0)),
            x1=float(d.get("x1", 0.0)),
            y1=float(d.get("y1", 0.0)),
            x2=float(d.get("x2", 0.0)),
            y2=float(d.get("y2", 0.0)),
            review_flag=bool(d.get("review_flag", False)),
            edited_by_human=d.get("edited_by_human"),
            track_id=d.get("track_id"),
        )


def load_points_bin(path: Path) -> np.ndarray:
    """(N,4) float32 velodyne 点（x,y,z,intensity），剔除 NaN/Inf。

    KittiFrame.load_points 与 web/payloads.frame_payload 共用（v0.3 P4 抽公共）。
    """
    pts = np.asarray(np.fromfile(path, dtype=np.float32)).reshape(-1, 4)
    return np.asarray(pts[np.isfinite(pts).all(axis=1)])


@dataclass
class KittiFrame:
    """KITTI 单帧（路径容器 + 懒加载；frame_id 为 6 位零填充字符串）。"""

    frame_id: str
    root: Path

    @property
    def image_path(self) -> Path:
        return self.root / "training" / "image_2" / f"{self.frame_id}.png"

    @property
    def velodyne_path(self) -> Path:
        return self.root / "training" / "velodyne" / f"{self.frame_id}.bin"

    @property
    def calib_path(self) -> Path:
        return self.root / "training" / "calib" / f"{self.frame_id}.txt"

    @property
    def label_path(self) -> Path:
        return self.root / "training" / "label_2" / f"{self.frame_id}.txt"

    def load_points(self) -> np.ndarray:
        """(N,4) float32 velodyne 点（x,y,z,intensity），剔除 NaN/Inf。"""
        return load_points_bin(self.velodyne_path)

    def load_calib(self) -> KittiCalib:
        return KittiCalib.from_file(self.calib_path)

    def load_image(self) -> np.ndarray:
        import cv2

        img = cv2.imread(str(self.image_path))
        if img is None:
            raise FileNotFoundError(f"图像读取失败: {self.image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def load_gt3d(self) -> list[Box3D]:
        """label_2 GT 3D 框（无 label 文件返回 []）。"""
        if not self.label_path.is_file():
            return []
        boxes: list[Box3D] = []
        for line in self.label_path.read_text(encoding="utf-8").strip().splitlines():
            parts = line.split()
            if len(parts) < 15:
                continue
            name = parts[0]
            if name in ("DontCare", "Misc"):
                continue
            h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
            x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
            ry = float(parts[14])
            boxes.append(Box3D.from_gt_row(name, h, w, l, x, y, z, ry))
        return boxes


@dataclass
class FrameResult:
    """单帧标注产物（Web 队列与 CLI 统计消费）。"""

    frame_id: str
    boxes3d: list[Box3D] = field(default_factory=list)
    masks: list = field(default_factory=list)  # 复用 auto2dlabel Mask（仅透传，不深依赖）
    dropped_no_points: int = 0  # mask 内零 LiDAR 点的实例数（宁缺勿假）
    warnings: list[str] = field(default_factory=list)
    proj_check_path: str | None = None
    bev_path: str | None = None
