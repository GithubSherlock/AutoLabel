"""nuScenes 3D 框数据模型（不共用 Box3D——全局系 + 四元数与 KITTI 相机系不兼容）。

坐标系红线：nuScenes 全局系 x 前 / y 左 / z 上，rotation 为四元数 (w,x,y,z)；
Box3D 是 KITTI 相机系（y 向下）+ yaw_bev——两者之间不做隐式转换，
转换点显式落在 tools/geometry.py（yaw_to_quat）与 data/nuscenes.py（GT 加载）。

mmdet3d nus-3d 模型输出即全局系 → NusBox 零旋转直映射。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NusBox:
    """nuScenes 全局系 3D 框。

    size 顺序 (w, l, h)（nuScenes 官方语义，与 Box3D 的 (h,w,l) 不同——消费点在此登记：
    1. export/nuscenes_json.build_submission_dict
    2. benchmarks/nuscenes_benchmark（size 直接喂评测 IoU，不做语义重排））
    """

    label: str  # 官方检测类名（NUSCENES_CLASSES 之一）
    confidence: float = 1.0
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)  # (x, y, z) 全局系
    size: tuple[float, float, float] = (0.0, 0.0, 0.0)  # (w, l, h)
    quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # (w, x, y, z)
    velocity: tuple[float, float] | None = None  # (vx, vy) 全局 x-y 平面，m/s
    track_id: str | None = None  # instance_token（P2 的 int track_id 语义不同，不混用）

    def to_dict(self) -> dict:
        """官方提交 JSON 的 box 字段结构（sample_token 在导出层补）。"""
        d: dict = {
            "detection_name": self.label,
            "detection_score": self.confidence,
            "translation": list(self.translation),
            "size": list(self.size),
            "rotation": list(self.quaternion),
        }
        if self.velocity is not None:
            d["velocity"] = list(self.velocity)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> NusBox:
        """to_dict 的官方提交键反向重建（tools/export.py nuscenes 分支消费）。

        track_id 可读（instance_token）但 to_dict 不输出——官方提交格式无此字段。
        """
        velocity_raw = d.get("velocity")
        trans_raw = d.get("translation", (0.0, 0.0, 0.0))
        size_raw = d.get("size", (0.0, 0.0, 0.0))
        rot_raw = d.get("rotation", (1.0, 0.0, 0.0, 0.0))
        return cls(
            label=str(d.get("detection_name", "")),
            confidence=float(d.get("detection_score", 1.0)),
            translation=(float(trans_raw[0]), float(trans_raw[1]), float(trans_raw[2])),
            size=(float(size_raw[0]), float(size_raw[1]), float(size_raw[2])),
            quaternion=(
                float(rot_raw[0]), float(rot_raw[1]), float(rot_raw[2]), float(rot_raw[3])
            ),
            velocity=(
                (float(velocity_raw[0]), float(velocity_raw[1]))
                if velocity_raw is not None
                else None
            ),
            track_id=d.get("track_id"),
        )
