"""统一内部标注数据结构。

所有 Tool 的输入/输出都使用这里的类型，导出层负责转换为目标格式。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Bbox:
    """边界框，使用 COCO 格式 [x, y, width, height]，坐标归一化或像素坐标均可。"""

    x: float
    y: float
    width: float
    height: float
    label: str
    confidence: float = 1.0
    id: int | None = None
    angle: float = 0.0  # 旋转角度（弧度），width 轴相对 x 轴，(-π/2, π/2]；HBB 恒为 0
    track_id: int | None = None  # 跟踪 ID（ByteTrack 关联；None=未跟踪）
    keypoints: list[tuple[float, float, float]] = field(default_factory=list)
    # 姿态关键点 [x, y, v]（COCO 17 点语义，像素坐标，v=0 未标注/1 标注/2 不可见）；空 = 无姿态
    edited_by_human: bool | None = None
    # 人工修正标记（Web 审核拖拽/缩放/旋转/改标签置位；None=AI 初稿，仅 True 时 to_dict 输出）

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        """转换为 [x1, y1, x2, y2] 格式。"""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    @property
    def num_keypoints(self) -> int:
        """可见关键点数量（v > 0）。"""
        return sum(1 for _x, _y, v in self.keypoints if v > 0)

    @classmethod
    def from_xyxy(
        cls,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        label: str,
        confidence: float = 1.0,
    ) -> Bbox:
        """从 [x1, y1, x2, y2] 格式创建 Bbox。"""
        return cls(x=x1, y=y1, width=x2 - x1, height=y2 - y1, label=label, confidence=confidence)

    def area(self) -> float:
        """返回边界框面积。"""
        return self.width * self.height

    def to_dict(self) -> dict[str, Any]:
        d = {
            "x": self.x, "y": self.y, "width": self.width, "height": self.height,
            "label": self.label, "confidence": self.confidence, "angle": self.angle,
        }
        if self.id is not None:
            d["id"] = self.id
        if self.track_id is not None:
            d["track_id"] = self.track_id
        if self.keypoints:  # 非空才输出（防普通检测 JSON 膨胀）
            d["keypoints"] = [list(k) for k in self.keypoints]
        if self.edited_by_human:
            d["edited_by_human"] = True
        return d


@dataclass
class ImageLabel:
    """图像级分类标签（CLIP / SigLIP 零样本分类结果）。"""

    label: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "score": self.score}


@dataclass
class Mask:
    """实例分割 mask。

    segmentation: COCO 格式的 polygon [[x1,y1, x2,y2, ...]] 或 RLE。
    """

    bbox: Bbox
    segmentation: list[list[float]]  # COCO polygon 格式
    area: float = 0.0

    def __post_init__(self):
        if self.area == 0.0 and self.segmentation:
            self.area = self.bbox.area()

    def to_dict(self) -> dict:
        return {
            "bbox": self.bbox.to_dict(),
            "segmentation": self.segmentation,
            "area": self.area,
        }


@dataclass
class Annotation:
    """单张图像的全部标注数据。

    对应一张输入图像的完整标注结果，包含所有检测/分割的物体。
    """

    image_path: str
    image_id: int | None = None
    image_size: tuple[int, int] = (0, 0)  # (width, height)
    bboxes: list[Bbox] = field(default_factory=list)
    masks: list[Mask] = field(default_factory=list)
    labels: list[ImageLabel] = field(default_factory=list)  # 图像级分类标签
    review_flags: list[int] = field(default_factory=list)  # bbox/mask 索引,标记为需人工审核
    metadata: dict = field(default_factory=dict)

    def add_bbox(self, bbox: Bbox) -> None:
        bbox.id = len(self.bboxes)
        self.bboxes.append(bbox)

    def add_mask(self, mask: Mask) -> None:
        self.masks.append(mask)

    def flag_for_review(self, index: int) -> None:
        if index not in self.review_flags:
            self.review_flags.append(index)

    @property
    def has_annotations(self) -> bool:
        return len(self.bboxes) > 0 or len(self.masks) > 0 or len(self.labels) > 0

    @property
    def summary(self) -> str:
        parts = []
        if self.bboxes:
            parts.append(f"{len(self.bboxes)} bboxes")
        if self.masks:
            parts.append(f"{len(self.masks)} masks")
        if self.labels:
            parts.append(f"{len(self.labels)} labels")
        if self.review_flags:
            parts.append(f"{len(self.review_flags)} flagged for review")
        return f"Annotation({self.image_path}: {', '.join(parts)})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "image_id": self.image_id,
            "image_size": list(self.image_size),
            "bboxes": [b.to_dict() for b in self.bboxes],
            "masks": [m.to_dict() for m in self.masks],
            "labels": [lab.to_dict() for lab in self.labels],
            "review_flags": self.review_flags,
            "metadata": self.metadata,
        }
