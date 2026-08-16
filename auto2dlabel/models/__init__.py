"""模型推理封装模块。"""

from auto2dlabel.models.detection import DetectionModel, DetectionResult, GroundingDINOModel
from auto2dlabel.models.segmentation import SAM2Model, SegmentationModel

__all__ = [
    "DetectionModel",
    "DetectionResult",
    "GroundingDINOModel",
    "SegmentationModel",
    "SAM2Model",
]
