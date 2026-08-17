"""模型推理封装模块。

库载入重构：os / pathlib.Path / PIL.Image 在此集中导入（显式 as 再导出，
mypy no-implicit-reexport 要求），子模块通过
`from auto2dlabel.models import os, Path, Image` 引用，消除各文件重复 import。
"""

import os as os
from pathlib import Path as Path

from PIL import Image as Image

from auto2dlabel.models.detection import DetectionModel, DetectionResult, GroundingDINOModel
from auto2dlabel.models.segmentation import SAM2Model, SegmentationModel

__all__ = [
    "DetectionModel",
    "DetectionResult",
    "GroundingDINOModel",
    "SegmentationModel",
    "SAM2Model",
]
