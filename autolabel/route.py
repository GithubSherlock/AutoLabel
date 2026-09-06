"""路由判据单一事实源（docs/Agentic_UI_plan.md §9.3）。

判据表：

| 判据 | 路由 |
| --- | --- |
| det_model ∈ 3D 引擎名（四名单并集） | 强制 3D |
| 指令含 3D 强特征词（点云/lidar/velodyne/3d bbox/3d框/3d检测 等） | 3D |
| 指令含独立 6 位帧号（非 .png/.jpg 等图像文件名） | 3D |
| 其余（含仅 kitti 字样） | 2D（默认域宁 2D 勿错） |

边界修正（§9.3）：「KITTI」不能作为 3D 强特征词——kitti_finetune / yolo11s_kitti
是 2D 模型，「检测 KITTI 数据集的汽车」是合法 2D 任务。
另一边界：「000860.png」等 2D 图像文件名含 6 位数字，须排除扩展名尾缀。
"""

from __future__ import annotations

import re

# ENGINE3D_NAMES 单一事实源在 auto3dlabel/configs/model_catalog.py（四名单并集，
# 协议纪律见其 docstring：BEVFUSION/FCOS3D/MONO3D 不进 DETECTOR3D_NAMES 但同为
# 3D 引擎）——route 只引用不复制
from auto3dlabel.configs.model_catalog import ENGINE3D_NAMES as ENGINE3D_NAMES  # 显式再导出（mypy）

# 3D 强特征词（指令小写后子串匹配；不含 kitti——见模块 docstring 边界修正）
_3D_KEYWORDS: tuple[str, ...] = (
    "点云",
    "lidar",
    "velodyne",
    "3d bbox",
    "3d框",
    "3d检测",
    "3d 检测",
    "3d标注",
    "3d 标注",
    "nuscenes",
)

# 独立 6 位帧号（KITTI frame_id），排除图像文件名（000860.png / .jpg 等）
# (?!\d)：COCO 12 位编号（000000000139.jpg）前 6 位后仍跟数字 → 不匹配（曾误判 3D）
_FRAME_ID_RE = re.compile(r"(?<![0-9A-Za-z])\d{6}(?!\d)(?!\s*\.(?:png|jpe?g|tiff?|bmp|webp)\b)")


def route_domain(instruction: str, det_model: str | None = None) -> str:
    """路由判据 → "2d" | "3d"。

    Args:
        instruction: 用户自然语言指令。
        det_model: CLI 指定检测模型名（None = 未指定）。

    Returns:
        "3d" 或 "2d"（默认域宁 2D 勿错）。
    """
    if det_model and det_model.lower() in ENGINE3D_NAMES:
        return "3d"
    text = instruction.lower()
    if any(keyword in text for keyword in _3D_KEYWORDS):
        return "3d"
    if _FRAME_ID_RE.search(instruction):
        return "3d"
    return "2d"
