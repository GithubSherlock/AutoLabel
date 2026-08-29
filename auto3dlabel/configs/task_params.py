"""3D 任务参数规格表（照 auto2dlabel/configs/task_params.py 模式，复用不复制）。

与 2D 表的关系：ParamSpec 复用 2D 定义（同一数据类，改一处不漏另一处）；
3D 只维护 3D 字段集（frame_id/prompts/conf/det/seg）。规格表驱动：
- Plan3D.missing_params（agent/planner3d.py）：required 字段值 falsy 即缺失
  （frame_id "" / prompts None）→ 3D cli 缺参检查（含 prompts——原只查 frame_id）
- 缺参清单文案（label 直接展示）

默认值引用 configs/kitti.py 常量（单一事实源——改常量不漏表），
与 Plan3D dataclass 默认同源。
"""

from __future__ import annotations

from auto2dlabel.configs.task_params import ParamSpec
from auto3dlabel.configs.kitti import DEFAULT_CONF, DEFAULT_DET_MODEL, DEFAULT_SEG_MODEL

PARAM_SPECS_3D: dict[str, ParamSpec] = {
    spec.key: spec
    for spec in (
        ParamSpec(
            "frame_id", "frame_id（KITTI 帧号）", "要标注哪个 KITTI 帧？(如 000123)", True, ""
        ),
        ParamSpec("prompts", "prompts（检测类别）", "要标注哪些类别？(如 car, person)", True, None),
        ParamSpec(
            "confidence_threshold",
            "confidence_threshold（置信度）",
            "",
            False,
            DEFAULT_CONF,
        ),
        ParamSpec("det_model", "det_model（检测模型）", "", False, DEFAULT_DET_MODEL),
        ParamSpec("seg_model", "seg_model（分割模型）", "", False, DEFAULT_SEG_MODEL),
    )
}
