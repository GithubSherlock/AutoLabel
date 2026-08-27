"""3D 标注日志（v0.2 工具层）——复用 auto2dlabel 日志机制，3D 语义适配。

红线「复用不复制」：AnnotationLogger / log_llm_call / log_python_api_call 零 2D 耦合，
直接 re-export；3D 差异只在 log_chat_call 的聚合键（boxes3d_count + 三档 HITL）
与 ANNOTATION_TYPES 词表。日志落 `AutoLabel/logs/`（与 2D 共享目录，时间戳命名不冲突）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from auto2dlabel.tools.log import (
    LOGS_DIR,
    AnnotationLogger,
    log_llm_call,
    log_python_api_call,
)

# 3D 标注任务词表（task_type 与 planner3d 对齐）
ANNOTATION_TYPES = {
    "kitti_3d": "KITTI 3D 标注",
    "nuscenes_3d": "nuScenes 3D 标注",
}


def log_chat_call(
    instruction: str,
    plan_summary: dict[str, Any],
    steps_results: list[dict[str, Any]],
    elapsed: float,
    llm_model: str = "",
    annotation_type: str = "kitti_3d",
    timestamp: str | None = None,
    error: str | None = None,
) -> Path:
    """便捷函数：记录一次 3D Chat 模式调用（镜像 2D log_chat_call，聚合键换 3D）。

    Args:
        instruction: 用户自然语言指令。
        plan_summary: TaskPlan 摘要（帧 ID、模型等）。
        steps_results: 每步执行结果 [{step_id, model, boxes3d_count, triage_summary}]。
        elapsed: 总耗时（秒）。
        llm_model: LLM 服务/模型名。
        annotation_type: 标注类别（kitti_3d / nuscenes_3d）。
        timestamp: 统一时间戳。
        error: 错误信息。
    """
    logger = AnnotationLogger(
        method="Chat",
        timestamp=timestamp,
        instruction=instruction,
        llm_model=llm_model,
        annotation_type=annotation_type,
        annotation_type_cn=ANNOTATION_TYPES.get(annotation_type, annotation_type),
    )
    logger.log_step("plan_parsed", {
        "instruction": instruction,
        "plan": plan_summary,
    })
    total_boxes = 0
    total_accepted = 0
    total_review = 0
    total_hard = 0
    for i, sr in enumerate(steps_results):
        logger.log_step(f"step_{i + 1}_done", sr)
        total_boxes += sr.get("boxes3d_count", 0)
        triage = sr.get("triage_summary", {})
        total_accepted += triage.get("accepted", 0)
        total_review += triage.get("review", 0)
        total_hard += triage.get("hard", 0)
    logger.set_result({
        "total_boxes3d": total_boxes,
        "total_accepted": total_accepted,
        "total_review": total_review,
        "total_hard": total_hard,
    })
    if error:
        logger.set_error(error)
    return logger.save()


__all__ = [
    "ANNOTATION_TYPES",
    "AnnotationLogger",
    "LOGS_DIR",
    "log_chat_call",
    "log_llm_call",
    "log_python_api_call",
]
