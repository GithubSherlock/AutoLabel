"""日志系统 —— 记录每次调用的完整信息。

日志文件命名: <method>_<timestamp>.log
  - method: "PythonAPI" | "LLM" | "Chat"
  - timestamp: YYYY-MM-DD-HH-MM-SS

日志保存在 AutoLabel/logs/ 目录中。

v0.1.0 新增：
  - log_chat_call() — 记录 Chat 模式调用（NL 指令 + 多步执行 + HITL）
  - 所有便捷函数支持 triage_stats 参数
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs"

# 标注类别常量
ANNOTATION_TYPES = {
    "object_detection": "目标检测",
    "semantic_segmentation": "语义分割",
    "instance_segmentation": "实例分割",
    "classification": "图像分类",
    "pose_estimation": "姿态估计",
    "obb_detection": "旋转框检测",
    "tracking": "目标跟踪",
}


def _ensure_logs_dir() -> Path:
    LOGS_DIR.mkdir(exist_ok=True)
    return LOGS_DIR


def _build_filename(method: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return f"{method}_{ts}.log"


class AnnotationLogger:
    """标注日志记录器。

    记录每次标注调用的：输入参数、中间步骤、输出结果、耗时和错误。
    """

    def __init__(self, method: str, timestamp: str | None = None, **metadata: Any):
        """
        Args:
            method: 运行方式 "PythonAPI" | "LLM"
            timestamp: 外部传入的统一时间戳，为 None 则自动生成。
            **metadata: 额外信息 (image_path, instruction, etc.)
        """
        self.method = method
        self._external_ts = timestamp
        self.metadata = metadata
        self.start_time = time.time()
        self.steps: list[dict] = []
        self.result: Any = None
        self.error: str | None = None

    def log_step(self, step_name: str, detail: dict | None = None) -> None:
        """记录一个执行步骤。"""
        self.steps.append({
            "step": step_name,
            "time": round(time.time() - self.start_time, 3),
            "detail": detail or {},
        })

    def set_result(self, result: Any) -> None:
        self.result = result

    def set_error(self, error: str) -> None:
        self.error = error

    def save(self) -> Path:
        """将日志写入文件。"""
        _ensure_logs_dir()

        elapsed = round(time.time() - self.start_time, 3)
        # 使用外部传入的统一时间戳，没有则自动生成
        ts = self._external_ts or datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = f"{self.method}_{ts}.log"
        filepath = LOGS_DIR / filename

        log_entry = {
            "method": self.method,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": elapsed,
            "metadata": self.metadata,
            "steps": self.steps,
            "result": self._serialize_result(),
            "error": self.error,
        }

        filepath.write_text(json.dumps(log_entry, indent=2, ensure_ascii=False, default=str))
        return filepath

    def _serialize_result(self) -> Any:
        if self.result is None:
            return None
        try:
            json.dumps(self.result, default=str)
            return self.result
        except (TypeError, ValueError):
            return str(self.result)


# ============================================================
# 便捷函数
# ============================================================

def log_python_api_call(
    image_path: str,
    prompts: list[str],
    results: list[Any],
    elapsed: float,
    model_name: str = "",
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.5,
    annotation_type: str = "object_detection",
    timestamp: str | None = None,
    quality: dict[str, Any] | None = None,
) -> Path:
    """便捷函数：记录一次 PythonAPI 直接调用。"""
    logger = AnnotationLogger(
        method="PythonAPI",
        timestamp=timestamp,
        image_path=image_path,
        prompts=prompts,
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        annotation_type=annotation_type,
        annotation_type_cn=ANNOTATION_TYPES.get(annotation_type, annotation_type),
        quality=quality,
    )
    logger.log_step("detect_start", {
        "prompts": prompts,
        "image": image_path,
        "model": model_name,
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "annotation_type": annotation_type,
    })
    logger.log_step("detect_done", {"count": len(results), "elapsed": round(elapsed, 3)})
    logger.set_result({"count": len(results)})
    return logger.save()


def log_llm_call(
    image_path: str,
    instruction: str,
    state_summary: dict,
    elapsed: float,
    llm_model: str = "",
    detection_model: str = "",
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.5,
    annotation_type: str = "object_detection",
    timestamp: str | None = None,
    error: str | None = None,
) -> Path:
    """便捷函数：记录一次 LLM Agent 调用。"""
    logger = AnnotationLogger(
        method="LLM",
        timestamp=timestamp,
        image_path=image_path,
        instruction=instruction,
        llm_model=llm_model,
        detection_model=detection_model,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        annotation_type=annotation_type,
        annotation_type_cn=ANNOTATION_TYPES.get(annotation_type, annotation_type),
    )
    logger.log_step("agent_start", {
        "instruction": instruction,
        "llm_model": llm_model,
        "detection_model": detection_model,
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "annotation_type": annotation_type,
    })
    logger.log_step("agent_done", state_summary)
    if error:
        logger.set_error(error)
    return logger.save()


def log_chat_call(
    instruction: str,
    plan_summary: dict,
    steps_results: list[dict],
    elapsed: float,
    llm_model: str = "",
    annotation_type: str = "object_detection",
    timestamp: str | None = None,
    error: str | None = None,
) -> Path:
    """便捷函数：记录一次 Chat 模式调用（v0.1.0 新增）。

    Args:
        instruction: 用户自然语言指令。
        plan_summary: TaskPlan 摘要（步骤数、每步参数）。
        steps_results: 每步执行结果列表 [{step_id, model, bbox_count, triage_summary}].
        elapsed: 总耗时（秒）。
        llm_model: LLM 模型名。
        annotation_type: 标注类别。
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
    total_bboxes = 0
    total_accepted = 0
    total_review = 0
    for i, sr in enumerate(steps_results):
        logger.log_step(f"step_{i+1}_done", sr)
        total_bboxes += sr.get("bbox_count", 0)
        triage = sr.get("triage_summary", {})
        total_accepted += triage.get("accepted", 0)
        total_review += triage.get("review", 0)
    logger.set_result({
        "total_bboxes": total_bboxes,
        "total_accepted": total_accepted,
        "total_review": total_review,
    })
    if error:
        logger.set_error(error)
    return logger.save()
