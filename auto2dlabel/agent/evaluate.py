"""代码级质量评估 —— M3 Evaluate 节点的代码级基础（0 框/缺类/超框数）。

纯函数、无 I/O：0 框判定 / 类别覆盖检查 / 超框数异常警告。
三条检测路径（Agent Loop / chat / no-LLM baseline）共用，保证行为一致。

评估在 tool/编排层自动执行，不增加 Agent Loop 迭代次数（max_iterations=3 红线）。
LLM Evaluate 节点仅在评估不通过（ok=False）时条件暴露，由 LLM 选择处置动作，
代码执行（apply_evaluate_action），动作不消耗额外迭代。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# LLM 选择 retry_lower_threshold 时的降阈值倍率（每图仅一次）
LLM_RETRY_FACTOR = 0.25


def pick_alternate_model(current: str) -> str:
    """按当前模型选备选检测模型（模型级重试，与 retry_lower_threshold 同构：每图一次）。

    规则（纯函数）：
    - yolo 系列 → fasterrcnn_resnet50_fpn_v2（高召回档，互补漏检）
    - fasterrcnn / grounding-dino → yolo26x.pt（均衡精度档）
    - 未知模型名 → DEFAULT_MODEL（yolo26x.pt）
    """
    from auto2dlabel.schema.task_plan import DEFAULT_MODEL

    name = current.lower()
    if "yolo" in name:
        return "fasterrcnn_resnet50_fpn_v2"
    if "fasterrcnn" in name or "grounding" in name or "/" in current:
        return DEFAULT_MODEL
    return DEFAULT_MODEL


@dataclass
class QualityReport:
    """单张图检测结果的代码级质量评估报告。"""

    image_path: str = ""
    total_boxes: int = 0
    prompts: list[str] = field(default_factory=list)
    covered_prompts: list[str] = field(default_factory=list)
    missing_prompts: list[str] = field(default_factory=list)
    retried: bool = False                  # 是否降阈值重试过
    retry_threshold: float | None = None   # 重试实际使用的阈值
    warnings: list[str] = field(default_factory=list)  # "0 框" / 超框数异常

    @property
    def ok(self) -> bool:
        """无警告且无缺类。"""
        return not self.warnings and not self.missing_prompts

    def to_dict(self) -> dict[str, Any]:
        """JSON 序列化（state.metadata / log）。"""
        return {
            "image_path": self.image_path,
            "total_boxes": self.total_boxes,
            "prompts": self.prompts,
            "covered_prompts": self.covered_prompts,
            "missing_prompts": self.missing_prompts,
            "retried": self.retried,
            "retry_threshold": self.retry_threshold,
            "warnings": self.warnings,
        }

    def summary_line(self) -> str:
        """单行摘要（注入 LLM / console）；无值得报告的信息时返回空串。

        只含对 LLM 有价值的信息（缺类 + 0 框重试结果）；超框数警告
        不进 LLM（防 token 膨胀），走 metadata/log 通道。
        """
        parts: list[str] = []
        if self.total_boxes == 0:
            parts.append("0 框" + ("（已降阈值重试）" if self.retried else ""))
        if self.missing_prompts:
            parts.append(f"缺类: {', '.join(self.missing_prompts)}")
        return " | ".join(parts)


def evaluate_detections(
    detections: list[Any],
    prompts: list[str],
    confidence_threshold: float = 0.3,
    image_path: str = "",
    max_box_warning: int = 200,
    retried: bool = False,
    retry_threshold: float | None = None,
    prompt_matcher: Callable[[str, list[str]], bool] | None = None,
) -> QualityReport:
    """纯函数：0 框判定 / 类别覆盖检查 / 超框数警告。不做任何 I/O。

    detections 元素兼容三种格式：
    - DetectionResult / Bbox（对象，有 .label / .confidence）
    - SAHI dict（{"name"/"label", "bbox", "conf"/"confidence"}）

    confidence_threshold 仅作为上下文保留，不参与判定。
    prompt_matcher: 类别覆盖检查用的匹配函数（OBB 场景传 _match_obb_prompt，
    默认 _match_prompt 双向子串）。
    """
    from auto2dlabel.models.detection import _match_prompt

    matcher = prompt_matcher or _match_prompt

    labels: list[str] = []
    for d in detections:
        if isinstance(d, dict):
            label = str(d.get("name") or d.get("label") or "")
        else:
            label = str(getattr(d, "label", ""))
        labels.append(label)

    covered: list[str] = []
    missing: list[str] = []
    for p in prompts:
        if any(matcher(lab, [p]) for lab in labels):
            covered.append(p)
        else:
            missing.append(p)

    warnings: list[str] = []
    if not detections:
        warnings.append("0 框")
    if len(detections) > max_box_warning:
        warnings.append(f"框数异常: {len(detections)} > {max_box_warning}")

    return QualityReport(
        image_path=image_path,
        total_boxes=len(detections),
        prompts=list(prompts),
        covered_prompts=covered,
        missing_prompts=missing,
        retried=retried,
        retry_threshold=retry_threshold,
        warnings=warnings,
    )


def apply_evaluate_action(
    action: str,
    *,
    report: QualityReport,
    retry_used: bool,
    base_threshold: float,
    detect_fn: Callable[[float], list[Any]] | None = None,
    model_retried: bool = False,
    swap_fn: Callable[[], list[Any]] | None = None,
) -> dict[str, Any]:
    """执行 LLM 选择的处置动作（纯函数，detect_fn/swap_fn 由编排器注入）。

    action ∈ {accept, flag_for_review, retry_lower_threshold, retry_swap_model}：
    - accept: 接受现状，不做任何事
    - flag_for_review: 返回标记意图，调用方把整图并入 review 档
    - retry_lower_threshold: 调 detect_fn(base×LLM_RETRY_FACTOR) 一次（每图一次）；
      retry_used=True（代码级已重试过）时拒绝重试，转为 accept
    - retry_swap_model: 调 swap_fn()（备选模型重检）一次（每图一次）；
      model_retried=True（已换过模型）时拒绝，转为 accept
    非法 action 抛 ValueError。
    """
    result: dict[str, Any] = {"action": action, "report": report.to_dict()}

    if action == "accept":
        result["accepted"] = True
    elif action == "flag_for_review":
        result["flagged"] = True
    elif action == "retry_lower_threshold":
        if retry_used:
            result["accepted"] = True
            result["note"] = "代码级已降阈值重试过，忽略再次重试请求"
            return result
        if detect_fn is None:
            raise ValueError("retry_lower_threshold 需要 detect_fn")
        lowered = base_threshold * LLM_RETRY_FACTOR
        result["retried"] = True
        result["threshold"] = lowered
        result["detections"] = detect_fn(lowered)
    elif action == "retry_swap_model":
        if model_retried:
            result["accepted"] = True
            result["note"] = "已换过模型，忽略再次换模型请求"
            return result
        if swap_fn is None:
            raise ValueError("retry_swap_model 需要 swap_fn")
        result["retried_swap"] = True
        result["detections"] = swap_fn()
    else:
        raise ValueError(
            f"非法 action: {action}"
            "（可选 accept/flag_for_review/retry_lower_threshold/retry_swap_model）"
        )

    return result
