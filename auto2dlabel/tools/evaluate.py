"""LLM Evaluate 节点 —— 条件暴露的质量评估工具。

仅在代码级质量评估不通过（quality.ok == False）时由编排器附加到当轮 tools，
正常图零 token 开销。**不进全局 registry**（防跨图状态污染），
由编排器持有实例并直接调用。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from auto2dlabel.tools.base import Tool

if TYPE_CHECKING:
    from auto2dlabel.agent.evaluate import QualityReport


class EvaluateTool(Tool):
    """让 LLM 在质量评估失败时选择处置动作。"""

    name = "evaluate_quality"
    description = (
        "质量评估未通过时选择处置动作: accept(接受现状) / "
        "flag_for_review(整图转入人工复核队列) / "
        "retry_lower_threshold(以更低置信度阈值重新检测一次) / "
        "retry_swap_model(换一个备选检测模型重新检测一次)"
    )

    def __init__(
        self,
        report: QualityReport,
        retry_used: bool,
        base_threshold: float,
        detect_fn: Callable[[float], list[Any]] | None = None,
        model_retried: bool = False,
        swap_fn: Callable[[], list[Any]] | None = None,
    ) -> None:
        self.report = report
        self.retry_used = retry_used
        self.base_threshold = base_threshold
        self.detect_fn = detect_fn
        self.model_retried = model_retried
        self.swap_fn = swap_fn

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "accept", "flag_for_review",
                        "retry_lower_threshold", "retry_swap_model",
                    ],
                    "description": "处置动作（只能调用一次）",
                },
            },
            "required": ["action"],
        }

    def forward(self, **kwargs: Any) -> dict[str, Any]:
        from auto2dlabel.agent.evaluate import apply_evaluate_action

        return apply_evaluate_action(
            str(kwargs.get("action", "")),
            report=self.report,
            retry_used=self.retry_used,
            base_threshold=self.base_threshold,
            detect_fn=self.detect_fn,
            model_retried=self.model_retried,
            swap_fn=self.swap_fn,
        )
