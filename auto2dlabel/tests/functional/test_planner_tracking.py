"""Planner tracking 解析测试（FakeLLM 固定 JSON，零网络）。

覆盖：tracking 步骤透传（prompts 固定类别 / model_name 不含跟踪器 /
export_format=mot）、_dict_to_plan 默认值填充、system prompt 含 tracking
规则子串（防 prompt 回退）。
"""

from __future__ import annotations

from typing import Any, cast

from auto2dlabel.agent.llm import LLMClient, LLMResponse
from auto2dlabel.agent.planner import _PLANNER_SYSTEM_PROMPT, TaskPlanner, _dict_to_plan


class _FixedLLM:
    """返回固定 JSON 内容的假 LLM。"""

    model = "fake-llm"

    def __init__(self, content: str) -> None:
        self._content = content

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.1,
    **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(content=self._content)


_TRACK_JSON = """{
  "steps": [{
    "step_id": 1,
    "task_type": "tracking",
    "source": "/data/video.mp4",
    "prompts": ["person", "car"],
    "confidence_threshold": 0.1,
    "iou_threshold": 0.3,
    "model_name": "yolo12n.pt",
    "export_format": "mot",
    "sahi": false
  }],
  "confirm_timeout": 30
}"""


def test_parse_tracking_step() -> None:
    """跟踪指令 → task_type=tracking：固定类别 prompts、mot 导出、模型名透传。"""
    planner = TaskPlanner(llm_client=cast(LLMClient, _FixedLLM(_TRACK_JSON)))
    plan = planner.parse("跟踪视频中的行人和车辆，使用 ByteTrack")

    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert step.task_type == "tracking"
    assert step.prompts == ["person", "car"]
    assert step.export_format == "mot"
    assert step.model_name == "yolo12n.pt"  # 跟踪器不进 model_name


def test_dict_to_plan_tracking_passthrough_and_defaults() -> None:
    """_dict_to_plan 直接解析：tracking 原样透传，缺省字段补默认值。"""
    plan = _dict_to_plan(
        {
            "steps": [{"step_id": 3, "task_type": "tracking", "source": "/d/img1"}],
            "confirm_timeout": 15,
        }
    )
    step = plan.steps[0]
    assert step.task_type == "tracking"
    assert step.confidence_threshold == 0.1  # DEFAULT_CONFIDENCE
    assert step.prompts == []
    assert plan.confirm_timeout == 15


def test_planner_system_prompt_has_tracking_rules() -> None:
    """System prompt 含 tracking 规则（防误改回退：枚举/触发词/导出/类别映射）。"""
    prompt = _PLANNER_SYSTEM_PROMPT
    assert "obb_detection|tracking" in prompt  # task_type 枚举
    assert "跟踪/追踪/track/video/视频/序列" in prompt  # 触发词规则
    assert ".mp4/.avi/.mov/.mkv" in prompt  # 视频 source
    assert "汽车/车辆→car" in prompt  # 类别映射
    assert "TRACKERS not models" in prompt  # ByteTrack/BoT-SORT 不进 model_name
    assert '"mot" if task_type=tracking' in prompt  # 导出格式
    assert "extract if user says 置信度/conf/阈值X" in prompt  # 指令级阈值提取
    assert "extract if user says iou/IoU X" in prompt
