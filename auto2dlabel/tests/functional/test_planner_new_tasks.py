"""Planner 新任务解析测试（FakeLLM 固定 JSON，零网络）。

覆盖：classification / obb_detection 步骤解析、模型名与导出格式透传、
默认值填充。
"""

from __future__ import annotations

from typing import Any, cast

from auto2dlabel.agent.llm import LLMClient, LLMResponse
from auto2dlabel.agent.planner import TaskPlanner, _dict_to_benchmark, _dict_to_plan


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


_CLS_JSON = """{
  "steps": [{
    "step_id": 1,
    "task_type": "classification",
    "source": "/data/imgs",
    "prompts": ["cat", "dog"],
    "confidence_threshold": 0.1,
    "iou_threshold": 0.3,
    "model_name": "openai/clip-vit-base-patch32",
    "export_format": "cls",
    "sahi": false
  }],
  "confirm_timeout": 30
}"""

_OBB_JSON = """{
  "steps": [{
    "step_id": 1,
    "task_type": "obb_detection",
    "source": "/data/aerial",
    "prompts": ["plane", "ship"],
    "confidence_threshold": 0.1,
    "iou_threshold": 0.3,
    "model_name": "yolo11n-obb.pt",
    "export_format": "dota",
    "sahi": false
  }],
  "confirm_timeout": 30
}"""


def test_parse_classification_step() -> None:
    """分类指令 → task_type=classification，模型/导出格式透传。"""
    planner = TaskPlanner(llm_client=cast(LLMClient, _FixedLLM(_CLS_JSON)))
    plan = planner.parse("把图片分类为猫和狗，用 clip")

    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert step.task_type == "classification"
    assert step.model_name == "openai/clip-vit-base-patch32"
    assert step.export_format == "cls"
    assert step.prompts == ["cat", "dog"]
    assert plan.raw_instruction == "把图片分类为猫和狗，用 clip"


def test_parse_obb_step() -> None:
    """旋转框指令 → task_type=obb_detection，dota 导出格式透传。"""
    planner = TaskPlanner(llm_client=cast(LLMClient, _FixedLLM(_OBB_JSON)))
    plan = planner.parse("检测航拍图中的旋转框目标，用 yolo11n-obb")

    step = plan.steps[0]
    assert step.task_type == "obb_detection"
    assert step.model_name == "yolo11n-obb.pt"
    assert step.export_format == "dota"


def test_dict_to_plan_defaults_and_new_task_types() -> None:
    """_dict_to_plan 直接解析：新 task_type 原样透传，缺省字段补默认值。"""
    plan = _dict_to_plan({
        "steps": [{"step_id": 7, "task_type": "classification", "source": "/d"}],
        "confirm_timeout": 12,
    })
    step = plan.steps[0]
    assert step.task_type == "classification"
    assert step.confidence_threshold == 0.1  # DEFAULT_CONFIDENCE
    assert plan.confirm_timeout == 12


def test_parse_empty_response_raises() -> None:
    """LLM 返回空内容 → 抛错。"""
    planner = TaskPlanner(llm_client=cast(LLMClient, _FixedLLM("")))
    try:
        planner.parse("分类")
        raise AssertionError("应抛出 ValueError")
    except ValueError as e:
        assert "空响应" in str(e)


_OBB_BENCH_JSON = """{
  "dataset": "dota_obb",
  "task_type": "obb_detection",
  "model": "yolo11n-obb.pt",
  "conf": 0.3,
  "iou": 0.5,
  "max_images": 50,
  "sahi": false
}"""


def test_parse_obb_benchmark() -> None:
    """「旋转框评测」→ dataset=dota_obb、task_type=obb_detection、OBB 权重透传。"""
    planner = TaskPlanner(llm_client=cast(LLMClient, _FixedLLM(_OBB_BENCH_JSON)))
    req = planner.parse_benchmark("在 DOTA 上评测旋转框检测")

    assert req.dataset == "dota_obb"
    assert req.task_type == "obb_detection"
    assert req.model == "yolo11n-obb.pt"


def test_dict_to_benchmark_obb_default_model() -> None:
    """dataset=dota_obb 且 LLM 未给 model → 推导 task_type + 默认 OBB 权重。"""
    req = _dict_to_benchmark({"dataset": "dota_obb"})
    assert req.task_type == "obb_detection"
    assert req.model == "yolo11n-obb.pt"  # yolo26x.pt 无 OBB 头，不得兜底


def test_dict_to_benchmark_plain_detection_keeps_default() -> None:
    """普通检测数据集未给 model → 保持 BENCHMARK_DEFAULT_MODEL。"""
    req = _dict_to_benchmark({"dataset": "voc2007"})
    assert req.task_type == "detection"
    assert req.model.endswith("yolo26x.pt")
