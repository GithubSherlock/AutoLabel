"""任务参数规格表 + 代码级守卫测试（configs/task_params.py 单一事实源，2026-08-29）。

覆盖：规格表覆盖 TaskStep 全部可提取字段（不含结构/展示字段）、required 派生
REQUIRED_PARAMS、DEFAULT_* 由表派生（改表不漏常量）、missing 判定表驱动
（文案不变）、摘要注入 planner prompt；守卫层：类型强转/值域/指令文本
代码级提取（实测洞「COCO2017」→ batch_size=2017）、planner parse 钩子。
"""

from __future__ import annotations

import json
from dataclasses import fields
from typing import Any, cast

from auto2dlabel.agent.llm import LLMClient, LLMResponse
from auto2dlabel.agent.planner import TaskPlanner
from auto2dlabel.configs.task_params import (
    TASK_PARAM_SPECS,
    coerce_bool,
    coerce_float,
    coerce_int,
    coerce_strs,
    format_task_params_summary,
    guard_batch_size,
    guard_num_workers,
)
from auto2dlabel.schema.task_plan import (
    BENCHMARK_DEFAULT_CONF,
    BENCHMARK_DEFAULT_IOU,
    BENCHMARK_DEFAULT_MAX_IMAGES,
    BENCHMARK_DEFAULT_MODEL,
    DEFAULT_CONFIDENCE,
    DEFAULT_EXPORT,
    DEFAULT_IOU,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    REQUIRED_PARAMS,
    BenchmarkRequest,
    TaskPlan,
    TaskStep,
    sanitize_benchmark_request,
    sanitize_task_plan,
)


class _FakeLLM:
    """固定内容单次响应（零网络零模型）。"""

    model = "fake-llm"
    has_credentials = True

    def __init__(self, content: str) -> None:
        self.content = content

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.1,
    ) -> LLMResponse:
        return LLMResponse(content=self.content, tool_calls=None)

# 用户可指定的任务参数字段（结构字段 step_id/task_type 与展示字段 model_hint 除外）
_EXTRACTABLE_FIELDS = {
    "source",
    "prompts",
    "confidence_threshold",
    "iou_threshold",
    "model_name",
    "export_format",
    "sahi",
    "num_workers",
    "batch_size",
}


def test_specs_cover_all_extractable_fields() -> None:
    """规格表恰好覆盖全部可提取参数字段；key 均为 TaskStep 真实字段。"""
    field_names = {f.name for f in fields(TaskStep)}
    assert set(TASK_PARAM_SPECS) == _EXTRACTABLE_FIELDS
    assert set(TASK_PARAM_SPECS) <= field_names
    # 结构/展示字段不进表（不是用户可指定参数）
    assert "model_hint" not in TASK_PARAM_SPECS
    assert "step_id" not in TASK_PARAM_SPECS
    assert "task_type" not in TASK_PARAM_SPECS


def test_required_params_derived_from_specs() -> None:
    """必填集合由规格表派生（source/prompts 为单一事实源）。"""
    required = {s.key for s in TASK_PARAM_SPECS.values() if s.required}
    assert REQUIRED_PARAMS == {"source", "prompts"} == required


def test_taskstep_defaults_match_specs() -> None:
    """TaskStep 实例默认值 == 规格表 default（派生零漂移，防手改发散）。"""
    step = TaskStep(step_id=1)
    for key, spec in TASK_PARAM_SPECS.items():
        if spec.default is None:
            assert getattr(step, key) is None
        else:
            assert getattr(step, key) == spec.default, key


def test_derived_default_constants() -> None:
    """DEFAULT_* 常量由规格表派生（改表不漏常量）。"""
    assert DEFAULT_CONFIDENCE == 0.1
    assert DEFAULT_IOU == 0.3
    assert DEFAULT_MODEL == "yolo26x.pt"
    assert DEFAULT_EXPORT == "coco"


def test_missing_params_table_driven() -> None:
    """missing 判定表驱动：required 字段值 falsy 即缺失（文案保持 v0.5 不变）。"""
    assert TaskStep(step_id=1).missing_params == [
        "source（数据路径）",
        "prompts（检测类别）",
    ]
    step = TaskStep(step_id=1, source="/data/img", prompts=[])
    assert step.missing_params == ["prompts（检测类别）"]
    step = TaskStep(step_id=1, source="/data/img", prompts=["car"])
    assert step.missing_params == []
    assert step.is_complete


def test_optional_params_never_missing() -> None:
    """可选参数保持默认（None/默认模型）不触发缺参（required 判定不受影响）。"""
    step = TaskStep(step_id=1, source="/data/img", prompts=["car"])
    assert step.num_workers is None and step.batch_size is None
    assert step.missing_params == []


def test_optional_missing_asks_model_and_thresholds() -> None:
    """追问扩展：缺模型选择/缺超参数（值为默认值=未指定）→ optional_missing。"""
    step = TaskStep(step_id=1, source="/data/img", prompts=["car"])
    missing = step.optional_missing()
    assert missing == [
        "confidence_threshold（置信度）（默认 0.1）",
        "iou_threshold（IoU）（默认 0.3）",
        "model_name（模型）（默认 yolo26x.pt）",
    ]
    # 指定后不再追问
    step.confidence_threshold = 0.5
    step.model_name = "fasterrcnn_resnet50_fpn_v2"
    assert step.optional_missing() == ["iou_threshold（IoU）（默认 0.3）"]
    step.iou_threshold = 0.4
    assert step.optional_missing() == []


def test_optional_missing_excludes_non_ask() -> None:
    """非 ask 字段（格式/切片/批参数）缺失不进入追问链（默认合理/独立交互）。"""
    step = TaskStep(step_id=1, source="/data/img", prompts=["car"])
    assert step.export_format == "coco" and step.sahi is False
    assert step.batch_size is None and step.num_workers is None
    texts = "\n".join(step.optional_missing())
    assert "export_format" not in texts and "sahi" not in texts
    assert "num_workers" not in texts and "batch_size" not in texts


def test_plan_all_optional_missing() -> None:
    """all_optional_missing 按 step_id 汇总（已指定非默认值的步骤不出现）。

    注：显式指定默认值（如 model_name="yolo26x.pt"）无法与「未指定」区分
    （值==默认即视为未指定）——宽判定、易跳过，回车即可。
    """
    plan = TaskPlan(
        steps=[
            TaskStep(step_id=1, source="/data", prompts=["car"]),  # 缺模型/阈值
            TaskStep(
                step_id=2, source="/data2", prompts=["person"],
                confidence_threshold=0.5, iou_threshold=0.4,
                model_name="fasterrcnn_resnet50_fpn_v2",
            ),  # 全指定（非默认）
        ]
    )
    assert plan.all_optional_missing() == {1: plan.steps[0].optional_missing()}


def test_guard_explicit_params() -> None:
    """指令显式提及判定（零 LLM 正则）：提及的模型/阈值 → key 集合。"""
    from auto2dlabel.configs.task_params import guard_explicit_params

    assert guard_explicit_params(
        "检测 dir 中的汽车，模型用 yolo26x.pt，置信度 0.5，IoU 0.4"
    ) == {"model_name", "confidence_threshold", "iou_threshold"}
    assert guard_explicit_params("检测 dir 中的汽车和行人，使用 fasterrcnn") == {
        "model_name"
    }
    assert guard_explicit_params("检测 dir 中的汽车") == set()
    assert guard_explicit_params("") == set()
    # 「用 COCO2017 里的图」宽匹配为模型提及——只损失一次询问机会
    # （不追问 → 默认执行，与回车跳过同语义），无害
    assert "model_name" in guard_explicit_params("检测用 COCO2017 里的图")
    # 2026-08-29 实测确认编辑措辞：「模型采用 sam3」「置信度改为 0.3」
    assert "model_name" in guard_explicit_params("改为实例分割任务, 模型采用 sam3")
    assert "confidence_threshold" in guard_explicit_params("置信度改为 0.3")
    assert "iou_threshold" in guard_explicit_params("iou 设置为 0.3")
    # 「confidence 和 iou 都设置为 0.3」——关键词后跟「和」不是数值，
    # 不算显式（值从 LLM 重解析拿，carry-over 兜底）
    assert guard_explicit_params("confidence 和 iou 都设置为 0.3") == set()


def test_optional_missing_explicit_override() -> None:
    """显式提及的可选参数即使值==默认也不追问（组 1 场景：模型用默认模型）。"""
    step = TaskStep(step_id=1, source="/data/img", prompts=["car"])
    assert step.optional_missing(explicit={"model_name"}) == [
        "confidence_threshold（置信度）（默认 0.1）",
        "iou_threshold（IoU）（默认 0.3）",
    ]
    assert step.optional_missing() == [
        "confidence_threshold（置信度）（默认 0.1）",
        "iou_threshold（IoU）（默认 0.3）",
        "model_name（模型）（默认 yolo26x.pt）",
    ]


def test_plan_all_missing_params_only_incomplete() -> None:
    """all_missing_params 只含不完整步骤（完整步骤不出现）。"""
    plan = TaskPlan(
        steps=[
            TaskStep(step_id=1, source="", prompts=["car"]),
            TaskStep(step_id=2, source="/data", prompts=["person"]),
        ]
    )
    assert plan.all_missing_params == {1: ["source（数据路径）"]}
    assert plan.first_incomplete is plan.steps[0]


def test_format_task_params_summary() -> None:
    """摘要含字段清单 + REQUIRED 标记 + 默认值（注入 planner prompt 用）。"""
    summary = format_task_params_summary()
    assert "Task parameters" in summary
    assert "source: REQUIRED" in summary
    assert "prompts: REQUIRED" in summary
    assert "confidence_threshold: 0.1" in summary
    assert "model_name: 'yolo26x.pt'" in summary
    assert "batch_size: None" in summary
    assert "model_hint" not in summary  # 展示字段不注入


def test_planner_prompt_injects_param_specs() -> None:
    """planner 系统提示注入规格摘要（LLM 拿到规范字段全集）。"""
    from auto2dlabel.agent.planner import _PLANNER_SYSTEM_PROMPT

    assert "Task parameters" in _PLANNER_SYSTEM_PROMPT
    assert "source: REQUIRED" in _PLANNER_SYSTEM_PROMPT
    assert "batch_size: None" in _PLANNER_SYSTEM_PROMPT
    assert "不要臆造路径/类别" in _PLANNER_SYSTEM_PROMPT
    # 路径数字不提取规则（与代码守卫双保险）
    assert "NEVER hyperparameters" in _PLANNER_SYSTEM_PROMPT


# ================================================================
# 代码级守卫（2026-08-29 防御纵深：LLM 输出不可全信）
# ================================================================


def test_coerce_primitives() -> None:
    """类型强转守卫：垃圾值回默认/None，绝不传播进执行层。"""
    assert coerce_float("abc", 0.1) == 0.1
    assert coerce_float(2017, 0.1) == 0.1  # 路径数字误提
    assert coerce_float("0.5", 0.1) == 0.5
    assert coerce_float(True, 0.1) == 0.1  # JSON true 不是 1.0
    assert coerce_float(0.0, 0.1) == 0.1  # 0 不入 (0,1]
    assert coerce_int(True) is None
    assert coerce_int("abc") is None
    assert coerce_int(8.0) == 8
    assert coerce_bool("false") is False  # Python bool("false")=True 的坑
    assert coerce_bool("true") is True
    assert coerce_bool(0) is False
    assert coerce_strs("car") == ["car"]
    assert coerce_strs([" car ", 2017, ""]) == ["car", "2017"]
    assert coerce_strs(None) == []


def test_guard_batch_size() -> None:
    """batch 守卫：指令文本代码级提取为准，LLM 值不采信。"""
    assert guard_batch_size("检测 COCO2017 中的汽车") is None  # 实测洞
    assert guard_batch_size("检测 dir 批量4 中的汽车") == 4
    assert guard_batch_size("检测 dir batch_size=16 中的汽车") == 16
    assert guard_batch_size("批量处理时跑最大") is None  # 无数字 → 自动实测
    assert guard_batch_size("批量1000") is None  # 超上限 → None
    assert guard_batch_size("检测 dir 中的汽车") is None  # 无表述


def test_guard_num_workers() -> None:
    """workers 守卫：同上；超 CPU 核数钳位（防 fork 炸弹）。"""
    import os

    assert guard_num_workers("检测 COCO2017 中的汽车") is None
    assert guard_num_workers("num_workers=4") == 4
    assert guard_num_workers("进程8") == 8
    assert guard_num_workers("num_workers=2017") == min(2017, os.cpu_count() or 4)


def test_sanitize_task_plan_garbage() -> None:
    """sanitize 全字段：垃圾值全部归位 + 修正记录可见。"""
    plan = TaskPlan(
        steps=[
            TaskStep(
                step_id=1,
                source=" /x ",
                prompts=cast(list[str], "car"),  # 单字符串 → 列表（防逐字符 join）
                confidence_threshold=2017,
                iou_threshold=cast(float, "abc"),
                batch_size=2017,
                num_workers=2017,
                sahi=cast(bool, "false"),  # 字符串 truthy 坑
                model_name="",
                export_format="garbage",
                task_type="garbage",
            )
        ],
        confirm_timeout=cast(int, "abc"),  # thread.join TypeError 隐患
    )
    plan.raw_instruction = "检测 COCO2017 中的汽车"
    fixes = sanitize_task_plan(plan)
    step = plan.steps[0]
    assert step.source == "/x"
    assert step.prompts == ["car"]
    assert step.confidence_threshold == DEFAULT_CONFIDENCE
    assert step.iou_threshold == DEFAULT_IOU
    assert step.batch_size is None
    assert step.num_workers is None
    assert step.sahi is False
    assert step.model_name == DEFAULT_MODEL
    assert step.export_format == DEFAULT_EXPORT
    assert step.task_type == "object_detection"
    assert plan.confirm_timeout == DEFAULT_TIMEOUT
    assert fixes  # 守卫触发必须可见（日志观测）


def test_sanitize_task_plan_keeps_valid() -> None:
    """sanitize 合法值零修正（零漂移）。"""
    plan = TaskPlan(
        steps=[
            TaskStep(
                step_id=1,
                source="/data",
                prompts=["car"],
                confidence_threshold=0.5,
                iou_threshold=0.4,
                batch_size=8,
                num_workers=4,
                sahi=True,
                export_format="yolo_obb",
                task_type="obb_detection",
            )
        ],
        confirm_timeout=30,
    )
    plan.raw_instruction = "检测 dir 批量8 num_workers=4 conf 0.5 中的汽车"
    assert sanitize_task_plan(plan) == []
    step = plan.steps[0]
    assert step.batch_size == 8 and step.num_workers == 4
    assert step.confidence_threshold == 0.5
    assert step.export_format == "yolo_obb"


def test_sanitize_normalizes_yolo_obb_dash() -> None:
    """yolo-obb（LLM 连字符写法）→ yolo_obb（注册表键）。"""
    plan = TaskPlan(
        steps=[TaskStep(step_id=1, source="/d", prompts=["car"], export_format="yolo-obb")]
    )
    plan.raw_instruction = "检测 dir 中的汽车"
    sanitize_task_plan(plan)
    assert plan.steps[0].export_format == "yolo_obb"


def test_planner_parse_sanitizes_garbage() -> None:
    """parse 钩子：LLM 输出垃圾 → 守卫归位（实测洞端到端）。"""
    content = json.dumps({
        "steps": [{
            "step_id": 1,
            "task_type": "object_detection",
            "source": "/data/COCO2017",
            "prompts": ["car"],
            "batch_size": 2017,
            "num_workers": 2017,
            "confidence_threshold": 2017,
        }]
    })
    planner = TaskPlanner(llm_client=cast(LLMClient, _FakeLLM(content)))
    plan = planner.parse("检测 COCO2017 中的汽车")
    step = plan.steps[0]
    assert step.batch_size is None
    assert step.num_workers is None
    assert step.confidence_threshold == DEFAULT_CONFIDENCE


def test_planner_parse_keeps_explicit_hyperparams() -> None:
    """parse 钩子：指令显式超参以指令文本为准（LLM 值不采信）。"""
    content = json.dumps({
        "steps": [{
            "step_id": 1,
            "task_type": "object_detection",
            "source": "/data",
            "prompts": ["car"],
            "batch_size": 2017,  # LLM 误提
            "confidence_threshold": 0.5,
        }]
    })
    planner = TaskPlanner(llm_client=cast(LLMClient, _FakeLLM(content)))
    plan = planner.parse("检测 /data 批量8 conf 0.5 中的汽车")
    step = plan.steps[0]
    assert step.batch_size == 8  # 指令文本权威
    assert step.confidence_threshold == 0.5


# ================================================================
# Benchmark 守卫
# ================================================================


def test_sanitize_benchmark_request_garbage() -> None:
    """benchmark 守卫：未知数据集/值域违规/垃圾类型全归位。"""
    req = BenchmarkRequest(
        dataset="unknown_ds",
        conf=2017,
        iou=cast(float, "abc"),
        max_images=2017,
        top_classes=cast(int, "abc"),
        sahi=cast(bool, "false"),
        model="",
    )
    fixes = sanitize_benchmark_request(req, "跑 COCO2017 检测")
    assert req.dataset == ""
    assert req.conf == BENCHMARK_DEFAULT_CONF
    assert req.iou == BENCHMARK_DEFAULT_IOU
    assert req.max_images == BENCHMARK_DEFAULT_MAX_IMAGES  # 无显式图片数 → 默认
    assert req.top_classes == 20
    assert req.sahi is False
    assert req.model == BENCHMARK_DEFAULT_MODEL
    assert fixes


def test_guard_max_images_explicit() -> None:
    """max_images 以指令文本为准：N张 → N；全部 → 0；无表述 → 默认。"""
    req = BenchmarkRequest(dataset="coco", max_images=2017)
    sanitize_benchmark_request(req, "跑 COCO 检测 200张")
    assert req.max_images == 200
    req2 = BenchmarkRequest(dataset="coco", max_images=2017)
    sanitize_benchmark_request(req2, "跑全部 COCO 检测")
    assert req2.max_images == 0
    req3 = BenchmarkRequest(dataset="coco", max_images=2017)
    sanitize_benchmark_request(req3, "跑 COCO2017 检测")
    assert req3.max_images == BENCHMARK_DEFAULT_MAX_IMAGES
