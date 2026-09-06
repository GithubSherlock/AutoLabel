"""任务规划器 —— 将自然语言指令解析为 TaskPlan。

使用 LLM 从用户文本中提取结构化标注参数。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from auto2dlabel.agent import json, logging
from auto2dlabel.agent.dialog import parse_with_dialog
from auto2dlabel.agent.llm import LLMClient
from auto2dlabel.agent.prompt_loader import PROFILES, load_prompt, render
from auto2dlabel.configs.datasets import format_datasets_summary
from auto2dlabel.configs.model_catalog import format_catalog_summary
from auto2dlabel.configs.task_params import format_task_params_summary
from auto2dlabel.schema.task_plan import (
    BENCHMARK_DATASETS,
    BENCHMARK_DEFAULT_CONF,
    BENCHMARK_DEFAULT_IOU,
    BENCHMARK_DEFAULT_MAX_IMAGES,
    BENCHMARK_DEFAULT_MODEL,
    BENCHMARK_DEFAULT_SEG_MODEL,
    DATASET_CN_MAP,
    DEFAULT_CONFIDENCE,
    DEFAULT_EXPORT,
    DEFAULT_IOU,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    BenchmarkRequest,
    PlanQuestion,
    TaskPlan,
    TaskStep,
    sanitize_benchmark_request,
    sanitize_task_plan,
)

logger = logging.getLogger(__name__)

# v1.0 P1+：prompt 正文外置 agent/prompts/*.md（frontmatter + $占位符注入，
# 借鉴 claude-code agents/*.md 模式）。import 期 render 保持「8 个测试直接
# import 常量断言子串」的旧语义——目录/数据集/参数摘要仍为运行时注入值。
_PLANNER_SPEC = load_prompt("planner.md")
_PLANNER_SYSTEM_PROMPT = render(
    _PLANNER_SPEC,
    catalog_summary=format_catalog_summary(),
    datasets_summary=format_datasets_summary(),
    task_params_summary=format_task_params_summary(),
)

_BENCHMARK_SPEC = load_prompt("benchmark.md")
_BENCHMARK_SYSTEM_PROMPT = render(
    _BENCHMARK_SPEC,
    default_model=BENCHMARK_DEFAULT_MODEL,
    default_seg_model=BENCHMARK_DEFAULT_SEG_MODEL,
    default_conf=BENCHMARK_DEFAULT_CONF,
    default_iou=BENCHMARK_DEFAULT_IOU,
    default_max_images=BENCHMARK_DEFAULT_MAX_IMAGES,
)


def _gpu_context_line() -> str:
    """注入 LLM 的设备信息 + batch 推荐规则（无 CUDA GPU 时返回空串）。

    让 LLM 在用户显式提到 GPU/显存相关措辞时有上下文可依；数值规则与
    schema.task_plan.recommend_batch_params 一致（代码兜底同源）。
    """
    from auto2dlabel.schema.task_plan import detect_gpu_memory_gb, recommend_batch_params

    mem = detect_gpu_memory_gb()
    if mem is None:
        return ""
    try:
        import torch

        name = torch.cuda.get_device_name(0)
    except ImportError:
        name = "CUDA GPU"

    lines = [f"Device context: GPU={name}, VRAM={mem}GB"]
    for task_type, label in [
        ("object_detection", "object_detection/obb_detection"),
        ("instance_segmentation", "instance_segmentation"),
        ("classification", "classification"),
    ]:
        bs, nw = recommend_batch_params(task_type, mem)
        lines.append(f"  recommended for {label}: batch_size={bs}, num_workers={nw}")
    return "\n".join(lines)


class TaskPlanner:
    """任务规划器 —— NL → TaskPlan / BenchmarkRequest。"""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def parse(self, instruction: str, confirm_timeout: int = DEFAULT_TIMEOUT) -> TaskPlan:
        """将自然语言指令解析为 TaskPlan。

        Args:
            instruction: 用户自然语言指令。
            confirm_timeout: 确认等待秒数（可通过 --no-wait 设为 0）。

        Returns:
            TaskPlan。
        """

        messages: list[dict[str, str]] = [
            {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": _gpu_context_line() + "\n\n" + instruction},
        ]

        response = self.llm.chat(
            messages,
            tools=None,
            **PROFILES[_PLANNER_SPEC.profile],  # planning 档（声明式，改档只动 prompt_loader）
            call_site="planner.parse",
        )

        if not response.content:
            raise ValueError("LLM 返回空响应，无法解析任务")

        plan = _parse_json_response(response.content)
        plan.raw_instruction = instruction

        for fix in sanitize_task_plan(plan):  # LLM 输出守卫（路径数字误提等）
            logger.info("参数守卫修正: %s", fix)

        _apply_confirm_timeout(plan, confirm_timeout)

        logger.info("Parsed plan: %d steps", len(plan.steps))
        for s in plan.steps:
            logger.info("  %s", s.summary)

        return plan

    def parse_dialog(
        self,
        instruction: str,
        ask_fn: Callable[[list[PlanQuestion]], str],
        confirm_timeout: int = DEFAULT_TIMEOUT,
        max_rounds: int = 3,
        on_delta: Callable[[str], None] | None = None,
    ) -> TaskPlan:
        """多轮对话解析（v0.6 P1）：缺参时 LLM 出 questions → 收集回答 → 回喂。

        复用 parse_with_dialog 通用骨架（agent/dialog.py，schema 无关）；
        system prompt 与 parse 同源（catalog 摘要 + GPU 上下文，设备信息放
        system 角色——与 parse 的 user 角色差异无语义影响）。
        轮次耗尽 / 用户放弃 → 返回缺参 plan（调用方 _fill_missing_params 兜底）。
        LLM 响应非法 / 空响应 → ValueError 传播（调用方降级链处理）。
        on_delta: LLM 流式增量回调（v1.0 P1；None = 非流式）。
        """
        system_prompt = _PLANNER_SYSTEM_PROMPT + "\n\n" + _gpu_context_line()
        plan = parse_with_dialog(
            llm=self.llm,
            system_prompt=system_prompt,
            parse_fn=_parse_json_response,
            ask_fn=ask_fn,
            instruction=instruction,
            max_rounds=max_rounds,
            call_site="planner.dialog",
            on_delta=on_delta,
        )
        for fix in sanitize_task_plan(plan):  # LLM 输出守卫（对话路径同源）
            logger.info("参数守卫修正: %s", fix)
        _apply_confirm_timeout(plan, confirm_timeout)
        logger.info("Parsed plan via dialog: %d steps", len(plan.steps))
        return plan

    def parse_benchmark(self, instruction: str) -> BenchmarkRequest:
        """将自然语言指令解析为 BenchmarkRequest。

        Args:
            instruction: 用户自然语言指令（如 "用 yolo26x 跑 COCO 检测，50 张图 conf=0.3"）。

        Returns:
            BenchmarkRequest。
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _BENCHMARK_SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]

        response = self.llm.chat(
            messages,
            tools=None,
            **PROFILES[_BENCHMARK_SPEC.profile],  # planning 档（与 parse 同档）
            call_site="planner.benchmark",
        )

        if not response.content:
            raise ValueError("LLM 返回空响应，无法解析 benchmark 参数")

        request = _parse_benchmark_json(response.content)
        for fix in sanitize_benchmark_request(request, instruction):
            logger.info("benchmark 参数守卫修正: %s", fix)
        logger.info("Parsed benchmark: %s", request.summary)
        return request


def _apply_confirm_timeout(plan: TaskPlan, confirm_timeout: int) -> None:
    """应用 CLI 指定 timeout（仅当 LLM 未显式给出时；v0.5 语义原样，parse/parse_dialog 共用）。"""
    if plan.confirm_timeout == DEFAULT_TIMEOUT:
        plan.confirm_timeout = confirm_timeout


def _parse_json_response(content: str) -> TaskPlan:
    """从 LLM 响应文本中提取 JSON 并解析为 TaskPlan。"""
    # 尝试直接解析
    try:
        data = json.loads(content)
        return _dict_to_plan(data)
    except json.JSONDecodeError:
        pass

    # 尝试从 markdown 代码块中提取
    import re
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
    if m:
        try:
            data = json.loads(m.group(1))
            return _dict_to_plan(data)
        except json.JSONDecodeError:
            pass

    # 最后尝试找 { ... } 块
    m = re.search(r"\{[\s\S]*\}", content)
    if m:
        try:
            data = json.loads(m.group(0))
            return _dict_to_plan(data)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"无法从 LLM 响应中解析 TaskPlan JSON: {content[:200]}...")


def _dict_to_plan(data: dict[str, Any]) -> TaskPlan:
    """将 LLM 返回的 dict 转换为 TaskPlan，补充默认值。"""
    steps_data = data.get("steps", [data] if "step_id" in data else [])
    steps: list[TaskStep] = []

    for i, sd in enumerate(steps_data):
        step = TaskStep(
            step_id=sd.get("step_id", i + 1),
            task_type=sd.get("task_type", "object_detection"),
            source=sd.get("source", ""),
            prompts=sd.get("prompts", []),
            confidence_threshold=sd.get("confidence_threshold", DEFAULT_CONFIDENCE),
            iou_threshold=sd.get("iou_threshold", DEFAULT_IOU),
            model_name=sd.get("model_name", DEFAULT_MODEL),
            model_hint=sd.get("model_hint", ""),
            export_format=sd.get("export_format", DEFAULT_EXPORT),
            sahi=sd.get("sahi", False),
            num_workers=sd.get("num_workers"),
            batch_size=sd.get("batch_size"),
        )
        steps.append(step)

    if not steps:
        raise ValueError("TaskPlan 至少需要一个步骤")

    return TaskPlan(
        steps=steps,
        confirm_timeout=data.get("confirm_timeout", DEFAULT_TIMEOUT),
        # v0.6 对话问题（致命点：LLM JSON → plan 走本函数，不经过 TaskPlan.from_dict）
        questions=PlanQuestion.from_list(data.get("questions")),
    )


def _parse_benchmark_json(content: str) -> BenchmarkRequest:
    """从 LLM 响应文本中提取 JSON 并解析为 BenchmarkRequest。"""
    # 尝试直接解析
    try:
        data = json.loads(content)
        return _dict_to_benchmark(data)
    except json.JSONDecodeError:
        pass

    # 尝试从 markdown 代码块中提取
    import re
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
    if m:
        try:
            data = json.loads(m.group(1))
            return _dict_to_benchmark(data)
        except json.JSONDecodeError:
            pass

    # 最后尝试找 { ... } 块
    m = re.search(r"\{[\s\S]*\}", content)
    if m:
        try:
            data = json.loads(m.group(0))
            return _dict_to_benchmark(data)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"无法从 LLM 响应中解析 BenchmarkRequest JSON: {content[:200]}...")


def _dict_to_benchmark(data: dict[str, Any]) -> BenchmarkRequest:
    """将 LLM 返回的 dict 转换为 BenchmarkRequest，补充默认值并验证。"""
    dataset = data.get("dataset", "").strip().lower()
    assert isinstance(dataset, str)  # LLM 返回 JSON 中 dataset 字段应为字符串

    # 中文名 → key 映射
    if dataset and dataset not in BENCHMARK_DATASETS:
        dataset = DATASET_CN_MAP.get(dataset, dataset)

    # 验证数据集名
    if dataset and dataset not in BENCHMARK_DATASETS:
        logger.warning("Unknown dataset '%s', using as-is", dataset)

    task_type = data.get("task_type", "detection")
    # 从数据集推导 task_type
    if dataset in BENCHMARK_DATASETS:
        task_type = BENCHMARK_DATASETS[dataset]["task_type"]

    # OBB 任务默认旋转框权重（yolo26x.pt 无 OBB 头）
    model = data.get("model", "")
    if not model:
        model = "yolo11n-obb.pt" if task_type == "obb_detection" else BENCHMARK_DEFAULT_MODEL

    return BenchmarkRequest(
        dataset=dataset,
        task_type=task_type,
        model=model,
        seg_model=data.get("seg_model", BENCHMARK_DEFAULT_SEG_MODEL),
        conf=float(data.get("conf", BENCHMARK_DEFAULT_CONF)),
        iou=float(data.get("iou", BENCHMARK_DEFAULT_IOU)),
        max_images=int(data.get("max_images", BENCHMARK_DEFAULT_MAX_IMAGES)),
        top_classes=int(data.get("top_classes", 20)),
        sahi=bool(data.get("sahi", False)),
    )
