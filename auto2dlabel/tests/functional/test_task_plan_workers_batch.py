"""批量推理超参数（num_workers / batch_size）测试。

覆盖 chat 全链的参数载体与推荐/解析纯函数（无 GPU / 无 LLM 依赖）：
- TaskStep 新字段默认 None + to_dict/from_dict 往返 + summary 展示
- recommend_batch_params 显存三档推荐（LLM prompt 与代码兜底共用）
- detect_gpu_memory_gb 无 CUDA 返回 None
- _parse_batch_input 各输入格式（= / 中文冒号 / 裸数字 / 非法）
"""

from __future__ import annotations

import pytest

from auto2dlabel.cli_commands import _is_auto_batch_input, _parse_batch_input
from auto2dlabel.schema.task_plan import (
    TaskStep,
    detect_gpu_memory_gb,
    recommend_batch_params,
)


def _make_step(**kwargs: object) -> TaskStep:
    defaults: dict[str, object] = {
        "step_id": 1,
        "task_type": "object_detection",
        "source": "/data/images",
        "prompts": ["car"],
    }
    defaults.update(kwargs)
    return TaskStep(**defaults)  # type: ignore[arg-type]


def test_task_step_defaults_none() -> None:
    """新字段默认 None（未指定时不干扰 planner/执行路径）。"""
    step = _make_step()
    assert step.num_workers is None
    assert step.batch_size is None


def test_task_step_dict_roundtrip() -> None:
    """to_dict/from_dict 往返保留 num_workers/batch_size。"""
    step = _make_step(num_workers=4, batch_size=8)
    restored = TaskStep.from_dict(step.to_dict())
    assert restored.num_workers == 4
    assert restored.batch_size == 8


def test_task_step_dict_roundtrip_none() -> None:
    """默认 None 往返后仍为 None（from_dict 用 d.get 缺省）。"""
    step = _make_step()
    restored = TaskStep.from_dict(step.to_dict())
    assert restored.num_workers is None
    assert restored.batch_size is None


def test_task_step_summary_shows_batch() -> None:
    """summary 展示批量参数；未指定时不出现 batch 字样。"""
    step = _make_step(batch_size=8, num_workers=4)
    assert "batch=8/4w" in step.summary
    assert "batch=" not in _make_step().summary


def test_task_step_summary_rich_safe() -> None:
    """回归（2026-08-29）：summary 经 console.print 渲染后 prompts 可见。

    半角方括号 [car, person] 会被 Rich 当无效 markup 标签吞掉（实测终端
    「参数已更新」行类别显示为空），故 summary 用全角括号。
    """
    from rich.console import Console

    step = _make_step(prompts=["car", "person"])
    console = Console(width=200)
    with console.capture() as capture:
        console.print(f"[dim]参数已更新: {step.summary}[/dim]")
    rendered = capture.get()
    assert "car, person" in rendered  # 曾为空（[car, person] 被 Rich 吞掉）


def test_recommend_no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """无 GPU → 逐图（batch_size=1）；workers 按 CPU 公式（2 核 → 0）。"""
    import os

    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    assert recommend_batch_params("object_detection", None) == (1, 0)


@pytest.mark.parametrize(
    ("task_type", "expected"),
    [
        ("object_detection", (8, 4)),
        ("obb_detection", (8, 4)),
        ("instance_segmentation", (4, 4)),
        ("semantic_segmentation", (4, 4)),
        ("classification", (16, 4)),
        ("image_classification", (16, 4)),
    ],
)
def test_recommend_large_gpu(
    task_type: str, expected: tuple[int, int], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """≥20GB（4090 档）按任务类型推荐；workers 按 CPU 公式（16 核 → 4）。"""
    import os

    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    assert recommend_batch_params(task_type, 24) == expected


def test_recommend_mid_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """10-20GB 档（batch 减半）；workers 按 CPU 公式（8 核 → 2）。"""
    import os

    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    assert recommend_batch_params("object_detection", 16) == (4, 2)
    assert recommend_batch_params("instance_segmentation", 12) == (2, 2)
    assert recommend_batch_params("classification", 10) == (8, 2)


def test_recommend_small_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """<10GB 档（batch 再减半）；workers 按 CPU 公式（2 核 → 0）。"""
    import os

    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    assert recommend_batch_params("object_detection", 8) == (2, 0)
    assert recommend_batch_params("instance_segmentation", 6) == (1, 0)
    assert recommend_batch_params("classification", 4) == (4, 0)


def test_detect_gpu_memory_none_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """无 CUDA → None（推荐回退逐图）。"""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert detect_gpu_memory_gb() is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("batch_size=8 num_workers=4", (8, 4)),
        ("batch_size:8, num_workers:4", (8, 4)),
        ("batch_size：8 num_workers：4", (8, 4)),   # 中文冒号
        ("8 4", (8, 4)),
        ("8 4 还有别的数字 99", (8, 4)),             # 取前两个数字
    ],
)
def test_parse_batch_input_valid(text: str, expected: tuple[int, int]) -> None:
    assert _parse_batch_input(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",                                    # 空
        "batch_size=8",                        # 只给半个
        "num_workers=4",                       # 只给半个
        "abc",                                 # 无数字
        "0 4",                                 # batch_size < 1
        "batch_size=x num_workers=4",          # 非数字
    ],
)
def test_parse_batch_input_invalid(text: str) -> None:
    assert _parse_batch_input(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "跑最大",
        "批量处理时跑最大",
        "最大",
        "用最大批量",
        "自动",
        "尽可能大",
        "拉满",
        "跑满",
        "max",
        "MAX",
        "auto",
    ],
)
def test_is_auto_batch_input_recognizes(text: str) -> None:
    """回归：自然语言「跑最大」→ 自动实测最大（曾被判格式无效白问一轮，2026-08-29）。"""
    assert _is_auto_batch_input(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "",           # 空
        "批量8",       # 带数字 = 显式指定，不是自动
        "batch_size=8 num_workers=4",
        "abc",        # 无关键词
        "   ",        # 纯空白
    ],
)
def test_is_auto_batch_input_rejects(text: str) -> None:
    assert _is_auto_batch_input(text) is False


def test_planner_batch_rule_recognizes_auto_max() -> None:
    """planner 规则明确「跑最大」→ batch_size=null（执行阶段自动实测最大 batch）。

    2026-08-29 实测：LLM 对「批量处理时跑最大」无规则可依，null 语义不被
    理解——规则现在写死「跑最大/最大批量/自动 → null = 自动实测最大（非 1）」。
    """
    from auto2dlabel.agent.planner import _PLANNER_SYSTEM_PROMPT

    assert "跑最大" in _PLANNER_SYSTEM_PROMPT
    assert "自动实测最大" in _PLANNER_SYSTEM_PROMPT
    assert "不是 1" in _PLANNER_SYSTEM_PROMPT
