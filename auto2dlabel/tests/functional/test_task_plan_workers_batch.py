"""批量推理超参数（num_workers / batch_size）测试。

覆盖 chat 全链的参数载体与推荐/解析纯函数（无 GPU / 无 LLM 依赖）：
- TaskStep 新字段默认 None + to_dict/from_dict 往返 + summary 展示
- recommend_batch_params 显存三档推荐（LLM prompt 与代码兜底共用）
- detect_gpu_memory_gb 无 CUDA 返回 None
- _parse_batch_input 各输入格式（= / 中文冒号 / 裸数字 / 非法）
"""

from __future__ import annotations

import pytest

from auto2dlabel.cli_commands import _parse_batch_input
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
