"""cli_execute 分割步测试 —— 零模型零权重。

2026-08-29 回归：SAM3 等自带检测分割模型在 prompts 为空时曾静默跳过
（SAM3Model.generate 空 prompt_bboxes → return []），导致「未指定类别」
的指令输出全零——修复为调用层黄字警告 + 跳过，不再静默。

另含（2026-08-29）：单图 OOM（大模型加载装不下，实测 SAM3 + 残留进程占显存）
→ 跳图继续不崩整批 + nvidia-smi 指引。
"""

from __future__ import annotations

from typing import Any

import pytest

from auto2dlabel.agent.planner import _dict_to_plan
from auto2dlabel.cli_execute import _segmentation_step, execute_plan
from auto2dlabel.schema.annotation import Annotation
from auto2dlabel.tests import Path


class _FakeSam3:
    """记录 generate 是否被调用的假 SAM3 模型。"""

    def __init__(self) -> None:
        self.called = False

    def generate(self, image_path: str, bboxes: list[Any]) -> list[Any]:
        self.called = True
        return []


def _seg_step(prompts: list[str]) -> Any:
    """构造 instance_segmentation 单步 plan 的 step。"""
    return _dict_to_plan(
        {
            "steps": [
                {
                    "step_id": 1,
                    "task_type": "instance_segmentation",
                    "source": "/nonexistent/dir",
                    "prompts": prompts,
                    "confidence_threshold": 0.5,
                    "iou_threshold": 0.5,
                    "model_name": "sam3.pt",
                    "export_format": "coco",
                }
            ],
            "confirm_timeout": 30,
        }
    ).steps[0]


def test_segmentation_step_self_detect_with_prompts_calls_generate() -> None:
    """prompts 非空 → SAM3 generate 被调用（正常路径）。"""
    step = _seg_step(["car", "person"])
    ann = Annotation(image_path="x.png")
    fake = _FakeSam3()

    _segmentation_step(step, Path("x.png"), ann, "sam3.pt", fake)

    assert fake.called is True


def test_segmentation_step_self_detect_empty_prompts_skips() -> None:
    """回归：prompts 为空 → 不调用 generate（SAM3 文本驱动无类别无法检测）。"""
    step = _seg_step([])
    ann = Annotation(image_path="x.png")
    fake = _FakeSam3()

    _segmentation_step(step, Path("x.png"), ann, "sam3.pt", fake)

    assert fake.called is False
    assert ann.bboxes == [] and ann.masks == []  # 无静默污染输出


def test_execute_plan_single_image_oom_skips_not_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """单图 OOM 回归（实测：残留进程占显存 → SAM3 加载 OOM 崩整批）。

    批量 OOM → 降级逐图 → 逐图仍 OOM → 跳图 + nvidia-smi 指引 + 继续，
    全程不抛异常（其余图像照常尝试）。
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("auto2dlabel.tools.log.log_chat_call", lambda **kw: None)
    for name in ("a.png", "b.png", "c.png"):
        (tmp_path / name).write_bytes(b"")

    monkeypatch.setattr(
        "auto2dlabel.models.segmentation.create_segmentation_model",
        lambda name: _FakeSam3(),
    )
    monkeypatch.setattr(
        "auto2dlabel.cli_execute._resolve_step_batch",
        lambda *a, **kw: (2, 0),  # 批 2 → 先批量 OOM → 降级逐图
    )
    calls: list[list[Path]] = []

    def _boom_chunk(step: Any, chunk: list[Path], *a: Any, **kw: Any) -> None:
        calls.append(chunk)
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr("auto2dlabel.cli_execute._execute_chunk", _boom_chunk)

    plan = _dict_to_plan(
        {
            "steps": [
                {
                    "step_id": 1,
                    "task_type": "instance_segmentation",
                    "source": str(tmp_path),
                    "prompts": ["car"],
                    "model_name": "sam3.pt",
                }
            ],
            "confirm_timeout": 30,
        }
    )
    plan.raw_instruction = "检测 dir 中的汽车"

    execute_plan(plan)  # 不抛异常（曾崩溃）

    out = capsys.readouterr().out
    assert "降级逐图" in out
    assert "单图仍 OOM" in out and "nvidia-smi" in out
    assert out.count("跳过:") == 3  # 3 图全 OOM → 全跳
    # 首次批量块 1 次 + 逐图 3 次
    assert len(calls) == 4
