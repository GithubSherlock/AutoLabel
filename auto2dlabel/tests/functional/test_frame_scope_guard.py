"""单帧指代句式检测回归测试（2026-09-06 实测 F6）。

case-092「标注 KITTI 帧 abc123 中的汽车」曾被 LLM 解析成 image_2 整目录
7481 张批量检测、爆跑 600s（outputs 灌入 1980+ 帧 JSON）——单帧请求目录化
漂移。护栏：指令指代单帧 + 源展开为目录多图 → ValueError 快速失败。
句式检测纯函数零模型；批量修饰词（所有帧等）不触发（合法批量不误伤）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from auto2dlabel.cli_execute import _is_single_frame_request
from auto2dlabel.schema.task_plan import TaskPlan, TaskStep

# ---------- 句式检测（纯函数） ----------

@pytest.mark.parametrize(
    "text",
    [
        "标注 KITTI 帧 abc123 中的汽车",  # 实测漂移案
        "标注 KITTI 帧 000049 中的汽车",
        "请标注第 000055 号 KITTI 帧中的汽车",
        "KITTI 000015 帧的行人，帮我标一下",
        "KITTI 000025 帧的汽车，标一下",
        "帮我标注 KITTI 帧 abc 里的车",  # 短帧名
    ],
)
def test_single_frame_requests_detected(text: str) -> None:
    assert _is_single_frame_request(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "标注 KITTI 数据集中所有帧的汽车",  # 帧后无帧号，不匹配句式
        "检测 KITTI image_2 全部帧里的汽车",  # 同上
        "把 KITTI 所有帧 000049 里的汽车标出来",  # 批量修饰紧邻 → 放行
        "检测 /data/images 中的汽车",  # 无帧句式
        "分割 000860.png 中的汽车",  # 单图无帧字
    ],
)
def test_batch_or_non_frame_requests_not_detected(text: str) -> None:
    assert _is_single_frame_request(text) is False


# ---------- execute_plan 目录源拦截 ----------

def _plan_with(source: str, raw: str) -> TaskPlan:
    return TaskPlan(
        steps=[TaskStep(step_id=1, task_type="object_detection", source=source, prompts=["car"])],
        raw_instruction=raw,
    )


def test_directory_source_with_single_frame_instruction_skips(tmp_path: Path) -> None:
    """单帧指代指令 + 目录源多图 → 红字提示跳过该步（不海跑，exit 0 收尾）。"""
    imgs = [tmp_path / f"{i:06d}.png" for i in range(10)]
    plan = _plan_with(str(tmp_path), "标注 KITTI 帧 abc123 中的汽车")

    from auto2dlabel.cli_execute import execute_plan

    msgs: list[Any] = []
    with patch("auto2dlabel.cli_execute._print_step_header") as _ph, patch(
        "auto2dlabel.cli_execute.collect_images", return_value=imgs
    ), patch("auto2dlabel.cli_execute.console") as _console:
        _console.print.side_effect = lambda *a, **k: msgs.append(a)
        execute_plan(plan)
    assert any("指令指代单个帧" in str(m) for m in msgs), msgs
    assert not any("模型" in str(m) for m in msgs)  # 未走到模型/执行阶段


def test_directory_source_without_frame_instruction_passes(tmp_path: Path) -> None:
    """无帧指代（真批量）不拦截——正常走后续逻辑（此处 collect_images 为空即停）。"""
    img_dir = tmp_path / "imgs2"
    img_dir.mkdir()
    plan = _plan_with(str(img_dir), "检测 /data/images 中的汽车")

    from auto2dlabel.cli_execute import execute_plan

    with patch("auto2dlabel.cli_execute._print_step_header") as _ph, patch(
        "auto2dlabel.cli_execute.collect_images", return_value=[]
    ) as _ci, patch("auto2dlabel.cli_execute.console") as _console:
        execute_plan(plan)  # 无图 → 黄字提示后 continue，不抛
    _ci.assert_called_once()
