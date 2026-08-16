"""apply_evaluate_action 纯函数测试（LLM Evaluate 节点动作执行层）。

覆盖全部动作分支：accept / flag_for_review / retry_lower_threshold
（含 retry_used 拒绝、缺 detect_fn 抛错、非法 action 抛错）。
"""

from __future__ import annotations

from typing import Any

import pytest

from auto2dlabel.agent.evaluate import (
    LLM_RETRY_FACTOR,
    QualityReport,
    apply_evaluate_action,
)


def _report() -> QualityReport:
    return QualityReport(image_path="a.jpg", total_boxes=0, prompts=["car"],
                         warnings=["0 框"])


def _noop(conf: float) -> list[Any]:
    return []


def test_accept() -> None:
    r = apply_evaluate_action("accept", report=_report(), retry_used=False,
                              base_threshold=0.3, detect_fn=_noop)
    assert r["accepted"] is True
    assert "flagged" not in r and "retried" not in r
    assert r["report"]["total_boxes"] == 0  # 报告随结果返回


def test_flag_for_review() -> None:
    r = apply_evaluate_action("flag_for_review", report=_report(), retry_used=False,
                              base_threshold=0.3, detect_fn=_noop)
    assert r["flagged"] is True
    assert "accepted" not in r and "retried" not in r


def test_retry_calls_detect_once_at_lowered_threshold() -> None:
    calls: list[float] = []

    def fn(conf: float) -> list[Any]:
        calls.append(conf)
        return [1, 2]

    r = apply_evaluate_action("retry_lower_threshold", report=_report(), retry_used=False,
                              base_threshold=0.4, detect_fn=fn)
    assert r["retried"] is True
    assert r["threshold"] == pytest.approx(0.4 * LLM_RETRY_FACTOR)
    assert calls == [pytest.approx(0.4 * LLM_RETRY_FACTOR)]  # 只调一次
    assert r["detections"] == [1, 2]


def test_retry_refused_when_already_retried() -> None:
    """代码级已重试过（retry_used=True）→ 拒绝再重试，转为 accept。"""
    calls: list[float] = []

    def fn(conf: float) -> list[Any]:
        calls.append(conf)
        return []

    r = apply_evaluate_action("retry_lower_threshold", report=_report(), retry_used=True,
                              base_threshold=0.3, detect_fn=fn)
    assert r["accepted"] is True
    assert "retried" not in r
    assert calls == []  # 未触发检测


def test_retry_without_detect_fn_raises() -> None:
    with pytest.raises(ValueError, match="detect_fn"):
        apply_evaluate_action("retry_lower_threshold", report=_report(), retry_used=False,
                              base_threshold=0.3, detect_fn=None)


def test_invalid_action_raises() -> None:
    with pytest.raises(ValueError, match="非法 action"):
        apply_evaluate_action("delete_everything", report=_report(), retry_used=False,
                              base_threshold=0.3, detect_fn=_noop)
