"""类别推荐器测试（v0.6 P2 落地形态锁定）：scan_classes / recommend_classes /
format_recommendation 纯逻辑 + _maybe_recommend_classes 集成。

零真实权重铁律：scan 用 Fake 模型注入；_maybe_recommend_classes 用
monkeypatch 替换扫描与确认（P2 形态 = 推荐保持代码级，对话轮不参与）。
"""

from __future__ import annotations

from typing import Any

import pytest

from auto2dlabel import cli_commands
from auto2dlabel.models.model_catalog import COCO_CLASSES
from auto2dlabel.schema.task_plan import TaskPlan, TaskStep
from auto2dlabel.tools.confirm import ConfirmResult
from auto2dlabel.tools.recommend import (
    format_recommendation,
    recommend_classes,
    scan_classes,
)


class _FakeResult:
    """检测结果 duck type（scan_classes 只取 label/confidence）。"""

    def __init__(self, label: str, confidence: float) -> None:
        self.label = label
        self.confidence = confidence


class _FakeModel:
    """检测模型 Fake：记录调用，返回固定结果（零真实权重铁律）。"""

    def __init__(self, results: list[_FakeResult]) -> None:
        self._results = results
        self.calls: list[tuple[str, list[str], float]] = []

    def detect(
        self, image_path: str, classes: list[str], confidence_threshold: float
    ) -> list[_FakeResult]:
        self.calls.append((image_path, list(classes), confidence_threshold))
        return self._results


def test_scan_classes_stats() -> None:
    """单图全类检测 → 每类 count/avg/min/max 统计；COCO 80 类注入。"""
    model = _FakeModel([
        _FakeResult("car", 0.9),
        _FakeResult("car", 0.5),
        _FakeResult("person", 0.7),
    ])
    stats = scan_classes("/tmp/x.jpg", model=model, confidence_threshold=0.3)

    assert stats == {
        "car": {"count": 2, "avg_conf": 0.7, "min_conf": 0.5, "max_conf": 0.9},
        "person": {"count": 1, "avg_conf": 0.7, "min_conf": 0.7, "max_conf": 0.7},
    }
    assert model.calls == [("/tmp/x.jpg", COCO_CLASSES, 0.3)]


def test_recommend_classes_thresholds_and_buckets() -> None:
    """count×avg_conf 排序；count<min / avg<min → skipped；count=0 → rejected。"""
    stats = {
        "car": {"count": 5, "avg_conf": 0.8, "min_conf": 0.6, "max_conf": 0.9},
        "person": {"count": 2, "avg_conf": 0.9, "min_conf": 0.8, "max_conf": 0.95},
        "cat": {"count": 4, "avg_conf": 0.2, "min_conf": 0.1, "max_conf": 0.3},
        "dog": {"count": 0, "avg_conf": 0.0, "min_conf": 0.0, "max_conf": 0.0},
    }
    recommended, skipped, rejected = recommend_classes(stats)

    assert recommended == ["car"]
    assert skipped == ["person", "cat"]  # 按 score 降序（1.8 > 0.8）
    assert rejected == ["dog"]


def test_recommend_classes_top_k() -> None:
    """top_k 截断：超额的合格类落 skipped。"""
    stats = {c: {"count": 4, "avg_conf": 0.8, "min_conf": 0.7, "max_conf": 0.9} for c in "abc"}
    recommended, skipped, _ = recommend_classes(stats, top_k=2)

    assert recommended == ["a", "b"]  # score 相同 → 保 dict 插入序
    assert skipped == ["c"]


def test_format_recommendation_text() -> None:
    """格式文本：推荐/跳过分区 + 百分比 + 超时提示；空跳过不渲染分区。"""
    stats = {
        "car": {"count": 5, "avg_conf": 0.8},
        "person": {"count": 1, "avg_conf": 0.9},
    }
    text = format_recommendation(["car"], ["person"], stats)
    assert "建议标注" in text
    assert "car: 5 框, avg 80%" in text
    assert "person: 1 框, avg 90%" in text
    assert "30 秒" in text
    assert "可跳过" not in format_recommendation(["car"], [], stats)


# ── _maybe_recommend_classes 集成（v0.6 P2 落地形态）───────

def _recommend_fail(*a: Any, **kw: Any) -> Any:
    pytest.fail("不应触发类别推荐")


def test_maybe_recommend_skips_when_prompts_filled(monkeypatch: Any) -> None:
    """有 prompts → 零扫描零询问（不覆盖用户已指定类别）。"""
    plan = TaskPlan(steps=[TaskStep(step_id=1, source="/x", prompts=["car"])])
    monkeypatch.setattr("auto2dlabel.tools.recommend.scan_classes", _recommend_fail)
    cli_commands._maybe_recommend_classes(plan, timeout=30, no_wait=False)


def test_maybe_recommend_skips_tracking(monkeypatch: Any) -> None:
    """tracking 步骤跳过（固定类别跟踪不走类别推荐）。"""
    plan = TaskPlan(steps=[TaskStep(step_id=1, task_type="tracking", source="v.mp4")])
    monkeypatch.setattr("auto2dlabel.tools.recommend.scan_classes", _recommend_fail)
    cli_commands._maybe_recommend_classes(plan, timeout=30, no_wait=False)


def test_maybe_recommend_skips_missing_source(monkeypatch: Any) -> None:
    """source 不存在 → 跳过（不扫描）。"""
    plan = TaskPlan(steps=[TaskStep(step_id=1, source="/nonexistent/dir")])
    monkeypatch.setattr("auto2dlabel.tools.recommend.scan_classes", _recommend_fail)
    cli_commands._maybe_recommend_classes(plan, timeout=30, no_wait=False)


def test_maybe_recommend_no_wait_auto_select(monkeypatch: Any, tmp_path: Any) -> None:
    """--no-wait：扫描后自动选择推荐类别，零交互。"""
    (tmp_path / "img.jpg").write_bytes(b"x")
    plan = TaskPlan(steps=[TaskStep(step_id=1, source=str(tmp_path))])
    stats = {
        "car": {"count": 5, "avg_conf": 0.8, "min_conf": 0.6, "max_conf": 0.9},
        "person": {"count": 3, "avg_conf": 0.7, "min_conf": 0.5, "max_conf": 0.8},
    }
    monkeypatch.setattr("auto2dlabel.tools.recommend.scan_classes", lambda p: stats)
    monkeypatch.setattr("auto2dlabel.tools.confirm.ask_with_timeout", _recommend_fail)

    cli_commands._maybe_recommend_classes(plan, timeout=0, no_wait=True)

    assert plan.steps[0].prompts == ["car", "person"]


def test_maybe_recommend_confirm_ok_and_custom(monkeypatch: Any, tmp_path: Any) -> None:
    """交互确认：'ok' → 推荐类别；自定义输入 → 用户类别（中文逗号兼容）。"""
    (tmp_path / "img.jpg").write_bytes(b"x")
    stats = {
        "car": {"count": 5, "avg_conf": 0.8, "min_conf": 0.6, "max_conf": 0.9},
    }
    monkeypatch.setattr("auto2dlabel.tools.recommend.scan_classes", lambda p: stats)

    # 回复 ok → 采用推荐
    plan_ok = TaskPlan(steps=[TaskStep(step_id=1, source=str(tmp_path))])
    monkeypatch.setattr(
        "auto2dlabel.tools.confirm.ask_with_timeout",
        lambda msg, timeout=30: ConfirmResult(confirmed=True, user_input="ok"),
    )
    cli_commands._maybe_recommend_classes(plan_ok, timeout=30, no_wait=False)
    assert plan_ok.steps[0].prompts == ["car"]

    # 自定义输入（中文逗号）→ 用户类别
    plan_custom = TaskPlan(steps=[TaskStep(step_id=1, source=str(tmp_path))])
    monkeypatch.setattr(
        "auto2dlabel.tools.confirm.ask_with_timeout",
        lambda msg, timeout=30: ConfirmResult(confirmed=True, user_input="狗，猫"),
    )
    cli_commands._maybe_recommend_classes(plan_custom, timeout=30, no_wait=False)
    assert plan_custom.steps[0].prompts == ["狗", "猫"]
