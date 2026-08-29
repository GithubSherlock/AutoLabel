"""检测类别过滤纯函数（models/detection._match_prompt）测试。

2026-08-29 回归：prompts 为空时 _match_prompt 曾恒 False → 所有框被过滤，
导致「未指定类别」的 chat 指令（如 KITTI 数据集检测）静默输出 7481 张零框。
修复语义：空 prompts = 用户未指定类别 = 不过滤（全类别输出）。
"""

from __future__ import annotations

from auto2dlabel.models.detection import _match_prompt


def test_match_prompt_exact() -> None:
    """完全匹配 → True。"""
    assert _match_prompt("car", ["car"]) is True


def test_match_prompt_mismatch() -> None:
    """不匹配 → False（过滤该类别）。"""
    assert _match_prompt("car", ["truck"]) is False


def test_match_prompt_case_insensitive() -> None:
    """大小写不敏感。"""
    assert _match_prompt("Car", ["car"]) is True
    assert _match_prompt("PEDESTRIAN", ["pedestrian"]) is True


def test_match_prompt_substring() -> None:
    """子串双向匹配（"light" 命中 "traffic light"；"car" 命中 "race car"）。"""
    assert _match_prompt("traffic light", ["light"]) is True
    assert _match_prompt("race car", ["car"]) is True


def test_match_prompt_multiple_any() -> None:
    """多 prompt 任一命中即 True。"""
    assert _match_prompt("person", ["car", "truck", "person"]) is True
    assert _match_prompt("dog", ["car", "truck", "person"]) is False


def test_match_prompt_empty_no_filter() -> None:
    """回归：空 prompts（未指定类别）→ 不过滤，全类别输出。"""
    assert _match_prompt("car", []) is True
    assert _match_prompt("anything", []) is True
