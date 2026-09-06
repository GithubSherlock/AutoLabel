"""prompt_loader 单元测试（v1.0 P1+：LLM prompt 外置 markdown 加载与渲染）。

铁律：零真实 LLM / 零真实权重——tmp_path 手写 .md 验证加载器行为，
对包内真实 prompt 文件只做结构断言（占位符存在性），不构造任何模型。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from auto2dlabel.agent.prompt_loader import (
    PROFILES,
    PromptLoadError,
    PromptSpec,
    load_prompt,
    render,
)

_FRONTMATTER = "---\nname: test\ndescription: 测试 prompt\nprofile: planning"


def _load(tmp_path: Path, content: str, filename: str = "t.md") -> PromptSpec:
    (tmp_path / filename).write_text(content, encoding="utf-8")
    return load_prompt(filename, prompts_dir=tmp_path)


def _load_body(tmp_path: Path, body: str) -> PromptSpec:
    return _load(tmp_path, f"{_FRONTMATTER}\n---\n{body}")


# ---------- load_prompt ----------


def test_load_prompt_happy_path(tmp_path: Path) -> None:
    spec = _load_body(tmp_path, "hello $who")
    assert spec.name == "test"
    assert spec.description == "测试 prompt"
    assert spec.profile == "planning"
    assert spec.body == "hello $who"
    assert spec.source == tmp_path / "t.md"


def test_load_prompt_file_missing(tmp_path: Path) -> None:
    with pytest.raises(PromptLoadError, match="不可读"):
        load_prompt("nope.md", prompts_dir=tmp_path)


def test_load_prompt_no_frontmatter(tmp_path: Path) -> None:
    with pytest.raises(PromptLoadError, match="缺 frontmatter"):
        _load(tmp_path, "hello")


def test_load_prompt_frontmatter_not_closed(tmp_path: Path) -> None:
    with pytest.raises(PromptLoadError, match="结束行"):
        _load(tmp_path, "---\nname: test\n")


def test_load_prompt_invalid_yaml(tmp_path: Path) -> None:
    with pytest.raises(PromptLoadError, match="YAML 非法"):
        _load(tmp_path, "---\nname: [unclosed\n---\nbody")


def test_load_prompt_frontmatter_not_mapping(tmp_path: Path) -> None:
    with pytest.raises(PromptLoadError, match="YAML 映射"):
        _load(tmp_path, "---\n- a\n- b\n---\nbody")


def test_load_prompt_missing_name(tmp_path: Path) -> None:
    with pytest.raises(PromptLoadError, match="缺 name"):
        _load(tmp_path, "---\ndescription: d\nprofile: planning\n---\nbody")


def test_load_prompt_unknown_profile(tmp_path: Path) -> None:
    with pytest.raises(PromptLoadError, match=r"profile 未知.*planning"):
        _load(tmp_path, "---\nname: t\ndescription: d\nprofile: bogus\n---\nbody")


def test_load_prompt_empty_body(tmp_path: Path) -> None:
    with pytest.raises(PromptLoadError, match="正文为空"):
        _load_body(tmp_path, "\n  \n")


# ---------- render ----------


def test_render_preserves_json_braces(tmp_path: Path) -> None:
    """正文字面 JSON 花括号不被渲染触碰（禁用 str.format 的动机）。"""
    spec = _load_body(tmp_path, '{"a": 1, "b": [$x]}')
    assert render(spec, x=2) == '{"a": 1, "b": [2]}'


def test_render_str_coercion(tmp_path: Path) -> None:
    assert render(_load_body(tmp_path, "[$n]"), n=3) == "[3]"


def test_render_prefix_single_pass(tmp_path: Path) -> None:
    """$a 不得污染 $ab（单遍 regex 替换——str.replace 顺序替换的典型坑）。"""
    assert render(_load_body(tmp_path, "$ab $a"), a=1, ab=2) == "2 1"


def test_render_lone_dollar_untouched(tmp_path: Path) -> None:
    assert render(_load_body(tmp_path, "100$ $x"), x=1) == "100$ 1"


def test_render_missing_placeholders_aggregated(tmp_path: Path) -> None:
    with pytest.raises(PromptLoadError, match=r"a, b"):
        render(_load_body(tmp_path, "$a $b $a"))


def test_render_extra_vars_ignored(tmp_path: Path) -> None:
    assert render(_load_body(tmp_path, "$x"), x=1, unused=9) == "1"


# ---------- PROFILES ----------


def test_profiles_planning_tier() -> None:
    assert set(PROFILES) == {"planning"}
    assert PROFILES["planning"] == {"temperature": 0.0, "max_tokens": 1024, "json_mode": True}


# ---------- 包内真实 prompt 文件结构 ----------


def test_planner_md_placeholders_present() -> None:
    spec = load_prompt("planner.md")
    for ph in ("$catalog_summary", "$datasets_summary", "$task_params_summary"):
        assert ph in spec.body


def test_benchmark_md_placeholders_present() -> None:
    spec = load_prompt("benchmark.md")
    for ph in (
        "$default_model",
        "$default_seg_model",
        "$default_conf",
        "$default_iou",
        "$default_max_images",
    ):
        assert ph in spec.body
    # f-string 双写花括号已还原单括号（JSON 样例原文可 copy）
    assert '"dataset":' in spec.body
    assert "{{" not in spec.body


def test_import_time_render_leaves_no_placeholder() -> None:
    from auto2dlabel.agent.planner import _BENCHMARK_SYSTEM_PROMPT, _PLANNER_SYSTEM_PROMPT

    assert "$" not in _PLANNER_SYSTEM_PROMPT
    assert "$" not in _BENCHMARK_SYSTEM_PROMPT
