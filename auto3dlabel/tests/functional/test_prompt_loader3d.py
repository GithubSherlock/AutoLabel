"""auto3dlabel prompt 外置加载（v1.0 P1+，与 2D 同模式；依赖方向 3D → 2D 合法）。

断言：planner3d.md 载入（本包 prompts 目录）/ Python `\\` 续行已并接（正文
零反斜杠）/ 占位符注入 + JSON 花括号保留 / import 期渲染无残留占位符。
零真实 LLM / 零真实权重。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from auto2dlabel.agent.prompt_loader import PromptLoadError, PromptSpec, load_prompt, render

_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "agent" / "prompts"


def _spec() -> PromptSpec:
    return load_prompt("planner3d.md", prompts_dir=_PROMPTS_DIR)


def test_planner3d_spec_loaded() -> None:
    spec = _spec()
    assert spec.name == "planner3d"
    assert spec.profile == "planning"
    assert spec.source == _PROMPTS_DIR / "planner3d.md"


def test_planner3d_body_no_backslash_continuations() -> None:
    """Python 字符串 `\\` 续行搬移进 .md 时必须按语义并成单行（否则 LLM 看到裸反斜杠）。"""
    spec = _spec()
    assert "\\" not in spec.body
    # 三处原续行点（nuScenes 批量意图 / det_model 关键词映射 / questions 规则）已并接
    assert "batch intent (多张/N张/随机N张/批量)" in spec.body
    assert "highest measured mAP" in spec.body


def test_planner3d_placeholders_injected() -> None:
    spec = _spec()
    rendered = render(spec, datasets_summary="DS-SUMMARY", catalog_summary="CAT-SUMMARY")
    # JSON 样例花括号原样保留
    assert '"frame_id": "000123"' in rendered
    # 注入段落到位且原文不再含占位符
    assert "$datasets_summary" in spec.body and "$catalog_summary" in spec.body
    assert "$" not in rendered
    assert "DS-SUMMARY" in rendered and "CAT-SUMMARY" in rendered


def test_planner3d_missing_placeholder_errors() -> None:
    with pytest.raises(PromptLoadError, match=r"datasets_summary, catalog_summary"):
        render(_spec())


def test_import_time_render3d_leaves_no_placeholder() -> None:
    from auto3dlabel.agent.planner3d import _PLANNER3D_SYSTEM_PROMPT

    assert "$" not in _PLANNER3D_SYSTEM_PROMPT
