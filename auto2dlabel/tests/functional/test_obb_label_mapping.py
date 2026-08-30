"""OBB 类名映射测试 — DOTA_CLASSES / _resolve_obb_label / _match_obb_prompt。

零权重依赖（纯函数）。
"""

from __future__ import annotations

import pytest

from auto2dlabel.configs.model_catalog import DOTA_CLASSES, DOTA_PROMPT_ALIASES
from auto2dlabel.models.obb import _match_obb_prompt, _resolve_obb_label


def test_dota_classes_length_and_order() -> None:
    """DOTA_CLASSES 必须与 ultralytics dota8.yaml 15 类权威顺序一致。"""
    assert len(DOTA_CLASSES) == 15
    assert DOTA_CLASSES[0] == "plane"
    assert DOTA_CLASSES[1] == "ship"
    assert DOTA_CLASSES[2] == "storage tank"
    assert DOTA_CLASSES[9] == "large vehicle"
    assert DOTA_CLASSES[10] == "small vehicle"
    assert DOTA_CLASSES[-1] == "swimming pool"


class TestResolveObbLabel:
    def test_names_priority(self) -> None:
        """checkpoint names dict 优先于 DOTA_CLASSES 回退表。"""
        assert _resolve_obb_label(0, {0: "custom-plane"}) == "custom-plane"

    def test_fallback_to_dota_classes(self) -> None:
        assert _resolve_obb_label(0) == "plane"
        assert _resolve_obb_label(14) == "swimming pool"

    def test_out_of_range_fallback_to_str_id(self) -> None:
        assert _resolve_obb_label(99) == "99"


class TestMatchObbPrompt:
    def test_alias_match(self) -> None:
        """DOTA 空格名经别名匹配 COCO 通用名（标注链路 prompt 兼容）。"""
        assert _match_obb_prompt("small vehicle", ["car"])
        assert _match_obb_prompt("small vehicle", ["vehicle"])
        assert _match_obb_prompt("large vehicle", ["truck"])
        assert _match_obb_prompt("ship", ["boat"])
        assert _match_obb_prompt("plane", ["airplane"])

    def test_hyphen_normalization_both_directions(self) -> None:
        assert _match_obb_prompt("small vehicle", ["small-vehicle"])
        assert _match_obb_prompt("small-vehicle", ["small vehicle"])

    def test_substring_match(self) -> None:
        assert _match_obb_prompt("small vehicle", ["small"])

    def test_no_match(self) -> None:
        assert not _match_obb_prompt("small vehicle", ["plane", "ship"])

    def test_aliases_table_consistent(self) -> None:
        """别名表键必须存在于 DOTA_CLASSES。"""
        assert set(DOTA_PROMPT_ALIASES) <= set(DOTA_CLASSES)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
