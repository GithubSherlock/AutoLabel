"""v1.1 P1 标注经验库测试 —— experience.py 纯函数 + planner 动态注入。

零真实权重铁律：embedding 用 FakeEmbed 注入（固定向量），绝不构造 CLIP
实例（沿用 ReID/约束过滤 Fake 注入铁律）；经验日志统一重定向 tmp_path。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from auto2dlabel.agent import experience as exp
from auto2dlabel.agent import planner as planner_mod
from auto2dlabel.agent.evaluate import QualityReport
from auto2dlabel.agent.experience import (
    ExperienceEntry,
    _cosine,
    _extract_human_action,
    append_entry,
    build_entry,
    format_fewshot,
    load_entries,
    retrieve,
)


@pytest.fixture(autouse=True)
def _exp_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """经验日志重定向 tmp_path（防测试写真实 logs/experience.jsonl）。"""
    path = tmp_path / "experience.jsonl"
    monkeypatch.setattr(exp, "EXPERIENCE_LOG_PATH", path)
    return path


class _FakeEmbed:
    """固定向量矩阵注入（零权重）：按文本哈希给确定向量，供 top-k 排序。"""

    def __init__(self, vecs: dict[str, list[float]]) -> None:
        self._vecs = vecs

    def __call__(self, text: str) -> list[float]:
        return self._vecs.get(text, [0.0] * 4)


def _entry(**kw: Any) -> ExperienceEntry:
    defaults: dict[str, Any] = {
        "instruction": "检测汽车", "domain": "2d", "dataset": "coco",
        "task_type": "object_detection", "model": "yolo26x.pt",
    }
    defaults.update(kw)
    return ExperienceEntry(**defaults)


# ---------- build_entry / 人工修正提取 ----------


def test_build_entry_extracts_quality_and_human() -> None:
    """纯代码构造：quality.ok → quality 字段；reviewed annotations →
    edited_by_human/issues 计数 → human_action。"""
    entry = build_entry(
        instruction="检测汽车", task_type="object_detection", model="yolo26x.pt",
        confidence_threshold=0.3,
        quality=QualityReport(total_boxes=0, missing_prompts=["person"]),
        reviewed_annotations=[
            {"edited_by_human": True},
            {"edited_by_human": True},
            {"issues": [1]},
        ],
    )
    assert entry.quality == "False"  # ok==False（缺类）
    assert entry.human_action == "人工修正 2 框 + 1 处区域问题"
    assert entry.confidence_threshold == 0.3


def test_build_entry_no_quality_ok() -> None:
    """quality=None → quality 字段空；无人工修正 → human_action 空。"""
    entry = build_entry(instruction="检测汽车")
    assert entry.quality == ""
    assert entry.human_action == ""


def test_extract_human_action_variants() -> None:
    """edited/issues 三态：仅框 / 仅问题 / 都无。"""
    assert _extract_human_action([{"edited_by_human": True}]) == "人工修正 1 框"
    # issues 按「含问题标注数」计（implementation 语义：逐 annotation 检查）
    assert _extract_human_action([{"issues": [1, 2]}, {"issues": [3]}]) == "2 处区域问题"
    assert _extract_human_action([{"label": "car"}]) == ""
    assert _extract_human_action([]) == ""


# ---------- append / load ----------


def test_append_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "e.jsonl"
    e = _entry()
    assert append_entry(path, e) is True
    assert append_entry(path, e) is False  # 同 instruction+model 去重
    loaded = load_entries(path)
    assert len(loaded) == 1
    assert loaded[0].instruction == "检测汽车"
    assert loaded[0].model == "yolo26x.pt"


def test_load_skips_corrupted_lines(tmp_path: Path) -> None:
    path = tmp_path / "e.jsonl"
    path.write_text(
        '{"instruction": "正常", "model": "a"}\n'
        '{"instruction": "损坏\n'
        '{"instruction": "脏类型", "model": "b", "confidence_threshold": "abc"}\n',
        encoding="utf-8",
    )
    loaded = load_entries(path)
    assert len(loaded) == 2  # 损坏行跳过；脏阈值容错
    assert loaded[1].confidence_threshold is None


def test_load_missing_file_empty() -> None:
    assert load_entries(Path("/nonexistent/x.jsonl")) == []


# ---------- metadata 过滤 + 向量 top-k ----------


def test_retrieve_meta_filter_excludes_mismatch() -> None:
    """metadata 精确过滤先行：dataset 不匹配的候选不进池。"""
    pool = [
        _entry(instruction="检测汽车", dataset="coco"),
        _entry(instruction="检测汽车", dataset="kitti"),
    ]
    hits = retrieve(
        "检测汽车", dataset="coco", entries=pool,
        embed_fn=_FakeEmbed({"检测汽车": [1.0, 0.0, 0.0, 0.0]}),
    )
    assert [h.dataset for h in hits] == ["coco"]


def test_retrieve_vector_ranks_within_candidates() -> None:
    """向量 top-k 只在候选集内排序：相似度高者优先。"""
    pool = [
        _entry(instruction="检测汽车", dataset="coco"),
        _entry(instruction="检测卡车", dataset="coco"),
        _entry(instruction="检测行人", dataset="coco"),
    ]
    # query 贴近「检测汽车」
    vecs = {
        "检测汽车": [1.0, 0.0, 0.0, 0.0],
        "检测卡车": [0.8, 0.2, 0.0, 0.0],
        "检测行人": [0.1, 0.9, 0.0, 0.0],
        "检测汽车和卡车": [0.95, 0.1, 0.0, 0.0],
    }
    hits = retrieve("检测汽车和卡车", dataset="coco", entries=pool, embed_fn=_FakeEmbed(vecs))
    assert [h.instruction for h in hits] == ["检测汽车", "检测卡车", "检测行人"]


def test_retrieve_empty_pool_and_no_meta() -> None:
    """空池 → []；无 metadata 条件 → 全池参与向量排序。"""
    assert retrieve("检测汽车", entries=[], embed_fn=_FakeEmbed({})) == []
    pool = [_entry(instruction="a"), _entry(instruction="b")]
    hits = retrieve(
        "x", entries=pool,
        embed_fn=_FakeEmbed({"x": [1.0, 0.0, 0.0, 0.0], "a": [0.9, 0, 0, 0], "b": [0.1, 0, 0, 0]}),
    )
    assert [h.instruction for h in hits] == ["a", "b"]


def test_retrieve_fallback_meta_only_when_embed_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """生产 embed 不可用（monkeypatch 返回 None，零真实权重）→ 仅 metadata
    过滤按原序截断 top-k。"""
    monkeypatch.setattr(exp, "_embed_fn", lambda: None)
    pool = [_entry(instruction="a"), _entry(instruction="b")]
    hits = retrieve("x", entries=pool, embed_fn=None)
    assert len(hits) <= 2 and hits  # 只断言数量（不排序），embedding 缺失回退


def test_retrieve_k_cap() -> None:
    pool = [_entry(instruction=f"i{n}", dataset="coco") for n in range(6)]
    hits = retrieve(
        "检测汽车", dataset="coco", k=3, entries=pool,
        embed_fn=_FakeEmbed({f"i{n}": [1.0, 0, 0, 0] for n in range(6)}),
    )
    assert len(hits) == 3


# ---------- 辅助 ----------


def test_cosine() -> None:
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert _cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    assert _cosine([1.0, 0.0], []) == 0.0  # 维度不匹配 → 0


def test_format_fewshot_empty_and_content() -> None:
    assert format_fewshot([]) == ""
    text = format_fewshot([_entry(instruction="检测汽车", model="yolo26x.pt", dataset="coco")])
    assert text.startswith("<history>")
    assert "检测汽车" in text and "yolo26x.pt" in text and "coco" in text


# ---------- planner 动态注入 ----------


def test_planner_system_prompt_injects_fewshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """检索命中 → system prompt 含 <history> 段（RAG few-shot 注入生效）。"""
    monkeypatch.setattr(
        exp,
        "retrieve",
        lambda instruction: [_entry(instruction="检测汽车", model="yolo26x.pt")],
    )
    sp = planner_mod._build_system_prompt("检测汽车")
    assert "<history>" in sp
    assert "yolo26x.pt" in sp


def test_planner_system_prompt_empty_history(monkeypatch: pytest.MonkeyPatch) -> None:
    """空库 → 无 <history> 段（规则行仍保留，LLM 忽略本节）。"""
    monkeypatch.setattr(exp, "retrieve", lambda instruction: [])
    sp = planner_mod._build_system_prompt("检测汽车")
    assert "<history>" not in sp
    assert "历史经验" in sp


def test_planner_system_prompt_retrieve_failure_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """检索抛异常 → 回退 import 期基础 prompt（静默降级绝不 raise）。"""
    monkeypatch.setattr(
        exp,
        "retrieve",
        lambda instruction: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    sp = planner_mod._build_system_prompt("检测汽车")
    assert "历史经验" in sp  # 基础版含规则行
    assert sp == planner_mod._PLANNER_SYSTEM_PROMPT


def test_planner_import_constant_no_history() -> None:
    """import 期常量无 <history> 段（动态注入在 parse 期；契约不回归）。"""
    assert "<history>" not in planner_mod._PLANNER_SYSTEM_PROMPT
