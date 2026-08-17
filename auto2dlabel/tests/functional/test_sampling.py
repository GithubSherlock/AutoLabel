"""主动学习采样（tools/sampling.py）测试。

覆盖：队列聚合（跳过 .reviewed/损坏/空文件）、score 计算与排序、
确定性 top-K、空目录、清单写入。
"""

from __future__ import annotations

from typing import Any

import pytest

from auto2dlabel.tests import Path, json
from auto2dlabel.tools.sampling import (
    collect_review_pool,
    sample_priority,
    score_image,
    write_priority_manifest,
)


def _write_queue(d: Path, name: str, annotations: list[dict[str, Any]],
                 image_path: str = "") -> None:
    (d / name).write_text(json.dumps({
        "image": name.removesuffix("_review.json"),
        "image_path": image_path,
        "annotations": annotations,
    }), encoding="utf-8")


def _box(label: str, conf: float) -> dict[str, Any]:
    return {"x": 0, "y": 0, "width": 1, "height": 1, "label": label, "confidence": conf}


def test_score_image_uncertainty() -> None:
    """conf 接近 0.5 得分高，极端置信度得分低；类别多样加分。"""
    assert score_image([0.5], 1) == pytest.approx(1.0 + 0.1)
    assert score_image([0.9], 1) == pytest.approx(1 - 2 * 0.4 + 0.1)
    assert score_image([0.5, 0.5], 2) == pytest.approx(1.0 + 0.2)
    assert score_image([], 3) == 0.0


def test_collect_pool_and_priority(tmp_path: Path) -> None:
    """聚合后按 score 降序；.reviewed/损坏/空文件被跳过。"""
    _write_queue(tmp_path, "a_review.json", [_box("car", 0.5), _box("dog", 0.6)])  # unc≈0.95, +0.2
    _write_queue(tmp_path, "b_review.json", [_box("car", 0.95)])  # unc=0.1, +0.1
    _write_queue(tmp_path, "c_review.json", [_box("car", 0.5)])  # unc=1.0, +0.1
    _write_queue(tmp_path, "d_review.json.reviewed", [_box("car", 0.5)])  # 已复核跳过
    (tmp_path / "e_review.json").write_text("{broken", encoding="utf-8")  # 损坏跳过
    _write_queue(tmp_path, "f_review.json", [])  # 空跳过

    pool = collect_review_pool(tmp_path)
    assert [p.file for p in pool] == ["a_review.json", "b_review.json", "c_review.json"]

    top = sample_priority(pool, top_k=2)
    assert [t.file for t in top] == ["a_review.json", "c_review.json"]
    assert top[0].score == pytest.approx(score_image([0.5, 0.6], 2))
    assert top[0].class_count == 2 and top[0].bbox_count == 2


def test_priority_tie_break_deterministic() -> None:
    """同分按文件名升序，结果可复现。"""
    from auto2dlabel.tools.sampling import ReviewItem

    pool = [ReviewItem(file="b", score=0.5), ReviewItem(file="a", score=0.5),
            ReviewItem(file="c", score=0.9)]
    top = sample_priority(pool, top_k=2)
    assert [t.file for t in top] == ["c", "a"]


def test_collect_empty_dir(tmp_path: Path) -> None:
    assert collect_review_pool(tmp_path) == []
    assert collect_review_pool(tmp_path / "nope") == []


def test_write_manifest(tmp_path: Path) -> None:
    _write_queue(tmp_path, "a_review.json", [_box("car", 0.5)], image_path="/data/a.png")
    top = sample_priority(collect_review_pool(tmp_path), top_k=1)
    out = write_priority_manifest(top, tmp_path / "sampling_manifest.json")

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["count"] == 1
    assert data["items"][0]["rank"] == 1
    assert data["items"][0]["file"] == "a_review.json"
    assert data["items"][0]["image_path"] == "/data/a.png"
