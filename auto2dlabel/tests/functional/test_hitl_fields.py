"""export_triage 顶层字段（image_path/image_size）测试。

Web 复核队列消费方依赖这两个字段显示原图；旧调用不传时置空容错。
"""

from __future__ import annotations

import json
from pathlib import Path

from auto2dlabel.schema.annotation import Bbox
from auto2dlabel.tools.hitl import TriageResult, export_triage


def test_review_queue_contains_image_fields(tmp_path: Path) -> None:
    """复核队列文件顶层包含 image_path 与 image_size，且合并 review+hard 两档。"""
    triage = TriageResult()
    triage.review.append(Bbox(x=1, y=2, width=3, height=4, label="car", confidence=0.5))
    triage.hard.append(Bbox(x=5, y=6, width=7, height=8, label="dog", confidence=0.1))

    files = export_triage(
        triage,
        output_dir=tmp_path,
        image_stem="img_01",
        timestamp="2026-08-16-10-00-00",
        image_path="/data/img_01.png",
        image_size=(640, 480),
    )
    data = json.loads(files["review"].read_text(encoding="utf-8"))
    assert data["image"] == "img_01"
    assert data["image_path"] == "/data/img_01.png"
    assert data["image_size"] == [640, 480]
    assert data["summary"] == {"review_count": 1, "hard_count": 1, "total_need_review": 2}
    # 复核队列 = 中置信度 + 困难样本两档（都需人工处理）
    assert [a["label"] for a in data["annotations"]] == ["car", "dog"]


def test_hard_file_contains_image_fields(tmp_path: Path) -> None:
    """hard 文件同样带 image_path/image_size。"""
    triage = TriageResult()
    triage.hard.append(Bbox(x=5, y=6, width=7, height=8, label="dog", confidence=0.1))

    files = export_triage(
        triage,
        output_dir=tmp_path,
        image_stem="img_02",
        image_path="/data/img_02.png",
        image_size=(100, 100),
    )
    hard = json.loads(files["hard"].read_text(encoding="utf-8"))
    assert hard["image_path"] == "/data/img_02.png"
    assert hard["image_size"] == [100, 100]


def test_hard_empty_not_written(tmp_path: Path) -> None:
    """hard 档为空时不写 hard 文件（回归：无条件写会留下空文件）。"""
    triage = TriageResult()
    triage.review.append(Bbox(x=1, y=2, width=3, height=4, label="car", confidence=0.5))

    files = export_triage(triage, output_dir=tmp_path, image_stem="img_03")
    assert files["review"].is_file()
    assert not files["hard"].exists()


def test_legacy_call_defaults(tmp_path: Path) -> None:
    """旧调用（不传 image_path/image_size）仍可写文件，字段置空容错。"""
    triage = TriageResult()
    triage.review.append(Bbox(x=1, y=2, width=3, height=4, label="car", confidence=0.5))

    files = export_triage(triage, output_dir=tmp_path, image_stem="img_04")
    data = json.loads(files["review"].read_text(encoding="utf-8"))
    assert data["image_path"] == ""
    assert data["image_size"] == [0, 0]
