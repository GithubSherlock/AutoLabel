"""批次清单（tools/batch.py）纯函数测试。

覆盖：roundtrip、原子写、resume 过滤（跳过 ok）、更新幂等、损坏/缺字段容错。
"""

from __future__ import annotations

import pytest

from auto2dlabel.tests import Path, json
from auto2dlabel.tools.batch import (
    STATUS_FAILED,
    STATUS_OK,
    STATUS_PENDING,
    BatchManifest,
    load_manifest,
    new_manifest,
    resume_targets,
    save_manifest,
    update_entry,
)


def _make_manifest() -> BatchManifest:
    return new_manifest(
        "检测汽车",
        config={"threshold": 0.3, "iou": 0.5},
        image_paths=["/data/a.jpg", "/data/b.jpg", "/data/c.jpg"],
    )


def test_manifest_roundtrip(tmp_path: Path) -> None:
    """保存后重新加载，字段一致。"""
    m = _make_manifest()
    update_entry(m, "/data/a.jpg", status=STATUS_OK, elapsed=1.5, bbox_count=3,
                 state_file="outputs/a_state.json")
    update_entry(m, "/data/b.jpg", status=STATUS_FAILED, error="boom", elapsed=0.2)

    out = save_manifest(m, tmp_path / "batch_manifest.json")
    loaded = load_manifest(out)

    assert loaded.instruction == "检测汽车"
    assert loaded.config == {"threshold": 0.3, "iou": 0.5}
    assert [e.path for e in loaded.images] == ["/data/a.jpg", "/data/b.jpg", "/data/c.jpg"]
    assert loaded.images[0].status == STATUS_OK
    assert loaded.images[0].bbox_count == 3
    assert loaded.images[1].status == STATUS_FAILED
    assert loaded.images[1].error == "boom"
    assert loaded.images[2].status == STATUS_PENDING


def test_save_manifest_atomic_no_tmp_left(tmp_path: Path) -> None:
    """原子写：完成后无 .tmp 残留。"""
    out = save_manifest(_make_manifest(), tmp_path / "m.json")
    assert out.is_file()
    assert not list(tmp_path.glob("*.tmp"))


def test_resume_targets_skips_ok(tmp_path: Path) -> None:
    """--resume 语义：跳过 ok，重跑 failed/pending。"""
    m = _make_manifest()
    update_entry(m, "/data/a.jpg", status=STATUS_OK)
    update_entry(m, "/data/b.jpg", status=STATUS_FAILED, error="x")

    targets = resume_targets(m)
    assert [t.path for t in targets] == ["/data/b.jpg", "/data/c.jpg"]  # failed + pending


def test_update_entry_missing_returns_false() -> None:
    """更新不存在的路径返回 False 不抛错。"""
    m = _make_manifest()
    assert update_entry(m, "/data/nope.jpg", status=STATUS_OK) is False


def test_load_manifest_missing_fields_tolerated(tmp_path: Path) -> None:
    """旧版/缺字段清单加载容错（全部字段带默认值）。"""
    p = tmp_path / "old.json"
    p.write_text(json.dumps({
        "instruction": "x",
        "images": [{"path": "/data/a.jpg"}, {"path": "/data/b.jpg", "status": "ok"}],
    }), encoding="utf-8")
    m = load_manifest(p)
    assert m.images[0].status == STATUS_PENDING
    assert m.images[1].status == STATUS_OK
    assert m.config == {}


def test_load_manifest_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "nope.json")
