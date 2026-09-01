"""test_nuscenes_labels：训练标签导出（{sample_token}.json）纯函数——零权重零 devkit。

格式契约（与官方 submission box dict 同键）：detection_name/detection_score/
translation/size/rotation/velocity（velocity 恒输出，None → [0,0]）；track_id 不入
文件（官方格式无此字段，见 nuscenes_labels docstring 已知简化）。
"""

from __future__ import annotations

import json
from pathlib import Path

from auto3dlabel.export.nuscenes_labels import (
    build_label_dict,
    load_dataset_labels,
    load_sample_label,
    write_sample_label,
)
from auto3dlabel.schema.nuscenes_box import NusBox


def _box(
    label: str = "car",
    confidence: float = 0.9,
    translation: tuple[float, float, float] = (10.0, 2.0, 0.5),
    velocity: tuple[float, float] | None = (1.0, 0.5),
) -> NusBox:
    return NusBox(
        label=label, confidence=confidence, translation=translation,
        size=(2.0, 4.0, 1.5), quaternion=(1.0, 0.0, 0.0, 0.0),
        velocity=velocity, track_id="inst-7",
    )


def test_build_label_dict_velocity_always_present() -> None:
    """velocity 恒输出：None → [0,0]（官方格式键必存）；track_id 不入文件。"""
    d = build_label_dict("tok1", [_box()])
    assert d["sample_token"] == "tok1"
    box = d["boxes"][0]
    assert box["detection_name"] == "car"
    assert box["detection_score"] == 0.9
    assert box["translation"] == [10.0, 2.0, 0.5]
    assert box["size"] == [2.0, 4.0, 1.5]
    assert box["velocity"] == [1.0, 0.5]
    assert "track_id" not in box  # 官方格式无此字段（已知简化）

    d_none = build_label_dict("tok2", [_box(velocity=None)])
    assert d_none["boxes"][0]["velocity"] == [0.0, 0.0]


def test_write_and_load_sample_label_roundtrip(tmp_path: Path) -> None:
    """写 {token}.json → 读回 (token, [NusBox]) 全字段一致。"""
    box = _box()
    path = write_sample_label("tok1", [box], tmp_path)
    assert path == tmp_path / "tok1.json"
    assert path.is_file()

    token, boxes = load_sample_label(path)
    assert token == "tok1" and len(boxes) == 1
    got = boxes[0]
    assert got.label == "car" and got.confidence == 0.9
    assert got.translation == (10.0, 2.0, 0.5)
    assert got.size == (2.0, 4.0, 1.5)
    assert got.velocity == (1.0, 0.5)


def test_load_dataset_labels_aggregate_and_skip(tmp_path: Path) -> None:
    """目录全量聚合：损坏 JSON / 缺 sample_token 跳过（宁缺勿假），其余读回。"""
    write_sample_label("tok1", [_box()], tmp_path)
    write_sample_label("tok2", [_box(label="truck", velocity=None)], tmp_path)
    (tmp_path / "bad.json").write_text("{corrupt", encoding="utf-8")  # 损坏跳过
    (tmp_path / "no_token.json").write_text(
        json.dumps({"boxes": []}), encoding="utf-8"  # 缺 sample_token 跳过
    )
    (tmp_path / "bad_box.json").write_text(
        json.dumps(
            {"sample_token": "tok3",
             "boxes": [{"detection_name": "x", "translation": ["a", "b", "c"]}]}
        ),
        encoding="utf-8",  # translation 非数值 → float() ValueError，整样本跳过
    )
    (tmp_path / "not_json.txt").write_text("hello", encoding="utf-8")  # 非 JSON 不匹配

    labels = load_dataset_labels(tmp_path)
    assert sorted(labels.keys()) == ["tok1", "tok2"]
    assert labels["tok1"][0].label == "car"
    assert labels["tok2"][0].label == "truck"
    # None velocity 经官方格式往返 = [0,0]（格式键必存纪律，见 docstring）
    assert labels["tok2"][0].velocity == (0.0, 0.0)

    assert load_dataset_labels(tmp_path / "empty") == {}  # 空目录 → 空 dict


def test_load_dataset_labels_accepts_single_file(tmp_path: Path) -> None:
    """labels 参数也可为单文件（smoke_nuscenes 回灌参数统一入口）。"""
    write_sample_label("tok9", [_box()], tmp_path)
    labels = load_dataset_labels(tmp_path / "tok9.json")
    assert list(labels.keys()) == ["tok9"]
