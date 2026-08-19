"""MOT16/20 导出测试 —— 行格式、跳过无 ID、ExportTool 接线（round-trip）。"""

from __future__ import annotations

import tempfile

from auto2dlabel.export.mot import export_mot
from auto2dlabel.schema.annotation import Annotation, Bbox
from auto2dlabel.tests import Path


def _tracked_ann(image_path: str, bboxes: list[Bbox]) -> Annotation:
    ann = Annotation(image_path=image_path, image_size=(640, 480))
    for b in bboxes:
        ann.add_bbox(b)
    return ann


def test_export_mot_two_frames_roundtrip() -> None:
    """2 帧 × 2 轨迹：frame 从 1 起，坐标/置信度格式与 MOT 规范一致。"""
    ann1 = _tracked_ann("f1.jpg", [
        Bbox(x=10.4, y=20.2, width=100.0, height=200.5,
             label="person", confidence=0.95, track_id=0),
        Bbox(x=300.0, y=400.0, width=50.0, height=60.0,
             label="person", confidence=0.8, track_id=1),
    ])
    ann2 = _tracked_ann("f2.jpg", [
        Bbox(x=12.0, y=22.0, width=100.0, height=200.5,
             label="person", confidence=0.93, track_id=0),
        Bbox(x=301.0, y=401.0, width=50.0, height=60.0,
             label="person", confidence=0.77, track_id=1),
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "mot.txt"
        export_mot([ann1, ann2], out)

        lines = out.read_text().strip().splitlines()
        assert lines == [
            "1,0,10.4,20.2,100.0,200.5,0.950000,-1,1",
            "1,1,300.0,400.0,50.0,60.0,0.800000,-1,1",
            "2,0,12.0,22.0,100.0,200.5,0.930000,-1,1",
            "2,1,301.0,401.0,50.0,60.0,0.770000,-1,1",
        ]


def test_export_mot_skips_untracked_bboxes() -> None:
    """无 track_id 的 bbox 跳过（MOT 格式无位置容纳未跟踪检测）。"""
    ann = _tracked_ann("f1.jpg", [
        Bbox(x=10.0, y=20.0, width=100.0, height=200.0,
             label="person", confidence=0.9, track_id=0),
        Bbox(x=300.0, y=400.0, width=50.0, height=60.0,
             label="person", confidence=0.3),  # track_id=None
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "mot.txt"
        export_mot([ann], out)

        lines = out.read_text().strip().splitlines()
        assert len(lines) == 1
        assert lines[0].startswith("1,0,")


def test_export_mot_empty_annotations() -> None:
    """无跟踪框时输出空文件（不报错）。"""
    ann = _tracked_ann("f1.jpg", [])
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "mot.txt"
        export_mot([ann], out)
        assert out.exists()
        assert out.read_text() == ""


def test_export_tool_mot_format_roundtrip() -> None:
    """经 ExportTool format="mot"：dict → _make_bbox 恢复 track_id → txt 行。"""
    from auto2dlabel.tools.export import ExportTool

    ann = _tracked_ann("f1.jpg", [
        Bbox(x=10.4, y=20.2, width=100.0, height=200.5,
             label="person", confidence=0.95, track_id=0),
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "via_tool.txt"
        tool = ExportTool()
        tool.forward(annotations=[ann.to_dict()], output_path=str(out), format="mot")

        lines = out.read_text().strip().splitlines()
        assert lines == ["1,0,10.4,20.2,100.0,200.5,0.950000,-1,1"]


def test_coco_export_preserves_track_id() -> None:
    """回归：COCO 导出从 Bbox 对象构造 dict（绕过 to_dict），track_id 曾静默丢失。"""
    from auto2dlabel.export.coco import build_coco_dict

    ann = _tracked_ann("f1.jpg", [
        Bbox(x=10.4, y=20.2, width=100.0, height=200.5,
             label="person", confidence=0.95, track_id=3),
        Bbox(x=300.0, y=400.0, width=50.0, height=60.0,
             label="person", confidence=0.3),  # track_id=None → 字段省略
    ])

    coco = build_coco_dict([ann])
    tracked = [a for a in coco["annotations"] if "track_id" in a]
    assert len(tracked) == 1
    assert tracked[0]["track_id"] == 3
    # 无 track_id 的框不带该键（与 Bbox.to_dict 行为一致）
    untracked = [a for a in coco["annotations"] if "track_id" not in a]
    assert len(untracked) == 1
    assert untracked[0]["score"] == 0.3
