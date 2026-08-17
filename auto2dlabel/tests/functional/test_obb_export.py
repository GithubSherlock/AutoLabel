"""OBB 旋转框导出测试（DOTA / YOLO-OBB）—— 零模型加载。

覆盖：rotated_corners 几何、dota 行格式、yolo_obb 行格式与 angle=0 HBB 等价、
angle 序列化 roundtrip（1e-6）、state/tool 反序列化带 angle。
"""

from __future__ import annotations

import math

import pytest

from auto2dlabel.agent.state import AgentState
from auto2dlabel.export.dota import export_dota, rotated_corners
from auto2dlabel.export.yolo import export_yolo, export_yolo_obb
from auto2dlabel.schema.annotation import Annotation, Bbox
from auto2dlabel.tests import Path
from auto2dlabel.tools.export import ExportTool


def test_rotated_corners_axis_aligned() -> None:
    """angle=0 → 4 角点即外接矩形角点（顺时针，按 atan2 排序）。"""
    cx, cy, w, h = 10.0, 20.0, 4.0, 2.0
    corners = rotated_corners(cx, cy, w, h, 0.0)
    assert corners == [(8.0, 19.0), (12.0, 19.0), (12.0, 21.0), (8.0, 21.0)]
    # 顺时针：相邻点距离递减检查（凸四边形、y 向下）
    assert len(corners) == 4


def test_rotated_corners_quarter_turn() -> None:
    """angle=π/2 → 宽高轴互换（width 轴转到 y 方向）。"""
    corners = rotated_corners(0.0, 0.0, 4.0, 2.0, math.pi / 2)
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    assert max(xs) - min(xs) == pytest.approx(2.0)  # 原 height 沿 x
    assert max(ys) - min(ys) == pytest.approx(4.0)  # 原 width 沿 y


def test_rotated_corners_45deg_square_extent() -> None:
    """边长 s 的正方形旋转 45° 后包围盒为 s·√2。"""
    s = 10.0
    corners = rotated_corners(0.0, 0.0, s, s, math.pi / 4)
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    assert max(xs) - min(xs) == pytest.approx(s * math.sqrt(2))
    assert max(ys) - min(ys) == pytest.approx(s * math.sqrt(2))


def test_dota_export_line_format(tmp_path: Path) -> None:
    """每行 <class_id> + 8 角点 + 0；angle=0 角点与外接矩形一致。"""
    ann = Annotation(image_path="aerial.jpg", image_size=(100, 100))
    ann.add_bbox(Bbox(x=10, y=20, width=40, height=10, label="plane", confidence=0.9))
    ann.add_bbox(Bbox(x=50, y=50, width=20, height=20, label="ship",
                      confidence=0.8, angle=math.pi / 4))

    export_dota([ann], tmp_path)

    assert (tmp_path / "classes.txt").read_text(encoding="utf-8") == "plane\nship"
    lines = (tmp_path / "aerial.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    # line 1: plane（angle=0）
    tokens = lines[0].split()
    assert tokens[0] == "0" and tokens[-1] == "0"
    coords = [float(t) for t in tokens[1:-1]]
    assert len(coords) == 8
    pts = [(coords[i], coords[i + 1]) for i in range(0, 8, 2)]
    assert set(pts) == {(10.0, 20.0), (50.0, 20.0), (50.0, 30.0), (10.0, 30.0)}

    # line 2: ship（angle=π/4），8 值齐全且非退化
    tokens2 = lines[1].split()
    assert tokens2[0] == "1" and tokens2[-1] == "0"
    coords2 = [float(t) for t in tokens2[1:-1]]
    assert len(coords2) == 8
    assert len(set(zip(coords2[0::2], coords2[1::2]))) == 4  # 4 个不同角点


def test_yolo_obb_line_format_and_hbb_equivalence(tmp_path: Path) -> None:
    """yolo_obb 行含 6 值；angle=0 时前 5 值与 export_yolo 完全一致。"""
    ann = Annotation(image_path="img.png", image_size=(100, 50))
    ann.add_bbox(Bbox(x=10, y=10, width=20, height=10, label="car", confidence=0.9))
    ann.add_bbox(Bbox(x=40, y=20, width=10, height=10, label="truck",
                      confidence=0.7, angle=0.3))

    export_yolo_obb([ann], tmp_path / "obb")
    export_yolo([ann], tmp_path / "hbb")

    obb_lines = (tmp_path / "obb" / "labels" / "img.txt").read_text(
        encoding="utf-8").splitlines()
    hbb_lines = (tmp_path / "hbb" / "labels" / "img.txt").read_text(
        encoding="utf-8").splitlines()

    assert len(obb_lines) == 2
    # angle=0 行与 HBB 行前 5 个 token 一致（第 6 值为 0.000000）
    assert obb_lines[0].split()[:5] == hbb_lines[0].split()
    assert obb_lines[0].split()[5] == "0.000000"
    # 旋转行：6 个值，angle 保留
    tokens = obb_lines[1].split()
    assert len(tokens) == 6
    assert float(tokens[5]) == pytest.approx(0.3, abs=1e-6)
    # cx cy w h 归一化
    assert float(tokens[1]) == pytest.approx(0.45, abs=1e-6)
    assert float(tokens[2]) == pytest.approx(0.5, abs=1e-6)


def test_bbox_angle_serialization_roundtrip() -> None:
    """angle 四处序列化路径均不丢角：to_dict / state / export tool。"""
    bbox = Bbox(x=1, y=2, width=3, height=4, label="ship",
                confidence=0.9, angle=-0.7853981633974483)
    d = bbox.to_dict()
    assert d["angle"] == pytest.approx(-0.7853981633974483, abs=1e-9)

    # AgentState.from_dict 反序列化
    ann = Annotation(image_path="a.jpg")
    ann.add_bbox(bbox)
    restored = AgentState.from_dict(
        {"annotations": [ann.to_dict()], "image_path": "a.jpg"}
    ).annotations[0].bboxes[0]
    assert restored.angle == pytest.approx(bbox.angle, abs=1e-9)

    # ExportTool._make_bbox 反序列化
    assert ExportTool._make_bbox(d).angle == pytest.approx(bbox.angle, abs=1e-9)


def test_bbox_angle_default_and_output() -> None:
    """默认 angle=0，to_dict 恒输出 angle 字段。"""
    bbox = Bbox(x=0, y=0, width=1, height=1, label="car")
    assert bbox.angle == 0.0
    assert bbox.to_dict()["angle"] == 0.0


def test_create_obb_model_dispatch() -> None:
    """create_obb_model：.pt 名 → UltralyticsOBBModel；非 .pt → 报错。"""
    from auto2dlabel.models.obb import UltralyticsOBBModel, create_obb_model

    for name in ("yolo11n-obb.pt", "yolo12n-obb.pt", "yolo26n-obb.pt"):
        assert isinstance(create_obb_model(name), UltralyticsOBBModel)
    try:
        create_obb_model("fasterrcnn_resnet50_fpn")
        raise AssertionError("应抛出 ValueError")
    except ValueError as e:
        assert "无法识别的 OBB 模型" in str(e)
