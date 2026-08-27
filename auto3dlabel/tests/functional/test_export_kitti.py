"""test_export_kitti：15 字段导出 + 底部中心 y 回写 + GT roundtrip + 类映射。"""

from __future__ import annotations

from typing import Any

import numpy as np

from auto3dlabel.configs.kitti import COCO_TO_KITTI
from auto3dlabel.export.kitti_label import build_label_file, line_from_box3d
from auto3dlabel.schema.box3d import Box3D
from auto3dlabel.tests.helpers.synth import gt_line


def _box(**kw: Any) -> Box3D:
    defaults: dict[str, Any] = dict(
        label="Car", confidence=0.85, cx=8.0, cy=1.4, cz=18.0,
        h=1.5, w=1.6, l=3.9, yaw_bev=np.pi / 2,  # → rotation_y=0
        x1=580.0, y1=170.0, x2=700.0, y2=230.0,
    )
    defaults.update(kw)
    return Box3D(**defaults)


def test_line_15_fields() -> None:
    parts = line_from_box3d(_box()).split()
    assert len(parts) == 15
    assert parts[0] == "Car"
    assert float(parts[8]) == 1.5 and float(parts[9]) == 1.6 and float(parts[10]) == 3.9


def test_y_bottom_center() -> None:
    """红线：内部 cy=1.4（体积中心）→ label 的 y = cy + h/2 = 2.15（底部中心）。"""
    parts = line_from_box3d(_box()).split()
    assert abs(float(parts[12]) - 2.15) < 1e-6


def test_rotation_y_in_line() -> None:
    """yaw=π/2 → rotation_y=0 出现在第 15 字段。"""
    parts = line_from_box3d(_box()).split()
    assert abs(float(parts[14]) - 0.0) < 1e-6


def test_gt_roundtrip_y_semantics() -> None:
    """GT 行（y 底部）→ from_gt_row → line_from_box3d 回写同值（闭环）。"""
    line = gt_line(x=8.0, y=-0.9, z=18.0, h=1.5, w=1.6, l=3.9, ry=0.0)
    parts = line.split()
    box = Box3D.from_gt_row(parts[0], *[float(v) for v in parts[8:15]])
    assert abs(box.cy - (-0.9 - 0.75)) < 1e-9  # cy = y - h/2
    out = line_from_box3d(box).split()
    assert abs(float(out[12]) - (-0.9)) < 1e-2  # 回写底部
    assert abs(float(out[11]) - 8.0) < 1e-2


def test_build_label_file_sorted_by_conf(tmp_path: Any) -> None:
    b_high = _box(confidence=0.9)
    b_low = _box(confidence=0.4, cx=1.0)
    path = build_label_file("000000", [b_low, b_high], tmp_path)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("Car 0.00 0 0.00 580.00")  # conf 高在前（同 2D 框坐标）
    assert abs(float(lines[1].split()[11]) - 1.0) < 1e-6  # b_low 的 cx


def test_build_label_file_empty(tmp_path: Any) -> None:
    path = build_label_file("000000", [], tmp_path)
    assert path.read_text() == ""


def test_coco_to_kitti_map() -> None:
    assert COCO_TO_KITTI["car"] == "Car"
    assert COCO_TO_KITTI["person"] == "Pedestrian"
    assert COCO_TO_KITTI["bicycle"] == "Cyclist"
    assert COCO_TO_KITTI["motorcycle"] == "Cyclist"  # 骑摩托车按 Cyclist 评测
    assert COCO_TO_KITTI["truck"] == "Truck"
    assert COCO_TO_KITTI["train"] == "Tram"
