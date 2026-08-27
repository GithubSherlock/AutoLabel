"""test_export3d：ExportTool——kitti 分支与 build_label_file 逐字节一致 / nuscenes
提交自检 / input_schema / register 纪律 / NusBox.from_dict 往返（零真实权重）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from auto2dlabel.tools.base import Tool
from auto2dlabel.tools.registry import ToolRegistry
from auto3dlabel.export.kitti_label import build_label_file
from auto3dlabel.schema.box3d import Box3D
from auto3dlabel.schema.nuscenes_box import NusBox
from auto3dlabel.tools.export import ExportTool


def _box(**kw: Any) -> Box3D:
    defaults: dict[str, Any] = dict(
        label="Car", confidence=0.85, cx=8.0, cy=1.4, cz=18.0,
        h=1.5, w=1.6, l=3.9, yaw_bev=np.pi / 2,  # → rotation_y=0
        x1=580.0, y1=170.0, x2=700.0, y2=230.0,
    )
    defaults.update(kw)
    return Box3D(**defaults)


def _nusbox(**kw: Any) -> NusBox:
    defaults: dict[str, Any] = dict(
        label="car", confidence=0.62, translation=(1.0, 2.0, 3.0),
        size=(1.8, 4.2, 1.5), quaternion=(0.9, 0.1, 0.2, 0.3), velocity=(0.5, -1.2),
    )
    defaults.update(kw)
    return NusBox(**defaults)


def test_kitti_dict_matches_build_label_file(tmp_path: Path) -> None:
    """dict 列表（乱序 conf）→ 与 build_label_file 逐字节一致（含 conf 降序 + 行尾换行）。"""
    boxes = [_box(confidence=0.4, cx=1.0), _box(confidence=0.9, cx=12.0)]
    expected = build_label_file("000000", boxes, tmp_path / "ref")
    got = ExportTool().forward(
        annotations=[b.to_dict() for b in boxes],
        output_path=str(tmp_path / "out" / "nested" / "000000.txt"),
    )
    assert Path(got) == (tmp_path / "out" / "nested" / "000000.txt").resolve()
    assert Path(got).read_text(encoding="utf-8") == expected.read_text(encoding="utf-8")
    lines = Path(got).read_text().strip().splitlines()
    assert len(lines) == 2
    assert float(lines[0].split()[11]) == 12.0  # conf 高在前


def test_kitti_accepts_box3d_objects(tmp_path: Path) -> None:
    """Box3D 对象直通（CLI chat 直接喂对象，不经 to_dict）。"""
    got = ExportTool().forward(
        annotations=[_box()], output_path=str(tmp_path / "obj.txt")
    )
    assert Path(got).read_text().startswith("Car 0.00 0 0.00 580.00 170.00 700.00 230.00")


def test_kitti_empty(tmp_path: Path) -> None:
    """空列表 → 空文件（与 build_label_file 空语义一致）。"""
    got = ExportTool().forward(annotations=[], output_path=str(tmp_path / "empty.txt"))
    assert Path(got).read_text(encoding="utf-8") == ""


def test_nuscenes_dict_of_dicts(tmp_path: Path) -> None:
    """{sample_token: [NusBox dict]} → 官方提交 JSON（自检通过 + 父目录自建）。"""
    out = tmp_path / "sub" / "results.json"
    got = ExportTool().forward(
        annotations={"s1": [_nusbox().to_dict()]},
        output_path=str(out),
        format="nuscenes",
    )
    sub = json.loads(Path(got).read_text(encoding="utf-8"))
    assert sub["meta"]["use_lidar"] is True
    assert sub["results"]["s1"][0]["detection_name"] == "car"
    assert sub["results"]["s1"][0]["velocity"] == [0.5, -1.2]
    assert sub["results"]["s1"][0]["attribute_name"] == "vehicle.moving"


def test_nuscenes_accepts_nusbox_objects(tmp_path: Path) -> None:
    """NusBox 对象直通（from_dict 分支旁路）。"""
    got = ExportTool().forward(
        annotations={"s2": [_nusbox()]},
        output_path=str(tmp_path / "obj_nus.json"),
        format="nuscenes",
    )
    sub = json.loads(Path(got).read_text(encoding="utf-8"))
    assert sub["results"]["s2"][0]["translation"] == [1.0, 2.0, 3.0]


def test_nuscenes_invalid_class_raises(tmp_path: Path) -> None:
    """非法类 → write_submission 自检 ValueError（宁缺勿假红线）。"""
    with pytest.raises(ValueError, match="自检未通过"):
        ExportTool().forward(
            annotations={"s1": [_nusbox(label="FlyingSaucer").to_dict()]},
            output_path=str(tmp_path / "bad.json"),
            format="nuscenes",
        )


def test_unknown_format_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="未知 3D 导出格式"):
        ExportTool().forward(annotations=[], output_path=str(tmp_path / "x.txt"), format="yolo")


def test_input_schema() -> None:
    schema = ExportTool().input_schema
    assert schema["properties"]["format"]["enum"] == ["kitti", "nuscenes"]
    assert schema["properties"]["format"]["default"] == "kitti"
    assert schema["required"] == ["annotations", "output_path"]


def test_register_into_registry() -> None:
    """register 进显式 registry（非全局单例）；ExportTool 是 Tool 子类。"""
    reg = ToolRegistry()
    from auto3dlabel.tools.export import register

    register(reg)
    tool = reg.get("export_annotations")
    assert isinstance(tool, Tool)
    assert "export_annotations" in reg.list_tools()


def test_nusbox_from_dict_roundtrip() -> None:
    """to_dict → from_dict 往返（velocity 有值/None 两分支；track_id 只读不输出）。"""
    rt = NusBox.from_dict(_nusbox().to_dict())
    assert rt == _nusbox()
    no_v = _nusbox(velocity=None)
    assert NusBox.from_dict(no_v.to_dict()) == no_v
    assert NusBox.from_dict({"detection_name": "bus", "track_id": "i1"}).track_id == "i1"
    assert NusBox.from_dict(_nusbox(track_id="i2").to_dict()).track_id is None
