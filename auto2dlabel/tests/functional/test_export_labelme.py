"""LabelMe JSON 导出 round-trip 测试。

对 LabelMe 格式：构造已知 Bbox/Mask → 导出 → json.loads 重新解析 → 断言结构
与坐标（保留浮点，误差 < 1e-6）。全部 mock 数据，不依赖图像文件。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, cast

import pytest

from auto2dlabel.schema.annotation import Annotation, Bbox, Mask


def _make_annotation() -> Annotation:
    """构造一个已知的测试 Annotation，包含 3 个 Bbox（与 roundtrip 测试同款）。"""
    ann = Annotation(
        image_path="test_image.jpg",
        image_id=1,
        image_size=(640, 480),
    )
    # 用整数坐标方便精确比较；不设 id → group_id 应为 null
    ann.bboxes = [
        Bbox(x=10, y=20, width=100, height=200, label="car", confidence=0.95),
        Bbox(x=300, y=400, width=50, height=60, label="person", confidence=0.78),
        Bbox(x=500, y=100, width=80, height=90, label="bicycle", confidence=0.62),
    ]
    return ann


def _make_annotation_with_mask() -> Annotation:
    """3 个 Bbox + 1 个四边形 mask；mask 的 bbox.id=0 覆盖 group_id int 分支。"""
    ann = _make_annotation()
    ann.masks = [
        Mask(
            bbox=Bbox(x=10, y=20, width=50, height=60, label="car", confidence=0.9, id=0),
            segmentation=[[10.0, 20.0, 60.0, 20.0, 60.0, 80.0, 10.0, 80.0]],
            area=0.0,  # 触发 __post_init__ 用 bbox 面积补齐
        )
    ]
    return ann


def _load(tmpdir: str, stem: str) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads((Path(tmpdir) / f"{stem}.json").read_text()))


class TestLabelMeBbox:
    def test_labelme_bbox_rectangle(self) -> None:
        """核心：3 个 bbox 导出为 rectangle shape，坐标浮点不取整。"""
        from auto2dlabel.export.labelme import export_labelme

        ann = _make_annotation()

        with tempfile.TemporaryDirectory() as tmpdir:
            export_labelme([ann], Path(tmpdir))

            data = _load(tmpdir, "test_image")
            assert data["version"] == "4.5.6"
            assert data["flags"] == {}
            assert data["imageData"] == ""
            assert data["imageWidth"] == 640
            assert data["imageHeight"] == 480
            assert data["imagePath"] == "test_image.jpg"

            shapes = data["shapes"]
            assert len(shapes) == 3, "应有 3 个 shape"

            # 按 x 排序以匹配 bbox 列表顺序
            shapes = sorted(shapes, key=lambda s: s["points"][0][0])
            for i, (original, shape) in enumerate(zip(ann.bboxes, shapes)):
                assert shape["shape_type"] == "rectangle", f"bbox {i} 应为 rectangle"
                assert shape["label"] == original.label, f"bbox {i} label 不匹配"
                assert shape["group_id"] is None, "未设 id 的 bbox 应为 null"
                assert shape["flags"] == {}
                assert len(shape["points"]) == 2
                x1, y1, x2, y2 = original.xyxy
                assert shape["points"][0] == pytest.approx([x1, y1], abs=1e-6), \
                    f"bbox {i} 左上角点不匹配"
                assert shape["points"][1] == pytest.approx([x2, y2], abs=1e-6), \
                    f"bbox {i} 右下角点不匹配"


class TestLabelMeMask:
    def test_labelme_mask_polygon(self) -> None:
        """mask 导出为 polygon shape，扁平点列按 2 个一组配对。"""
        from auto2dlabel.export.labelme import export_labelme

        ann = _make_annotation_with_mask()

        with tempfile.TemporaryDirectory() as tmpdir:
            export_labelme([ann], Path(tmpdir))

            data = _load(tmpdir, "test_image")
            shapes = data["shapes"]
            assert len(shapes) == 4, "3 bbox + 1 mask = 4 个 shape"

            poly = shapes[-1]
            assert poly["shape_type"] == "polygon"
            assert poly["label"] == "car"
            assert poly["group_id"] == 0, "mask 的 bbox.id 应透传为 group_id"
            expected = [[10, 20], [60, 20], [60, 80], [10, 80]]
            assert len(poly["points"]) == 4
            for pt, exp in zip(poly["points"], expected):
                assert pt == pytest.approx(exp, abs=1e-6), f"polygon 顶点 {pt} 不匹配"

    def test_labelme_bbox_and_mask_order(self) -> None:
        """shapes 顺序：bbox 在前、mask 在后。"""
        from auto2dlabel.export.labelme import export_labelme

        ann = _make_annotation_with_mask()

        with tempfile.TemporaryDirectory() as tmpdir:
            export_labelme([ann], Path(tmpdir))

            data = _load(tmpdir, "test_image")
            types = [s["shape_type"] for s in data["shapes"]]
            assert types == ["rectangle", "rectangle", "rectangle", "polygon"]


class TestLabelMeEdge:
    def test_labelme_empty_annotation(self) -> None:
        """空 annotation：文件写出且 shapes 为空。"""
        from auto2dlabel.export.labelme import export_labelme

        ann = Annotation(image_path="empty.jpg", image_size=(640, 480))

        with tempfile.TemporaryDirectory() as tmpdir:
            export_labelme([ann], Path(tmpdir))

            data = _load(tmpdir, "empty")
            assert data["shapes"] == []

    def test_labelme_multiple_images(self) -> None:
        """多张图像各写一个 JSON 文件。"""
        from auto2dlabel.export.labelme import export_labelme

        anns = [
            Annotation(
                image_path="img1.jpg", image_size=(100, 100),
                bboxes=[Bbox(x=1, y=2, width=3, height=4, label="a")],
            ),
            Annotation(
                image_path="img2.jpg", image_size=(200, 200),
                bboxes=[Bbox(x=5, y=6, width=7, height=8, label="b")],
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            export_labelme(anns, Path(tmpdir))

            d1 = _load(tmpdir, "img1")
            d2 = _load(tmpdir, "img2")
            assert d1["imagePath"] == "img1.jpg"
            assert d2["imagePath"] == "img2.jpg"
            assert d1["shapes"][0]["label"] == "a"
            assert d2["shapes"][0]["label"] == "b"

    def test_labelme_unknown_image_size(self) -> None:
        """image_size=(0,0) 时输出 0（labelme 读取端要求键存在，与三器兜底一致）。"""
        from auto2dlabel.export.labelme import export_labelme

        ann = Annotation(
            image_path="unknown_size.jpg",
            bboxes=[Bbox(x=1, y=2, width=3, height=4, label="a")],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            export_labelme([ann], Path(tmpdir))

            data = _load(tmpdir, "unknown_size")
            assert data["imageWidth"] == 0
            assert data["imageHeight"] == 0


class TestLabelMeTool:
    def test_labelme_via_export_tool(self) -> None:
        """走 ExportTool.forward 全链路，防 tools 层接线回归。"""
        from auto2dlabel.tools.export import ExportTool

        with tempfile.TemporaryDirectory() as tmpdir:
            out = ExportTool().forward(
                [{
                    "image_path": "via_tool.jpg",
                    "image_size": [640, 480],
                    "bboxes": [
                        {"x": 10, "y": 20, "width": 100, "height": 200,
                         "label": "car", "confidence": 0.9},
                    ],
                }],
                tmpdir,
                format="labelme",
            )

            data = json.loads((Path(out) / "via_tool.json").read_text())
            assert len(data["shapes"]) == 1
            assert data["shapes"][0]["shape_type"] == "rectangle"
            assert data["imageWidth"] == 640


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
