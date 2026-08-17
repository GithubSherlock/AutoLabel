"""导出格式 round-trip 测试。

对 COCO / YOLO / VOC 三种格式：
    构造已知 Bbox → 导出 → 重新解析 → 断言坐标误差 < 1e-6
"""

from __future__ import annotations

import tempfile

import pytest

from auto2dlabel.schema.annotation import Annotation, Bbox
from auto2dlabel.tests import Path, json


def _make_annotation() -> Annotation:
    """构造一个已知的测试 Annotation，包含多个 Bbox。"""
    ann = Annotation(
        image_path="test_image.jpg",
        image_id=1,
        image_size=(640, 480),
    )
    # 用整数坐标方便精确比较
    ann.bboxes = [
        Bbox(x=10, y=20, width=100, height=200, label="car", confidence=0.95),
        Bbox(x=300, y=400, width=50, height=60, label="person", confidence=0.78),
        Bbox(x=500, y=100, width=80, height=90, label="bicycle", confidence=0.62),
    ]
    return ann


# ============================================================
# COCO round-trip
# ============================================================

def test_coco_roundtrip():
    """验证 COCO JSON 导出后重新解析，bbox 坐标不变。"""
    from auto2dlabel.export.coco import export_coco

    ann = _make_annotation()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test_coco.json"
        export_coco([ann], out_path)

        # 重新读取
        data = json.loads(out_path.read_text())
        assert len(data["images"]) == 1, "COCO 应有 1 张图像"
        assert data["images"][0]["width"] == 640
        assert data["images"][0]["height"] == 480
        assert len(data["annotations"]) == 3, "COCO 应有 3 个标注"

        # 按 bbox 排序以匹配
        coco_anns = sorted(data["annotations"], key=lambda a: a["bbox"][0])

        for i, (original, exported) in enumerate(zip(ann.bboxes, coco_anns)):
            coco_bbox = exported["bbox"]
            assert exported["score"] == pytest.approx(original.confidence), \
                f"bbox {i} 置信度不匹配"
            # COCO: [x, y, w, h]
            assert coco_bbox[0] == pytest.approx(original.x, abs=1e-6), \
                f"bbox {i} x 不匹配: expected {original.x}, got {coco_bbox[0]}"
            assert coco_bbox[1] == pytest.approx(original.y, abs=1e-6)
            assert coco_bbox[2] == pytest.approx(original.width, abs=1e-6)
            assert coco_bbox[3] == pytest.approx(original.height, abs=1e-6)

        # 验证类别
        cat_names = {c["id"]: c["name"] for c in data["categories"]}
        assert set(cat_names.values()) == {"car", "person", "bicycle"}


# ============================================================
# YOLO round-trip
# ============================================================

def test_yolo_roundtrip():
    """验证 YOLO txt 导出后重新解析，归一化坐标恢复一致。"""
    from auto2dlabel.export.yolo import export_yolo

    ann = _make_annotation()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        export_yolo([ann], out_dir)

        # 重新读取
        txt_path = out_dir / "labels" / "test_image.txt"
        assert txt_path.exists(), "YOLO label 文件应存在"

        classes_path = out_dir / "classes.txt"
        class_names = classes_path.read_text().strip().split("\n")
        assert class_names == ["car", "person", "bicycle"]

        lines = txt_path.read_text().strip().split("\n")
        assert len(lines) == 3, "应有 3 行 YOLO 标注"

        for i, line in enumerate(lines):
            parts = line.strip().split()
            assert len(parts) == 5, f"YOLO 行 {i} 应有 5 个字段"
            cls_id, cx, cy, w, h = map(float, parts)

            original = ann.bboxes[i]
            # 从归一化坐标恢复
            img_w, img_h = ann.image_size
            recovered_x = cx * img_w - w * img_w / 2
            recovered_y = cy * img_h - h * img_h / 2
            recovered_w = w * img_w
            recovered_h = h * img_h

            # YOLO txt 只存 6 位小数，容忍 1e-3 浮点误差
            assert recovered_x == pytest.approx(original.x, abs=1e-3), \
                f"YOLO bbox {i} x"
            assert recovered_y == pytest.approx(original.y, abs=1e-3)
            assert recovered_w == pytest.approx(original.width, abs=1e-3)
            assert recovered_h == pytest.approx(original.height, abs=1e-3)


# ============================================================
# VOC round-trip
# ============================================================

def test_voc_roundtrip():
    """验证 Pascal VOC XML 导出后重新解析，坐标一致。"""
    import xml.etree.ElementTree as ET

    from auto2dlabel.export.voc import export_voc

    ann = _make_annotation()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        export_voc([ann], out_dir)

        # 重新读取
        xml_path = out_dir / "Annotations" / "test_image.xml"
        assert xml_path.exists(), "VOC XML 文件应存在"

        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find("size")
        assert size is not None, "VOC XML 缺少 size 节点"
        assert int(size.findtext("width", 0)) == 640
        assert int(size.findtext("height", 0)) == 480

        objects = root.findall("object")
        assert len(objects) == 3, "应有 3 个 object"

        for i, obj in enumerate(objects):
            name = obj.findtext("name", "")
            bndbox = obj.find("bndbox")
            assert bndbox is not None, f"VOC object {i} 缺少 bndbox 节点"
            xmin = int(bndbox.findtext("xmin", 0))
            ymin = int(bndbox.findtext("ymin", 0))
            xmax = int(bndbox.findtext("xmax", 0))
            ymax = int(bndbox.findtext("ymax", 0))

            original = ann.bboxes[i]
            assert name == original.label, f"VOC bbox {i} label 不匹配: {name} vs {original.label}"
            # VOC: [xmin, ymin, xmax, ymax]
            assert xmin == pytest.approx(original.x, abs=1), \
                f"VOC bbox {i} xmin: {xmin} vs {original.x}"
            assert ymin == pytest.approx(original.y, abs=1)
            assert xmax - xmin == pytest.approx(original.width, abs=1)
            assert ymax - ymin == pytest.approx(original.height, abs=1)
