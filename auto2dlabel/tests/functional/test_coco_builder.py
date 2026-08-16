"""build_coco_dict 纯函数测试。

该函数是文件导出（export_coco）与 Web /api/export-coco 端点的单一事实源，
保证两条导出路径结果一致。
"""

from __future__ import annotations

from auto2dlabel.export.coco import build_coco_dict
from auto2dlabel.schema.annotation import Annotation, Bbox, Mask


def test_build_coco_basic() -> None:
    """纯 bbox 标注 → 图像/类别/标注结构正确。"""
    ann = Annotation(image_path="a.jpg", image_size=(100, 200))
    ann.add_bbox(Bbox(x=1, y=2, width=3, height=4, label="cat", confidence=0.5))

    coco = build_coco_dict([ann])
    assert coco["images"] == [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 200}]
    assert [c["name"] for c in coco["categories"]] == ["cat"]
    assert coco["annotations"][0]["bbox"] == [1, 2, 3, 4]
    assert coco["annotations"][0]["category_id"] == 1
    assert coco["annotations"][0]["score"] == 0.5


def test_build_coco_with_masks() -> None:
    """含 mask 时 segmentation/area 透传，mask 复用其 bbox 的类别。"""
    ann = Annotation(image_path="/data/test_image.jpg", image_size=(640, 480))
    car = Bbox(x=10, y=20, width=100, height=200, label="car", confidence=0.95)
    ann.add_bbox(car)
    ann.add_bbox(Bbox(x=300, y=400, width=50, height=60, label="person", confidence=0.78))
    ann.add_mask(Mask(
        bbox=car,
        segmentation=[[10, 20, 110, 20, 110, 220, 10, 220]],
        area=20000.0,
    ))

    coco = build_coco_dict([ann])
    assert len(coco["annotations"]) == 3  # 2 bbox + 1 mask

    mask_ann = [a for a in coco["annotations"] if "segmentation" in a][0]
    assert mask_ann["segmentation"] == [[10, 20, 110, 20, 110, 220, 10, 220]]
    assert mask_ann["area"] == 20000.0
    assert mask_ann["category_id"] == 1  # car 是第一个注册的类别


def test_build_coco_category_id_alignment() -> None:
    """两图共享类别时 category id 跨图对齐、不重复注册。"""
    a1 = Annotation(image_path="a.jpg", image_size=(10, 10))
    a1.add_bbox(Bbox(x=0, y=0, width=1, height=1, label="car", confidence=0.9))
    a2 = Annotation(image_path="b.jpg", image_size=(10, 10))
    a2.add_bbox(Bbox(x=0, y=0, width=1, height=1, label="car", confidence=0.8))
    a2.add_bbox(Bbox(x=2, y=2, width=1, height=1, label="dog", confidence=0.7))

    coco = build_coco_dict([a1, a2])
    cat_ids = {c["name"]: c["id"] for c in coco["categories"]}
    assert set(cat_ids) == {"car", "dog"}

    dog_ann = [a for a in coco["annotations"] if a["category_id"] == cat_ids["dog"]][0]
    assert dog_ann["image_id"] == 2
    # car 在两张图各一个，共享同一 category_id
    car_anns = [a for a in coco["annotations"] if a["category_id"] == cat_ids["car"]]
    assert [a["image_id"] for a in car_anns] == [1, 2]
