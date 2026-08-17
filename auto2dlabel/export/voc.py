"""Pascal VOC XML 格式导出。"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from auto2dlabel.export import Path
from auto2dlabel.schema.annotation import Annotation


def export_voc(annotations: list[Annotation], output_dir: Path) -> None:
    """将标注列表导出为 Pascal VOC XML 格式。

    每张图像对应一个同名的 .xml 文件。

    Args:
        annotations: Annotation 对象列表。
        output_dir: 输出目录（会在其中创建 Annotations/ 子目录）。
    """
    ann_dir = output_dir / "Annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    for ann in annotations:
        stem = Path(ann.image_path).stem
        xml_path = ann_dir / f"{stem}.xml"

        root = ET.Element("annotation")

        ET.SubElement(root, "folder").text = str(Path(ann.image_path).parent)
        ET.SubElement(root, "filename").text = Path(ann.image_path).name
        ET.SubElement(root, "path").text = ann.image_path

        source = ET.SubElement(root, "source")
        ET.SubElement(source, "database").text = "AutoLabel"

        size = ET.SubElement(root, "size")
        w, h = ann.image_size if ann.image_size != (0, 0) else (0, 0)
        ET.SubElement(size, "width").text = str(int(w))
        ET.SubElement(size, "height").text = str(int(h))
        ET.SubElement(size, "depth").text = "3"

        ET.SubElement(root, "segmented").text = "1" if ann.masks else "0"

        for bbox in ann.bboxes:
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = bbox.label
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"

            bndbox = ET.SubElement(obj, "bndbox")
            x1, y1, x2, y2 = bbox.xyxy
            ET.SubElement(bndbox, "xmin").text = str(int(x1))
            ET.SubElement(bndbox, "ymin").text = str(int(y1))
            ET.SubElement(bndbox, "xmax").text = str(int(x2))
            ET.SubElement(bndbox, "ymax").text = str(int(y2))

        # 格式化输出
        ET.indent(root, space="  ")
        tree = ET.ElementTree(root)
        tree.write(str(xml_path), encoding="utf-8", xml_declaration=True)
