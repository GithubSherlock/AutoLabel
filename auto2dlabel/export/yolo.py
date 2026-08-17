"""YOLO txt 格式导出（HBB + OBB）。"""

from __future__ import annotations

from auto2dlabel.export import Path
from auto2dlabel.schema.annotation import Annotation


def _collect_classes(annotations: list[Annotation]) -> tuple[list[str], dict[str, int]]:
    """收集类别名 → ID 映射（按首次出现顺序）。"""
    class_names: list[str] = []
    class_name_to_id: dict[str, int] = {}
    for ann in annotations:
        for bbox in ann.bboxes:
            if bbox.label not in class_name_to_id:
                class_name_to_id[bbox.label] = len(class_names)
                class_names.append(bbox.label)
    return class_names, class_name_to_id


def export_yolo(annotations: list[Annotation], output_dir: Path) -> None:
    """将标注列表导出为 YOLO txt 格式。

    每张图像对应一个同名的 .txt 文件，每行一个标注：
    <class_id> <cx> <cy> <w> <h>
    坐标归一化到 [0, 1]。

    同时生成一个 classes.txt 文件记录类别名称。

    Args:
        annotations: Annotation 对象列表。
        output_dir: 输出目录（会在其中创建 labels/ 子目录）。
    """
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # 收集类别名称 → 类别 ID
    class_names, class_name_to_id = _collect_classes(annotations)

    # 写入 classes.txt
    (output_dir / "classes.txt").write_text("\n".join(class_names))

    img_w, img_h = 1.0, 1.0  # 默认归一化坐标的除数值

    for ann in annotations:
        # 获取图像尺寸用于归一化
        if ann.image_size != (0, 0):
            img_w, img_h = ann.image_size

        stem = Path(ann.image_path).stem
        txt_path = labels_dir / f"{stem}.txt"

        lines = []
        for bbox in ann.bboxes:
            cls_id = class_name_to_id[bbox.label]
            # 转换为 YOLO 格式: [cx, cy, w, h] 归一化
            cx = (bbox.x + bbox.width / 2) / img_w
            cy = (bbox.y + bbox.height / 2) / img_h
            nw = bbox.width / img_w
            nh = bbox.height / img_h

            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        txt_path.write_text("\n".join(lines))


def export_yolo_obb(annotations: list[Annotation], output_dir: Path) -> None:
    """将旋转框标注导出为 YOLO-OBB txt 格式。

    每张图像对应一个同名的 .txt 文件，每行一个标注：
    <class_id> <cx> <cy> <w> <h> <angle>
    cx/cy/w/h 归一化到 [0, 1]，angle 为弧度（width 轴相对 x 轴，(-π/2, π/2]），
    与 ultralytics xywhr 约定零转换；angle=0 时几何与 export_yolo 一致。

    同时生成 classes.txt 文件记录类别名称。

    Args:
        annotations: Annotation 对象列表（bboxes 带 angle）。
        output_dir: 输出目录（会在其中创建 labels/ 子目录）。
    """
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    class_names, class_name_to_id = _collect_classes(annotations)
    (output_dir / "classes.txt").write_text("\n".join(class_names))

    img_w, img_h = 1.0, 1.0  # 默认归一化坐标的除数值

    for ann in annotations:
        if ann.image_size != (0, 0):
            img_w, img_h = ann.image_size

        stem = Path(ann.image_path).stem
        txt_path = labels_dir / f"{stem}.txt"

        lines = []
        for bbox in ann.bboxes:
            cls_id = class_name_to_id[bbox.label]
            cx = (bbox.x + bbox.width / 2) / img_w
            cy = (bbox.y + bbox.height / 2) / img_h
            nw = bbox.width / img_w
            nh = bbox.height / img_h

            lines.append(
                f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f} {bbox.angle:.6f}"
            )

        txt_path.write_text("\n".join(lines))
