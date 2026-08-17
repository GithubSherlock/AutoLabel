"""可视化 Tool —— 在图像上绘制边界框和 mask，保存结果。"""

from __future__ import annotations

import cv2
import numpy as np

from auto2dlabel.schema.annotation import Annotation, Bbox, Mask
from auto2dlabel.schema.task_plan import DEFAULT_MODEL
from auto2dlabel.tools import Path
from auto2dlabel.tools.base import Tool

# 类别颜色（BGR 格式，OpenCV 用）
CLASS_COLORS: dict[str, tuple[int, int, int]] = {}


def _get_color(label: str) -> tuple[int, int, int]:
    """为每个类别分配一个固定颜色。"""
    if label not in CLASS_COLORS:
        # 使用 HSV → BGR 生成区分度高的颜色
        import colorsys
        import hashlib

        h = int(hashlib.md5(label.encode()).hexdigest(), 16) % 360
        r, g, b = colorsys.hsv_to_rgb(h / 360, 0.8, 0.9)
        CLASS_COLORS[label] = (int(b * 255), int(g * 255), int(r * 255))
    return CLASS_COLORS[label]


def draw_bboxes(
    image: np.ndarray,
    bboxes: list[Bbox],
    show_label: bool = True,
    show_conf: bool = True,
    line_thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """在图像上绘制边界框。

    Args:
        image: BGR 格式的 numpy 数组 (H, W, 3)。
        bboxes: 边界框列表。
        show_label: 是否显示类别名。
        show_conf: 是否显示置信度。
        line_thickness: 线宽。
        font_scale: 字体大小。

    Returns:
        绘制后的图像（BGR）。
    """
    img = image.copy()
    h, w = img.shape[:2]

    for bbox in bboxes:
        color = _get_color(bbox.label)
        x1, y1, x2, y2 = bbox.xyxy

        # 限制坐标在图像范围内
        x1_i = max(0, int(x1))
        y1_i = max(0, int(y1))
        x2_i = min(w, int(x2))
        y2_i = min(h, int(y2))

        # 绘制矩形（旋转框画多边形角点连线）
        if abs(bbox.angle) < 1e-9:
            cv2.rectangle(img, (x1_i, y1_i), (x2_i, y2_i), color, line_thickness)
        else:
            from auto2dlabel.export.dota import rotated_corners

            corners = rotated_corners(
                bbox.x + bbox.width / 2, bbox.y + bbox.height / 2,
                bbox.width, bbox.height, bbox.angle,
            )
            pts = np.array(
                [[round(px), round(py)] for px, py in corners], dtype=np.int32
            ).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=line_thickness)

        # 标签文字
        parts = []
        if show_label:
            parts.append(bbox.label)
        if show_conf:
            parts.append(f"{bbox.confidence:.0%}")

        if parts:
            label_text = " ".join(parts)
            # 文字背景
            (text_w, text_h), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thickness
            )
            cv2.rectangle(
                img,
                (x1_i, y1_i - text_h - baseline - 4),
                (x1_i + text_w + 4, y1_i),
                color,
                -1,
            )
            cv2.putText(
                img,
                label_text,
                (x1_i + 2, y1_i - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                line_thickness,
                cv2.LINE_AA,
            )

    return img


def draw_masks(
    image: np.ndarray,
    masks: list[Mask],
    alpha: float = 0.4,
) -> np.ndarray:
    """在图像上叠加分割 mask。

    Args:
        image: BGR 格式的 numpy 数组 (H, W, 3)。
        masks: Mask 列表。
        alpha: 透明度 (0-1)。

    Returns:
        叠加后的图像（BGR）。
    """
    overlay = image.copy()

    for mask in masks:
        color = _get_color(mask.bbox.label)

        for polygon in mask.segmentation:
            if len(polygon) < 6:
                continue
            # polygon: [x1, y1, x2, y2, ...]
            pts = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(overlay, [pts], color)

    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def visualize_annotation(
    image_path: Path,
    annotation: Annotation,
    output_path: Path,
    draw_mask: bool = True,
) -> Path:
    """可视化一个标注结果并保存。

    Args:
        image_path: 原始图像路径。
        annotation: 标注数据。
        output_path: 输出路径。
        draw_mask: 是否绘制 mask。

    Returns:
        输出文件路径。
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    if annotation.masks and draw_mask:
        image = draw_masks(image, annotation.masks)

    if annotation.bboxes:
        image = draw_bboxes(image, annotation.bboxes)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return output_path


def _resolve_output_path(
    input_path: Path,
    output_parent_dir: Path,
) -> Path:
    """根据输入路径计算输出路径。

    - 单文件 /a/b/photo.jpg → output_parent_dir/output_photo.jpg
    - 目录 /a/b/images/    → output_parent_dir/output_images/
    """
    if input_path.is_file():
        output_name = f"output_{input_path.name}"
        return output_parent_dir / output_name
    else:
        output_name = f"output_{input_path.name}"
        return output_parent_dir / output_name


def _collect_images(source: Path) -> list[Path]:
    """收集待处理的图像文件列表。"""
    extensions = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}

    if source.is_file():
        if source.suffix.lower() in extensions:
            return [source]
        return []

    if source.is_dir():
        return sorted(
            p for p in source.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        )

    return []


class VisualizeTool(Tool):
    """可视化 Tool —— 在图像上绘制标注结果。"""

    name = "visualize"
    description = (
        "Draw bounding boxes and segmentation masks on images to visualize "
        "annotation results. Supports single images or batch directory processing."
    )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the image or directory.",
                },
                "annotation": {
                    "type": "object",
                    "description": "Annotation data dict with bboxes and masks.",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Directory to save the output image.",
                },
                "draw_mask": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to draw segmentation masks.",
                },
            },
            "required": ["image_path", "annotation", "output_dir"],
        }

    def forward(
        self,
        image_path: str,
        annotation: dict,
        output_dir: str,
        draw_mask: bool = True,
    ) -> str:
        """执行可视化。"""
        from auto2dlabel.schema.annotation import Bbox, Mask

        bboxes = [
            Bbox(**b) if isinstance(b, dict) else b
            for b in annotation.get("bboxes", [])
        ]
        masks = [
            Mask(
                bbox=Bbox(**m["bbox"]) if isinstance(m["bbox"], dict) else m["bbox"],
                segmentation=m.get("segmentation", []),
                area=m.get("area", 0.0),
            )
            for m in annotation.get("masks", [])
        ]

        ann = Annotation(
            image_path=image_path,
            bboxes=bboxes,
            masks=masks,
            review_flags=annotation.get("review_flags", []),
            metadata=annotation.get("metadata", {}),
        )

        input_path = Path(image_path)
        output_parent = Path(output_dir)
        output_path = _resolve_output_path(input_path, output_parent)

        return str(visualize_annotation(input_path, ann, output_path, draw_mask=draw_mask))


def register(registry=None) -> None:
    """向全局注册表注册此 Tool。"""
    from auto2dlabel.tools.registry import registry as reg

    target = registry or reg
    target.register(VisualizeTool())


# ============================================================
# 便捷函数：直接从检测结果绘制并保存
# ============================================================

def detect_and_visualize(
    source: str | Path,
    prompts: list[str],
    output_parent: str | Path | None = None,
    confidence_threshold: float = 0.1,
    iou_threshold: float = 0.3,
    model_name: str = DEFAULT_MODEL,
    with_segmentation: bool = False,
    seg_model_name: str = "sam2_l.pt",
) -> list[Path]:
    """检测图像中的目标并可视化，可选分割。

    一键完成 detect → [segment] → visualize 流程。

    Args:
        source: 图像路径或目录。
        prompts: 检测的文本 prompt 列表。
        output_parent: 输出根目录。默认 vis_outputs/。
        confidence_threshold: 置信度阈值。
        iou_threshold: IoU 阈值。
        model_name: 检测模型名（支持所有 model_catalog 中的模型）。
        with_segmentation: 是否对高置信度框运行分割。
        seg_model_name: 分割模型名: "fastsam" | "maskrcnn" | "sam2" | "sam3"。

    Returns:
        所有输出文件的 Path 列表。
    """
    import time as _time
    from datetime import datetime as _datetime

    from auto2dlabel.models.detection import create_detection_model
    from auto2dlabel.schema.annotation import Annotation, Bbox
    from auto2dlabel.tools.export import ExportTool

    _t0 = _time.time()
    _ts = _datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    source_path = Path(source).resolve()
    if output_parent is None:
        output_parent = Path("vis_outputs")
    else:
        output_parent = Path(output_parent).resolve()

    images = _collect_images(source_path)
    if not images:
        raise ValueError(f"未找到可处理的图像文件: {source}")

    det_model = create_detection_model(model_name, iou_threshold=iou_threshold)
    seg_model = None
    if with_segmentation:
        from auto2dlabel.models.segmentation import create_segmentation_model
        seg_model = create_segmentation_model(seg_model_name)

    output_files: list[Path] = []
    vis_dir = output_parent
    vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"检测 {len(images)} 张图像 → {vis_dir}")

    for img_path in images:
        stem = img_path.stem
        print(f"  {img_path.name} ... ", end="", flush=True)

        # 检测
        results = det_model.detect(str(img_path), prompts, confidence_threshold)
        ann = Annotation(image_path=str(img_path))
        try:
            from PIL import Image as _Image
            im = _Image.open(img_path)
            ann.image_size = (im.width, im.height)
        except Exception:
            pass
        for r in results:
            ann.add_bbox(Bbox(x=r.x, y=r.y, width=r.width, height=r.height,
                              label=r.label, confidence=r.confidence))
            if r.confidence < confidence_threshold:
                ann.flag_for_review(len(ann.bboxes) - 1)

        # 分割
        if seg_model and ann.bboxes:
            high_conf = [b for b in ann.bboxes if b.confidence >= 0.5]
            if high_conf:
                masks = seg_model.generate(str(img_path), high_conf)
                for m in masks:
                    ann.add_mask(m)
                print(f"{len(results)} det + {len(masks)} seg → ", end="", flush=True)
            else:
                print(f"{len(results)} det → ", end="", flush=True)
        else:
            print(f"{len(results)} det → ", end="", flush=True)

        # 保存可视化（带时间戳）
        vis_path = vis_dir / f"vis_{stem}_{_ts}.png"
        visualize_annotation(img_path, ann, vis_path, draw_mask=bool(ann.masks))
        output_files.append(vis_path)
        print(vis_path.name)

        # 导出 COCO JSON（与 CLI 格式一致）
        et = ExportTool()
        json_dir = Path("outputs")
        json_dir.mkdir(exist_ok=True)
        et.forward(
            annotations=[ann.to_dict()],
            output_path=str(json_dir / f"{stem}_{_ts}.json"),
            format="coco",
        )

        # HITL 分流
        from auto2dlabel.tools.hitl import export_triage, triage_annotations
        triage = triage_annotations([ann])
        export_triage(triage, output_dir="outputs", image_stem=stem, timestamp=_ts)

    elapsed = round(_time.time() - _t0, 3)
    print(f"完成! {len(output_files)} 张 → {vis_dir}")

    # 日志
    from auto2dlabel.tools.log import log_python_api_call
    log_python_api_call(
        image_path=str(source_path), prompts=prompts, results=output_files,
        elapsed=elapsed, model_name=model_name,
        confidence_threshold=confidence_threshold, iou_threshold=iou_threshold,
        annotation_type="instance_segmentation" if with_segmentation else "object_detection",
        timestamp=_ts,
    )
    return output_files
