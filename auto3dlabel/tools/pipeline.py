"""单帧管线组装：detect → SAM2 mask → 反投影 → 聚类 → 拟合 → FrameResult。

检测/分割模型经参数注入（测试传 Fake 零真实权重铁律）；
默认模型工厂见 configs（DEFAULT_DET_MODEL/DEFAULT_SEG_MODEL）。
"""

from __future__ import annotations

from pathlib import Path

from auto2dlabel.models.detection import DetectionModel, create_detection_model
from auto2dlabel.models.segmentation import SegmentationModel, create_segmentation_model
from auto2dlabel.schema.annotation import Bbox
from auto2dlabel.tools.device import disable_tf32

from auto3dlabel.configs.kitti import (
    COCO_TO_KITTI,
    DEFAULT_CONF,
    DEFAULT_DET_MODEL,
    DEFAULT_IOU,
    DEFAULT_SEG_MODEL,
)
from auto3dlabel.schema.box3d import FrameResult, KittiFrame
from auto3dlabel.tools.backproject import backproject_semantics
from auto3dlabel.tools.cluster import cluster_instance
from auto3dlabel.tools.fit import fit_box3d
from auto3dlabel.tools.viz import draw_bev, draw_projection_check


def annotate_frame(
    frame: KittiFrame,
    prompts: list[str],
    det_model: DetectionModel | None = None,
    seg_model: SegmentationModel | None = None,
    confidence: float = DEFAULT_CONF,
    viz: bool = True,
    out_dir: str | Path | None = None,
) -> FrameResult:
    """KITTI 单帧 → 3D bbox 初稿（代码级直跑入口，同 auto2dlabel cli_execute 定位）。"""
    disable_tf32()
    image = str(frame.image_path)
    if det_model is None:
        det_model = create_detection_model(DEFAULT_DET_MODEL, iou_threshold=DEFAULT_IOU)
    if seg_model is None:
        seg_model = create_segmentation_model(DEFAULT_SEG_MODEL)

    dets = det_model.detect(image, prompts, confidence_threshold=confidence)
    result = FrameResult(frame_id=frame.frame_id)
    if not dets:
        return result

    bboxes = [
        Bbox(x=d.x, y=d.y, width=d.width, height=d.height, label=d.label, confidence=d.confidence)
        for d in dets
    ]
    masks = seg_model.generate(image, bboxes, mode="box")
    back = backproject_semantics(frame, [m.segmentation for m in masks])
    result.dropped_no_points = back.dropped

    for i in range(len(masks)):
        label_coco = str(dets[i].label)
        kitti_label = COCO_TO_KITTI.get(label_coco, label_coco)
        pts = back.points_of(i)
        if len(pts) == 0:
            result.warnings.append(f"实例{i}({label_coco}) mask 内零 LiDAR 点，丢弃")
            continue
        cluster = cluster_instance(pts, kitti_label)
        if cluster is None:
            result.warnings.append(f"实例{i}({label_coco}) 簇太小({len(pts)}点)，丢弃")
            continue
        box = fit_box3d(
            cluster.points_cam,
            kitti_label,
            float(dets[i].confidence),
            float(dets[i].x),
            float(dets[i].y),
            float(dets[i].x + dets[i].width),
            float(dets[i].y + dets[i].height),
            adhesion_flag=cluster.adhesion_flag,
        )
        if box is None:
            result.warnings.append(f"实例{i}({label_coco}) 拟合退化，丢弃")
            continue
        result.boxes3d.append(box)

    if viz and out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        result.proj_check_path = draw_projection_check(frame, out / f"{frame.frame_id}_proj.png")
        result.bev_path = draw_bev(
            frame,
            out / f"{frame.frame_id}_bev.png",
            points_cam=back.points_cam if len(back.points_cam) else None,
            boxes=result.boxes3d,
            gt_boxes=frame.load_gt3d(),
        )
    return result
