"""3D Agent 工具（同名策略：Detect3DTool.name="detect_objects" →
orchestrator 防重复/摘要/质量评估挂点零修改复用）。

forward 返回 list[dict]（Box3D.to_dict）：orchestrator _summarize_tool_result 的 dict 分支原样透传，
evaluate_detections 只消费 label 键——3D dict 兼容零改动。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from auto2dlabel.tools.base import Tool
from auto2dlabel.tools.registry import ToolRegistry

from auto3dlabel.schema.box3d import KittiFrame
from auto3dlabel.tools.pipeline import annotate_frame
from auto3dlabel.tools.viz import draw_bev

if TYPE_CHECKING:
    from auto2dlabel.models.detection import DetectionModel
    from auto2dlabel.models.segmentation import SegmentationModel


def attach_coco_names(results: list[dict]) -> list[dict]:
    """dict 加 name 键（COCO 名），供 evaluate_detections prompt 覆盖检查匹配。

    KITTI 类名（Pedestrian 不含 person 子串）按 KITTI_TO_COCO 映射，
    否则 _match_prompt 子串匹配会误报缺类。
    """
    from auto3dlabel.configs.kitti import KITTI_TO_COCO

    for d in results:
        d["name"] = KITTI_TO_COCO.get(str(d.get("label", "")), str(d.get("label", "")).lower())
    return results


class Detect3DTool(Tool):
    """KITTI 单帧 3D 检测工具（2D 检测 → SAM2 mask → 反投影 → 聚类拟合）。"""

    name = "detect_objects"
    description = (
        "Detect 3D objects in a KITTI frame (LiDAR + camera). Runs the full pipeline: "
        "2D detection → SAM2 mask → LiDAR backprojection → DBSCAN clustering → 3D box fitting. "
        "Returns 3D boxes with cx/cy/cz center, h/w/l size, rotation_y, and fit_points "
        "(number of LiDAR points supporting the fit). Use this tool when the user asks to "
        "annotate 3D objects (cars, pedestrians, cyclists) in a KITTI frame."
    )

    def __init__(
        self,
        frame: KittiFrame,
        det_model_name: str | None = None,
        seg_model_name: str | None = None,
        out_dir: str | Path | None = None,
    ) -> None:
        self.frame = frame
        self.det_model_name = det_model_name
        self.seg_model_name = seg_model_name
        self.out_dir = out_dir
        self.last_retried = False  # orchestrator 质量评估读取（同 2D DetectionTool）
        self.last_retry_threshold: float | None = None

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "prompts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Object classes to detect, e.g. ['car', 'person', 'bicycle']. "
                        "English COCO names."
                    ),
                },
                "confidence_threshold": {
                    "type": "number",
                    "default": 0.3,
                    "description": "Minimum confidence score (0-1) for the 2D detector.",
                },
            },
            "required": ["prompts"],
        }

    def _models(self) -> tuple[DetectionModel | None, SegmentationModel | None]:
        from auto2dlabel.models.detection import create_detection_model
        from auto2dlabel.models.segmentation import create_segmentation_model

        det = create_detection_model(self.det_model_name) if self.det_model_name else None
        seg = create_segmentation_model(self.seg_model_name) if self.seg_model_name else None
        return det, seg

    def forward(
        self,
        prompts: list[str] | None = None,
        confidence_threshold: float = 0.3,
        **kwargs: Any,
    ) -> list[dict]:
        """跑完整 3D 管线；0 框自动降阈值重试一次（同 2D 工具，对 Agent Loop 透明）。

        基类 forward 为 **kwargs 调用协议（orchestrator 以 tool.forward(**arguments) 调用），
        故具名参数全部给默认值 + **kwargs 吸收其余，规避 override 不兼容。
        """
        from auto2dlabel.models.detection import detect_with_retry

        prompts = prompts or ["car"]
        det_model, seg_model = self._models()  # None 时 annotate_frame 内部走默认工厂

        def _run(conf: float) -> list[dict]:
            result = annotate_frame(
                self.frame,
                prompts,
                det_model=det_model,
                seg_model=seg_model,
                confidence=conf,
                viz=False,
            )
            return [b.to_dict() for b in result.boxes3d]

        results, self.last_retried, self.last_retry_threshold = detect_with_retry(
            lambda conf: _run(conf), confidence_threshold
        )
        # name 键携带 COCO 名（见 attach_coco_names docstring）
        attach_coco_names(results)
        results.sort(key=lambda d: -float(d.get("confidence", 0)))
        return results


class Visualize3DTool(Tool):
    """生成 BEV 鸟瞰图（预测框 + GT 框对照），供 LLM/用户直观复核。"""

    name = "visualize_bev"
    description = (
        "Render a bird's-eye-view image of the KITTI frame with predicted 3D boxes "
        "(green) and ground-truth boxes (blue). Returns the output image path. "
        "Use this tool when the user asks to see or check the 3D annotation result."
    )

    def __init__(self, frame: KittiFrame, out_dir: str | Path) -> None:
        self.frame = frame
        self.out_dir = out_dir

    @property
    def input_schema(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    def forward(self, **kwargs: object) -> dict:
        out = Path(self.out_dir) / f"{self.frame.frame_id}_bev.png"
        path = draw_bev(
            self.frame,
            out,
            boxes=_boxes_from_kwargs(kwargs),
            gt_boxes=self.frame.load_gt3d(),
        )
        return {"success": True, "path": str(path)}


def _boxes_from_kwargs(kwargs: dict[str, object]) -> list:
    from auto3dlabel.schema.box3d import Box3D

    boxes = kwargs.get("boxes", [])
    if not isinstance(boxes, list):
        return []
    return [
        b if isinstance(b, Box3D) else Box3D.from_dict(b) if isinstance(b, dict) else b
        for b in boxes
    ]


def build_3d_registry(
    frame: KittiFrame,
    det_model_name: str | None = None,
    seg_model_name: str | None = None,
    out_dir: str | Path | None = None,
) -> ToolRegistry:
    """独立 ToolRegistry 实例（非全局单例，规避与 2D chat 同名冲突）。"""
    registry = ToolRegistry()
    registry.register(Detect3DTool(frame, det_model_name, seg_model_name, out_dir))
    registry.register(Visualize3DTool(frame, Path(out_dir) if out_dir else Path("outputs")))
    return registry
