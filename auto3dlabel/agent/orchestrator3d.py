"""3D Agent 编排薄壳：复用 AgentOrchestrator 循环（3D 工具注册表注入），
事后从 state.tool_calls 提取 3D 结果重建 Box3D + fit_points 判据写 metadata。

同名工具策略：Detect3DTool.name="detect_objects" → orchestrator 的防重复调用、
工具摘要（dict 分支透传）、evaluate_detections 质量挂点全部零修改复用。
state.annotations（2D _sync_annotations 残留）不消费——3D 结果只从 tool_calls 提取。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from auto2dlabel.agent.orchestrator import AgentOrchestrator
from auto2dlabel.agent.state import AgentState
from auto3dlabel.agent.tools3d import build_3d_registry
from auto3dlabel.configs.kitti import MIN_FIT_POINTS
from auto3dlabel.schema.box3d import Box3D, KittiFrame


def extract_boxes3d(state: AgentState) -> list[Box3D]:
    """AgentState.tool_calls → Box3D 列表（detect_objects 的摘要 objects 直通重建）。

    摘要结构：{"success", "count", "objects": [Box3D dict]}；失败/异常时返回 []。
    """
    boxes: list[Box3D] = []
    for tc in state.tool_calls:
        if tc.get("tool_name") != "detect_objects":
            continue
        result = tc.get("result", {})
        if not isinstance(result, dict):
            continue
        items = result.get("objects") or result.get("data") or []
        for item in items:
            if isinstance(item, Box3D):
                boxes.append(item)
            elif isinstance(item, dict) and item.get("label"):
                try:
                    boxes.append(Box3D.from_dict(item))
                except (TypeError, ValueError):
                    continue
    return boxes


def _box3d_to_bbox2d(d: dict) -> Any:
    """3D dict → 2D Bbox 视图（x1..y2 投影框），供 orchestrator 2D 挂点消费。"""
    from auto2dlabel.schema.annotation import Bbox

    return Bbox(
        x=float(d.get("x1", 0.0)),
        y=float(d.get("y1", 0.0)),
        width=max(0.0, float(d.get("x2", 0.0)) - float(d.get("x1", 0.0))),
        height=max(0.0, float(d.get("y2", 0.0)) - float(d.get("y1", 0.0))),
        label=str(d.get("label", "")),
        confidence=float(d.get("confidence", 0.0)),
    )


def _wrap_evaluate_retry(orchestrator: AgentOrchestrator, registry: Any) -> None:
    """包装 _prepare_evaluate：retry 的 detect_fn 返回 2D Bbox 视图。

    2D orchestrator 的 _sync_annotations 把 _evaluate_retry_bboxes 直接
    ann.add_bbox（Bbox 语义硬编码）；3D 工具 forward 返回 Box3D dict 会崩
    （dict 无 .id）。故 evaluate 重试仍跑 3D 管线（低阈值多检出），但结果以
    2D 投影框视图回填——orchestrator 全链零修改，3D 主结果以 tool_calls 为准。
    """
    orig_prepare = orchestrator._prepare_evaluate

    def _prepare_3d(image_path: str, arguments: dict, quality: Any, base_threshold: float) -> None:
        orig_prepare(image_path, arguments, quality, base_threshold)
        tool3d = registry.get("detect_objects")
        if tool3d is None:
            return

        def _retry_3d_as_2d(conf: float) -> list[Any]:
            results = tool3d.forward(
                prompts=list(arguments.get("prompts", [])),
                confidence_threshold=conf,
            )
            return [_box3d_to_bbox2d(d) for d in results]

        assert orchestrator._pending_evaluate is not None  # orig_prepare 已构造
        orchestrator._pending_evaluate.detect_fn = _retry_3d_as_2d

    orchestrator._prepare_evaluate = _prepare_3d  # type: ignore[method-assign]


def run_3d_agent(
    frame: KittiFrame,
    instruction: str,
    llm_client: Any = None,
    det_model_name: str | None = None,
    seg_model_name: str | None = None,
    out_dir: str | Path | None = None,
    max_iterations: int = 3,
) -> tuple[list[Box3D], AgentState]:
    """3D Agent Loop：复用 2D orchestrator 循环，返回 (Box3D 列表, state)。"""
    registry = build_3d_registry(frame, det_model_name, seg_model_name, out_dir)
    orchestrator = AgentOrchestrator(
        llm_client=llm_client,
        tool_registry=registry,
        max_iterations=max_iterations,
        detection_model=det_model_name,
    )
    # 质量评估挂点（2D 测试同款）：_ensure_tools_registered 仅在 registry 空时设置
    # _detect_tool，3D 预置非空 registry 需显式回填，否则 evaluate 的 retry 处置
    # detect_fn 为 None（"retry_lower_threshold 需要 detect_fn"）。
    orchestrator._detect_tool = registry.get("detect_objects")
    _wrap_evaluate_retry(orchestrator, registry)
    state = orchestrator.run(
        image_path=str(frame.image_path),
        instruction=instruction,
        confidence_threshold=0.3,
    )
    boxes = extract_boxes3d(state)
    # fit_points 判据写 metadata（3D 质量补充，HITL 兜底依据）
    state.metadata["boxes3d_count"] = len(boxes)
    state.metadata["low_fit_points"] = sum(1 for b in boxes if b.fit_points < MIN_FIT_POINTS)
    state.metadata["review_flags"] = sum(1 for b in boxes if b.review_flag)
    return boxes, state
