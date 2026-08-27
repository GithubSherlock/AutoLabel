"""test_agent3d：3D Agent 闭环（FakeLLM + Fake 工具，零真实权重）。

覆盖：planner3d 三级解析、build_3d_registry、extract_boxes3d（yaw 往返）、
_box3d_to_bbox2d 视图、_wrap_evaluate_retry、attach_coco_names（Pedestrian/person
误报缺类回归）、run_3d_agent 全链 + metadata 判据、cli 纯函数。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from auto2dlabel.agent.evaluate import QualityReport
from auto2dlabel.agent.llm import LLMClient
from auto2dlabel.agent.orchestrator import AgentOrchestrator
from auto2dlabel.agent.state import AgentState
from auto2dlabel.tools.base import Tool
from auto2dlabel.tools.registry import ToolRegistry
from auto3dlabel.agent.orchestrator3d import (
    _box3d_to_bbox2d,
    _wrap_evaluate_retry,
    extract_boxes3d,
    run_3d_agent,
)
from auto3dlabel.agent.planner3d import Plan3D, TaskPlanner3D, parse_plan3d_json
from auto3dlabel.agent.tools3d import (
    Detect3DTool,
    Visualize3DTool,
    attach_coco_names,
    build_3d_registry,
)
from auto3dlabel.schema.box3d import Box3D, KittiFrame
from auto3dlabel.tests.helpers.fakes import FakeLLM, tool_call
from auto3dlabel.tests.helpers.synth import write_frame

# 两个 3D 框（一个低 fit_points 强制 review）
_BOXES = [
    Box3D(
        label="Car", confidence=0.85, cx=10.0, cy=1.5, cz=20.0,
        h=1.5, w=1.6, l=3.9, yaw_bev=0.15, fit_points=42,
        x1=100.0, y1=120.0, x2=300.0, y2=260.0,
    ),
    Box3D(
        label="Pedestrian", confidence=0.5, cx=-4.0, cy=1.6, cz=8.0,
        h=1.7, w=0.6, l=0.8, yaw_bev=-0.8, fit_points=8,  # < MIN_FIT_POINTS
        x1=400.0, y1=150.0, x2=500.0, y2=320.0,
    ),
]


class _FakeDetect3DTool(Tool):
    """detect_objects 假工具：返回预设 Box3D dict（模拟真实 Detect3DTool.forward）。"""

    name = "detect_objects"
    description = "fake 3D detection"

    def __init__(self, boxes: list[Box3D] | None = None) -> None:
        self.boxes = boxes if boxes is not None else _BOXES
        self.calls: list[tuple[list[str], float]] = []

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompts": {"type": "array", "items": {"type": "string"}},
                "confidence_threshold": {"type": "number"},
            },
            "required": ["prompts"],
        }

    def forward(self, **kwargs: Any) -> list[dict]:
        prompts = kwargs.get("prompts", [])
        conf = float(kwargs.get("confidence_threshold", 0.3))
        self.calls.append((list(prompts), conf))
        return [b.to_dict() for b in self.boxes]


def _detect_call_result() -> dict[str, Any]:
    return {
        "success": True,
        "count": len(_BOXES),
        "objects": [b.to_dict() for b in _BOXES],
    }


# ── planner3d 三级解析 ─────────────────────────────────────────

def test_parse_plan_direct_json() -> None:
    plan = parse_plan3d_json(json.dumps({
        "frame_id": "000123", "prompts": ["car", "person"],
        "confidence_threshold": 0.25, "det_model": "yolo11s.pt", "seg_model": "sam2_l.pt",
    }))
    assert plan.frame_id == "000123"
    assert plan.prompts == ["car", "person"]
    assert abs(plan.confidence_threshold - 0.25) < 1e-9
    assert plan.det_model == "yolo11s.pt" and plan.seg_model == "sam2_l.pt"


def test_parse_plan_fenced_json() -> None:
    plan = parse_plan3d_json(
        '好的，计划如下：\n```json\n{"frame_id": "123", "prompts": ["car"]}\n```\n完毕'
    )
    assert plan.frame_id == "123" and plan.prompts == ["car"]


def test_parse_plan_regex_fallback() -> None:
    plan = parse_plan3d_json('结果是 {"frame_id": "000456", "prompts": ["bicycle"]}，请执行')
    assert plan.frame_id == "000456" and plan.prompts == ["bicycle"]


def test_parse_plan_defaults() -> None:
    plan = parse_plan3d_json('{"frame_id": "000007"}')
    assert plan.prompts == []  # 缺省 → 空列表（planner 契约）
    assert abs(plan.confidence_threshold - 0.3) < 1e-9  # DEFAULT_CONF
    assert "grounding-dino" in plan.det_model and plan.seg_model == "sam2_l.pt"


def test_parse_plan_garbage_raises() -> None:
    with pytest.raises(ValueError):
        parse_plan3d_json("完全不是 JSON 的响应")


def test_task_planner_parse_via_fake_llm() -> None:
    llm = FakeLLM([{"content": json.dumps({
        "frame_id": "000123", "prompts": ["car", "person"], "confidence_threshold": 0.4,
    })}])
    plan = TaskPlanner3D(cast(LLMClient, llm)).parse("标注 000123 中的汽车和行人")
    assert isinstance(plan, Plan3D)
    assert plan.frame_id == "000123" and plan.prompts == ["car", "person"]
    assert abs(plan.confidence_threshold - 0.4) < 1e-9
    assert "frame=000123" in plan.summary


# ── registry / tools3d ─────────────────────────────────────────

def test_build_3d_registry_names(tmp_path: Any) -> None:
    frame = write_frame(tmp_path)
    reg = build_3d_registry(frame, out_dir=tmp_path / "out")
    assert isinstance(reg.get("detect_objects"), Detect3DTool)
    assert isinstance(reg.get("visualize_bev"), Visualize3DTool)
    # 独立实例：第二个 registry 的 detect 工具绑定另一帧
    frame2 = write_frame(tmp_path, frame_id="000001")
    reg2 = build_3d_registry(frame2)
    assert reg2.get("detect_objects") is not reg.get("detect_objects")
    assert reg2.get("detect_objects").frame.frame_id == "000001"  # type: ignore[attr-defined]


def test_visualize_tool_forward(tmp_path: Any) -> None:
    frame = write_frame(tmp_path)
    tool = Visualize3DTool(frame, tmp_path / "out")
    out = tool.forward(boxes=[_BOXES[0].to_dict()])
    assert out["success"] is True
    assert (tmp_path / "out" / "000000_bev.png").is_file()


def test_attach_coco_names_regression() -> None:
    """Pedestrian → person 回归：KITTI 名不含 COCO 子串会误报缺类。"""
    out = attach_coco_names([
        {"label": "Pedestrian", "confidence": 0.5},
        {"label": "Car", "confidence": 0.9},
        {"label": "Mystery", "confidence": 0.1},
    ])
    assert [d["name"] for d in out] == ["person", "car", "mystery"]


# ── extract_boxes3d / 视图转换 ─────────────────────────────────

def _state_with(tool_calls: list[dict[str, Any]]) -> AgentState:
    state = AgentState()
    state.tool_calls = tool_calls
    return state


def test_extract_boxes3d_yaw_roundtrip() -> None:
    state = _state_with([{
        "tool_name": "detect_objects",
        "arguments": {"prompts": ["car"]},
        "result": _detect_call_result(),
    }])
    boxes = extract_boxes3d(state)
    assert len(boxes) == 2
    for orig, rebuilt in zip(_BOXES, boxes):
        assert rebuilt.label == orig.label
        assert abs(rebuilt.cx - orig.cx) < 1e-6
        assert abs(rebuilt.yaw_bev - orig.yaw_bev) < 1e-6  # rotation_y 往返
        assert rebuilt.fit_points == orig.fit_points


def test_extract_boxes3d_skips_other_tools_and_corrupt() -> None:
    state = _state_with([
        {"tool_name": "visualize_bev", "result": {"path": "/tmp/x.png"}},
        {"tool_name": "detect_objects", "result": {"objects": [_BOXES[0].to_dict(), {"cx": 1.0}]}},
        {"tool_name": "detect_objects", "result": None},
        {"tool_name": "detect_objects", "result": {"objects": [_BOXES[1]]}},  # Box3D 直通
    ])
    boxes = extract_boxes3d(state)
    assert len(boxes) == 2  # visualize 跳过、无 label dict 跳过、None 跳过
    assert boxes[0].label == "Car" and boxes[1].label == "Pedestrian"


def test_extract_boxes3d_empty() -> None:
    assert extract_boxes3d(_state_with([])) == []
    assert extract_boxes3d(_state_with([{"tool_name": "detect_objects", "result": {}}])) == []


def test_box3d_to_bbox2d_view() -> None:
    bbox = _box3d_to_bbox2d(_BOXES[0].to_dict())
    assert bbox.x == 100.0 and bbox.y == 120.0
    assert bbox.width == 200.0 and bbox.height == 140.0
    assert bbox.label == "Car" and bbox.confidence == 0.85


# ── _wrap_evaluate_retry ───────────────────────────────────────

def test_wrap_evaluate_retry_detect_fn_2d_view() -> None:
    """evaluate 重试的 detect_fn 返回 2D Bbox 视图（_sync_annotations 只吃 Bbox）。"""
    fake = _FakeDetect3DTool()
    reg = ToolRegistry()
    reg.register(fake)
    orch = AgentOrchestrator(llm_client=cast(LLMClient, FakeLLM([{}])), tool_registry=reg)
    orch._detect_tool = fake
    _wrap_evaluate_retry(orch, reg)

    quality = QualityReport(
        total_boxes=2, prompts=["car", "person"], covered_prompts=["car", "person"],
    )
    orch._prepare_evaluate("/tmp/fake.png", {"prompts": ["car", "person"]}, quality, 0.3)
    assert orch._pending_evaluate is not None
    views = orch._pending_evaluate.detect_fn(0.2)
    assert len(views) == 2
    assert all(hasattr(v, "id") for v in views)  # Bbox 语义（非 dict）
    assert views[0].label == "Car"
    assert fake.calls == [(["car", "person"], 0.2)]  # 重试按新阈值跑 3D 管线


# ── orchestrator 全循环 + run_3d_agent ─────────────────────────

def _two_round_llm(prompts: list[str]) -> FakeLLM:
    return FakeLLM([
        {"tool_calls": [tool_call("detect_objects", {"prompts": prompts}, "c1")]},
        {"content": "完成。"},  # 第二轮无 tool_call → 硬性收尾
    ])


def test_orchestrator_loop_dict_passthrough() -> None:
    """orchestrator 循环 → _summarize dict 透传 → tool_calls 记录完整 3D dict。"""
    fake = _FakeDetect3DTool()
    reg = ToolRegistry()
    reg.register(fake)
    orch = AgentOrchestrator(
        llm_client=cast(LLMClient, _two_round_llm(["car", "person"])),
        tool_registry=reg, max_iterations=3,
    )
    orch._detect_tool = fake
    _wrap_evaluate_retry(orch, reg)
    state = orch.run(
        image_path="/tmp/fake.png", instruction="检测汽车和行人", confidence_threshold=0.3,
    )

    assert fake.calls == [(["car", "person"], 0.3)], fake.calls
    assert state.iteration == 2
    assert state.metadata["quality_report"]["total_boxes"] == 2
    boxes = extract_boxes3d(state)
    assert len(boxes) == 2


def test_run_3d_agent_full(monkeypatch: Any) -> None:
    """run_3d_agent 薄壳：build_3d_registry mock 后走同一循环 + metadata 判据。"""
    import auto3dlabel.agent.orchestrator3d as orch3d

    fake = _FakeDetect3DTool()
    reg = ToolRegistry()
    reg.register(fake)
    monkeypatch.setattr(orch3d, "build_3d_registry", lambda *a, **k: reg)

    # KittiFrame 仅被 build_3d_registry 消费（已 mock），占位即可
    frame = KittiFrame(frame_id="000000", root=Path("/tmp"))
    boxes, state = run_3d_agent(
        frame, "检测汽车和行人",
        llm_client=cast(LLMClient, _two_round_llm(["car"])),
        out_dir="/tmp/out3d",
    )
    assert len(boxes) == 2
    assert state.metadata["boxes3d_count"] == 2
    assert state.metadata["low_fit_points"] == 1  # Pedestrian 8 点
    assert state.metadata["review_flags"] == 0


# ── cli 纯函数 ─────────────────────────────────────────────────

def test_cli_parse_frame_ids(tmp_path: Any) -> None:
    from auto3dlabel import cli

    assert cli._parse_frame_ids("000123") == ["000123"]
    assert cli._parse_frame_ids("123") == ["000123"]
    assert cli._parse_frame_ids("003712-003713") == ["003712", "003713"]  # 含两端
    d = tmp_path / "image_2"
    d.mkdir()
    (d / "000005.png").write_bytes(b"")
    assert cli._parse_frame_ids(str(d)) == ["000005"]
    with pytest.raises(Exception):  # typer.BadParameter
        cli._parse_frame_ids("not-a-frame")


def test_cli_resolve_det_model() -> None:
    from auto3dlabel import cli

    assert cli._resolve_det_model(None) == "IDEA-Research/grounding-dino-tiny"
    assert cli._resolve_det_model("yolo11s.pt") == "yolo11s.pt"  # 直通
    # kitti_finetune → 微调权重路径（存在时）；否则回退默认
    out = cli._resolve_det_model("kitti_finetune")
    assert out.endswith("best.pt") or out == "IDEA-Research/grounding-dino-tiny"
