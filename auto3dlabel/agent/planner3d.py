"""3D 任务规划器：NL → Plan3D（照 auto2dlabel planner 模式：JSON 三级解析兜底）。

三级解析：直接 JSON → ```json 代码块 → 正则找 { } 块（同 planner.py _parse_json_response）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from auto2dlabel.agent import json
from auto2dlabel.agent.llm import LLMClient

from auto3dlabel.configs.kitti import DEFAULT_CONF, DEFAULT_DET_MODEL, DEFAULT_SEG_MODEL

_PLANNER3D_SYSTEM_PROMPT = """You are a task planner for a KITTI 3D object annotation tool. \
Parse user's NL into JSON.

Output ONLY valid JSON:
{
  "frame_id": "000123",
  "prompts": ["car", "person"],
  "confidence_threshold": 0.3,
  "det_model": "IDEA-Research/grounding-dino-tiny",
  "seg_model": "sam2_l.pt"
}

Rules:
- frame_id: KITTI frame number, 6-digit zero-padded (e.g. 123 → "000123"). "" if not specified.
- prompts: English COCO names only. Map: 汽车/车辆→car, 行人/人→person, \
自行车/单车/骑行者→bicycle, 摩托车→motorcycle, 卡车→truck, 公交车/公共汽车→bus, 火车→train
- confidence_threshold: default 0.3; extract if user says 置信度/conf/阈值X
- det_model: 2D detector for the pipeline. Default "IDEA-Research/grounding-dino-tiny".
  Map: yolo→yolo11s.pt, yolo26→yolo26x.pt, grounding dino/gdino→IDEA-Research/grounding-dino-tiny,
  kitti微调/kitti权重/kitti_yolo→kitti_finetune (a KITTI-finetuned YOLO).
  If user names a model ending with .pt or containing /, use it directly.
- seg_model: SAM mask model. Default "sam2_l.pt". Map: sam/sam2→sam2_l.pt, \
fastsam→FastSAM-s.pt, sam3→sam3.pt
- Only JSON. No other text."""


@dataclass
class Plan3D:
    """3D 标注任务参数（planner 输出 → CLI/agent 消费）。"""

    frame_id: str = ""
    prompts: list[str] | None = None
    confidence_threshold: float = DEFAULT_CONF
    det_model: str = DEFAULT_DET_MODEL
    seg_model: str = DEFAULT_SEG_MODEL

    @property
    def summary(self) -> str:
        return (
            f"frame={self.frame_id or '?'} prompts={self.prompts or []} "
            f"conf={self.confidence_threshold} det={self.det_model} seg={self.seg_model}"
        )


def _dict_to_plan3d(data: dict) -> Plan3D:
    prompts = data.get("prompts", [])
    return Plan3D(
        frame_id=str(data.get("frame_id", "")),
        prompts=[str(p) for p in prompts] if isinstance(prompts, list) else None,
        confidence_threshold=float(data.get("confidence_threshold", DEFAULT_CONF)),
        det_model=str(data.get("det_model", DEFAULT_DET_MODEL)),
        seg_model=str(data.get("seg_model", DEFAULT_SEG_MODEL)),
    )


def parse_plan3d_json(content: str) -> Plan3D:
    """LLM 响应 → Plan3D（三级解析；全失败抛 ValueError）。"""
    try:
        return _dict_to_plan3d(json.loads(content))
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
    if m:
        try:
            return _dict_to_plan3d(json.loads(m.group(1)))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{[\s\S]*\}", content)
    if m:
        try:
            return _dict_to_plan3d(json.loads(m.group(0)))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"无法从 LLM 响应解析 Plan3D JSON: {content[:200]}...")


class TaskPlanner3D:
    """3D 任务规划器。"""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def parse(self, instruction: str) -> Plan3D:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _PLANNER3D_SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]
        response = self.llm.chat(messages, tools=None, temperature=0.0)
        if not response.content:
            raise ValueError("LLM 返回空响应，无法解析 3D 任务")
        return parse_plan3d_json(response.content)
