"""3D 任务规划器：NL → Plan3D（照 auto2dlabel planner 模式：JSON 三级解析兜底）。

三级解析：直接 JSON → ```json 代码块 → 正则找 { } 块（同 planner.py _parse_json_response）。
v0.3 P1：对话式规划（parse_dialog）复用 auto2dlabel/agent/dialog.py 通用骨架。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from auto2dlabel.agent import json, logging
from auto2dlabel.agent.dialog import parse_with_dialog
from auto2dlabel.agent.llm import LLMClient
from auto2dlabel.configs.task_params import coerce_float, coerce_str
from auto2dlabel.schema.task_plan import PlanQuestion
from auto3dlabel.configs.datasets import format_datasets_summary
from auto3dlabel.configs.kitti import DEFAULT_CONF, DEFAULT_DET_MODEL, DEFAULT_SEG_MODEL
from auto3dlabel.configs.task_params import PARAM_SPECS_3D

logger = logging.getLogger(__name__)

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
- confidence_threshold: default 0.3; extract ONLY from explicit 置信度/conf/阈值X
  phrasing. Digits inside the frame number/paths are NEVER hyperparameters —
  do NOT extract them as confidence_threshold.
- det_model: 2D detector or LiDAR 3D detector for the pipeline.
  Default "IDEA-Research/grounding-dino-tiny".
  Map: yolo→yolo11s.pt, yolo26→yolo26x.pt, grounding dino/gdino→
  IDEA-Research/grounding-dino-tiny, kitti微调/kitti权重/kitti_yolo→kitti_finetune
  (a KITTI-finetuned YOLO).
  LiDAR 3D engines: pointpillars/点柱→pointpillars_kitti (KITTI),
  pvrcnn/pv-rcnn/PV-RCNN→pvrcnn_kitti (KITTI),
  centerpoint→centerpoint_nus (nuScenes)
  (LiDAR-only detectors, no SAM needed).
  If user names a model ending with .pt or containing /, use it directly.
- seg_model: SAM mask model. Default "sam2_l.pt". Map: sam/sam2→sam2_l.pt, \
fastsam→FastSAM-s.pt, sam3→sam3.pt
- questions: ONLY when key fields are missing (frame_id / prompts), e.g.
  "questions": [{"id": "frame_id", "question": "要标注哪个 KITTI 帧？(如 000123)"}].
  id = bare field name. questions MUST accompany the full JSON (missing fields left
  empty/default). No questions when nothing is missing.
- Dialog: next round user's answers are fed back as
  "补充信息（用户在以下问题的回答，请据此更新 JSON）：" + "- [frame_id] <question>"
  + "用户回答：<answers>". Then fill the previously-missing fields from the
  answers and output empty/omit questions.
- Only JSON. No other text."""

# 追加数据集摘要（字符串拼接：原文含 JSON 花括号，不可改 f-string；与 2D planner 同模式）
_PLANNER3D_SYSTEM_PROMPT = _PLANNER3D_SYSTEM_PROMPT + f"""

{format_datasets_summary()}
Dataset rules:
- 3D 帧根目录由 CLI 决定，LLM 不解析路径——「KITTI 数据集/对 KITTI」→ 照常输出 6 位 frame_id
- 指令含自建数据集名 → 该数据集可用，帧标注方式同 KITTI（frame_id 照填）"""


@dataclass
class Plan3D:
    """3D 标注任务参数（planner 输出 → CLI/agent 消费）。"""

    frame_id: str = ""
    prompts: list[str] | None = None
    confidence_threshold: float = DEFAULT_CONF
    det_model: str = DEFAULT_DET_MODEL
    seg_model: str = DEFAULT_SEG_MODEL
    questions: list[PlanQuestion] = field(default_factory=list)  # v0.3 P1 对话问题

    @property
    def summary(self) -> str:
        return (
            f"frame={self.frame_id or '?'} prompts={self.prompts or []} "
            f"conf={self.confidence_threshold} det={self.det_model} seg={self.seg_model}"
        )

    @property
    def missing_params(self) -> list[str]:
        """缺失的必填参数（规格表驱动；frame_id "" / prompts None → 缺失）。

        v0.3.1：原 cli 只查 frame_id——prompts 缺失会漏到下游；改为规格表
        全量检查（label 直接进缺参清单文案）。
        """
        return [
            spec.label
            for spec in PARAM_SPECS_3D.values()
            if spec.required and not getattr(self, spec.key)
        ]


def _dict_to_plan3d(data: dict) -> Plan3D:
    prompts = data.get("prompts", [])
    return Plan3D(
        frame_id=str(data.get("frame_id", "")),
        prompts=[str(p) for p in prompts] if isinstance(prompts, list) else None,
        # coerce_float：垃圾值/值域外回默认（帧号数字被误提为 conf 的守卫；
        # 防 "conf": "abc" 直接 float() 抛异常中断解析）
        confidence_threshold=coerce_float(
            data.get("confidence_threshold", DEFAULT_CONF), DEFAULT_CONF, lo=0.0, hi=1.0
        ),
        det_model=coerce_str(data.get("det_model"), DEFAULT_DET_MODEL) or DEFAULT_DET_MODEL,
        seg_model=coerce_str(data.get("seg_model"), DEFAULT_SEG_MODEL) or DEFAULT_SEG_MODEL,
        # v0.3 P1 对话问题（致命点：LLM JSON → plan 走本函数）
        questions=PlanQuestion.from_list(data.get("questions")),
    )


def sanitize_plan3d(plan: Plan3D) -> list[str]:
    """LLM 输出守卫（同 2D 漏洞类，2026-08-29）：返回修正记录供日志。

    - frame_id 须纯数字（null → "None"、"abc" 等垃圾 → "" 缺参走追问/
      BadParameter，不再让 resolve_frame 裸抛 ValueError traceback）
    - prompts 过滤空串；过滤后为空 → None（缺参语义）
    - det/seg 模型名去空白（空串在 _dict_to_plan3d 已回默认，此处兜底）
    """
    fixes: list[str] = []

    def _fix(key: str, new: object) -> None:
        old = getattr(plan, key)
        if old != new:
            fixes.append(f"{key}: {old!r} -> {new!r}")
            setattr(plan, key, new)

    fid = (plan.frame_id or "").strip()
    _fix("frame_id", fid if fid.isdigit() else "")
    prompts = plan.prompts
    if prompts is not None:
        # 只过滤空串项；空结果保持 []（零行为漂移：None/[] 均按缺参处理）
        _fix("prompts", [p for p in (s.strip() for s in prompts) if p])
    _fix(
        "confidence_threshold",
        coerce_float(plan.confidence_threshold, DEFAULT_CONF, lo=0.0, hi=1.0),
    )
    _fix("det_model", coerce_str(plan.det_model, DEFAULT_DET_MODEL) or DEFAULT_DET_MODEL)
    _fix("seg_model", coerce_str(plan.seg_model, DEFAULT_SEG_MODEL) or DEFAULT_SEG_MODEL)
    return fixes


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
        plan = parse_plan3d_json(response.content)
        for fix in sanitize_plan3d(plan):  # LLM 输出守卫（帧号数字误提等）
            logger.info("3D 参数守卫修正: %s", fix)
        return plan

    def parse_dialog(
        self,
        instruction: str,
        ask_fn: Callable[[list[PlanQuestion]], str],
        max_rounds: int = 3,
    ) -> Plan3D:
        """多轮对话解析（v0.3 P1）：缺参（frame_id/prompts）→ questions → 收集 → 回喂。

        复用 auto2dlabel/agent/dialog.py 通用骨架（2D v0.6 同源，复用不复制）；
        轮次耗尽 / 用户放弃 → 返回缺参 plan（3D cli 保持 BadParameter 兜底）；
        LLM 响应非法 / 空响应 → ValueError 传播（调用方降级链处理）。
        """
        plan = parse_with_dialog(
            llm=self.llm,
            system_prompt=_PLANNER3D_SYSTEM_PROMPT,
            parse_fn=parse_plan3d_json,
            ask_fn=ask_fn,
            instruction=instruction,
            max_rounds=max_rounds,
        )
        for fix in sanitize_plan3d(plan):  # LLM 输出守卫（对话路径同源）
            logger.info("3D 参数守卫修正: %s", fix)
        return plan
