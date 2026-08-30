"""批次级策略（3b）—— 大批量检测前抽样统计 + LLM 一次性调参（每批 1 次调用）。

与序列级 LLM 规划（cli_track._llm_plan_once）同构：LLM 每批只调 1 次，
逐图执行仍走纯代码级管线（零 LLM）。任何一步失败调用方降级代码级默认参数，
无 API key 时零影响。

数据流（execute_plan --batch-strategy）：
sample_stats（确定性抽样 ≤8 张代码级检测）→ llm_tune_strategy（单轮 JSON 调参
建议）→ apply_strategy（conf 立即覆写 + 模型建议进 model_hint）。抽样与
aggregation 全为纯函数/注入模型，单测零真实权重。
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BatchStats:
    """抽样检测的聚合统计（喂给 LLM 调参）。"""

    images_sampled: int = 0
    prompts: list[str] = field(default_factory=list)
    total_boxes: int = 0
    zero_box_images: int = 0
    missing_prompts: list[str] = field(default_factory=list)  # 任一抽样图缺失的类别
    threshold: float = 0.3

    @property
    def avg_boxes(self) -> float:
        """平均每图框数（无抽样时为 0）。"""
        return self.total_boxes / self.images_sampled if self.images_sampled else 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON 序列化（LLM 调参入参）。"""
        return {
            "images_sampled": self.images_sampled,
            "prompts": self.prompts,
            "total_boxes": self.total_boxes,
            "avg_boxes": round(self.avg_boxes, 2),
            "zero_box_images": self.zero_box_images,
            "missing_prompts": self.missing_prompts,
            "threshold": self.threshold,
        }


def sample_stats(
    image_paths: Sequence[str | Path],
    prompts: list[str],
    confidence_threshold: float = 0.3,
    *,
    model: Any | None = None,
    det_model_name: str | None = None,
    max_samples: int = 8,
) -> BatchStats:
    """确定性抽样（步长采样，≤max_samples 张）代码级检测 → 聚合统计。

    model 复用调用方已加载的检测模型（None 时按 det_model_name 创建）。
    单张抽样异常按 0 框计入（宁多勿漏——让 LLM 看到问题信号）。
    """
    from auto2dlabel.agent.evaluate import evaluate_detections
    from auto2dlabel.models.detection import create_detection_model

    if model is None:
        model = create_detection_model(det_model_name)
    paths = [str(p) for p in image_paths]
    if len(paths) <= max_samples:
        sampled = paths
    else:
        # 步长采样：ceil 保证抽样数 ≤ max_samples，跨批次确定性
        stride = math.ceil(len(paths) / max_samples)
        sampled = paths[::stride]

    stats = BatchStats(
        images_sampled=len(sampled),
        prompts=list(prompts),
        threshold=confidence_threshold,
    )
    missing: set[str] = set()
    for path in sampled:
        try:
            dets = model.detect(path, prompts, confidence_threshold)
        except Exception:
            dets = []  # 单张异常按 0 框计入
        stats.total_boxes += len(dets)
        if not dets:
            stats.zero_box_images += 1
        report = evaluate_detections(dets, prompts, confidence_threshold, path)
        missing.update(report.missing_prompts)
    stats.missing_prompts = sorted(missing)
    return stats


def llm_tune_strategy(
    llm: Any,
    instruction: str,
    stats: BatchStats,
) -> dict[str, Any]:
    """单轮 LLM 调参建议（每批 1 次调用，逐图仍零 LLM）。

    输出 JSON {confidence_threshold, suggest_model, note}；字段非法抛 ValueError
    （调用方降级代码级默认参数）。
    """
    response = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "你是自动标注系统的批量参数调优器。基于抽样统计输出建议 JSON：\n"
                    '{"confidence_threshold": 0到1浮点数, '
                    '"suggest_model": "建议替换的检测模型名或 null", '
                    '"note": "一句中文理由"}\n'
                    "confidence_threshold 将覆盖批量执行的检测阈值；"
                    "suggest_model 可填 yolo26x.pt / fasterrcnn_resnet50_fpn_v2 等"
                    "（无建议则 null）。只输出 JSON，不要其他文字。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"指令: {instruction}\n"
                    f"抽样统计: {json.dumps(stats.to_dict(), ensure_ascii=False)}"
                ),
            },
        ],
        temperature=0.0,
        max_tokens=512,
        json_mode=True,
        call_site="batch.strategy",
    )
    content = (response.content or "").strip()
    # 剥离可能的 markdown 代码块（仿 cli_track._llm_plan_once）
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:]
    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError(f"LLM 调参输出非对象: {data}")

    raw_conf = data.get("confidence_threshold")
    if (
        not isinstance(raw_conf, (int, float))
        or isinstance(raw_conf, bool)
        or not 0 < raw_conf <= 1
    ):
        raise ValueError(f"LLM 调参输出无效 confidence_threshold: {data}")
    suggest = data.get("suggest_model")
    if suggest is not None and (not isinstance(suggest, str) or not suggest.strip()):
        raise ValueError(f"LLM 调参输出无效 suggest_model: {data}")
    note = data.get("note")
    if not isinstance(note, str):
        note = ""

    return {
        "confidence_threshold": float(raw_conf),
        "suggest_model": suggest,
        "note": note,
    }


def apply_strategy(step: Any, strategy: dict[str, Any]) -> dict[str, Any]:
    """把调参建议落到步骤：conf 立即覆写（钳位 [0.05, 0.95]）；模型建议只进
    model_hint（权重已加载，换模型留作提示，不中途重建）。返回生效值。"""
    conf = float(strategy.get("confidence_threshold", step.confidence_threshold))
    conf = min(0.95, max(0.05, conf))
    old = step.confidence_threshold
    step.confidence_threshold = conf
    suggest = strategy.get("suggest_model")
    if isinstance(suggest, str) and suggest.strip():
        current = f"（当前 {step.model_name}）" if step.model_name else ""
        step.model_hint = f"批次策略建议: {suggest}{current}"
    return {
        "confidence_threshold": conf,
        "old_confidence_threshold": old,
        "suggest_model": suggest,
        "note": strategy.get("note", ""),
    }
