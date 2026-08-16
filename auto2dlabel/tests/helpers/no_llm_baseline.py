"""--no-llm baseline 测试工具。

绕过 LLM Agent Loop，直接用「中→英关键词映射 + 检测模型」生成标注结果，
输出格式与 LLM 路径完全一致（JSON + PNG + LOG），可直接 diff。

原实现为 auto2dlabel/cli.py 的 `_run_no_llm()`，迁入此处供 pytest 使用。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from rich.console import Console

from auto2dlabel.agent.evaluate import QualityReport, evaluate_detections
from auto2dlabel.models.detection import (
    DetectionResult,
    create_detection_model,
    detect_with_retry,
)
from auto2dlabel.schema.annotation import Annotation, Bbox
from auto2dlabel.schema.task_plan import DEFAULT_MODEL
from auto2dlabel.tools.export import ExportTool
from auto2dlabel.tools.log import log_python_api_call
from auto2dlabel.tools.visualize import visualize_annotation

console = Console()

# 与 AgentOrchestrator 共用同一份中→英关键词映射表
CN_EN_MAP = {
    "行人": "person", "人": "person", "汽车": "car", "车": "car",
    "自行车": "bicycle", "单车": "bicycle", "摩托车": "motorcycle",
    "公交车": "bus", "卡车": "truck", "狗": "dog", "猫": "cat",
    "红绿灯": "traffic light", "交通灯": "traffic light",
}


@dataclass
class NoLlmResult:
    """单张图的 no-LLM baseline 结果。"""

    image_path: str
    prompts: list[str]
    detections: list[DetectionResult] = field(default_factory=list)
    annotation: Annotation | None = None
    json_path: Path | None = None
    vis_path: Path | None = None
    elapsed: float = 0.0
    quality: QualityReport | None = None


def extract_prompts(instruction: str) -> list[str]:
    """从用户指令中提取英文 prompts（与 orchestrator 的 hint 映射一致）。

    Raises:
        ValueError: 指令中不含任何映射关键词。
    """
    prompts: list[str] = []
    for cn, en in CN_EN_MAP.items():
        if cn in instruction and en not in prompts:
            prompts.append(en)
    if not prompts:
        raise ValueError("无法从指令中提取关键词。请直接使用英文 prompt。")
    return prompts


def run_no_llm_baseline(
    image_files: list[Path],
    instruction: str,
    threshold: float = 0.3,
    iou: float = 0.5,
    export_fmt: str = "coco",
    output: str = "outputs",
    det_model: str | None = None,
    sahi: bool = False,
    vis_dir: str = "vis_outputs",
) -> list[NoLlmResult]:
    """no-LLM baseline：绕过 Agent Loop，直接调检测模型。

    输出格式与 LLM 路径完全一致（JSON + PNG + LOG + HITL 分流），可直接 diff。
    返回每张图的结构化结果，供测试断言。

    Raises:
        ValueError: 无法从指令提取关键词。
    """
    from auto2dlabel.tools.hitl import export_triage, triage_annotations

    prompts = extract_prompts(instruction)
    det_name = det_model or DEFAULT_MODEL

    console.print(f"[dim]检测模型: {det_name}  置信度: {threshold}  IoU: {iou}[/dim]")
    console.print(f"[dim]提取的关键词: {', '.join(prompts)}[/dim]")

    model = create_detection_model(det_name, iou_threshold=iou)
    results_out: list[NoLlmResult] = []

    for img_path in image_files:
        console.print(f"[dim]━━━ 标注: {img_path.name} ━━━[/dim]")

        _t0 = time.time()
        _ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        # 直接检测，不经过 Agent（与 LLM 路径一致：0 框降阈值重试一次）
        if sahi:
            from auto2dlabel.benchmarks.common import detect_image_sahi

            console.print("[dim]SAHI 切片推理...[/dim]")
            dets, retried, retry_threshold = detect_with_retry(
                lambda conf: detect_image_sahi(
                    model, str(img_path), prompts, confidence_threshold=conf,
                ),
                threshold,
            )
            detections = [
                DetectionResult(
                    x=d["bbox"][0], y=d["bbox"][1],
                    width=d["bbox"][2] - d["bbox"][0],
                    height=d["bbox"][3] - d["bbox"][1],
                    label=d["name"], confidence=d["conf"],
                )
                for d in dets
            ]
        else:
            detections, retried, retry_threshold = detect_with_retry(
                lambda conf: model.detect(str(img_path), prompts, conf),
                threshold,
            )

        # 代码级质量评估（与 Agent Loop / chat 路径共用判据）
        quality = evaluate_detections(
            detections, prompts, threshold, str(img_path),
            retried=retried, retry_threshold=retry_threshold,
        )
        if quality.warnings:
            console.print(f"[yellow]质量警告: {'; '.join(quality.warnings)}[/yellow]")

        # 构建 Annotation（与 LLM 路径格式一致）
        ann = Annotation(image_path=str(img_path))
        try:
            from PIL import Image

            im = Image.open(img_path)
            ann.image_size = (im.width, im.height)
        except Exception:
            pass
        for r in detections:
            ann.add_bbox(Bbox(x=r.x, y=r.y, width=r.width, height=r.height,
                              label=r.label, confidence=r.confidence))
            if r.confidence < threshold:
                ann.flag_for_review(len(ann.bboxes) - 1)

        elapsed = round(time.time() - _t0, 3)

        # 导出 JSON
        out_json = Path(ExportTool().forward(
            annotations=[ann.to_dict()],
            output_path=f"{output}/{img_path.stem}_{_ts}.json",
            format=export_fmt,
        ))

        # 可视化
        vis_path = Path(vis_dir) / f"vis_{img_path.stem}_{_ts}.png"
        vis_path.parent.mkdir(parents=True, exist_ok=True)
        visualize_annotation(img_path, ann, vis_path, draw_mask=bool(ann.masks))

        # HITL 置信度分流（与 LLM 路径一致）
        tau_low = max(0.1, threshold * 0.5)
        triage = triage_annotations([ann], tau_high=0.7, tau_low=tau_low)
        export_triage(triage, output_dir=output, image_stem=img_path.stem, timestamp=_ts)

        # 日志（method=PythonAPI 而非 LLM）
        log_python_api_call(
            image_path=str(img_path),
            prompts=prompts,
            results=detections,
            elapsed=elapsed,
            model_name=det_name,
            confidence_threshold=threshold,
            iou_threshold=iou,
            annotation_type="object_detection",
            timestamp=_ts,
            quality=quality.to_dict(),
        )

        results_out.append(NoLlmResult(
            image_path=str(img_path),
            prompts=prompts,
            detections=detections,
            annotation=ann,
            json_path=out_json,
            vis_path=vis_path,
            elapsed=elapsed,
            quality=quality,
        ))

    return results_out
