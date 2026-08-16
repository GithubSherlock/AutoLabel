"""任务规划器 —— 将自然语言指令解析为 TaskPlan。

使用 LLM 从用户文本中提取结构化标注参数。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from auto2dlabel.agent.llm import LLMClient
from auto2dlabel.schema.task_plan import (
    BENCHMARK_DATASETS,
    BENCHMARK_DEFAULT_CONF,
    BENCHMARK_DEFAULT_IOU,
    BENCHMARK_DEFAULT_MAX_IMAGES,
    BENCHMARK_DEFAULT_MODEL,
    BENCHMARK_DEFAULT_SEG_MODEL,
    BenchmarkRequest,
    DATASET_CN_MAP,
    DEFAULT_CONFIDENCE,
    DEFAULT_EXPORT,
    DEFAULT_IOU,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    OPTIONAL_PARAMS,
    TaskPlan,
    TaskStep,
    TaskType,
)

logger = logging.getLogger(__name__)

_PLANNER_SYSTEM_PROMPT = """You are a task planner for an image annotation tool. Parse user's NL into JSON.

Output ONLY valid JSON:
{
  "steps": [
    {
      "step_id": 1,
      "task_type": "object_detection | instance_segmentation | classification | obb_detection",
      "source": "/path/to/images",
      "prompts": ["car", "person"],
      "confidence_threshold": 0.1,
      "iou_threshold": 0.3,
      "model_name": "yolo26x.pt",
      "export_format": "coco",
      "sahi": false
    }
  ],
  "confirm_timeout": 30
}

Rules:
- task_type: "classification" if user says 分类/classify/打标签/图片分类 (prompts are candidate labels); "obb_detection" if user says 旋转框/obb/rotate/oriented (prompts are classes); "instance_segmentation" if user says 分割/segmentation/mask; otherwise "object_detection"
- source: data path. "" if not specified.
- prompts: English only. Map: 汽车→car, 行人/人→person, 自行车/单车→bicycle, 摩托车→motorcycle, 公共汽车/公交车→bus, 卡车→truck, 狗→dog, 猫→cat
- confidence_threshold: default 0.1
- iou_threshold: default 0.3
- model_name: map hints to names. Key mappings: faster rcnn→fasterrcnn_resnet50_fpn_v2, yolo→yolo26x.pt, yolov8→yolov8n.pt, rtdetr/rt-detr/detr→rtdetr-l.pt, grounding dino→IDEA-Research/grounding-dino-tiny, clip→openai/clip-vit-base-patch32, siglip→google/siglip-base-patch16-224, obb→yolo11n-obb.pt. sam3/sam/maskrcnn/fastsam are SEGMENTATION models — keep them as model_name (the system auto-routes them). default: yolo26x.pt
- export_format: "cls" if task_type=classification; "dota" if task_type=obb_detection;
  otherwise "coco"
- sahi: true if user mentions SAHI/切片/切块/slicing/sahi/大图. default false.
- Multiple tasks separated by 然后/再/；/; → multiple steps.
- Only JSON. No other text."""


_BENCHMARK_SYSTEM_PROMPT = f"""You are a benchmark planner for an image annotation tool. Parse user's NL into JSON.

Output ONLY valid JSON:
{{
  "dataset": "<dataset_key>",
  "task_type": "detection | segmentation",
  "model": "<model_name>",
  "seg_model": "<seg_model_name>",
  "conf": 0.3,
  "iou": 0.5,
  "max_images": 50,
  "sahi": false
}}

Available datasets (key → description):
- coco: COCO 2017 val detection
- voc2007: Pascal VOC 2007 detection
- kitti: KITTI object detection
- dota: DOTA aerial detection (航拍)
- mot: MOT17+MOT20 pedestrian detection (密集行人)
- coco_seg: COCO 2017 val instance segmentation
- cityscapes: Cityscapes instance segmentation (城市街景)
- nuimages: nuImages instance segmentation
- d2sa: D2SA retail shelf instance segmentation (零售货架)

Rules:
- dataset: Map user's words to dataset keys. 中文映射: COCO/COCO2017→coco, VOC→voc2007, 航拍→dota, MOT/行人→mot, COCO分割→coco_seg, 街景/cityscapes→cityscapes, nuImages→nuimages, D2SA/零售/货架→d2sa
- task_type: "segmentation" if user mentions 分割/segmentation/mask/segment or dataset is coco_seg/cityscapes/nuimages/d2sa; otherwise "detection"
- model: Map hints. 默认{BENCHMARK_DEFAULT_MODEL}. Key mappings: yolo/yolov8→yolov8x.pt, yolo26→yolo26x.pt, yolov8n→yolov8n.pt, rtdetr/rt-detr/detr→rtdetr-l.pt, faster rcnn/frcnn→fasterrcnn_resnet50_fpn_v2, grounding dino→IDEA-Research/grounding-dino-tiny. If user says a specific model name that looks like a real model (ends with .pt or contains /), use it directly.
- seg_model: For segmentation only. 默认{BENCHMARK_DEFAULT_SEG_MODEL}. Map: fastsam→FastSAM-s.pt, fastsam-x→FastSAM-x.pt, sam/sam2/sam_b/sam_l→sam_b.pt, sam3→sam3.pt, maskrcnn→maskrcnn_resnet50_fpn_v2
- conf: Extract from "conf=X" or "阈值X" or "置信度X". 默认{BENCHMARK_DEFAULT_CONF}
- iou: Extract from "iou=X". 默认{BENCHMARK_DEFAULT_IOU}
- max_images: Extract from "N张图" or "N张" or "N images" or "max=N". 默认{BENCHMARK_DEFAULT_MAX_IMAGES}
- sahi: true if user mentions SAHI/切片/切块/slicing/sahi. 默认false
- If user doesn't specify dataset, leave dataset="".
- Only JSON. No other text."""


class TaskPlanner:
    """任务规划器 —— NL → TaskPlan / BenchmarkRequest。"""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def parse(self, instruction: str, confirm_timeout: int = DEFAULT_TIMEOUT) -> TaskPlan:
        """将自然语言指令解析为 TaskPlan。

        Args:
            instruction: 用户自然语言指令。
            confirm_timeout: 确认等待秒数（可通过 --no-wait 设为 0）。

        Returns:
            TaskPlan。
        """
        from auto2dlabel.agent.llm import LLMResponse

        messages: list[dict] = [
            {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]

        response = self.llm.chat(messages, tools=None, temperature=0.0)

        if not response.content:
            raise ValueError("LLM 返回空响应，无法解析任务")

        plan = _parse_json_response(response.content)
        plan.raw_instruction = instruction

        # 应用用户指定的 timeout
        for step in plan.steps:
            if plan.confirm_timeout == DEFAULT_TIMEOUT:
                plan.confirm_timeout = confirm_timeout

        logger.info("Parsed plan: %d steps", len(plan.steps))
        for s in plan.steps:
            logger.info("  %s", s.summary)

        return plan

    def parse_benchmark(self, instruction: str) -> BenchmarkRequest:
        """将自然语言指令解析为 BenchmarkRequest。

        Args:
            instruction: 用户自然语言指令（如 "用 YOLOv8x 跑 COCO 检测，50 张图 conf=0.3"）。

        Returns:
            BenchmarkRequest。
        """
        messages: list[dict] = [
            {"role": "system", "content": _BENCHMARK_SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]

        response = self.llm.chat(messages, tools=None, temperature=0.0)

        if not response.content:
            raise ValueError("LLM 返回空响应，无法解析 benchmark 参数")

        request = _parse_benchmark_json(response.content)
        logger.info("Parsed benchmark: %s", request.summary)
        return request


def _parse_json_response(content: str) -> TaskPlan:
    """从 LLM 响应文本中提取 JSON 并解析为 TaskPlan。"""
    # 尝试直接解析
    try:
        data = json.loads(content)
        return _dict_to_plan(data)
    except json.JSONDecodeError:
        pass

    # 尝试从 markdown 代码块中提取
    import re
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
    if m:
        try:
            data = json.loads(m.group(1))
            return _dict_to_plan(data)
        except json.JSONDecodeError:
            pass

    # 最后尝试找 { ... } 块
    m = re.search(r"\{[\s\S]*\}", content)
    if m:
        try:
            data = json.loads(m.group(0))
            return _dict_to_plan(data)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"无法从 LLM 响应中解析 TaskPlan JSON: {content[:200]}...")


def _dict_to_plan(data: dict[str, Any]) -> TaskPlan:
    """将 LLM 返回的 dict 转换为 TaskPlan，补充默认值。"""
    steps_data = data.get("steps", [data] if "step_id" in data else [])
    steps: list[TaskStep] = []

    for i, sd in enumerate(steps_data):
        step = TaskStep(
            step_id=sd.get("step_id", i + 1),
            task_type=sd.get("task_type", "object_detection"),
            source=sd.get("source", ""),
            prompts=sd.get("prompts", []),
            confidence_threshold=sd.get("confidence_threshold", DEFAULT_CONFIDENCE),
            iou_threshold=sd.get("iou_threshold", DEFAULT_IOU),
            model_name=sd.get("model_name", DEFAULT_MODEL),
            export_format=sd.get("export_format", DEFAULT_EXPORT),
            sahi=sd.get("sahi", False),
        )
        steps.append(step)

    if not steps:
        raise ValueError("TaskPlan 至少需要一个步骤")

    return TaskPlan(
        steps=steps,
        confirm_timeout=data.get("confirm_timeout", DEFAULT_TIMEOUT),
    )


def _parse_benchmark_json(content: str) -> BenchmarkRequest:
    """从 LLM 响应文本中提取 JSON 并解析为 BenchmarkRequest。"""
    # 尝试直接解析
    try:
        data = json.loads(content)
        return _dict_to_benchmark(data)
    except json.JSONDecodeError:
        pass

    # 尝试从 markdown 代码块中提取
    import re
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
    if m:
        try:
            data = json.loads(m.group(1))
            return _dict_to_benchmark(data)
        except json.JSONDecodeError:
            pass

    # 最后尝试找 { ... } 块
    m = re.search(r"\{[\s\S]*\}", content)
    if m:
        try:
            data = json.loads(m.group(0))
            return _dict_to_benchmark(data)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"无法从 LLM 响应中解析 BenchmarkRequest JSON: {content[:200]}...")


def _dict_to_benchmark(data: dict[str, Any]) -> BenchmarkRequest:
    """将 LLM 返回的 dict 转换为 BenchmarkRequest，补充默认值并验证。"""
    dataset = data.get("dataset", "").strip().lower()
    assert isinstance(dataset, str)  # LLM 返回 JSON 中 dataset 字段应为字符串

    # 中文名 → key 映射
    if dataset and dataset not in BENCHMARK_DATASETS:
        dataset = DATASET_CN_MAP.get(dataset, dataset)

    # 验证数据集名
    if dataset and dataset not in BENCHMARK_DATASETS:
        logger.warning("Unknown dataset '%s', using as-is", dataset)

    task_type = data.get("task_type", "detection")
    # 从数据集推导 task_type
    if dataset in BENCHMARK_DATASETS:
        task_type = BENCHMARK_DATASETS[dataset]["task_type"]

    return BenchmarkRequest(
        dataset=dataset,
        task_type=task_type,
        model=data.get("model", BENCHMARK_DEFAULT_MODEL),
        seg_model=data.get("seg_model", BENCHMARK_DEFAULT_SEG_MODEL),
        conf=float(data.get("conf", BENCHMARK_DEFAULT_CONF)),
        iou=float(data.get("iou", BENCHMARK_DEFAULT_IOU)),
        max_images=int(data.get("max_images", BENCHMARK_DEFAULT_MAX_IMAGES)),
        top_classes=int(data.get("top_classes", 20)),
        sahi=bool(data.get("sahi", False)),
    )
