"""任务规划器 —— 将自然语言指令解析为 TaskPlan。

使用 LLM 从用户文本中提取结构化标注参数。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from auto2dlabel.agent import json, logging
from auto2dlabel.agent.dialog import parse_with_dialog
from auto2dlabel.agent.llm import LLMClient
from auto2dlabel.configs.datasets import format_datasets_summary
from auto2dlabel.configs.task_params import format_task_params_summary
from auto2dlabel.models.model_catalog import format_catalog_summary
from auto2dlabel.schema.task_plan import (
    BENCHMARK_DATASETS,
    BENCHMARK_DEFAULT_CONF,
    BENCHMARK_DEFAULT_IOU,
    BENCHMARK_DEFAULT_MAX_IMAGES,
    BENCHMARK_DEFAULT_MODEL,
    BENCHMARK_DEFAULT_SEG_MODEL,
    DATASET_CN_MAP,
    DEFAULT_CONFIDENCE,
    DEFAULT_EXPORT,
    DEFAULT_IOU,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    BenchmarkRequest,
    PlanQuestion,
    TaskPlan,
    TaskStep,
    sanitize_benchmark_request,
    sanitize_task_plan,
)

logger = logging.getLogger(__name__)

_PLANNER_SYSTEM_PROMPT = """You are a task planner for an image annotation tool. Parse user's NL into JSON.

Output ONLY valid JSON:
{
  "steps": [
    {
      "step_id": 1,
      "task_type": "object_detection|instance_segmentation|classification|obb_detection|tracking|pose_estimation",
      "source": "/path/to/images",
      "prompts": ["car", "person"],
      "confidence_threshold": 0.1,
      "iou_threshold": 0.3,
      "model_name": "yolo26x.pt",
      "export_format": "coco",
      "sahi": false,
      "num_workers": null,
      "batch_size": null
    }
  ],
  "confirm_timeout": 30
}

Rules:
- task_type: "tracking" if user says 跟踪/追踪/track/video/视频/序列 (prompts are fixed classes; ByteTrack/BoT-SORT are trackers chosen by the CLI, NOT part of this JSON); "classification" if user says 分类/classify/打标签/图片分类 (prompts are candidate labels); "obb_detection" if user says 旋转框/obb/rotate/oriented (prompts are classes); "pose_estimation" if user says 姿态/姿势/关键点/keypoint/pose/骨骼 (prompts are person classes); "instance_segmentation" if user says 分割/segmentation/mask; otherwise "object_detection"
- source: data path (image dir, or video .mp4/.avi/.mov/.mkv for tracking). "" if not specified.
- prompts: English only. Map: 汽车/车辆→car, 行人/人→person, 自行车/单车→bicycle, 摩托车→motorcycle, 公共汽车/公交车→bus, 卡车→truck, 狗→dog, 猫→cat
- confidence_threshold: default 0.1; extract if user says 置信度/conf/阈值X (e.g. 置信度0.5)
- iou_threshold: default 0.3; extract if user says iou/IoU X (e.g. iou0.5)
- model_name: map hints to names. Key mappings: faster rcnn→fasterrcnn_resnet50_fpn_v2, yolo→yolo26x.pt, yolo12→yolo12x.pt, rtdetr/rt-detr/detr→rtdetr-l.pt, grounding dino→IDEA-Research/grounding-dino-tiny, clip→openai/clip-vit-base-patch32, siglip→google/siglip-base-patch16-224, convnext→convnext_large, swin→swin_b, maxvit→maxvit_t, efficientnet→efficientnet_v2_l, vit→vit_b_16, resnet→resnet50, resnext→resnext101_32x8d, obb→yolo11n-obb.pt, pose→yolo11n-pose.pt, fcn→fcn_resnet50, deeplab→deeplabv3_resnet50, lraspp→lraspp_mobilenet_v3_large. cityscapes 街景分割→maskrcnn_r50_cityscapes. sam3/sam/maskrcnn/fastsam/fcn*/deeplabv3*/lraspp* are SEGMENTATION models — keep them as model_name (the system auto-routes them). ByteTrack/BoT-SORT are TRACKERS not models — ignore them for model_name. default: yolo26x.pt
- export_format: "cls" if task_type=classification; "dota" if task_type=obb_detection;
  "mot" if task_type=tracking; otherwise "coco" (pose keypoints 内嵌 COCO JSON，无需单独格式)
- sahi: true if user mentions SAHI/切片/切块/slicing/sahi/大图. default false.
- num_workers/batch_size: extract ONLY if the user explicitly specifies them
  (e.g. "batch_size=8", "num_workers=4", "批量4"). 用户说「跑最大/最大批量/自动/
  尽可能大」→ batch_size=null（null 语义 = 执行阶段自动实测最大 batch，不是 1；
  无批量能力的模型自动逐图）。Otherwise null — the CLI will ask interactively
  or auto-fill a GPU-based recommendation.
- HYPERPARAM values come ONLY from explicit specs (batch_size=N/批量N/
  num_workers=N/进程N/conf X/iou X). Digits inside paths/filenames
  (COCO2017, 000000000139.jpg, 000123.png) are NEVER hyperparameters —
  do NOT extract them as batch_size/num_workers/confidence_threshold.
- questions: ONLY when key fields are missing (source / prompts), e.g.
  "questions": [{"id": "step1.source", "question": "图像目录在哪里？"}],
  "questions": [{"id": "step1.prompts", "question": "要检测哪些类别？(如 car, person)"}].
  id = "step{step_id}.{field}". questions MUST accompany the full steps JSON
  (same step list, missing fields left empty). No questions when nothing is missing.
- Dialog: next round user's answers are fed back as
  "补充信息（用户在以下问题的回答，请据此更新 JSON）：" + "- [step1.source] <question>" + "用户回答：<answers>".
  Then fill the previously-missing fields from the answers and output empty/omit questions.
- Multiple tasks separated by 然后/再/；/; → multiple steps.
- Only JSON. No other text."""

# 追加模型目录摘要（字符串拼接：原文含 JSON 花括号，不可改 f-string）
_PLANNER_SYSTEM_PROMPT = _PLANNER_SYSTEM_PROMPT + f"""

{format_catalog_summary()}
Model selection:
- 用户点名或描述能力（速度/精度/开放词汇/旋转框/分割）→ 从上述目录选最贴合项
- 用户未指定 → object_detection→yolo26x.pt, obb_detection→yolo11n-obb.pt,
  instance_segmentation→sam_b.pt, classification→openai/clip-vit-base-patch32,
  semantic_segmentation→fcn_resnet50
- 自主选型时在该 step 加 "model_hint": "<一行中文理由>"
"""

# 追加数据集路径摘要（方案 A：LLM 路径引导；运行时取当前值，env 覆盖生效）
_PLANNER_SYSTEM_PROMPT = _PLANNER_SYSTEM_PROMPT + f"""

{format_datasets_summary()}
Dataset resolution:
- 指令中的数据集名（如「检测 COCO2017 验证集」/「KITTI 帧」）→ 用上表路径构造 source
  （如 /root/autodl-tmp/Documents/datasets/COCO2017/val2017），source 用绝对路径
- 图像在子目录时，source 指向图像所在子目录而非数据集根目录：
  KITTI → .../KITTI/object/training/image_2；COCO → .../COCO2017/val2017；
  其余按上表 subdirs/note 定位图像目录
- 指令给的是具体路径/文件名 → source 原样保留，不查上表
"""

# 追加任务参数规格摘要（规格表单一事实源：REQUIRED 字段缺失时 CLI 会追问）
_PLANNER_SYSTEM_PROMPT = _PLANNER_SYSTEM_PROMPT + f"""

{format_task_params_summary()}
- 缺参指令（未提 REQUIRED 字段）→ 对应字段留空/默认 + questions 追问，
  不要臆造路径/类别
"""


_BENCHMARK_SYSTEM_PROMPT = f"""You are a benchmark planner for an image annotation tool. Parse user's NL into JSON.

Output ONLY valid JSON:
{{
  "dataset": "<dataset_key>",
  "task_type": "detection | segmentation | classification | obb_detection",
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
- dota_obb: DOTA Task1 oriented bounding box detection (旋转框)
- mot: MOT17+MOT20 pedestrian detection (密集行人)
- coco_seg: COCO 2017 val instance segmentation
- cityscapes: Cityscapes instance segmentation (城市街景)
- nuimages: nuImages instance segmentation
- d2sa: D2SA retail shelf instance segmentation (零售货架)
- imagenet100: ImageNet100 image classification (图像分类)

Rules:
- dataset: Map user's words to dataset keys. 中文映射: COCO/COCO2017→coco, VOC→voc2007, 航拍→dota, 旋转框/OBB→dota_obb, MOT/行人→mot, COCO分割→coco_seg, 街景/cityscapes→cityscapes, nuImages→nuimages, D2SA/零售/货架→d2sa, ImageNet→imagenet100
- task_type: "classification" if user mentions 分类/classify/classification or dataset is imagenet100; "segmentation" if user mentions 分割/segmentation/mask/segment or dataset is coco_seg/cityscapes/nuimages/d2sa; "obb_detection" if user mentions 旋转框/obb/rotate/oriented or dataset is dota_obb; otherwise "detection"
- model: Map hints. 默认{BENCHMARK_DEFAULT_MODEL}. Key mappings: yolo→{BENCHMARK_DEFAULT_MODEL}, yolo12→yolo12x.pt, yolo26→yolo26x.pt, obb→yolo11n-obb.pt, rtdetr/rt-detr/detr→rtdetr-l.pt, faster rcnn/frcnn→fasterrcnn_resnet50_fpn_v2, grounding dino→IDEA-Research/grounding-dino-tiny. If user says a specific model name that looks like a real model (ends with .pt or contains /), use it directly.
- seg_model: For segmentation only. 默认{BENCHMARK_DEFAULT_SEG_MODEL}. Map: fastsam→FastSAM-s.pt, fastsam-x→FastSAM-x.pt, sam/sam_b/sam2_b→sam_b.pt, sam_l/sam2_l→sam2_l.pt, sam3→sam3.pt, maskrcnn→maskrcnn_resnet50_fpn_v2, cityscapes 域内/cityscapes 分割→maskrcnn_r50_cityscapes
- conf: Extract from "conf=X" or "阈值X" or "置信度X". 默认{BENCHMARK_DEFAULT_CONF}
- iou: Extract from "iou=X". 默认{BENCHMARK_DEFAULT_IOU}
- max_images: Extract from "N张图" or "N张" or "N images" or "max=N". 默认{BENCHMARK_DEFAULT_MAX_IMAGES}
- sahi: true if user mentions SAHI/切片/切块/slicing/sahi. 默认false
- If user doesn't specify dataset, leave dataset="".
- Only JSON. No other text."""


def _gpu_context_line() -> str:
    """注入 LLM 的设备信息 + batch 推荐规则（无 CUDA GPU 时返回空串）。

    让 LLM 在用户显式提到 GPU/显存相关措辞时有上下文可依；数值规则与
    schema.task_plan.recommend_batch_params 一致（代码兜底同源）。
    """
    from auto2dlabel.schema.task_plan import detect_gpu_memory_gb, recommend_batch_params

    mem = detect_gpu_memory_gb()
    if mem is None:
        return ""
    try:
        import torch

        name = torch.cuda.get_device_name(0)
    except ImportError:
        name = "CUDA GPU"

    lines = [f"Device context: GPU={name}, VRAM={mem}GB"]
    for task_type, label in [
        ("object_detection", "object_detection/obb_detection"),
        ("instance_segmentation", "instance_segmentation"),
        ("classification", "classification"),
    ]:
        bs, nw = recommend_batch_params(task_type, mem)
        lines.append(f"  recommended for {label}: batch_size={bs}, num_workers={nw}")
    return "\n".join(lines)


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

        messages: list[dict[str, str]] = [
            {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": _gpu_context_line() + "\n\n" + instruction},
        ]

        response = self.llm.chat(messages, tools=None, temperature=0.0)

        if not response.content:
            raise ValueError("LLM 返回空响应，无法解析任务")

        plan = _parse_json_response(response.content)
        plan.raw_instruction = instruction

        for fix in sanitize_task_plan(plan):  # LLM 输出守卫（路径数字误提等）
            logger.info("参数守卫修正: %s", fix)

        _apply_confirm_timeout(plan, confirm_timeout)

        logger.info("Parsed plan: %d steps", len(plan.steps))
        for s in plan.steps:
            logger.info("  %s", s.summary)

        return plan

    def parse_dialog(
        self,
        instruction: str,
        ask_fn: Callable[[list[PlanQuestion]], str],
        confirm_timeout: int = DEFAULT_TIMEOUT,
        max_rounds: int = 3,
    ) -> TaskPlan:
        """多轮对话解析（v0.6 P1）：缺参时 LLM 出 questions → 收集回答 → 回喂。

        复用 parse_with_dialog 通用骨架（agent/dialog.py，schema 无关）；
        system prompt 与 parse 同源（catalog 摘要 + GPU 上下文，设备信息放
        system 角色——与 parse 的 user 角色差异无语义影响）。
        轮次耗尽 / 用户放弃 → 返回缺参 plan（调用方 _fill_missing_params 兜底）。
        LLM 响应非法 / 空响应 → ValueError 传播（调用方降级链处理）。
        """
        system_prompt = _PLANNER_SYSTEM_PROMPT + "\n\n" + _gpu_context_line()
        plan = parse_with_dialog(
            llm=self.llm,
            system_prompt=system_prompt,
            parse_fn=_parse_json_response,
            ask_fn=ask_fn,
            instruction=instruction,
            max_rounds=max_rounds,
        )
        for fix in sanitize_task_plan(plan):  # LLM 输出守卫（对话路径同源）
            logger.info("参数守卫修正: %s", fix)
        _apply_confirm_timeout(plan, confirm_timeout)
        logger.info("Parsed plan via dialog: %d steps", len(plan.steps))
        return plan

    def parse_benchmark(self, instruction: str) -> BenchmarkRequest:
        """将自然语言指令解析为 BenchmarkRequest。

        Args:
            instruction: 用户自然语言指令（如 "用 yolo26x 跑 COCO 检测，50 张图 conf=0.3"）。

        Returns:
            BenchmarkRequest。
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _BENCHMARK_SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]

        response = self.llm.chat(messages, tools=None, temperature=0.0)

        if not response.content:
            raise ValueError("LLM 返回空响应，无法解析 benchmark 参数")

        request = _parse_benchmark_json(response.content)
        for fix in sanitize_benchmark_request(request, instruction):
            logger.info("benchmark 参数守卫修正: %s", fix)
        logger.info("Parsed benchmark: %s", request.summary)
        return request


def _apply_confirm_timeout(plan: TaskPlan, confirm_timeout: int) -> None:
    """应用 CLI 指定 timeout（仅当 LLM 未显式给出时；v0.5 语义原样，parse/parse_dialog 共用）。"""
    if plan.confirm_timeout == DEFAULT_TIMEOUT:
        plan.confirm_timeout = confirm_timeout


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
            model_hint=sd.get("model_hint", ""),
            export_format=sd.get("export_format", DEFAULT_EXPORT),
            sahi=sd.get("sahi", False),
            num_workers=sd.get("num_workers"),
            batch_size=sd.get("batch_size"),
        )
        steps.append(step)

    if not steps:
        raise ValueError("TaskPlan 至少需要一个步骤")

    return TaskPlan(
        steps=steps,
        confirm_timeout=data.get("confirm_timeout", DEFAULT_TIMEOUT),
        # v0.6 对话问题（致命点：LLM JSON → plan 走本函数，不经过 TaskPlan.from_dict）
        questions=PlanQuestion.from_list(data.get("questions")),
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

    # OBB 任务默认旋转框权重（yolo26x.pt 无 OBB 头）
    model = data.get("model", "")
    if not model:
        model = "yolo11n-obb.pt" if task_type == "obb_detection" else BENCHMARK_DEFAULT_MODEL

    return BenchmarkRequest(
        dataset=dataset,
        task_type=task_type,
        model=model,
        seg_model=data.get("seg_model", BENCHMARK_DEFAULT_SEG_MODEL),
        conf=float(data.get("conf", BENCHMARK_DEFAULT_CONF)),
        iou=float(data.get("iou", BENCHMARK_DEFAULT_IOU)),
        max_images=int(data.get("max_images", BENCHMARK_DEFAULT_MAX_IMAGES)),
        top_classes=int(data.get("top_classes", 20)),
        sahi=bool(data.get("sahi", False)),
    )
