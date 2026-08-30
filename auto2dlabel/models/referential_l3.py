"""指代检测 L3 解析器（Qwen2-VL-7B-Instruct 4bit，v0.5 GPU 项）。

L2（Florence-2）也覆盖不了的复杂上下文指代（「第二辆车后面穿红衣服的人」）
由 Qwen2-VL-7B 看图解析。与 L2 **同一调用点**（ReferentialResolver Protocol，
TrackingTool 零改动），按能力/成本阶梯升级：

- `--refer-l3` 显式直用 L3；未显式指定时 `CascadeReferentialResolver`
  L2 失败自动升级 L3（L2 resolver 记录 last_failed 状态）。
- **序列级一次调用**（首帧锁定 → 后续帧轨迹-检测匹配维持），不做逐帧。

GPU 必需：NF4 4bit ~4.5G 权重 + KV cache 12GB 显存可用；CPU 分钟级不可用
（_load 直接拒绝）。权重：官方 fp16 16GB 经 bitsandbytes NF4 量化加载，
下载脚本 `weights/download_qwen_l3.sh`（HF 镜像；未就位快速失败不触发
静默 16GB 下载）。

接口（与 L2 同协议，duck typing 注入 Fake 零真实权重单测铁律）：
    resolve(image_path, phrase, bboxes) -> list[Bbox]
输出 = 输入框子集（保原框）；解析失败「宁多勿漏」返回全部。
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from auto2dlabel.configs.model_catalog import WEIGHTS_DIR
from auto2dlabel.models.referential import (
    ReferentialResolver,
    match_boxes_by_center,
)
from auto2dlabel.schema.annotation import Bbox

# 权重缓存位置（HF_HOME=weights/hf 的 hub 布局）；_load 前检查，未就位
# 快速失败（提示下载脚本），绝不静默触发 16GB 下载
L3_MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
L3_CACHE_DIR = WEIGHTS_DIR / "hf" / "hub" / "models--Qwen--Qwen2-VL-7B-Instruct"

# 官方 grounding prompt 模式（Qwen2-VL 官方 demo 同款措辞：要求 JSON 输出）
GROUNDING_PROMPT = (
    'Locate "{phrase}" in this image and output the bounding box '
    'coordinates in JSON: [{{"bbox_2d": [x1, y1, x2, y2], "label": "target"}}].'
)

# 官方坐标约定：bbox 值在 1000×1000 归一化网格（官方 demo 后处理 /1000）；
# 兼容 [0,1] 归一化（实测启发式：任一坐标绝对值 > 1 视为 1000 系——
# [0,1] 系坐标恒 ≤1，1000 系坐标几乎恒 >1）
COORD_GRID_SCALE = 1000.0


def parse_qwen_bboxes(
    text: str, orig_w: int, orig_h: int
) -> list[tuple[float, float, float, float]]:
    """Qwen2-VL 输出文本 → 原图像素坐标框列表（纯函数，测试直测）。

    支持三种输出形态（按优先级）：
    1. JSON 数组 {"bbox_2d": [x1,y1,x2,y2]}（官方 prompt 要求的形式）
    2. <|box_start|>(x1,y1),(x2,y2)<|box_end|>（模型偶尔输出原生 box token）
    3. 裸 "[x1, y1, x2, y2]" 数字列表
    坐标 1000 系（官方）与 [0,1] 系自动判别；无框返回 []。
    """
    boxes: list[tuple[float, float, float, float]] = []

    # 形态 1：JSON bbox_2d
    for m in re.finditer(r'"bbox_2d"\s*:\s*\[([^\]]+)\]', text):
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", m.group(1))]
        if len(nums) >= 4:
            boxes.append((nums[0], nums[1], nums[2], nums[3]))

    # 形态 2：<|box_start|>(x1,y1),(x2,y2)<|box_end|>
    for m in re.finditer(
        r"<\|box_start\|>\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)"
        r"\s*,\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)\s*<\|box_end\|>",
        text,
    ):
        boxes.append(tuple(float(g) for g in m.groups()))  # type: ignore[arg-type]

    # 形态 3：裸数字列表（仅当 1/2 均无命中时兜底）
    if not boxes:
        bare = re.search(r"\[\s*[-\d.,\s]+\s*\]", text)
        if bare:
            nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", bare.group(0))]
            if len(nums) == 4:
                boxes.append((nums[0], nums[1], nums[2], nums[3]))

    out: list[tuple[float, float, float, float]] = []
    for x1, y1, x2, y2 in boxes:
        if x1 == x2 or y1 == y2:
            continue  # 退化框
        scale = COORD_GRID_SCALE if max(abs(x1), abs(x2)) > 1.0 else 1.0
        out.append((
            min(x1, x2) / scale * orig_w,
            min(y1, y2) / scale * orig_h,
            max(x1, x2) / scale * orig_w,
            max(y1, y2) / scale * orig_h,
        ))
    return out


class QwenVLReferentialResolver:
    """Qwen2-VL-7B 指代解析（懒加载 + GPU 守卫 + 权重就位守卫）。

    resolve 流程：grounding JSON prompt → generate(do_sample=False) →
    parse_qwen_bboxes → match_boxes_by_center 匹配回原框。
    last_failed 供 CascadeReferentialResolver 阶梯升级判定。
    """

    def __init__(
        self,
        model_name: str = L3_MODEL_NAME,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._processor: Any = None
        self._model: Any = None
        self._last_failed = False

    @property
    def last_failed(self) -> bool:
        """最近一次 resolve 是否失败（供阶梯升级判定；单测 FakeL2 同款属性）。"""
        return self._last_failed

    def _load(self) -> tuple[Any, Any]:
        """懒加载 processor + model（幂等；无 GPU / 权重未就位快速失败）。

        4bit NF4 量化加载（bitsandbytes），12GB 显存可用；CPU 直接拒绝
        （分钟级不可用，红线见 milestone/v0.5.md）。
        """
        if self._processor is not None and self._model is not None:
            return self._processor, self._model
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("指代 L3（Qwen2-VL-7B）需 GPU；当前无 CUDA 设备")
        if not L3_CACHE_DIR.exists():
            raise RuntimeError(
                f"指代 L3 权重未就位: {L3_CACHE_DIR}\n"
                "请先运行 auto2dlabel/weights/download_qwen_l3.sh 下载"
            )
        os.environ.setdefault("HF_HOME", str(WEIGHTS_DIR / "hf"))
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        try:
            from transformers import (
                AutoModelForImageTextToText,
                AutoProcessor,
                BitsAndBytesConfig,
            )
        except ImportError:
            raise ImportError("transformers 未安装，请运行: pip install transformers")

        self._processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            self._model_name
        )
        quantization_config = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            self._model_name, quantization_config=quantization_config
        )
        self._model.eval()
        return self._processor, self._model

    def resolve(self, image_path: str, phrase: str, bboxes: list[Bbox]) -> list[Bbox]:
        """Qwen2-VL grounding 解析 → 原框子集；任何失败「宁多勿漏」返回全部。"""
        self._last_failed = False
        if not bboxes:
            return []
        try:
            processor, model = self._load()
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
            orig_w, orig_h = image.size
            prompt = GROUNDING_PROMPT.format(phrase=phrase)
            # transformers 5.x Qwen2-VL：必须经 chat template 注入
            # <|image_pad|> 图像占位 token，否则 "Image features and image
            # tokens do not match"（官方 demo 同款用法）
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }]
            text_prompt = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False)
            inputs = processor(text=[text_prompt], images=[image], return_tensors="pt")
            import torch

            device = self._device or ("cuda:0" if torch.cuda.is_available() else "cpu")
            # 全部张量（input_ids + pixel_values 等）上设备；非张量原样保留
            inputs = {
                k: (v.to(device) if hasattr(v, "to") else v)
                for k, v in inputs.items()
            }
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs, max_new_tokens=512, do_sample=False
                )
            text = processor.batch_decode(
                generated_ids[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=False,
            )[0]
            f_boxes = parse_qwen_bboxes(text, orig_w, orig_h)
            if not f_boxes:
                self._last_failed = True
                return list(bboxes)
            kept = match_boxes_by_center(f_boxes, bboxes)
            if not kept:
                self._last_failed = True
            return kept if kept else list(bboxes)
        except Exception as e:
            self._last_failed = True
            from auto2dlabel.cli_common import console

            console.print(f"[yellow]⚠ 指代 L3 解析失败（保留全部框）: {e}[/yellow]")
            return list(bboxes)


class CascadeReferentialResolver:
    """阶梯升级解析器：L2 失败自动升级 L3（能力/成本阶梯，milestone v0.5 §3）。

    primary 的 last_failed 状态（Florence2ReferentialResolver / Fake 同款
    属性）为升级信号；fallback 为 None 时纯 L2 语义（CPU 环境零影响）。
    """

    def __init__(
        self,
        primary: ReferentialResolver,
        fallback: ReferentialResolver | None = None,
    ) -> None:
        self._primary = primary
        self._fallback = fallback

    @property
    def last_failed(self) -> bool:
        """最近一次 resolve 是否失败（外层再包装时可用）。"""
        return bool(getattr(self._primary, "last_failed", False)) and (
            self._fallback is None or bool(getattr(self._fallback, "last_failed", False))
        )

    def resolve(self, image_path: str, phrase: str, bboxes: list[Bbox]) -> list[Bbox]:
        """L2 先试；last_failed 且 fallback 存在 → 升级 L3。"""
        result = self._primary.resolve(image_path, phrase, bboxes)
        if (
            self._fallback is not None
            and getattr(self._primary, "last_failed", False)
        ):
            from auto2dlabel.cli_common import console

            console.print("[dim]指代 L2 未解析 → 升级 L3 (Qwen2-VL-7B)[/dim]")
            return self._fallback.resolve(image_path, phrase, bboxes)
        return result


def create_referential_l3_resolver(
    model_name: str = L3_MODEL_NAME,
    **kwargs: Any,
) -> ReferentialResolver:
    """工厂函数：创建 L3 解析器（Qwen2-VL-7B）。"""
    return QwenVLReferentialResolver(model_name=model_name, **kwargs)


def create_cascade_referential_resolver(
    l2_kwargs: dict[str, Any] | None = None,
    l3_kwargs: dict[str, Any] | None = None,
) -> ReferentialResolver:
    """工厂函数：L2 → 失败升级 L3 阶梯解析器（v0.5 指代调用点默认形态）。

    均懒加载：构造零加载零下载；L3 仅在 L2 失败且权重就位/GPU 可用时
    真正加载，否则 resolve 内快速失败黄字降级（宁多勿漏）。
    """
    from auto2dlabel.models.referential import create_referential_resolver

    l2 = create_referential_resolver(**(l2_kwargs or {}))
    l3 = create_referential_l3_resolver(**(l3_kwargs or {}))
    return CascadeReferentialResolver(l2, l3)


def parse_json_for_tests(text: str) -> Any:
    """测试辅助：宽容 JSON 解析（输出片段可能带前后杂质）。"""
    m = re.search(r"\[.*\]", text, re.DOTALL)
    return json.loads(m.group(0)) if m else None
