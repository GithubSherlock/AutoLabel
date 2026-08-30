"""指代检测 L2 解析器（Florence-2 `<OD>` 开放词汇检测兜底，v0.5）。

v0.4 3a 过滤链的 L2 兜底：关系/复合指代（「红车旁边的行人」）在 L1
（属性 CLIP 逐框 + 方位分位）覆盖不了时，由 Florence-2 直接看图出框。
**序列级一次调用**（首帧解析锁定目标 → 后续帧轨迹-检测匹配维持），
不做逐帧（CPU 单图 5-30s，成本红线见 milestone/v0.5.md）。

接口（Protocol，duck typing 注入 Fake 零真实权重单测铁律）：
    resolve(image_path, phrase, bboxes) -> list[Bbox]
输出 = 输入框子集（保原框对象，track_id/置信度无损）；解析失败
「宁多勿漏」返回全部（L2 是兜底，过滤失败不得丢检测）。

实现：vendored 官方代码（models/vendor_florence2/，Apache-2.0，
不 trust_remote_code）——transformers 5.15 内置 florence2 集成有根本缺陷
（缺 tokenizer 类 + 视觉骨干重写与官方 checkpoint 不兼容），vendor 动机
与补丁清单见 vendor_florence2/README.md。权重进 auto2dlabel/weights/hf/。
"""

from __future__ import annotations

import math
import os
from typing import Any, Protocol

from auto2dlabel.configs.model_catalog import WEIGHTS_DIR
from auto2dlabel.schema.annotation import Bbox

# Florence-2 指代任务词条（开放词汇检测：接受英文指代短语输入）
REFER_TASK = "<OPEN_VOCABULARY_DETECTION>"

# 中心距匹配默认阈值 = max(50px, 原框短边/2)——Florence 输出框与原检测框
# 中心偏移超过该值视为不同目标（小框用小阈值，大框放宽）
DEFAULT_MATCH_DIST_BASE = 50.0

# 官方 preprocessor_config.json 的训练/推理尺寸（768×768 方形）。官方
# _encode_image 要求方形特征图（DaViT stride 32 → 24×24=576+1=577 tokens），
# 非方输入 assert 直接炸——推理前必须 squash 到该尺寸（不保宽高比，坐标
# 按 x/y 分别反变换回原图系，线性可逆）
FLORENCE_INFER_SIZE = 768


class ReferentialResolver(Protocol):
    """指代解析器协议 —— 测试注入 FakeResolver（零真实权重铁律，同 ReID 模式）。"""

    def resolve(self, image_path: str, phrase: str, bboxes: list[Bbox]) -> list[Bbox]:
        """解析英文指代短语 → 输入候选框子集（保原框；失败宁多勿漏返回全部）。"""
        ...


def match_boxes_by_center(
    florence_boxes: list[tuple[float, float, float, float]],
    bboxes: list[Bbox],
    dist_limit: float | None = None,
) -> list[Bbox]:
    """Florence 输出框 → 原检测框子集（贪心最近中心距匹配，保原框）。

    Florence-2 是看图出框（无 track 语义），必须匹配回原检测框才能
    保住置信度/track 字段。每个 Florence 框匹配最近的未占用原框；
    dist_limit 为 None 时用默认阈值（每框独立：max(50, 短边/2)），
    显式注入时全局统一（测试用）。
    """
    if not florence_boxes or not bboxes:
        return []
    centers = [(b.x + b.width / 2, b.y + b.height / 2) for b in bboxes]
    used = [False] * len(bboxes)
    out: list[Bbox] = []
    for fx1, fy1, fx2, fy2 in florence_boxes:
        fc = ((fx1 + fx2) / 2, (fy1 + fy2) / 2)
        best_i, best_d = -1, math.inf
        for i, (cx, cy) in enumerate(centers):
            if used[i]:
                continue
            d = math.hypot(cx - fc[0], cy - fc[1])
            if d < best_d:
                best_i, best_d = i, d
        if best_i < 0:
            continue
        if dist_limit is None:
            b = bboxes[best_i]
            if best_d > max(DEFAULT_MATCH_DIST_BASE, min(b.width, b.height) / 2):
                continue
        elif best_d > dist_limit:
            continue
        used[best_i] = True
        out.append(bboxes[best_i])
    return out


class Florence2ReferentialResolver:
    """Florence-2 指代解析（懒加载 + ImportError 守卫，仿 classification.py 模式）。

    resolve 流程：<OPEN_VOCABULARY_DETECTION> 任务词条 + 英文短语 →
    generate(num_beams=3) → post_process_generation 解析 bboxes
    （transformers 已 dequantize 为像素坐标）→ match_boxes_by_center 匹配回原框。
    """

    def __init__(
        self,
        model_name: str = "microsoft/Florence-2-base",
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._processor: Any = None
        self._model: Any = None
        self._last_failed = False

    @property
    def last_failed(self) -> bool:
        """最近一次 resolve 是否失败（供 L3 阶梯升级判定，referential_l3.py）。"""
        return self._last_failed

    def _load(self) -> tuple[Any, Any]:
        """懒加载 processor + model（幂等；transformers 未装抛 ImportError）。

        使用 vendored 官方代码（models/vendor_florence2/，不 trust_remote_code）：
        5.15 内置 florence2 集成缺 tokenizer 类且视觉骨干与官方 checkpoint
        不兼容（详见 vendor_florence2/README.md）。官方 checkpoint 以 fp16 存储
        （config torch_dtype=float16）：CPU 推理必须 fp32 加载（fp16 权重与
        fp32 图像输入 dtype 冲突），GPU 保持 fp16。
        """
        if self._processor is not None and self._model is not None:
            return self._processor, self._model
        os.environ.setdefault("HF_HOME", str(WEIGHTS_DIR / "hf"))
        # AutoDL 环境 huggingface.co 不可直连（缓存未命中时需镜像；setdefault 不覆盖用户设置）
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        try:
            import torch

            from auto2dlabel.models.vendor_florence2.modeling_florence2 import (
                Florence2ForConditionalGeneration,
            )
            from auto2dlabel.models.vendor_florence2.processing_florence2 import (
                Florence2Processor,
            )
        except ImportError:
            raise ImportError("transformers 未安装，请运行: pip install transformers")
        dtype = torch.float32 if not torch.cuda.is_available() else torch.float16
        self._processor = Florence2Processor.from_pretrained(self._model_name)
        # vendored 官方代码无类型标注（mypy overrides 只抑制内部错误，跨模块
        # 调用仍需定点 ignore）
        self._model = Florence2ForConditionalGeneration.from_pretrained(  # type: ignore[no-untyped-call]
            self._model_name, dtype=dtype
        )
        return self._processor, self._model

    def resolve(self, image_path: str, phrase: str, bboxes: list[Bbox]) -> list[Bbox]:
        """Florence-2 <OD> 解析 → 原框子集；任何失败「宁多勿漏」返回全部。"""
        self._last_failed = False
        if not bboxes:
            return []
        try:
            processor, model = self._load()
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
            orig_w, orig_h = image.size
            # 官方 _encode_image 要求方形特征图：squash 到训练尺寸 768×768
            # （不保宽高比——坐标按 x/y 分别反变换回原图系，线性可逆）
            image = image.resize((FLORENCE_INFER_SIZE, FLORENCE_INFER_SIZE))
            inputs = processor(
                text=f"{REFER_TASK} {phrase}", images=image, return_tensors="pt"
            )
            import torch

            device = self._device or ("cuda:0" if torch.cuda.is_available() else "cpu")
            model.to(device)  # 已在目标设备时为 no-op
            # 浮点输入对齐模型 dtype（GPU fp16 模型 + fp32 输入 dtype 冲突
            # 直接 RuntimeError；input_ids 等整型保持不动）
            model_dtype = next(model.parameters()).dtype
            inputs = {
                k: (
                    v.to(device)
                    if not v.is_floating_point()
                    else v.to(device=device, dtype=model_dtype)
                )
                for k, v in inputs.items()
            }
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
            )
            text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed = processor.post_process_generation(
                text,
                task=REFER_TASK,
                image_size=(FLORENCE_INFER_SIZE, FLORENCE_INFER_SIZE),
            )
            od = parsed.get(REFER_TASK) or {}
            raw_boxes: list[Any] = od.get("bboxes") or []
            if not raw_boxes:
                self._last_failed = True
                return list(bboxes)
            # 768 系 → 原图系（x/y 独立缩放，反 squash）
            sx, sy = orig_w / FLORENCE_INFER_SIZE, orig_h / FLORENCE_INFER_SIZE
            f_boxes = [
                (float(b[0]) * sx, float(b[1]) * sy, float(b[2]) * sx, float(b[3]) * sy)
                for b in raw_boxes
            ]
            kept = match_boxes_by_center(f_boxes, bboxes)
            if not kept:
                self._last_failed = True
            return kept if kept else list(bboxes)
        except Exception as e:
            self._last_failed = True
            from auto2dlabel.cli_common import console

            console.print(f"[yellow]⚠ 指代 L2 解析失败（保留全部框）: {e}[/yellow]")
            return list(bboxes)


def create_referential_resolver(
    model_name: str = "microsoft/Florence-2-base",
    **kwargs: Any,
) -> ReferentialResolver:
    """工厂函数：创建指代解析器（当前仅 Florence-2 一种实现）。"""
    return Florence2ReferentialResolver(model_name=model_name, **kwargs)
