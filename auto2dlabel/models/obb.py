"""旋转框检测模型抽象层（YOLO-OBB）。

封装 Ultralytics YOLO11/12/26-OBB 系列（yolo11/12/26 n/s/m/l/x-obb.pt），提供统一接口：
detect_obb(image_path, prompts, confidence_threshold) -> list[OBBResult]。

Oriented R-CNN 等 mmrotate 系模型延后（依赖重，见 milestone/v0.3.md 取舍注记）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

from auto2dlabel.models.model_catalog import DOTA_CLASSES, DOTA_PROMPT_ALIASES

if TYPE_CHECKING:
    # 仅用于类型标注与 cast（字符串前向引用），运行时保持懒加载
    from ultralytics.engine.results import Results


def _resolve_obb_label(cls_id: int, names: dict[int, str] | None = None) -> str:
    """DOTA 类别 id → 类名（空格命名，与 ultralytics dota8.yaml 一致）。

    names: 模型 checkpoint 的 names dict（优先，权威）；缺省回退 DOTA_CLASSES。
    """
    if names and cls_id in names:
        return str(names[cls_id])
    return DOTA_CLASSES[cls_id] if cls_id < len(DOTA_CLASSES) else str(cls_id)


def _match_obb_prompt(label: str, prompts: list[str]) -> bool:
    """检查 OBB 标签是否匹配用户 prompt（连字符↔空格归一 + DOTA 别名 + 双向子串）。

    例：label "small vehicle" 匹配 prompt "car"（别名）/"small-vehicle"（归一）/"vehicle"（子串）。
    """
    norm = label.replace("-", " ").lower()
    candidates = {norm, *DOTA_PROMPT_ALIASES.get(norm, ())}
    for p in prompts:
        p_norm = p.replace("-", " ").lower()
        if p_norm in norm or norm in p_norm or p_norm in candidates:
            return True
    return False


@dataclass
class OBBResult:
    """单个旋转框检测结果（像素坐标）。

    cx, cy: 中心点。
    width, height: 外接矩形宽高。
    angle: 旋转角度（弧度），width 轴相对 x 轴，(-π/2, π/2]，与 ultralytics xywhr 零转换。
    """

    cx: float
    cy: float
    width: float
    height: float
    angle: float
    label: str
    confidence: float


class OBBModel(Protocol):
    """旋转框检测模型接口（Protocol，允许 duck typing）。"""

    def detect_obb(
        self,
        image_path: str,
        prompts: list[str],
        confidence_threshold: float = 0.3,
    ) -> list[OBBResult]:
        ...


class UltralyticsOBBModel:
    """Ultralytics YOLO11-OBB 模型封装。

    Results.obb.xywhr 直接取 (cx, cy, w, h, angle)（弧度）。
    模型被当作普通检测运行时无 obb 属性 → 容错返回空列表。
    """

    def __init__(
        self,
        model_name: str = "yolo11n-obb.pt",
        device: str | None = None,
        iou_threshold: float = 0.5,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._iou = iou_threshold
        self._model: Any = None  # ultralytics stub 导出不稳定，类型按 Any 处理

    def _load(self) -> Any:
        """加载模型（幂等）。ultralytics stub 的 YOLO 导出不稳定，类型按 Any 处理。"""
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO, settings  # type: ignore
        except ImportError:
            raise ImportError("ultralytics 未安装，请运行: pip install ultralytics")

        from auto2dlabel.models.model_catalog import WEIGHTS_DIR

        # 权重目录和下载目录统一指向 auto2dlabel/weights/
        cast(Any, settings).update({
            "weights_dir": str(WEIGHTS_DIR),
            "datasets_dir": str(WEIGHTS_DIR / "datasets"),
        })

        # 优先从本地 weights/ 加载，没有则自动下载
        local_path = WEIGHTS_DIR / self._model_name
        if local_path.exists():
            self._model = YOLO(str(local_path))
        else:
            self._model = YOLO(self._model_name)
        return self._model

    def detect_obb(
        self,
        image_path: str,
        prompts: list[str],
        confidence_threshold: float = 0.3,
    ) -> list[OBBResult]:
        model = self._load()

        # YOLO.__call__ stub 为 Results | Tensor 联合，运行时恒为 list[Results]
        preds = cast("list[Results]", model(
            image_path, conf=confidence_threshold, iou=self._iou,
            device=self._device, verbose=False,
        ))

        return self._parse_pred(preds[0], prompts)

    def detect_obb_batch(
        self,
        image_paths: list[str],
        prompts: list[str],
        confidence_threshold: float = 0.3,
        num_workers: int = 0,
    ) -> list[list[OBBResult]]:
        """批量旋转框检测：ultralytics 原生 batch 推理（model(paths, batch=, workers=)）。

        每图解析与 detect_obb 完全一致（同一 _parse_pred）。
        """
        model = self._load()

        kwargs: dict[str, Any] = {
            "conf": confidence_threshold, "iou": self._iou,
            "device": self._device, "verbose": False,
        }
        if num_workers:
            kwargs["workers"] = num_workers
        if len(image_paths) > 1:
            kwargs["batch"] = len(image_paths)

        preds = cast("list[Results]", model(image_paths, **kwargs))
        return [self._parse_pred(p, prompts) for p in preds]

    def _parse_pred(self, pred: "Results", prompts: list[str]) -> list[OBBResult]:
        """单个 ultralytics Results → OBBResult 列表（单图/批量共用）。"""
        # checkpoint 的 names dict（权威类名表，dict[int, str]）
        model_names = cast(Any, self._model).names

        results = []
        obb = getattr(pred, "obb", None)  # 非 OBB 模型容错
        if obb is None:
            return results
        # stub 未覆盖 OBB 字段类型，统一按 Any 处理
        rows = cast(Any, obb).xywhr.tolist()
        confs = cast(Any, obb).conf.tolist()
        cls_ids = cast(Any, obb).cls.tolist()
        for (cx, cy, w, h, angle), conf, cls_id in zip(rows, confs, cls_ids):
            label = _resolve_obb_label(int(cls_id), model_names)
            if not _match_obb_prompt(label, prompts):
                continue  # 跳过不匹配的类别
            results.append(OBBResult(
                cx=float(cx), cy=float(cy),
                width=float(w), height=float(h),
                angle=float(angle),
                label=label, confidence=float(conf),
            ))
        return results


def create_obb_model(
    model_name: str = "yolo11n-obb.pt",
    **kwargs: Any,
) -> OBBModel:
    """工厂函数：创建旋转框检测模型。"""
    if not model_name.endswith(".pt"):
        from auto2dlabel.models.model_catalog import ULTRALYTICS_OBB_MODELS

        raise ValueError(
            f"无法识别的 OBB 模型: '{model_name}'。\n"
            f"可用: {', '.join(ULTRALYTICS_OBB_MODELS)}"
        )
    return UltralyticsOBBModel(model_name=model_name, **kwargs)
