"""姿态估计模型抽象层（YOLO-pose）。

封装 Ultralytics YOLO11/12/26-pose 系列（yolo11/12/26 n/s/m/l/x-pose.pt），提供统一接口：
detect_pose(image_path, prompts, confidence_threshold) -> list[PoseResult]。

keypoints 语义：COCO 17 点 [x, y, v]（像素坐标；v=0 未标注 / 1 标注可见 / 2 标注不可见），
挂 Bbox.keypoints 随 bbox 数据流穿透。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

from auto2dlabel.configs.model_catalog import POSE_MODELS, WEIGHTS_DIR

if TYPE_CHECKING:
    # 仅用于类型标注与 cast（字符串前向引用），运行时保持懒加载
    from ultralytics.engine.results import Results


@dataclass
class PoseResult:
    """单个人体姿态检测结果（像素坐标）。

    keypoints: COCO 17 点 [x, y, v] 列表，与 boxes 逐框对齐。
    """

    x: float
    y: float
    width: float
    height: float
    label: str
    confidence: float
    keypoints: list[tuple[float, float, float]]


class PoseModel(Protocol):
    """姿态模型接口（Protocol，允许 duck typing）。"""

    def detect_pose(
        self,
        image_path: str,
        prompts: list[str],
        confidence_threshold: float = 0.3,
    ) -> list[PoseResult]:
        ...


class UltralyticsPoseModel:
    """Ultralytics YOLO-pose 封装（_load 懒加载 + 批量 rect=False 铁律）。"""

    def __init__(
        self,
        model_name: str = "yolo11n-pose.pt",
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

    def detect_pose(
        self,
        image_path: str,
        prompts: list[str],
        confidence_threshold: float = 0.3,
    ) -> list[PoseResult]:
        model = self._load()

        # rect=False：统一单图/批量 letterbox（批量 parity 铁律，与检测/OBB 一致）
        preds = cast("list[Results]", model(
            image_path, conf=confidence_threshold, iou=self._iou,
            device=self._device, verbose=False, rect=False,
        ))

        return self._parse_pred(preds[0], prompts)

    def detect_pose_batch(
        self,
        image_paths: list[str],
        prompts: list[str],
        confidence_threshold: float = 0.3,
        num_workers: int = 0,
    ) -> list[list[PoseResult]]:
        """批量姿态检测：ultralytics 原生 batch 推理（model(paths, batch=, workers=)）。

        每图解析与 detect_pose 完全一致（同一 _parse_pred）。
        """
        model = self._load()

        kwargs: dict[str, Any] = {
            "conf": confidence_threshold, "iou": self._iou,
            "device": self._device, "verbose": False, "rect": False,
        }
        if num_workers:
            kwargs["workers"] = num_workers
        if len(image_paths) > 1:
            kwargs["batch"] = len(image_paths)

        preds = cast("list[Results]", model(image_paths, **kwargs))
        return [self._parse_pred(p, prompts) for p in preds]

    def _parse_pred(self, pred: Results, prompts: list[str]) -> list[PoseResult]:
        """单个 ultralytics Results → PoseResult 列表（单图/批量共用）。

        keypoints.data 与 boxes 逐框对齐（同一次推理输出，顺序一致）；
        无 keypoints 属性时（误加载普通检测权重）容错返回空列表。
        """
        from auto2dlabel.models.detection import _match_prompt

        model_names = cast(Any, self._model).names
        if not hasattr(pred, "keypoints") or pred.keypoints is None:
            return []

        # ultralytics stub 中 boxes/keypoints 为 Optional，按 Any 处理（与 obb._parse_pred 同款）
        boxes = cast(Any, pred.boxes)
        kpts_xy = cast(Any, pred.keypoints).xy
        kpts_conf = cast(Any, pred.keypoints).conf
        results: list[PoseResult] = []
        for i in range(len(boxes)):
            label = str(model_names[int(boxes.cls[i])])
            if not _match_prompt(label, prompts):
                continue
            x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[i])
            kpts: list[tuple[float, float, float]] = []
            for j in range(len(kpts_xy[i])):
                px = float(kpts_xy[i][j][0])
                py = float(kpts_xy[i][j][1])
                # v 取整：conf>0 → 标注（1/2 不区分），0 → 未标注
                v = 1 if float(kpts_conf[i][j]) > 0 else 0
                kpts.append((px, py, float(v)))
            results.append(PoseResult(
                x=x1, y=y1, width=x2 - x1, height=y2 - y1,
                label=label, confidence=float(boxes.conf[i]), keypoints=kpts,
            ))
        return results


def create_pose_model(
    model_name: str = "yolo11n-pose.pt",
    **kwargs: Any,
) -> PoseModel:
    """工厂函数：创建姿态估计模型。

    自动识别：ultralytics YOLO-pose（-pose.pt 或目录内名）/
    mmpose RTMPose（前缀 "rtmpose"，top-down 精度档）。
    """
    if model_name.lower().startswith("rtmpose"):
        # mmpose RTMPose 精度档（v0.6 Phase 3b）：config/权重经
        # download_mmpose_weights.sh 就位；类内 ImportError 守卫（零加载）
        from auto2dlabel.models.mmpose_engines import MMposeRTMPoseModel

        kwargs.setdefault(
            "model_name",
            MMposeRTMPoseModel.DEFAULT_NAME if model_name.lower() == "rtmpose" else model_name,
        )
        return MMposeRTMPoseModel(**kwargs)
    if not model_name.endswith("-pose.pt") and model_name not in POSE_MODELS:
        raise ValueError(
            f"无法识别的姿态模型: '{model_name}'。\n"
            f"可用: {', '.join(POSE_MODELS)}"
        )
    return UltralyticsPoseModel(model_name=model_name, **kwargs)
