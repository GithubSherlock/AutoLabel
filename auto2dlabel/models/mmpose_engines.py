"""mmpose 运行时模型封装（v0.6 Phase 3b：RTMPose 姿态精度档）。

RTMPose 是 top-down 姿态模型：先 person 检测（复用本项目检测引擎，单一
事实源 DEFAULT_MODEL）→ mmpose inference_topdown 逐框单人姿态。与
mmdet_engines 同款设计：

- 零加载 __init__ + 幂等 _load + ImportError 守卫（未装 mmpose 仍可实例化）
- config + 权重落 auto2dlabel/weights/（mmpose v1.3.2 sparse-checkout）
- keypoints 语义 COCO 17 点 [x, y, v]（v=1 有分 / 0 无分，
  与 UltralyticsPoseModel._parse_pred 同口径）
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from auto2dlabel.configs.model_catalog import WEIGHTS_DIR
from auto2dlabel.models.mmdet_engines import _allow_legacy_checkpoint_globals, _require_files
from auto2dlabel.models.pose import PoseResult

# configs：mmpose v1.3.2 sparse-checkout（与已装 mmpose 版本绑定——升级
# mmpose 需同步重新 checkout 对应 tag）
MMPOSE_CONFIGS_DIR: Path = WEIGHTS_DIR / "mmpose_configs" / "configs"
MMPOSE_CKPT_DIR: Path = WEIGHTS_DIR / "mmpose_checkpoints"

# name → (config 相对路径, 权重相对路径)；URL 单一事实源 = download_mmpose_weights.sh
RTMPOSE_MODELS: dict[str, tuple[str, str]] = {
    "rtmpose_l": (
        "body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_coco-256x192.py",
        "rtmpose_l/"
        "rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.pth",
    ),
}

NUM_KEYPOINTS = 17  # COCO 17 点（与 PoseResult.keypoints 语义一致）


def _import_mmpose() -> tuple[Any, Any]:
    """mmpose 运行时导入（ImportError 守卫：返回 (init_model, inference_topdown)）。"""
    try:
        from mmpose.apis import inference_topdown, init_model
    except ImportError:
        raise ImportError(
            "mmpose 未安装（RTMPose 需要）。请运行: pip install mmpose==1.3.2"
        ) from None
    return init_model, inference_topdown


def _resolve_paths(
    model_name: str, table: dict[str, tuple[str, str]],
) -> tuple[Path, Path]:
    """模型名 → (config 路径, 权重路径)；未知模型名抛 ValueError（列出可用名）。"""
    if model_name not in table:
        raise ValueError(
            f"未知的 mmpose 模型名: '{model_name}'。可用: {', '.join(sorted(table))}"
        )
    config_rel, ckpt_rel = table[model_name]
    return MMPOSE_CONFIGS_DIR / config_rel, MMPOSE_CKPT_DIR / ckpt_rel


def _parse_topdown(
    pred_instances: Any,
    bbox: tuple[float, float, float, float],
    confidence: float,
    num_keypoints: int = NUM_KEYPOINTS,
) -> PoseResult:
    """mmpose PoseDataSample.pred_instances → PoseResult（纯函数，可单测）。

    真实 mmpose 输出 keypoints (N,K,2) + keypoint_scores (N,K) 多人张量
    （inference_topdown 逐框调用时 N=1，取首实例）；v=1 if score>0 else 0。
    K 与 17 不一致时截断/补齐 (0,0,0)（宁多勿漏，防越界崩溃）。
    """
    kpts = getattr(pred_instances, "keypoints", None)
    if kpts is None:
        kpts = np.zeros((0, 2), dtype=float)
    kpts_xy = np.asarray(kpts)
    if kpts_xy.ndim == 3:  # (N,K,2) → 逐框推理 N=1，取首实例
        kpts_xy = kpts_xy[0]
    scores = getattr(pred_instances, "keypoint_scores", None)
    if scores is None:
        scores = np.zeros((len(kpts_xy),), dtype=float)
    scores = np.asarray(scores)
    if scores.ndim == 2:  # (N,K) → 取首实例
        scores = scores[0]

    keypoints: list[tuple[float, float, float]] = []
    for j in range(num_keypoints):
        if j < len(kpts_xy):
            keypoints.append((
                float(kpts_xy[j][0]), float(kpts_xy[j][1]),
                1.0 if float(scores[j]) > 0 else 0.0,
            ))
        else:
            keypoints.append((0.0, 0.0, 0.0))

    x1, y1, x2, y2 = bbox
    return PoseResult(
        x=x1, y=y1, width=x2 - x1, height=y2 - y1,
        label="person", confidence=confidence, keypoints=keypoints,
    )


class MMposeRTMPoseModel:
    """mmpose RTMPose top-down 姿态模型（精度档）。

    两段式：person 检测（复用 create_detection_model + DEFAULT_MODEL，检测框
    即 PoseResult.bbox）→ inference_topdown 逐框单人姿态。零加载 __init__ +
    幂等 _load；config/权重经
    `bash auto2dlabel/weights/download_mmpose_weights.sh` 下载。
    """

    DEFAULT_NAME = "rtmpose_l"

    def __init__(
        self,
        model_name: str = DEFAULT_NAME,
        device: str | None = None,
        iou_threshold: float = 0.5,
    ):
        from auto2dlabel.tools.device import get_device

        self._model_name = model_name
        self._device = device or get_device()
        self._iou = iou_threshold  # 透传 person 检测器（检测框质量影响姿态）
        self._model = None
        self._det_model: Any = None  # person 检测器（懒加载）

    def _load_detector(self) -> Any:
        """person 检测器（懒加载 + 幂等）：复用检测工厂 + 默认模型。"""
        if self._det_model is not None:
            return self._det_model
        from auto2dlabel.models.detection import create_detection_model
        from auto2dlabel.schema.task_plan import DEFAULT_MODEL

        self._det_model = create_detection_model(DEFAULT_MODEL, iou_threshold=self._iou)
        return self._det_model

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        init_model, _ = _import_mmpose()
        config_path, ckpt_path = _resolve_paths(self._model_name, RTMPOSE_MODELS)
        _require_files(config_path, ckpt_path, download_script="download_mmpose_weights.sh")
        device = "cuda:0" if self._device == "cuda" else self._device
        # torch 2.6+ weights_only 拒 2023 老权重 → 先放行非 torch 全局
        # （复用 mmdet_engines._allow_legacy_checkpoint_globals，见其 docstring）
        _allow_legacy_checkpoint_globals()
        self._model = init_model(str(config_path), str(ckpt_path), device=device)
        return self._model

    def detect_pose(
        self,
        image_path: str,
        prompts: list[str],
        confidence_threshold: float = 0.3,
    ) -> list[PoseResult]:
        """两段式姿态：person 检测 → RTMPose 逐框单人姿态（只出 person）。"""
        model = self._load()
        _, inference_topdown = _import_mmpose()

        dets = self._load_detector().detect(
            image_path, prompts or ["person"], confidence_threshold,
        )
        # RTMPose 只支持 person；检测器输出非 person 类时过滤（prompts 未含
        # person 时宁多勿漏地保留 person 类）
        dets = [d for d in dets if d.label.lower() == "person"]
        if not dets:
            return []

        boxes = [[d.x, d.y, d.x + d.width, d.y + d.height] for d in dets]
        samples = inference_topdown(model, image_path, bboxes=boxes, bbox_format="xyxy")

        return [
            _parse_topdown(
                sample.pred_instances,
                (d.x, d.y, d.x + d.width, d.y + d.height),
                d.confidence,
            )
            for d, sample in zip(dets, samples)
        ]
