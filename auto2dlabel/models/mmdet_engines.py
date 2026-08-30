"""mmdet 运行时模型封装（v0.6 Phase 3a：Mask2Former 分割质量档 + RTMDet 高召回检测档）。

统一走 mmdet init_detector + inference_detector 运行时；设计要点：

- 零加载 __init__：mmdet 导入守卫在 _load 内——未装 mmdet 的环境仍可经工厂
  实例化（chat/planner 链路安全），首次推理才报 ImportError 提示安装
- 幂等 _load：重复调用返回同一模型实例
- config + 权重落 auto2dlabel/weights/（configs 为 mmdetection v3.3.0
  sparse-checkout 嵌套仓库，不入库；缺文件 → FileNotFoundError 提示下载脚本）
- 类别 COCO 80 类（与 PyTorchVisionModel / MaskRCNNModel 同源 COCO_CLASSES）；
  检测按 _match_prompt 过滤（空 prompts 不过滤）；Mask2Former 与 MaskRCNNModel
  同款「检测 + 分割一步完成」模式（generate 忽略输入 bboxes）
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np

from auto2dlabel.configs.model_catalog import COCO_CLASSES, WEIGHTS_DIR
from auto2dlabel.models.detection import DetectionResult, _match_prompt
from auto2dlabel.models.segmentation import _mask_to_polygon
from auto2dlabel.schema.annotation import Bbox, Mask

# configs：mmdetection v3.3.0 sparse-checkout（与已装 mmdet 版本绑定——config
# 引用的组件必须存在于已装 mmdet；升级 mmdet 需同步重新 checkout 对应 tag）
MMDET_CONFIGS_DIR: Path = WEIGHTS_DIR / "mmdet_configs" / "configs"
MMDET_CKPT_DIR: Path = WEIGHTS_DIR / "mmdet_checkpoints"

# name → (config 相对路径, 权重相对路径)；URL 单一事实源 = download_mmdet_weights.sh
# （权重子目录 = 下载脚本 out_dir 布局：CKPT_DIR/<name>/<file>）
RTMDET_MODELS: dict[str, tuple[str, str]] = {
    "rtmdet_s": (
        "rtmdet/rtmdet_s_8xb32-300e_coco.py",
        "rtmdet_s/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth",
    ),
    "rtmdet_m": (
        "rtmdet/rtmdet_m_8xb32-300e_coco.py",
        "rtmdet_m/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth",
    ),
    "rtmdet_l": (
        "rtmdet/rtmdet_l_8xb32-300e_coco.py",
        "rtmdet_l/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth",
    ),
    "rtmdet_x": (
        "rtmdet/rtmdet_x_8xb32-300e_coco.py",
        "rtmdet_x/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth",
    ),
}
MASK2FORMER_MODELS: dict[str, tuple[str, str]] = {
    "mask2former_r50_8xb2-lsj-50e_coco": (
        "mask2former/mask2former_r50_8xb2-lsj-50e_coco.py",
        "mask2former_r50_8xb2-lsj-50e_coco/"
        "mask2former_r50_8xb2-lsj-50e_coco_20220506_191028-41b088b6.pth",
    ),
}


def _import_mmdet() -> tuple[Any, Any]:
    """mmdet 运行时导入（ImportError 守卫：返回 (init_detector, inference_detector)）。"""
    try:
        from mmdet.apis import inference_detector, init_detector
    except ImportError:
        raise ImportError(
            "mmdet 未安装（RTMDet / Mask2Former 需要）。请运行: pip install mmdet==3.3.0"
        ) from None
    return init_detector, inference_detector


def _allow_legacy_checkpoint_globals() -> None:
    """白名单放行老 checkpoint 的非 torch 全局（torch>=2.6 weights_only 兼容）。

    torch 2.6+ torch.load 默认 weights_only=True，拒绝 2021-2023 训练
    权重的非 torch 全局——实测两类：
    - numpy：RTMPose 20230127 权重报 numpy.core.multiarray._reconstruct
      （numpy.ndarray / numpy.dtype / numpy.dtypes dtype 类随 ndarray
      pickle 一并出现，全套放行）
    - mmengine：RTMDet 20220719 权重报 mmengine.logging.HistoryBuffer
      （mmengine 训练缓存对象，数据载体无代码执行；其 __reduce__ 引用
      内建 bytes 构造器 + getattr，一并放行）
    幂等可重复调用；torch<2.6 无 add_safe_globals 或组件缺失时静默返回
    （行为退化为修复前）。mmpose_engines 复用本函数（复用不复制）。
    """
    try:
        from torch.serialization import add_safe_globals
    except ImportError:
        return  # torch < 2.6：weights_only 默认关闭，无需白名单
    candidates: list[Any] = [np.ndarray, np.dtype, bytes, getattr]
    for mod_name in ("numpy.core.multiarray", "numpy._core.multiarray"):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        # _reconstruct（ndarray rebuild）+ scalar（RTMDet 权重实测的 numpy
        # scalar 类型）
        for attr in ("_reconstruct", "scalar"):
            fn = getattr(mod, attr, None)
            if fn is not None:
                candidates.append(fn)
    try:
        dtypes = importlib.import_module("numpy.dtypes")
    except ImportError:
        dtypes = None  # numpy < 1.25：无 dtype 类全局
    if dtypes is not None:
        for name in dir(dtypes):
            obj = getattr(dtypes, name)
            if isinstance(obj, type):
                candidates.append(obj)
    history_buffer: Any | None = None
    try:
        from mmengine.logging import HistoryBuffer

        history_buffer = HistoryBuffer
    except ImportError:
        pass  # 未装 mmengine：仅 numpy 白名单
    if history_buffer is not None:
        candidates.append(history_buffer)
    add_safe_globals(candidates)


def _mmdet_device(device: str) -> str:
    """get_device() 的 "cuda" → mmdet 惯例 "cuda:0"；其余原样透传。"""
    return "cuda:0" if device == "cuda" else device


def _resolve_paths(
    model_name: str, table: dict[str, tuple[str, str]],
) -> tuple[Path, Path]:
    """模型名 → (config 路径, 权重路径)；未知模型名抛 ValueError（列出可用名）。"""
    if model_name not in table:
        raise ValueError(
            f"未知的 mmdet 模型名: '{model_name}'。可用: {', '.join(sorted(table))}"
        )
    config_rel, ckpt_file = table[model_name]
    return MMDET_CONFIGS_DIR / config_rel, MMDET_CKPT_DIR / ckpt_file


def _require_files(
    config_path: Path, ckpt_path: Path,
    download_script: str = "download_mmdet_weights.sh",
) -> None:
    """config/权重任一缺失 → FileNotFoundError（提示下载脚本，防 init 深埋报错）。

    mmpose 复用本函数时传 download_script="download_mmpose_weights.sh"。
    """
    missing = [str(p) for p in (config_path, ckpt_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "缺少模型文件: " + ", ".join(missing)
            + f"。请运行: bash auto2dlabel/weights/{download_script}"
        )


def _pred_tensor(value: Any) -> list[Any]:
    """pred_instances 张量 → 普通 list（torch Tensor 与 numpy 数组通吃）。"""
    if value is None:
        return []
    if hasattr(value, "cpu"):
        value = value.cpu().numpy()
    return list(np.asarray(value).tolist())


def _parse_detections(
    pred_instances: Any,
    class_names: list[str],
    confidence_threshold: float,
    prompts: list[str],
) -> list[DetectionResult]:
    """mmdet pred_instances → DetectionResult 列表（纯函数，可单测）。

    bboxes (N,4) xyxy / scores (N,) / labels (N,) 0-based；置信度过滤 +
    _match_prompt 过滤（空 prompts 不过滤）+ 标签越界丢弃。
    """
    bboxes = _pred_tensor(getattr(pred_instances, "bboxes", None))
    scores = _pred_tensor(getattr(pred_instances, "scores", None))
    labels = _pred_tensor(getattr(pred_instances, "labels", None))

    results: list[DetectionResult] = []
    for box, score, cls_id in zip(bboxes, scores, labels):
        conf = float(score)
        if conf < confidence_threshold:
            continue
        idx = int(cls_id)
        if idx < 0 or idx >= len(class_names):
            continue
        label = class_names[idx]
        if not _match_prompt(label, prompts):
            continue
        x1, y1, x2, y2 = (float(v) for v in box[:4])
        results.append(DetectionResult(
            x=x1, y=y1, width=x2 - x1, height=y2 - y1,
            label=label, confidence=conf,
        ))
    return results


def _parse_masks(
    pred_instances: Any,
    class_names: list[str],
    confidence_threshold: float,
) -> list[Mask]:
    """mmdet pred_instances（含 masks）→ Mask 列表（纯函数，可单测）。

    masks (N,H,W) → bool >0.5 → _mask_to_polygon（COCO polygon）；
    bbox xyxy → Bbox；置信度过滤 + 标签越界丢弃
    （与 MaskRCNNModel._parse_output 同口径 0.5 阈值）。
    """
    masks = getattr(pred_instances, "masks", None)
    bboxes = _pred_tensor(getattr(pred_instances, "bboxes", None))
    scores = _pred_tensor(getattr(pred_instances, "scores", None))
    labels = _pred_tensor(getattr(pred_instances, "labels", None))
    if masks is None or (hasattr(masks, "__len__") and len(masks) == 0):
        return []
    masks_np = np.asarray(masks.cpu().numpy() if hasattr(masks, "cpu") else masks)

    results: list[Mask] = []
    for i, (box, score, cls_id) in enumerate(zip(bboxes, scores, labels)):
        conf = float(score)
        if conf < confidence_threshold:
            continue
        idx = int(cls_id)
        if idx < 0 or idx >= len(class_names):
            continue
        mask_bool = masks_np[i] > 0.5
        x1, y1, x2, y2 = (float(v) for v in box[:4])
        results.append(Mask(
            bbox=Bbox.from_xyxy(
                x1, y1, x2, y2, label=class_names[idx], confidence=conf,
            ),
            segmentation=_mask_to_polygon(mask_bool),
            area=float(mask_bool.sum()),
        ))
    return results


def _infer_single(inference_detector: Any, model: Any, image_path: str) -> Any:
    """inference_detector 单图调用 → 首个 DetDataSample（list 返回兼容）。"""
    result = inference_detector(model, image_path)
    if isinstance(result, (list, tuple)):
        result = result[0]
    return result


class MMDetRTMDetModel:
    """mmdet RTMDet 检测模型（高召回档，COCO 80 类）。

    零加载 __init__ + 幂等 _load；config/权重经
    `bash auto2dlabel/weights/download_mmdet_weights.sh` 下载后首次推理自动就位。
    """

    DEFAULT_NAME = "rtmdet_l"

    def __init__(
        self,
        model_name: str = DEFAULT_NAME,
        device: str | None = None,
        iou_threshold: float = 0.5,
    ):
        from auto2dlabel.tools.device import get_device

        self._model_name = model_name
        self._device = device or get_device()
        self._iou = iou_threshold  # 工厂 parity 参数：mmdet 测试管线自管 NMS，推理期未用
        self._model = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        init_detector, _ = _import_mmdet()
        config_path, ckpt_path = _resolve_paths(self._model_name, RTMDET_MODELS)
        _require_files(config_path, ckpt_path)
        # torch 2.6+ weights_only 拒 2022 老权重 → 先放行非 torch 全局（见函数 docstring）
        _allow_legacy_checkpoint_globals()
        self._model = init_detector(
            str(config_path), str(ckpt_path), device=_mmdet_device(self._device)
        )
        return self._model

    def detect(
        self,
        image_path: str,
        prompts: list[str],
        confidence_threshold: float = 0.3,
    ) -> list[DetectionResult]:
        """检测单图 → DetectionResult 列表（COCO 80 类，按 prompt 过滤）。"""
        model = self._load()
        _, inference_detector = _import_mmdet()
        result = _infer_single(inference_detector, model, image_path)
        return _parse_detections(
            result.pred_instances, COCO_CLASSES, confidence_threshold, prompts,
        )


class MMDetMask2FormerModel:
    """mmdet Mask2Former 实例分割模型（分割质量档，COCO 80 类）。

    与 MaskRCNNModel 同款「检测 + 分割一步完成」模式：generate 忽略输入
    bboxes，模型自己检测（coco_seg_benchmark 自带检测模型兼容先例）。
    """

    DEFAULT_NAME = "mask2former_r50_8xb2-lsj-50e_coco"

    def __init__(
        self,
        model_name: str = DEFAULT_NAME,
        device: str | None = None,
        confidence_threshold: float = 0.5,
    ):
        from auto2dlabel.tools.device import get_device

        self._model_name = model_name
        self._device = device or get_device()
        self._confidence_threshold = confidence_threshold  # 与 MaskRCNNModel 0.5 同口径
        self._model = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        init_detector, _ = _import_mmdet()
        config_path, ckpt_path = _resolve_paths(self._model_name, MASK2FORMER_MODELS)
        _require_files(config_path, ckpt_path)
        # torch 2.6+ weights_only 拒 2022 老权重 → 先放行非 torch 全局（见函数 docstring）
        _allow_legacy_checkpoint_globals()
        self._model = init_detector(
            str(config_path), str(ckpt_path), device=_mmdet_device(self._device)
        )
        return self._model

    def generate(
        self,
        image_path: str,
        bboxes: list[Bbox],
        mode: str = "box",
    ) -> list[Mask]:
        """检测 + 分割一步完成（忽略输入的 bboxes，模型自己检测）。"""
        model = self._load()
        _, inference_detector = _import_mmdet()
        result = _infer_single(inference_detector, model, image_path)
        return _parse_masks(
            result.pred_instances, COCO_CLASSES, self._confidence_threshold,
        )
