"""LiDAR 3D 检测器封装（v0.2 P1）——mmdet3d CenterPoint/PointPillars 直检引擎。

母版 = auto2dlabel/models/lane.py 四段式：Protocol（测试注入 Fake）→ 具体类
（零加载 __init__ + 幂等 _load + ImportError 守卫）→ create_xxx 工厂（名字路由）。
红线：mmdet3d/mmcv 全部 import 在 _load 守卫内——mmdet3d 未装状态下
import 本模块、走工厂、单测均全绿。

推理链（手写最小链，不跑 DataLoader）：
    Config.fromfile → MODELS.build → load_checkpoint → Points(velodyne) →
    model.data_preprocessor（Det3DDataPreprocessor，内含 voxelize CUDA op）→
    model(mode='predict') → pred_instances_3d（bboxes_3d/scores_3d/labels_3d）。

输出语义（mmdet3d KITTI 模型）：相机系 [x,y,z,l,h,w,ry]，x/y/z = 底面中心
（y 向下）——与 label_2 语义同构，直接复用 Box3D.from_gt_row 转换
（cy = y - h/2；yaw_bev = rotation_y_to_yaw(ry)，唯一转换点纪律不变）。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from auto3dlabel.configs.kitti import (
    DEFAULT_CONF,
    KITTI_IMG_H,
    KITTI_IMG_W,
    MMDET3D_CONFIG_DIR,
    WEIGHTS_DIR,
)
from auto3dlabel.configs.model_catalog import DETECTOR3D_NAMES
from auto3dlabel.schema.box3d import Box3D, KittiFrame

# mmdet3d KITTI 3-class config 默认类序（config 多源提取失败时的 fallback）
MMDET3D_KITTI_CLASSES = ("Pedestrian", "Cyclist", "Car")


@dataclass
class Det3DResult:
    """LiDAR 直检单目标输出（相机系 7 值 = mmdet3d KITTI 输出语义）。"""

    label: str  # KITTI 类名（Car/Pedestrian/Cyclist）
    confidence: float
    bbox: list[float]  # [x, y, z, l, h, w, ry] 底面中心（相机系，y 向下）
    x1: float = 0.0  # 2D 投影外接框（导出字段 4-7，8 角点投影极值）
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0

    def to_box3d(self) -> Box3D:
        """→ Box3D（cy = y - h/2 底面→体积中心；yaw_bev = rotation_y_to_yaw(ry)）。"""
        x, y, z, l, h, w, ry = (float(v) for v in self.bbox)
        box = Box3D.from_gt_row(self.label, h, w, l, x, y, z, ry)
        box.confidence = self.confidence
        box.x1, box.y1, box.x2, box.y2 = self.x1, self.y1, self.x2, self.y2
        return box


def suggest_batch_size(
    per_frame_bytes: int, free_bytes: int, safety_factor: float = 0.85, max_batch: int = 32
) -> int:
    """批大小 = 空闲显存 × 安全系数 ÷ 单帧峰值（钳 [1, max_batch]）。

    cli run --batch LiDAR 引擎自动实测用：目标「跑满 GPU」但留安全余量
    （默认 0.85 与 2D resolve_batch_params 同源；实测路径传 0.92 + OOM 减半兜底）；
    max_batch 为估算失真护栏（实测偏差由运行时 OOM 减半吸收）；
    输入非正 → 1（保守逐帧）。
    """
    if per_frame_bytes <= 0 or free_bytes <= 0:
        return 1
    batch = int(free_bytes * safety_factor // per_frame_bytes)
    return max(1, min(max_batch, batch))


class Detector3D(Protocol):
    """3D 检测器协议 —— 测试注入 FakeDetector3D（零真实权重铁律）。"""

    @property
    def class_names(self) -> tuple[str, ...]:
        """类名元组（懒加载后为 config/dataset_meta 真实类序）。"""
        ...

    def detect(
        self, frame: KittiFrame, conf_threshold: float | None = None
    ) -> list[Det3DResult]:
        """KITTI 单帧点云 → 3D 检测列表（conf_threshold 覆盖构造默认）。"""
        ...

    def detect_batch(
        self, frames: list[KittiFrame], conf_threshold: float | None = None
    ) -> list[list[Det3DResult]]:
        """多帧点云一次 forward → 每帧检测列表（批处理：显存随帧数线性增长）。"""
        ...


def _patch_pretrained_init(cfg: Any) -> None:
    """删 backbone init_cfg Pretrained（阻断 open-mmlab:// 额外权重下载，v0.3 P6b）。

    全量 checkpoint 已含 backbone 权重，init_cfg Pretrained 只会触发冗余下载
    （网络受限环境 open-mmlab:// 可能不可达）——删段跳过预训练初始化。
    FreeAnchor regnet-400mf 实测需要（BEVFusion img_backbone 同款 patch 先例）。
    """
    model = cfg.get("model", {})
    for key in ("pts_backbone", "img_backbone", "backbone"):
        init_cfg = (model.get(key) or {}).get("init_cfg")
        if isinstance(init_cfg, dict) and init_cfg.get("type") == "Pretrained":
            del model[key]["init_cfg"]


def _init_model_trusted(config: Any, checkpoint_path: str, device: str) -> Any:
    """init_model + 可信来源权重加载（torch 2.6+ weights_only 兼容，v0.3 P2）。

    config: config 文件路径（str/Path）或 mmengine Config 对象——v0.3 P3 放宽：
    BEVFusion Swin init_cfg 需运行时 patch（checkpoint=None）后传对象，
    init_model 官方签名即接受 Union[str, Path, Config]。
    torch 2.6 起 torch.load 默认 weights_only=True：2022 年 zoo checkpoint
    （pv_rcnn 含 numpy scalar/dtype 与 mmengine HistoryBuffer 对象）会被拒。
    权重来自 download_detector3d.sh 钉死的 openmmlab 官方直链（可信来源），
    按 torch 官方指引以 weights_only=False 加载；patch 作用域仅限本次
    init_model 调用（finally 恢复，进程内其他加载路径零影响）。
    """
    import torch
    from mmdet3d.apis import init_model  # pyright: ignore[reportMissingImports]

    orig_load = torch.load

    def _trusted_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return orig_load(*args, **kwargs)

    setattr(torch, "load", _trusted_load)
    try:
        return init_model(config=config, checkpoint=checkpoint_path, device=device)
    finally:
        setattr(torch, "load", orig_load)


class Mmdet3dDetector:
    """mmdet3d 检测器（懒加载：构造零加载，detect 时 _load 幂等 + ImportError 守卫）。"""

    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path,
        conf_threshold: float = DEFAULT_CONF,
        device: str | None = None,
    ) -> None:
        self._config_path = str(config_path)
        self._checkpoint_path = str(checkpoint_path)
        self.conf_threshold = conf_threshold
        self._device = device  # None = cuda 可用则 cuda，否则 cpu
        self._model: Any = None
        self._class_names: tuple[str, ...] = MMDET3D_KITTI_CLASSES

    def _load(self) -> Any:
        """懒加载 mmdet3d 模型（幂等；mmdet3d 未装抛 ImportError，守卫路径）。

        官方 init_model（含 register_all_modules / SyncBN 转换 / dataset_meta）。
        """
        if self._model is not None:
            return self._model
        try:
            import mmdet3d  # noqa: F401  # 守卫探测：未装时抛 ImportError（helper 内部再 import apis）
            import torch
            from mmengine.config import Config
        except ImportError as e:
            raise ImportError(
                "mmdet3d/mmcv 未安装，无法使用 LiDAR 3D 检测器"
                "（安装步骤见 milestone/v0.2.md M1）"
            ) from e
        device = self._device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        # 确定性红线（批量 parity，同 2D 铁律）：cudnn 按 shape 启发式选卷积算法，
        # 不同批大小 → 不同 shape → 可能不同 kernel → 末位浮点漂移。
        # 2026-08-28 实测（400 帧 pointpillars_kitti，RTX 4090）：batch≤16 逐位一致；
        # batch=32（跑满显存档）5/400 帧 x 差 1cm（deterministic 只限候选集，
        # 无法跨 shape 强绑 kernel）——远低于人工标注方差（±10cm），接受并记录。
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # 深拷贝 Config 一次：阻断 backbone init_cfg Pretrained 下载（P6b
        # FreeAnchor regnet），再把对象传给 _init_model_trusted（官方签名
        # 接受 str|Path|Config，BEVFusion 同款路径）与 _extract_class_names
        cfg = copy.deepcopy(Config.fromfile(self._config_path))
        _patch_pretrained_init(cfg)
        model = _init_model_trusted(cfg, self._checkpoint_path, device)
        self._class_names = _extract_class_names(cfg, model)
        self._model = model
        return self._model

    @property
    def class_names(self) -> tuple[str, ...]:
        """类名元组（触发懒加载——加载后为 config/dataset_meta 真实类序）。

        构造后默认 KITTI 3 类；只有 _load 后才反映模型真实类序，
        故读取即触发加载（未装 mmdet3d 时抛 ImportError，同 detect）。
        """
        self._load()
        return self._class_names

    def detect_points(
        self, pts: np.ndarray, conf_threshold: float | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """LiDAR 点云 → (boxes, scores, labels)——模型原始输出系（不做坐标系转换）。

        KITTI 模型：LiDAR 系（x 前/y 左/z 上，z=底面中心，origin=(0.5,0.5,0)）；
        nuScenes 模型：自车系（同向，z=体积中心，origin=(0.5,0.5,0.5)）。
        调用方按模型语义负责 origin 与坐标系转换（KITTI detect / nuScenes 冒烟各持一端）。
        """
        from mmdet3d.apis import inference_detector  # pyright: ignore[reportMissingImports]

        model = self._load()
        # 官方推理链（LoadPointsFromDict pipeline）；1.4.0 实际返回 (results[0], data[0])
        # 类型标注误写为 Det3DDataSample（不可迭代）→ cast 对齐真实返回
        out: Any = cast(
            Any, inference_detector(model, np.asarray(pts, dtype=np.float32))
        )[0]
        pred: Any = out.pred_instances_3d
        boxes = pred.bboxes_3d.tensor.detach().cpu().numpy()
        scores = pred.scores_3d.detach().cpu().numpy()
        labels = pred.labels_3d.detach().cpu().numpy()
        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold
        mask = scores >= threshold
        return boxes[mask], scores[mask], labels[mask]

    def detect_points_batch(
        self, pts_list: list[np.ndarray], conf_threshold: float | None = None
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """多帧点云一次 forward → 每帧 (boxes, scores, labels)（原始输出系，同 detect_points）。

        inference_detector 官方 API 原生支持 Sequence[ndarray]（is_batch=True）：
        伪 collate 后单次 model.test_step 整批推理——激活显存随帧数线性增长
        （batch 大小 = 显存换吞吐，cli run --batch 用它跑满 GPU）。
        """
        from mmdet3d.apis import inference_detector  # pyright: ignore[reportMissingImports]

        model = self._load()
        # 批量返回 (results_list, data_list)——类型标注误写为单样本 → cast 对齐真实返回
        outs: Any = cast(
            Any, inference_detector(model, [np.asarray(p, dtype=np.float32) for p in pts_list])
        )[0]
        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold
        results: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for out in outs:
            pred: Any = out.pred_instances_3d
            boxes = pred.bboxes_3d.tensor.detach().cpu().numpy()
            scores = pred.scores_3d.detach().cpu().numpy()
            labels = pred.labels_3d.detach().cpu().numpy()
            mask = scores >= threshold
            results.append((boxes[mask], scores[mask], labels[mask]))
        return results

    def detect(
        self, frame: KittiFrame, conf_threshold: float | None = None
    ) -> list[Det3DResult]:
        """单帧点云 → 检测列表（≥阈值）；含 8 角点投影 2D 外接框。"""
        boxes, scores, labels = self.detect_points(frame.load_points(), conf_threshold)
        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold
        return self._det3d_results(frame, boxes, scores, labels, threshold)

    def detect_batch(
        self, frames: list[KittiFrame], conf_threshold: float | None = None
    ) -> list[list[Det3DResult]]:
        """多帧点云一次 forward → 每帧检测列表（语义同 detect：相机系 + 2D 投影框）。"""
        per_frame = self.detect_points_batch(
            [f.load_points() for f in frames], conf_threshold
        )
        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold
        return [
            self._det3d_results(frame, b, s, l, threshold)
            for frame, (b, s, l) in zip(frames, per_frame, strict=True)
        ]

    def _det3d_results(
        self,
        frame: KittiFrame,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        threshold: float,
    ) -> list[Det3DResult]:
        """单帧原始输出（LiDAR 系 7 值）→ Det3DResult 列表（相机系 + 2D 投影框）。

        detect / detect_batch 共用（批处理不改转换语义——parity 铁律同 2D 批量）：
        唯一转换点 = 官方 convert_to(CAM) + calib 4x4（R0_rect @ Tr_velo_to_cam）。
        """
        # mmdet3d KITTI 模型输出 LiDAR 系（x 前/y 左/z 上，底面中心）→ 相机系
        from mmdet3d.structures import (  # pyright: ignore[reportMissingImports]
            Box3DMode,
            LiDARInstance3DBoxes,
        )

        calib = frame.load_calib()
        # KITTI LiDAR 系 z = 底面中心（mmdet3d 默认 origin=(0.5,0.5,0)）——传 (0.5,0.5,0.5)
        # 会被 convert_to 误当「中心」再做一次底面换算，y_cam 凭空 +h/2（3D AP 全灭）
        lidar_boxes = LiDARInstance3DBoxes(boxes, box_dim=7, origin=(0.5, 0.5, 0.0))
        cam_boxes = lidar_boxes.convert_to(
            Box3DMode.CAM, rt_mat=calib.velo_to_cam_matrix(), correct_yaw=True
        )
        boxes = cam_boxes.tensor.detach().cpu().numpy()

        results: list[Det3DResult] = []
        for i in range(len(boxes)):
            if float(scores[i]) < threshold:
                continue
            label = self._class_names[int(labels[i])]
            x, y, z, l, h, w, ry = (float(v) for v in boxes[i])
            box = Box3D.from_gt_row(label, h, w, l, x, y, z, ry)
            u, v, valid = calib.project_cam_to_image(
                box.corners_cam(), KITTI_IMG_W, KITTI_IMG_H
            )
            if valid.any():
                x1, y1 = float(u[valid].min()), float(v[valid].min())
                x2, y2 = float(u[valid].max()), float(v[valid].max())
            else:
                x1 = y1 = x2 = y2 = 0.0  # 目标在视野外：2D 框哨兵（导出字段不参与评测）
            results.append(
                Det3DResult(
                    label=label,
                    confidence=float(scores[i]),
                    bbox=[x, y, z, l, h, w, ry],
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )
        return results


def _extract_class_names(cfg: Any, model: Any | None = None) -> tuple[str, ...]:
    """→ 类名元组（多源提取：model.dataset_meta > config metainfo > class_names）。

    全失败默认 3 类序（KITTI：Pedestrian/Cyclist/Car）。
    """
    meta = getattr(model, "dataset_meta", None)
    classes: Any = None
    if isinstance(meta, dict):
        classes = meta.get("classes")
    if not classes:
        metainfo = cfg.get("metainfo")
        if isinstance(metainfo, dict):
            classes = metainfo.get("classes")
    if not classes:
        classes = cfg.get("class_names")
    if classes:
        return tuple(str(c) for c in classes)
    return MMDET3D_KITTI_CLASSES


def create_detector3d(model_name: str | None) -> Detector3D | None:
    """工厂：DETECTOR3D_NAMES 路由 → Mmdet3dDetector；None/未知名 → None（2D 引擎）。

    判定单一事实源：调用方以返回值非 None 判「3D 引擎」（cli/tools3d 共用）。
    构造零加载——config/权重文件存在性延迟到 _load（未装/缺权重时抛错给调用方兜底）。
    """
    if not model_name or model_name not in DETECTOR3D_NAMES:
        return None
    entry = DETECTOR3D_NAMES[model_name]
    config_path = MMDET3D_CONFIG_DIR / entry["config"]
    checkpoint_path = WEIGHTS_DIR / entry["weights_dir"] / entry["checkpoint"]
    return Mmdet3dDetector(config_path=config_path, checkpoint_path=checkpoint_path)


def create_detector3d_any(model_name: str | None) -> Any | None:
    """三引擎统一工厂（v0.4 P2）：LiDAR（detect_points）/ 融合（bevfusion）/
    单目（fcos3d）——按名字逐工厂路由，未知名 → None。

    配合 tools/nuscenes_pipeline.predict_sample_nus 的三协议分派；KITTI 侧调用方
    继续用 create_detector3d（语义不变：非 None = LiDAR 引擎）。
    """
    det = create_detector3d(model_name)
    if det is not None:
        return det
    from auto3dlabel.models.bevfusion3d import create_bevfusion_detector

    fusion_det = create_bevfusion_detector(model_name)
    if fusion_det is not None:
        return fusion_det
    from auto3dlabel.models.mono3d import create_fcos3d_detector

    return create_fcos3d_detector(model_name)
