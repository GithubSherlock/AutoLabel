"""BEVFusion 检测器封装（v0.3 P3）——mmdet3d projects/BEVFusion（nuScenes 融合引擎）。

定位：nuScenes 简化评测出表（vs pointpillars_nus 17.7 / centerpoint_nus 23.4
同口径，test-v0.3.md P2 实测）。母版 = models/detection3d.py 四段式：零加载
__init__ + 幂等 _load + ImportError 守卫 + create_xxx 工厂（名字路由）。
红线：mmdet3d/mmcv 全部 import 在 _load 守卫内——mmdet3d 未装状态下
import 本模块、走工厂、单测均全绿。

推理链（手写最小链，绕开 ann_file pkl）：
    sys.path += MMDET3D_CONFIG_DIR（projects.BEVFusion 可解析，custom_imports 依赖）
    → Config.fromfile 深拷贝 + 删 img_backbone.init_cfg（Swin 预训练 URL 无网络）
    → _init_model_trusted(Config 对象)（官方 init_model 内部处理 custom_imports）
    → data_ dict → Compose(test_pipeline) → pseudo_collate → model.test_step。

data_ 契约（官方 inference_multi_modality_detector 骨架）：
    {images: {CAM_x: {img_path, cam2img 3x3, lidar2cam 4x4}},
     lidar_points: {lidar_path}, timestamp（秒，pkl infos 约定 /1e6）,
     box_type_3d, box_mode_3d}
box_type_3d 必须传 get_box_type('LiDAR') 返回的**类**——BEVFusion.predict 不做
str→类转换（实码确认，与 mono3d 同坑；Base3DDetector.predict 的转换不覆盖此路径）。

输出语义：pred_instances_3d.bboxes_3d.tensor（LiDAR 传感器系）=
[x,y,z,w,l,h,yaw,vx,vy]——TransFusionBBoxCoder.decode 拼接
cat([center,height,dim,rot,vel])，dim head 训练目标 = gt_bboxes_3d.dims =
[w,l,h]（LiDARInstance3DBoxes 张量序，实码锁定）——与 DeltaXYZWLHRBBoxCoder 系
pointpillars/centerpoint 的 idx3=l **相反**。detect_sample 归一化到统一契约
[x,y,z,l,w,h,yaw,vx,vy]（idx3=l），供 boxes_sensor_to_global 原样复用。
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np

from auto3dlabel.configs.kitti import DEFAULT_CONF, MMDET3D_CONFIG_DIR, WEIGHTS_DIR
from auto3dlabel.configs.nuscenes import BEVFUSION_NAMES, NUSCENES_CAMERAS
from auto3dlabel.data.nuscenes import lidar_sample_data, rot_matrix
from auto3dlabel.models.detection3d import _extract_class_names, _init_model_trusted


def _pose_matrix(calib: dict) -> np.ndarray:
    """calibrated_sensor 记录（translation + rotation 四元数）→ 4x4 位姿（传感器 → ego）。"""
    m = np.eye(4)
    m[:3, :3] = rot_matrix(tuple(calib["rotation"]))
    m[:3, 3] = np.asarray(calib["translation"], dtype=np.float64)
    return m


def build_bevfusion_data(
    nusc: Any,
    sample: dict,
    dataroot: Path,
    box_type_3d: Any = None,
    box_mode_3d: Any = None,
    cameras: tuple[str, ...] = NUSCENES_CAMERAS,
) -> dict[str, Any]:
    """nuScenes sample → BEVFusion data_ dict（官方骨架，绕开 ann_file pkl）。

    标定链（红线）：lidar2cam = inv(cam_pose) @ lidar_pose，cam_pose/lidar_pose
    为 calibrated_sensor 的传感器→ego 位姿；cam2img = camera_intrinsic 3x3。
    纯函数零 mmdet3d 依赖：box_type_3d/box_mode_3d 由调用方 _load 提供
    （get_box_type 返回的类/枚举；测试可传 None）。
    """
    lidar_data = lidar_sample_data(sample, nusc)
    lidar_pose = _pose_matrix(nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"]))
    images: dict[str, dict[str, Any]] = {}
    for cam in cameras:
        sd = nusc.get("sample_data", sample["data"][cam])
        calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        cam_pose = _pose_matrix(calib)
        lidar2cam = np.linalg.inv(cam_pose) @ lidar_pose
        images[cam] = {
            "img_path": str(dataroot / sd["filename"]),
            "cam2img": np.asarray(calib["camera_intrinsic"], dtype=np.float64),
            "lidar2cam": lidar2cam,
        }
    return {
        "images": images,
        "lidar_points": {"lidar_path": str(dataroot / lidar_data["filename"])},
        "timestamp": float(lidar_data["timestamp"]) / 1e6,  # 微秒 → 秒（pkl infos 约定）
        "box_type_3d": box_type_3d,
        "box_mode_3d": box_mode_3d,
    }


def _load_checkpoint_state(path: str) -> dict[str, Any]:
    """torch checkpoint → state_dict。

    weights_only=False：官方权重含 mmengine HistoryBuffer 全局（2.6+ 默认拒载）。
    """
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    return state if isinstance(state, dict) else {}


def _transpose_sparse_conv_weights(
    state: dict[str, Any], prefix: str = "pts_middle_encoder"
) -> dict[str, Any]:
    """mmcv 1.x 布局 (out,kx,ky,kz,in) → 本地 mmcv 2.x 稀疏卷积布局 (kx,ky,kz,in,out)。

    实码锁定（2026-08-29 冒烟实证）：BEVFusion v1.1.0_models 官方 checkpoint 的
    SECOND 稀疏编码器权重为 mmcv 1.x 布局，本地 mmcv 2.x 编译的 SubMConv3d 期望
    spconv 原生布局 → load_checkpoint 形状不匹配**整批跳过** → 编码器随机初始化 →
    单帧 200 框分数全 <0.2（conf 0.3 零框）。补载换序后 conf 0.3 单帧 22 框、
    max score 0.814。仅 5D 权重换序（bias 1D 不受影响，其余键原样不动）。
    """
    return {
        k: v.permute(1, 2, 3, 4, 0)
        for k, v in state.items()
        if getattr(v, "ndim", None) == 5 and k.startswith(prefix)
    }


def _normalize_bevfusion_boxes(boxes: np.ndarray) -> np.ndarray:
    """TransFusionBBoxCoder 输出 → 统一契约 [x,y,z,l,w,h,yaw,vx,vy]（恒等直通）。

    实证锁定（2026-08-29 单帧 + 配对诊断）：decode 输出已是 [x,y,z,l,w,h,yaw,vx,vy]
    （car 输出 idx3=4.47≈GT l 4.7、idx4=1.85≈GT w 2.0；提交 JSON 换 L-W 后 car BEV
    IoU 0.254→0.795，yaw 与中心零误差）——官方 checkpoint 训练时代（mmdet3d 1.x）
    dims 序为 (l,w,h)，decode 代码不重排、输出即训练语义，与 pointpillars/centerpoint
    旧权重「idx3=l 实测校准」同源。曾误判 idx3=w 做过 swap（[0,1,2,4,3,5,6]）→
    car 全灭 mAP 5.8；修正为恒等直通。保留函数作归一化单一事实源（新模型按需在此换序）。
    """
    return boxes


class BevFusionDetector:
    """mmdet3d BEVFusion 检测器（懒加载：构造零加载，detect 时 _load 幂等 + ImportError 守卫）。"""

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
        self._class_names: tuple[str, ...] = ()
        self._pipeline: Any = None
        self._box_type_3d: Any = None
        self._box_mode_3d: Any = None

    def _load(self) -> Any:
        """懒加载 BEVFusion 模型（幂等；mmdet3d 未装抛 ImportError，守卫路径）。"""
        if self._model is not None:
            return self._model
        try:
            import mmdet3d  # noqa: F401  # 守卫探测：未装时抛 ImportError
            import torch
            from mmdet3d.structures import get_box_type  # pyright: ignore[reportMissingImports]
            from mmengine.config import Config
        except ImportError as e:
            raise ImportError(
                "mmdet3d/mmcv 未安装，无法使用 BEVFusion"
                "（安装步骤见 milestone/v0.2.md M1）"
            ) from e
        # custom_imports 指向 projects.BEVFusion.bevfusion：configs 仓库根入 sys.path
        if str(MMDET3D_CONFIG_DIR) not in sys.path:
            sys.path.insert(0, str(MMDET3D_CONFIG_DIR))
        device = self._device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        # 确定性红线（同 Mmdet3dDetector，批量 parity 铁律）
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        cfg = copy.deepcopy(Config.fromfile(self._config_path))
        # Swin init_cfg 指向 GitHub 预训练 URL（本地无网络）→ 整段删除跳过预训练
        # 初始化；融合 checkpoint 自带 img_backbone 权重（load_checkpoint 全量覆盖）
        if "init_cfg" in cfg.model.img_backbone:
            del cfg.model.img_backbone["init_cfg"]
        model = _init_model_trusted(cfg, self._checkpoint_path, device)
        # 稀疏卷积权重换序补载：_init_model_trusted 的 load_checkpoint 对形状不匹配
        # 的键整批跳过（日志 size mismatch），此处按 mmcv 2.x 布局补载（见 helper）
        model.load_state_dict(
            _transpose_sparse_conv_weights(_load_checkpoint_state(self._checkpoint_path)),
            strict=False,
        )
        self._class_names = _extract_class_names(cfg, model)
        # 测试管线与 box 语义直接取自 config（单一事实源）
        from mmcv.transforms import Compose  # pyright: ignore[reportMissingImports]

        self._pipeline = Compose(cfg.test_dataloader.dataset.pipeline)
        # box_type_3d 必须是**类**（见模块 docstring 的 BEVFusion.predict 实码结论）
        self._box_type_3d, self._box_mode_3d = get_box_type(
            cfg.test_dataloader.dataset.box_type_3d
        )
        self._model = model
        return self._model

    @property
    def class_names(self) -> tuple[str, ...]:
        """类名元组（触发懒加载——config 真实类序，非 devkit 序）。"""
        self._load()
        return self._class_names

    def detect_sample(
        self,
        sample: dict,
        nusc: Any,
        dataroot: Path,
        conf_threshold: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """nuScenes 单 sample → (boxes, scores, labels)（传感器系统一契约，≥阈值）。

        boxes = [x,y,z,l,w,h,yaw,vx,vy]（_normalize_bevfusion_boxes 归一化），
        与 centerpoint/pointpillars 的 detect_points 同契约 → 补偿链直用。
        """
        import torch
        from mmengine.dataset import pseudo_collate  # pyright: ignore[reportMissingImports]

        model = self._load()
        data_ = build_bevfusion_data(
            nusc,
            sample,
            dataroot,
            box_type_3d=self._box_type_3d,
            box_mode_3d=self._box_mode_3d,
        )
        data_ = self._pipeline(data_)
        collated = pseudo_collate([data_])
        with torch.no_grad():
            results = model.test_step(collated)
        pred: Any = results[0].pred_instances_3d
        boxes = _normalize_bevfusion_boxes(pred.bboxes_3d.tensor.detach().cpu().numpy())
        scores = pred.scores_3d.detach().cpu().numpy()
        labels = pred.labels_3d.detach().cpu().numpy()
        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold
        mask = scores >= threshold
        return boxes[mask], scores[mask], labels[mask]


def create_bevfusion_detector(model_name: str | None) -> BevFusionDetector | None:
    """工厂：BEVFUSION_NAMES 路由 → BevFusionDetector；None/未知名 → None。

    构造零加载——config/权重文件存在性延迟到 _load（同 create_detector3d）。
    """
    if not model_name or model_name not in BEVFUSION_NAMES:
        return None
    entry = BEVFUSION_NAMES[model_name]
    config_path = MMDET3D_CONFIG_DIR / entry["config"]
    checkpoint_path = WEIGHTS_DIR / entry["weights_dir"] / entry["checkpoint"]
    return BevFusionDetector(config_path=config_path, checkpoint_path=checkpoint_path)
