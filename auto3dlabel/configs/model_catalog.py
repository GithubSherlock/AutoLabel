"""模型目录（auto3dlabel）。

所有 mmdet3d 3D 检测模型的路由表单一事实源（v0.3 P6 收拢自
configs/kitti.py 与 configs/nuscenes.py）：模型名 → {config 相对路径,
checkpoint 文件名, weights_dir}。

- config 相对路径以 ``configs/kitti.MMDET3D_CONFIG_DIR`` 为根
  （sparse-checkout 的 mmdet3d configs 仓库，不入库）；权重落
  ``configs/kitti.WEIGHTS_DIR/<weights_dir>/``
- config/权重 URL 以 sparse-checkout 的 configs/*/metafile.yml 为准
  （M3 下载脚本提取核对）
- 协议纪律：DETECTOR3D_NAMES 走 detect(frame) 单帧协议（Mmdet3dDetector）；
  BEVFUSION_NAMES / FCOS3D_NAMES 走 detect_sample(sample, nusc, ...)
  nuScenes 多模态/单目协议，**不进 DETECTOR3D_NAMES**（协议不兼容）；
  MONO3D_NAMES 走 detect(image) KITTI 单目协议
- ``format_catalog_summary3d()`` 供 planner3d system prompt 注入
  （token 可控；新增模型无需改 prompt 文本）
"""

from __future__ import annotations

# ============================================================
# LiDAR 3D 检测器（mmdet3d，detect(frame) 单帧协议；Mmdet3dDetector）
# ============================================================
DETECTOR3D_NAMES = {
    "pointpillars_kitti": {
        "config": "configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py",
        "checkpoint": (
            "hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth"
        ),
        "weights_dir": "pointpillars_kitti",
    },
    "pointpillars_nus": {
        "config": "configs/pointpillars/pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d.py",
        "checkpoint": (
            "hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth"
        ),
        "weights_dir": "pointpillars_nus",
    },
    "centerpoint_nus": {
        "config": (
            "configs/centerpoint/"
            "centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py"
        ),
        "checkpoint": (
            "centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_"
            "20220811_031844-191a3822.pth"
        ),
        "weights_dir": "centerpoint_nus",
    },
    # v0.3 P2：KITTI 精度升级（v1.4.0 无 KITTI centerpoint → 走 PV-RCNN；zoo car moderate 81.4）
    "pvrcnn_kitti": {
        "config": "configs/pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-3class.py",
        "checkpoint": "pv_rcnn_8xb2-80e_kitti-3d-3class_20221117_234428-b384d22f.pth",
        "weights_dir": "pvrcnn_kitti",
    },
    # v0.3 P6b：速度档（对照 CenterPoint 基线）——FreeAnchor regnet-400mf 轻骨干
    # （VoxelNeXt/TransFusion 不在 v1.4.0 主线 configs（OpenPCDet 框架）→ 不接入，
    # 如实记录见 tests/test-v0.3.md P6 段；backbone init_cfg Pretrained 下载
    # 由 detection3d._patch_pretrained_init 阻断，全量 ckpt 已含 backbone 权重）
    "free_anchor_nus": {
        "config": (
            "configs/free_anchor/"
            "pointpillars_hv_regnet-400mf_fpn_head-free-anchor_sbn-all_8xb4-2x_nus-3d.py"
        ),
        "checkpoint": (
            "hv_pointpillars_regnet-400mf_fpn_sbn-all_free-anchor_4x8_2x_nus-3d"
            "_20210827_213939-a2dd3fff.pth"
        ),
        "weights_dir": "free_anchor_nus",
    },
}

# ============================================================
# KITTI 单目 3D 检测器（v0.3 P3：LiDAR 不可用时的降级方案，交叉验证基准；
# detect(image) 协议，Mono3dDetector）
# 输出 = 相机系 7 值 [x,y,z,l,h,w,ry] 底面中心（与 label_2 同构，Det3DResult 直通）
# ============================================================
MONO3D_NAMES = {
    "pgd_kitti": {
        "config": "configs/pgd/pgd_r101-caffe_fpn_head-gn_4xb3-4x_kitti-mono3d.py",
        "checkpoint": "pgd_r101_caffe_fpn_gn-head_3x4_4x_kitti-mono3d_20211022_102608-8a97533b.pth",
        "weights_dir": "pgd_kitti",
    },
}

# ============================================================
# BEVFusion 相机-LiDAR 融合（v0.3 P3；mmdet3d projects/BEVFusion，
# detect_sample(sample, nusc, dataroot) 多模态协议——BevFusionDetector）
# config 位于 projects/BEVFusion/configs/（custom_imports 注册组件）
# ============================================================
BEVFUSION_NAMES = {
    "bevfusion_nus": {
        "config": (
            "projects/BEVFusion/configs/"
            "bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
        ),
        "checkpoint": (
            "bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d"
            "-5239b1af.pth"
        ),
        "weights_dir": "bevfusion_nus",
    },
    # lidar-only 备选（R3 降级链：融合 OOM/失败时切换）
    "bevfusion_lidar_nus": {
        "config": (
            "projects/BEVFusion/configs/"
            "bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
        ),
        "checkpoint": (
            "bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d"
            "-2628f933.pth"
        ),
        "weights_dir": "bevfusion_lidar_nus",
    },
}

# ============================================================
# FCOS3D nuScenes 单目（v0.3 P6b：单目质量档扩展；
# detect_sample(sample, nusc, conf) 协议——Fcos3dNuScenesDetector）
# 输出语义（2026-08-30 实测定案，详见 models/mono3d.py Fcos3dNuScenesDetector）：
# 相机系 9 值几何中心。head 实测输出 tensor[:,3:6] 为 **(l,h,w) 序**、yaw 为
# **负相机系 yaw**（2021-07 权重经 v1.4.0 head 推理的实测语义：单样本探针 + Mini
# 全量 4 组合评测 D「dims 重排 (t5,t3,t4) + yaw 负号」mAP 0.8 / car 8.0，其余三组
# 全 0）——与 v0.15 源码考古结论（(w,l,h) 直传 + yaw 无负号）不符，以实测为准。
# 适配在 models/mono3d._fcos3d_reorder_dims_yaw 归一为 (w,l,h)+正 yaw 后，
# 全局转换仍复刻 v0.15 版 output_to_nusc_box（data/nuscenes.boxes_cam_to_global）。
# ============================================================
FCOS3D_NAMES = {
    "fcos3d_nus": {
        "config": (
            "configs/fcos3d/"
            "fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune.py"
        ),
        "checkpoint": (
            "fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune"
            "_20210717_095645-8d806dc2.pth"
        ),
        "weights_dir": "fcos3d_nus",
    },
}

# ============================================================
# 目录摘要（供 planner3d system prompt 注入，单一事实源）
# ============================================================
_CATALOG_SUMMARY_GROUPS: list[tuple[str, list[str], str]] = [
    ("LiDAR 3D [KITTI]", ["pointpillars_kitti", "pvrcnn_kitti"],
     "pointpillars 快速扫 / pvrcnn 精度档"),
    ("LiDAR 3D [nuScenes]", ["pointpillars_nus", "centerpoint_nus", "free_anchor_nus"],
     "centerpoint 精度 / free_anchor 速度档"),
    ("mono 3D [KITTI]", ["pgd_kitti"], "无 LiDAR 降级交叉验证基准"),
    ("fusion 3D [nuScenes, LiDAR+camera]", ["bevfusion_nus", "bevfusion_lidar_nus"],
     "融合档（lidar 变体为降级备选）"),
    ("mono 3D [nuScenes]", ["fcos3d_nus"], "单目质量档扩展"),
]


def format_catalog_summary3d() -> str:
    """生成 3D 引擎目录摘要（供 planner3d system prompt 注入，token 可控）。

    调用方需先加载模型名 → 引擎的映射语义；模型名 = 目录表 key（exact names）。
    """
    lines = ["LiDAR/mono 3D engines (use exact model names):"]
    for group, names, hint in _CATALOG_SUMMARY_GROUPS:
        lines.append(f"- {group}: {', '.join(names)} — {hint}")
    return "\n".join(lines)
