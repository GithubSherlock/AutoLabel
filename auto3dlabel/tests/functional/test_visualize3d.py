"""BEV 点云预测图回归（2026-09-06 点云线 BEV 产物）。

红线：零真实权重——合成帧 + FakeDetector3D；cv2 读回断言像素级内容
（点云散点渲染、GT 蓝框、预测框类别色）。验收语义：LiDAR 引擎的
可视化产物 = 点云散点 + 预测 3D 框，chat（Detect3DTool/Visualize3DTool）
与 cli run（pipeline._draw_lidar_bev）共用 draw_lidar_bev_predictions。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import cv2
import numpy as np

from auto3dlabel.agent.tools3d import Detect3DTool, Visualize3DTool
from auto3dlabel.schema.box3d import Box3D
from auto3dlabel.tests.helpers.fakes import FakeDetection, FakeDetector3D, FakeSegmentation, det3d
from auto3dlabel.tests.helpers.synth import write_frame
from auto3dlabel.tools.visualize import draw_bev, draw_lidar_bev_predictions

_BG = np.array([32, 32, 32], dtype=np.uint8)  # draw_bev canvas 背景灰
_GT_BLUE = np.array([255, 128, 0], dtype=np.uint8)  # GT 框 BGR
_CAR_GREEN = np.array([0, 255, 0], dtype=np.uint8)
_PED_CYAN = np.array([255, 255, 0], dtype=np.uint8)

_GT_LINE = "Car 0.0 0 0 10 10 50 50 1.5 1.6 3.9 8.0 -0.9 18.0 -1.57"


def _car_frame(tmp_path: Path, n_points: int = 4000) -> Any:
    """合成帧：velodyne 地面点散布（经真实标定映射到相机系 x∈[-20,20], z∈[10,30]
    的大区域——BEV 像素占比可断言）+ 1 条 GT 车（相机系 x≈8,z≈18）。

    真实 calib 的 velo→cam 轴变换：cam_x ≈ −velo_y、cam_z ≈ velo_x，
    故 velodyne 生成 vx∈[10,30], vy∈[−20,20] 才落到大相机区域
    （rect_points 直接生成相机系语义会经变换被压成 20×20px 小簇）。
    """
    rng = np.random.RandomState(0)
    vx = rng.uniform(10.0, 30.0, n_points)
    vy = rng.uniform(-20.0, 20.0, n_points)
    vz = rng.uniform(-1.5, 0.0, n_points)
    pts = np.stack([vx, vy, vz, np.ones(n_points)], axis=1).astype(np.float32)
    return write_frame(tmp_path, points=pts, label_lines=[_GT_LINE])


def _non_bg_ratio(png: Path) -> float:
    img = cv2.imread(str(png))
    assert img is not None, f"无法读回 {png}"
    return float((~(np.all(img == _BG, axis=2))).mean())


def _has_color(png: Path, bgr: np.ndarray) -> bool:
    img = cv2.imread(str(png))
    assert img is not None, png
    return bool((np.all(img == bgr, axis=2)).any())


def _pred_box(label: str = "Car", cx: float = 8.0, cz: float = 18.0) -> Box3D:
    return Box3D(
        label=label, confidence=0.9, cx=cx, cy=-0.9, cz=cz,
        h=1.5, w=1.6, l=3.9, yaw_bev=0.0,
    )


# ---------- draw_lidar_bev_predictions ----------


def test_lidar_bev_renders_points_box_gt(tmp_path: Path) -> None:
    """BEV 点云预测图：点云散点（非背景占比）+ GT 蓝框（专色）+ 预测框（与空框版不同）。

    预测框放 (0,24)——与 GT(8,18) 不重叠，蓝框专色不被绿线覆盖。
    """
    frame = _car_frame(tmp_path)
    out = tmp_path / "bev.png"
    path = draw_lidar_bev_predictions(
        frame, out, [_pred_box(cx=0.0, cz=24.0)], gt_boxes=frame.load_gt3d()
    )
    assert Path(path).is_file()
    assert _non_bg_ratio(out) > 0.003, "点云散点/框未渲染"
    assert _has_color(out, _GT_BLUE), "GT 蓝框缺失"
    out_no_box = tmp_path / "bev_nobox.png"
    draw_lidar_bev_predictions(frame, out_no_box, [], gt_boxes=frame.load_gt3d())
    assert out.read_bytes() != out_no_box.read_bytes(), "预测框未渲染"


def test_lidar_bev_subsamples_large_cloud(tmp_path: Path) -> None:
    """大点云（10 万点）→ 比例抽样后仍落图（≤ _BEV_MAX_POINTS，不爆内存不超时）。"""
    frame = _car_frame(tmp_path, n_points=100_000)
    out = tmp_path / "bev_big.png"
    path = draw_lidar_bev_predictions(frame, out, [])
    assert Path(path).is_file()
    assert _non_bg_ratio(out) > 0.01


# ---------- draw_bev 类别色 ----------


def test_bev_prediction_class_colors(tmp_path: Path) -> None:
    """预测框按 KITTI 类着色：Car 绿 / Pedestrian 青（无点云散点时颜色无歧义）。"""
    frame = _car_frame(tmp_path)
    out = tmp_path / "bev_classes.png"
    draw_bev(frame, out, boxes=[_pred_box("Car"), _pred_box("Pedestrian", cx=-6.0)])
    assert _has_color(out, _CAR_GREEN), "Car 绿框缺失"
    assert _has_color(out, _PED_CYAN), "Pedestrian 青框缺失"


# ---------- Detect3DTool / Visualize3DTool（chat 路径）----------


def test_detect_tool_lidar_engine_writes_bev(tmp_path: Path) -> None:
    """chat LiDAR 引擎：detect 成功后自动落 BEV 点云预测图（0 框也落图）。"""
    frame = _car_frame(tmp_path)
    out_dir = tmp_path / "out"
    tool = Detect3DTool(frame, det_model_name="pointpillars_kitti", out_dir=out_dir)
    fake = FakeDetector3D([det3d(confidence=0.9, bbox=(8.0, -0.9, 18.0, 3.9, 1.5, 1.6, 0.0))])
    with patch("auto3dlabel.models.detection3d.create_detector3d", return_value=fake):
        results = tool.forward(prompts=["car"], confidence_threshold=0.3)
    assert len(results) == 1
    bev = out_dir / "000000_bev.png"
    assert bev.is_file()
    assert _non_bg_ratio(bev) > 0.003, "BEV 点云预测图未渲染"


def test_detect_tool_2d_path_writes_no_bev(tmp_path: Path) -> None:
    """chat 2D→3D 反投影路径不落 LiDAR BEV（BEV 产物仅点云线）。"""
    frame = _car_frame(tmp_path)
    out_dir = tmp_path / "out2d"
    tool = Detect3DTool(frame, det_model_name=None, out_dir=out_dir)
    with patch(
        "auto3dlabel.tools.pipeline.create_detection_model", return_value=FakeDetection([])
    ), patch(
        "auto3dlabel.tools.pipeline.create_segmentation_model", return_value=FakeSegmentation([])
    ):
        results = tool.forward(prompts=["car"], confidence_threshold=0.3)
    assert results == []
    assert not (out_dir / "000000_bev.png").exists()


def test_visualize_tool_renders_point_cloud(tmp_path: Path) -> None:
    """Visualize3DTool 显式调用：BEV 图含点云散点（非纯框图）。"""
    frame = _car_frame(tmp_path)
    out_dir = tmp_path / "outviz"
    tool = Visualize3DTool(frame, out_dir)
    out = tool.forward(boxes=[_pred_box().to_dict()])
    assert out["success"] is True
    bev = Path(str(out["path"]))
    assert bev.is_file()
    assert _non_bg_ratio(bev) > 0.003, "点云散点未渲染"
