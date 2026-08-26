"""nuScenes Mini 冒烟（M7 执行器，真实权重）：pointpillars_nus → 简化评测出表 + 提交 JSON 自检。

用法：
    python3 -m auto3dlabel.benchmarks.smoke_nuscenes [model] [conf]

管线：Mini val 2 场景 → 每 sample LIDAR_TOP 主点云 → detect_points（自车系）→
ego_pose 补偿转全局（NusBox）→ run_nuscenes_benchmark 简化评测出表 →
build_submission_json + validate + write 自检。

口径（如实记录，见 nuscenes_benchmark 模块 docstring）：Mini 2 场景 vs 官方 val 150 场景；
预测只用主 LIDAR_TOP（不合并 sweeps）——官方评测用全量扫，recall 会偏低，数字仅对照参考。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

from auto3dlabel.benchmarks.nuscenes_benchmark import run_nuscenes_benchmark
from auto3dlabel.configs.kitti import DEFAULT_CONF
from auto3dlabel.configs.nuscenes import NUSCENES_CLASSES
from auto3dlabel.data.nuscenes import (
    gt_boxes_of_sample,
    load_nuscenes,
    samples_of_scene,
    val_scene_names,
)
from auto3dlabel.export.nuscenes_json import (
    build_submission_json,
    validate_submission,
    write_submission,
)
from auto3dlabel.models.detection3d import create_detector3d
from auto3dlabel.schema.nuscenes_box import NusBox
from auto3dlabel.tools.geometry import yaw_to_quat


def _rot_matrix(q: tuple[float, float, float, float]) -> np.ndarray:
    """四元数 (w,x,y,z) → 3x3 旋转矩阵（Hamilton 约定，nus devkit 同）。"""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _quat_mul(
    q1: tuple[float, float, float, float], q2: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """Hamilton 积 q1 ⊗ q2（nus devkit Quaternion 同约定）。"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _load_lidar(sample: dict, nusc: Any, dataroot: Path) -> np.ndarray:
    """sample → LIDAR_TOP 主点云 (N,4)（简化口径：不合并 sweeps）。

    文件名为 scene__sensor__timestamp 格式（v1.0 samples 命名）——经 devkit
    sample_data 表映射（nusc.get('sample_data', token)['filename']）。
    """
    lidar_token = sample["data"]["LIDAR_TOP"]
    filename = nusc.get("sample_data", lidar_token)["filename"]
    raw = np.fromfile(dataroot / filename, dtype=np.float32)
    return raw.reshape(-1, 5)  # (x,y,z,intensity,elongation)——nus pipeline 要 5 通道


def _predict_sample(
    det: Any,
    sample: dict,
    nusc: Any,
    dataroot: Path,
    conf: float,
    class_names: list[str],
) -> list[NusBox]:
    """单 sample 检测：自车系 → ego_pose 补偿 → 全局系 NusBox。"""
    pts = _load_lidar(sample, nusc, dataroot)
    boxes, scores, labels = det.detect_points(pts, conf)
    if len(boxes) == 0:
        return []
    # ego_pose 挂在 LIDAR_TOP 的 sample_data 记录上（sample 表无此字段）
    lidar_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ego = nusc.get("ego_pose", lidar_data["ego_pose_token"])
    r_mat = _rot_matrix(tuple(ego["rotation"]))
    t_ego = np.asarray(ego["translation"], dtype=np.float64)
    q_ego = tuple(ego["rotation"])
    out: list[NusBox] = []
    for i in range(len(boxes)):
        # nus 模型 9 值输出 [x,y,z,w,l,h,yaw,vx,vy]（自车系中心 + 自车系速度）
        x, y, z, w, l, h, yaw = (float(v) for v in boxes[i, :7])
        center = r_mat @ np.array([x, y, z]) + t_ego
        quat = _quat_mul(q_ego, yaw_to_quat(yaw))
        velocity: tuple[float, float] | None = None
        if boxes.shape[1] >= 9:
            v_local = r_mat @ np.array([float(boxes[i, 7]), float(boxes[i, 8]), 0.0])
            velocity = (float(v_local[0]), float(v_local[1]))
        out.append(
            NusBox(
                label=class_names[int(labels[i])],
                confidence=float(scores[i]),
                translation=(float(center[0]), float(center[1]), float(center[2])),
                size=(w, l, h),
                quaternion=quat,
                velocity=velocity,
            )
        )
    return out


def _format_table(bench: dict[str, Any]) -> str:
    """整体 + 距离分桶 mAP 表格（每类 AP %）。"""
    import io

    buf = io.StringIO()
    console = Console(file=buf, record=True, width=110, force_terminal=False)
    table = Table(title="nuScenes Mini 简化评测（AP %，x-y 旋转矩形 IoU 0.5）", show_lines=True)
    table.add_column("范围", style="bold")
    for cls in NUSCENES_CLASSES:
        table.add_column(cls, justify="right")
    overall = bench["overall"]
    row = ["overall"]
    for cls in NUSCENES_CLASSES:
        ap = float(overall[cls]["ap"]) * 100
        row.append(f"{ap:5.1f} ({overall[cls]['gt_count']})")
    table.add_row(*row)
    for bin_name, classes in bench["distance"].items():
        row = [bin_name]
        for cls in NUSCENES_CLASSES:
            row.append(f"{float(classes[cls]['ap']) * 100:5.1f}")
        table.add_row(*row)
    console.print(table)
    return buf.getvalue()


def main(argv: list[str] | None = None) -> None:
    argv = argv or sys.argv[1:]
    model = argv[0] if len(argv) > 0 else "pointpillars_nus"
    conf = float(argv[1]) if len(argv) > 1 else DEFAULT_CONF

    det = create_detector3d(model)
    if det is None:
        raise SystemExit(f"未知名 3D 模型：{model}")
    nusc = load_nuscenes()
    dataroot = Path(nusc.dataroot)
    class_names = list(det.class_names)  # 触发懒加载——nus config 真实类序（10 类）
    console = Console()
    console.print(f"[bold]冒烟[/bold] model={model} conf={conf} dataroot={dataroot}")

    gt_map: dict[str, list[NusBox]] = {}
    pred_map: dict[str, list[NusBox]] = {}
    t0 = time.perf_counter()
    n_scenes = 0
    n_samples = 0
    for scene_name in val_scene_names():
        n_scenes += 1
        samples = samples_of_scene(nusc, scene_name)
        for sample in samples:
            n_samples += 1
            gt_map[sample["token"]] = gt_boxes_of_sample(nusc, sample["token"])
            pred_map[sample["token"]] = _predict_sample(
                det, sample, nusc, dataroot, conf, class_names
            )
    elapsed = time.perf_counter() - t0
    n_pred = sum(len(v) for v in pred_map.values())
    console.print(
        f"[bold]耗时[/bold] {elapsed:.1f}s（{elapsed / max(n_samples, 1):.2f}s/sample）"
        f"{n_scenes} 场景 {n_samples} samples 共 {n_pred} 个预测框"
    )

    bench = run_nuscenes_benchmark(gt_map, pred_map)
    console.print(_format_table(bench))
    console.print(f"mAP = {bench['mAP'] * 100:.1f}")

    # 提交 JSON 自检（官方 8 类映射 + meta；Mini 仅为格式验证）
    sub = build_submission_json(pred_map)
    errors = validate_submission(sub)
    if errors:
        raise SystemExit("提交 JSON 自检失败:\n" + "\n".join(errors))
    Path("outputs").mkdir(exist_ok=True)
    out_path = write_submission(sub, Path("outputs") / f"nuscenes_submission_{model}.json")
    console.print(f"[bold]产物[/bold] {out_path}（{len(sub['results'])} samples，自检通过）")


if __name__ == "__main__":
    main()
