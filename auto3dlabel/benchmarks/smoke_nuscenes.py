"""nuScenes Mini 冒烟（M7 执行器，真实权重）：pointpillars_nus → 简化评测出表 + 提交 JSON 自检。

用法：
    python3 -m auto3dlabel.benchmarks.smoke_nuscenes [model] [conf] [labels]

管线：Mini val 2 场景 → 每 sample LIDAR_TOP 主点云 → detect_points（传感器系）→
calib + ego_pose 两级补偿转全局（NusBox）→ run_nuscenes_benchmark 简化评测出表 →
build_submission_json + validate + write 自检。

回灌模式（P2 E2E 验收）：第三个位置参数给 labels 文件或目录（Web 复核导出的
{sample_token}.json，load_dataset_labels 读回）→ **当 pred** 与官方 GT 出表——复核
标签可解析 + 标注质量一站验收；此模式不加载检测器（零显存）。

口径（如实记录，见 nuscenes_benchmark 模块 docstring）：Mini 2 场景 vs 官方 val 150 场景；
预测只用主 LIDAR_TOP（不合并 sweeps）——官方评测用全量扫，recall 会偏低，数字仅对照参考。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from auto3dlabel.benchmarks.nuscenes_benchmark import run_nuscenes_benchmark
from auto3dlabel.configs.kitti import DEFAULT_CONF
from auto3dlabel.configs.nuscenes import NUSCENES_CLASSES
from auto3dlabel.data.nuscenes import (
    boxes_sensor_to_global,
    gt_boxes_of_sample,
    lidar_sample_data,
    load_lidar_points,
    load_nuscenes,
    samples_of_scene,
    val_scene_names,
)
from auto3dlabel.export.nuscenes_json import (
    build_submission_json,
    validate_submission,
    write_submission,
)
from auto3dlabel.export.nuscenes_labels import load_dataset_labels
from auto3dlabel.models.detection3d import create_detector3d
from auto3dlabel.schema.nuscenes_box import NusBox


def _predict_sample(
    det: Any,
    sample: dict,
    nusc: Any,
    dataroot: Path,
    conf: float,
    class_names: list[str],
) -> list[NusBox]:
    """单 sample 检测：传感器系 → calib 位姿 → ego 地面系 → ego_pose → 全局系 NusBox。

    补偿链为公共纯函数 boxes_sensor_to_global（data/nuscenes.py，smoke_bevfusion 复用）。
    """
    pts = load_lidar_points(sample, nusc, dataroot)
    boxes, scores, labels = det.detect_points(pts, conf)
    if len(boxes) == 0:
        return []
    lidar_data = lidar_sample_data(sample, nusc)
    ego = nusc.get("ego_pose", lidar_data["ego_pose_token"])
    calib = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
    return boxes_sensor_to_global(boxes, scores, labels, class_names, ego, calib)


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
    labels_arg = argv[2] if len(argv) > 2 else ""  # 回灌模式：Web 复核 labels 当 pred

    console = Console()
    nusc = load_nuscenes()
    dataroot = Path(nusc.dataroot)
    pred_map: dict[str, list[NusBox]] = {}
    if labels_arg:
        # 回灌模式（P2 E2E 验收）：复核导出的 {token}.json 当 pred 与官方 GT 出表
        det = None
        class_names: list[str] = []
        pred_map = load_dataset_labels(labels_arg)
        console.print(f"[bold]回灌[/bold] labels={labels_arg}（{len(pred_map)} samples 当 pred）")
        if not pred_map:
            raise SystemExit(f"labels 目录无有效样本: {labels_arg}")
    else:
        det = create_detector3d(model)
        if det is None:
            raise SystemExit(f"未知名 3D 模型：{model}")
        class_names = list(det.class_names)  # 触发懒加载——nus config 真实类序（10 类）
        console.print(f"[bold]冒烟[/bold] model={model} conf={conf} dataroot={dataroot}")

    gt_map: dict[str, list[NusBox]] = {}
    egos: dict[str, tuple[float, float]] = {}  # 距离分桶相对自车（官方 distance 口径）
    t0 = time.perf_counter()
    n_scenes = 0
    n_samples = 0
    for scene_name in val_scene_names():
        n_scenes += 1
        samples = samples_of_scene(nusc, scene_name)
        for sample in samples:
            n_samples += 1
            gt_map[sample["token"]] = gt_boxes_of_sample(nusc, sample["token"])
            if det is not None:
                pred_map[sample["token"]] = _predict_sample(
                    det, sample, nusc, dataroot, conf, class_names
                )
            lidar_data = lidar_sample_data(sample, nusc)
            ego = nusc.get("ego_pose", lidar_data["ego_pose_token"])
            egos[sample["token"]] = (float(ego["translation"][0]), float(ego["translation"][1]))
    elapsed = time.perf_counter() - t0
    n_pred = sum(len(v) for v in pred_map.values())
    console.print(
        f"[bold]耗时[/bold] {elapsed:.1f}s（{elapsed / max(n_samples, 1):.2f}s/sample）"
        f"{n_scenes} 场景 {n_samples} samples 共 {n_pred} 个预测框"
    )

    bench = run_nuscenes_benchmark(gt_map, pred_map, egos=egos)
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
