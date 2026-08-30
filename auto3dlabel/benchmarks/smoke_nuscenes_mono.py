"""nuScenes Mini 单目冒烟（P6b 执行器，真实权重）：fcos3d_nus → 简化评测出表 + 提交 JSON 自检。

用法：
    python3 -m auto3dlabel.benchmarks.smoke_nuscenes_mono [model] [conf]

管线：Mini val 2 场景 → 每 sample CAM_FRONT 图像 → Fcos3dNuScenesDetector.detect_sample
（相机系 9 值 → v0.15 定式两级补偿转全局 NusBox）→ run_nuscenes_benchmark 简化评测
出表 → build_submission_json + validate + write 自检。

口径（如实记录，见 nuscenes_benchmark 模块 docstring）：Mini 2 场景 vs 官方 val 150
场景；**单相机 CAM_FRONT**（官方 FCOS3D 评测走 6 相机全量，recall 受前视 FOV 限制
系统性偏低，数字仅作质量档对照参考）。
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
    gt_boxes_of_sample,
    lidar_sample_data,
    load_nuscenes,
    samples_of_scene,
    val_scene_names,
)
from auto3dlabel.export.nuscenes_json import (
    build_submission_json,
    validate_submission,
    write_submission,
)
from auto3dlabel.models.mono3d import create_fcos3d_detector
from auto3dlabel.schema.nuscenes_box import NusBox


def _format_table(bench: dict[str, Any]) -> str:
    """整体 + 距离分桶 mAP 表格（每类 AP %）——同 smoke_nuscenes 口径。"""
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
    model = argv[0] if len(argv) > 0 else "fcos3d_nus"
    conf = float(argv[1]) if len(argv) > 1 else DEFAULT_CONF

    det = create_fcos3d_detector(model)
    if det is None:
        raise SystemExit(f"未知名单目 nuScenes 模型：{model}")
    nusc = load_nuscenes()
    console = Console()
    console.print(f"[bold]单目冒烟[/bold] model={model} conf={conf} dataroot={nusc.dataroot}")

    gt_map: dict[str, list[NusBox]] = {}
    pred_map: dict[str, list[NusBox]] = {}
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
            pred_map[sample["token"]] = det.detect_sample(sample, nusc, conf)
            lidar_data = lidar_sample_data(sample, nusc)
            ego = nusc.get("ego_pose", lidar_data["ego_pose_token"])
            egos[sample["token"]] = (float(ego["translation"][0]), float(ego["translation"][1]))
    elapsed = time.perf_counter() - t0
    n_pred = sum(len(v) for v in pred_map.values())
    console.print(
        f"[bold]耗时[/bold] {elapsed:.1f}s（{elapsed / max(n_samples, 1):.2f}s/sample）"
        f"{n_scenes} 场景 {n_samples} samples 共 {n_pred} 个预测框（CAM_FRONT 单相机）"
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
