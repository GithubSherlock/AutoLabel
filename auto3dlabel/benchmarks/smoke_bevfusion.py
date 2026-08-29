"""BEVFusion nuScenes Mini 冒烟（P3 执行器，真实权重）：融合引擎 → 简化评测出表 + 提交 JSON 自检。

用法：
    python3 -m auto3dlabel.benchmarks.smoke_bevfusion [model] [conf]

管线：Mini val 2 场景（81 samples）→ 每 sample build_bevfusion_data（6 相机 +
LIDAR_TOP）→ detect_sample（传感器系）→ boxes_sensor_to_global 两级补偿（与
smoke_nuscenes 同链，复用不复制）→ run_nuscenes_benchmark 出表（vs
pointpillars_nus 17.7 / centerpoint_nus 23.4，test-v0.3.md P2 同口径）→
submission JSON 自检。

口径（如实记录，见 nuscenes_benchmark 模块 docstring）：Mini 2 场景 vs 官方
val 150 场景；预测只用主 LIDAR_TOP + pad_empty_sweeps 重复 keyframe（无真实
sweeps，recall 偏低）；融合引擎需 6 相机齐备（build_bevfusion_data 标定链）。
仅做整体评测，不改任何测试代码。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from rich.console import Console

from auto3dlabel.benchmarks.nuscenes_benchmark import run_nuscenes_benchmark
from auto3dlabel.benchmarks.smoke_nuscenes import _format_table
from auto3dlabel.configs.kitti import DEFAULT_CONF
from auto3dlabel.data.nuscenes import (
    boxes_sensor_to_global,
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
from auto3dlabel.models.bevfusion3d import create_bevfusion_detector
from auto3dlabel.schema.nuscenes_box import NusBox


def _predict_sample(
    det: Any,
    sample: dict,
    nusc: Any,
    dataroot: Path,
    conf: float,
    class_names: list[str],
) -> list[NusBox]:
    """单 sample 检测：传感器系（detect_sample 归一化契约）→ 补偿链 → 全局系 NusBox。"""
    boxes, scores, labels = det.detect_sample(sample, nusc, dataroot, conf)
    if len(boxes) == 0:
        return []
    lidar_data = lidar_sample_data(sample, nusc)
    ego = nusc.get("ego_pose", lidar_data["ego_pose_token"])
    calib = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
    return boxes_sensor_to_global(boxes, scores, labels, class_names, ego, calib)


def main(argv: list[str] | None = None) -> None:
    argv = argv or sys.argv[1:]
    model = argv[0] if len(argv) > 0 else "bevfusion_nus"
    conf = float(argv[1]) if len(argv) > 1 else DEFAULT_CONF

    det = create_bevfusion_detector(model)
    if det is None:
        raise SystemExit(f"未知名 BEVFusion 模型：{model}")
    nusc = load_nuscenes()
    dataroot = Path(nusc.dataroot)
    class_names = list(det.class_names)  # 触发懒加载——config 真实类序（10 类）
    console = Console()
    console.print(f"[bold]冒烟[/bold] model={model} conf={conf} dataroot={dataroot}")

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
