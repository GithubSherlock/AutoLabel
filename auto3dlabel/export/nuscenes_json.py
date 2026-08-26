"""nuScenes 官方提交 JSON 导出（v0.2 P3）：build 纯函数 + 自检。

格式对齐 devkit nuscenes/eval/detection/loaders.py create_submission 的 box dict
（sample_token / translation / size / rotation / velocity / detection_name /
detection_score / attribute_name）。

MVP 已知简化（如实记录，不静默伪造）：
- attribute_name 恒 "vehicle.moving"（按 plan；非 vehicle 类官方评测忽略 attr）
- velocity 缺失（None）→ [0.0, 0.0]（官方格式要求 velocity 键存在）
"""

from __future__ import annotations

import json
from pathlib import Path

from auto3dlabel.configs.nuscenes import NUSCENES_CLASSES
from auto3dlabel.schema.nuscenes_box import NusBox


def build_submission_json(
    results: dict[str, list[NusBox]],
    use_camera: bool = True,
    use_lidar: bool = True,
) -> dict:
    """{sample_token: [NusBox]} → 官方提交 dict（meta + results）。"""
    submission: dict[str, object] = {
        "meta": {
            "use_camera": use_camera,
            "use_lidar": use_lidar,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        },
        "results": {},
    }
    out_results: dict[str, list[dict]] = {}
    for sample_token, boxes in results.items():
        box_dicts: list[dict] = []
        for b in boxes:
            d = b.to_dict()
            d["sample_token"] = sample_token
            d["velocity"] = list(b.velocity) if b.velocity is not None else [0.0, 0.0]
            d["attribute_name"] = "vehicle.moving"  # MVP 恒值（见 docstring）
            box_dicts.append(d)
        out_results[sample_token] = box_dicts
    submission["results"] = out_results
    return submission


def validate_submission(sub: dict) -> list[str]:
    """自检 → 问题列表（空 = 通过）。类名/分数域/字段长度逐项校验。"""
    problems: list[str] = []
    results = sub.get("results", {})
    if not isinstance(results, dict):
        return ["results 不是 dict"]
    for token, boxes in results.items():
        for i, b in enumerate(boxes):
            tag = f"{token}[{i}]"
            if b.get("detection_name") not in NUSCENES_CLASSES:
                problems.append(f"{tag}: 未知类 {b.get('detection_name')!r}")
            score = b.get("detection_score")
            if not isinstance(score, (int, float)) or not 0 <= float(score) <= 1:
                problems.append(f"{tag}: detection_score 越界 {score!r}")
            for key, n in (("translation", 3), ("size", 3), ("rotation", 4), ("velocity", 2)):
                v = b.get(key)
                if not isinstance(v, (list, tuple)) or len(v) != n:
                    problems.append(f"{tag}: {key} 应为 {n} 值, 实为 {v!r}")
    return problems


def write_submission(sub: dict, path: str | Path) -> Path:
    """落盘 JSON（ensure_ascii=False；写前自检，问题非空抛 ValueError 宁缺勿假）。"""
    problems = validate_submission(sub)
    if problems:
        raise ValueError(f"提交 JSON 自检未通过（{len(problems)} 项）: {problems[:5]}")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sub, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
