"""nuScenes 训练标签导出（v0.4 P2）：Web save 后逐样本 {sample_token}.json 落盘。

单样本文件 = {"sample_token", "boxes": [NusBox.to_dict() + velocity 填充]}——与官方
submission box dict 同键（detection_name/detection_score/translation/size/rotation/velocity），
devkit create_submission 可直接读；数据集级合并/自检**复用 export/nuscenes_json**
（build_submission_json / validate_submission，不重复造格式）。

已知简化（如实记录）：track_id 不入文件（to_dict 不输出，官方格式无此字段）；
velocity 缺失 → [0.0, 0.0]（官方格式要求键存在，同 nuscenes_json 纪律）。
"""

from __future__ import annotations

import json
from pathlib import Path

from auto3dlabel.schema.nuscenes_box import NusBox


def build_label_dict(sample_token: str, boxes: list[NusBox]) -> dict:
    """单样本标签文件 dict：{sample_token, boxes: [box dict]}（velocity 恒输出）。"""
    out_boxes: list[dict] = []
    for b in boxes:
        d = b.to_dict()
        d["velocity"] = list(b.velocity) if b.velocity is not None else [0.0, 0.0]
        out_boxes.append(d)
    return {"sample_token": sample_token, "boxes": out_boxes}


def write_sample_label(
    sample_token: str, boxes: list[NusBox], labels_dir: str | Path
) -> Path:
    """sample → {labels_dir}/{sample_token}.json（Web save 与 CLI 导出共用）。"""
    out = Path(labels_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{sample_token}.json"
    path.write_text(
        json.dumps(build_label_dict(sample_token, boxes), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def load_sample_label(path: Path) -> tuple[str, list[NusBox]]:
    """{sample_token}.json → (token, [NusBox])（NusBox.from_dict 读回）。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    boxes = [NusBox.from_dict(b) for b in data.get("boxes", []) if isinstance(b, dict)]
    return str(data.get("sample_token", "")), boxes


def load_dataset_labels(labels_dir: str | Path) -> dict[str, list[NusBox]]:
    """labels 目录（或单文件）全量 → {sample_token: [NusBox]}（损坏/缺 token 跳过，宁缺勿假）。

    回灌评测入口：smoke_nuscenes 的 labels 参数（文件或目录均可）→ 直接喂
    run_nuscenes_benchmark。
    """
    p = Path(labels_dir)
    paths = [p] if p.is_file() else sorted(p.glob("*.json"))
    out: dict[str, list[NusBox]] = {}
    for path in paths:
        try:
            token, boxes = load_sample_label(path)
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            continue
        if token:
            out[token] = boxes
    return out
