#!/usr/bin/env python3
"""KITTI 域内微调工具（v0.4 Phase 2 GPU 项）—— label_2 → YOLO 转换 + 微调训练。

闭环（cityscapes 0.0082→0.5149 先例）：KITTI GT 转换 → YOLO 微调 →
`kitti_benchmark` 对比基线（yolo26x 零样本 0.4081）。

类别映射与评测口径一致（`benchmarks/kitti_benchmark.py` KITTI_TO_COCO 单一
事实源）：Car/Van→car、Pedestrian/Person_sitting→person、Cyclist→bicycle、
Truck→truck、Tram→train；Misc/DontCare 与不满足任何 difficulty 档
（easy/moderate/hard 官方判据，同 `kitti_difficulty`）的对象剔除。

布局（零图像复制——images 为 symlink 指向 training/image_2）:
    DATASETS_ROOT/KITTI/yolo/
        images/{train,val}/*.png   (symlink)
        labels/{train,val}/*.txt   (YOLO 归一化)
        data.yaml

用法:
    python3 -m auto2dlabel.tools.train_kitti              # 转换 + 训练（需 GPU）
    python3 -m auto2dlabel.tools.train_kitti --dry-run    # 仅转换（CPU 可跑）
    python3 -m auto2dlabel.tools.train_kitti --epochs 40 --batch 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

from auto2dlabel.benchmarks.datasets import DATASETS_ROOT
from auto2dlabel.benchmarks.kitti_benchmark import KITTI_TO_COCO, kitti_difficulty

KITTI_ROOT = DATASETS_ROOT / "KITTI" / "object"
YOLO_ROOT = DATASETS_ROOT / "KITTI" / "yolo"

# 与评测口径一致：KITTI_TO_COCO 单一事实源 → YOLO 类索引（排序后确定性）
KITTI_YOLO_NAMES = sorted(set(KITTI_TO_COCO.values()))
KITTI_TO_YOLO = {k: KITTI_YOLO_NAMES.index(v) for k, v in KITTI_TO_COCO.items()}


def kitti_line_to_yolo(line: str, img_w: int, img_h: int) -> str | None:
    """单行 label_2 → YOLO 行 "cls cx cy w h"（归一化）。

    剔除 Misc/DontCare 与不满足任何 difficulty 档的对象（与 benchmark
    评测 ignore 语义一致）；行格式残缺返回 None。
    """
    parts = line.split()
    if len(parts) < 8:
        return None
    yolo_cls = KITTI_TO_YOLO.get(parts[0])
    if yolo_cls is None:
        return None
    truncated = float(parts[1])
    occluded = int(parts[2])
    x1, y1, x2, y2 = (float(p) for p in parts[4:8])
    if kitti_difficulty(truncated, occluded, y2 - y1) is None:
        return None
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    return (
        f"{yolo_cls} {(x1 + x2) / 2 / img_w:.6f} {(y1 + y2) / 2 / img_h:.6f} "
        f"{w / img_w:.6f} {h / img_h:.6f}"
    )


def convert_label_file(
    label_path: Path, out_path: Path, img_w: int, img_h: int
) -> int:
    """单个 label_2 → YOLO txt；返回写入行数（已存在非空时幂等跳过，返回 -1）。"""
    if out_path.exists() and out_path.stat().st_size > 0:
        return -1
    lines = [
        converted
        for converted in (
            kitti_line_to_yolo(ln, img_w, img_h)
            for ln in label_path.read_text().splitlines()
        )
        if converted
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def get_image_size(image_path: Path) -> tuple[int, int]:
    """PIL 读图像尺寸（YOLO 归一化需要；KITTI object 全为 1242×375 但按实际读）。"""
    from PIL import Image

    with Image.open(image_path) as im:
        return im.size


def split_train_val(
    files: list[Path], val_ratio: float = 0.1, seed: int = 42
) -> tuple[list[Path], list[Path]]:
    """确定性划分：随机种子 shuffle 后前 val_ratio 进 val（可复现）。"""
    import random

    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    n_val = int(len(shuffled) * val_ratio)
    return shuffled[n_val:], shuffled[:n_val]


def write_data_yaml(yolo_root: Path) -> Path:
    """写 ultralytics data.yaml（names 与 KITTI_YOLO_NAMES 一致）。"""
    yaml_path = yolo_root / "data.yaml"
    yaml_path.write_text(
        f"path: {yolo_root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(KITTI_YOLO_NAMES))
    )
    return yaml_path


def convert_dataset(
    kitti_root: Path = KITTI_ROOT,
    yolo_root: Path = YOLO_ROOT,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Path:
    """label_2 → YOLO 布局（转换 + 划分 + symlink + data.yaml，幂等）。

    返回 data.yaml 路径；images 用 symlink 指向 training/image_2（零复制）。
    """
    label_dir = kitti_root / "training" / "label_2"
    image_dir = kitti_root / "training" / "image_2"
    label_files = sorted(label_dir.glob("*.txt"))
    if not label_files:
        raise FileNotFoundError(f"KITTI 标注缺失: {label_dir}")

    train_files, val_files = split_train_val(label_files, val_ratio, seed)
    for split_name, files in (("train", train_files), ("val", val_files)):
        out_img_dir = yolo_root / "images" / split_name
        out_lbl_dir = yolo_root / "labels" / split_name
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)
        for lf in files:
            img = image_dir / f"{lf.stem}.png"
            if not img.exists():
                print(f"跳过（缺图）: {img.name}")
                continue
            link = out_img_dir / img.name
            if not link.exists():
                link.symlink_to(img)
            w, h = get_image_size(img)
            convert_label_file(lf, out_lbl_dir / f"{lf.stem}.txt", w, h)

    yaml_path = write_data_yaml(yolo_root)
    print(
        f"✓ KITTI YOLO 布局: train={len(train_files)}  val={len(val_files)}"
        f"（{yaml_path}）"
    )
    return yaml_path


def train(
    data_yaml: Path,
    base_model: str = "yolo11s.pt",
    epochs: int = 80,
    batch: int = 16,
    imgsz: int = 640,
    patience: int = 15,
    project: Path | None = None,
    name: str = "yolo11s_kitti",
) -> Path:
    """ultralytics YOLO 微调（需 GPU）；返回 best.pt 路径。

    权重落 auto2dlabel/weights/kitti_finetune/（不入库）；base_model 自动
    下载到统一权重目录。显存 12GB 档：yolo11s batch=16。
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("KITTI 微调需 GPU")

    from ultralytics import YOLO, settings  # type: ignore

    from auto2dlabel.configs.model_catalog import WEIGHTS_DIR

    settings.update({  # type: ignore[no-untyped-call]
        "weights_dir": str(WEIGHTS_DIR),
        "datasets_dir": str(WEIGHTS_DIR / "datasets"),
    })

    print(f"🏋 微调 {base_model}: epochs={epochs} batch={batch} imgsz={imgsz}")
    # 显式路径：裸文件名在 ultralytics 新版本会下载到 CWD（权重管理红线：
    # 一律进 weights/）
    base_path = WEIGHTS_DIR / base_model
    model = YOLO(str(base_path))
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=0,
        patience=patience,
        seed=42,
        project=str(project or WEIGHTS_DIR / "kitti_finetune"),
        name=name,
        exist_ok=True,
    )

    best = (project or WEIGHTS_DIR / "kitti_finetune") / name / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"训练产物缺失: {best}")
    print(f"✓ 微调完成: {best}")
    print(
        "  benchmark: "
        f"python3 -m auto2dlabel.benchmarks.kitti_benchmark "
        f"--model {best} --max-images 0 --difficulty all"
    )
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="KITTI 域内微调（v0.4 Phase 2）")
    parser.add_argument("--base", default="yolo11s.pt", help="基础模型（自动下载）")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=16, help="训练 batch（12GB 档 16）")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument(
        "--dry-run", action="store_true", help="仅转换数据不训练（CPU 可跑）"
    )
    args = parser.parse_args()

    data_yaml = convert_dataset(val_ratio=args.val_ratio)
    if args.dry_run:
        return
    train(
        data_yaml,
        base_model=args.base,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
