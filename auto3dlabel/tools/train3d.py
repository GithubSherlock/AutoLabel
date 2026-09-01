#!/usr/bin/env python3
"""KITTI 3D 微调管线（v0.4 P3）：mmdet3d pointpillars_kitti 小样本微调（官方权重热启）。

管线（train_kitti.py 2D 先例的 3D 版）：
1. prepare_kitti_imagesets：ImageSets train 000000-003711 / val 003712-007480 / test 占位
2. run_create_data：mmdet3d create_data（subprocess cwd=.mim/tools + PYTHONPATH=$PWD；
   产物齐全幂等跳过——infos pkl×5 + dbinfos + velodyne_reduced 7481 帧）
3. subsample_infos：按 image_idx 升序截取 → 冒烟 pkl（train 300 / val 40，确定性）
4. build_finetune_config：官方 config 对象级改造 → 自足 config（data_root 绝对化、
   ann_file 指 subsample pkl、**删 GT-AUG（db_sampler + ObjectSample）**、batch/lr/epoch、
   param_scheduler 换 LinearLR 预热 + 恒定（官方 T_max 绑 epoch_num，只改 epoch 会 LR
   崩溃）、load_from 官方 pth）
5. build_train_cmd / train：单卡训练（需 GPU）

布局（零数据复制）：权重与 config 落 weights/kitti3d_finetune/（不入库）。

用法:
    python3 -m auto3dlabel.tools.train3d --dry-run   # 数据准备 + config 生成（CPU 可跑）
    python3 -m auto3dlabel.tools.train3d             # + 训练（需 GPU）
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import subprocess
import sys
from pathlib import Path

from auto3dlabel.configs.kitti import (
    DEFAULT_KITTI_ROOT,
    MMDET3D_CONFIG_DIR,
    WEIGHTS_DIR,
)
from auto3dlabel.configs.model_catalog import DETECTOR3D_NAMES

# 训练产物目录（不入库，同 weights/kitti_finetune 纪律）
FINETUNE_DIR = WEIGHTS_DIR / "kitti3d_finetune"


def mim_tools_dir() -> Path:
    """mmdet3d 安装包内 .mim/tools（create_data/train 脚本所在；动态定位免硬编码）。"""
    import mmdet3d  # noqa: PLC0415  # 延迟 import：dry-run 前置步骤不依赖

    return Path(mmdet3d.__file__).resolve().parent / ".mim" / "tools"


def prepare_kitti_imagesets(
    kitti_root: Path = DEFAULT_KITTI_ROOT,
    n_train: int = 3712,
    n_val: int = 3769,
    n_test: int = 7518,
) -> Path:
    """ImageSets train 000000-003711 / val 003712-007480 / test 占位（内容确定，幂等）。"""
    sets_dir = kitti_root / "ImageSets"
    sets_dir.mkdir(parents=True, exist_ok=True)
    (sets_dir / "train.txt").write_text(
        "".join(f"{i:06d}\n" for i in range(n_train)), encoding="utf-8"
    )
    (sets_dir / "val.txt").write_text(
        "".join(f"{i:06d}\n" for i in range(n_train, n_train + n_val)), encoding="utf-8"
    )
    (sets_dir / "test.txt").write_text(
        "".join(f"{i:06d}\n" for i in range(n_test)), encoding="utf-8"
    )
    return sets_dir


def run_create_data(kitti_root: Path = DEFAULT_KITTI_ROOT) -> None:
    """mmdet3d create_data 幂等封装：infos pkl×5 + dbinfos 齐全即跳过（首跑 30-60 分钟）。

    教训（v0.4 实测）：.mim/tools 无 __init__.py → 必须 cwd=.mim/tools +
    PYTHONPATH=$PWD 命名空间包导入；不得用 `| tail` 管道（吞退出码假象），
    直接 subprocess check 退出码。
    """
    required = [
        "kitti_infos_train.pkl",
        "kitti_infos_val.pkl",
        "kitti_infos_test.pkl",
        "kitti_infos_trainval.pkl",
        "kitti_dbinfos_train.pkl",
    ]
    velodyne = kitti_root / "training" / "velodyne_reduced"
    if all((kitti_root / f).is_file() for f in required) and any(velodyne.iterdir()):
        return  # 幂等跳过
    cmd = [
        sys.executable,
        "create_data.py",  # .mim/tools 平级（无 tools/ 子目录；dry-run 实测教训）
        "kitti",
        "--root-path",
        str(kitti_root),
        "--out-dir",
        str(kitti_root),
        "--extra-tag",
        "kitti",
    ]
    env = {**os.environ, "PYTHONPATH": str(mim_tools_dir())}
    subprocess.run(cmd, cwd=mim_tools_dir(), env=env, check=True)


def _read_data_list(src: Path) -> tuple[dict | None, list]:
    """infos pkl → (metainfo, data_list)：v1.x 格式 {metainfo, data_list} 与旧 list 双兼容。"""
    with open(src, "rb") as f:
        data = pickle.load(f)  # noqa: S301  # 自产数据格式
    if isinstance(data, dict) and "data_list" in data:
        return data.get("metainfo"), data["data_list"]
    if not isinstance(data, list):
        raise ValueError(f"未知 infos pkl 格式: {type(data).__name__}")
    return None, data


def subsample_infos(src: Path, out: Path, n: int) -> int:
    """infos pkl 按 sample_idx 升序取前 n 帧 → 小样本 pkl（确定性；n 超全集取全）。

    v1.x 格式（update_infos_to_v2 产物）的 metainfo（categories 类表）原样保留。
    """
    metainfo, data_list = _read_data_list(src)
    keyed = sorted(data_list, key=lambda i: int(i["sample_idx"]))
    payload: dict | list = (
        {"metainfo": metainfo, "data_list": keyed[:n]}
        if metainfo is not None
        else keyed[:n]
    )
    out.write_bytes(pickle.dumps(payload))
    return len(keyed[:n])


def build_finetune_config(
    official_config: Path,
    kitti_root: Path,
    train_infos: Path,
    val_infos: Path,
    checkpoint: Path,
    out_path: Path,
    batch_size: int = 2,
    epochs: int = 4,
    lr: float = 0.0003,
    accumulative_counts: int = 1,
) -> Path:
    """官方 config → 自足微调 config（dump 全展开，无 _base_ 引用）。

    改造清单（v0.4 P3 定案）：data_root/ann_file 绝对化指 subsample pkl；
    删 db_sampler + ObjectSample（GT-AUG 关，DB 采样在小样本上反而噪声）；
    param_scheduler 整段替换 LinearLR 预热 + 恒定（官方 CosineAnnealing 的
    T_max 绑 epoch_num=80，只改 epoch 会 LR 崩溃）；lr 0.0003 微调档；
    batch 2（12GB 档，OOM 降级 batch 1 + accumulative_counts=2）；epoch 4；
    load_from 官方 pth（热启）；val_interval=1（冒烟看曲线）。
    """
    from mmengine.config import Config  # noqa: PLC0415

    # 全路径绝对化：训练 cwd=.mim/tools，相对路径（WEIGHTS_DIR/FINETUNE_DIR）必炸
    kitti_root = kitti_root.resolve()
    train_infos = train_infos.resolve()
    val_infos = val_infos.resolve()
    checkpoint = checkpoint.resolve()
    out_path = out_path.resolve()

    cfg = Config.fromfile(str(official_config))
    cfg.data_root = f"{kitti_root}/"
    cfg.lr = lr
    cfg.epoch_num = epochs
    cfg.load_from = str(checkpoint)
    # 内置 KittiMetric val 全关（val_begin=epochs+1）：其 IoU 依赖 numba CUDA
    # kernel（rotate_iou.py import 期编译），numba 0.67 × CUDA 13 下 Signature
    # mismatch 环境级不可用（v0.4 P3 实测）。val_interval 大数无效——mmengine
    # 强制最终 epoch（_epoch == _max_epochs）跑一次 val；val_begin 越界才彻底关。
    # 评测由 smoke_kitti 自写 40-point 口径（kitti_official_ap，纯 numpy）在
    # P3-5 承担，loss 曲线监控训练。
    cfg.train_cfg = dict(
        by_epoch=True,
        max_epochs=epochs,
        val_begin=epochs + 1,
        val_interval=1000,
    )
    cfg.optim_wrapper.optimizer.lr = lr
    cfg.optim_wrapper.accumulative_counts = accumulative_counts
    cfg.train_dataloader.batch_size = batch_size
    # RepeatDataset times 保留官方 2（少改少风险；300 帧 × 4 epoch × 2 ≈ 1200 iter）
    cfg.train_dataloader.dataset.dataset.ann_file = str(train_infos)
    cfg.val_dataloader.dataset.ann_file = str(val_infos)
    cfg.test_dataloader.dataset.ann_file = str(val_infos)
    cfg.val_evaluator.ann_file = str(val_infos)
    cfg.test_evaluator.ann_file = str(val_infos)
    # 嵌套 data_root：Config 合并时变量引用已求值成 'data/kitti/'，顶层
    # cfg.data_root 重绑不会传播到 dataset 内嵌键——显式逐处改
    cfg.train_dataloader.dataset.dataset.data_root = f"{kitti_root}/"
    cfg.val_dataloader.dataset.data_root = f"{kitti_root}/"
    cfg.test_dataloader.dataset.data_root = f"{kitti_root}/"
    # GT-AUG 关：删 db_sampler 与 ObjectSample 管线步（小样本 DB 采样收益低 + 省 30 分钟建库）
    # （mmengine Config 无 __delitem__，删除走 pop）
    cfg.pop("db_sampler", None)
    # train_pipeline 重绑顶层键不会同步 dataset.dataset.pipeline（官方 config
    # `dataset=dict(dataset=dict(pipeline=train_pipeline))` 合并后同一 list 两处
    # 引用）——两处都显式赋过滤后的列表
    pipeline = [p for p in cfg.train_pipeline if p.get("type") != "ObjectSample"]
    cfg.train_pipeline = pipeline
    cfg.train_dataloader.dataset.dataset.pipeline = pipeline
    # LR 调度：LinearLR 预热 + ConstantLR 恒定（官方 T_max 绑 epoch_num 的教训）
    # 纯 iter-based（begin/end 直接 iter 数）——不得加 convert_to_iter_based
    # （mmengine 断言仅 epoch-based scheduler 可转 iter-based，实测炸）
    times = int(cfg.train_dataloader.dataset.times)
    _, train_list = _read_data_list(train_infos)
    n_train = len(train_list)
    iters_per_epoch = math.ceil(n_train / batch_size)
    total_iters = iters_per_epoch * epochs * times
    warmup_iters = min(50, total_iters // 4)
    cfg.param_scheduler = [
        dict(
            type="LinearLR",
            start_factor=1e-3,
            by_epoch=False,
            begin=0,
            end=warmup_iters,
        ),
        dict(
            type="ConstantLR",
            factor=1.0,
            by_epoch=False,
            begin=warmup_iters,
            end=total_iters,
        ),
    ]
    cfg.dump(str(out_path))
    return out_path


def build_train_cmd(config_path: Path, work_dir: Path, resume: bool = False) -> list[str]:
    """单卡训练命令（调用方 cwd=mim_tools_dir() + PYTHONPATH=$PWD）。

    config/work_dir 一律 resolve：训练 cwd=.mim/tools，相对路径必炸（dry-run 实测）。
    """
    cmd = [
        sys.executable,
        "train.py",
        str(config_path.resolve()),
        "--work-dir",
        str(work_dir.resolve()),
    ]
    if resume:
        cmd.append("--resume")
    return cmd


def train(config_path: Path, work_dir: Path, resume: bool = False) -> int:
    """subprocess 跑训练（需 GPU）；返回 returncode（非零抛 CalledProcessError）。"""
    cmd = build_train_cmd(config_path, work_dir, resume=resume)
    env = {**os.environ, "PYTHONPATH": str(mim_tools_dir())}
    proc = subprocess.run(cmd, cwd=mim_tools_dir(), env=env, check=True)
    return proc.returncode


def prepare_finetune(
    n_train: int = 300,
    n_val: int = 40,
    epochs: int = 4,
    batch_size: int = 2,
    lr: float = 0.0003,
    accumulative_counts: int = 1,
) -> tuple[Path, Path]:
    """数据准备 + config 生成 → (finetune_config, work_dir)；纯 CPU 可跑（dry-run）。"""
    kitti_root = DEFAULT_KITTI_ROOT
    prepare_kitti_imagesets(kitti_root)
    run_create_data(kitti_root)

    out = FINETUNE_DIR
    out.mkdir(parents=True, exist_ok=True)
    train_infos = out / f"kitti_infos_train_{n_train}.pkl"
    val_infos = out / f"kitti_infos_val_{n_val}.pkl"
    subsample_infos(kitti_root / "kitti_infos_train.pkl", train_infos, n_train)
    subsample_infos(kitti_root / "kitti_infos_val.pkl", val_infos, n_val)

    entry = DETECTOR3D_NAMES["pointpillars_kitti"]
    official = MMDET3D_CONFIG_DIR / entry["config"]
    checkpoint = WEIGHTS_DIR / entry["weights_dir"] / entry["checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"官方权重缺失: {checkpoint}")
    config_path = out / "finetune_config.py"
    build_finetune_config(
        official_config=official,
        kitti_root=kitti_root,
        train_infos=train_infos,
        val_infos=val_infos,
        checkpoint=checkpoint,
        out_path=config_path,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        accumulative_counts=accumulative_counts,
    )
    print(f"✓ 微调 config: {config_path}")
    print(f"  train={n_train} 帧 / val={n_val} 帧 / epoch={epochs} / batch={batch_size} / lr={lr}")
    print(f"  load_from={checkpoint.name}（官方热启）")
    return config_path, out


def main() -> None:
    parser = argparse.ArgumentParser(description="KITTI 3D 微调管线（v0.4 P3）")
    parser.add_argument("--n-train", type=int, default=300, help="冒烟训练帧数")
    parser.add_argument("--n-val", type=int, default=40, help="冒烟验证帧数")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch", type=int, default=2, help="12GB 档 2；OOM → 1 + --accum 2")
    parser.add_argument("--accum", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--lr", type=float, default=0.0003, help="微调档学习率")
    parser.add_argument("--resume", action="store_true", help="从 work_dir 断点续训")
    parser.add_argument(
        "--dry-run", action="store_true", help="仅数据准备 + config 生成（CPU 可跑）"
    )
    args = parser.parse_args()

    config_path, work_dir = prepare_finetune(
        n_train=args.n_train,
        n_val=args.n_val,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        accumulative_counts=args.accum,
    )
    if args.dry_run:
        return
    train(config_path, work_dir, resume=args.resume)


if __name__ == "__main__":
    main()
