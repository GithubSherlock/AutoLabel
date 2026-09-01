"""test_train3d：P3 微调管线纯函数（build_finetune_config 需 mmdet3d 环境，skipif 守卫）。

数据准备/截取/命令组装零权重零网络；config 改造对官方 config 文本级断言
（不加载模型——Config.fromfile 只做 dict 合并）。
"""

from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path

import pytest

from auto3dlabel.configs.kitti import MMDET3D_CONFIG_DIR
from auto3dlabel.tools.train3d import (
    build_finetune_config,
    build_train_cmd,
    prepare_kitti_imagesets,
    subsample_infos,
)

MMDET3D_INSTALLED = importlib.util.find_spec("mmdet3d") is not None


def test_prepare_kitti_imagesets(tmp_path: Path) -> None:
    """ImageSets 三文件：train 000000-003711 / val 003712-007480 / test 占位（幂等）。"""
    sets_dir = prepare_kitti_imagesets(tmp_path, n_train=5, n_val=3, n_test=4)
    assert (sets_dir / "train.txt").read_text().splitlines() == [
        "000000", "000001", "000002", "000003", "000004",
    ]
    assert (sets_dir / "val.txt").read_text().splitlines() == [
        "000005", "000006", "000007",
    ]
    assert (sets_dir / "test.txt").read_text().splitlines() == [
        "000000", "000001", "000002", "000003",
    ]
    # 幂等：重复调用内容确定（覆盖写）
    prepare_kitti_imagesets(tmp_path, n_train=5, n_val=3, n_test=4)
    assert len((sets_dir / "train.txt").read_text().splitlines()) == 5


def _write_infos(path: Path, idxes: list[str]) -> None:
    path.write_bytes(pickle.dumps([{"sample_idx": i} for i in idxes]))


def test_subsample_infos_sorted_and_truncated(tmp_path: Path) -> None:
    """按 sample_idx 升序取前 n（乱序输入也确定）；n 超全集取全。"""
    src = tmp_path / "src.pkl"
    _write_infos(src, ["000005", "000001", "000009", "000003"])
    out = tmp_path / "sub.pkl"
    assert subsample_infos(src, out, 2) == 2
    with open(out, "rb") as f:
        got = pickle.load(f)  # noqa: S301
    assert [i["sample_idx"] for i in got] == ["000001", "000003"]

    assert subsample_infos(src, tmp_path / "all.pkl", 99) == 4  # 超全集取全


def test_subsample_infos_v1x_keeps_metainfo(tmp_path: Path) -> None:
    """v1.x 格式 {metainfo, data_list}：categories 类表原样保留（KittiDataset 必读）。"""
    src = tmp_path / "src.pkl"
    src.write_bytes(pickle.dumps({
        "metainfo": {"categories": {"Car": 0, "Pedestrian": 1}, "dataset": "kitti"},
        "data_list": [{"sample_idx": 9}, {"sample_idx": 2}, {"sample_idx": 5}],
    }))
    out = tmp_path / "sub.pkl"
    assert subsample_infos(src, out, 2) == 2
    with open(out, "rb") as f:
        got = pickle.load(f)  # noqa: S301
    assert got["metainfo"]["categories"] == {"Car": 0, "Pedestrian": 1}
    assert [i["sample_idx"] for i in got["data_list"]] == [2, 5]


def test_build_train_cmd() -> None:
    """训练命令：train.py（.mim/tools 平级）+ config 绝对路径 + --work-dir
    （cwd/PYTHONPATH 由 train() 设）。"""
    cmd = build_train_cmd(Path("/tmp/finetune_config.py"), Path("/tmp/wd"))
    assert Path(cmd[0]).name in ("python", "python3")  # sys.executable 全路径
    assert "train.py" in cmd  # 不是 tools/train.py（.mim/tools 无 tools/ 子目录）
    assert "/tmp/finetune_config.py" in cmd
    assert cmd[cmd.index("--work-dir") + 1] == "/tmp/wd"
    assert "--resume" not in cmd
    assert "--resume" in build_train_cmd(
        Path("/tmp/c.py"), Path("/tmp/wd"), resume=True
    )


@pytest.mark.skipif(not MMDET3D_INSTALLED, reason="mmdet3d 未装，config 改造跳过")
def test_build_finetune_config_rewrites(tmp_path: Path) -> None:
    """官方 config → 自足微调 config：GT-AUG 关 / LR 调度替换 / 热启 / 小样本 pkl。"""
    from mmengine.config import Config

    official = (
        MMDET3D_CONFIG_DIR
        / "configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py"
    )
    train_infos = tmp_path / "train.pkl"
    _write_infos(train_infos, [f"{i:06d}" for i in range(300)])
    val_infos = tmp_path / "val.pkl"
    _write_infos(val_infos, [f"{i:06d}" for i in range(40)])
    # v1.x 格式（metainfo 保留）下的 n_train 读取（LR 调度 total_iters 计算依赖）
    out_path = tmp_path / "finetune_config.py"

    built = build_finetune_config(
        official_config=official,
        kitti_root=tmp_path / "kitti",
        train_infos=train_infos,
        val_infos=val_infos,
        checkpoint=tmp_path / "official.pth",
        out_path=out_path,
        batch_size=2,
        epochs=4,
        lr=0.0003,
    )
    assert built == out_path

    cfg = Config.fromfile(str(out_path))  # dump 后自足可读回（无 _base_ 依赖）
    assert cfg.data_root == f"{tmp_path / 'kitti'}/"
    assert cfg.load_from == str(tmp_path / "official.pth")
    assert cfg.epoch_num == 4
    assert cfg.optim_wrapper.optimizer.lr == 0.0003
    assert cfg.train_dataloader.batch_size == 2
    assert cfg.train_dataloader.dataset.dataset.ann_file == str(train_infos)
    assert cfg.val_dataloader.dataset.ann_file == str(val_infos)
    assert cfg.val_evaluator.ann_file == str(val_infos)
    # 内置 KittiMetric val 全关（val_begin 越界——mmengine 强制最终 epoch val，
    # val_interval 大数无效；numba CUDA kernel 环境级不可用）
    assert cfg.train_cfg == {
        "by_epoch": True,
        "max_epochs": 4,
        "val_begin": 5,
        "val_interval": 1000,
    }
    # GT-AUG 关：db_sampler 删除 + ObjectSample 管线步移除
    assert not hasattr(cfg, "db_sampler")
    assert "ObjectSample" not in [p.get("type") for p in cfg.train_pipeline]
    # 嵌套 pipeline 同步（官方 config dataset=dict(dataset=dict(pipeline=train_pipeline))
    # 合并后同一 list 两处引用——单改顶层键会漏，训练时仍跑 DB 采样）
    nested = cfg.train_dataloader.dataset.dataset.pipeline
    assert "ObjectSample" not in [p.get("type") for p in nested]
    assert len(nested) == len(cfg.train_pipeline)
    # dump 文本级回归：无 db_sampler / ObjectSample 残留（内嵌 dict 一并消失）
    text = out_path.read_text(encoding="utf-8")
    assert "db_sampler" not in text
    assert "ObjectSample" not in text
    # test_evaluator 与三处嵌套 data_root 一并改造（训练 cwd=.mim/tools 下相对路径必炸）
    assert cfg.test_evaluator.ann_file == str(val_infos)
    for dataset in (
        cfg.train_dataloader.dataset.dataset,
        cfg.val_dataloader.dataset,
        cfg.test_dataloader.dataset,
    ):
        assert dataset.data_root == f"{tmp_path / 'kitti'}/"
    # 进入 config 的路径全绝对（load_from / ann_file×4 / data_root）
    assert Path(cfg.load_from).is_absolute()
    assert Path(cfg.train_dataloader.dataset.dataset.ann_file).is_absolute()
    assert Path(cfg.val_evaluator.ann_file).is_absolute()
    # LR 调度：LinearLR 预热 + ConstantLR 恒定（官方 T_max 绑 epoch_num 的教训）
    assert [s["type"] for s in cfg.param_scheduler] == ["LinearLR", "ConstantLR"]
    linear, const = cfg.param_scheduler
    assert linear["start_factor"] == 1e-3 and linear["end"] == 50  # warmup 50 iter
    assert const["factor"] == 1.0
    # 300 帧 / batch 2 / 4 epoch / RepeatDataset times 2 → 150×4×2 = 1200 iter
    assert const["end"] == 1200
