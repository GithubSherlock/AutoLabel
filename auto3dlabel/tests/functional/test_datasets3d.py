"""3D 数据集知识层测试（configs/datasets.py + planner3d 注入）。

2026-08-29：auto3dlabel/configs/datasets.py 曾为空占位——补全为
内置路径表（引用 configs/{kitti,nuscenes}.py 常量，单一事实源）+ 自建注册
（复用 auto2dlabel.configs.datasets 读写，独立 yaml）+ planner3d prompt 注入，
使 3D chat 达到 2D 一样的数据集识别效果。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from auto3dlabel.configs.datasets import (
    DATASET_DIRS,
    DATASET_ENV,
    format_datasets_summary,
    register_user_dataset,
    remove_user_dataset,
    resolve_dataset_dir,
)
from auto3dlabel.configs.kitti import DEFAULT_KITTI_ROOT
from auto3dlabel.configs.nuscenes import DEFAULT_NUSCENES_ROOT


def test_dataset_dirs_references_configs_constants() -> None:
    """内置路径表引用 kitti/nuscenes 常量——单一事实源（改常量不漏表）。"""
    assert DATASET_DIRS["kitti"].path == DEFAULT_KITTI_ROOT
    assert DATASET_DIRS["nuscenes_mini"].path == DEFAULT_NUSCENES_ROOT
    assert set(DATASET_DIRS) == {"kitti", "nuscenes_mini"}
    assert DATASET_ENV == {"kitti": "KITTI_OBJECT_ROOT", "nuscenes_mini": "NUSCENES_ROOT"}


def test_resolve_dataset_dir_default() -> None:
    """默认路径 = configs 常量值（绝对路径）。"""
    for name, info in DATASET_DIRS.items():
        assert resolve_dataset_dir(name) == info.path
        assert info.path.is_absolute()


def test_resolve_dataset_dir_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """env 覆盖（KITTI_OBJECT_ROOT / NUSCENES_ROOT）优先于默认路径。"""
    monkeypatch.setenv("KITTI_OBJECT_ROOT", "/tmp/fake_kitti_3d")
    monkeypatch.setenv("NUSCENES_ROOT", "/tmp/fake_nus_3d")
    assert resolve_dataset_dir("kitti") == Path("/tmp/fake_kitti_3d")
    assert resolve_dataset_dir("nuscenes_mini") == Path("/tmp/fake_nus_3d")


def test_resolve_dataset_dir_unknown() -> None:
    """未知数据集名 → 明确 KeyError（列可选名）。"""
    with pytest.raises(KeyError, match="未知数据集"):
        resolve_dataset_dir("not_a_dataset")


def test_format_datasets_summary_contains_all() -> None:
    """摘要含全部内置数据集：名 / 绝对路径 / 任务描述 / env 提示。"""
    summary = format_datasets_summary()
    for name, info in DATASET_DIRS.items():
        assert name in summary
        assert str(info.path) in summary
        assert info.task in summary
    assert "KITTI_OBJECT_ROOT" in summary
    assert "内置数据集" in summary


def test_format_datasets_summary_env_reflects_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """摘要运行时取 env 当前值——LLM 不会拿到过期路径。"""
    monkeypatch.setenv("KITTI_OBJECT_ROOT", "/tmp/fake_kitti_3d")
    assert "/tmp/fake_kitti_3d" in format_datasets_summary()


def test_register_remove_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """自建注册 → 落盘 → 摘要合并 → 移除（写独立 3D yaml，不碰 2D 文件）。"""
    cfg = tmp_path / "user_datasets.yaml"
    monkeypatch.setattr("auto3dlabel.configs.datasets.USER_DATASETS_FILE", cfg)
    data_dir = tmp_path / "mydata3d"
    data_dir.mkdir()
    info = register_user_dataset(
        "mydata3d", str(data_dir), subdirs=["training"], task="3D 检测", note="自建",
    )
    assert info.path == data_dir and info.task == "3D 检测"
    assert cfg.exists()  # 落盘到 3D 自己的 yaml
    summary = format_datasets_summary()
    assert "用户自建 3D 数据集" in summary and "mydata3d" in summary
    assert str(data_dir) in summary
    assert remove_user_dataset("mydata3d") is True
    assert remove_user_dataset("mydata3d") is False


def test_register_invalid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """非法注册拒绝：空名 / 与 3D 内置重名 / 路径不存在。"""
    cfg = tmp_path / "user_datasets.yaml"
    monkeypatch.setattr("auto3dlabel.configs.datasets.USER_DATASETS_FILE", cfg)
    data_dir = tmp_path / "mydata3d"
    data_dir.mkdir()
    with pytest.raises(ValueError, match="不能为空"):
        register_user_dataset("  ", str(data_dir))
    with pytest.raises(ValueError, match="重名"):
        register_user_dataset("kitti", str(data_dir))  # 3D 内置表检查
    with pytest.raises(ValueError, match="不存在"):
        register_user_dataset("mydata3d", str(tmp_path / "nope"))


def test_planner3d_injects_datasets_summary() -> None:
    """planner3d prompt 注入摘要 + 数据集规则（LLM 可识别数据集名/路径语义）。"""
    from auto3dlabel.agent.planner3d import _PLANNER3D_SYSTEM_PROMPT

    assert "内置数据集" in _PLANNER3D_SYSTEM_PROMPT
    assert "training" in _PLANNER3D_SYSTEM_PROMPT
    assert "KITTI 数据集/对 KITTI" in _PLANNER3D_SYSTEM_PROMPT
    assert str(DEFAULT_KITTI_ROOT) in _PLANNER3D_SYSTEM_PROMPT
