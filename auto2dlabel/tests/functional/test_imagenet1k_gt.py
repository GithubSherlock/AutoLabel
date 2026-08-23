"""ImageNet1k（ILSVRC2012 val）分类扩展测试 —— GT 解析 / 分层抽样 / 幂等解压 / 元数据。

零真实权重铁律与真实归档无关：GT 解析、成员抽样、幂等解压用 tmp_path 合成
tar（3 类 × 2 图微型夹具）；仅 meta.mat / GT 全文覆盖测试依赖真实 devkit
（skipif 归档缺失）。
"""

from __future__ import annotations

import tarfile
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

import auto2dlabel.benchmarks.datasets as ds
from auto2dlabel.benchmarks.datasets import (
    ensure_imagenet1k_val,
    load_imagenet1k_ground_truth,
    load_imagenet1k_meta,
    parse_imagenet1k_validation_gt,
    select_imagenet1k_members,
)

# ---------- tmp_path 微型夹具（零真实归档） ----------

def _make_devkit_tar(tmp_path: Path, gt_lines: list[str]) -> Path:
    """合成 devkit tar.gz（validation_ground_truth.txt 成员）。"""
    p = tmp_path / "devkit.tar.gz"
    data = ("\n".join(gt_lines) + "\n").encode("utf-8")
    with tarfile.open(p, "w:gz") as tf:
        info = tarfile.TarInfo("ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt")
        info.size = len(data)
        tf.addfile(info, BytesIO(data))
    return p


def _make_val_tar(tmp_path: Path, names: list[str]) -> Path:
    """合成纯 val tar（扁平 JPEG 成员，内容无关）。"""
    p = tmp_path / "val.tar"
    with tarfile.open(p, "w") as tf:
        for n in names:
            info = tarfile.TarInfo(n)
            info.size = 3
            tf.addfile(info, BytesIO(b"abc"))
    return p


# ---------- GT 解析纯函数 ----------

def test_parse_validation_gt(tmp_path: Path) -> None:
    """"ILSVRC2012_val_00000001.JPEG 490" → {文件名: 类 ID}。"""
    devkit = _make_devkit_tar(tmp_path, [
        "ILSVRC2012_val_00000001.JPEG 490",
        "ILSVRC2012_val_00000002.JPEG 361",
    ])
    gt = parse_imagenet1k_validation_gt(devkit_archive=devkit)
    assert gt == {"ILSVRC2012_val_00000001.JPEG": 490,
                  "ILSVRC2012_val_00000002.JPEG": 361}


def test_parse_validation_gt_single_column(tmp_path: Path) -> None:
    """单列类 ID（行序即图序，本归档副本格式）→ 文件名按行号推导。"""
    devkit = _make_devkit_tar(tmp_path, ["490", "361"])
    gt = parse_imagenet1k_validation_gt(devkit_archive=devkit)
    assert gt == {"ILSVRC2012_val_00000001.JPEG": 490,
                  "ILSVRC2012_val_00000002.JPEG": 361}


def test_parse_validation_gt_missing_archive() -> None:
    """归档缺失 → FileNotFoundError。"""
    with pytest.raises(FileNotFoundError):
        parse_imagenet1k_validation_gt(devkit_archive=Path("/nonexistent.tar.gz"))


def test_parse_validation_gt_missing_member(tmp_path: Path) -> None:
    """devkit 内无 GT 成员 → FileNotFoundError。"""
    p = tmp_path / "empty.tar.gz"
    with tarfile.open(p, "w:gz") as tf:
        info = tarfile.TarInfo("other.txt")
        info.size = 1
        tf.addfile(info, BytesIO(b"x"))
    with pytest.raises(FileNotFoundError):
        parse_imagenet1k_validation_gt(devkit_archive=p)


# ---------- 分层抽样纯函数 ----------

def test_select_members_stratified(tmp_path: Path) -> None:
    """3 类 × 2 图：per_class=1 → 每类取文件名排序前 1（共 3）；per_class=0 → 全量。"""
    names = [
        "ILSVRC2012_val_00000001.JPEG",  # cls 1
        "ILSVRC2012_val_00000002.JPEG",  # cls 1
        "ILSVRC2012_val_00000003.JPEG",  # cls 2
        "ILSVRC2012_val_00000004.JPEG",  # cls 2
        "ILSVRC2012_val_00000005.JPEG",  # cls 3
        "ILSVRC2012_val_00000006.JPEG",  # cls 3
    ]
    tar = _make_val_tar(tmp_path, names)
    gt = {n: (i - 1) // 2 + 1 for i, n in enumerate(names, start=1)}  # 01,02→1; 03,04→2; 05,06→3

    picked1 = select_imagenet1k_members(tar, gt, per_class=1)
    assert picked1 == [
        "ILSVRC2012_val_00000001.JPEG",
        "ILSVRC2012_val_00000003.JPEG",
        "ILSVRC2012_val_00000005.JPEG",
    ]
    picked_all = select_imagenet1k_members(tar, gt, per_class=0)
    assert picked_all == sorted(names)


def test_select_members_ignores_unknown(tmp_path: Path) -> None:
    """GT 不含的成员跳过（不静默纳入）。"""
    tar = _make_val_tar(tmp_path, ["ILSVRC2012_val_00000001.JPEG", "stray.JPEG"])
    gt = {"ILSVRC2012_val_00000001.JPEG": 1}
    assert select_imagenet1k_members(tar, gt, per_class=1) == [
        "ILSVRC2012_val_00000001.JPEG"
    ]


# ---------- ensure_imagenet1k_val 幂等（monkeypatch 归档路径） ----------

@pytest.fixture()
def _mini_archives(tmp_path: Path, monkeypatch: Any) -> Path:
    """把模块级归档/DATASETS_ROOT 指向 tmp_path 微型夹具。"""
    gt_lines = [f"ILSVRC2012_val_{i:08d}.JPEG {(i - 1) // 2 + 1}" for i in range(1, 7)]
    devkit = _make_devkit_tar(tmp_path, gt_lines)
    val_tar = _make_val_tar(tmp_path, [f"ILSVRC2012_val_{i:08d}.JPEG" for i in range(1, 7)])
    monkeypatch.setattr(ds, "IMAGENET1K_DEVKIT", devkit)
    monkeypatch.setattr(ds, "IMAGENET1K_VAL_TAR", val_tar)
    monkeypatch.setattr(ds, "DATASETS_ROOT", tmp_path / "ds")
    return tmp_path


def test_ensure_imagenet1k_val_extracts_and_manifest(_mini_archives: Path) -> None:
    """首调解压 per_class 张 + 写 manifest；二调幂等跳过。"""
    dest = ensure_imagenet1k_val(per_class=1)
    jpegs = sorted(p.name for p in dest.iterdir() if p.suffix == ".JPEG")
    assert jpegs == ["ILSVRC2012_val_00000001.JPEG",  # 每类取文件名排序前 1
                     "ILSVRC2012_val_00000003.JPEG",
                     "ILSVRC2012_val_00000005.JPEG"]
    assert (dest / "_selected_pc1.txt").exists()

    # 二调：manifest 命中，全部存在 → 不重解压
    before = sorted(p.name for p in dest.iterdir())
    dest2 = ensure_imagenet1k_val(per_class=1)
    assert dest2 == dest
    assert sorted(p.name for p in dest.iterdir()) == before


def test_ensure_imagenet1k_val_resumes_missing(_mini_archives: Path) -> None:
    """manifest 在但文件缺失 → 只补缺。"""
    dest = ensure_imagenet1k_val(per_class=1)
    missing = dest / "ILSVRC2012_val_00000001.JPEG"
    missing.unlink()
    ensure_imagenet1k_val(per_class=1)
    assert missing.exists()


def test_ensure_imagenet1k_val_missing_archive(_mini_archives: Path) -> None:
    """归档缺失 → FileNotFoundError（manifest 未写）。"""
    monkeypatch_setattr = pytest.MonkeyPatch()
    monkeypatch_setattr.setattr(ds, "IMAGENET1K_VAL_TAR", Path("/nonexistent.tar"))
    with pytest.raises(FileNotFoundError):
        ensure_imagenet1k_val(per_class=1)
    monkeypatch_setattr.undo()


# ---------- load_imagenet1k_ground_truth ----------

def test_load_imagenet1k_gt_labels_and_truncation(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """扁平 JPEG → devkit GT → meta 英文名；manifest 跳过；max_images 跨类均匀。"""
    val_dir = tmp_path / "val"
    val_dir.mkdir()
    gt_orig = {f"ILSVRC2012_val_{i:08d}.JPEG": i % 2 + 1 for i in range(1, 5)}
    for name in gt_orig:
        (val_dir / name).write_bytes(b"x")
    (val_dir / "_selected_pc2.txt").write_text("manifest")
    (val_dir / "notes.md").write_text("x")  # 非图像跳过

    monkeypatch.setattr(ds, "load_imagenet1k_meta", lambda: {"1": "kit fox", "2": "goldfish"})
    monkeypatch.setattr(ds, "parse_imagenet1k_validation_gt", lambda: gt_orig)

    gt = load_imagenet1k_ground_truth(val_dir)
    assert len(gt) == 4
    assert gt["ILSVRC2012_val_00000001.JPEG"]["label"] == "goldfish"  # 1%2+1=2
    assert gt["ILSVRC2012_val_00000002.JPEG"]["label"] == "kit fox"   # 2%2+1=1

    # max_images 跨类均匀截断：4 图 → 2 图，两类各 1
    truncated = load_imagenet1k_ground_truth(val_dir, max_images=2)
    assert len(truncated) == 2
    labels = {v["label"] for v in truncated.values()}
    assert labels == {"kit fox", "goldfish"}


def test_load_imagenet1k_gt_unknown_id_fallback(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """meta 缺类 ID → 回退 class_{id}，不静默丢图。"""
    val_dir = tmp_path / "val"
    val_dir.mkdir()
    (val_dir / "ILSVRC2012_val_00000001.JPEG").write_bytes(b"x")
    monkeypatch.setattr(ds, "load_imagenet1k_meta", lambda: {})
    monkeypatch.setattr(
        ds, "parse_imagenet1k_validation_gt",
        lambda: {"ILSVRC2012_val_00000001.JPEG": 490},
    )
    gt = load_imagenet1k_ground_truth(val_dir)
    assert gt["ILSVRC2012_val_00000001.JPEG"]["label"] == "class_490"


# ---------- 真实 devkit 覆盖（skipif 归档缺失） ----------

@pytest.mark.skipif(
    not ds.IMAGENET1K_DEVKIT.exists(), reason="ILSVRC2012 devkit 归档缺失"
)
def test_parse_validation_gt_real_devkit() -> None:
    """真实 devkit：5 万行单列，行 1 → 00000001.JPEG → 490（已实测）。"""
    gt = parse_imagenet1k_validation_gt()
    assert len(gt) == 50000
    assert gt["ILSVRC2012_val_00000001.JPEG"] == 490


@pytest.mark.skipif(
    not ds.IMAGENET1K_DEVKIT.exists(), reason="ILSVRC2012 devkit 归档缺失"
)
def test_load_imagenet1k_meta_real_devkit() -> None:
    """真实 meta.mat：恰 1000 个 1k 类，ID 1 → kit fox。"""
    meta = load_imagenet1k_meta()
    assert len(meta) == 1000
    assert set(meta) == {str(i) for i in range(1, 1001)}
    assert meta["1"] == "kit fox"
