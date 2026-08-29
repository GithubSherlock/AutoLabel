"""cli_common 公共层测试：collect_images 收集语义（含递归兜底）。

- 单文件 / 顶层目录 / batch=False 提示
- 2026-08-29 递归兜底：LLM 路径引导可能把 source 指向数据集根目录而非图像
  目录（如 KITTI training/ 的图像在 training/image_2/）——顶层无图时递归
  子目录收集；顶层有图时保持历史语义（不递归）
"""

from __future__ import annotations

from pathlib import Path

from auto2dlabel.cli_common import collect_images


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_collect_images_single_file(tmp_path: Path) -> None:
    """单文件 → 只收该文件（图片扩展名）。"""
    img = _touch(tmp_path / "a.png")
    assert collect_images(img, batch=False) == [img]


def test_collect_images_single_file_non_image(tmp_path: Path) -> None:
    """单文件非图片扩展名 → []。"""
    txt = _touch(tmp_path / "a.txt")
    assert collect_images(txt, batch=False) == []


def test_collect_images_top_level_only(tmp_path: Path) -> None:
    """顶层有图 → 只收顶层（历史语义：不递归子目录）。"""
    top = _touch(tmp_path / "a.png")
    _touch(tmp_path / "sub" / "b.png")
    assert collect_images(tmp_path, batch=True) == [top]


def test_collect_images_recursive_fallback(tmp_path: Path) -> None:
    """顶层无图 + 子目录有图 → 递归收集（KITTI training/ → image_2/ 场景）。"""
    nested = _touch(tmp_path / "image_2" / "000001.png")
    _touch(tmp_path / "calib" / "000001.txt")  # 非图片不混入
    assert collect_images(tmp_path, batch=True) == [nested]


def test_collect_images_recursive_multiple_sorted(tmp_path: Path) -> None:
    """递归兜底：多子目录多图 → 全局排序确定序。"""
    b = _touch(tmp_path / "b" / "2.png")
    a = _touch(tmp_path / "a" / "1.png")
    assert collect_images(tmp_path, batch=True) == [a, b]


def test_collect_images_recursive_no_images(tmp_path: Path) -> None:
    """顶层与子目录均无图 → []（与递归前行为一致）。"""
    _touch(tmp_path / "labels" / "a.txt")
    assert collect_images(tmp_path, batch=True) == []


def test_collect_images_dir_without_batch(tmp_path: Path) -> None:
    """目录模式需 --batch（batch=False → []）。"""
    _touch(tmp_path / "a.png")
    assert collect_images(tmp_path, batch=False) == []


def test_collect_images_missing_file(tmp_path: Path) -> None:
    """不存在的路径 → []。"""
    assert collect_images(tmp_path / "nope.png", batch=False) == []
