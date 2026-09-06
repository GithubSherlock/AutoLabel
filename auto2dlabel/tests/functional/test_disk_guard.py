"""磁盘保护回归测试（2026-09-03 实测：视频跟踪逐帧 PNG 写爆盘级联 Errno 122）。

F3 两项修复：
- chat_command viz 默认 False（chat 跟踪不产逐帧 PNG；auto2dlabel chat 经
  --no-viz 显式传入，老行为不变）
- TrackingTool 逐帧 PNG 写前余量检查（< MIN_FREE_DISK_GB 黄字降级跳过）

F5 任务级输出预算预检（2026-09-03 实测：XFS 配额盘上长视频中途耗尽，
逐帧 JSON 写入 EDQUOT 裸 traceback）：
- estimate_tracking_output_gb 按实测基线估算（抽帧 JPG+JSON 0.22MB/帧、
  viz PNG 3MB/帧、成片视频 ×1.5）
- run() 入口预检：预算 ×2 安全系数 > 余量 - 安全边际 → ValueError 提前失败
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from auto2dlabel.cli_commands import chat_command
from auto2dlabel.tools.tracking import (
    MIN_FREE_DISK_GB,
    TrackingTool,
    disk_free_gb,
    estimate_tracking_output_gb,
)


def _usage(free: int) -> SimpleNamespace:
    """disk_usage 返回 namedtuple（.free 属性），用 SimpleNamespace 模拟。"""
    return SimpleNamespace(total=100, used=50, free=free)


def test_disk_free_gb_converts_bytes_to_gb() -> None:
    """disk_usage free 字节 → GB（/1024**3）。"""
    with patch("shutil.disk_usage", return_value=_usage(10 * 1024**3)):
        assert disk_free_gb(Path(".")) == 10.0


def test_disk_free_gb_accepts_nonexistent_path(tmp_path: Path) -> None:
    """不存在路径不抛错（真实实现：向上解析到最近已存在祖先）。

    2026-09-03 实测回归：shutil.disk_usage 对不存在路径抛 FileNotFoundError，
    曾致 output_dir 尚未创建时预检崩溃——mock 版本掩盖了该缺陷。
    """
    nonexistent = tmp_path / "not_created_yet" / "out"
    assert disk_free_gb(nonexistent) > 0  # 解析到 tmp_path，不抛错


def test_min_free_disk_gb_threshold_sane() -> None:
    """阈值常量：1GB 级（写爆级联实测后的保护线）。"""
    assert 0.1 < MIN_FREE_DISK_GB <= 10


def test_chat_command_viz_defaults_false() -> None:
    """chat 跟踪默认不产逐帧 PNG（磁盘保护）；auto2dlabel chat 显式传 viz 不受影响。"""
    sig = inspect.signature(chat_command)
    assert sig.parameters["viz"].default is False


def test_estimate_output_no_viz() -> None:
    """viz 关：100 帧 ≈ 抽帧 JPG + 逐帧 JSON（0.22MB/帧）。"""
    est = estimate_tracking_output_gb(100, viz=False)
    assert est == pytest.approx(100 * 0.22 / 1024)


def test_estimate_output_with_viz() -> None:
    """viz 开：逐帧 PNG 3MB/帧 是大头（2026-09-02 实测 2.7MB/帧）。"""
    est = estimate_tracking_output_gb(100, viz=True)
    assert est == pytest.approx(100 * 3.22 / 1024)


def test_estimate_output_video_movie() -> None:
    """成片视频 = 源视频 ×1.5（mp4v 重编码，帧目录不产片）。"""
    est = estimate_tracking_output_gb(0, viz=False, video_mb=100.0)
    assert est == pytest.approx(150.0 / 1024)


def test_run_rejects_budget_over_free(tmp_path: Path) -> None:
    """入口预检：长视频预算 ×2 > 余量 → ValueError 提前失败（不跑十几分钟再崩）。

    预检在 collect_frames / 模型加载之前，构造与调用零真实权重。
    """
    fake_mp4 = tmp_path / "fake.mp4"
    fake_mp4.write_bytes(b"x" * 1024)
    cap_mock = MagicMock()
    cap_mock.isOpened.return_value = True
    cap_mock.get.return_value = 10000  # 10000 帧 ≈ 2.1GB 预算
    with patch("cv2.VideoCapture", return_value=cap_mock):
        with patch("auto2dlabel.tools.tracking.disk_free_gb", return_value=0.5):
            with pytest.raises(ValueError, match="磁盘余量不足"):
                TrackingTool(viz=False).forward(str(fake_mp4), ["person"])
