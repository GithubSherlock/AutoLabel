"""P3 进度/取消协议测试 —— TrackingTool progress_cb/cancel_event + CLI 行协议。

零真实权重（FakeDetModel + 合成帧序列/视频，沿用 test_tracking_tool.py 模式）。
覆盖：progress_cb (done,total) 序列、cancel_event → TrackingCancelledError
（ValueError 子类，既有 except ValueError 收敛路径零改动）、视频源取消后
finally 链 release 成片可读、取消后 MOT/成片不产出、cli_common 协议行
格式与 SIGTERM→KeyboardInterrupt handler。
"""

from __future__ import annotations

import os
import signal
import threading
from typing import Any

import cv2
import pytest

from auto2dlabel.cli_common import (
    AL_PROGRESS_PREFIX,
    install_sigterm_interrupt,
    print_al_progress,
)
from auto2dlabel.models.detection import DetectionResult
from auto2dlabel.tests import Path, np
from auto2dlabel.tools.tracking import TrackingCancelledError, TrackingTool


class FakeDetModel:
    """单目标逐帧右移（100×80 @ conf 0.9）：相邻帧 IoU ≥ 0.8，3+ 帧维持同一轨迹。"""

    def __init__(self) -> None:
        self.detect_calls: list[str] = []

    def detect(
        self,
        image_path: str,
        prompts: list[str],
        confidence_threshold: float = 0.3,
    ) -> list[DetectionResult]:
        self.detect_calls.append(image_path)
        i = len(self.detect_calls) - 1
        return [
            DetectionResult(
                x=10.0 + 4.0 * i,
                y=20.0,
                width=100.0,
                height=80.0,
                label=prompts[0],
                confidence=0.9,
            )
        ]


def _make_seq(d: Path, n: int = 3) -> Path:
    """合成 n 帧序列（128×128，白色矩形逐帧右移）。"""
    seq = d / "seq"
    seq.mkdir()
    for i in range(n):
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        frame[40:88, 30 + i * 4 : 80 + i * 4] = 255
        cv2.imwrite(str(seq / f"f{i}.png"), frame)
    return seq


def _make_video(d: Path, n: int = 3) -> Path:
    """cv2.VideoWriter 合成 n 帧 mp4（零真实权重，mp4v 编码）。"""
    video = d / "clip.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(video), fourcc, 10.0, (128, 128))
    assert writer.isOpened()
    for i in range(n):
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        frame[40:88, 30 + i * 4 : 80 + i * 4] = 255
        writer.write(frame)
    writer.release()
    return video


def test_progress_cb_sequence(tmp_path: Path, monkeypatch: Any) -> None:
    """progress_cb 收到 (done,total) 严格序列（每帧完成一次，末帧 done==total）。"""
    monkeypatch.chdir(tmp_path)
    seq = _make_seq(tmp_path, n=4)
    calls: list[tuple[int, int]] = []
    tool = TrackingTool(
        model=FakeDetModel(),
        batch_size=1,
        output_dir=str(tmp_path / "out"),
    )

    result = tool.forward(
        str(seq), ["person"],
        progress_cb=lambda d, t: calls.append((d, t)),
    )

    assert calls == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert result["frames"] == 4  # 取消未发生，正常收尾


def test_cancel_event_pre_set_raises_before_any_output(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    """cancel_event 预置位 → TrackingCancelledError（ValueError 子类），零产物。"""
    monkeypatch.chdir(tmp_path)
    seq = _make_seq(tmp_path, n=3)
    cancel = threading.Event()
    cancel.set()
    tool = TrackingTool(
        model=FakeDetModel(),
        batch_size=1,
        output_dir=str(tmp_path / "out"),
    )

    with pytest.raises(TrackingCancelledError):
        tool.forward(str(seq), ["person"], cancel_event=cancel)

    assert issubclass(TrackingCancelledError, ValueError)  # 既有收敛路径零改动
    assert not list(tmp_path.rglob("*.json"))  # 无逐帧 JSON
    assert not list(tmp_path.rglob("*_mot.txt"))  # MOT 不产出


def test_cancel_mid_video_releases_writer_and_no_mot(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    """视频源取消：finally 链 release 成片（可回读 1 帧），MOT/汇总不产出。

    第 1 帧完成回调里置位 cancel_event——第 2 帧循环头检查抛
    TrackingCancelledError，video_writer.release 必经 finally 执行；
    成片视频可回读且帧数 = 已写入的 1 帧（未 release 则文件不可读）。
    """
    monkeypatch.chdir(tmp_path)
    video = _make_video(tmp_path, n=3)
    cancel = threading.Event()
    calls: list[tuple[int, int]] = []

    def cb(done: int, total: int) -> None:
        calls.append((done, total))
        if done == 1:
            cancel.set()

    tool = TrackingTool(
        model=FakeDetModel(),
        batch_size=1,
        output_dir=str(tmp_path / "out"),
    )

    with pytest.raises(TrackingCancelledError):
        tool.forward(str(video), ["person"], progress_cb=cb, cancel_event=cancel)

    assert calls == [(1, 3)]
    out_video = tmp_path / "output_clip.mp4"
    assert out_video.exists()
    cap = cv2.VideoCapture(str(out_video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert n == 1  # release 已执行，只有取消前写入的 1 帧
    assert not list(tmp_path.rglob("*_mot.txt"))  # MOT 导出在 try 外，不产出


def test_protocol_line_format(capsys: Any) -> None:
    """行协议 `[AL_PROGRESS] done/total` 纯文本（TUI 解析端单一约定）。"""
    print_al_progress(3, 10)
    captured = capsys.readouterr()
    assert captured.out == f"{AL_PROGRESS_PREFIX} 3/10\n"
    assert AL_PROGRESS_PREFIX == "[AL_PROGRESS]"


def test_sigterm_raises_keyboard_interrupt() -> None:
    """SIGTERM → KeyboardInterrupt（TUI /cancel 打断主线程序列触发 finally 链）。"""
    old = signal.getsignal(signal.SIGTERM)
    try:
        install_sigterm_interrupt()
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGTERM)
    finally:
        signal.signal(signal.SIGTERM, old)
