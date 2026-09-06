"""P3 进度/取消协议测试 —— TrackingTool progress_cb/cancel_event + CLI 行协议。

零真实权重（FakeDetModel + 合成帧序列/视频，沿用 test_tracking_tool.py 模式）。
覆盖：progress_cb (done,total) 序列、cancel_event → TrackingCancelledError
（ValueError 子类，既有 except ValueError 收敛路径零改动）、视频源取消后
finally 链 release 成片可读、取消后 MOT/成片不产出、cli_common 协议行
格式与 SIGTERM→KeyboardInterrupt handler；2026-09-07 审查回归：成片隐藏
临时名（.output_*.mp4 点前缀，非 .tmp 后缀——OpenCV 按扩展名选 muxer）+
finally 原子改名（#3）、collect_frames 抽帧循环取消检查（#6）、
estimate_tracking_output_gb_parts 拆分（#5）。
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
    PROGRESS_ENV,
    install_sigterm_interrupt,
    print_al_progress,
)
from auto2dlabel.models.detection import DetectionResult
from auto2dlabel.tests import Path, np
from auto2dlabel.tools.tracking import (
    TrackingCancelledError,
    TrackingTool,
    collect_frames,
    estimate_tracking_output_gb,
    estimate_tracking_output_gb_parts,
)


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
    审查 #3 防损坏同时断言：release 前写盘名是隐藏临时名 .output_clip.mp4
    （正式名不出现——SIGKILL 打穿 finally 时同样只留临时名，下游不会误读
    损坏成片），finally 原子改名后正式名出现且临时名不残留。注：临时名须
    以 .mp4 结尾（OpenCV 按扩展名选 muxer，.mp4.tmp 这类 .tmp 后缀 writer
    开不了），故用点前缀隐藏名而非 .tmp 后缀。
    """
    monkeypatch.chdir(tmp_path)
    video = _make_video(tmp_path, n=3)
    cancel = threading.Event()
    calls: list[tuple[int, int]] = []
    out_video = tmp_path / "output_clip.mp4"
    tmp_video = tmp_path / ".output_clip.mp4"

    def cb(done: int, total: int) -> None:
        calls.append((done, total))
        if done == 1:
            # 第 1 帧刚写入、尚未 release：正式名不得存在，写盘名仍为临时名
            assert not out_video.exists(), "release 前正式名不得出现"
            assert tmp_video.exists(), "release 前写盘名应为隐藏临时名 .output_clip.mp4"
            cancel.set()

    tool = TrackingTool(
        model=FakeDetModel(),
        batch_size=1,
        output_dir=str(tmp_path / "out"),
    )

    with pytest.raises(TrackingCancelledError):
        tool.forward(str(video), ["person"], progress_cb=cb, cancel_event=cancel)

    assert calls == [(1, 3)]
    assert out_video.exists()  # finally release + 原子改名后正式名出现
    assert not tmp_video.exists()  # 隐藏临时名不残留
    cap = cv2.VideoCapture(str(out_video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert n == 1  # release 已执行，只有取消前写入的 1 帧
    assert not list(tmp_path.rglob("*_mot.txt"))  # MOT 导出在 try 外，不产出


def test_protocol_line_format(capsys: Any, monkeypatch: Any) -> None:
    """行协议 `[AL_PROGRESS] done/total` 纯文本（TUI 解析端单一约定）。

    协议行由 env AUTOLABEL_PROGRESS=1 门控（仅 TUI 子进程开启）——
    断言打印必须先置 env（cli_common.print_al_progress 门控）。
    """
    monkeypatch.setenv(PROGRESS_ENV, "1")
    print_al_progress(3, 10)
    captured = capsys.readouterr()
    assert captured.out == f"{AL_PROGRESS_PREFIX} 3/10\n"
    assert AL_PROGRESS_PREFIX == "[AL_PROGRESS]"


def test_protocol_line_gated_by_env(capsys: Any, monkeypatch: Any) -> None:
    """未置 AUTOLABEL_PROGRESS=1 时协议行不打印（普通 CLI 输出零污染）。"""
    monkeypatch.delenv(PROGRESS_ENV, raising=False)
    print_al_progress(3, 10)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_collect_frames_video_cancel_pre_set(tmp_path: Path, monkeypatch: Any) -> None:
    """审查 #6：视频源抽帧循环每帧检查 cancel_event——预置位在首帧写盘前
    抛 TrackingCancelledError，零帧落盘（取消首查点提前到抽帧段）。"""
    monkeypatch.chdir(tmp_path)
    video = _make_video(tmp_path, n=3)
    cancel = threading.Event()
    cancel.set()

    with pytest.raises(TrackingCancelledError):
        collect_frames(video, tmp_path / "out", cancel_event=cancel)

    assert not list((tmp_path / "out" / "track_frames" / "clip").glob("*.jpg"))


def test_collect_frames_video_cancel_mid_extraction(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    """审查 #6：抽帧中途置位 → 循环内停止抽取并抛 TrackingCancelledError。

    看门线程观察到首帧落盘即置位（每帧抽帧+写盘毫秒级，200 帧给足取消
    窗口）；断言已抽帧数 < 总帧数（取消打断在抽帧段而非等抽完）。
    """
    monkeypatch.chdir(tmp_path)
    video = _make_video(tmp_path, n=200)
    cancel = threading.Event()

    def _watch() -> None:
        frame_dir = tmp_path / "out" / "track_frames" / "clip"
        while not cancel.is_set():
            if list(frame_dir.glob("*.jpg")):
                cancel.set()
                return

    watcher = threading.Thread(target=_watch, daemon=True)
    watcher.start()
    with pytest.raises(TrackingCancelledError):
        collect_frames(video, tmp_path / "out", cancel_event=cancel)
    watcher.join(5.0)

    written = list((tmp_path / "out" / "track_frames" / "clip").glob("*.jpg"))
    assert 0 < len(written) < 200  # 已抽一部分即被取消，未跑完全部帧


def test_collect_frames_video_no_cancel_returns_all(tmp_path: Path, monkeypatch: Any) -> None:
    """基准：不传 cancel_event 行为不变——抽全 n 帧返回有序列表。"""
    monkeypatch.chdir(tmp_path)
    video = _make_video(tmp_path, n=3)
    frames = collect_frames(video, tmp_path / "out")
    assert [p.name for p in frames] == ["frame_000000.jpg", "frame_000001.jpg",
                                        "frame_000002.jpg"]


def test_estimate_tracking_output_gb_parts_split() -> None:
    """审查 #5：parts 拆分 (non_viz, png)；兼容入口 = sum；viz=False png 恒 0。

    帧目录/视频量按类型归边：non_viz = 抽帧 JSON 项 + 成片项（output_dir 盘），
    png = 逐帧 PNG 项（vis_outputs 盘）——两盘不同时预检各查各的。
    """
    frame_count, video_mb = 1000, 50.0
    non_viz, png = estimate_tracking_output_gb_parts(frame_count, viz=True, video_mb=video_mb)
    assert non_viz == pytest.approx(frame_count * 0.22 / 1024 + video_mb * 1.5 / 1024)
    assert png == pytest.approx(frame_count * 3.0 / 1024)
    assert png > non_viz  # 逐帧 PNG 是大头（实测 3MB/帧 vs 0.22MB/帧）
    # 兼容入口数值不变 = 两 parts 之和
    assert non_viz + png == pytest.approx(
        estimate_tracking_output_gb(frame_count, viz=True, video_mb=video_mb)
    )
    # viz=False：png 项为零（#25 长视频 --no-viz 不再被 png 预算误拒）
    non_viz2, png2 = estimate_tracking_output_gb_parts(frame_count, viz=False, video_mb=video_mb)
    assert png2 == 0.0
    assert non_viz2 == pytest.approx(non_viz)


def test_sigterm_raises_keyboard_interrupt() -> None:
    """SIGTERM → KeyboardInterrupt（TUI /cancel 打断主线程序列触发 finally 链）。"""
    old = signal.getsignal(signal.SIGTERM)
    try:
        install_sigterm_interrupt()
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGTERM)
    finally:
        signal.signal(signal.SIGTERM, old)
