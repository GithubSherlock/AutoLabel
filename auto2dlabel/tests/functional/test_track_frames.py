"""tools.tracking.collect_frames 测试 —— 视频抽帧 / 帧目录收集 / 幂等重跑。

视频合成用 MJPG+AVI（OpenCV 自带编码器，无 FFmpeg 依赖）。
"""

from __future__ import annotations

import tempfile

import cv2

from auto2dlabel.tests import Path, np
from auto2dlabel.tools.tracking import collect_frames


def _make_video(path: Path, num_frames: int = 5) -> Path:
    """合成 num_frames 帧 AVI（64×64，白色矩形逐帧右移）。"""
    fourcc = getattr(cv2, "VideoWriter_fourcc")  # cv2 stub 缺该属性，getattr 兜底
    writer = cv2.VideoWriter(
        str(path), fourcc(*"MJPG"), 10, (64, 64),
    )
    assert writer.isOpened(), f"无法创建测试视频: {path}"
    for i in range(num_frames):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[20:40, 5 + i * 4:25 + i * 4] = 255
        writer.write(frame)
    writer.release()
    return path


def test_video_extraction() -> None:
    """视频 → 帧目录：帧数一致，命名 frame_%06d.jpg，可被 cv2 读回。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        video = _make_video(Path(tmpdir) / "test.avi", num_frames=5)
        frames = collect_frames(video, Path(tmpdir) / "out")

        assert len(frames) == 5
        assert [p.name for p in frames] == [f"frame_{i:06d}.jpg" for i in range(5)]
        assert frames[0].parent.name == "test"  # 抽帧目录按视频 stem 命名
        img = cv2.imread(str(frames[2]))
        assert img is not None and img.shape == (64, 64, 3)


def test_video_extraction_idempotent() -> None:
    """重复调用不重复抽帧（已存在帧跳过），结果一致。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        video = _make_video(Path(tmpdir) / "test.avi", num_frames=3)
        first = collect_frames(video, Path(tmpdir) / "out")
        second = collect_frames(video, Path(tmpdir) / "out")
        assert first == second


def test_frame_dir_collection() -> None:
    """帧目录输入：按文件名排序返回图像列表。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir) / "seq"
        d.mkdir()
        for i in (2, 0, 1):
            cv2.imwrite(str(d / f"f{i}.png"), np.zeros((8, 8, 3), dtype=np.uint8))

        frames = collect_frames(d, Path(tmpdir) / "out")
        assert [p.name for p in frames] == ["f0.png", "f1.png", "f2.png"]


def test_single_image_input() -> None:
    """单张图像输入（无时序上下文的最小序列）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        img = Path(tmpdir) / "single.jpg"
        cv2.imwrite(str(img), np.zeros((8, 8, 3), dtype=np.uint8))

        frames = collect_frames(img, Path(tmpdir) / "out")
        assert frames == [img]
