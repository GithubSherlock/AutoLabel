"""TrackingTool 单测 —— 零真实权重（Fake 检测模型 + FakeReIDModel duck typing）。

覆盖：forward 端到端（ByteTrack / BoT-SORT）、MOT 导出与逐帧 JSON、
空目录 ValueError、detect_tracker_kind 参数化、显式 batch_size=1 不探 GPU。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cv2
import pytest

from auto2dlabel.models import Image
from auto2dlabel.models.detection import DetectionResult
from auto2dlabel.tests import Path, json, np
from auto2dlabel.tools.tracking import (
    DEFAULT_REID_MODEL,
    TrackingTool,
    detect_tracker_kind,
)


class FakeDetModel:
    """单目标逐帧右移（100×80 @ conf 0.9）：相邻帧 IoU ≥ 0.8，3 帧维持同一轨迹。

    每帧恰好 1 次 detect 调用（首调用即有框，不触发降阈值重试），
    帧位置由调用序号推导。
    """

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


class FakeReIDModel:
    """常量特征向量（任意两批 cosine=1.0），用于 BoT-SORT 关联跑通。"""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def extract(
        self,
        images: Sequence[Image.Image],
    ) -> list[np.ndarray[Any, Any]]:
        self.calls.append(len(images))
        return [
            np.full(4, float(len(self.calls)) * 100 + i, dtype=np.float32)
            for i in range(len(images))
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


def _assert_mot_ok(mot_path: Path, n_frames: int, expected_ids: set[int]) -> None:
    """MOT txt：帧序 1 起、每行 9 列、ID 集合正确。"""
    lines = mot_path.read_text().strip().splitlines()
    assert len(lines) == n_frames
    for line in lines:
        parts = line.split(",")
        assert len(parts) == 9, f"MOT 行应为 9 列: {line}"
    assert {int(line.split(",")[0]) for line in lines} == set(range(1, n_frames + 1))
    assert {int(line.split(",")[1]) for line in lines} == expected_ids


def test_forward_bytetrack_end_to_end(tmp_path: Path, monkeypatch: Any) -> None:
    """ByteTrack 端到端：3 帧合成序列 → 1 条轨迹 → MOT + 逐帧 JSON（track_id 透出）。

    batch_size=1 显式传参：resolve_batch_params 走显式短路，不触发动态实测探 GPU。
    """
    monkeypatch.chdir(tmp_path)  # vis_outputs/logs/outputs 均为 CWD 相对路径
    seq = _make_seq(tmp_path)
    fake = FakeDetModel()
    tool = TrackingTool(
        model=fake,
        batch_size=1,
        output_dir=str(tmp_path / "out"),
    )

    result = tool.forward(str(seq), ["person"])

    assert result["frames"] == 3
    assert result["bboxes"] == 3
    assert result["track_ids"] == [0]
    assert result["video_path"] is None  # 帧目录不产片（避免污染数据集目录）
    assert not list(tmp_path.glob("output_*.mp4"))

    mot_path = Path(result["mot_path"])
    assert mot_path.name == "seq_mot.txt"  # 按 source stem 命名
    _assert_mot_ok(mot_path, n_frames=3, expected_ids={0})

    # 逐帧 JSON：3 个文件，bboxes 内 track_id 透出（coco 格式注解级字段）
    per_frame = sorted((tmp_path / "out").glob("*.json"))
    assert len(per_frame) == 3
    for p in per_frame:
        data = json.loads(p.read_text())
        assert data["annotations"][0]["track_id"] == 0

    assert len(fake.detect_calls) == 3  # 逐图路径，每帧恰好 1 次


def test_forward_bot_sort_end_to_end(tmp_path: Path, monkeypatch: Any) -> None:
    """BoT-SORT 端到端：ReID 特征注入 + ECC 补偿路径跑通（FakeReIDModel 零权重）。"""
    monkeypatch.chdir(tmp_path)
    seq = _make_seq(tmp_path)
    fake = FakeDetModel()
    fake_reid = FakeReIDModel()
    tool = TrackingTool(
        model=fake,
        use_bot_sort=True,
        reid_model=fake_reid,
        batch_size=1,
        output_dir=str(tmp_path / "out"),
    )

    result = tool.forward(str(seq), ["person"])

    assert result["frames"] == 3
    assert result["track_ids"] == [0]
    assert fake_reid.calls, "ReID 特征应被提取"
    _assert_mot_ok(Path(result["mot_path"]), n_frames=3, expected_ids={0})


def test_forward_video_source_writes_annotated_video(tmp_path: Path, monkeypatch: Any) -> None:
    """视频源端到端出片：output_<原名>.mp4 落在源视频同目录，帧数一致且已绘制标注。

    cv2.VideoWriter 合成 3 帧 mp4 作为输入（零真实权重），验证抽帧→跟踪→
    成片合成全链路；输出视频首帧像素应不同于原帧（框/标签已绘制）。
    """
    monkeypatch.chdir(tmp_path)
    video = tmp_path / "clip.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(video), fourcc, 10.0, (128, 128))
    assert writer.isOpened()
    for i in range(3):
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        frame[40:88, 30 + i * 4 : 80 + i * 4] = 255
        writer.write(frame)
    writer.release()

    fake = FakeDetModel()
    tool = TrackingTool(
        model=fake,
        batch_size=1,
        output_dir=str(tmp_path / "out"),
    )

    result = tool.forward(str(video), ["person"])

    assert result["video_path"] == str(tmp_path / "output_clip.mp4")
    out_video = Path(result["video_path"])
    assert out_video.exists()

    # 输出视频可回读：3 帧，首帧含绘制内容（与原帧像素不同）
    cap = cv2.VideoCapture(str(out_video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first = cap.read()
    cap.release()
    assert n == 3
    assert ok
    assert not np.array_equal(first, np.zeros((128, 128, 3), dtype=np.uint8))


def test_forward_viz_false_skips_pngs(tmp_path: Path, monkeypatch: Any) -> None:
    """viz=False（--no-viz）：跳过 vis_outputs 逐帧 PNG，MOT/逐帧 JSON/成片视频不受影响。"""
    monkeypatch.chdir(tmp_path)
    video = tmp_path / "clip.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(video), fourcc, 10.0, (128, 128))
    assert writer.isOpened()
    for i in range(3):
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        frame[40:88, 30 + i * 4 : 80 + i * 4] = 255
        writer.write(frame)
    writer.release()

    tool = TrackingTool(
        model=FakeDetModel(),
        batch_size=1,
        output_dir=str(tmp_path / "out"),
        viz=False,
    )

    result = tool.forward(str(video), ["person"])

    assert result["video_path"] == str(tmp_path / "output_clip.mp4")  # 成片视频照常
    assert (tmp_path / "output_clip.mp4").exists()
    assert not list(tmp_path.glob("vis_outputs/**/vis_*.png"))  # 无逐帧 PNG
    _assert_mot_ok(Path(result["mot_path"]), n_frames=3, expected_ids={0})
    assert len(sorted((tmp_path / "out").glob("*.json"))) == 3  # 逐帧 JSON 照常


def test_forward_empty_dir_raises(tmp_path: Path) -> None:
    """空目录 → ValueError（CLI 层转 typer.Exit）。"""
    empty = tmp_path / "empty"
    empty.mkdir()
    tool = TrackingTool(model=FakeDetModel(), batch_size=1)

    with pytest.raises(ValueError, match="未找到可跟踪的帧"):
        tool.forward(str(empty), ["person"])


def test_input_schema() -> None:
    """input_schema：source/prompts 必填，置信度默认 0.1。"""
    schema = TrackingTool(model=FakeDetModel()).input_schema
    assert schema["required"] == ["source", "prompts"]
    assert schema["properties"]["confidence_threshold"]["default"] == 0.1


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("使用 BoT-SORT", "bot_sort"),
        ("使用 botsort", "bot_sort"),
        ("bot_sort 精度档", "bot_sort"),
        ("bot sort 跟踪", "bot_sort"),
        ("BOT-SORT", "bot_sort"),
        ("使用 ByteTrack", "bytetrack"),
        ("跟踪行人", "bytetrack"),
        ("", "bytetrack"),
    ],
)
def test_detect_tracker_kind(text: str, expected: str) -> None:
    assert detect_tracker_kind(text) == expected


def test_default_reid_model_constant() -> None:
    """DEFAULT_REID_MODEL 单一事实源（cli_execute/cli_track 引用，防硬编码发散）。"""
    assert DEFAULT_REID_MODEL == "openai/clip-vit-base-patch32"
