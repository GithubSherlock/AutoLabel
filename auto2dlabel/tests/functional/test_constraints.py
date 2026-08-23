"""指代约束过滤层单测（v0.4 3a）—— 零真实权重（FakeScorer duck typing）。

覆盖：parse_referential 参数化（属性/方位/退化）、parse_roi 矩形/多边形/
非法输入、filter_by_spatial（ROI 射线法 + 方位分位 + AND 交集）、
filter_by_attributes（候选对阈值 + 多属性 AND）、TrackingTool 端到端
（属性约束过滤后仅 1 条轨迹进 MOT）。
"""

from __future__ import annotations

from typing import Any

import cv2
import pytest

from auto2dlabel.models.detection import DetectionResult
from auto2dlabel.schema.annotation import Bbox
from auto2dlabel.tests import Path, np
from auto2dlabel.tools.constraints import (
    ReferentialConstraint,
    filter_by_attributes,
    filter_by_spatial,
    parse_referential,
    parse_roi,
)
from auto2dlabel.tools.tracking import TrackingTool


def _b(x: float, y: float, label: str = "person") -> Bbox:
    return Bbox(x=x, y=y, width=100.0, height=100.0, label=label, confidence=0.9)


class FakeScorer:
    """脚本化属性打分器：每次调用第 0 个 crop 命中（正候选概率 1.0）。

    score_crops 返回概率矩阵（候选原序），模拟 CLIP 语义：正候选 =
    "a {attr} {label}" 开头者。测试的 boxes 列表首个框恒为「命中目标」。
    """

    def __init__(self) -> None:
        self.calls: list[tuple[int, list[str]]] = []

    def score_crops(
        self,
        images: list[Any],
        candidates: list[str],
    ) -> list[list[float]]:
        self.calls.append((len(images), list(candidates)))
        return [
            [1.0 if (idx == 0 and c.startswith("a ")) else 0.0 for c in candidates]
            for idx in range(len(images))
        ]


# ---------- parse_referential ----------

@pytest.mark.parametrize(
    ("text", "prompts", "attributes", "position"),
    [
        ("检测行人", ["person"], [], None),
        ("检测红色的汽车", ["car"], ["red"], None),
        ("跟踪穿白色衣服的人", ["person"], ["white"], None),
        ("检测左边的人", ["person"], [], "left"),
        ("检测中间的行人", ["person"], [], "center"),
        ("检测右边红色的车", ["car"], ["red"], "right"),
        ("检测蓝色和白色的汽车", ["car"], ["blue", "white"], None),  # 多属性 AND
    ],
)
def test_parse_referential(
    text: str, prompts: list[str], attributes: list[str], position: str | None,
) -> None:
    c = parse_referential(text)
    assert c.prompts == prompts
    assert c.attributes == attributes
    assert c.position == position
    assert c.is_plain == (not attributes and position is None)


def test_parse_referential_no_class_raises() -> None:
    with pytest.raises(ValueError, match="无法从指令中提取关键词"):
        parse_referential("红色的东西")


# ---------- parse_roi ----------

def test_parse_roi_rectangle() -> None:
    assert parse_roi("100,100,500,400") == [
        (100.0, 100.0), (500.0, 100.0), (500.0, 400.0), (100.0, 400.0)
    ]


def test_parse_roi_polygon() -> None:
    assert parse_roi("0,0;100,0;100,100") == [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)]


@pytest.mark.parametrize("bad", ["100,100,500", "0,0;10,10", "a,b,c,d", ""])
def test_parse_roi_invalid_raises(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_roi(bad)


# ---------- filter_by_spatial ----------

def test_filter_spatial_roi_keeps_center_inside() -> None:
    roi = [(0.0, 0.0), (400.0, 0.0), (400.0, 400.0), (0.0, 400.0)]
    boxes = [_b(50, 50), _b(500, 50), _b(300, 300)]  # 第 2 个中心在外
    out = filter_by_spatial(boxes, roi=roi)
    assert [b.x for b in out] == [50.0, 300.0]


def test_filter_spatial_position_thirds() -> None:
    boxes = [_b(10, 10), _b(200, 10), _b(500, 10)]  # 700 宽画面：左/中/右
    left = filter_by_spatial(boxes, position="left", image_size=(700, 400))
    center = filter_by_spatial(boxes, position="center", image_size=(700, 400))
    right = filter_by_spatial(boxes, position="right", image_size=(700, 400))
    assert [b.x for b in left] == [10.0]
    assert [b.x for b in center] == [200.0]
    assert [b.x for b in right] == [500.0]


def test_filter_spatial_roi_and_position_intersect() -> None:
    roi = [(0.0, 0.0), (400.0, 0.0), (400.0, 400.0), (0.0, 400.0)]
    boxes = [_b(50, 50), _b(300, 50), _b(500, 50)]
    out = filter_by_spatial(boxes, roi=roi, position="right", image_size=(700, 400))
    assert out == []  # ROI 内无右侧目标 → AND 交集为空


def test_filter_spatial_none_constraint_passthrough() -> None:
    boxes = [_b(50, 50), _b(500, 50)]
    assert filter_by_spatial(boxes) == boxes


# ---------- filter_by_attributes ----------

def test_filter_attributes_keeps_matching_only() -> None:
    from PIL import Image

    img = Image.new("RGB", (400, 100))
    boxes = [_b(0, 0), _b(200, 0)]  # FakeScorer：左框正候选 1.0，右框 0.0
    scorer = FakeScorer()
    out = filter_by_attributes(img, boxes, ["red"], scorer)
    assert [b.x for b in out] == [0.0]
    # 候选对构造："a red person" / "a person"（唯一化后按原序概率）
    assert scorer.calls[0][0] == 2
    assert set(scorer.calls[0][1]) == {"a red person", "a person"}


def test_filter_attributes_multi_attr_and() -> None:
    from PIL import Image

    img = Image.new("RGB", (400, 100))
    boxes = [_b(0, 0), _b(200, 0)]
    scorer = FakeScorer()
    # 左框全部属性命中（多属性 AND 保留），右框全不命中
    out = filter_by_attributes(img, boxes, ["red", "white"], scorer)
    assert [b.x for b in out] == [0.0]


def test_filter_attributes_no_attrs_passthrough() -> None:
    from PIL import Image

    img = Image.new("RGB", (400, 100))
    boxes = [_b(0, 0)]
    scorer = FakeScorer()
    assert filter_by_attributes(img, boxes, [], scorer) == boxes
    assert scorer.calls == []


def test_filter_attributes_empty_boxes_skips_scorer() -> None:
    from PIL import Image

    img = Image.new("RGB", (400, 100))
    scorer = FakeScorer()
    assert filter_by_attributes(img, [], ["red"], scorer) == []
    assert scorer.calls == []


# ---------- TrackingTool 端到端（属性约束） ----------

class TwoDetModel:
    """两目标逐帧右移（各 60×60 @ conf 0.9）：左框 + 右框。"""

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
            DetectionResult(x=10.0 + 4.0 * i, y=20.0, width=60.0, height=60.0,
                            label=prompts[0], confidence=0.9),
            DetectionResult(x=60.0 + 4.0 * i, y=20.0, width=60.0, height=60.0,
                            label=prompts[0], confidence=0.9),
        ]


def test_tracking_tool_attribute_constraint_filters_to_one_track(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    """端到端：属性约束（FakeScorer 只保留左框）→ MOT 仅 1 条轨迹。

    合成帧 3 张，左框 x 从 10 起（<200 → 命中），右框 x 从 60 起（crop 后
    bbox 相对坐标——FakeScorer 按 crop 图 bbox 左边界 <200 判定，两 crop
    合成图都是 60×60，需让 FakeScorer 可区分 → 按调用序：偶数框保留）。
    """
    monkeypatch.chdir(tmp_path)
    seq = tmp_path / "seq"
    seq.mkdir()
    for i in range(3):
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        frame[20:80, 10 + i * 4:70 + i * 4] = 255
        cv2.imwrite(str(seq / f"f{i}.png"), frame)

    class OrderedScorer:
        def score_crops(self, images: list[Any], candidates: list[str]) -> list[list[float]]:
            # 奇数索引 crop 判为命中（每帧两框 → 只保留第一个）
            out = []
            for idx in range(len(images)):
                hit = idx % 2 == 0
                out.append([1.0 if (hit and c.startswith("a ")) else 0.0 for c in candidates])
            return out

    constraint = ReferentialConstraint(prompts=["person"], attributes=["red"])
    tool = TrackingTool(
        model=TwoDetModel(),
        batch_size=1,
        output_dir=str(tmp_path / "out"),
        constraint=constraint,
        attribute_scorer=OrderedScorer(),
    )

    result = tool.forward(str(seq), ["person"])

    assert result["bboxes"] == 3  # 每帧过滤后剩 1 框
    assert result["track_ids"] == [0]
    mot = (tmp_path / "out" / "seq_mot.txt").read_text().strip().splitlines()
    assert len(mot) == 3
    for i, line in enumerate(mot):
        # 只剩左框（逐帧右移 4px）：x ≈ 10 + 4i；右框 x≈60+4i 已被属性过滤剔除
        assert float(line.split(",")[2]) == pytest.approx(10.0 + 4.0 * i, abs=2.0)
