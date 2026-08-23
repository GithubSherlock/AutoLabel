"""指代检测 L2（Florence-2 兜底）测试。

零真实权重铁律：Florence2ReferentialResolver 直接注入伪 processor/model
（_processor/_model 属性，绕过 _load 权重下载）；TrackingTool/execute_plan
用 FakeResolver + Fake 检测模型注入。纯函数（关系词触发/短语构造/中心距
匹配/锁定帧匹配）单独覆盖。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from auto2dlabel.models.detection import DetectionResult
from auto2dlabel.models.referential import (
    Florence2ReferentialResolver,
    create_referential_resolver,
    match_boxes_by_center,
)
from auto2dlabel.schema.annotation import Bbox
from auto2dlabel.tests import Path
from auto2dlabel.tools.constraints import (
    build_referential_phrase,
    has_relation,
    parse_referential,
)
from auto2dlabel.tools.tracking import TrackingTool, match_dets_to_locked

# ---------- 关系词触发（has_relation） ----------

@pytest.mark.parametrize(
    "instruction",
    [
        "红车旁边的行人", "跟踪穿红衣服的人旁边", "person next to the red car",
        "behind the tree", "两个人之间的自行车", "靠近卡车的行人",
    ],
)
def test_has_relation_true(instruction: str) -> None:
    """中英文关系词任一命中 → True。"""
    assert has_relation(instruction)


@pytest.mark.parametrize(
    "instruction",
    [
        "跟踪穿红衣服的人",       # 纯属性（L1 覆盖）
        "左边红色的汽车",          # 方位词（L1 覆盖）
        "检测行人",               # 纯类别
    ],
)
def test_has_relation_false(instruction: str) -> None:
    """无关系词（属性/方位/纯类别）→ False。"""
    assert not has_relation(instruction)


# ---------- L2 英文短语构造（build_referential_phrase） ----------

def test_build_phrase_subject_after_relation() -> None:
    """「红车旁边的行人」→ "the person next to the red car"（主体=关系词后最近类别）。"""
    c = parse_referential("红车旁边的行人")
    assert c.prompts == ["person", "car"] and c.attributes == ["red"]
    assert build_referential_phrase("红车旁边的行人", c) == "the person next to the red car"


def test_build_phrase_english_relation() -> None:
    """英文关系词混合指令（中文类别 + 英文关系词）：英译直取。"""
    c = parse_referential("红车 next to 行人")
    assert build_referential_phrase("红车 next to 行人", c) == "the person next to the red car"


def test_build_phrase_no_relation_fallback() -> None:
    """无关系词：退化 "属性 类别"（L2 兜底也接受）。"""
    c = parse_referential("跟踪穿红衣服的人")
    assert build_referential_phrase("跟踪穿红衣服的人", c) == "red person"


# ---------- Florence 输出框 → 原框匹配（match_boxes_by_center） ----------

def _bbox(x: float, y: float, w: float, h: float, label: str = "person") -> Bbox:
    return Bbox(x=x, y=y, width=w, height=h, label=label, confidence=0.9)


def test_match_boxes_by_center_preserves_original_boxes() -> None:
    """匹配返回原 Bbox 对象（track_id/置信度无损）。"""
    src = _bbox(10, 10, 20, 20)
    src.track_id = 7
    out = match_boxes_by_center([(11, 11, 29, 29)], [src], dist_limit=100.0)
    assert out == [src] and out[0].track_id == 7


def test_match_boxes_by_center_drops_far_box() -> None:
    """Florence 框与所有原框中心距超阈值 → 丢弃。"""
    out = match_boxes_by_center([(200, 200, 220, 220)], [_bbox(10, 10, 20, 20)],
                                dist_limit=50.0)
    assert out == []


def test_match_boxes_by_center_default_threshold_short_side() -> None:
    """默认阈值 = max(50, 短边/2)：60px 偏移对小框（20 短边→50 阈值）超限。"""
    out = match_boxes_by_center([(70, 10, 90, 30)], [_bbox(10, 10, 20, 20)])
    assert out == []
    # 大框（短边 200 → 阈值 100）60px 偏移保留
    out = match_boxes_by_center([(170, 110, 230, 250)], [_bbox(100, 100, 200, 200)])
    assert len(out) == 1


def test_match_boxes_by_center_greedy_one_to_one() -> None:
    """贪心一一匹配：两个 Florence 框不重复占用同一原框。"""
    out = match_boxes_by_center(
        [(11, 11, 29, 29), (31, 11, 49, 29)],
        [_bbox(10, 10, 20, 20), _bbox(30, 10, 20, 20)],
        dist_limit=100.0,
    )
    assert len(out) == 2


# ---------- 锁定轨迹匹配（match_dets_to_locked） ----------

def test_match_dets_to_locked_matches_and_drops() -> None:
    """锁定位置匹配最近候选，远处干扰框丢弃（目标集合首帧已定）。"""
    bboxes = [_bbox(10, 10, 20, 20, "person"), _bbox(300, 300, 20, 20, "dog")]
    out = match_dets_to_locked(bboxes, {1: (12.0, 12.0)})
    assert [b.label for b in out] == ["person"]


def test_match_dets_to_locked_empty_positions() -> None:
    """无锁定位置（首帧 0 框）→ 空（不建新轨迹）。"""
    assert match_dets_to_locked([_bbox(10, 10, 20, 20)], {}) == []


# ---------- Florence2ReferentialResolver（注入伪 processor/model） ----------

class _FakeTensor:
    """伪张量：is_floating_point/to(device, dtype) 链式（resolve 会把输入搬上
    device 并对齐浮点 dtype）。"""

    def __init__(self, floating: bool = True) -> None:
        self._floating = floating

    def is_floating_point(self) -> bool:
        return self._floating

    def to(self, device: str = "cpu", dtype: Any = None) -> _FakeTensor:
        return self


class _FakeFlorenceProcessor:
    """伪 Florence2Processor：__call__/batch_decode/post_process_generation。"""

    def __init__(self, od_out: Any) -> None:
        self._od_out = od_out

    def __call__(self, text: str, images: Any, return_tensors: str) -> dict[str, Any]:
        return {"input_ids": _FakeTensor(floating=False), "pixel_values": _FakeTensor()}

    def batch_decode(self, ids: Any, **kw: Any) -> list[str]:
        return ["<OD>ok</OD>"]

    def post_process_generation(self, text: str, task: str, image_size: Any) -> dict[str, Any]:
        return {task: self._od_out}


def _make_fake_florence(od_out: Any) -> Florence2ReferentialResolver:
    """注入伪 processor/model（绕过 _load，零权重下载）。"""
    r = Florence2ReferentialResolver()
    r._processor = _FakeFlorenceProcessor(od_out)
    r._model = SimpleNamespace(
        generate=lambda **kw: SimpleNamespace(),
        to=lambda device: None,
        parameters=lambda: iter([SimpleNamespace(dtype="fake")]),
    )
    return r


def test_florence_resolve_matches_original_boxes(tmp_path: Path) -> None:
    """伪 Florence 输出像素框 → 中心匹配回原框子集。"""
    r = _make_fake_florence(
        {"bboxes": [[11.0, 11.0, 29.0, 29.0]], "labels": ["person"]}
    )
    img = tmp_path / "a.jpg"
    Image.new("RGB", (64, 64)).save(img)
    src = _bbox(10, 10, 20, 20)
    out = r.resolve(str(img), "the person next to the red car", [src, _bbox(300, 300, 20, 20)])
    assert out == [src]


def test_florence_resolve_no_boxes_keeps_all() -> None:
    """Florence 无输出 → 宁多勿漏保留全部。"""
    r = _make_fake_florence({"bboxes": [], "labels": []})
    boxes = [_bbox(10, 10, 20, 20)]
    assert r.resolve("/a.jpg", "phrase", boxes) == boxes


def test_florence_resolve_failure_keeps_all() -> None:
    """resolve 内部异常（图片打开失败等）→ 保留全部（兜底不丢检测）。"""
    r = _make_fake_florence({"bboxes": [], "labels": []})
    boxes = [_bbox(10, 10, 20, 20)]
    out = r.resolve("/nonexistent.jpg", "phrase", boxes)  # Image.open 抛异常
    assert out == boxes


def test_florence_resolve_empty_bboxes_noop() -> None:
    """空候选框：不加载模型直接返回 []。"""
    r = Florence2ReferentialResolver()  # 不注入，验证未触发 _load
    assert r.resolve("/a.jpg", "phrase", []) == []


def test_create_referential_resolver_factory() -> None:
    """工厂返回 Florence2ReferentialResolver 实现（Protocol duck typing）。"""
    r = create_referential_resolver()
    assert isinstance(r, Florence2ReferentialResolver)
    assert hasattr(r, "resolve")


# ---------- TrackingTool 指代 L2 端到端（Fake 检测 + FakeResolver） ----------

class _FakeDetModel:
    """Fake 检测模型：帧 1 目标 + 干扰，帧 2/3 目标移动 + 新干扰。"""

    def detect(self, image_path: str, prompts: list[str],
               confidence_threshold: float = 0.3) -> list[DetectionResult]:
        frame = int(image_path.rsplit("/", 1)[-1].split("_")[-1].split(".")[0])
        # 2px/帧移动：20×20 框相邻 IoU≈0.82 > ByteTrack 0.8 匹配阈值，轨迹 ID 持续
        dets = [DetectionResult(x=10.0 + frame * 2, y=10.0, width=20.0, height=20.0,
                                label="person", confidence=0.9)]
        if frame >= 2:
            dets.append(DetectionResult(x=300.0, y=300.0, width=20.0, height=20.0,
                                        label="dog", confidence=0.9))
        return dets

    def detect_batch(self, paths: list[str], prompts: list[str],
                     confidence_threshold: float, num_workers: int = 0,
                     ) -> list[list[DetectionResult]]:
        return [self.detect(p, prompts, confidence_threshold) for p in paths]


class _FakeResolver:
    """Fake 指代解析器：只保留 person 框（模拟「红车旁边的行人」锁定）。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    def resolve(self, image_path: str, phrase: str, bboxes: list[Bbox]) -> list[Bbox]:
        self.calls.append((image_path, phrase, len(bboxes)))
        return [b for b in bboxes if b.label == "person"]


@pytest.fixture()
def _track_frames(tmp_path: Path) -> Path:
    """3 帧合成帧目录（32x32 小图，帧名 frame_00000X.jpg 供 Fake 解析）。"""
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    for i in range(3):
        Image.new("RGB", (32, 32)).save(frame_dir / f"frame_{i}.jpg")
    return frame_dir


def test_tracking_referential_l2_locks_targets(
    tmp_path: Path, _track_frames: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TrackingTool + FakeResolver：首帧解析恰一次，后续帧锁定目标（dog 不进）。"""
    monkeypatch.setattr("auto2dlabel.tools.log.log_python_api_call", lambda **kw: None)
    monkeypatch.chdir(tmp_path)

    resolver = _FakeResolver()
    tool = TrackingTool(
        model=_FakeDetModel(),
        viz=False,
        referential=resolver,
        referential_phrase="the person next to the red car",
    )
    summary = tool.forward(str(_track_frames), ["person", "dog"], 0.3)

    # 首帧解析恰一次（序列级调用红线）；后续帧锁定匹配
    assert len(resolver.calls) == 1
    assert resolver.calls[0][1] == "the person next to the red car"
    # 3 帧 annotation 全部不含 dog（锁定过滤）；person 框 3 帧在列
    import json

    # 逐帧 JSON 名为 frame_N_<时间戳>.json；排除 triage 产出的 *_review.json
    jsons = sorted(
        p for p in (tmp_path / "outputs").glob("frame_*.json")
        if not p.name.endswith("_review.json")
    )
    assert len(jsons) == 3
    for jp in jsons:
        data = json.loads(jp.read_text(encoding="utf-8"))
        # 逐帧 JSON 为 COCO 格式：annotation 用 category_id，经 categories 表映射
        id2name = {c["id"]: c["name"] for c in data["categories"]}
        labels = [id2name[a["category_id"]] for a in data["annotations"]]
        assert labels == ["person"], jp.name
    # 跟踪汇总：person 单轨迹（dog 从未进 tracker）
    assert summary["frames"] == 3 and len(summary["track_ids"]) == 1


def test_tracking_without_referential_keeps_all(
    tmp_path: Path, _track_frames: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """无 referential（向后兼容）：帧 2/3 的 dog 正常进 annotation。"""
    monkeypatch.setattr("auto2dlabel.tools.log.log_python_api_call", lambda **kw: None)
    monkeypatch.chdir(tmp_path)

    tool = TrackingTool(model=_FakeDetModel(), viz=False)
    tool.forward(str(_track_frames), ["person", "dog"], 0.3)

    import json

    # 逐帧 JSON 名为 frame_2_<时间戳>.json；glob 取帧 2（排除 _review.json）
    frame2_files = [
        p for p in (tmp_path / "outputs").glob("frame_2_*.json")
        if not p.name.endswith("_review.json")
    ]
    assert len(frame2_files) == 1
    frame2 = json.loads(frame2_files[0].read_text(encoding="utf-8"))
    id2name = {c["id"]: c["name"] for c in frame2["categories"]}
    labels = [id2name[a["category_id"]] for a in frame2["annotations"]]
    assert sorted(labels) == ["dog", "person"]


# ---------- execute_plan tracking 步骤 L2 透传 ----------

def test_execute_plan_refer_l2_hooks_resolver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """execute_plan(refer_l2=True)：Fake 工厂 → TrackingTool 挂接 → resolve 恰一次。"""
    from auto2dlabel.cli_execute import execute_plan
    from auto2dlabel.schema.task_plan import TaskPlan, TaskStep

    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    for i in range(2):
        Image.new("RGB", (32, 32)).save(frame_dir / f"frame_{i}.jpg")

    resolver = _FakeResolver()
    monkeypatch.setattr(
        "auto2dlabel.models.referential.create_referential_resolver",
        lambda name="microsoft/Florence-2-base", **kw: resolver,
    )
    monkeypatch.setattr("auto2dlabel.tools.log.log_python_api_call", lambda **kw: None)
    # TrackingTool.model property 内部走 tools.tracking 模块级引用
    monkeypatch.setattr("auto2dlabel.tools.tracking.create_detection_model",
                        lambda name, iou_threshold=0.5: _FakeDetModel())
    monkeypatch.chdir(tmp_path)

    step = TaskStep(
        step_id=1, task_type="tracking", source=str(frame_dir),
        prompts=["person", "dog"], model_name="yolo11n.pt",
        confidence_threshold=0.3, iou_threshold=0.3, export_format="mot",
    )
    constraint = parse_referential("红车旁边的行人")
    execute_plan(TaskPlan(steps=[step], confirm_timeout=0, raw_instruction="红车旁边的行人"),
                 constraint=constraint, refer_l2=True)

    assert len(resolver.calls) == 1
    assert resolver.calls[0][1] == "the person next to the red car"
