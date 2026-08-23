"""Pose 姿态估计测试（YOLO-pose 抽象层 + keypoints 穿透 + 导出/可视化/CLI 分支）。

零真实权重：UltralyticsPoseModel 直接注入伪模型（_model 属性），
execute_plan 分支 monkeypatch create_pose_model 工厂注入 FakePoseModel。
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from PIL import Image

from auto2dlabel.agent.state import AgentState
from auto2dlabel.cli_execute import execute_plan
from auto2dlabel.export.coco import build_coco_dict, export_coco
from auto2dlabel.models.pose import PoseResult, UltralyticsPoseModel, create_pose_model
from auto2dlabel.schema.annotation import Annotation, Bbox
from auto2dlabel.schema.task_plan import TaskPlan, TaskStep
from auto2dlabel.tests import Path
from auto2dlabel.tools.export import ExportTool
from auto2dlabel.web import server

# ---------- 伪 ultralytics 结果（_parse_pred 输入） ----------

class _FakeBoxes:
    """ultralytics Boxes 鸭子类型：xyxy/cls/conf + __len__。"""

    def __init__(self, xyxy: list[list[float]], cls: list[int], conf: list[float]) -> None:
        self.xyxy, self.cls, self.conf = xyxy, cls, conf

    def __len__(self) -> int:
        return len(self.xyxy)


def _make_fake_pred() -> Any:
    """2 框（person/bicycle）各 17 点：xy 按 (11+j*2, 21+j*3) 排列，conf 交替 0.5/0。

    返回 Any：_parse_pred 参数标注为 ultralytics Results（stub 类型），
    测试伪对象经 Any 注入（duck typing，与零真实权重铁律一致）。
    """
    xy = np.zeros((2, 17, 2), dtype=np.float32)
    conf = np.zeros((2, 17), dtype=np.float32)
    for i in range(2):
        for j in range(17):
            xy[i][j] = (11 + j * 2, 21 + j * 3)
            conf[i][j] = 0.5 if j % 2 == 0 else 0.0
    return SimpleNamespace(
        boxes=_FakeBoxes([[10.0, 20.0, 30.0, 60.0], [100.0, 5.0, 150.0, 80.0]],
                         [0, 1], [0.9, 0.6]),
        keypoints=SimpleNamespace(xy=xy, conf=conf),
    )


def _make_model(names: dict[int, str] | None = None) -> UltralyticsPoseModel:
    """注入伪模型（绕过 _load，零 ultralytics 导入）。"""
    m = UltralyticsPoseModel()
    m._model = SimpleNamespace(names=names or {0: "person", 1: "bicycle"})
    return m


# ---------- _parse_pred：17 点对齐 / 过滤 / 容错 ----------

def test_parse_pred_aligns_17_keypoints() -> None:
    """伪 Results → PoseResult：坐标逐点对齐，v = conf>0 取整（1/2 不区分）。"""
    results = _make_model()._parse_pred(_make_fake_pred(), ["person", "bicycle"])
    assert len(results) == 2
    r0, r1 = results
    assert (r0.x, r0.y, r0.width, r0.height) == (10.0, 20.0, 20.0, 40.0)
    assert r0.label == "person" and r1.label == "bicycle"
    assert len(r0.keypoints) == 17
    for j, (px, py, v) in enumerate(r0.keypoints):
        assert (px, py) == pytest.approx((11 + j * 2, 21 + j * 3))
        assert v == (1 if j % 2 == 0 else 0)
    assert r1.confidence == pytest.approx(0.6)


def test_parse_pred_filters_by_prompt() -> None:
    """prompts 只含 person → bicycle 框被过滤。"""
    results = _make_model()._parse_pred(_make_fake_pred(), ["person"])
    assert [r.label for r in results] == ["person"]


def test_parse_pred_no_keypoints_tolerates() -> None:
    """pred 无 keypoints 属性（误载普通检测权重）→ 空列表不崩。"""
    pred = SimpleNamespace(boxes=None)
    assert _make_model()._parse_pred(cast(Any, pred), ["person"]) == []


def test_parse_pred_prompt_matching() -> None:
    """custom 目录名 labels（如 "自定义:person"）仍匹配 prompt。"""
    xy = np.zeros((1, 17, 2), dtype=np.float32)
    conf = np.ones((1, 17), dtype=np.float32)
    pred = SimpleNamespace(
        boxes=_FakeBoxes([[1.0, 2.0, 3.0, 4.0]], [0], [0.8]),
        keypoints=SimpleNamespace(xy=xy, conf=conf),
    )
    m = _make_model({0: "custom:person"})
    results = m._parse_pred(cast(Any, pred), ["person"])
    assert len(results) == 1


# ---------- detect_pose_batch：rect=False 铁律 ----------

def test_detect_pose_batch_rect_false_and_batch_kwargs() -> None:
    """批量推理显式 rect=False；单图不传 batch、多图传 batch=len。"""
    calls: list[dict[str, Any]] = []

    class _FakeYOLO:
        names = {0: "person", 1: "bicycle"}

        def __call__(self, source: Any, **kwargs: Any) -> list[Any]:
            calls.append(kwargs)
            paths = source if isinstance(source, list) else [source]
            return [_make_fake_pred() for _ in paths]

    m = _make_model({0: "person", 1: "bicycle"})
    m._model = _FakeYOLO()
    m.detect_pose_batch(["/a/1.jpg"], ["person"], 0.3)
    m.detect_pose_batch(["/a/1.jpg", "/a/2.jpg"], ["person"], 0.3, num_workers=2)
    assert len(calls) == 2
    assert calls[0]["rect"] is False and "batch" not in calls[0]
    assert calls[1]["rect"] is False and calls[1]["batch"] == 2 and calls[1]["workers"] == 2


# ---------- 工厂 ----------

def test_create_pose_model_dispatch() -> None:
    """create_pose_model：-pose.pt 名 → UltralyticsPoseModel；非 pose 名 → ValueError。"""
    for name in ("yolo11n-pose.pt", "yolo12n-pose.pt", "yolo26n-pose.pt"):
        assert isinstance(create_pose_model(name), UltralyticsPoseModel)
    with pytest.raises(ValueError):
        create_pose_model("yolo11n.pt")
    with pytest.raises(ValueError):
        create_pose_model("fasterrcnn_resnet50_fpn")


# ---------- keypoints 序列化穿透（to_dict / state / export / web） ----------

_KPTS: list[tuple[float, float, float]] = [
    (11.0, 21.0, 1.0), (13.0, 21.0, 1.0), (15.0, 26.0, 0.0),
]


def test_keypoints_serialization_paths() -> None:
    """keypoints 三路径（to_dict / state.from_dict / ExportTool._make_bbox）不丢。"""
    bbox = Bbox(x=1, y=2, width=3, height=4, label="person",
                confidence=0.9, keypoints=_KPTS)
    d = bbox.to_dict()
    assert d["keypoints"] == [list(k) for k in _KPTS]

    ann = Annotation(image_path="a.jpg")
    ann.add_bbox(bbox)
    restored = AgentState.from_dict(
        {"annotations": [ann.to_dict()], "image_path": "a.jpg"}
    ).annotations[0].bboxes[0]
    assert restored.keypoints == _KPTS

    assert ExportTool._make_bbox(d).keypoints == _KPTS
    assert server._bbox_from_dict(d).keypoints == _KPTS


def test_keypoints_empty_not_serialized() -> None:
    """空 keypoints 不输出（防 JSON 膨胀）；num_keypoints=0。"""
    bbox = Bbox(x=0, y=0, width=1, height=1, label="person")
    assert "keypoints" not in bbox.to_dict()
    assert bbox.num_keypoints == 0


def test_num_keypoints_counts_visible() -> None:
    """num_keypoints = v>0 点数。"""
    bbox = Bbox(x=0, y=0, width=1, height=1, label="person", keypoints=_KPTS)
    assert bbox.num_keypoints == 2


# ---------- COCO 导出：keypoints 内嵌 + 无 keypoints 逐位回归 ----------

def test_coco_export_keypoints_embedded(tmp_path: Path) -> None:
    """COCO JSON：keypoints 展平 51 值 + num_keypoints 字段。"""
    ann = Annotation(image_path="a.jpg", image_size=(100, 100))
    ann.add_bbox(Bbox(x=10, y=10, width=30, height=40, label="person",
                      confidence=0.9, keypoints=_KPTS))
    out = tmp_path / "a.json"
    export_coco([ann], out)
    data = json.loads(out.read_text(encoding="utf-8"))
    ann0 = data["annotations"][0]
    assert ann0["num_keypoints"] == 2
    assert len(ann0["keypoints"]) == 9
    assert ann0["keypoints"][:3] == [11.0, 21.0, 1.0]


def test_coco_export_no_keypoints_bit_identical() -> None:
    """无 keypoints 框导出与旧输出逐位一致（带 keypoints 版本去掉两字段后全等）。"""
    ann_kp = Annotation(image_path="a.jpg", image_size=(100, 100))
    ann_kp.add_bbox(Bbox(x=10, y=10, width=30, height=40, label="person",
                         confidence=0.9, keypoints=_KPTS))
    ann_plain = Annotation(image_path="a.jpg", image_size=(100, 100))
    ann_plain.add_bbox(Bbox(x=10, y=10, width=30, height=40, label="person",
                            confidence=0.9))

    d_kp = json.loads(json.dumps(build_coco_dict([ann_kp])))
    d_plain = json.loads(json.dumps(build_coco_dict([ann_plain])))

    kp_ann = d_kp["annotations"][0]
    assert "keypoints" in kp_ann and "num_keypoints" in kp_ann
    del kp_ann["keypoints"], kp_ann["num_keypoints"]
    assert kp_ann == d_plain["annotations"][0]
    assert d_kp["images"] == d_plain["images"]


# ---------- draw_keypoints：像素命中 / v=0 跳过 / 无 keypoints 不变 ----------

def test_draw_keypoints_pixels() -> None:
    """骨架线中点像素着色、可见点圆点着色、v=0 点不着色。"""
    from auto2dlabel.tools.visualize import _get_color, draw_keypoints

    kpts: list[tuple[float, float, float]] = [
        (10.0, 10.0, 1.0),   # 0 鼻
        (20.0, 10.0, 1.0),   # 1 左眼 → 线对 [0,1]
        (30.0, 20.0, 0.0),   # 2 右眼 v=0 → 线对 [0,2] 不画
    ] + [(0.0, 0.0, 0.0)] * 14
    bbox = Bbox(x=5, y=5, width=50, height=60, label="person",
                confidence=0.9, keypoints=kpts)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = draw_keypoints(img, [bbox])

    color = _get_color("person")  # BGR
    assert out[10, 15].tolist() == list(color)      # [0,1] 线中点
    assert out[10, 10].tolist() == list(color)      # 点 0 圆内
    assert out[20, 30].tolist() == [0, 0, 0]        # 点 2 v=0 不着色
    assert out[99, 99].tolist() == [0, 0, 0]


def test_draw_keypoints_empty_is_noop() -> None:
    """无 keypoints 框 / 空列表 → 图像逐位不变。"""
    from auto2dlabel.tools.visualize import draw_keypoints

    img = np.zeros((50, 50, 3), dtype=np.uint8)
    bbox = Bbox(x=0, y=0, width=10, height=10, label="person")
    out = draw_keypoints(img, [bbox])
    assert np.array_equal(out, img)


# ---------- execute_plan pose 分支（FakePoseModel 注入） ----------

_KPTS17: list[tuple[float, float, float]] = [
    (float(10 + j), float(20 + j), 1.0 if j % 2 == 0 else 0.0) for j in range(17)
]


class _FakePoseModel:
    """duck typing PoseModel：逐图/批量返回同一 PoseResult。"""

    def detect_pose(self, image_path: str, prompts: list[str],
                    confidence_threshold: float = 0.3) -> list[PoseResult]:
        return [PoseResult(x=5, y=10, width=30, height=40, label="person",
                           confidence=0.9, keypoints=_KPTS17)]

    def detect_pose_batch(self, paths: list[str], prompts: list[str],
                          confidence_threshold: float, num_workers: int = 0,
                          ) -> list[list[PoseResult]]:
        return [self.detect_pose(p, prompts, confidence_threshold) for p in paths]


def test_execute_plan_pose_step_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """execute_plan pose_estimation 步骤：Fake 工厂 → 推理 → coco 导出含 keypoints。"""
    img = tmp_path / "person.jpg"
    Image.new("RGB", (64, 64)).save(img)

    monkeypatch.setattr("auto2dlabel.models.pose.create_pose_model",
                        lambda name="yolo11n-pose.pt", **kw: _FakePoseModel())
    monkeypatch.setattr("auto2dlabel.tools.log.log_python_api_call", lambda **kw: None)
    monkeypatch.setattr("auto2dlabel.tools.log.log_chat_call", lambda **kw: None)
    monkeypatch.chdir(tmp_path)

    step = TaskStep(
        step_id=1, task_type="pose_estimation", source=str(img),
        prompts=["person"], model_name="yolo11n-pose.pt",
        confidence_threshold=0.5, iou_threshold=0.3, export_format="coco",
    )
    execute_plan(TaskPlan(steps=[step], confirm_timeout=0))

    # coco 导出文件含 keypoints（outputs/person_<ts>.json；排除 triage 的 *_review.json）
    jsons = [p for p in (tmp_path / "outputs").glob("person_*.json")
             if not p.name.endswith("_review.json")]
    assert len(jsons) == 1
    data = json.loads(jsons[0].read_text(encoding="utf-8"))
    ann0 = data["annotations"][0]
    assert ann0["num_keypoints"] == sum(1 for k in _KPTS17 if k[2] > 0)
    assert len(ann0["keypoints"]) == 51
    # 可视化骨架图已生成（vis_outputs）
    assert len(list((tmp_path / "vis_outputs").glob("vis_person_*.png"))) == 1


def test_execute_plan_pose_batch_two_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """两张图步骤：块级 detect_pose_batch 分派（Fake 记录调用）。"""
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for name in ("a.jpg", "b.jpg"):
        Image.new("RGB", (32, 32)).save(img_dir / name)

    batch_calls: list[int] = []

    class _CountingPoseModel(_FakePoseModel):
        def detect_pose_batch(self, paths: list[str], prompts: list[str],
                              confidence_threshold: float, num_workers: int = 0,
                              ) -> list[list[PoseResult]]:
            batch_calls.append(len(paths))
            return super().detect_pose_batch(paths, prompts, confidence_threshold, num_workers)

    monkeypatch.setattr("auto2dlabel.models.pose.create_pose_model",
                        lambda name="yolo11n-pose.pt", **kw: _CountingPoseModel())
    monkeypatch.setattr("auto2dlabel.tools.log.log_python_api_call", lambda **kw: None)
    monkeypatch.setattr("auto2dlabel.tools.log.log_chat_call", lambda **kw: None)
    monkeypatch.setattr("auto2dlabel.cli_execute._resolve_step_batch",
                        lambda *a, **kw: (2, 0))
    monkeypatch.chdir(tmp_path)

    step = TaskStep(
        step_id=1, task_type="pose_estimation", source=str(img_dir),
        prompts=["person"], model_name="yolo11n-pose.pt",
        confidence_threshold=0.5, iou_threshold=0.3, export_format="coco",
    )
    execute_plan(TaskPlan(steps=[step], confirm_timeout=0))

    assert batch_calls == [2]  # 块级批量路径恰一次（2 图一个 chunk）
    coco_jsons = [p for p in (tmp_path / "outputs").glob("*.json")
                  if not p.name.endswith("_review.json")]
    assert len(coco_jsons) == 2
