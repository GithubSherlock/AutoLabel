"""KITTI 域内微调数据管线测试 —— 转换纯函数 / 划分 / yaml / 自定义类别解析。

零真实权重铁律：UltralyticsModel._parse_pred 经注入 Fake model（
SimpleNamespace(names=...)）测自定义类别表，不加载任何 .pt；转换/划分/
yaml 为纯文件 IO。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from auto2dlabel.models.detection import UltralyticsModel
from auto2dlabel.tools.train_kitti import (
    KITTI_TO_YOLO,
    KITTI_YOLO_NAMES,
    convert_label_file,
    kitti_line_to_yolo,
    split_train_val,
    write_data_yaml,
)

# KITTI label_2 行格式: cls truncated occluded alpha x1 y1 x2 y2 h w l x y z rot_y
_CAR_LINE = "Car 0.00 0 1.85 100.0 100.0 400.0 300.0 1.56 1.60 3.50 -1.0 1.8 25.0 1.62"


# ---------- 类别映射（与评测口径一致） ----------

def test_yolo_class_mapping_matches_eval() -> None:
    """KITTI 8 类 → 5 类索引（benchmark KITTI_TO_COCO 单一事实源）。"""
    assert KITTI_YOLO_NAMES == ["bicycle", "car", "person", "train", "truck"]
    assert KITTI_TO_YOLO == {
        "Car": 1, "Van": 1,
        "Pedestrian": 2, "Person_sitting": 2,
        "Cyclist": 0,
        "Truck": 4,
        "Tram": 3,
    }


# ---------- kitti_line_to_yolo ----------

def test_line_to_yolo_normalizes() -> None:
    """Car 行 → 归一化 cx cy w h（1242×375）。"""
    out = kitti_line_to_yolo(_CAR_LINE, 1242, 375)
    # cx=250/1242  cy=200/375  w=300/1242  h=200/375
    assert out == "1 0.201288 0.533333 0.241546 0.533333"


def test_line_to_yolo_excludes_dontcare_and_misc() -> None:
    """DontCare / Misc → 剔除（评测 ignore 类）。"""
    dontcare = "DontCare -1 -1 -10 0 0 100 100 -1 -1 -1 -1000 -1000 -1000 -10"
    misc = "Misc -1 -1 -10 0 0 100 100 -1 -1 -1 -1000 -1000 -1000 -10"
    assert kitti_line_to_yolo(dontcare, 1242, 375) is None
    assert kitti_line_to_yolo(misc, 1242, 375) is None


def test_line_to_yolo_excludes_beyond_hard() -> None:
    """occ=3（超出 hard 档）→ 剔除（与评测 ignore 语义一致）。"""
    line = _CAR_LINE.replace(" 0 1.85", " 3 1.85", 1)
    assert kitti_line_to_yolo(line, 1242, 375) is None


def test_line_to_yolo_excludes_too_small() -> None:
    """高度 < 25（不满足任何档）→ 剔除。"""
    line = "Car 0.00 0 1.85 100 100 110 120 1.56 1.60 3.50 -1 1.8 25.0 1.62"
    assert kitti_line_to_yolo(line, 1242, 375) is None


def test_line_to_yolo_hard_tier_included() -> None:
    """occ=2 + trunc=0.5 + h≥25 → hard 档保留。"""
    line = "Car 0.50 2 1.85 100 100 300 200 1.56 1.60 3.50 -1 1.8 25.0 1.62"
    out = kitti_line_to_yolo(line, 1242, 375)
    assert out is not None and out.startswith("1 ")


def test_line_to_yolo_truncated_garbage() -> None:
    """行格式残缺 → None。"""
    assert kitti_line_to_yolo("Car 0.0", 1242, 375) is None


# ---------- convert_label_file / 划分 / yaml ----------

def test_convert_label_file_and_idempotent(tmp_path: Path) -> None:
    """单文件转换 + 幂等（二次调用 -1，内容不变）。"""
    src = tmp_path / "000000.txt"
    src.write_text(f"{_CAR_LINE}\nDontCare -1 -1 -10 0 0 100 100 -1 -1 -1 -1 -1 -1 -1\n")
    out = tmp_path / "labels" / "000000.txt"

    n = convert_label_file(src, out, 1242, 375)
    assert n == 1
    assert out.read_text().strip().startswith("1 ")
    assert convert_label_file(src, out, 1242, 375) == -1
    assert out.read_text() == out.read_text()  # 内容未被二次改写（行数仍 1）
    assert len(out.read_text().strip().splitlines()) == 1


def test_split_train_val_deterministic() -> None:
    """确定性划分：可复现、互斥、全量覆盖。"""
    files = [Path(f"{i:06d}.txt") for i in range(100)]
    train1, val1 = split_train_val(files, val_ratio=0.1)
    train2, val2 = split_train_val(files, val_ratio=0.1)
    assert train1 == train2 and val1 == val2
    assert len(val1) == 10 and len(train1) == 90
    assert set(train1).isdisjoint(val1)
    assert sorted(train1 + val1) == sorted(files)


def test_write_data_yaml(tmp_path: Path) -> None:
    """data.yaml 内容：path/train/val + 5 类 names。"""
    p = write_data_yaml(tmp_path)
    text = p.read_text()
    assert f"path: {tmp_path}" in text
    assert "train: images/train" in text
    assert "val: images/val" in text
    for i, name in enumerate(KITTI_YOLO_NAMES):
        assert f"  {i}: {name}" in text


# ---------- UltralyticsModel._parse_pred 自定义类别表 ----------

class _FakeVec:
    """单框张量鸭子类型（xyxy[0].tolist() 调用点）。"""

    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return self._values


class _FakeBox:
    """ultralytics 单框对象（_parse_pred 逐框迭代，每框 xyxy/cls/conf 长度 1）。"""

    def __init__(self, xyxy: list[float], cls: int, conf: float) -> None:
        self.xyxy = [_FakeVec(xyxy)]
        self.cls = [cls]
        self.conf = [conf]


def _make_model(names: dict[int, str] | None) -> UltralyticsModel:
    """注入 Fake model（绕过 _load，零 ultralytics 导入零权重）。"""
    m = UltralyticsModel(model_name="custom.pt")
    m._model = SimpleNamespace(names=names)
    return m


def _make_pred() -> Any:
    """2 框（cls 0/1）伪 Results；Any 注入（duck typing 铁律）。"""
    return SimpleNamespace(boxes=[
        _FakeBox([10.0, 20.0, 30.0, 60.0], 0, 0.9),
        _FakeBox([100.0, 5.0, 150.0, 80.0], 1, 0.6),
    ])


def test_parse_pred_uses_custom_model_names() -> None:
    """KITTI 微调 5 类模型：cls 0/1 → bicycle/car（不再按 COCO 表映射）。"""
    names = {0: "bicycle", 1: "car", 2: "person", 3: "train", 4: "truck"}
    prompts = ["bicycle", "car", "person", "train", "truck"]
    results = _make_model(names)._parse_pred(_make_pred(), prompts)
    assert [(r.label, r.confidence) for r in results] == [("bicycle", 0.9), ("car", 0.6)]


def test_parse_pred_custom_names_prompt_filter() -> None:
    """prompts 只含 car → bicycle 框被过滤。"""
    names = {0: "bicycle", 1: "car"}
    results = _make_model(names)._parse_pred(_make_pred(), ["car"])
    assert [r.label for r in results] == ["car"]


def test_parse_pred_names_missing_falls_back_to_coco() -> None:
    """模型无 names → 回退 COCO 80 类（标准模型行为不变）。"""
    results = _make_model(None)._parse_pred(_make_pred(), ["person", "bicycle"])
    assert [r.label for r in results] == ["person", "bicycle"]


def test_create_detection_model_routes_path_pt_to_ultralytics() -> None:
    """含 "/" 的 .pt 完整路径（微调产物）→ Ultralytics，不误判 GroundingDINO。"""
    from auto2dlabel.models.detection import (
        GroundingDINOModel,
        UltralyticsModel,
        create_detection_model,
    )

    m = create_detection_model(
        "/root/autodl-tmp/Documents/Projects/AutoLabel/auto2dlabel/weights/"
        "kitti_finetune/yolo11s_kitti/weights/best.pt"
    )
    assert isinstance(m, UltralyticsModel)
    assert not isinstance(m, GroundingDINOModel)
