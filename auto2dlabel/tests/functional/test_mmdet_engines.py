"""mmdet 引擎测试（v0.6 Phase 3a，零真实权重——Fake 注入 mmdet 协议 + 路由）。

覆盖：ImportError 守卫（未装 mmdet 可实例化、首次推理才报错提示安装）；
缺文件提示下载脚本；_parse_detections / _parse_masks 纯函数（置信度/prompt/
越界标签过滤、bbox 换算、mask→polygon）；幂等 _load；设备规整 cuda→cuda:0；
工厂路由（rtmdet / mask2former + 裸别名默认档）。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import auto2dlabel.models.mmdet_engines as engines
from auto2dlabel.models.detection import create_detection_model
from auto2dlabel.models.segmentation import create_segmentation_model
from auto2dlabel.schema.annotation import Bbox
from auto2dlabel.tests import Path, np, sys


class _FakeInit:
    """fake init_detector：记录调用参数，返回固定模型对象。"""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.model: Any = object()

    def __call__(self, config: Any, checkpoint: Any = None, device: str = "cuda:0") -> Any:
        self.calls.append((config, checkpoint, device))
        return self.model


class _FakeInfer:
    """fake inference_detector：按预设 DetDataSample 返回，记录 (model, image_path)。"""

    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[tuple[Any, Any]] = []

    def __call__(self, model: Any, image_path: Any) -> Any:
        self.calls.append((model, image_path))
        return self.result


def _patch_mmdet(monkeypatch: pytest.MonkeyPatch, result: Any) -> tuple[_FakeInit, _FakeInfer]:
    fake_init, fake_infer = _FakeInit(), _FakeInfer(result)
    monkeypatch.setattr(engines, "_import_mmdet", lambda: (fake_init, fake_infer))
    return fake_init, fake_infer


def _prepare_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """把 config/权重目录指向 tmp（空文件即可——_require_files 只看存在性）。"""
    monkeypatch.setattr(engines, "MMDET_CONFIGS_DIR", tmp_path / "configs")
    monkeypatch.setattr(engines, "MMDET_CKPT_DIR", tmp_path / "ckpts")
    (tmp_path / "configs" / "rtmdet").mkdir(parents=True)
    (tmp_path / "configs" / "mask2former").mkdir(parents=True)
    cfg_rtmdet = "rtmdet/rtmdet_l_8xb32-300e_coco.py"
    (tmp_path / "configs" / cfg_rtmdet).write_text("# fake")
    cfg_m2f = "mask2former/mask2former_r50_8xb2-lsj-50e_coco.py"
    (tmp_path / "configs" / cfg_m2f).write_text("# fake")
    ckpt_rtmdet = "rtmdet_l/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth"
    (tmp_path / "ckpts" / ckpt_rtmdet).parent.mkdir(parents=True)
    (tmp_path / "ckpts" / ckpt_rtmdet).write_bytes(b"")
    ckpt_m2f = (
        "mask2former_r50_8xb2-lsj-50e_coco/"
        "mask2former_r50_8xb2-lsj-50e_coco_20220506_191028-41b088b6.pth"
    )
    (tmp_path / "ckpts" / ckpt_m2f).parent.mkdir(parents=True)
    (tmp_path / "ckpts" / ckpt_m2f).write_bytes(b"")


def _pred_instances(
    bboxes: Any = None, scores: Any = None, labels: Any = None, masks: Any = None,
) -> Any:
    return SimpleNamespace(
        bboxes=None if bboxes is None else np.asarray(bboxes, dtype=float),
        scores=None if scores is None else np.asarray(scores, dtype=float),
        labels=None if labels is None else np.asarray(labels, dtype=int),
        masks=masks,
    )


# ============ ImportError 守卫 / 缺文件提示 ============


def test_rtmdet_import_error_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """未装 mmdet：detect 首次推理抛 ImportError 带安装提示（走真实守卫分支）。"""
    monkeypatch.setitem(sys.modules, "mmdet.apis", None)  # 强制 import mmdet.apis 失败
    model = engines.MMDetRTMDetModel(device="cpu")
    with pytest.raises(ImportError, match="pip install mmdet"):
        model.detect("a.jpg", ["car"])


def test_rtmdet_missing_files_hint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """config/权重未下载 → FileNotFoundError 提示下载脚本（防深埋 init 报错）。"""
    monkeypatch.setattr(engines, "MMDET_CONFIGS_DIR", tmp_path / "configs")
    monkeypatch.setattr(engines, "MMDET_CKPT_DIR", tmp_path / "ckpts")
    _patch_mmdet(monkeypatch, None)
    model = engines.MMDetRTMDetModel(device="cpu")
    with pytest.raises(FileNotFoundError, match="download_mmdet_weights.sh"):
        model.detect("a.jpg", ["car"])


def test_resolve_paths_unknown_name() -> None:
    """未知模型名 → ValueError 列出可用名。"""
    with pytest.raises(ValueError, match="rtmdet_l"):
        engines._resolve_paths("rtmdet-unknown", engines.RTMDET_MODELS)


# ============ RTMDet detect ============


def test_rtmdet_detect_parses_boxes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pred_instances → DetectionResult：xyxy→x/y/w/h、类别名映射、设备 cuda→cuda:0。"""
    _prepare_files(monkeypatch, tmp_path)
    fake_init, _ = _patch_mmdet(monkeypatch, SimpleNamespace(pred_instances=_pred_instances(
        bboxes=[[10.0, 20.0, 60.0, 80.0], [0.0, 0.0, 5.0, 5.0]],
        scores=[0.9, 0.1],
        labels=[2, 5],
    )))
    model = engines.MMDetRTMDetModel(model_name="rtmdet_l", device="cuda")
    dets = model.detect("a.jpg", ["car"], confidence_threshold=0.3)

    assert len(dets) == 1  # 第二框 conf 0.1 < 0.3 被过滤
    d = dets[0]
    assert (d.x, d.y, d.width, d.height) == (10.0, 20.0, 50.0, 60.0)
    assert d.label == "car" and d.confidence == pytest.approx(0.9)
    assert fake_init.calls == [(
        str(tmp_path / "configs" / "rtmdet" / "rtmdet_l_8xb32-300e_coco.py"),
        str(tmp_path / "ckpts" / "rtmdet_l"
            / "rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth"),
        "cuda:0",
    )]


def test_rtmdet_prompt_filter_and_empty_prompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """prompt 过滤（person 过滤 car）；空 prompts 不过滤（全类输出）。"""
    _prepare_files(monkeypatch, tmp_path)
    _patch_mmdet(monkeypatch, SimpleNamespace(pred_instances=_pred_instances(
        bboxes=[[1.0, 2.0, 3.0, 4.0]],
        scores=[0.9],
        labels=[2],  # car
    )))
    model = engines.MMDetRTMDetModel(device="cpu")
    assert model.detect("a.jpg", ["person"]) == []
    assert len(model.detect("a.jpg", [])) == 1


def test_rtmdet_out_of_range_label_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """标签越界（≥80）→ 丢弃而非崩溃。"""
    _prepare_files(monkeypatch, tmp_path)
    _patch_mmdet(monkeypatch, SimpleNamespace(pred_instances=_pred_instances(
        bboxes=[[1.0, 2.0, 3.0, 4.0]], scores=[0.9], labels=[99],
    )))
    assert engines.MMDetRTMDetModel(device="cpu").detect("a.jpg", []) == []


def test_rtmdet_empty_pred_instances(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """空 pred_instances → []（无 bboxes 属性）。"""
    _prepare_files(monkeypatch, tmp_path)
    _patch_mmdet(monkeypatch, SimpleNamespace(pred_instances=SimpleNamespace()))
    assert engines.MMDetRTMDetModel(device="cpu").detect("a.jpg", []) == []


def test_rtmdet_idempotent_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """幂等 _load：两次 detect 只 init 一次（同模型实例复用）。"""
    _prepare_files(monkeypatch, tmp_path)
    fake_init, _ = _patch_mmdet(monkeypatch, SimpleNamespace(pred_instances=SimpleNamespace()))
    model = engines.MMDetRTMDetModel(device="cpu")
    model.detect("a.jpg", [])
    model.detect("a.jpg", [])
    assert len(fake_init.calls) == 1


# ============ Mask2Former generate ============


def test_mask2former_generate_ignores_bboxes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """自带检测模式：generate 忽略输入 bboxes（inference 只收 model + image_path）。"""
    _prepare_files(monkeypatch, tmp_path)
    square = np.zeros((4, 4), dtype=bool)
    square[1:3, 1:3] = True
    _, fake_infer = _patch_mmdet(monkeypatch, SimpleNamespace(pred_instances=_pred_instances(
        bboxes=[[1.0, 1.0, 3.0, 3.0]], scores=[0.8], labels=[2], masks=np.array([square]),
    )))
    model = engines.MMDetMask2FormerModel(device="cpu")
    masks = model.generate("a.jpg", [Bbox(x=0, y=0, width=1, height=1, label="x", confidence=0.9)])

    assert len(masks) == 1  # 输入 1 个 bbox 但模型自检，输出只看 pred_instances
    assert len(fake_infer.calls) == 1
    assert fake_infer.calls[0][1] == "a.jpg"  # inference 只收 (model, image_path)，无 bboxes


def test_mask2former_parses_masks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """masks → Mask：polygon 非空、bbox 换算、area=mask 面积、conf<0.5 过滤。"""
    _prepare_files(monkeypatch, tmp_path)
    square = np.zeros((4, 4), dtype=bool)
    square[1:3, 1:3] = True
    _patch_mmdet(monkeypatch, SimpleNamespace(pred_instances=_pred_instances(
        bboxes=[[1.0, 2.0, 5.0, 6.0], [0.0, 0.0, 1.0, 1.0]],
        scores=[0.8, 0.4],
        labels=[2, 5],
        masks=np.array([square, square]),
    )))
    masks = engines.MMDetMask2FormerModel(device="cpu").generate("a.jpg", [])

    assert len(masks) == 1  # conf 0.4 < 0.5 被过滤
    m = masks[0]
    assert m.bbox.label == "car"
    assert (m.bbox.x, m.bbox.y, m.bbox.width, m.bbox.height) == (1.0, 2.0, 4.0, 4.0)
    assert m.area == pytest.approx(4.0)
    assert m.segmentation and all(len(poly) >= 6 for poly in m.segmentation)  # 闭合 polygon


def test_mask2former_empty_masks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """无 masks → []。"""
    _prepare_files(monkeypatch, tmp_path)
    _patch_mmdet(monkeypatch, SimpleNamespace(pred_instances=_pred_instances(
        bboxes=[[1.0, 2.0, 3.0, 4.0]], scores=[0.9], labels=[2],
    )))
    assert engines.MMDetMask2FormerModel(device="cpu").generate("a.jpg", []) == []


# ============ 工厂路由 + 零加载 ============


def test_factory_routes_rtmdet() -> None:
    """create_detection_model("rtmdet_l"/裸 "rtmdet") → MMDetRTMDetModel（默认 l 档）。"""
    model = create_detection_model("rtmdet_l", iou_threshold=0.7)
    assert isinstance(model, engines.MMDetRTMDetModel)
    assert model._model_name == "rtmdet_l" and model._iou == 0.7

    bare = create_detection_model("rtmdet")
    assert isinstance(bare, engines.MMDetRTMDetModel)
    assert bare._model_name == engines.MMDetRTMDetModel.DEFAULT_NAME


def test_factory_routes_mask2former() -> None:
    """create_segmentation_model("mask2former*") → MMDetMask2FormerModel（裸别名默认 R50）。"""
    model = create_segmentation_model("mask2former_r50_8xb2-lsj-50e_coco")
    assert isinstance(model, engines.MMDetMask2FormerModel)
    assert model._model_name == "mask2former_r50_8xb2-lsj-50e_coco"

    bare = create_segmentation_model("mask2former")
    assert isinstance(bare, engines.MMDetMask2FormerModel)
    assert bare._model_name == engines.MMDetMask2FormerModel.DEFAULT_NAME


def test_factory_zero_load_without_mmdet(monkeypatch: pytest.MonkeyPatch) -> None:
    """零加载：mmdet 未安装时工厂仍可实例化（构造不触发 ImportError）。"""
    monkeypatch.setattr(
        engines, "_import_mmdet",
        lambda: (_ for _ in ()).throw(ImportError("mmdet 未安装")),
    )
    model = create_detection_model("rtmdet_l")  # 不应抛 ImportError
    assert isinstance(model, engines.MMDetRTMDetModel)
    assert model._model is None  # 未加载


def test_device_mps_passthrough(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """非 "cuda" 设备（mps）原样透传 init_detector。"""
    _prepare_files(monkeypatch, tmp_path)
    fake_init, _ = _patch_mmdet(monkeypatch, SimpleNamespace(pred_instances=SimpleNamespace()))
    engines.MMDetRTMDetModel(device="mps").detect("a.jpg", [])
    assert fake_init.calls[0][2] == "mps"


# ============ weights_only 兼容（torch>=2.6） ============


def test_allow_legacy_checkpoint_globals(tmp_path: Path) -> None:
    """回归：白名单放行含 mmengine 全局的老 checkpoint（torch 2.6+ weights_only）。

    2022 RTMDet 权重在 torch 2.13 默认 weights_only=True 下加载失败
    （Unsupported global: mmengine.logging.history_buffer.HistoryBuffer，
    benchmark 实测报错记录见 test-v0.6.md）；白名单后同构 checkpoint 应可载。
    （「白名单前拒载」为 torch 默认行为，是本次故障的事实前提——
    若白名单被删/改坏，本测试与 test_load_whitelists_legacy_globals_before_init
    组合会红。）
    """
    torch = pytest.importorskip("torch")
    mmengine_logging = pytest.importorskip("mmengine.logging")
    if not hasattr(torch.serialization, "add_safe_globals"):
        pytest.skip("torch < 2.6：无 weights_only 默认策略")
    ckpt = tmp_path / "legacy.pth"
    torch.save({"hb": mmengine_logging.HistoryBuffer()}, ckpt)
    engines._allow_legacy_checkpoint_globals()
    loaded = torch.load(ckpt, weights_only=True)  # 白名单后：可载
    assert isinstance(loaded["hb"], mmengine_logging.HistoryBuffer)


def test_load_whitelists_legacy_globals_before_init(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RTMDet _load 在 init_detector 前执行白名单（顺序敏感——先放行后加载）。"""
    _prepare_files(monkeypatch, tmp_path)
    order: list[str] = []

    monkeypatch.setattr(
        engines, "_allow_legacy_checkpoint_globals",
        lambda: order.append("whitelist"),
    )

    def fake_import() -> tuple[Any, Any]:
        def init(config: Any, checkpoint: Any = None, device: str = "cuda:0") -> Any:
            order.append("init_detector")
            return object()

        return init, _FakeInfer(None)

    monkeypatch.setattr(engines, "_import_mmdet", fake_import)
    engines.MMDetRTMDetModel(device="cpu")._load()
    assert order == ["whitelist", "init_detector"]
