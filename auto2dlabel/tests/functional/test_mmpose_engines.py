"""mmpose 引擎测试（v0.6 Phase 3b，零真实权重——Fake 注入 mmpose 协议 + 路由）。

覆盖：ImportError 守卫（未装 mmpose 可实例化、首次推理才报错提示安装）；
缺文件提示下载脚本；_parse_topdown 纯函数（17 点截断/补齐、v 语义、bbox 换算）；
两段式 detect_pose（person 过滤、xyxy 检测框传入 inference_topdown、逐框对齐）；
幂等 _load；设备规整 cuda→cuda:0；工厂路由（rtmpose 前缀 + 裸别名默认档）。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import auto2dlabel.models.mmpose_engines as engines
from auto2dlabel.models.detection import DetectionResult
from auto2dlabel.models.mmdet_engines import _allow_legacy_checkpoint_globals
from auto2dlabel.models.pose import create_pose_model
from auto2dlabel.tests import Path, np, sys


class _FakeInit:
    """fake init_model：记录调用参数，返回固定模型对象。"""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.model: Any = object()

    def __call__(self, config: Any, checkpoint: Any = None, device: str = "cuda:0") -> Any:
        self.calls.append((config, checkpoint, device))
        return self.model


class _FakeInfer:
    """fake inference_topdown：按 bboxes 逐框返回预置 PoseDataSample。"""

    def __init__(self, pred_instances: Any) -> None:
        self.pred_instances = pred_instances
        self.calls: list[tuple[Any, ...]] = []

    def __call__(self, model: Any, image_path: Any, bboxes: Any = None,
                 bbox_format: str = "xyxy") -> list[Any]:
        self.calls.append((model, image_path, bboxes, bbox_format))
        n = len(bboxes) if bboxes is not None else 0
        return [SimpleNamespace(pred_instances=self.pred_instances) for _ in range(n)]


class _FakeDetector:
    """fake person 检测器：记录 detect 调用，返回预置 DetectionResult 列表。"""

    def __init__(self, dets: list[DetectionResult]) -> None:
        self.dets = dets
        self.calls: list[tuple[Any, ...]] = []

    def detect(self, image_path: Any, prompts: Any,
               confidence_threshold: float = 0.3) -> list[DetectionResult]:
        self.calls.append((image_path, prompts, confidence_threshold))
        return list(self.dets)


def _patch_mmpose(monkeypatch: pytest.MonkeyPatch, pred_instances: Any,
                  ) -> tuple[_FakeInit, _FakeInfer]:
    fake_init, fake_infer = _FakeInit(), _FakeInfer(pred_instances)
    monkeypatch.setattr(engines, "_import_mmpose", lambda: (fake_init, fake_infer))
    return fake_init, fake_infer


def _patch_mmpose_probe(monkeypatch: pytest.MonkeyPatch, kpts: Any,
                        ) -> tuple[_FakeInit, _FakeInfer]:
    """_patch_mmpose + 17 点全置信 keypoints 的 pred_instances（多数用例共用）。"""
    scores = [0.9] * len(kpts)
    return _patch_mmpose(monkeypatch, _pred_instances(kpts, scores))


def _patch_detector(monkeypatch: pytest.MonkeyPatch,
                    dets: list[DetectionResult]) -> _FakeDetector:
    fake = _FakeDetector(dets)
    monkeypatch.setattr(
        engines.MMposeRTMPoseModel, "_load_detector", lambda self: fake,
    )
    return fake


def _prepare_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """把 config/权重目录指向 tmp（空文件即可——_require_files 只看存在性）。"""
    monkeypatch.setattr(engines, "MMPOSE_CONFIGS_DIR", tmp_path / "configs")
    monkeypatch.setattr(engines, "MMPOSE_CKPT_DIR", tmp_path / "ckpts")
    cfg_dir = tmp_path / "configs" / "body_2d_keypoint" / "rtmpose" / "coco"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "rtmpose-l_8xb256-420e_coco-256x192.py").write_text("# fake")
    ckpt = "rtmpose_l/rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.pth"
    (tmp_path / "ckpts" / ckpt).parent.mkdir(parents=True)
    (tmp_path / "ckpts" / ckpt).write_bytes(b"")


def _pred_instances(keypoints: Any = None, scores: Any = None) -> Any:
    return SimpleNamespace(
        keypoints=None if keypoints is None else np.asarray(keypoints, dtype=float),
        keypoint_scores=None if scores is None else np.asarray(scores, dtype=float),
    )


def _person(x: float = 0, y: float = 0, w: float = 10, h: float = 20) -> DetectionResult:
    return DetectionResult(x=x, y=y, width=w, height=h, label="person", confidence=0.9)


# ============ ImportError 守卫 / 缺文件提示 ============


def test_rtmpose_import_error_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """未装 mmpose：detect_pose 首次推理抛 ImportError 带安装提示（真实守卫分支）。"""
    monkeypatch.setitem(sys.modules, "mmpose.apis", None)  # 强制 import mmpose.apis 失败
    model = engines.MMposeRTMPoseModel(device="cpu")
    with pytest.raises(ImportError, match="pip install mmpose"):
        model.detect_pose("a.jpg", ["person"])


def test_rtmpose_missing_files_hint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """config/权重未下载 → FileNotFoundError 提示下载脚本（防深埋 init 报错）。"""
    monkeypatch.setattr(engines, "MMPOSE_CONFIGS_DIR", tmp_path / "configs")
    monkeypatch.setattr(engines, "MMPOSE_CKPT_DIR", tmp_path / "ckpts")
    _patch_mmpose(monkeypatch, None)
    model = engines.MMposeRTMPoseModel(device="cpu")
    with pytest.raises(FileNotFoundError, match="download_mmpose_weights.sh"):
        model.detect_pose("a.jpg", ["person"])


def test_resolve_paths_unknown_name() -> None:
    """未知模型名 → ValueError 列出可用名。"""
    with pytest.raises(ValueError, match="rtmpose_l"):
        engines._resolve_paths("rtmpose-unknown", engines.RTMPOSE_MODELS)


# ============ _parse_topdown 纯函数 ============


def test_parse_topdown_full_keypoints() -> None:
    """17 点全量：像素坐标透传、v=1/0 按 score、bbox→x/y/w/h、label=person。"""
    kpts = [[float(i), float(i * 2)] for i in range(17)]
    scores = [0.9 if i % 2 == 0 else 0.0 for i in range(17)]
    r = engines._parse_topdown(
        _pred_instances(kpts, scores), (10.0, 20.0, 30.0, 50.0), 0.8,
    )
    assert (r.x, r.y, r.width, r.height) == (10.0, 20.0, 20.0, 30.0)
    assert r.label == "person" and r.confidence == pytest.approx(0.8)
    assert len(r.keypoints) == 17
    assert r.keypoints[0] == (0.0, 0.0, 1.0)
    assert r.keypoints[1] == (1.0, 2.0, 0.0)  # score=0 → v=0
    assert r.keypoints[16] == (16.0, 32.0, 1.0)


def test_parse_topdown_pads_short_keypoints() -> None:
    """K<17 → 补齐 (0,0,0)；K>17 → 截断（防越界崩溃，宁多勿漏）。"""
    kpts = [[1.0, 2.0]] * 20
    scores = [0.7] * 20
    r = engines._parse_topdown(_pred_instances(kpts, scores), (0, 0, 1, 1), 0.5)
    assert len(r.keypoints) == 17
    assert r.keypoints[0] == (1.0, 2.0, 1.0)  # 截断保留前 17
    assert r.keypoints[16] == (1.0, 2.0, 1.0)

    r2 = engines._parse_topdown(
        _pred_instances([[1.0, 2.0], [3.0, 4.0]], [0.6, 0.6]), (0, 0, 1, 1), 0.5,
    )
    assert r2.keypoints[2:] == [(0.0, 0.0, 0.0)] * 15  # 补齐


def test_parse_topdown_empty_keypoints() -> None:
    """无 keypoints 属性 → 全 (0,0,0) 且 v=0（容错不崩溃）。"""
    r = engines._parse_topdown(SimpleNamespace(), (0, 0, 1, 1), 0.5)
    assert r.keypoints == [(0.0, 0.0, 0.0)] * 17


def test_parse_topdown_takes_first_instance_of_nd_tensor() -> None:
    """回归：真实 mmpose 输出 (N,K,2)/(N,K) 多人张量 → 取首实例。

    inference_topdown 逐框调用 N=1，但输出仍是三维——
    曾按 (K,2) 处理致 float() 崩溃（benchmark 实测，见 test-v0.6.md）。
    """
    kpts = np.zeros((2, 17, 2), dtype=float)
    kpts[0, :, 0] = np.arange(17.0)  # 首实例 x = 索引
    kpts[1] = 99.0  # 次实例（应被忽略）
    scores = np.zeros((2, 17), dtype=float)
    scores[0] = 0.9
    r = engines._parse_topdown(_pred_instances(kpts, scores), (0, 0, 1, 1), 0.5)
    assert len(r.keypoints) == 17
    assert r.keypoints[0] == (0.0, 0.0, 1.0)
    assert r.keypoints[16] == (16.0, 0.0, 1.0)  # 首实例数据，非 99.0


# ============ detect_pose 两段式 ============


def test_detect_pose_two_stage_person_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """两段式：person 检测 → 只把 person 框传给 inference_topdown（xyxy 格式）。"""
    _prepare_files(monkeypatch, tmp_path)
    kpts = [[float(i), 0.0] for i in range(17)]
    fake_init, fake_infer = _patch_mmpose_probe(monkeypatch, kpts)
    dets = [_person(1, 2, 10, 20), _person(50, 60, 5, 5),
            DetectionResult(x=0, y=0, width=1, height=1, label="car", confidence=0.9)]
    fake_det = _patch_detector(monkeypatch, dets)

    results = engines.MMposeRTMPoseModel(device="cuda").detect_pose(
        "a.jpg", ["person", "car"], confidence_threshold=0.3,
    )

    assert len(results) == 2  # car 被过滤，只出 person
    # inference_topdown 只收 2 个 person 框，xyxy 格式
    assert len(fake_infer.calls) == 1
    _, img, boxes, fmt = fake_infer.calls[0]
    assert img == "a.jpg" and fmt == "xyxy"
    assert boxes == [[1.0, 2.0, 11.0, 22.0], [50.0, 60.0, 55.0, 65.0]]
    # 结果与检测框逐框对齐（bbox/置信度透传，keypoints 来自 mmpose）
    assert (results[0].x, results[0].y) == (1.0, 2.0)
    assert results[1].confidence == pytest.approx(0.9)
    assert results[0].keypoints[0] == (0.0, 0.0, 1.0)
    # 检测器透传原始 prompts + conf
    assert fake_det.calls == [("a.jpg", ["person", "car"], 0.3)]
    # 设备规整：cuda→cuda:0
    assert fake_init.calls[0][2] == "cuda:0"


def test_detect_pose_no_person_short_circuits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """检测无 person → 不调 inference_topdown，直接空返回。"""
    _prepare_files(monkeypatch, tmp_path)
    _, fake_infer = _patch_mmpose(monkeypatch, _pred_instances([[0, 0]], [1.0]))
    _patch_detector(monkeypatch, [DetectionResult(
        x=0, y=0, width=1, height=1, label="car", confidence=0.9,
    )])
    results = engines.MMposeRTMPoseModel(device="cpu").detect_pose("a.jpg", ["car"])
    assert results == []
    assert fake_infer.calls == []


def test_detect_pose_empty_prompts_defaults_person(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """空 prompts → 检测器兜底 ["person"]（top-down 姿态只出 person）。"""
    _prepare_files(monkeypatch, tmp_path)
    _patch_mmpose_probe(monkeypatch, [[float(i), 0.0] for i in range(17)])
    fake_det = _patch_detector(monkeypatch, [_person()])
    engines.MMposeRTMPoseModel(device="cpu").detect_pose("a.jpg", [])
    assert fake_det.calls[0][1] == ["person"]


def test_rtmpose_idempotent_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """幂等 _load：两次 detect_pose 只 init_model 一次。"""
    _prepare_files(monkeypatch, tmp_path)
    fake_init, _ = _patch_mmpose_probe(monkeypatch, [[0.0, 0.0]] * 17)
    _patch_detector(monkeypatch, [_person()])
    model = engines.MMposeRTMPoseModel(device="cpu")
    model.detect_pose("a.jpg", ["person"])
    model.detect_pose("a.jpg", ["person"])
    assert len(fake_init.calls) == 1


# ============ weights_only 兼容（torch>=2.6） ============


def test_allow_legacy_checkpoint_globals(tmp_path: Path) -> None:
    """回归：白名单放行含 numpy 全局的老 checkpoint（torch 2.6+ weights_only）。

    2023 RTMPose 权重在 torch 2.13 默认 weights_only=True 下加载失败
    （Unsupported global: numpy.core.multiarray._reconstruct，benchmark
    实测报错记录见 test-v0.6.md）；白名单后同构 checkpoint 应可载。
    （「白名单前拒载」为 torch 默认行为，是本次故障的事实前提——
    若白名单被删/改坏，本测试与 test_load_whitelists_legacy_globals_before_init
    组合会红。）
    """
    torch = pytest.importorskip("torch")
    if not hasattr(torch.serialization, "add_safe_globals"):
        pytest.skip("torch < 2.6：无 weights_only 默认策略")
    ckpt = tmp_path / "legacy.pth"
    torch.save({"arr": np.zeros(3, dtype=np.float32)}, ckpt)
    _allow_legacy_checkpoint_globals()
    loaded = torch.load(ckpt, weights_only=True)  # 白名单后：可载
    assert loaded["arr"].shape == (3,)


def test_load_whitelists_legacy_globals_before_init(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_load 在 init_model 前执行白名单（顺序敏感——先放行后加载）。"""
    _prepare_files(monkeypatch, tmp_path)
    order: list[str] = []

    monkeypatch.setattr(
        engines, "_allow_legacy_checkpoint_globals",
        lambda: order.append("whitelist"),
    )

    def fake_import() -> tuple[Any, Any]:
        def init(config: Any, checkpoint: Any = None, device: str = "cuda:0") -> Any:
            order.append("init_model")
            return object()

        return init, _FakeInfer(_pred_instances([[0.0, 0.0]] * 17))

    monkeypatch.setattr(engines, "_import_mmpose", fake_import)
    engines.MMposeRTMPoseModel(device="cpu")._load()
    assert order == ["whitelist", "init_model"]


# ============ 工厂路由 + 零加载 ============


def test_factory_routes_rtmpose() -> None:
    """create_pose_model("rtmpose_l"/裸 "rtmpose") → MMposeRTMPoseModel（默认 l 档）。"""
    model = create_pose_model("rtmpose_l", iou_threshold=0.7)
    assert isinstance(model, engines.MMposeRTMPoseModel)
    assert model._model_name == "rtmpose_l" and model._iou == 0.7

    bare = create_pose_model("rtmpose")
    assert isinstance(bare, engines.MMposeRTMPoseModel)
    assert bare._model_name == engines.MMposeRTMPoseModel.DEFAULT_NAME


def test_factory_zero_load_without_mmpose(monkeypatch: pytest.MonkeyPatch) -> None:
    """零加载：mmpose 未安装时工厂仍可实例化（构造不触发 ImportError）。"""
    monkeypatch.setattr(
        engines, "_import_mmpose",
        lambda: (_ for _ in ()).throw(ImportError("mmpose 未安装")),
    )
    model = create_pose_model("rtmpose_l")  # 不应抛 ImportError
    assert isinstance(model, engines.MMposeRTMPoseModel)
    assert model._model is None  # 未加载
