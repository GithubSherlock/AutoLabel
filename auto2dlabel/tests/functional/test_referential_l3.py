"""指代 L3（Qwen2-VL-7B）测试 —— 解析纯函数 / 阶梯升级 / 守卫 / CLI 路由。

零真实权重铁律：QwenVLReferentialResolver 预注入 FakeProcessor/FakeModel
（_load 短路，不触 GPU/权重）；Cascade 用 FakeL2/FakeL3（last_failed 信号
duck typing，同 L2 测试模式）；CLI 路由 monkeypatch 工厂。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

import auto2dlabel.models.referential_l3 as l3
from auto2dlabel.models.referential_l3 import (
    GROUNDING_PROMPT,
    CascadeReferentialResolver,
    QwenVLReferentialResolver,
    create_cascade_referential_resolver,
    create_referential_l3_resolver,
    parse_qwen_bboxes,
)
from auto2dlabel.schema.annotation import Bbox


def _bbox(x: float, y: float, w: float = 50.0, h: float = 50.0) -> Bbox:
    return Bbox(x=x, y=y, width=w, height=h, label="person", confidence=0.9)


# ---------- parse_qwen_bboxes 纯函数 ----------

def test_parse_json_bbox_1000_grid() -> None:
    """官方 JSON [0,1000] 归一化 → 原图像素（/1000）。"""
    text = '[{"bbox_2d": [100, 200, 300, 400], "label": "person"}]'
    boxes = parse_qwen_bboxes(text, orig_w=1000, orig_h=500)
    assert boxes == [(100.0, 100.0, 300.0, 200.0)]  # 1000 系 → w=1000,h=500


def test_parse_json_bbox_unit_grid() -> None:
    """[0,1] 归一化 → 像素（坐标 ≤1 判别）。"""
    text = '[{"bbox_2d": [0.1, 0.2, 0.3, 0.4]}]'
    boxes = parse_qwen_bboxes(text, orig_w=1000, orig_h=500)
    assert boxes == [(100.0, 100.0, 300.0, 200.0)]


def test_parse_box_tokens() -> None:
    """<|box_start|>(x1,y1),(x2,y2)<|box_end|> 原生 box token 形态。"""
    text = "<|box_start|>(100,200),(300,400)<|box_end|>"
    boxes = parse_qwen_bboxes(text, orig_w=1000, orig_h=1000)
    assert boxes == [(100.0, 200.0, 300.0, 400.0)]


def test_parse_bare_list_fallback() -> None:
    """裸数字列表兜底形态。"""
    boxes = parse_qwen_bboxes("[100, 200, 300, 400]", orig_w=1000, orig_h=1000)
    assert boxes == [(100.0, 200.0, 300.0, 400.0)]


def test_parse_no_box() -> None:
    """无框输出 → 空列表。"""
    assert parse_qwen_bboxes("I cannot locate the target.", 1000, 1000) == []


def test_parse_skips_degenerate() -> None:
    """退化框（x1==x2）丢弃。"""
    text = '[{"bbox_2d": [100, 200, 100, 400]}, {"bbox_2d": [10, 20, 30, 40]}]'
    boxes = parse_qwen_bboxes(text, 1000, 1000)
    assert boxes == [(10.0, 20.0, 30.0, 40.0)]


def test_grounding_prompt_requires_json() -> None:
    """官方 grounding prompt 形态：要求 bbox_2d JSON + 短语占位。"""
    assert "{phrase}" in GROUNDING_PROMPT
    assert "bbox_2d" in GROUNDING_PROMPT
    assert 'Locate "the red car"' in GROUNDING_PROMPT.format(phrase="the red car")


# ---------- 阶梯升级 Cascade ----------

class _FakeL2:
    def __init__(self, failed: bool, result: list[Bbox]) -> None:
        self._failed = failed
        self._result = result

    @property
    def last_failed(self) -> bool:
        return self._failed

    def resolve(self, image_path: str, phrase: str, bboxes: list[Bbox]) -> list[Bbox]:
        return self._result


class _FakeL3:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def resolve(self, image_path: str, phrase: str, bboxes: list[Bbox]) -> list[Bbox]:
        self.calls.append((image_path, phrase))
        return bboxes[:1]


def test_cascade_escalates_on_l2_failure() -> None:
    """L2 last_failed → 升级 L3（同参传入）。"""
    boxes = [_bbox(0, 0), _bbox(100, 100)]
    l3r = _FakeL3()
    cascade = CascadeReferentialResolver(_FakeL2(failed=True, result=boxes), l3r)
    out = cascade.resolve("img.jpg", "the person", boxes)
    assert out == boxes[:1]
    assert l3r.calls == [("img.jpg", "the person")]


def test_cascade_no_escalation_on_success() -> None:
    """L2 成功 → L3 不调用（成本红线：不白白跑 7B）。"""
    boxes = [_bbox(0, 0)]
    l3r = _FakeL3()
    cascade = CascadeReferentialResolver(_FakeL2(failed=False, result=boxes), l3r)
    assert cascade.resolve("img.jpg", "p", boxes) == boxes
    assert l3r.calls == []


def test_cascade_without_fallback_is_pure_l2() -> None:
    """fallback None（CPU 环境）→ 纯 L2 语义。"""
    boxes = [_bbox(0, 0)]
    cascade = CascadeReferentialResolver(_FakeL2(failed=True, result=boxes), None)
    assert cascade.resolve("img.jpg", "p", boxes) == boxes


# ---------- QwenVLReferentialResolver（Fake 注入，零权重） ----------

class _FakeProcessor:
    def __init__(self, text: str) -> None:
        self.text = text

    def apply_chat_template(
        self, messages: Any, add_generation_prompt: bool = False, tokenize: bool = False
    ) -> str:
        return "<image>locate"

    def __call__(
        self, text: Any, images: Any, return_tensors: str = "pt"
    ) -> dict[str, Any]:
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "pixel_values": torch.ones(1, 3, 8, 8),
        }

    def batch_decode(self, ids: Any, skip_special_tokens: bool = False) -> list[str]:
        return [self.text]


class _FakeModel:
    def __init__(self, raise_on_generate: bool = False) -> None:
        self._boom = raise_on_generate

    def generate(self, **kwargs: Any) -> Any:
        if self._boom:
            raise RuntimeError("oom")
        return torch.tensor([[1, 2, 3, 7, 8]])


def _make_img(tmp_path: Path, w: int = 1000, h: int = 1000) -> str:
    from PIL import Image

    p = tmp_path / "frame.jpg"
    Image.new("RGB", (w, h)).save(p)
    return str(p)


def _injected_resolver(
    text: str, raise_on_generate: bool = False
) -> QwenVLReferentialResolver:
    """预注入 Fake processor/model（_load 短路，零 GPU 零权重）。"""
    r = QwenVLReferentialResolver()
    r._processor = _FakeProcessor(text)
    r._model = _FakeModel(raise_on_generate=raise_on_generate)
    return r


def test_resolve_matches_by_center(tmp_path: Path) -> None:
    """生成 JSON 框 → 中心距匹配回原框（保原框）。"""
    boxes = [_bbox(90, 190, 40, 40), _bbox(500, 500, 40, 40)]  # 中心 110,210 / 520,520
    text = '[{"bbox_2d": [100, 200, 120, 220], "label": "person"}]'  # 中心 110,210
    r = _injected_resolver(text)
    out = r.resolve(_make_img(tmp_path), "the person", boxes)
    assert out == [boxes[0]]
    assert r.last_failed is False


def test_resolve_failure_keeps_all(tmp_path: Path) -> None:
    """生成异常 → 宁多勿漏返回全部 + last_failed=True（升级信号）。"""
    boxes = [_bbox(0, 0), _bbox(100, 100)]
    r = _injected_resolver("whatever", raise_on_generate=True)
    assert r.resolve(_make_img(tmp_path), "p", boxes) == boxes
    assert r.last_failed is True


def test_resolve_no_bbox_keeps_all_and_fails(tmp_path: Path) -> None:
    """无框输出 → 返回全部 + last_failed。"""
    boxes = [_bbox(0, 0)]
    r = _injected_resolver("not found")
    assert r.resolve(_make_img(tmp_path), "p", boxes) == boxes
    assert r.last_failed is True


def test_resolve_empty_candidates(tmp_path: Path) -> None:
    """空候选 → 空返回，不计失败。"""
    r = _injected_resolver("x")
    assert r.resolve(_make_img(tmp_path), "p", []) == []
    assert r.last_failed is False


def test_load_requires_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """无 CUDA → RuntimeError（CPU 分钟级不可用红线）。"""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="GPU"):
        QwenVLReferentialResolver()._load()


def test_load_requires_cached_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """权重未就位 → 快速失败并提示下载脚本（不静默触发 16GB 下载）。"""
    monkeypatch.setattr(l3, "L3_CACHE_DIR", tmp_path / "nonexistent")
    with pytest.raises(RuntimeError, match="download_qwen_l3"):
        QwenVLReferentialResolver()._load()


# ---------- 工厂 ----------

def test_l3_factory_zero_load() -> None:
    """工厂构造零加载（无 GPU/权重依赖）。"""
    r = create_referential_l3_resolver()
    assert isinstance(r, QwenVLReferentialResolver)
    assert r._processor is None and r._model is None


def test_cascade_factory_builds_l2_then_l3() -> None:
    """阶梯工厂：primary = L2（Florence）、fallback = L3（均零加载）。"""
    from auto2dlabel.models.referential import Florence2ReferentialResolver

    c = create_cascade_referential_resolver()
    assert isinstance(c, CascadeReferentialResolver)
    assert isinstance(c._primary, Florence2ReferentialResolver)
    assert isinstance(c._fallback, QwenVLReferentialResolver)


# ---------- CLI 路由（monkeypatch 工厂，零权重） ----------

def _patch_cli_track(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """monkeypatch cli_track：resolve_plan 返回空约束、TrackingTool 捕获 kwargs。"""
    from auto2dlabel import cli_track
    from auto2dlabel.tools.constraints import ReferentialConstraint

    captured: dict[str, Any] = {}

    class _FakeTool:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def forward(
            self, source: str, prompts: Any, confidence_threshold: float
        ) -> None:
            pass

    monkeypatch.setattr(
        cli_track, "resolve_plan",
        lambda *a, **k: ReferentialConstraint(prompts=["person"]),
    )
    monkeypatch.setattr(cli_track, "TrackingTool", _FakeTool)
    return captured


def _run_track(tmp_path: Path, **kwargs: Any) -> None:
    from auto2dlabel import cli_track

    cli_track.run_tracking(
        source=str(tmp_path),
        instruction=kwargs.pop("instruction", "跟踪红车旁边的行人"),
        threshold=0.3,
        iou=0.3,
        export="mot",
        output=str(tmp_path),
        det_model=None,
        sahi=False,
        verbose=False,
        **kwargs,
    )


def test_cli_track_refer_l3_routes_l3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--refer-l3 → 直用 L3 工厂（不建 cascade）。"""
    captured = _patch_cli_track(monkeypatch)
    monkeypatch.setattr(l3, "create_referential_l3_resolver", _FakeL3)

    _run_track(tmp_path, refer_l3=True)
    assert isinstance(captured["referential"], _FakeL3)
    assert captured["referential_phrase"] is not None


def test_cli_track_relation_auto_routes_cascade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """关系词（旁边）自动触发 → 阶梯工厂（默认指代路径）。"""
    class _FakeCascade:
        pass

    captured = _patch_cli_track(monkeypatch)
    monkeypatch.setattr(l3, "create_cascade_referential_resolver", _FakeCascade)

    _run_track(tmp_path, refer_l3=False, refer_l2=False)
    assert isinstance(captured["referential"], _FakeCascade)


def test_cli_track_plain_instruction_no_referential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """无关系词无 flag → 无指代解析器（零成本路径不受 L3 改动影响）。"""
    captured = _patch_cli_track(monkeypatch)
    _run_track(tmp_path, instruction="检测行人", refer_l3=False, refer_l2=False)
    assert captured["referential"] is None
    assert captured["referential_phrase"] is None
