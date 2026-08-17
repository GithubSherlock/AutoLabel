"""benchmarks/common.py 批量 helper 测试（无 GPU / 无模型依赖）。

覆盖：
- *_batch_or_fallback 批量路径 == 逐图路径逐位相等（parity）
- 无批量方法 → 逐图回退
- 批量 OOM → 当前块降级逐图 + empty_cache 被调
- 非 OOM 异常向上传播
- 单图块（len=1）不触发批量
- sample_image_paths 探针采样
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from auto2dlabel.benchmarks.common import (
    classify_batch_or_fallback,
    detect_batch_or_fallback,
    generate_batch_or_fallback,
    obb_batch_or_fallback,
    sample_image_paths,
)

# ── Fake 模型：批量路径与逐图路径产出可区分的记录 ──────────────


class _FakeDetModel:
    """detect/detect_batch 返回按路径可区分的列表（parity 校验用）。"""

    def __init__(self, oom_on_batch: bool = False) -> None:
        self.oom_on_batch = oom_on_batch
        self.batch_calls: list[int] = []
        self.single_calls: list[str] = []

    def detect(self, path: str, prompts: list[str], confidence_threshold: float = 0.3) -> list[Any]:
        self.single_calls.append(path)
        return [{"path": path, "prompts": prompts, "conf": confidence_threshold}]

    def detect_batch(
        self, paths: list[str], prompts: list[str],
        confidence_threshold: float = 0.3, num_workers: int = 0,
    ) -> list[list[Any]]:
        self.batch_calls.append(len(paths))
        if self.oom_on_batch:
            raise RuntimeError("CUDA out of memory")
        return [self.detect(p, prompts, confidence_threshold) for p in paths]


class _FakeClsModel:
    def __init__(self, oom_on_batch: bool = False) -> None:
        self.oom_on_batch = oom_on_batch

    def classify(self, path: str, candidates: list[str], top_k: int = 5) -> list[Any]:
        return [{"path": path, "candidates": candidates, "top_k": top_k}]

    def classify_batch(
        self, paths: list[str], candidates: list[str], top_k: int = 5,
    ) -> list[list[Any]]:
        if self.oom_on_batch:
            raise RuntimeError("CUDA out of memory")
        return [self.classify(p, candidates, top_k) for p in paths]


class _FakeNoBatchModel:
    """只有单图接口（无批量能力）。"""

    def detect(self, path: str, prompts: list[str], confidence_threshold: float = 0.3) -> list[Any]:
        return [{"path": path}]


# ── 检测 helper ─────────────────────────────────────────────────


def test_detect_batch_parity() -> None:
    """批量路径与逐图路径逐位相等，且走批量分支。"""
    model = _FakeDetModel()
    paths = ["/a.jpg", "/b.jpg", "/c.jpg"]
    per_img = detect_batch_or_fallback(model, paths, ["cat"], 0.5, num_workers=2)
    assert model.batch_calls == [3]
    assert per_img == [
        [{"path": p, "prompts": ["cat"], "conf": 0.5}] for p in paths
    ]


def test_detect_no_batch_method_falls_back() -> None:
    """模型无 detect_batch → 逐图（结果与 paths 对齐）。"""
    model = _FakeNoBatchModel()
    paths = ["/a.jpg", "/b.jpg"]
    per_img = detect_batch_or_fallback(model, paths, ["cat"], 0.3)
    assert per_img == [[{"path": "/a.jpg"}], [{"path": "/b.jpg"}]]


def test_detect_single_path_skips_batch() -> None:
    """len(paths)==1 时直接逐图（不触发批量调用）。"""
    model = _FakeDetModel()
    per_img = detect_batch_or_fallback(model, ["/only.jpg"], ["cat"], 0.3)
    assert model.batch_calls == []
    assert per_img == [[{"path": "/only.jpg", "prompts": ["cat"], "conf": 0.3}]]


def test_detect_oom_falls_back_per_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """批量 OOM → 当前块逐图 + empty_cache 被调。"""
    import torch

    cache_calls: list[bool] = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: cache_calls.append(True))
    model = _FakeDetModel(oom_on_batch=True)
    paths = ["/a.jpg", "/b.jpg"]
    per_img = detect_batch_or_fallback(model, paths, ["cat"], 0.3)
    assert cache_calls  # OOM 后释放缓存
    assert len(per_img) == 2  # 逐图结果保序


def test_detect_non_oom_error_propagates() -> None:
    """非 OOM 异常向上传播（不吞真实错误）。"""

    class ValueErrorModel(_FakeDetModel):
        def detect_batch(
            self, paths: list[str], prompts: list[str],
            confidence_threshold: float = 0.3, num_workers: int = 0,
        ) -> list[list[Any]]:
            raise ValueError("bad model")

    with pytest.raises(ValueError, match="bad model"):
        detect_batch_or_fallback(ValueErrorModel(), ["/a.jpg", "/b.jpg"], ["cat"], 0.3)


# ── OBB / 分类 / 分割 helper 同构 parity ────────────────────────


def test_obb_batch_parity() -> None:
    """obb helper：批量路径与逐图路径一致。"""

    class FakeObb(_FakeDetModel):
        def detect_obb(
            self, path: str, prompts: list[str], confidence_threshold: float = 0.3,
        ) -> list[Any]:
            self.single_calls.append(path)
            return [{"path": path, "angle": 0.5}]

        def detect_obb_batch(
            self, paths: list[str], prompts: list[str],
            confidence_threshold: float = 0.3, num_workers: int = 0,
        ) -> list[list[Any]]:
            self.batch_calls.append(len(paths))
            return [self.detect_obb(p, prompts, confidence_threshold) for p in paths]

    model = FakeObb()
    paths = ["/a.jpg", "/b.jpg"]
    per_img = obb_batch_or_fallback(model, paths, ["ship"], 0.4, num_workers=1)
    assert model.batch_calls == [2]
    assert per_img == [[{"path": "/a.jpg", "angle": 0.5}], [{"path": "/b.jpg", "angle": 0.5}]]


def test_classify_batch_parity() -> None:
    """分类 helper：批量路径与逐图路径一致。"""
    model = _FakeClsModel()
    paths = ["/a.jpg", "/b.jpg"]
    per_img = classify_batch_or_fallback(model, paths, ["cat", "dog"], top_k=3)
    assert per_img == [
        [{"path": p, "candidates": ["cat", "dog"], "top_k": 3}] for p in paths
    ]


def test_classify_oom_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """分类批量 OOM → 逐图 + empty_cache。"""
    import torch

    cache_calls: list[bool] = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: cache_calls.append(True))
    model = _FakeClsModel(oom_on_batch=True)
    per_img = classify_batch_or_fallback(model, ["/a.jpg", "/b.jpg"], ["cat"], 5)
    assert cache_calls
    assert len(per_img) == 2


def test_generate_batch_parity() -> None:
    """分割 helper：批量路径与逐图路径一致（bboxes 参数透传）。"""

    class FakeSeg:
        def __init__(self) -> None:
            self.batch_calls: list[int] = []

        def generate(self, path: str, bboxes: list[Any], mode: str = "box") -> list[Any]:
            return [{"path": path, "mode": mode, "n_bboxes": len(bboxes)}]

        def generate_batch(
            self, paths: list[str], bboxes: list[Any], mode: str = "box",
        ) -> list[list[Any]]:
            self.batch_calls.append(len(paths))
            return [self.generate(p, bboxes, mode) for p in paths]

    model = FakeSeg()
    bboxes = [{"label": "cat"}]
    per_img = generate_batch_or_fallback(model, ["/a.jpg", "/b.jpg"], bboxes)
    assert model.batch_calls == [2]
    assert per_img == [
        [{"path": "/a.jpg", "mode": "box", "n_bboxes": 1}],
        [{"path": "/b.jpg", "mode": "box", "n_bboxes": 1}],
    ]


def test_generate_no_batch_method_falls_back() -> None:
    """无 generate_batch（SAM2 等）→ 逐图。"""

    class Sam2Like:
        def generate(self, path: str, bboxes: list[Any], mode: str = "box") -> list[Any]:
            return [{"path": path}]

    per_img = generate_batch_or_fallback(Sam2Like(), ["/a.jpg"], [])
    assert per_img == [[{"path": "/a.jpg"}]]


# ── 探针采样 ────────────────────────────────────────────────────


def test_sample_image_paths(tmp_path: Path) -> None:
    """取前 limit 张存在的图路径；缺失文件跳过。"""
    gt = {
        1: {"file_name": "a.jpg"},
        2: {"file_name": "missing.jpg"},
        3: {"file_name": "b.jpg"},
        4: {"file_name": "c.jpg"},
    }
    image_dir = tmp_path
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.jpg").write_bytes(b"x")
    (tmp_path / "c.jpg").write_bytes(b"x")

    paths = sample_image_paths(gt, image_dir, limit=2)
    assert paths == [str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg")]


def test_sample_image_paths_empty_dir(tmp_path: Path) -> None:
    """全部缺失 → 空列表。"""
    gt = {1: {"file_name": "nope.jpg"}}
    assert sample_image_paths(gt, tmp_path) == []
