"""tools/device.py 批量调优原语测试（mock torch.cuda，无 GPU 依赖）。

覆盖：
- recommend_num_workers CPU 核数公式（与 GPU 无关）
- get_gpu_free_memory_gb（无 CUDA → None）
- measure_single_image_memory 增量法（脚本化 memory_reserved 序列）
- auto_tune_batch_size 预算数学与钳制
- resolve_batch_params 四档优先级（显式 > 动态实测 > 静态表）
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from auto2dlabel.tests import Path, np
from auto2dlabel.tools.device import (
    auto_tune_batch_size,
    disable_tf32,
    get_gpu_free_memory_gb,
    measure_single_image_memory,
    recommend_num_workers,
    resolve_batch_params,
)

# ── 探针图（cv2 造图，PIL 可读）──────────────────────────────────


def _make_probe_images(tmp_path: Path, sizes: list[tuple[int, int]]) -> list[str]:
    """写多张纯黑图，返回路径列表。"""
    import cv2

    paths = []
    for i, (w, h) in enumerate(sizes):
        p = tmp_path / f"probe_{i}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), np.zeros((h, w, 3), dtype=np.uint8))
        paths.append(str(p))
    return paths


# ── CUDA 环境 mock 工具 ──────────────────────────────────────────


def _mock_cuda_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """torch.cuda 各 API 默认可用（具体测试再覆盖行为）。"""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _dev=0: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _dev=0: (24 * 1024 ** 3, 24 * 1024 ** 3))


def _mock_reserved_sequence(
    monkeypatch: pytest.MonkeyPatch, values: list[int],
) -> list[int]:
    """memory_reserved 按调用次序返回 values，记录实际调用次数。"""
    import torch

    calls: list[int] = []
    seq = iter(values)

    def _reserved(_dev: int = 0) -> int:
        calls.append(1)
        return next(seq, values[-1])

    monkeypatch.setattr(torch.cuda, "memory_reserved", _reserved)
    return calls


class _RecordingInfer:
    """记录每次调用图数的推理闭包。"""

    def __init__(self) -> None:
        self.calls: list[int] = []
        self.paths_calls: list[list[str]] = []

    def __call__(self, paths: list[str]) -> list[Any]:
        self.calls.append(len(paths))
        self.paths_calls.append(list(paths))
        return []


# ── num_workers 公式 ─────────────────────────────────────────────


@pytest.mark.parametrize(
    ("cores", "expected"),
    [(1, 0), (4, 1), (8, 2), (16, 4), (32, 4)],  # 32 核钳制到 4
)
def test_recommend_num_workers_by_cpu(
    monkeypatch: pytest.MonkeyPatch, cores: int, expected: int,
) -> None:
    """workers = min(4, cores // 4)，与 GPU 无关。"""
    monkeypatch.setattr(os, "cpu_count", lambda: cores)
    assert recommend_num_workers() == expected


def test_recommend_num_workers_cpu_count_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """cpu_count 返回 None → 视为 1 核 → 0。"""
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert recommend_num_workers() == 0


# ── 空闲显存 ─────────────────────────────────────────────────────


def test_get_gpu_free_memory_none_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """无 CUDA → None。"""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert get_gpu_free_memory_gb() is None


def test_get_gpu_free_memory_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """mem_get_info 空闲 8GB → 返回 8.0。"""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _dev=0: (8 * 1024 ** 3, 24 * 1024 ** 3))
    assert get_gpu_free_memory_gb() == pytest.approx(8.0)


# ── 单图显存增量测量 ─────────────────────────────────────────────


def test_measure_batch2_increment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """r0→warmup→r1→probe(batch2)→r2：per_img = r2 - r1，probe_batch=2。"""
    _mock_cuda_ok(monkeypatch)
    import torch

    seq = iter([1, 6, 10])

    def _reserved(_dev: int = 0) -> int:
        return next(seq, 10)

    monkeypatch.setattr(torch.cuda, "memory_reserved", _reserved)
    infer = _RecordingInfer()
    paths = _make_probe_images(tmp_path, [(100, 100), (80, 80)])

    result = measure_single_image_memory(infer, paths)

    assert result == (4, 2)  # per_img = 10 - 6
    assert infer.calls == [1, 2]  # warmup 单图 → probe 两张


def test_measure_probe_oom_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """probe（batch=2）OOM → per_img = r1 - r0 保守上界，probe_batch=1，empty_cache 被调。"""
    _mock_cuda_ok(monkeypatch)
    import torch

    seq = iter([1, 6])
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _dev=0: next(seq, 6))
    cache_calls: list[bool] = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: cache_calls.append(True))

    class OomOn2:
        def __call__(self, paths: list[str]) -> list[Any]:
            if len(paths) > 1:
                raise RuntimeError("CUDA out of memory")
            return []

    result = measure_single_image_memory(
        OomOn2(), _make_probe_images(tmp_path, [(10, 10), (9, 9)]),
    )

    assert result == (5, 1)  # r1 - r0 = 6 - 1
    assert cache_calls  # OOM 后释放缓存


def test_measure_warmup_oom_returns_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """warmup（batch=1）OOM → None（单图都放不下）。"""
    _mock_cuda_ok(monkeypatch)

    class OomAlways:
        def __call__(self, _paths: list[str]) -> list[Any]:
            raise RuntimeError("CUDA out of memory")

    result = measure_single_image_memory(OomAlways(), _make_probe_images(tmp_path, [(10, 10)]))
    assert result is None


def test_measure_non_oom_error_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """非 OOM 异常向上传播（不吞真实错误）。"""
    _mock_cuda_ok(monkeypatch)

    class ValueErrorRaiser:
        def __call__(self, _paths: list[str]) -> list[Any]:
            raise ValueError("bad model")

    with pytest.raises(ValueError, match="bad model"):
        measure_single_image_memory(ValueErrorRaiser(), _make_probe_images(tmp_path, [(10, 10)]))


def test_measure_picks_largest_images(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """探针选面积最大的 2 张不同图（小图不参与探测）。"""
    _mock_cuda_ok(monkeypatch)
    paths = _make_probe_images(tmp_path, [(10, 10), (200, 200), (150, 150)])
    infer = _RecordingInfer()

    measure_single_image_memory(infer, paths)

    assert infer.calls == [1, 2]
    # 探针 = 面积最大的两张：200x200 与 150x150；10x10 被排除
    assert infer.paths_calls[0] == [paths[1]]
    assert infer.paths_calls[1] == [paths[1], paths[2]]


def test_measure_no_cuda_returns_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """无 CUDA → None 且不调用推理闭包。"""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    infer = _RecordingInfer()
    assert measure_single_image_memory(infer, _make_probe_images(tmp_path, [(10, 10)])) is None
    assert infer.calls == []


def test_measure_no_valid_images_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """探针候选图全部缺失 → None。"""
    _mock_cuda_ok(monkeypatch)
    assert measure_single_image_memory(_RecordingInfer(), ["/nope/1.png", "/nope/2.png"]) is None


# ── 最大 batch 计算 ──────────────────────────────────────────────


def test_auto_tune_budget_math(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """free=20G、per_img=2G、safety=0.85 → 1 + floor(17/2) = 9。"""
    _mock_cuda_ok(monkeypatch)
    import torch

    # measure 路径：r0=0 → warmup → r1=2G → probe → r2=4G → per_img=2G
    seq = iter([0, 2 * 1024 ** 3, 4 * 1024 ** 3])
    monkeypatch.setattr(
        torch.cuda, "memory_reserved", lambda _dev=0: next(seq, 0),
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _dev=0: (20 * 1024 ** 3, 24 * 1024 ** 3))

    bs = auto_tune_batch_size(_RecordingInfer(), _make_probe_images(tmp_path, [(10, 10)]))
    assert bs == 9


def test_auto_tune_max_batch_clamp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """计算值超 max_batch → 钳制。"""
    _mock_cuda_ok(monkeypatch)
    import torch

    seq = iter([0, 1024 ** 3, 2 * 1024 ** 3])  # per_img=1G，free=24G → 21 张
    monkeypatch.setattr(
        torch.cuda, "memory_reserved", lambda _dev=0: next(seq, 0),
    )

    bs = auto_tune_batch_size(
        _RecordingInfer(), _make_probe_images(tmp_path, [(10, 10)]), max_batch=4,
    )
    assert bs == 4


def test_auto_tune_no_cuda_returns_min(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """无 CUDA → min_batch。"""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert auto_tune_batch_size(_RecordingInfer(), _make_probe_images(tmp_path, [(10, 10)])) == 1


def test_auto_tune_probe_batch1_returns_min(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """probe（batch=2）OOM → 只能逐图 → min_batch。"""
    _mock_cuda_ok(monkeypatch)

    class OomOn2:
        def __call__(self, paths: list[str]) -> list[Any]:
            if len(paths) > 1:
                raise RuntimeError("CUDA out of memory")
            return []

    bs = auto_tune_batch_size(OomOn2(), _make_probe_images(tmp_path, [(10, 10), (9, 9)]))
    assert bs == 1


def test_auto_tune_min_batch_floor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """实测成功但 min_batch=2 → 至少 2。"""
    _mock_cuda_ok(monkeypatch)
    import torch

    seq = iter([0, 1024 ** 3, 2 * 1024 ** 3])
    monkeypatch.setattr(
        torch.cuda, "memory_reserved", lambda _dev=0: next(seq, 0),
    )

    bs = auto_tune_batch_size(
        _RecordingInfer(), _make_probe_images(tmp_path, [(10, 10)]), min_batch=2,
    )
    assert bs >= 2


# ── 四档来源统一入口 ─────────────────────────────────────────────


def test_resolve_explicit_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """显式 batch/workers 优先，不触碰实测闭包。"""
    _mock_cuda_ok(monkeypatch)
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    infer = _RecordingInfer()

    bs, nw = resolve_batch_params(
        "object_detection", infer, _make_probe_images(tmp_path, [(10, 10)]),
        explicit_batch=8, explicit_workers=2,
    )
    assert (bs, nw) == (8, 2)
    assert infer.calls == []  # 实测未触发


def test_resolve_partial_explicit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """只显式给 workers → batch 走动态实测。"""
    _mock_cuda_ok(monkeypatch)
    import torch

    seq = iter([0, 1024 ** 3, 2 * 1024 ** 3])  # per_img=1G → free 24G×0.85=20.4 → bs=21→钳 64
    monkeypatch.setattr(
        torch.cuda, "memory_reserved", lambda _dev=0: next(seq, 0),
    )

    bs, nw = resolve_batch_params(
        "object_detection", _RecordingInfer(), _make_probe_images(tmp_path, [(10, 10)]),
        explicit_workers=3,
    )
    assert nw == 3
    assert bs == 21  # 动态实测值（free=24G, per_img=1G, 0.85 安全系数）


def test_resolve_no_cuda_static(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """无 CUDA → 静态表兜底（24GB 档 bs=8），workers 按 CPU 公式。"""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(os, "cpu_count", lambda: 8)

    bs, nw = resolve_batch_params(
        "object_detection", _RecordingInfer(), _make_probe_images(tmp_path, [(10, 10)]),
    )
    assert nw == 2  # CPU 公式（8 核）
    assert bs == 1  # 静态表在无 GPU（None）时回退逐图


def test_resolve_no_infer_fn_static(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """模型无批量能力（infer_fn=None）→ 静态表（含 GPU 显存 24GB 档）。"""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _dev=0: type(
        "Props", (), {"total_memory": 24 * 1024 ** 3},
    )())
    monkeypatch.setattr(os, "cpu_count", lambda: 16)

    bs, nw = resolve_batch_params("classification", None, [])
    assert (bs, nw) == (16, 4)  # 静态表 4090 档


def test_resolve_dynamic_used_when_batchable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CUDA + 有批量能力 → 动态实测值覆盖静态表。"""
    _mock_cuda_ok(monkeypatch)
    import torch

    seq = iter([0, 1024 ** 3, 2 * 1024 ** 3])  # per_img=1G
    monkeypatch.setattr(
        torch.cuda, "memory_reserved", lambda _dev=0: next(seq, 0),
    )
    monkeypatch.setattr(os, "cpu_count", lambda: 16)

    bs, nw = resolve_batch_params(
        "object_detection", _RecordingInfer(), _make_probe_images(tmp_path, [(10, 10)]),
    )
    assert bs == 21  # 动态（free=24G → 20.4G/1G + 1），非静态表 8
    assert nw == 4


# ── TF32 关闭（批量 vs 逐图一致性使能）────────────────────────────


def test_disable_tf32_sets_both_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """cudnn 与 matmul 的 TF32 开关均置 False。"""
    import torch

    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True, raising=False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True, raising=False)

    disable_tf32()

    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.backends.cuda.matmul.allow_tf32 is False


def test_resolve_batch_params_disables_tf32(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """批量入口 resolve_batch_params 顺带关闭 TF32（回归：TF32 曾致 OBB 批量 22 框差异）。"""
    _mock_cuda_ok(monkeypatch)
    import torch

    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True, raising=False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True, raising=False)

    resolve_batch_params(
        "object_detection", None, _make_probe_images(tmp_path, [(10, 10)]),
        explicit_batch=2,
    )

    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.backends.cuda.matmul.allow_tf32 is False
