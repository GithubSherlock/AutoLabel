"""设备检测工具 — 自动检测 CUDA / MPS / CPU。

所有需要 GPU 推理的脚本统一调用此模块，避免重复代码。
另含批量推理超参数动态调优原语（显存测量 / 最大 batch 实测），
见 measure_single_image_memory / auto_tune_batch_size / resolve_batch_params。
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any


def get_device() -> str:
    """检测最佳可用设备：CUDA > MPS > CPU。

    Returns:
        设备字符串: "cuda" | "mps" | "cpu"
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def get_device_info() -> str:
    """返回设备的可读描述。

    Returns:
        如 "CUDA (NVIDIA GeForce RTX 4090)" | "MPS (Apple Silicon GPU)" | "CPU"
    """
    try:
        import torch

        if torch.cuda.is_available():
            return f"CUDA ({torch.cuda.get_device_name(0)})"
        elif torch.backends.mps.is_available():
            return "MPS (Apple Silicon GPU)"
    except ImportError:
        pass
    return "CPU"


def print_device() -> None:
    """打印当前使用的设备（benchmark 入口：顺带关闭 TF32 保证批量一致性）。"""
    disable_tf32()
    print(f"设备: {get_device_info()}")


# ================================================================
# 批量推理超参数动态调优（执行阶段用：模型已加载后实测）
# ================================================================


def disable_tf32() -> None:
    """关闭 TF32（Ampere+ GPU 的 19-bit 尾数加速格式），保证批量推理一致性。

    实测（RTX 4090，yolo26x-obb，2026-08-18）：TF32 开启时同一模型对
    batch=1 与 batch>1 走不同 cudnn kernel，首层卷积即产生 ~8e-5 差异，
    经 200 层 + 注意力 softmax + 角度 argmax 放大后，批量与逐图结果差
    20+ 个旋转框（raw 输出最大差 ~1e3）；关闭后 raw 差 <1e-4、结果逐位
    一致（逐图推理自洽不受影响，纯精度提升）。

    幂等；无 torch 时静默跳过。吞吐代价 ~10%（正确性优先）。
    """
    try:
        import torch
    except ImportError:
        return
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def get_gpu_free_memory_gb() -> float | None:
    """当前空闲显存（GB）；synchronize + empty_cache 后读取；无 CUDA 返回 None。"""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    free, _total = torch.cuda.mem_get_info(0)
    return free / (1024 ** 3)


def recommend_num_workers() -> int:
    """DataLoader workers 按 CPU 核数推荐（与 GPU 显存无关）。

    workers 是 CPU 数据加载并行度，不消耗显存；每 4 核 1 个、上限 4。
    """
    cores = os.cpu_count() or 1
    return min(4, max(0, cores // 4))


def _is_oom(e: BaseException) -> bool:
    """判断异常是否为 CUDA OOM。"""
    return "out of memory" in str(e).lower()


def _pick_probe_images(image_paths: list[str]) -> list[str]:
    """选面积最大的 2 张不同图作显存探针（torchvision 批显存≈面积和，最大图最保守）。"""
    from PIL import Image

    sizes: list[tuple[int, str]] = []
    for p in image_paths:
        try:
            with Image.open(p) as im:
                sizes.append((im.size[0] * im.size[1], p))
        except Exception:
            continue  # 读取失败的图不参与探测
    sizes.sort(reverse=True)
    unique: list[str] = []
    for _area, p in sizes:
        if p not in unique:
            unique.append(p)
        if len(unique) == 2:
            break
    return unique


def measure_single_image_memory(
    infer_fn: Callable[[list[str]], Any],
    image_paths: list[str],
) -> tuple[int, int] | None:
    """实测单图峰值显存增量，返回 (per_img_bytes, probe_batch)。

    增量法（memory_reserved 只增不减，只测差值、从不 reset）：
    - 选面积最大 2 张不同图作探针
    - r0 = reserved()；warmup infer_fn([img1]) → r1；probe infer_fn([img1, img2]) → r2
    - per_img = r2 - r1（batch=2 的线性增量）；probe OOM → per_img = r1 - r0
      （含 warmup 一次性成本，保守上界），probe_batch = 1
    - 无 CUDA / warmup OOM / per_img <= 0 → None（调用方回退静态档位）
    - 非 OOM 异常向上传播（不吞真实错误）
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None

    probes = _pick_probe_images(image_paths)
    if not probes:
        return None

    r0 = torch.cuda.memory_reserved(0)
    try:
        infer_fn([probes[0]])
        torch.cuda.synchronize()
    except RuntimeError as e:
        if _is_oom(e):
            return None
        raise
    r1 = torch.cuda.memory_reserved(0)

    try:
        infer_fn(probes)  # batch=2（两图可能重复：仅 1 张有效图时）
        torch.cuda.synchronize()
    except RuntimeError as e:
        if _is_oom(e):
            torch.cuda.empty_cache()
            return (r1 - r0, 1)  # batch=2 放不下 → 单图全量增量作保守上界
        raise
    r2 = torch.cuda.memory_reserved(0)

    per_img = r2 - r1
    if per_img <= 0:
        return None
    return per_img, 2


def auto_tune_batch_size(
    infer_fn: Callable[[list[str]], Any],
    image_paths: list[str],
    min_batch: int = 1,
    max_batch: int = 64,
    safety_factor: float = 0.85,
) -> int:
    """动态实测推荐最大 batch_size；无 GPU / 失败 / 探测仅容单图 → min_batch。

    budget = 空闲显存 × safety_factor（留余量给其他进程与估计误差）；
    bs = 1 + floor(budget / per_img)（batch=1 已驻留，每加一张花 per_img）；
    钳制到 [min_batch, max_batch]。
    """
    try:
        import torch
    except ImportError:
        return min_batch
    if not torch.cuda.is_available():
        return min_batch

    measured = measure_single_image_memory(infer_fn, image_paths)
    if measured is None:
        return min_batch
    per_img, probe_batch = measured
    if probe_batch == 1:
        return min_batch  # batch=2 已放不下 → 只能逐图

    torch.cuda.empty_cache()
    free, _total = torch.cuda.mem_get_info(0)
    budget = free * safety_factor
    bs = 1 + int(budget // per_img)
    return max(min_batch, min(max_batch, bs))


def resolve_batch_params(
    task_type: str,
    infer_fn: Callable[[list[str]], Any] | None,
    image_paths: list[str],
    explicit_batch: int | None = None,
    explicit_workers: int | None = None,
    max_batch: int = 64,
) -> tuple[int, int]:
    """批量推理超参数执行阶段统一入口：显式 > 动态实测 > 静态表 > 逐图。

    1) explicit_batch / explicit_workers 任一给定 → 直接采用（缺的另一半走推荐）
    2) CUDA 且 infer_fn 非 None → auto_tune_batch_size 实测（失败静默降级）
    3) 其余 → recommend_batch_params 静态表（规划阶段兜底，模型未加载场景）
    4) num_workers 一律 recommend_num_workers()（与显存档位解耦）

    Args:
        task_type: object_detection / obb_detection / instance_segmentation /
            semantic_segmentation / classification / image_classification
        infer_fn: 批量推理闭包（paths → 结果）；None 表示模型无批量能力（回退静态表）
        image_paths: 实测探针候选图路径
    """
    from auto2dlabel.schema.task_plan import (
        detect_gpu_memory_gb,
        recommend_batch_params,
    )

    # TF32 会破坏批量 vs 逐图结果一致性（batch 不同 → cudnn kernel 不同 →
    # 数值发散经 argmax 放大），所有批量推理入口统一关闭
    disable_tf32()

    nw = recommend_num_workers()
    if explicit_workers is not None:
        nw = max(0, explicit_workers)

    # 1) 显式给定 batch → 直接采用（workers 已有，独立）
    if explicit_batch is not None:
        return max(1, explicit_batch), nw

    # 2) 动态实测（CUDA 可用时结果即结论：1 表示确认只能逐图，不回退静态表）
    try:
        import torch

        cuda_ok = torch.cuda.is_available()
    except ImportError:
        cuda_ok = False
    if infer_fn is not None and cuda_ok:
        return auto_tune_batch_size(infer_fn, image_paths, max_batch=max_batch), nw

    # 3) 静态表兜底（MPS/CPU 或模型无批量能力；num_workers 仍用 CPU 公式）
    bs, _ = recommend_batch_params(task_type, detect_gpu_memory_gb())
    return bs, nw
