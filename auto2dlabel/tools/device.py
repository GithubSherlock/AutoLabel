"""设备检测工具 — 自动检测 CUDA / MPS / CPU。

所有需要 GPU 推理的脚本统一调用此模块，避免重复代码。
"""

from __future__ import annotations


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
    """打印当前使用的设备。"""
    print(f"设备: {get_device_info()}")
