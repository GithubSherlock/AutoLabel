"""Benchmark 测试工具（自 auto2dlabel/cli.py 迁入）。

原 chat 命令中的 benchmark 路径已从 CLI 移除，功能测试可直接调用这里：

- `is_benchmark_intent`: 关键词意图检测（NL → benchmark vs annotation）
- `execute_benchmark`: subprocess 执行 benchmark 脚本

路径基于 auto2dlabel 包位置解析，不依赖 cli.py 所在目录。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from rich.console import Console

import auto2dlabel
from auto2dlabel.schema.task_plan import BENCHMARK_DATASETS, BenchmarkRequest

console = Console()

_PKG_ROOT = Path(auto2dlabel.__file__).resolve().parent  # auto2dlabel/
_PROJECT_ROOT = _PKG_ROOT.parent                          # 仓库根


def is_benchmark_intent(text: str) -> bool:
    """检测用户指令是否为 benchmark 意图（vs 标注意图）。

    策略：关键词匹配。数据集名 + 测评上下文 → benchmark。
    """
    text_lower = text.lower()

    # 数据集关键词
    dataset_kw = [
        "coco", "voc", "kitti", "dota", "mot",
        "cityscapes", "nuimages", "d2sa",
        "航拍", "行人检测", "密集行人", "城市街景", "零售货架",
    ]
    has_dataset = any(kw in text_lower for kw in dataset_kw)

    # Benchmark 上下文关键词
    benchmark_kw = [
        "benchmark", "基准测试", "基准", "测评", "跑分", "评估",
        "mAP", "对比", "测一下模型", "测试模型", "评测",
        "跑一下", "跑个", "跑一次", "run benchmark",
    ]
    has_benchmark_ctx = any(kw in text_lower for kw in benchmark_kw)

    # 标注上下文（排除）
    annotation_kw = [
        "标注这张", "标注这个", "标一下", "annotate this",
        "检测这张图", "检测这个图",
    ]
    has_annotation_ctx = any(kw in text_lower for kw in annotation_kw)

    # 明显是标注（含图片路径后缀）
    looks_like_image = any(
        ext in text_lower for ext in [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"]
    )

    if looks_like_image or has_annotation_ctx:
        return False

    return has_dataset and has_benchmark_ctx


def execute_benchmark(
    request: BenchmarkRequest,
    timeout: int = 7200,
) -> subprocess.CompletedProcess[str]:
    """执行 benchmark：构建 CLI 参数，subprocess 运行对应脚本。

    Args:
        request: BenchmarkRequest（参数已解析完整）。
        timeout: subprocess 超时秒数（SAHI 可能很慢）。

    Raises:
        ValueError: 数据集未知。
        subprocess.TimeoutExpired: 超时。

    Returns:
        CompletedProcess（returncode != 0 表示 benchmark 失败）。
    """
    if request.dataset not in BENCHMARK_DATASETS:
        available = ", ".join(BENCHMARK_DATASETS.keys())
        raise ValueError(f"未知数据集: {request.dataset}（可用: {available}）")

    info = BENCHMARK_DATASETS[request.dataset]
    script = _PKG_ROOT / "benchmarks" / info["script"]
    cli_args = request.to_cli_args()

    cmd = [sys.executable, str(script), *cli_args]

    console.print(f"[bold]━━━ 运行 Benchmark: {request.dataset} ━━━[/bold]")
    console.print(f"[dim]脚本: {script.name}[/dim]")
    console.print(f"[dim]命令: {' '.join(cmd)}[/dim]")

    result = subprocess.run(
        cmd,
        cwd=str(_PROJECT_ROOT),
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        console.print(f"[red]Benchmark 失败 (exit code: {result.returncode})[/red]")
        if result.stderr:
            console.print(f"[red]STDERR:[/red]\n{result.stderr[-1000:]}")
    else:
        console.print(f"[bold green]✓ Benchmark {request.dataset} 完成[/bold green]")
    return result
