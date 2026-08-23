#!/usr/bin/env python3
"""批量运行所有 benchmark 并生成汇总报告。

用法:
  python -m auto2dlabel.benchmarks.run_all          # 全部 11 数据集
  python -m auto2dlabel.benchmarks.run_all --only voc2007  # 仅指定数据集
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from auto2dlabel.benchmarks import datetime

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

BENCHMARKS = {
    # 目标检测
    "coco2017": {
        "script": "coco_benchmark.py",
        "model": "yolo26x.pt",
        "default_args": "--max-images 50 --conf 0.3",
    },
    "voc2007": {
        "script": "voc_benchmark.py",
        "model": "yolo26x.pt",
        "default_args": "--max-images 50 --conf 0.3",
    },
    "kitti": {
        "script": "kitti_benchmark.py",
        "model": "yolo26x.pt",
        "default_args": "--max-images 50 --conf 0.3",
    },
    # 航拍检测
    "dota": {
        "script": "dota_benchmark.py",
        "model": "yolo26x.pt",
        "default_args": "--max-images 50 --conf 0.3",
    },
    # 旋转框检测
    "dota_obb": {
        "script": "dota_obb_benchmark.py",
        "model": "yolo11n-obb.pt",
        "default_args": "--max-images 50 --conf 0.3",
    },
    # 密集行人检测
    "mot": {
        "script": "mot_benchmark.py",
        "model": "yolo26x.pt",
        "default_args": "--max-images 50 --conf 0.3",
    },
    # 实例分割
    "coco_seg": {
        "script": "coco_seg_benchmark.py",
        "model": "FastSAM-s.pt",
        "default_args": "--max-images 50 --conf 0.3 --seg-model sam2_l.pt",
    },
    "cityscapes": {
        "script": "cityscapes_benchmark.py",
        "model": "FastSAM-s.pt",
        "default_args": "--max-images 50",
    },
    "nuimages": {
        "script": "nuimages_benchmark.py",
        "model": "FastSAM-s.pt",
        "default_args": "--max-images 50",
    },
    "d2sa": {
        "script": "d2sa_benchmark.py",
        "model": "FastSAM-s.pt",
        "default_args": "--max-images 50 --seg-model sam2_l.pt",
    },
    # 图像分类
    "imagenet100": {
        "script": "classification_benchmark.py",
        "model": "resnet18",
        "default_args": "--dataset imagenet100 --max-images 50 --per-class 50",
    },
    "imagenet1k": {
        "script": "classification_benchmark.py",
        "model": "resnet18",
        "default_args": "--dataset imagenet1k --per-class 2 --max-images 200",
    },
}

OUTPUT_DIR = _PROJECT_ROOT / "benchmarks_outputs"


def run_benchmark(name: str, info: dict, extra_args: str = "") -> dict:
    """运行单个 benchmark，返回结果摘要。"""
    script = _PROJECT_ROOT / "auto2dlabel" / "benchmarks" / info["script"]
    cmd = [
        sys.executable, str(script),
        *info["default_args"].split(),
        *extra_args.split(),
    ]
    cmd_str = " ".join(cmd)

    print(f"\n{'=' * 60}")
    print(f"▶ {name}: {cmd_str}")
    print(f"{'=' * 60}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, cwd=str(_PROJECT_ROOT))
        success = result.returncode == 0
        output = result.stdout[-2000:] if result.stdout else ""
        if not success:
            output += f"\nSTDERR:\n{result.stderr[-1000:]}"
    except subprocess.TimeoutExpired:
        success = False
        output = "TIMEOUT (3600s)"
    except Exception as e:
        success = False
        output = str(e)

    return {"name": name, "model": info["model"], "success": success, "output": output}


def generate_summary(results: list[dict]) -> str:
    """生成汇总 Markdown 报告。"""
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    lines = [
        f"# AutoLabel Benchmark Summary",
        f"",
        f"**时间**: {ts}",
        f"",
        f"## 概览",
        f"",
        f"| 数据集 | 模型 | 状态 |",
        f"| --- | --- | --- |",
    ]

    success_count = 0
    for r in results:
        status = "✅ 成功" if r["success"] else "❌ 失败"
        if r["success"]:
            success_count += 1
        lines.append(f"| {r['name']} | {r['model']} | {status} |")

    lines.append(f"")
    lines.append(f"**总计**: {success_count}/{len(results)} 通过")
    lines.append(f"")

    # 详细输出
    lines.append(f"## 详细输出")
    lines.append(f"")
    for r in results:
        lines.append(f"### {r['name']} ({r['model']})")
        lines.append(f"```")
        lines.append(r["output"][:3000])
        lines.append(f"```")
        lines.append(f"")

    summary = "\n".join(lines)
    md_path = OUTPUT_DIR / f"summary_{ts}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(summary)
    print(f"\n汇总报告: {md_path}")
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run all benchmarks")
    parser.add_argument("--only", type=str, help="仅运行指定数据集（逗号分隔）")
    parser.add_argument("--extra", type=str, default="", help="额外 CLI 参数")
    args = parser.parse_args()

    if args.only:
        names = [n.strip() for n in args.only.split(",")]
        selected = {n: BENCHMARKS[n] for n in names if n in BENCHMARKS}
        if not selected:
            print(f"未找到数据集: {args.only}")
            print(f"可用: {', '.join(BENCHMARKS.keys())}")
            sys.exit(1)
    else:
        selected = BENCHMARKS

    print(f"将运行 {len(selected)} 个 benchmark...")

    results = []
    for name, info in selected.items():
        results.append(run_benchmark(name, info, extra_args=args.extra))

    generate_summary(results)

    failed = [r for r in results if not r["success"]]
    if failed:
        print(f"\n{failed} 失败 ({len(failed)}/{len(results)})")
        sys.exit(1)
    else:
        print(f"\n全部 {len(results)} 个 benchmark 通过!")


if __name__ == "__main__":
    main()
