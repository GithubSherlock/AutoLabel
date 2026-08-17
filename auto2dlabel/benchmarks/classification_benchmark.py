#!/usr/bin/env python3
"""ImageNet100 分类 Benchmark — 监督（torchvision）vs 零样本（CLIP/SigLIP）top-K 准确率。

评测协议（两种标签空间，报告与实测数据须注明）：
- 监督路径（resnet18 等）：完整 ImageNet1K 1000 类空间取 top-K（官方评测协议），
  GT 英文名与预测英文名精确字符串比较
- 零样本路径（clip/siglip）：100 类候选空间取 top-K（候选即 GT 类名集合）
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from auto2dlabel.benchmarks import datetime, time  # noqa: E402
from auto2dlabel.benchmarks.common import (  # noqa: E402
    OUTPUT_DIR,
    build_parser,
    evaluate_classification,
    format_classification_table,
    save_results,
)
from auto2dlabel.benchmarks.datasets import (  # noqa: E402
    ensure_imagenet100,
    load_imagenet100_ground_truth,
)
from auto2dlabel.models.classification import create_classification_model  # noqa: E402

# ============================================================
# 1. 推理
# ============================================================

def run_classification(
    gt: dict[str, dict[str, Any]],
    dataset_root: Path,
    model_name: str,
    top_k: int,
    class_names: list[str],
) -> tuple[dict[str, list[dict[str, Any]]], float]:
    """逐图分类，返回 (预测 top-K 列表, 推理耗时秒数)。

    监督路径 candidates=[]（纯 1000 类 top-K）；零样本路径 candidates=100 类英文名。
    """
    model = create_classification_model(model_name)
    cast(Any, model)._load()  # 预加载权重，避免首图加载耗时污染吞吐计时

    is_zeroshot = "clip" in model_name or "siglip" in model_name
    predictions: dict[str, list[dict[str, Any]]] = {}

    _t0 = time.time()
    for img_id, info in gt.items():
        img_path = dataset_root / info["file_name"]
        if not img_path.exists():
            continue
        candidates = class_names if is_zeroshot else []
        labels = model.classify(str(img_path), candidates, top_k=top_k)
        predictions[img_id] = [{"label": lb.label, "score": lb.score} for lb in labels]

    elapsed = time.time() - _t0
    total = max(len(predictions), 1)
    print(f"完成！{elapsed:.1f}s, {total/elapsed:.1f} img/s")
    return predictions, elapsed


# ============================================================
# 2. 主流程
# ============================================================

def main() -> None:
    parser = build_parser("ImageNet100 分类 Benchmark（监督 vs 零样本 top-K）")
    parser.set_defaults(model="resnet18")
    parser.add_argument("--dataset", default="imagenet100", choices=["imagenet100"],
                        help="分类数据集（wnid 目录即标签）")
    parser.add_argument("--top-k", type=int, default=5, help="评测 top-K 值")
    parser.add_argument("--per-class", type=int, default=50, help="每类抽样解压张数")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    from auto2dlabel.tools.device import print_device
    print_device()
    print()
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # 数据准备（每类均匀抽样解压，幂等）
    print(f"加载 {args.dataset}（每类 {args.per_class} 张）...")
    dataset_root = ensure_imagenet100(per_class=args.per_class)
    gt = load_imagenet100_ground_truth(dataset_root, max_images=args.max_images)
    print(f"已加载 {len(gt)} 张图像\n")
    if not gt:
        print("GT 为空，请检查解压结果")
        sys.exit(1)

    class_names = sorted({str(info["label"]) for info in gt.values()})

    # 推理
    print(f"运行分类（模型 {args.model}，top-{args.top_k}）...")
    predictions, elapsed = run_classification(gt, dataset_root, args.model, args.top_k, class_names)
    print()

    # 可视化（--viz：文本条 GT + top-K）
    if args.viz:
        from auto2dlabel.benchmarks.viz import visualize_dataset
        visualize_dataset(
            "imagenet100", gt, predictions,
            lambda img_id, info: dataset_root / info["file_name"],
            gt_label_fn=lambda info: str(info["label"]),
        )
        print()

    # 指标
    results = evaluate_classification(gt, predictions, top_k=args.top_k)
    all_classes = sorted({str(info["label"]) for info in gt.values()})
    summary = format_classification_table(results, all_classes, top_n=args.top_classes)
    print(summary)
    print()

    # 保存（模型名含 "/" 时替换为 "-"，避免破坏产物文件名）
    total = max(len(predictions), 1)
    result_data = {
        "timestamp": timestamp, "dataset": args.dataset,
        "model": args.model, "top_k": args.top_k,
        "image_count": len(gt), "class_count": len(all_classes),
        "seconds_per_image": round(elapsed / total, 4),
        "images_per_second": round(total / elapsed, 4),
        "summary": {"top-1": results["top1"], f"top-{args.top_k}": results[f"top{args.top_k}"]},
        "per_class": results["per_class"],
    }

    safe_model = args.model.replace("/", "-")
    json_path, md_path = save_results(result_data, args.dataset, safe_model, timestamp)
    md_path.write_text("\n".join([
        f"# {args.dataset} 分类 Benchmark",
        f"- **模型**: {args.model} | **top-K**: {args.top_k} | "
        f"**top-1**: {results['top1']:.4f} | **top-{args.top_k}**: {results[f'top{args.top_k}']:.4f}",
        f"- **吞吐**: {total/elapsed:.1f} img/s（{elapsed/total:.3f} s/图，"
        f"{len(gt)} 图）",
        f"```\n{summary}\n```",
    ]))
    print(f"结果: {json_path}")
    print(f"报告: {md_path}")


if __name__ == "__main__":
    main()
