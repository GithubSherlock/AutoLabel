#!/usr/bin/env python3
"""Experience few-shot 注入 Benchmark（v1.1 P1）—— 注入组 vs 无注入组。

对比「RAG few-shot 注入是否改善 LLM 规划质量」：同源 COCO GT / 同检测器 /
同评估协议（复用 benchmarks/common.py），两组唯一差异 = planner system prompt
是否注入 `retrieve()` 检索的 top-k 历史经验。

指标：① mAP@0.5（检测质量）② planner LLM 调用数（台账 aggregate_usage，
call_site=planner.dialog 注入组 vs 无注入组）③ 费用对比。

复用不复制：GT 加载 + 检测 + 评估全走既有 benchmark 设施（coco_benchmark.py
的 load_coco_ground_truth / run_detection + common.py 的 evaluate_per_class /
format_result_table / save_results）。

用法:
  python -m auto2dlabel.benchmarks.experience_benchmark --max-images 50
      [--inject]        # 注入组（默认关 = 基线组）
      [--seed N]        # 经验库种子（默认读 configs/experience_seeds.jsonl）
      [--usage PATH]    # 台账路径（默认 logs/llm_usage.jsonl；对比两组的 LLM 调用数）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from auto2dlabel.benchmarks import datetime, np  # noqa: E402
from auto2dlabel.benchmarks.coco_benchmark import (  # noqa: E402
    CONFIG,
    load_coco_ground_truth,
    run_detection,
)
from auto2dlabel.benchmarks.common import (  # noqa: E402
    IOU_MATCH_THRESHOLD,
    evaluate_per_class,
    format_result_table,
    save_results,
)


def _usage_stats(usage_path: Path | None, call_sites: set[str]) -> dict[str, Any]:
    """台账聚合指定 call_site 的调用数 + 费用（复用 aggregate_usage 单一事实源）。"""
    from auto2dlabel.agent.llm import USAGE_LOG_PATH, aggregate_usage, load_usage_logs

    target = usage_path or USAGE_LOG_PATH
    rows = aggregate_usage(load_usage_logs(target))
    stats: dict[str, Any] = {"calls": 0, "cost_rmb": 0.0, "tokens": 0}
    for r in rows:
        if r["call_site"] in call_sites:
            stats["calls"] += int(r["calls"])
            stats["cost_rmb"] += float(r["cost_rmb"])
            stats["tokens"] += int(r["prompt_tokens"]) + int(r["completion_tokens"])
    stats["cost_rmb"] = round(stats["cost_rmb"], 6)
    return stats


def _identity_gt(gt: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """GT 透传（COCO/DOTA 已是 bbox 结构，无 mask 转换）——与 cityscapes
    `_gt_bbox_view` 签名一致，供 `_cs_bbox_view` 在非 mask 分支复用（mypy 类型恒定）。"""
    return gt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experience few-shot 注入 vs 基线（COCO 检测，同源同协议）"
    )
    parser.add_argument("--max-images", type=int, default=50, help="最多图像数（0=全部）")
    parser.add_argument("--conf", type=float, default=CONFIG["confidence_threshold"])
    parser.add_argument("--model", type=str, default=CONFIG["model_name"])
    parser.add_argument("--iou", type=float, default=CONFIG["iou_threshold"])
    parser.add_argument("--top-classes", type=int, default=20)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--inject", action="store_true",
        help="注入 RAG few-shot（默认关 = 无注入基线组）",
    )
    parser.add_argument(
        "--seed-fail", action="store_true",
        help="种子检索失败场景测试（monkeypatch retrieve 抛异常，验证静默降级）",
    )
    parser.add_argument("--usage", type=str, default="", help="台账 JSONL 路径")
    parser.add_argument(
        "--dataset", type=str, default="coco",
        choices=["coco", "cityscapes", "dota"],
        help="GT 源：coco（默认全类负例）| cityscapes（基线已硬编码域内权重，负例）"
             "| dota（基线无「航拍域失效」经验，RAG 增量正例）",
    )
    args = parser.parse_args()

    CONFIG["confidence_threshold"] = args.conf
    CONFIG["model_name"] = args.model
    CONFIG["iou_threshold"] = args.iou

    group = "inject" if args.inject else "baseline"
    print(f"\n=== Experience Benchmark [{'注入组' if args.inject else '基线组'}] ===")
    print(f"模型: {args.model} | conf: {args.conf} | max-images: {args.max_images}")
    print(f"数据集: {args.dataset}")

    if args.seed_fail:
        import auto2dlabel.agent.planner as _planner_mod

        def _boom(*_: object, **_kw: object) -> object:
            _ = (_kw,)
            raise RuntimeError("种子检索失败（测试静默降级）")

        setattr(_planner_mod, "retrieve", _boom)
        print("[dim]种子检索失败场景：验证 planner 静默降级[/dim]")

    # 单组运行：真实 LLM 规划（chat 路径）→ 检测 → 评估
    from auto2dlabel.agent.llm import create_client
    from auto2dlabel.agent.planner import TaskPlanner

    # 数据集分支：cityscapes 域指令（检索命中「域内权重」种子）+ 域 GT；
    # coco 全类指令（检索无命中 → 两组行为一致，对照负例）
    if args.dataset == "cityscapes":
        from auto2dlabel.benchmarks.cityscapes_benchmark import (
            _gt_bbox_view as _cs_bbox_view,
        )
        from auto2dlabel.benchmarks.cityscapes_benchmark import (
            load_cityscapes_ground_truth,
            run_detection_only,
        )
        from auto2dlabel.benchmarks.datasets import ensure_cityscapes_val

        cs_root = ensure_cityscapes_val()
        gt_dir = cs_root / "gtFine" / "val"
        img_dir = cs_root / "leftImg8bit" / "val"
        print("\n加载 cityscapes Ground Truth（instanceIds.png，8 类 thing）...")
        gt = load_cityscapes_ground_truth(gt_dir, img_dir, max_images=args.max_images)
        print(f"已加载 {len(gt)} 张图像\n")

        all_cats = sorted(set(o["name"] for g in gt.values() for o in g["objects"]))
        # 域指令：明确提到街景小目标——注入组应检索命中 cityscapes 种子
        # （model=maskrcnn_r50_cityscapes，经验「COCO 预训练对小目标失效」）
        instruction = (
            "检测 cityscapes 街景中的小目标（30~50px）: " + ", ".join(all_cats)
        )
        print(f"指令: {instruction[:160]}...\n")

        def run_det(gt_: Any, model_name: str) -> Any:
            # det-only：GT 对象是 mask，需转 bbox 视图（cityscapes_benchmark 同款）
            return run_detection_only(
                _cs_bbox_view(gt_), img_dir, args.conf, args.iou, model_name,
                batch=args.batch, workers=args.workers,
            )
    elif args.dataset == "dota":
        from auto2dlabel.benchmarks.datasets import ensure_dota_val
        from auto2dlabel.benchmarks.dota_benchmark import (
            load_dota_ground_truth,
        )
        from auto2dlabel.benchmarks.dota_benchmark import (
            run_detection as run_dota_detection,
        )

        dota_root = ensure_dota_val()
        img_dir = dota_root / "images"
        label_dir = dota_root / "labels"
        print("\n加载 DOTA Ground Truth（HBB，4 类可映射 COCO）...")
        gt = load_dota_ground_truth(img_dir, label_dir, max_images=args.max_images)
        print(f"已加载 {len(gt)} 张图像\n")

        all_cats = sorted(set(o["name"] for g in gt.values() for o in g["objects"]))
        # 域指令：航拍域 + 旋转框——RAG 增量正例（基线 prompt 无「航拍域 COCO
        # 权重失效需域微调」经验；种子库 DOTA 条目携带该经验）
        instruction = "检测航拍图像中的飞机和舰船（旋转框）: " + ", ".join(all_cats)
        print(f"指令: {instruction[:160]}...\n")

        def run_det(gt_: Any, model_name: str) -> Any:
            return run_dota_detection(
                gt_, img_dir, model_name, args.conf, args.iou,
                batch=args.batch, workers=args.workers,
            )

        _cs_bbox_view = _identity_gt  # DOTA GT 已是 bbox 直接评估
    else:
        print("\n加载 COCO val 2017 Ground Truth...")
        gt = load_coco_ground_truth(max_images=args.max_images)
        print(f"已加载 {len(gt)} 张图像\n")

        all_cats = sorted(set(o["name"] for g in gt.values() for o in g["objects"]))
        instruction = f"检测以下图像中的物体：{', '.join(all_cats)}"
        print(f"指令: {instruction[:120]}...\n")

        def run_coco_det(gt_: Any, model_name: str) -> Any:
            # coco_benchmark.run_detection 内部用 CONFIG["model_name"]（main 已覆写）
            _ = model_name  # 签名统一（cityscapes/dota 分支用 plan 模型；此处已由 CONFIG 生效）
            return run_detection(
                gt_, batch=args.batch, workers=args.workers,
            )

        run_det = run_coco_det
        _cs_bbox_view = _identity_gt  # COCO GT 直接评估（无 mask 转换）

    llm = create_client(provider="deepseek")
    planner = TaskPlanner(llm_client=llm)
    # 规划：注入组 system prompt 含 RAG few-shot（_build_system_prompt 动态检索）；
    # 基线组用 import 期常量（无 history 段）。plan 决定检测类别/阈值。
    plan = planner.parse(instruction, confirm_timeout=0)
    print(f"计划: {len(plan.steps)} 步 | {plan.summary.replace(chr(10), ' ')[:160]}")

    # 执行检测（用 plan 决定的模型——验证 RAG 注入是否改变模型选型；
    # 去掉 --model 锁死后，注入组若命中域内权重种子应选 maskrcnn_r50_cityscapes）
    plan_model = plan.steps[0].model_name if plan.steps else args.model
    CONFIG["model_name"] = plan_model  # coco 路径 run_detection 内部读 CONFIG
    print(f"[检测模型（来自 plan）: {plan_model}]")
    predictions = run_det(gt, plan_model)

    # 评估（cityscapes GT 是 mask，需转 bbox 视图；coco/dota 已是 bbox 直接评估）
    eval_gt = _cs_bbox_view(gt)
    print(f"\n计算指标（IoU@{IOU_MATCH_THRESHOLD}）...")
    results = {cls: evaluate_per_class(eval_gt, predictions, cls, IOU_MATCH_THRESHOLD)
               for cls in all_cats}
    summary = format_result_table(results, all_cats, top_n=args.top_classes)
    print(summary)

    map_ap = float(np.mean([results[cls]["ap"] for cls in all_cats if cls in results]))

    # LLM 调用数 / 费用对比（台账按 call_site 聚合）
    usage = _usage_stats(
        Path(args.usage) if args.usage else None, {"planner.dialog", "planner.parse"}
    )

    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    result_data = {
        "timestamp": ts, "dataset": args.dataset,
        "group": group, "inject": args.inject,
        "model": args.model, "confidence_threshold": args.conf,
        "iou_match_threshold": IOU_MATCH_THRESHOLD, "image_count": len(gt),
        "planner_instruction": instruction,
        "summary": {"mAP@0.5": round(map_ap, 4)},
        "usage": usage,
        "per_class": {cls: results[cls] for cls in all_cats},
    }
    json_path, md_path = save_results(result_data, f"experience_{group}", args.model, ts)
    md_path.write_text("\n".join([
        f"# Experience {group} Benchmark",
        f"- **模型**: {args.model} | **conf**: {args.conf} | **mAP@0.5**: {map_ap:.4f}",
        f"- **LLM 调用**: {usage['calls']} 次 / {usage['tokens']} tokens / ¥{usage['cost_rmb']}",
        f"- **注入**: {args.inject}",
        f"```\n{summary}\n```",
    ]))
    print(f"\n[{group}] mAP@0.5 = {map_ap:.4f} | LLM 调用 = {usage['calls']} 次 | "
          f"费用 = ¥{usage['cost_rmb']}")
    print(f"结果: {json_path}")
    print(f"报告: {md_path}")


if __name__ == "__main__":
    main()
