"""CLI 入口 —— 命令行标注工具。

Usage:
    auto2dlabel run image.jpg "detect all cars and pedestrians"
    auto2dlabel run image.jpg "detect all cars" --threshold 0.5 --export coco
    auto2dlabel run ./images/ "detect cars, bikes, people" --batch --export yolo
    auto2dlabel run video.mp4 "检测行人" --track
"""

from __future__ import annotations

import typer

from auto2dlabel.cli_commands import (
    chat_command,
    cost_report_command,
    dataset_add_command,
    dataset_list_command,
    dataset_remove_command,
    sample_command,
    tools_command,
)
from auto2dlabel.cli_run import run_command

app = typer.Typer(
    name="auto2dlabel",
    help="Agentic 2D image annotation — AI-first, human-in-the-loop.",
    no_args_is_help=True,
)


@app.command()
def run(
    image: str = typer.Argument("", help="图像路径或目录（支持 JPG/PNG/TIFF）；--resume 时忽略"),
    instruction: str = typer.Argument(
        "", help="标注指令，用自然语言描述要标注什么；--resume 时从清单读取"
    ),
    threshold: float = typer.Option(0.1, "--threshold", "-t", help="置信度阈值 (0-1)"),
    iou: float = typer.Option(0.3, "--iou", help="IoU 阈值 (0-1)，NMS 去重力度"),
    export: str = typer.Option(
        "coco", "--export", "-e", help="导出格式: coco, yolo, voc, labelme"
    ),
    output: str = typer.Option("outputs", "--output", "-o", help="输出目录"),
    batch: bool = typer.Option(False, "--batch", "-b", help="如果是目录, 批量处理所有图像"),
    resume: str = typer.Option(
        "", "--resume", help="从批次清单续跑（跳过 ok，重跑 failed/pending）"
    ),
    provider: str = typer.Option(
        "deepseek", "--provider", "-p", help="LLM provider: openai, anthropic, deepseek"
    ),
    model: str | None = typer.Option(None, "--model", "-m", help="LLM 模型名"),
    det_model: str | None = typer.Option(
        None,
        "--det-model",
        "-d",
        help="检测模型: grounding-dino-tiny | yolo26x.pt | fasterrcnn_resnet50_fpn | ...",
    ),
    api_key: str = typer.Option(None, "--api-key", help="LLM API key"),
    base_url: str = typer.Option(None, "--base-url", help="LLM API base URL"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志输出"),
    sahi: bool = typer.Option(False, "--sahi", help="启用 SAHI 切片推理（大分辨率图像）"),
    track: bool = typer.Option(
        False, "--track",
        help="跟踪模式：视频/帧目录逐帧检测 + ByteTrack ID 维持 + MOT 导出",
    ),
    llm: bool = typer.Option(
        False, "--llm",
        help="跟踪模式下启用 LLM 一次性指令解析（复杂指令；失败自动回退代码级解析）",
    ),
    bot_sort: bool = typer.Option(
        False, "--bot-sort",
        help="跟踪模式精度档：BoT-SORT（ReID 外观关联 + ECC 相机运动补偿）",
    ),
    reid_model: str = typer.Option(
        "openai/clip-vit-base-patch32",
        "--reid-model",
        help="BoT-SORT ReID 特征模型（clip 或 siglip，见 model_catalog.REID_MODELS）",
    ),
    no_viz: bool = typer.Option(
        False, "--no-viz",
        help="跟踪模式：跳过逐帧 PNG 可视化（vis_outputs），省磁盘；MOT/JSON/成片视频不受影响",
    ),
    roi: str = typer.Option(
        None, "--roi",
        help="跟踪模式空间约束（ROI）：矩形 'x1,y1,x2,y2' 或分号多边形 'x1,y1;x2,y2;...'，"
        "只保留框中心点在区域内的目标（如车道区域）；'auto' 自动检测自车车道多边形"
        "（首帧 UFLD 车道线模型，权重下载 auto2dlabel/weights/download_lane_weights.sh）",
    ),
    refer_l2: bool = typer.Option(
        False, "--refer-l2",
        help="指代 L2（Florence-2 兜底）：关系/复合指代（如「红车旁边的行人」）首帧"
        "解析锁定目标（指令含 旁边/附近/之间 等关系词时自动触发）",
    ),
    refer_l3: bool = typer.Option(
        False, "--refer-l3",
        help="指代 L3（Qwen2-VL-7B，GPU）：复杂上下文指代直用 L3 首帧解析"
        "（默认指代路径为阶梯升级——L2 失败自动升级 L3；权重下载 "
        "auto2dlabel/weights/download_qwen_l3.sh）",
    ),
) -> None:
    """对图像运行 Agentic 标注。"""
    run_command(
        image=image,
        instruction=instruction,
        threshold=threshold,
        iou=iou,
        export=export,
        output=output,
        batch=batch,
        resume=resume,
        provider=provider,
        model=model,
        det_model=det_model,
        api_key=api_key,
        base_url=base_url,
        verbose=verbose,
        sahi=sahi,
        track=track,
        use_llm=llm,
        bot_sort=bot_sort,
        reid_model=reid_model,
        viz=not no_viz,
        roi=roi,
        refer_l2=refer_l2,
        refer_l3=refer_l3,
    )


@app.command()
def sample(
    top_k: int = typer.Option(10, "--top-k", "-k", help="返回前 K 个最不确定的样本"),
    review_dir: str = typer.Option("outputs", "--review-dir", help="复核队列目录"),
    output: str = typer.Option(
        "outputs/sampling_manifest.json", "--output", "-o", help="采样清单输出路径"
    ),
) -> None:
    """主动学习采样：从 HITL 复核队列挑选信息量最高的样本。"""
    sample_command(top_k=top_k, review_dir=review_dir, output=output)


@app.command()
def tools() -> None:
    """列出所有可用的标注 Tool。"""
    tools_command()


@app.command()
def cost_report(
    path: str = typer.Option("", "--path", help="台账 JSONL 路径（默认 logs/llm_usage.jsonl）"),
) -> None:
    """按调用点聚合 LLM token 用量与费用（v0.6 Phase 4 usage 台账）。"""
    cost_report_command(path=path or None)


dataset_app = typer.Typer(help="管理用户自建数据集注册（供 LLM 路径引导）")
app.add_typer(dataset_app, name="dataset")


@dataset_app.command("add")
def dataset_add(
    name: str = typer.Argument(..., help="数据集名（chat 指令中引用的名字）"),
    path: str = typer.Argument(..., help="数据集根目录（必须存在）"),
    subdirs: str = typer.Option(
        "", "--subdirs", help="关键子目录，逗号分隔（如 images,annotations）"
    ),
    task: str = typer.Option("", "--task", help="任务描述（如 实例分割）"),
    note: str = typer.Option("", "--note", help="备注"),
) -> None:
    """注册用户自建数据集 → configs/user_datasets.yaml（LLM 路径引导用）。"""
    dataset_add_command(
        name=name,
        path=path,
        subdirs=[s.strip() for s in subdirs.split(",") if s.strip()],
        task=task,
        note=note,
    )


@dataset_app.command("list")
def dataset_list() -> None:
    """列出已注册的自建数据集。"""
    dataset_list_command()


@dataset_app.command("remove")
def dataset_remove(
    name: str = typer.Argument(..., help="数据集名"),
) -> None:
    """删除已注册的自建数据集。"""
    dataset_remove_command(name=name)


@app.command()
def chat(
    instruction: str = typer.Argument(None, help="自然语言标注指令（省略则进入交互模式）"),
    det_model: str = typer.Option(None, "--det-model", "-d", help="覆盖检测模型（如 rtdetr-l.pt）"),
    confirm_timeout: int = typer.Option(30, "--timeout", help="确认等待秒数（0 = 跳过确认）"),
    no_wait: bool = typer.Option(False, "--no-wait", help="跳过所有确认，直接执行"),
    provider: str = typer.Option("deepseek", "--provider", "-p"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    sahi: bool = typer.Option(False, "--sahi", help="启用 SAHI 切片推理（大分辨率图像）"),
    batch_size: int = typer.Option(
        None, "--batch-size", "-b", min=1,
        help="批量推理每批图像数（1=逐图；缺省交互询问，未指定按 GPU 显存自动推荐）",
    ),
    num_workers: int = typer.Option(
        None, "--num-workers", min=0,
        help="DataLoader 子进程数（缺省交互询问，未指定按 GPU 显存自动推荐）",
    ),
    no_viz: bool = typer.Option(
        False, "--no-viz",
        help="跟踪指令：跳过逐帧 PNG 可视化（vis_outputs），省磁盘；MOT/JSON/成片视频不受影响",
    ),
    batch_strategy: bool = typer.Option(
        False, "--batch-strategy",
        help="批量检测：抽样统计后让 LLM 一次性调参（阈值覆写 + 模型建议；每批 1 次 LLM 调用）",
    ),
    refer_l2: bool = typer.Option(
        False, "--refer-l2",
        help="指代 L2（Florence-2 兜底）：跟踪指令含关系词（旁边/附近/之间）自动触发，"
        "也可显式启用——首帧解析锁定目标",
    ),
    refer_l3: bool = typer.Option(
        False, "--refer-l3",
        help="指代 L3（Qwen2-VL-7B，GPU）：复杂上下文指代直用 L3 首帧解析"
        "（默认指代路径为阶梯升级——L2 失败自动升级 L3）",
    ),
) -> None:
    """自然语言驱动的 Agentic 标注——无需记忆 CLI 参数。

    示例:
        auto2dlabel chat "检测 000860.png 中的汽车和行人，conf=0.5"
        auto2dlabel chat "检测汽车和行人" -d rtdetr-x.pt --no-wait
        auto2dlabel chat "检测汽车和行人" --batch-size 8 --num-workers 4
        auto2dlabel chat "检测汽车和行人" --batch-strategy  # 大批量：抽样 + LLM 调参
        auto2dlabel chat "跟踪 video.mp4 中红车旁边的行人" --no-wait  # 关系指代 → L2 自动触发
        auto2dlabel chat  →  进入交互对话模式
    """
    chat_command(
        instruction=instruction,
        det_model=det_model,
        confirm_timeout=confirm_timeout,
        no_wait=no_wait,
        provider=provider,
        verbose=verbose,
        sahi=sahi,
        batch_size=batch_size,
        num_workers=num_workers,
        viz=not no_viz,
        batch_strategy=batch_strategy,
        refer_l2=refer_l2,
        refer_l3=refer_l3,
    )


if __name__ == "__main__":
    app()
