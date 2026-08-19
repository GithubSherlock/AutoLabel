"""CLI 入口 —— 命令行标注工具。

Usage:
    auto2dlabel run image.jpg "detect all cars and pedestrians"
    auto2dlabel run image.jpg "detect all cars" --threshold 0.5 --export coco
    auto2dlabel run ./images/ "detect cars, bikes, people" --batch --export yolo
    auto2dlabel run video.mp4 "检测行人" --track
"""

from __future__ import annotations

import typer

from auto2dlabel.cli_commands import chat_command, sample_command, tools_command
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
) -> None:
    """自然语言驱动的 Agentic 标注——无需记忆 CLI 参数。

    示例:
        auto2dlabel chat "检测 000860.png 中的汽车和行人，conf=0.5"
        auto2dlabel chat "检测汽车和行人" -d rtdetr-x.pt --no-wait
        auto2dlabel chat "检测汽车和行人" --batch-size 8 --num-workers 4
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
    )


if __name__ == "__main__":
    app()
