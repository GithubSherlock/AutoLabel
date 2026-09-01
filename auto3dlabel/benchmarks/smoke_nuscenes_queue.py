"""nuScenes 复核队列冒烟（P2）：val 场景 → {sample_token}_review.json 队列（resume 幂等）。

用法：python -m auto3dlabel.benchmarks.smoke_nuscenes_queue [model] [conf] [out_dir]
默认 pointpillars_nus / 0.3 / outputs/nuscenes_queue。
验收：队列文件 dataset=="nuscenes"、annotations 为 cam_like 渲染 dict、summary 三档计数。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rich.console import Console

from auto3dlabel.configs.kitti import DEFAULT_CONF
from auto3dlabel.models.detection3d import create_detector3d_any
from auto3dlabel.tools.nuscenes_pipeline import generate_review_queue


def main(argv: list[str] | None = None) -> None:
    argv = argv or sys.argv[1:]
    model = argv[0] if len(argv) > 0 else "pointpillars_nus"
    conf = float(argv[1]) if len(argv) > 1 else DEFAULT_CONF
    out_dir = Path(argv[2]) if len(argv) > 2 else Path("outputs/nuscenes_queue")

    det = create_detector3d_any(model)
    if det is None:
        raise SystemExit(f"未知名 3D 模型：{model}")
    console = Console()
    console.print(f"[bold]冒烟[/bold] model={model} conf={conf} out_dir={out_dir}")

    written = generate_review_queue(det, out_dir, conf=conf)
    n_skip = len(list(out_dir.glob("*_review.json"))) - len(written)
    console.print(f"[bold]队列[/bold] 新写 {len(written)} / 跳过 {n_skip}（resume）")

    # 抽查：全部队列文件 protocol 键 + summary 汇总
    n_ann = 0
    n_need = 0
    for path in sorted(out_dir.glob("*_review.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["dataset"] == "nuscenes" and data["sample_token"] in path.name
        assert len(data["cameras"]) == 6 and data["ego_translation"] is not None
        n_ann += len(data["annotations"])
        n_need += data["summary"]["total_need_review"]
    console.print(f"[bold]汇总[/bold] {len(list(out_dir.glob('*_review.json')))} 个队列文件 "
                  f"/ 待复核框 {n_need} / annotations {n_ann}")


if __name__ == "__main__":
    main()
