"""no-LLM baseline 与 benchmark runner 工具测试。

覆盖 tests/helpers/ 中自 cli.py 迁入的工具：

- extract_prompts：中→英关键词映射（--no-llm 模式的核心）
- run_no_llm_baseline：端到端输出与 LLM 路径格式一致（JSON/PNG/HITL）
- is_benchmark_intent：benchmark vs annotation 意图检测
- execute_benchmark：数据集校验
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from auto2dlabel.tests.helpers.benchmark_runner import (
    execute_benchmark,
    is_benchmark_intent,
)
from auto2dlabel.tests.helpers.no_llm_baseline import (
    extract_prompts,
    run_no_llm_baseline,
)

_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights"
_YOLOV8X = _WEIGHTS_DIR / "yolov8x.pt"


# ============================================================
# extract_prompts：中→英关键词映射
# ============================================================

class TestExtractPrompts:
    def test_cn_to_en(self) -> None:
        assert extract_prompts("检测汽车和行人") == ["person", "car"]

    def test_dedup_synonyms(self) -> None:
        # 「车」与「汽车」都映射到 car，只保留一次
        assert extract_prompts("检测车和汽车") == ["car"]

    def test_no_match_raises(self) -> None:
        # 映射表为子串匹配，「飞机」不含任何关键词
        with pytest.raises(ValueError, match="无法从指令中提取关键词"):
            extract_prompts("检测飞机")

    def test_substring_match(self) -> None:
        # 「人」是子串关键词，「外星人」也会命中（与原 CLI 行为一致）
        assert extract_prompts("检测外星人") == ["person"]

    def test_english_only_raises(self) -> None:
        # 映射表只含中文关键词（与 CLI 原行为一致）
        with pytest.raises(ValueError):
            extract_prompts("detect cars")


# ============================================================
# run_no_llm_baseline：端到端（需要检测模型权重）
# ============================================================

@pytest.mark.skipif(not _YOLOV8X.exists(), reason="需要 yolov8x.pt 权重（auto2dlabel/weights/）")
class TestNoLlmBaselineE2E:
    def test_end_to_end_outputs(self, sample_image: Path, tmp_path: Path) -> None:
        """baseline 输出与 LLM 路径格式一致：JSON / 可视化 / Annotation 结构。"""
        out = tmp_path / "outputs"
        vis = tmp_path / "vis"
        results = run_no_llm_baseline(
            [sample_image],
            "检测汽车和行人",
            threshold=0.1,
            det_model="yolov8x.pt",
            output=str(out),
            vis_dir=str(vis),
        )

        assert len(results) == 1
        r = results[0]
        assert r.prompts == ["person", "car"]
        assert r.annotation is not None
        assert len(r.detections) == len(r.annotation.bboxes)

        # JSON 导出可解析且图像尺寸正确
        assert r.json_path is not None and r.json_path.exists()
        data = json.loads(r.json_path.read_text())
        assert data["images"][0]["width"] == 640
        assert data["images"][0]["height"] == 480

        # 可视化 PNG
        assert r.vis_path is not None and r.vis_path.exists()

        # Annotation 结构完整
        assert r.annotation.image_size == (640, 480)

    def test_sahi_branch(self, sample_image: Path, tmp_path: Path) -> None:
        """SAHI 分支：640×480 ≤ 切片尺寸时走直接推理路径，输出结构一致。"""
        results = run_no_llm_baseline(
            [sample_image],
            "检测汽车",
            threshold=0.1,
            det_model="yolov8x.pt",
            sahi=True,
            output=str(tmp_path / "out"),
            vis_dir=str(tmp_path / "vis"),
        )

        assert len(results) == 1
        assert results[0].json_path is not None and results[0].json_path.exists()
        assert results[0].vis_path is not None and results[0].vis_path.exists()


# ============================================================
# benchmark_runner 工具
# ============================================================

class TestIsBenchmarkIntent:
    def test_benchmark_zh(self) -> None:
        assert is_benchmark_intent("用 yolov8x 跑 COCO 检测 benchmark，50 张图")

    def test_benchmark_eval_only(self) -> None:
        assert is_benchmark_intent("评估 KITTI 数据集上的检测性能")

    def test_annotation_with_image_path(self) -> None:
        assert not is_benchmark_intent("检测 000860.png 中的汽车")

    def test_annotation_explicit(self) -> None:
        assert not is_benchmark_intent("标注这张图里的行人")

    def test_dataset_without_benchmark_ctx(self) -> None:
        assert not is_benchmark_intent("检测 COCO 图片")


class TestExecuteBenchmark:
    def test_unknown_dataset_raises(self) -> None:
        from auto2dlabel.schema.task_plan import BenchmarkRequest

        with pytest.raises(ValueError, match="未知数据集"):
            execute_benchmark(BenchmarkRequest(dataset="nope"))
