"""分类评测函数单元测试 — top-1/top-K 准确率 + 逐类统计 + 表格格式化（零 torch、零下载）。

另含 datasets.py 分类接入的纯函数部分：均匀抽样解压（临时 zip 构造）。
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from auto2dlabel.benchmarks.common import (
    evaluate_classification,
    format_classification_table,
)
from auto2dlabel.benchmarks.datasets import extract_zip_members_uniform


def _gt(*labels: str) -> dict[str, dict[str, object]]:
    return {f"img{i}": {"file_name": f"img{i}.jpg", "label": lab}
            for i, lab in enumerate(labels)}


def _pred(**kwargs: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    return kwargs


class TestEvaluateClassification:
    def test_perfect_top1_and_top5(self) -> None:
        """全部 top-1 命中 → top1/top5 均为 1.0。"""
        gt = _gt("cat", "dog", "bird")
        pred = _pred(
            img0=[{"label": "cat", "score": 0.9}],
            img1=[{"label": "dog", "score": 0.9}],
            img2=[{"label": "bird", "score": 0.9}],
        )
        result = evaluate_classification(gt, pred)
        assert result["top1"] == 1.0
        assert result["top5"] == 1.0

    def test_top5_hit_not_top1(self) -> None:
        """GT 位于第 3 位 → top1 0.0、top5 1.0。"""
        gt = _gt("cat")
        pred = _pred(
            img0=[
                {"label": "dog", "score": 0.9},
                {"label": "bird", "score": 0.8},
                {"label": "cat", "score": 0.7},
            ],
        )
        result = evaluate_classification(gt, pred)
        assert result["top1"] == 0.0
        assert result["top5"] == 1.0

    def test_miss_beyond_topk(self) -> None:
        """GT 在第 6 位 → top5 不命中。"""
        gt = _gt("cat")
        pred = _pred(
            img0=[{"label": f"cls{i}", "score": 1.0 - i * 0.1} for i in range(5)]
            + [{"label": "cat", "score": 0.4}],
        )
        result = evaluate_classification(gt, pred)
        assert result["top1"] == 0.0
        assert result["top5"] == 0.0

    def test_missing_pred_key_counts_miss(self) -> None:
        """GT 无对应 pred 键 → 计 miss（不抛异常）。"""
        gt = _gt("cat", "dog")
        pred = _pred(img0=[{"label": "cat", "score": 0.9}])
        result = evaluate_classification(gt, pred)
        assert result["top1"] == 0.5
        assert result["top5"] == 0.5

    def test_empty_pred_list_counts_miss(self) -> None:
        """pred 空列表 → 计 miss。"""
        gt = _gt("cat")
        pred = _pred(img0=[])
        result = evaluate_classification(gt, pred)
        assert result["top1"] == 0.0
        assert result["top5"] == 0.0

    def test_top_k_variants(self) -> None:
        """top_k=1 与 top_k=3 键名与判定边界。"""
        gt = _gt("cat")
        pred = _pred(
            img0=[
                {"label": "a", "score": 0.9},
                {"label": "b", "score": 0.8},
                {"label": "cat", "score": 0.7},
            ],
        )
        assert evaluate_classification(gt, pred, top_k=1)["top1"] == 0.0
        assert evaluate_classification(gt, pred, top_k=1)["top1"] == 0.0
        r3 = evaluate_classification(gt, pred, top_k=3)
        assert r3["top3"] == 1.0

    def test_per_class_consistent_with_global(self) -> None:
        """逐类统计与全局口径一致。"""
        gt = _gt("cat", "cat", "dog")
        pred = _pred(
            img0=[{"label": "cat", "score": 0.9}],
            img1=[{"label": "dog", "score": 0.9}],  # cat 类错判
            img2=[{"label": "dog", "score": 0.9}],
        )
        result = evaluate_classification(gt, pred)
        assert result["per_class"]["cat"] == {"correct": 1, "total": 2, "accuracy": 0.5}
        assert result["per_class"]["dog"] == {"correct": 1, "total": 1, "accuracy": 1.0}
        # 全局 correct = 逐类 correct 之和
        assert sum(r["correct"] for r in result["per_class"].values()) == 2
        assert result["top1"] == round(2 / 3, 4)


class TestFormatClassificationTable:
    def test_header_and_summary(self) -> None:
        """表格含列头、逐类行与 top-1/top-5 汇总行。"""
        gt = _gt("cat", "dog")
        pred = _pred(
            img0=[{"label": "cat", "score": 0.9}],
            img1=[{"label": "cat", "score": 0.9}],
        )
        result = evaluate_classification(gt, pred)
        table = format_classification_table(result, ["cat", "dog"])
        assert "Class" in table and "Total" in table and "Correct" in table and "Acc" in table
        assert "cat" in table and "dog" in table
        assert "top-1: 0.5000" in table
        assert "top5: 0.5000" in table

    def test_top_n_limits_rows(self) -> None:
        """top_n 限制展示行数。"""
        labels = [f"cls{i}" for i in range(30)]
        gt = _gt(*labels)
        pred = {img_id: [{"label": labels[i], "score": 1.0}]
                for i, img_id in enumerate(gt)}
        result = evaluate_classification(gt, pred)
        table = format_classification_table(result, labels, top_n=10)
        # 10 个逐类行 + 分隔线，cls29 不在展示行内
        assert "cls0" in table
        assert "cls29" not in table
        assert "总体 (30 cls)" in table


class TestExtractZipMembersUniform:
    """均匀抽样解压（临时 zip 构造，零外部依赖）。"""

    def _make_zip(self, tmp_path: Path, n_classes: int = 3, n_files: int = 5) -> Path:
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            # 目录条目刻意放在文件之前（复刻 ImageNet100 zip 结构）
            for c in range(n_classes):
                zf.writestr(f"data/cls{c}/", "")
                for f in range(n_files):
                    zf.writestr(f"data/cls{c}/img{f}.jpg", b"x")
        return zip_path

    def test_per_class_sampling(self, tmp_path: Path) -> None:
        """每类取前 N 张，目录条目被过滤。"""
        zip_path = self._make_zip(tmp_path)
        dest = tmp_path / "out"
        dest.mkdir()
        extract_zip_members_uniform(zip_path, dest, member_prefix="data/", per_class=2)
        for c in range(3):
            files = sorted((dest / "data" / f"cls{c}").iterdir())
            assert [f.name for f in files] == ["img0.jpg", "img1.jpg"]

    def test_idempotent(self, tmp_path: Path) -> None:
        """二次调用零新增（幂等，缺失检查）。"""
        zip_path = self._make_zip(tmp_path)
        dest = tmp_path / "out"
        dest.mkdir()
        extract_zip_members_uniform(zip_path, dest, member_prefix="data/", per_class=1)
        before = sorted(p.relative_to(dest) for p in dest.rglob("*.jpg"))
        extract_zip_members_uniform(zip_path, dest, member_prefix="data/", per_class=1)
        after = sorted(p.relative_to(dest) for p in dest.rglob("*.jpg"))
        assert before == after

    def test_per_class_zero_means_all(self, tmp_path: Path) -> None:
        """per_class=0 提取全部文件。"""
        zip_path = self._make_zip(tmp_path)
        dest = tmp_path / "out"
        dest.mkdir()
        extract_zip_members_uniform(zip_path, dest, member_prefix="data/", per_class=0)
        total = len(list(dest.rglob("*.jpg")))
        assert total == 15
