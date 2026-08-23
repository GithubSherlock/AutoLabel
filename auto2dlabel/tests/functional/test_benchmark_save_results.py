"""save_results 文件名 sanitize 回归 — 模型名含路径分隔符时不得形成嵌套目录。

根因（2026-08-23）：微调产物 best.pt 以完整路径传入 kitti_benchmark，
f"kitti_{model}_{ts}" 拼出 "kitti_/root/.../best.pt_..." 非法嵌套路径，
指标算完却崩在保存。修复在 common.save_results 单一收口点。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from auto2dlabel.benchmarks import common
from auto2dlabel.benchmarks.common import save_results


def _saved_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[Path]:
    """monkeypatch 输出目录为 tmp_path（不污染真实 benchmarks_outputs）。"""
    monkeypatch.setattr(common, "OUTPUT_DIR", tmp_path)
    return list(tmp_path.iterdir())


def test_save_results_sanitizes_model_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """微调产物完整路径 → 单层输出文件，无嵌套目录。"""
    _saved_files(monkeypatch, tmp_path)
    model = "/root/autodl-tmp/Documents/Projects/AutoLabel/auto2dlabel/weights/" \
        "kitti_finetune/yolo11s_kitti/weights/best.pt"
    json_path, md_path = save_results({"mAP": 0.8867}, "kitti", model, ts="T")

    assert json_path.parent == tmp_path and md_path.parent == tmp_path
    assert json_path.name == md_path.name.replace(".md", ".json")
    assert json_path.name == "kitti__root_autodl-tmp_Documents_Projects_AutoLabel_" \
        "auto2dlabel_weights_kitti_finetune_yolo11s_kitti_weights_best.pt_T.json"
    assert json_path.exists()


def test_save_results_hf_repo_id_sanitized(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """HF repo id（IDEA-Research/grounding-dino-tiny）同样含 "/" → 下划线。"""
    _saved_files(monkeypatch, tmp_path)
    json_path, _ = save_results({}, "detection", "IDEA-Research/grounding-dino-tiny", ts="T")
    assert json_path.parent == tmp_path
    assert json_path.name == "detection_IDEA-Research_grounding-dino-tiny_T.json"
    assert json_path.exists()


def test_save_results_plain_model_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """普通模型名（yolo26x.pt）行为不变。"""
    _saved_files(monkeypatch, tmp_path)
    json_path, _ = save_results({}, "kitti", "yolo26x.pt", ts="T")
    assert json_path.name == "kitti_yolo26x.pt_T.json"
    assert json_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
