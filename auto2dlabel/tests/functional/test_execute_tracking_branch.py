"""execute_plan tracking 分支测试 —— monkeypatch TrackingTool，零模型零权重。

覆盖：tracking 步骤短路进 TrackingTool（不进 collect_images）、参数透传
（batch 显式优先 / mot→coco 映射 / 跟踪器选择）、steps_results 汇总、
ValueError 优雅跳过、非 tracking 步骤回归不受影响。
"""

from __future__ import annotations

from typing import Any

from auto2dlabel.agent.planner import _dict_to_plan
from auto2dlabel.cli_execute import execute_plan
from auto2dlabel.tests import Path


class _FakeTool:
    """记录构造参数与 forward 入参的假 TrackingTool（返回固定汇总）。"""

    instances: list[_FakeTool] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        _FakeTool.instances.append(self)

    def forward(
        self,
        source: str,
        prompts: list[str],
        confidence_threshold: float = 0.1,
    ) -> dict[str, Any]:
        self.last = {
            "source": source,
            "prompts": prompts,
            "confidence_threshold": confidence_threshold,
        }
        return {
            "frames": 5,
            "bboxes": 12,
            "track_ids": [0, 1],
            "mot_path": "outputs/x_mot.txt",
        }


def _tracking_plan() -> Any:
    return _dict_to_plan(
        {
            "steps": [
                {
                    "step_id": 1,
                    "task_type": "tracking",
                    "source": "/nonexistent/video.mp4",
                    "prompts": ["person", "car"],
                    "model_name": "yolo12n.pt",
                    "export_format": "mot",
                }
            ],
            "confirm_timeout": 30,
        }
    )


def test_tracking_step_short_circuits_into_tool(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    """tracking 步骤 → TrackingTool（参数透传正确），不进 collect_images 循环。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("auto2dlabel.tools.tracking.TrackingTool", _FakeTool)
    logged: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "auto2dlabel.tools.log.log_chat_call",
        lambda **kw: logged.append(kw),
    )

    execute_plan(
        _tracking_plan(),
        use_bot_sort=True,
        explicit_batch_size=2,
        output_dir="/tmp/out",
    )

    out = capsys.readouterr().out
    assert "未找到图像" not in out  # 未走 collect_images 路径
    assert "跟踪失败" not in out

    assert len(_FakeTool.instances) == 1
    tool = _FakeTool.instances[0]
    assert tool.kwargs["model_name"] == "yolo12n.pt"
    assert tool.kwargs["use_bot_sort"] is True
    assert tool.kwargs["batch_size"] == 2  # CLI 显式优先
    assert tool.kwargs["export_format"] == "coco"  # mot → 逐帧 coco 映射
    assert tool.last == {
        "source": "/nonexistent/video.mp4",
        "prompts": ["person", "car"],
        "confidence_threshold": 0.1,
    }

    # steps_results 汇总并入顶层 chat 日志
    assert len(logged) == 1
    results = logged[0]["steps_results"]
    assert len(results) == 1
    entry = results[0]
    assert entry["bbox_count"] == 12
    assert entry["track_ids"] == [0, 1]
    assert entry["mot_path"] == "outputs/x_mot.txt"
    assert entry["video_path"] is None  # forward 未带视频输出时 .get 兜底
    assert entry["step_id"] == 1


def test_tracking_value_error_skips_step_gracefully(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    """TrackingTool.forward 抛 ValueError → 红字提示、不冒泡、本步跳过。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("auto2dlabel.tools.log.log_chat_call", lambda **kw: None)

    class _BoomTool(_FakeTool):
        def forward(
            self,
            source: str,
            prompts: list[str],
            confidence_threshold: float = 0.1,
        ) -> dict[str, Any]:
            raise ValueError("未找到可跟踪的帧/视频")

    monkeypatch.setattr("auto2dlabel.tools.tracking.TrackingTool", _BoomTool)

    execute_plan(_tracking_plan())  # 不抛异常

    out = capsys.readouterr().out
    assert "跟踪失败" in out
    assert "未找到可跟踪的帧/视频" in out


def test_non_tracking_step_path_regression(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    """非 tracking 步骤不受影响：仍走 collect_images（源不存在 → 未找到图像）。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("auto2dlabel.tools.log.log_chat_call", lambda **kw: None)
    plan = _dict_to_plan(
        {
            "steps": [
                {
                    "step_id": 1,
                    "task_type": "object_detection",
                    "source": "/nonexistent/imgs",
                    "prompts": ["person"],
                    "model_name": "yolo12n.pt",
                }
            ],
            "confirm_timeout": 30,
        }
    )

    execute_plan(plan)

    out = capsys.readouterr().out
    assert "未找到图像" in out
