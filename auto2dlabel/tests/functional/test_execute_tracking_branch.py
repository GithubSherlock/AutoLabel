"""execute_plan tracking 分支测试 —— monkeypatch TrackingTool，零模型零权重。

覆盖：tracking 步骤短路进 TrackingTool（不进 collect_images）、参数透传
（batch 显式优先 / mot→coco 映射 / 跟踪器选择）、steps_results 汇总、
ValueError 优雅跳过、取消（TrackingCancelledError）短路剩余步骤、
失败计数语义（#8：execute_plan 返回失败数）、非 tracking 步骤回归。
"""

from __future__ import annotations

from typing import Any

from auto2dlabel.agent.planner import _dict_to_plan
from auto2dlabel.cli_execute import execute_plan
from auto2dlabel.tests import Path
from auto2dlabel.tools.tracking import TrackingCancelledError


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
        progress_cb: Any = None,
    ) -> dict[str, Any]:
        self.last = {
            "source": source,
            "prompts": prompts,
            "confidence_threshold": confidence_threshold,
            "progress_cb": progress_cb,
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


def _two_tracking_plan() -> Any:
    """两步 tracking 计划（模型名作失败开关，见各测试的 fake）。"""
    return _dict_to_plan(
        {
            "steps": [
                {
                    "step_id": 1,
                    "task_type": "tracking",
                    "source": "/nonexistent/a.mp4",
                    "prompts": ["person"],
                    "model_name": "yolo12n.pt",
                    "export_format": "mot",
                },
                {
                    "step_id": 2,
                    "task_type": "tracking",
                    "source": "/nonexistent/b.mp4",
                    "prompts": ["car"],
                    "model_name": "yolo12n.pt",
                    "export_format": "mot",
                },
            ],
            "confirm_timeout": 30,
        }
    )


def _success_summary() -> dict[str, Any]:
    return {
        "frames": 5,
        "bboxes": 12,
        "track_ids": [0, 1],
        "mot_path": "outputs/x_mot.txt",
    }


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

    failures = execute_plan(
        _tracking_plan(),
        use_bot_sort=True,
        explicit_batch_size=2,
        output_dir="/tmp/out",
    )

    assert failures == 0  # #8 契约：全部成功 → 失败计数 0
    out = capsys.readouterr().out
    assert "未找到图像" not in out  # 未走 collect_images 路径
    assert "跟踪失败" not in out

    assert len(_FakeTool.instances) == 1
    tool = _FakeTool.instances[0]
    assert tool.kwargs["model_name"] == "yolo12n.pt"
    assert tool.kwargs["use_bot_sort"] is True
    assert tool.kwargs["batch_size"] == 2  # CLI 显式优先
    assert tool.kwargs["export_format"] == "coco"  # mot → 逐帧 coco 映射
    assert tool.last["source"] == "/nonexistent/video.mp4"
    assert tool.last["prompts"] == ["person", "car"]
    assert tool.last["confidence_threshold"] == 0.1
    assert callable(tool.last["progress_cb"])  # v1.0 P3 协议行默认挂接

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
    """TrackingTool.forward 抛 ValueError → 红字提示、不冒泡、本步跳过、
    计入失败计数（#8 契约：execute_plan 返回 1）。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("auto2dlabel.tools.log.log_chat_call", lambda **kw: None)

    class _BoomTool(_FakeTool):
        def forward(
            self,
            source: str,
            prompts: list[str],
            confidence_threshold: float = 0.1,
            progress_cb: Any = None,
        ) -> dict[str, Any]:
            raise ValueError("未找到可跟踪的帧/视频")

    monkeypatch.setattr("auto2dlabel.tools.tracking.TrackingTool", _BoomTool)

    failures = execute_plan(_tracking_plan())  # 不抛异常

    assert failures == 1  # #8：失败返回计数（chat 层据此非零退出）
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


def test_tracking_cancel_short_circuits_remaining_steps(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    """审查 #7：TrackingTool.forward 抛 TrackingCancelledError → 取消短路——
    不视为失败（返回 0）、后续步骤不执行（实例不再构造）、console 提示取消。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("auto2dlabel.tools.log.log_chat_call", lambda **kw: None)

    class _CancelTool(_FakeTool):
        def forward(
            self,
            source: str,
            prompts: list[str],
            confidence_threshold: float = 0.1,
            progress_cb: Any = None,
        ) -> dict[str, Any]:
            raise TrackingCancelledError("跟踪已取消")

    monkeypatch.setattr("auto2dlabel.tools.tracking.TrackingTool", _CancelTool)

    before = len(_FakeTool.instances)
    failures = execute_plan(_two_tracking_plan())
    assert failures == 0  # 取消不计数为失败
    assert len(_FakeTool.instances) == before + 1  # 第 2 步被短路，未构造
    out = capsys.readouterr().out
    assert "已取消" in out  # 步骤内提示 + 主循环「跳过剩余步骤」


def test_tracking_failure_counted_and_later_steps_run(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    """审查 #8：失败只跳过本步——后续步骤照常执行、成功步骤仍进 steps_results
    日志；execute_plan 返回失败步数 1。"""
    monkeypatch.chdir(tmp_path)
    logged: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "auto2dlabel.tools.log.log_chat_call",
        lambda **kw: logged.append(kw),
    )

    class _SelectiveFailTool(_FakeTool):
        def forward(
            self,
            source: str,
            prompts: list[str],
            confidence_threshold: float = 0.1,
            progress_cb: Any = None,
        ) -> dict[str, Any]:
            if self.kwargs.get("model_name") == "boom.pt":
                raise ValueError("磁盘余量不足（预检）")
            return _success_summary()

    monkeypatch.setattr("auto2dlabel.tools.tracking.TrackingTool", _SelectiveFailTool)

    plan = _two_tracking_plan()
    plan.steps[0].model_name = "boom.pt"  # 第 1 步失败、第 2 步成功
    before = len(_FakeTool.instances)

    failures = execute_plan(plan)

    assert failures == 1
    assert len(_FakeTool.instances) == before + 2  # 失败隔离：第 2 步照常执行
    out = capsys.readouterr().out
    assert "跟踪失败" in out
    assert "磁盘余量不足" in out
    # 成功步骤照常并入顶层日志（失败的步骤不入）
    assert len(logged) == 1
    assert len(logged[0]["steps_results"]) == 1
    assert logged[0]["steps_results"][0]["step_id"] == 2
