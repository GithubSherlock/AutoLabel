"""Agent 状态定义。

AgentState 贯穿整个 Agent Loop 生命周期，记录当前标注会话的完整状态。
支持 JSON 序列化，为未来 checkpoint/resume 预留基础。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from auto2dlabel.schema.annotation import Annotation


@dataclass
class AgentState:
    """Agent 标注会话的完整状态。

    每次 Agent Loop 迭代都会修改此状态。
    """

    # === 输入 ===
    image_path: str = ""
    user_instruction: str = ""
    confidence_threshold: float = 0.3

    # === 累积的标注结果 ===
    annotations: list[Annotation] = field(default_factory=list)

    # === 对话历史（Message 格式，兼容 OpenAI/Anthropic API）===
    messages: list[dict[str, Any]] = field(default_factory=list)

    # === Tool 调用记录 ===
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    # === 任务控制 ===
    max_iterations: int = 10  # 最大 Agent Loop 迭代次数
    iteration: int = 0  # 当前迭代次数
    done: bool = False  # Agent 是否完成

    # === 元数据 ===
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def current_annotation(self) -> Annotation | None:
        """获取当前正在处理的 Annotation（最后一个）。"""
        if self.annotations:
            return self.annotations[-1]
        return None

    @property
    def system_prompt(self) -> str:
        """生成 Agent 的 system prompt。"""
        return (
            "You are an annotation agent. Call detect_objects ONCE. "
            "CRITICAL: Extract ALL objects from user instruction and pass them ALL "
            "to prompts (e.g. '汽车 行人 自行车' → prompts=['car','person','bicycle']). "
            "Do NOT skip any object. "
            "After results, output count per class in one line. STOP. No questions."
        )

    def add_message(self, role: str, content: str, tool_calls: list | None = None) -> None:
        """向对话历史添加一条消息。"""
        msg: dict[str, Any] = {"role": role, "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.messages.append(msg)

    def add_tool_result(self, tool_call_id: str, tool_name: str, result: dict) -> None:
        """向对话历史添加 tool 调用结果。"""
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": str(result),
        })
        self.tool_calls.append({
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "result": result,
        })

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict（用于 checkpoint）。"""
        return {
            "image_path": self.image_path,
            "user_instruction": self.user_instruction,
            "confidence_threshold": self.confidence_threshold,
            "annotations": [a.to_dict() for a in self.annotations],
            "messages": self.messages,
            "tool_calls": self.tool_calls,
            "max_iterations": self.max_iterations,
            "iteration": self.iteration,
            "done": self.done,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentState:
        """从 dict 恢复状态（用于 checkpoint resume）。"""
        state = cls(
            image_path=data.get("image_path", ""),
            user_instruction=data.get("user_instruction", ""),
            confidence_threshold=data.get("confidence_threshold", 0.3),
        )
        state.messages = data.get("messages", [])
        state.tool_calls = data.get("tool_calls", [])
        state.max_iterations = int(data.get("max_iterations", 10))
        state.iteration = data.get("iteration", 0)
        state.done = data.get("done", False)
        state.metadata = data.get("metadata", {})

        # 恢复 annotations
        state.annotations = [
            _annotation_from_dict(a) for a in data.get("annotations", [])
        ]
        return state


def _annotation_from_dict(d: dict[str, Any]) -> Annotation:
    """从 dict 重建 Annotation 对象。"""
    from auto2dlabel.schema.annotation import Bbox, ImageLabel, Mask

    bboxes = [
        Bbox(
            x=b["x"], y=b["y"],
            width=b["width"], height=b["height"],
            label=b.get("label", ""),
            confidence=b.get("confidence", 1.0),
            angle=b.get("angle", 0.0),
            track_id=b.get("track_id"),
            keypoints=[
                (float(k[0]), float(k[1]), float(k[2]))
                for k in b.get("keypoints", [])
            ],
        )
        for b in d.get("bboxes", [])
    ]
    masks = [
        Mask(
            bbox=Bbox(
                x=m["bbox"]["x"], y=m["bbox"]["y"],
                width=m["bbox"]["width"], height=m["bbox"]["height"],
                label=m["bbox"].get("label", ""),
                confidence=m["bbox"].get("confidence", 1.0),
                angle=m["bbox"].get("angle", 0.0),
                track_id=m["bbox"].get("track_id"),
                keypoints=[
                    (float(k[0]), float(k[1]), float(k[2]))
                    for k in m["bbox"].get("keypoints", [])
                ],
            ),
            segmentation=m.get("segmentation", []),
            area=m.get("area", 0.0),
        )
        for m in d.get("masks", [])
    ]
    labels = [
        ImageLabel(label=lab.get("label", ""), score=lab.get("score", 0.0))
        for lab in d.get("labels", [])
    ]
    return Annotation(
        image_path=d["image_path"],
        image_size=cast(tuple[int, int], tuple(d.get("image_size", (0, 0)))),
        bboxes=bboxes,
        masks=masks,
        labels=labels,
        review_flags=d.get("review_flags", []),
        metadata=d.get("metadata", {}),
    )
