"""Agent 编排器 —— Agent Loop 的核心实现。

参考 Anthropic/OpenAI tool-use 协议：
1. 用户意图 → LLM 规划 → Tool Call → 观察结果 → 循环

这是 v0.1 最重要的模块。
"""

from __future__ import annotations

from typing import Any, cast

from auto2dlabel.agent import json, logging
from auto2dlabel.agent.evaluate import QualityReport, evaluate_detections
from auto2dlabel.agent.llm import LLMClient, create_client
from auto2dlabel.agent.state import AgentState
from auto2dlabel.schema.annotation import Bbox
from auto2dlabel.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def _summarize_tool_result(tool_name: str, result: Any) -> dict[str, Any]:
    """精简 tool 返回结果，避免把完整 Bbox 对象发给 LLM 浪费 token。"""
    if tool_name == "detect_objects" and isinstance(result, list):
        items = []
        for i, item in enumerate(result):
            if isinstance(item, Bbox):
                items.append({
                    "id": i,
                    "label": item.label,
                    "conf": round(item.confidence, 3),
                    "bbox": [
                        round(item.x, 0), round(item.y, 0),
                        round(item.width, 0), round(item.height, 0),
                    ],
                })
            elif isinstance(item, dict):
                items.append(item)
        return {"success": True, "count": len(items), "objects": items}
    if tool_name == "export_annotations":
        return {"success": True, "path": str(result)}
    if tool_name == "evaluate_quality" and isinstance(result, dict):
        out: dict[str, Any] = {"action": result.get("action")}
        if result.get("accepted"):
            out["accepted"] = True
        if result.get("flagged"):
            out["flagged"] = True
        if result.get("retried"):
            out["retried"] = True
            out["new_count"] = len(result.get("detections", []))
        return out
    return {"success": True, "data": str(result)[:200]}


class AgentOrchestrator:
    """Agent 编排器。

    管理 Agent Loop 的生命周期：接收用户指令 → 调用 LLM → 执行 Tool → 循环直到完成。

    Usage:
        orchestrator = AgentOrchestrator(llm_client)
        state = await orchestrator.run(
            image_path="photo.jpg",
            instruction="检测所有汽车和行人",
        )
        print(state.annotations)
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        tool_registry: ToolRegistry | None = None,
        max_iterations: int = 3,
        detection_model: str | None = None,
        iou_threshold: float = 0.5,
        use_sahi: bool = False,
    ):
        self.llm = llm_client
        self.registry = tool_registry or ToolRegistry.get_instance()
        self.max_iterations = max_iterations
        self.detection_model = detection_model
        self.iou_threshold = iou_threshold
        self.use_sahi = use_sahi
        self._detect_tool: Any | None = None  # 保留引用以读取重试状态
        # LLM Evaluate 节点状态（每次 run 重置，条件暴露）
        self._pending_evaluate: Any | None = None  # EvaluateTool 实例
        self._evaluate_called = False
        self._evaluate_retry_bboxes: list[Bbox] = []  # 重试检测结果，供 _sync_annotations 并入

    def _ensure_tools_registered(self) -> None:
        """确保默认 Tool 已注册。"""
        if len(self.registry) == 0:
            from auto2dlabel.models.detection import create_detection_model
            from auto2dlabel.tools.detection import DetectionTool

            model = create_detection_model(self.detection_model, iou_threshold=self.iou_threshold)
            self._detect_tool = DetectionTool(model=model, use_sahi=self.use_sahi)
            self.registry.register(self._detect_tool)
            # Export 不暴露给 LLM，由 CLI 直接调用

    def _ensure_llm(self) -> LLMClient:
        """确保 LLM 客户端可用。默认用 DeepSeek（如果配了 API key），否则 fallback 到 OpenAI。"""
        if self.llm is None:
            import os

            if os.environ.get("DEEPSEEK_API_KEY"):
                provider = "deepseek"
            elif os.environ.get("OPENAI_API_KEY"):
                provider = "openai"
            else:
                provider = "openai"
            self.llm = create_client(provider=provider)
        return self.llm

    def run(
        self,
        image_path: str,
        instruction: str,
        confidence_threshold: float = 0.3,
    ) -> AgentState:
        """运行 Agent Loop（同步版本）。

        Args:
            image_path: 待标注图像的路径。
            instruction: 用户的标注指令（自然语言）。
            confidence_threshold: 置信度阈值。

        Returns:
            AgentState 包含所有标注结果。
        """
        self._ensure_tools_registered()
        llm = self._ensure_llm()

        state = AgentState(
            image_path=image_path,
            user_instruction=instruction,
            confidence_threshold=confidence_threshold,
            max_iterations=self.max_iterations,
        )

        # 构建初始消息
        state.add_message("system", state.system_prompt)

        # 中→英关键词映射，确保 LLM 不遗漏（单一事实源 tools.prompts.CN_EN_MAP）
        from auto2dlabel.tools.prompts import CN_EN_MAP

        _hints = []
        for cn, en in CN_EN_MAP.items():
            if cn in instruction:
                _hints.append(f"{cn}={en}")
        hint_text = ("\nKeyword hints: " + ", ".join(_hints)) if _hints else ""

        state.add_message(
            "user",
            f"Image: {image_path}\n"
            f"Instruction: {instruction}\n"
            f"Threshold: {confidence_threshold}"
            f"{hint_text}",
        )

        logger.info("Agent Loop started: %s", instruction)

        _detect_called = False  # 防止重复调用检测
        # 重置 LLM Evaluate 节点状态（单图作用域）
        self._pending_evaluate = None
        self._evaluate_called = False
        self._evaluate_retry_bboxes = []

        while state.iteration < self.max_iterations and not state.done:
            state.iteration += 1
            logger.info("Iteration %d/%d", state.iteration, self.max_iterations)

            # 每轮重建 tools：质量未通过且尚未处置时，条件暴露 evaluate_quality
            tools = self.registry.to_openai_tools()
            if self._pending_evaluate is not None and not self._evaluate_called:
                tools.append(self._pending_evaluate.to_openai_tool())

            # Step 1: Call LLM
            response = llm.chat(state.messages, tools=tools, temperature=0.1)
            logger.info("LLM response: content=%s, tool_calls=%s",
                         response.content[:100] if response.content else None,
                         response.wants_tool_call)

            # 硬性收尾：检测已跑过且本轮无 tool_call，直接结束
            if _detect_called and not response.wants_tool_call:
                state.done = True
                if response.content:
                    state.add_message("assistant", response.content)
                logger.info("Agent finished after detection + summary")
                break

            # Step 2: Handle tool calls
            if response.wants_tool_call:
                assert response.tool_calls is not None

                # 过滤重复调用：detect_objects 一次 / evaluate_quality 一次
                for tc in list(response.tool_calls):
                    name = tc["function"]["name"]
                    if name == "detect_objects" and _detect_called:
                        logger.info("Skipping duplicate detect_objects call")
                        response.tool_calls.remove(tc)
                        state.add_tool_result(
                            tc["id"], "detect_objects",
                            {"skipped": True, "reason": "already called, use previous results"},
                        )
                    if name == "evaluate_quality" and self._evaluate_called:
                        logger.info("Skipping duplicate evaluate_quality call")
                        response.tool_calls.remove(tc)
                        state.add_tool_result(
                            tc["id"], "evaluate_quality",
                            {"skipped": True, "reason": "already called, only once per image"},
                        )

                if not response.tool_calls:
                    state.add_message("assistant", "检测已完成，请直接总结结果。")
                    continue
                state.add_message("assistant", response.content or "", response.tool_calls)

                for tc in response.tool_calls:
                    if tc["function"]["name"] == "detect_objects":
                        _detect_called = True
                    tool_name = tc["function"]["name"]
                    arguments = json.loads(tc["function"]["arguments"])

                    logger.info("Calling tool: %s(%s)", tool_name, arguments)

                    try:
                        # 只为需要图像的工具注入 image_path
                        _image_tools = {"detect_objects", "segment_mask", "visualize"}
                        if tool_name in _image_tools and "image_path" not in arguments:
                            arguments["image_path"] = image_path
                        # 清理其他工具多余的 image_path（LLM 可能误传）
                        if tool_name not in _image_tools:
                            arguments.pop("image_path", None)
                        if tool_name == "evaluate_quality":
                            # 不进全局 registry：直接调用条件暴露的实例；
                            # 同批重复调用也跳过（仅每图一次）
                            if self._evaluate_called or self._pending_evaluate is None:
                                state.add_tool_result(
                                    tc["id"], tool_name,
                                    {"skipped": True, "reason": "only once per image"},
                                )
                                continue
                            result = self._pending_evaluate.forward(**arguments)
                            self._evaluate_called = True
                        else:
                            result = self.registry.call(tool_name, **arguments)
                        # 精简返回给 LLM 的内容，避免浪费 token
                        summary = _summarize_tool_result(tool_name, result)
                        state.add_tool_result(tc["id"], tool_name, summary)

                        if tool_name == "evaluate_quality":
                            # 执行 LLM 选择的处置动作（代码执行，不追加迭代）
                            if result.get("flagged"):
                                state.metadata["llm_review_flagged"] = True
                            if result.get("retried"):
                                self._evaluate_retry_bboxes = list(result.get("detections", []))
                            state.add_message(
                                "user",
                                "处置完成。Summarize NOW. No tools. One line per class.",
                            )

                        # 检测完成后：代码级质量评估 + 注入收尾指令强制 LLM 停止
                        if tool_name == "detect_objects":
                            # 优先用保留的引用；注册表被外部预置时回退到注册实例
                            tool = self._detect_tool or self.registry.get("detect_objects")
                            quality = evaluate_detections(
                                result,
                                arguments.get("prompts", []),
                                confidence_threshold=arguments.get(
                                    "confidence_threshold", confidence_threshold
                                ),
                                image_path=image_path,
                                retried=bool(getattr(tool, "last_retried", False)),
                                retry_threshold=getattr(tool, "last_retry_threshold", None),
                            )
                            # 报告三通道：tool 摘要（LLM 可见）+ metadata + log
                            summary["quality"] = quality.to_dict()
                            state.metadata["quality_report"] = quality.to_dict()
                            if quality.warnings:
                                logger.warning(
                                    "质量警告 [%s]: %s", image_path, "; ".join(quality.warnings)
                                )
                            note = quality.summary_line()
                            if quality.ok:
                                state.add_message(
                                    "user",
                                    "Summarize NOW. No tools. One line per class."
                                    + (f" Quality: {note}" if note else ""),
                                )
                            else:
                                # 质量未通过：挂起 Evaluate 节点（下一轮条件暴露）
                                self._prepare_evaluate(
                                    image_path, arguments, quality, confidence_threshold
                                )
                                state.add_message(
                                    "user",
                                    f"质量评估未通过: {note}。请调用 evaluate_quality 工具"
                                    "选择处置动作（仅一次）: accept / flag_for_review / "
                                    "retry_lower_threshold。",
                                )
                    except Exception as e:
                        logger.error("Tool call failed: %s", e)
                        state.add_tool_result(
                            tc["id"], tool_name, {"success": False, "error": str(e)}
                        )

            elif response.content:
                state.add_message("assistant", response.content)
            else:
                # No tool calls & no content → Agent 认为任务完成
                state.done = True
                logger.info("Agent signaled completion (no tool calls)")

        # 将 tool call 结果汇总到 Annotation 对象
        self._sync_annotations(state)

        logger.info("Agent Loop finished after %d iterations", state.iteration)
        return state

    def _prepare_evaluate(
        self,
        image_path: str,
        arguments: dict[str, Any],
        quality: QualityReport,
        base_threshold: float,
    ) -> None:
        """质量未通过时挂起 EvaluateTool（下一轮条件暴露给 LLM）。

        detect_fn 复用 DetectionTool.forward（内部已含降阈值重试），
        每图仅允许一次 evaluate 处置（_evaluate_called 守卫）。
        """
        from auto2dlabel.tools.evaluate import EvaluateTool

        tool = self._detect_tool
        prompts = list(arguments.get("prompts", []))

        def _retry_detect(conf: float) -> list[Any]:
            # 复用 DetectionTool.forward（内部已含降阈值重试）
            assert tool is not None  # 挂起时必有检测工具
            return cast(list[Any], tool.forward(
                image_path=image_path, prompts=prompts, confidence_threshold=conf
            ))

        self._pending_evaluate = EvaluateTool(
            report=quality,
            retry_used=bool(getattr(tool, "last_retried", False)),
            base_threshold=base_threshold,
            detect_fn=_retry_detect if tool is not None else None,
        )

    def _sync_annotations(self, state: AgentState) -> None:
        """将 tool_call 记录同步回 Annotation 数据结构。"""
        from PIL import Image

        from auto2dlabel.schema.annotation import Annotation, Bbox

        ann = Annotation(image_path=state.image_path)
        found_any = False

        # 读取图像尺寸
        try:
            img = Image.open(state.image_path)
            ann.image_size = (img.width, img.height)
        except Exception:
            pass

        for tc in state.tool_calls:
            if tc["tool_name"] == "detect_objects":
                result = tc.get("result", {})
                if not isinstance(result, dict):
                    continue
                # 兼容两种格式：精简的 "objects" 和原始的 "data"
                items = result.get("objects") or result.get("data") or []
                for item in items:
                    if isinstance(item, Bbox):
                        ann.add_bbox(item)
                        found_any = True
                        if item.confidence < state.confidence_threshold:
                            assert item.id is not None  # add_bbox 已为 bbox 分配 id
                            ann.flag_for_review(item.id)
                    elif isinstance(item, dict):
                        # 兼容两种格式: {x, y, width, height} 或 {bbox: [x,y,w,h]}
                        if "bbox" in item:
                            bx, by, bw, bh = item["bbox"]
                        else:
                            bx, by = item.get("x", 0), item.get("y", 0)
                            bw, bh = item.get("width", 0), item.get("height", 0)
                        bbox = Bbox(
                            x=bx, y=by, width=bw, height=bh,
                            label=item.get("label", ""),
                            confidence=item.get("confidence", item.get("conf", 1.0)),
                        )
                        ann.add_bbox(bbox)
                        found_any = True
                        if bbox.confidence < state.confidence_threshold:
                            assert bbox.id is not None  # add_bbox 已为 bbox 分配 id
                            ann.flag_for_review(bbox.id)

        # LLM evaluate 重试的检测结果并入（追加，不替换原检测）
        for item in self._evaluate_retry_bboxes:
            ann.add_bbox(item)
            found_any = True
            if item.confidence < state.confidence_threshold:
                assert item.id is not None  # add_bbox 已为 bbox 分配 id
                ann.flag_for_review(item.id)

        if found_any:
            state.annotations.append(ann)

    def run_batch(
        self,
        image_paths: list[str],
        instruction: str,
        confidence_threshold: float = 0.3,
    ) -> list[AgentState]:
        """批量标注多张图像。

        Args:
            image_paths: 图像路径列表。
            instruction: 标注指令。
            confidence_threshold: 置信度阈值。

        Returns:
            AgentState 列表。
        """
        results = []
        for i, path in enumerate(image_paths):
            logger.info("Batch %d/%d: %s", i + 1, len(image_paths), path)
            state = self.run(path, instruction, confidence_threshold=confidence_threshold)
            results.append(state)
        return results
