"""条件 LLM 质量评估工具（3D）——EvaluateTool 本体零 2D 耦合，直接复用。

3D 使用契约（agent/orchestrator3d._wrap_evaluate_retry 实现）：
- detect_fn 注入 3D 重检闭包（Detect3DTool 降阈值重跑，Box3D dict → 2D Bbox 视图供
  auto2dlabel 质量评估循环消费）；swap_fn 注入备选模型闭包
- 与 2D 同红线：**绝不进 build_3d_registry**——orchestrator 持有实例直调，
  防跨图状态污染（auto2dlabel 设计不变量）
"""

from __future__ import annotations

from auto2dlabel.tools.evaluate import EvaluateTool

__all__ = ["EvaluateTool"]
