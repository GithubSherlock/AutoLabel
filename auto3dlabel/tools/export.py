"""3D 标注导出工具（v0.2 工具层）——仿 auto2dlabel/tools/export.py 骨架。

红线：Export 不暴露给 LLM——CLI 代码直调（cli.py chat 经本模块导 KITTI label），
build_3d_registry 从不注册本工具（2D 同纪律：仅 tools 命令显式 register）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from auto2dlabel.tools.base import Tool
from auto3dlabel.schema.box3d import Box3D


class ExportTool(Tool):
    """3D 标注导出：Box3D 列表 → KITTI label_2；NusBox 映射 → nuScenes 提交 JSON。"""

    name = "export_annotations"
    description = "导出 3D 标注为标准格式文件（kitti / nuscenes）"

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "annotations": {
                    "type": "array",
                    "description": (
                        "Box3D dict 列表（kitti）或 {sample_token: [NusBox dict]}（nuscenes）"
                    ),
                },
                "format": {
                    "type": "string",
                    "enum": ["kitti", "nuscenes"],
                    "default": "kitti",
                },
                "output_path": {"type": "string", "description": "输出文件路径"},
            },
            "required": ["annotations", "output_path"],
        }

    def forward(
        self,
        annotations: Any = None,
        output_path: str = "",
        format: str = "kitti",
        **kwargs: Any,
    ) -> str:
        """执行导出（全部具名参数带默认值——Tool 基类 forward 为 **kwargs 协议，
        orchestrator 以 forward(**arguments) 调用，必需性由 input_schema 声明）。"""
        if annotations is None or not output_path:
            raise ValueError("export_annotations 需要 annotations 与 output_path")
        if format == "kitti":
            return self._export_kitti(annotations, output_path)
        if format == "nuscenes":
            return self._export_nuscenes(annotations, output_path)
        raise ValueError(f"未知 3D 导出格式: {format}（可选 kitti / nuscenes）")

    @staticmethod
    def _as_box3d(item: Any) -> Box3D:
        """Box3D 对象直通；dict 走 from_dict（to_dict 中间格式往返）。"""
        return item if isinstance(item, Box3D) else Box3D.from_dict(item)

    def _export_kitti(self, annotations: Any, output_path: str) -> str:
        """Box3D 列表（或 to_dict dict 列表）→ label_2 文件。

        与 export/kitti_label.build_label_file 逐字节一致（conf 降序 + 行尾换行语义）。
        """
        from auto3dlabel.export.kitti_label import line_from_box3d

        boxes = [self._as_box3d(a) for a in annotations]
        lines = [line_from_box3d(b) for b in sorted(boxes, key=lambda b: -b.confidence)]
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return str(out.resolve())

    def _export_nuscenes(self, annotations: Any, output_path: str) -> str:
        """{sample_token: [NusBox|dict]} → 官方提交 JSON（自检不过 write_submission 抛
        ValueError）。"""
        from auto3dlabel.export.nuscenes_json import build_submission_json, write_submission
        from auto3dlabel.schema.nuscenes_box import NusBox

        results: dict[str, list[NusBox]] = {}
        for token, boxes in annotations.items():
            results[token] = [
                b if isinstance(b, NusBox) else NusBox.from_dict(b) for b in boxes
            ]
        sub = build_submission_json(results)
        return str(write_submission(sub, Path(output_path)).resolve())


def register(registry: Any = None) -> None:
    """向注册表注册本工具（缺省 2D 全局 registry；build_3d_registry 从不调用——红线）。"""
    from auto2dlabel.tools.registry import registry as reg

    # is not None 判断：ToolRegistry 定义 __len__，空注册表 falsy，`or` 会误落全局单例
    target = registry if registry is not None else reg
    target.register(ExportTool())
