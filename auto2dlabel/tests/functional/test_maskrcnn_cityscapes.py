"""cityscapes 域内 Mask R-CNN 接入测试 —— 零权重加载、零下载。

覆盖：工厂分发（maskrcnn_r50_cityscapes → MaskRCNNModel 且 model_name 透传；
COCO v2 路径不回归）、mmdet 权重转换纯函数（fc_cls 背景行重排 / fc_reg 与
mask_fcn_logits 背景通道 padding / 键名映射）、cityscapes 类别映射常量、
GT instanceIds 解析（class_id = inst_id // 1000 回归防护）、CLI self-detect 路由。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from auto2dlabel.cli_execute import _is_self_detect_seg
from auto2dlabel.models.segmentation import (
    CITYSCAPES_THING_NAMES,
    MaskRCNNModel,
    create_segmentation_model,
)
from auto2dlabel.tests import np
from auto2dlabel.tools.convert_mmdet_cityscapes_maskrcnn import (
    _pad_channel,
    convert_state_dict,
)

# ============================================================
# 工厂分发
# ============================================================

class TestDispatch:
    def test_cityscapes_name_routes_to_maskrcnn(self) -> None:
        """maskrcnn_r50_cityscapes → MaskRCNNModel 且 model_name 透传。"""
        model = create_segmentation_model("maskrcnn_r50_cityscapes")
        assert isinstance(model, MaskRCNNModel)
        assert model._model_name == "maskrcnn_r50_cityscapes"

    def test_coco_v2_path_unchanged(self) -> None:
        """COCO 路径不回归：v2 名字仍走 MaskRCNNModel + COCO 默认名。"""
        model = create_segmentation_model("maskrcnn_resnet50_fpn_v2")
        assert isinstance(model, MaskRCNNModel)
        assert model._model_name == "maskrcnn_resnet50_fpn_v2"

    def test_lazy_loading(self) -> None:
        """构造零加载（不 import torchvision / 不读权重文件）。"""
        model = MaskRCNNModel(model_name="maskrcnn_r50_cityscapes")
        assert model._model is None

    def test_self_detect_routing(self) -> None:
        """CLI self-detect 路由识别 cityscapes 模型（自带检测，跳过检测步）。"""
        assert _is_self_detect_seg("maskrcnn_r50_cityscapes")
        assert _is_self_detect_seg("maskrcnn_resnet50_fpn_v2")


# ============================================================
# cityscapes 类别映射（mmdet 权重下标权威顺序）
# ============================================================

class TestCityscapesClassNames:
    def test_order(self) -> None:
        """下标顺序与 mmdet CLASSES meta 一致（person..bicycle，8 类）。"""
        assert CITYSCAPES_THING_NAMES == [
            "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
        ]

    def test_label_idx_mapping(self) -> None:
        """torchvision 1-based label → 0-based 下标 → 名字（generate 映射规则）。"""
        for label_idx, name in enumerate(CITYSCAPES_THING_NAMES, start=1):
            assert CITYSCAPES_THING_NAMES[label_idx - 1] == name


# ============================================================
# mmdet → torchvision 转换纯函数
# ============================================================

class TestConvertStateDict:
    def test_fc_cls_background_row_reorder(self) -> None:
        """mmdet fc_cls 背景在最后一维 → torchvision 背景在 index 0（行重排）。"""
        # 9 行：8 前景（行值 0..7）+ 背景（行值 8）
        weight = torch.arange(9, dtype=torch.float32).reshape(9, 1)
        state = {"roi_head.bbox_head.fc_cls.weight": weight}
        out, unmapped = convert_state_dict(state)
        assert not unmapped
        converted = out["roi_heads.box_predictor.cls_score.weight"]
        assert converted.shape == (9, 1)
        assert converted[0, 0] == 8.0  # 背景行移到最前
        for i in range(8):
            assert converted[i + 1, 0] == float(i)

    def test_fc_cls_bias_reorder(self) -> None:
        """bias 与 weight 同规则重排（背景 bias 移到 index 0）。"""
        bias = torch.arange(9, dtype=torch.float32)
        state = {"roi_head.bbox_head.fc_cls.bias": bias}
        out, _ = convert_state_dict(state)
        converted = out["roi_heads.box_predictor.cls_score.bias"]
        assert converted[0] == 8.0
        assert converted[1] == 0.0
        assert converted[8] == 7.0

    def test_fc_reg_background_padding(self) -> None:
        """fc_reg 8 类×4 无背景 → 前置补零 4 行（背景坐标），后 32 行原样。"""
        weight = torch.ones(32, 1024)
        state = {"roi_head.bbox_head.fc_reg.weight": weight}
        out, _ = convert_state_dict(state)
        converted = out["roi_heads.box_predictor.bbox_pred.weight"]
        assert converted.shape == (36, 1024)
        assert (converted[:4] == 0).all()
        assert torch.equal(converted[4:], weight)

    def test_mask_fcn_logits_background_padding(self) -> None:
        """mask_fcn_logits 8 通道无背景 → 前置补零 1 通道。"""
        weight = torch.ones(8, 256, 1, 1)
        state = {"roi_head.mask_head.conv_logits.weight": weight}
        out, _ = convert_state_dict(state)
        converted = out["roi_heads.mask_predictor.mask_fcn_logits.weight"]
        assert converted.shape == (9, 256, 1, 1)
        assert (converted[:1] == 0).all()
        assert torch.equal(converted[1:], weight)

    def test_key_remapping(self) -> None:
        """代表键映射：backbone/FPN/RPN/box head/mask head 命名空间转换。"""
        state = {
            "backbone.conv1.weight": torch.zeros(1),
            "neck.lateral_convs.0.conv.weight": torch.zeros(1),
            "neck.fpn_convs.3.conv.bias": torch.zeros(1),
            "rpn_head.rpn_conv.weight": torch.zeros(1),
            "rpn_head.rpn_cls.weight": torch.zeros(1),
            "rpn_head.rpn_reg.weight": torch.zeros(1),
            "roi_head.bbox_head.shared_fcs.0.weight": torch.zeros(1),
            "roi_head.bbox_head.shared_fcs.1.bias": torch.zeros(1),
            "roi_head.mask_head.convs.2.conv.weight": torch.zeros(1),
            "roi_head.mask_head.upsample.weight": torch.zeros(1),
        }
        out, unmapped = convert_state_dict(state)
        assert not unmapped
        assert "backbone.body.conv1.weight" in out
        assert "backbone.fpn.inner_blocks.0.0.weight" in out
        assert "backbone.fpn.layer_blocks.3.0.bias" in out
        assert "rpn.head.conv.0.0.weight" in out
        assert "rpn.head.cls_logits.weight" in out
        assert "rpn.head.bbox_pred.weight" in out
        assert "roi_heads.box_head.fc6.weight" in out
        assert "roi_heads.box_head.fc7.bias" in out
        assert "roi_heads.mask_head.2.0.weight" in out
        assert "roi_heads.mask_predictor.conv5_mask.weight" in out

    def test_unmapped_keys_reported(self) -> None:
        """未知键进未映射清单（转换完整性自检）。"""
        _, unmapped = convert_state_dict({"optimizer.state": torch.zeros(1)})
        assert unmapped == ["optimizer.state"]

    def test_pad_channel_noop_when_already_target(self) -> None:
        """已达目标通道数时 no-op（fc_reg 36 行输入不被二次 padding）。"""
        t = torch.ones(36, 4)
        assert torch.equal(_pad_channel(t, 36), t)


# ============================================================
# GT instanceIds 解析回归（class_id = inst_id // 1000）
# ============================================================

class TestCityscapesGroundTruthParsing:
    """instanceIds.png 编码: pixel_value = label_id * 1000 + instance_idx。

    回归防护：曾误用 `inst_id % 1000` 对 thing 类像素（24000+）得 0 → 全部跳过，
    GT 近乎空（详见 cityscapes_benchmark.load_cityscapes_ground_truth）。
    """

    def _make_scene(self, tmp_path: Path, instance_ids: np.ndarray[Any, Any]) -> tuple[Path, Path]:
        from PIL import Image

        gt_dir = tmp_path / "gtFine" / "val"
        img_dir = tmp_path / "leftImg8bit" / "val"
        (gt_dir).mkdir(parents=True)
        (img_dir / "testcity").mkdir(parents=True)
        gt_path = gt_dir / "testcity_000000_000001_gtFine_instanceIds.png"
        Image.fromarray(instance_ids.astype(np.uint16)).save(gt_path)
        # 对应原图（GT 加载只检查存在性，内容无关）
        (img_dir / "testcity" / "testcity_000000_000001_leftImg8bit.png").write_bytes(b"\x00")
        return gt_dir, img_dir

    def test_thing_classes_parsed(self, tmp_path: Path) -> None:
        """24000(person)/26005(car)/31000(train) 解析为实例；stuff(12) 跳过。"""
        ids = np.zeros((4, 4), dtype=np.uint16)
        ids[0, 0] = 24000   # person 实例 0
        ids[1, 1] = 26005   # car 实例 5
        ids[2, 2] = 31000   # train 实例 0
        ids[3, 3] = 12      # stuff（无实例语义）→ 跳过
        gt_dir, img_dir = self._make_scene(tmp_path, ids)

        from auto2dlabel.benchmarks.cityscapes_benchmark import (
            load_cityscapes_ground_truth,
        )

        gt = load_cityscapes_ground_truth(gt_dir, img_dir)
        assert len(gt) == 1
        objects = gt[0]["objects"]
        names = sorted(o["name"] for o in objects)
        assert names == ["car", "person", "train"]
        # mask 内容正确（单像素实例）
        by_name = {o["name"]: o["mask"] for o in objects}
        assert by_name["person"].shape == (4, 4)
        assert by_name["person"][0, 0] and by_name["person"].sum() == 1
        assert by_name["car"][1, 1] and by_name["car"].sum() == 1
        assert by_name["train"][2, 2] and by_name["train"].sum() == 1

    def test_rider_folds_to_person(self, tmp_path: Path) -> None:
        """25(rider) 折叠为 person（THING_CLASSES 映射）。"""
        ids = np.zeros((2, 2), dtype=np.uint16)
        ids[0, 0] = 25000  # rider 实例 0
        gt_dir, img_dir = self._make_scene(tmp_path, ids)

        from auto2dlabel.benchmarks.cityscapes_benchmark import (
            load_cityscapes_ground_truth,
        )

        gt = load_cityscapes_ground_truth(gt_dir, img_dir)
        assert [o["name"] for o in gt[0]["objects"]] == ["person"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
