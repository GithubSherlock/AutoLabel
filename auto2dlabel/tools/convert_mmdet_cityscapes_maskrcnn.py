"""一次性转换脚本：mmdet Cityscapes Mask R-CNN 权重 → torchvision v1 命名空间。

背景：cityscapes 检测步域边界（COCO 预训练模型对 30~50px 小目标失效，bbox mAP ≈ 0）
的解决方案——引入 mmdetection 官方 cityscapes 域内权重（Box AP 40.9 / Mask AP 36.4），
经本脚本一次性键名转换后由 torchvision 纯本地推理（不引入 mmdet/mmcv 依赖）。

用法:
    python3 -m auto2dlabel.tools.convert_mmdet_cityscapes_maskrcnn

输入: weights/maskrcnn_r50_cityscapes.mmdet.pth（open-mmlab 官方下载）
输出: weights/maskrcnn_r50_cityscapes.pth（torchvision 命名空间 state_dict）

结构对齐要点（torchvision 0.28 与 mmdet 2.7 已实测一致）:
- mask head: MaskRCNNHeads 默认即 4 层 conv + ReLU，与 mmdet FCNMaskHead(num_convs=4) 同构
- backbone/FPN/RPN/box head 逐键对应（见 convert_state_dict 注释）
- 背景通道 padding: mmdet fc_reg/mask_fcn_logits 不含背景 → 32→36、8→9 前置补零
- 背景位置重排: mmdet fc_cls 为 (9, 1024) 且背景在**最后一维**（mmdet 2.x 标签约定
  label=num_classes，softmax 后取 [:, :-1]）→ torchvision 背景在 index 0，
  需行重排 [fg×8, bg] → [bg, fg×8]（weight/bias 同规则，实测逐位验证）
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch


def _pad_channel(tensor: torch.Tensor, target_channels: int) -> torch.Tensor:
    """dim=0 前置补零到目标通道数（背景通道）。"""
    pad = target_channels - tensor.size(0)
    if pad <= 0:
        return tensor
    shape = list(tensor.shape)
    shape[0] = pad
    return torch.cat([tensor.new_zeros(shape), tensor], dim=0)


def convert_state_dict(
    mmdet_state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """mmdet 2.x 键名 → torchvision v1 键名。返回 (转换结果, 未映射键清单)。"""
    out: dict[str, torch.Tensor] = {}
    unmapped: list[str] = []

    for key, tensor in mmdet_state.items():
        new_key: str | None = None
        # backbone.* → backbone.body.*
        if key.startswith("backbone."):
            new_key = "backbone.body." + key[len("backbone."):]
        # neck 横向 1x1 / 上采样 3x3 卷积 → FPN
        elif (m := re.match(r"neck\.lateral_convs\.(\d)\.conv\.(.+)", key)):
            new_key = f"backbone.fpn.inner_blocks.{m.group(1)}.0.{m.group(2)}"
        elif (m := re.match(r"neck\.fpn_convs\.(\d)\.conv\.(.+)", key)):
            new_key = f"backbone.fpn.layer_blocks.{m.group(1)}.0.{m.group(2)}"
        # RPN（rpn_conv → rpn.head.conv.0.0，Conv2dNormActivation 嵌套）
        elif key.startswith("rpn_head.rpn_conv."):
            new_key = "rpn.head.conv.0.0." + key[len("rpn_head.rpn_conv."):]
        elif key.startswith("rpn_head.rpn_cls."):
            new_key = "rpn.head.cls_logits." + key[len("rpn_head.rpn_cls."):]
        elif key.startswith("rpn_head.rpn_reg."):
            new_key = "rpn.head.bbox_pred." + key[len("rpn_head.rpn_reg."):]
        # box head（shared_fcs.0/1 → fc6/fc7；fc_cls/fc_reg → box_predictor）
        elif key.startswith("roi_head.bbox_head.shared_fcs."):
            idx = key.split(".")[3]
            rest = ".".join(key.split(".")[4:])
            fc_name = "fc6" if idx == "0" else "fc7"
            new_key = f"roi_heads.box_head.{fc_name}.{rest}"
        elif key.startswith("roi_head.bbox_head.fc_cls."):
            new_key = (
                "roi_heads.box_predictor.cls_score."
                + key[len("roi_head.bbox_head.fc_cls."):]
            )
            # mmdet 背景在最后一维 → torchvision 背景在 index 0（行重排 [fg..., bg] → [bg, fg...]）
            tensor = torch.cat([tensor[-1:], tensor[:-1]], dim=0)
        elif key.startswith("roi_head.bbox_head.fc_reg."):
            new_key = (
                "roi_heads.box_predictor.bbox_pred."
                + key[len("roi_head.bbox_head.fc_reg."):]
            )
            tensor = _pad_channel(tensor, 36)  # 8 类 ×4 无背景 → 9 类 ×4
        # mask head（convs.{i}.conv → mask_head.{i}.0，upsample → deconv）
        elif (m := re.match(r"roi_head\.mask_head\.convs\.(\d)\.conv\.(.+)", key)):
            new_key = f"roi_heads.mask_head.{m.group(1)}.0.{m.group(2)}"
        elif key.startswith("roi_head.mask_head.upsample."):
            new_key = (
                "roi_heads.mask_predictor.conv5_mask."
                + key[len("roi_head.mask_head.upsample."):]
            )
        elif key.startswith("roi_head.mask_head.conv_logits."):
            new_key = (
                "roi_heads.mask_predictor.mask_fcn_logits."
                + key[len("roi_head.mask_head.conv_logits."):]
            )
            tensor = _pad_channel(tensor, 9)  # 8 类无背景 → 9 类（通道 0 背景补零）
        else:
            unmapped.append(key)
            continue

        out[new_key] = tensor
    return out, unmapped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="mmdet checkpoint 路径（默认 weights/maskrcnn_r50_cityscapes.mmdet.pth）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出路径（默认 weights/maskrcnn_r50_cityscapes.pth）",
    )
    args = parser.parse_args()

    from auto2dlabel.models.model_catalog import WEIGHTS_DIR

    src = args.input or WEIGHTS_DIR / "maskrcnn_r50_cityscapes.mmdet.pth"
    dst = args.output or WEIGHTS_DIR / "maskrcnn_r50_cityscapes.pth"
    if not src.exists():
        raise SystemExit(
            f"输入权重不存在: {src}。请先下载: "
            "https://download.openmmlab.com/mmdetection/v2.0/cityscapes/"
            "mask_rcnn_r50_fpn_1x_cityscapes/mask_rcnn_r50_fpn_1x_cityscapes_"
            "20201211_133733-d2858245.pth"
        )

    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    mmdet_state = ckpt.get("state_dict", ckpt)
    converted, unmapped = convert_state_dict(mmdet_state)
    if unmapped:
        print(f"未映射键 {len(unmapped)} 个，转换不完整:")
        for k in unmapped:
            print("  ", k)
        raise SystemExit(1)

    # 严格校验：构建目标模型，比对键集合 + 逐键加载
    from auto2dlabel.models.segmentation import build_maskrcnn_cityscapes

    model = build_maskrcnn_cityscapes("cpu")
    missing, unexpected = model.load_state_dict(converted, strict=False)
    if missing or unexpected:
        print("键集合不匹配（strict 加载将失败）:")
        for k in missing:
            print("  missing:  ", k)
        for k in unexpected:
            print("  unexpected:", k)
        raise SystemExit(1)

    torch.save(converted, dst)
    print(f"转换完成: {len(converted)} 键（输入 {len(mmdet_state)} 键）→ {dst}")


if __name__ == "__main__":
    main()
