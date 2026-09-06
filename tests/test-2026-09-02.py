"""145 条 autolabel CLI 命令可靠性实测（2026-09-02/06，GPU 3080 Ti）。

50 条 2D + 95 条 3D（50 基础/边界 + 45 点云线），四档难度：
- 简单：标准句式、单类
- 中等：多类、阈值、模型/引擎指定、分割/分类/OBB
- 困难：复杂句式、转述、泛指、多步骤、视频跟踪、3D 引擎协议
- 边界：不存在图片/帧、非法参数、超范围阈值、路由歧义、协议不兼容
- 点云线（101-145）：PointPillars/PV-RCNN 真实检出（KITTI velodyne）+
  BEV 点云预测图产物（expect 后置检查：非空 label + {frame}_bev.png）

每条真实执行 `autolabel`（子进程，选项前置——typer 位置参数后选项会报
No such command，见 /tmp/kitti3d_tests/FINDINGS.md F3）。判定：

- PASS：进程在超时内退出、exit 0、输出无 Traceback，且 expect 产物检查全过
  （业务性 0 框/黄字降级不算失败——那是模型能力或 HITL 设计意图）
- FAIL：崩溃（traceback）/ 非零 exit / 超时挂起 / expect 产物缺失 —— 即本轮要发现的漏洞

expect 语义：每条 (glob, min_bytes) 须命中至少一个 ≥ min_bytes 的文件
（相对路径以项目根为基准）；argv_prefix 允许换入口（auto3dlabel run 批量路径）。
结果与日志落 /tmp/autolabel_100/（results/*.json + logs/*.log）。
运行：python -m pytest tests/test-2026-09-02.py -q（约 60-90 分钟，串行）。
"""

from __future__ import annotations

import glob
import json
import subprocess
import time
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

PROJ = Path(__file__).resolve().parents[1]
RESULT_DIR = Path("/tmp/autolabel_100")
LOG_DIR = RESULT_DIR / "logs"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

VIDEO = "/root/autodl-tmp/Documents/videos/sportscheck_2min.mp4"
COCO_IMG = "/root/autodl-tmp/Documents/datasets/COCO2017/val2017/000000000139.jpg"

# 点云线（101-145）输出目录（独立于 outputs/，避免与基线用例互相覆盖）
OUT_PP = "/tmp/autolabel_100/out_pp"
OUT_PP_BATCH = "/tmp/autolabel_100/out_pp_batch"
PP_CHAT = ("--no-wait", "-d", "pointpillars_kitti", "-o", OUT_PP)


def _pp_expect(fid: str) -> tuple[tuple[str, int], ...]:
    """点云线 chat 用例 expect：非空 label（真实检出）+ BEV 点云预测图。"""
    return ((f"{OUT_PP}/labels/{fid}.txt", 60), (f"{OUT_PP}/{fid}_bev.png", 500))


def _pp_bev_only(fid: str) -> tuple[tuple[str, int], ...]:
    """只查 BEV 产物（阈值/类别过滤场景允许 0 框——业务性，label 可能为空）。"""
    return ((f"{OUT_PP}/{fid}_bev.png", 500),)


def _batch_expect(fids: tuple[str, ...]) -> tuple[tuple[str, int], ...]:
    """auto3dlabel run 批量用例 expect：各帧非空 label。"""
    return tuple((f"{OUT_PP_BATCH}/labels/{f}.txt", 60) for f in fids)


@dataclass(frozen=True)
class Case:
    id: int
    domain: str
    instruction: str
    opts: tuple[str, ...] = ("--no-wait",)
    timeout: int = 600
    note: str = ""
    argv_prefix: tuple[str, ...] = ("autolabel",)
    expect: tuple[tuple[str, int], ...] = ()


def C(
    i: int, domain: str, instruction: str, note: str = "", opts: tuple[str, ...] = ("--no-wait",),
    timeout: int = 600, argv_prefix: tuple[str, ...] = ("autolabel",),
    expect: tuple[tuple[str, int], ...] = ()
) -> Case:
    return Case(i, domain, instruction, opts, timeout, note, argv_prefix, expect)


def _expect_misses(expect: tuple[tuple[str, int], ...]) -> str:
    """→ 未满足的 expect 清单文本（空串 = 全满足）；相对 pattern 以 PROJ 为基准。"""
    miss: list[str] = []
    for pattern, min_bytes in expect:
        pat = pattern if Path(pattern).is_absolute() else str(PROJ / pattern)
        hit = any(
            Path(p).is_file() and Path(p).stat().st_size >= min_bytes
            for p in glob.glob(pat)
        )
        if not hit:
            miss.append(f"{pattern}≥{min_bytes}B")
    return ", ".join(miss)


CASES: list[Case] = [
    # ================= 2D 简单（1-10）=================
    C(1, "2d", "检测 000860.png 中的汽车", "标准句式单类"),
    C(2, "2d", "检测 000860.png 中的行人", "标准句式单类"),
    C(3, "2d", "检测 000860.png 中的汽车和行人", "双类"),
    C(4, "2d", "用 yolo12n.pt 检测 000860.png 中的汽车", "指定 nano 模型"),
    C(5, "2d", "用 yolo26x.pt 检测 000860.png 中的汽车和行人", "指定高精度模型"),
    C(6, "2d", f"检测 {COCO_IMG} 中的行人", "绝对路径单图"),
    C(7, "2d", f"用 fasterrcnn 检测 {COCO_IMG} 中的汽车和行人", "torchvision 引擎"),
    C(8, "2d", "检测 000860.png 中的公交车", "bus 中文名"),
    C(9, "2d", "检测 000860.png 中的汽车，置信度 0.4", "指定阈值"),
    C(10, "2d", "检测 000860.png 中的 car 和 person", "中英混排类别"),
    # ================= 2D 中等（11-20）=================
    C(11, "2d", "分割 000860.png 中的汽车", "默认分割模型"),
    C(12, "2d", "用 sam_b.pt 分割 000860.png 中的行人", "SAM 两段式"),
    C(13, "2d", "用 maskrcnn 检测并分割 000860.png 中的行人", "Mask R-CNN 一步"),
    C(14, "2d", "用 maskrcnn_r50_cityscapes 分割 000860.png 中的汽车", "cityscapes 域内"),
    C(15, "2d", "语义分割 000860.png", "语义分割无类别"),
    C(16, "2d", "分类 000860.png 为室内场景和室外场景", "CLIP 中文候选"),
    C(17, "2d", "用 clip 分类 000860.png 为猫和狗", "CLIP 指定"),
    C(18, "2d", "用 resnet50 分类 000860.png 为 car 和 person", "torchvision 英文候选"),
    C(19, "2d", "检测 000860.png 中的旋转框", "OBB 默认模型"),
    C(20, "2d", "检测 000860.png 中的汽车，阈值 0.8", "高阈值→0 框→降阈值重试"),
    # ================= 2D 困难（21-32）=================
    C(21, "2d", "用 yolo11n-obb.pt 检测 000860.png 中的汽车旋转框", "OBB 指定+DOTA 类映射"),
    C(22, "2d", "检测 000860.png 中的汽车，置信度 0.15，iou 0.6", "多参数组合"),
    C(23, "2d", "用 grounding-dino-tiny 检测 000860.png 中的红色汽车", "开放词汇+属性"),
    C(24, "2d", "用 rtdetr-x.pt 检测 000860.png 中的汽车和行人", "RT-DETR 引擎"),
    C(25, "2d", "检测 000860.png 中的交通工具", "泛指类别"),
    C(26, "2d", "帮我看看 000860.png 里有什么值得标注的目标", "类别推荐模式"),
    C(27, "2d", "在 000860.png 上检测汽车；然后分割行人", "多步骤分号"),
    C(28, "2d", "标注 000860.png 中的行人姿态关键点", "pose 任务"),
    C(29, "2d", "用 yolo11s 检测 000860.png 中的汽车和卡车", "yolo11s 引擎"),
    C(30, "2d", "检测 KITTI 数据集 image_2 目录中 000049.png 的汽车", "KITTI 2D 单图+目录名解析"),
    C(31, "2d", f"跟踪 {VIDEO} 中的行人", "视频跟踪 ByteTrack", timeout=900),
    C(32, "2d", f"跟踪 {VIDEO} 中穿红衣服的人", "属性指代 L1", timeout=900),
    # ================= 2D 边界/异常（33-50）=================
    C(33, "2d", "检测 not_exist_xxx.png 中的汽车", "不存在图片"),
    C(34, "2d", "检测 000860.png 中的独角兽", "不在类别表"),
    C(35, "2d", "检测 000860.png 中的汽车，置信度 -0.5", "负阈值"),
    C(36, "2d", "检测 000860.png 中的汽车，置信度 1.5", "超 1 阈值"),
    C(37, "2d", "检测 000860.png 中的汽车和汽车和汽车和汽车", "重复类别"),
    C(38, "2d", "detect cars in 000860.png", "纯英文指令"),
    C(39, "2d", "检测 000860.png 中的 aeroplane、train 和 cow", "COCO 全类但图内无"),
    C(40, "2d", "检测 000860.png 中的汽车，用不存在的模型 not_a_model.pt", "不存在模型名"),
    C(41, "2d", "检测 000860.png 中的汽车，用 kitti 微调权重", "2D 微调权重 5 类"),
    C(42, "2d", "检测 000860.png 和 000000000139.jpg 中的汽车", "多文件探测"),
    C(43, "2d", "检测 000860.png 中的汽车和行人", "SAHI 切片", opts=("--no-wait", "--sahi")),
    C(44, "2d", "分类 000860.png 为 dog、cat、bird、airplane、automobile", "5 候选含 ImageNet 无类"),
    C(45, "2d", "检测 000860.png 中的小汽车和大卡车", "同义词变体"),
    C(46, "2d", "用 yolo26x-obb.pt 检测 000860.png 中的船舶旋转框", "OBB ship 类"),
    C(47, "2d", f"用 fasterrcnn 检测 {COCO_IMG} 中的 bus 和 traffic light", "英文类别 COCO 图"),
    C(48, "2d", "分类 000860.png 为汽车、行人和自行车", "3 候选中文"),
    C(49, "2d", "检测 000860.png 中的汽车，置信度 0.05", "极低阈值"),
    C(50, "2d", "检测 000860.png 中的汽车，用 yolo11n.pt", "nano 快速"),
    # ================= 3D 基础（51-70）=================
    C(51, "3d", "标注 KITTI 帧 000049 中的汽车和行人", "双类默认引擎"),
    C(52, "3d", "标注 KITTI 帧 000010 中的汽车，用 kitti 微调权重", "单类+微调"),
    C(53, "3d", "用 kitti 微调权重标注 KITTI 帧 000008 中的 car", "英文类名+微调"),
    C(54, "3d", "标注 KITTI 帧 000067 中的汽车和卡车，置信度 0.4", "双类+conf"),
    C(55, "3d", "标注 KITTI 帧 000021 中的车辆，阈值设 0.5", "泛指车辆"),
    C(56, "3d", "用 yolo26x 权重标注 KITTI 帧 000025 中的汽车和卡车", "COCO 引擎"),
    C(57, "3d", "请帮我标注 KITTI 帧 000038 中的汽车", "请求句式"),
    C(58, "3d", "标注 KITTI 帧 000045 中的汽车，用 kitti 微调权重，阈值 0.2", "微调+低阈值"),
    C(59, "3d", "把 KITTI 帧 000050 里的汽车都标出来", "动词变体"),
    C(60, "3d", "KITTI 帧 000036，请标注其中的汽车和行人", "帧前置语序"),
    C(61, "3d", "标注 KITTI 帧 000011 中的行人和汽车", "行人主位"),
    C(62, "3d", "标注 KITTI 帧 000043 中的行人，置信度 0.5", "单行人+conf"),
    C(63, "3d", "标注 KITTI 帧 000048 中的行人和自行车", "双类混合"),
    C(64, "3d", "标注 KITTI 帧 000073 中的自行车和行人", "自行车主位"),
    C(65, "3d", "标注 KITTI 帧 000076 中的行人和骑自行车的人", "转述式类别"),
    C(66, "3d", "标注 KITTI 帧 000134 中的 bicycle、person 和 car", "三英文类"),
    C(67, "3d", "请标注 KITTI 帧 000142 中的行人、自行车和汽车", "三中文类"),
    C(68, "3d", "用 yolo11s 检测 KITTI 帧 000145 中的车辆和行人", "yolo11s 引擎"),
    C(69, "3d", "标注 KITTI 帧 000055 中的卡车，用 kitti 微调权重", "单卡车+微调"),
    C(70, "3d", "KITTI 000015 帧的行人，帮我标一下", "帧号无前导零"),
    # ================= 3D 特征词/引擎（71-80）=================
    C(71, "3d", "对 KITTI 帧 000049 做 3D 检测，标出汽车", "3D 特征词"),
    C(72, "3d", "标注 KITTI 帧 000010 点云中的汽车", "点云特征词"),
    C(73, "3d", "lidar 点云标注 KITTI 帧 000008 中的汽车", "lidar 特征词"),
    C(74, "3d", "用 pointpillars 标注 KITTI 帧 000067 中的汽车", "pointpillars 引擎"),
    C(75, "3d", "用 pgd 做 KITTI 帧 000021 的单目 3D 检测，标出汽车", "MONO3D 引擎"),
    C(76, "3d", "标注 KITTI 帧 000025 中的汽车和行人，输出 3D 框", "3D 框字面"),
    C(77, "3d", "3D标注 KITTI 帧 000038 中的汽车", "3D 前缀连写"),
    C(78, "3d", "标注 KITTI 帧 000045 中的 bus", "微调 5 类外 bus"),
    C(79, "3d", "标注 KITTI 帧 000050 中的摩托车", "motorcycle→Cyclist 映射"),
    C(80, "3d", "标注 KITTI 帧 000036 中的汽车、行人、卡车、自行车", "四类"),
    # ================= 3D 困难/参数（81-90）=================
    C(81, "3d", "标注 KITTI 帧 000011 中的汽车，置信度 0.9", "高阈值"),
    C(82, "3d", "标注 KITTI 帧 000043 中的行人，置信度 0.1", "低阈值"),
    C(83, "3d", "标注 KITTI 帧 000048 中的汽车，阈值 3.0", "超范围阈值"),
    C(84, "3d", "标注 KITTI 帧 000073 中的汽车，阈值 -0.2", "负阈值"),
    C(85, "3d", "标注 KITTI 帧 000076 中的汽车和行人，用 kitti 微调权重，置信度 0.35，3D 框", "微调+多参数"),
    C(86, "3d", "用 pvrcnn 标注 KITTI 帧 000134 中的汽车", "PV-RCNN 引擎"),
    C(87, "3d", "用 pgd_kitti 标注 KITTI 帧 000142 中的汽车", "MONO3D 全名"),
    C(88, "3d", "帮我给 KITTI 帧 000145 的汽车和行人画 3D 包围盒", "转述式 3D"),
    C(89, "3d", "请标注第 000055 号 KITTI 帧中的汽车", "第 N 号帧句式"),
    C(90, "3d", "请标注 KITTI 帧 000015 中的所有车辆", "泛指所有车辆"),
    # ================= 3D 边界/异常（91-100）=================
    C(91, "3d", "标注 KITTI 帧 999999 中的汽车", "不存在帧"),
    C(92, "3d", "标注 KITTI 帧 abc123 中的汽车", "非法帧号"),
    C(93, "3d", "标注 KITTI 帧 000049 中的独角兽", "不存在类别"),
    C(94, "3d", "标注 KITTI 帧 000010", "无类别"),
    C(95, "3d", "标注 KITTI 帧 000008 中的汽车，用不存在的引擎 not_a_3d_model", "不存在引擎"),
    C(96, "3d", "用 bevfusion 标注 KITTI 帧 000067 中的汽车", "nuScenes 协议引擎×KITTI 帧"),
    C(97, "3d", "KITTI 000025 帧的汽车，标一下", "无「帧」字变体"),
    C(98, "3d", "标注 KITTI 帧 000038 中的汽车", "自定义输出目录", opts=("--no-wait", "-o", "/tmp/autolabel_100/out3d")),
    C(99, "3d", "标注 KITTI 帧 000045 中的汽车", "最大迭代 5", opts=("--no-wait", "--max-iterations", "5")),
    C(100, "3d", "标注 nuScenes 数据集中随机 2 个场景的汽车", "nuScenes 批量意图"),
    # ================= 3D 点云线：真实检出（101-122，PointPillars + GT 富集帧）=================
    C(101, "3d", "用 pointpillars 标注 KITTI 帧 000049 中的汽车", "14 Car GT",
      opts=PP_CHAT, expect=_pp_expect("000049")),
    C(102, "3d", "用 pointpillars 标注 KITTI 帧 000038 中的汽车", "11 Car GT",
      opts=PP_CHAT, expect=_pp_expect("000038")),
    C(103, "3d", "用 pointpillars 标注 KITTI 帧 000010 中的汽车", "8 Car GT",
      opts=PP_CHAT, expect=_pp_expect("000010")),
    C(104, "3d", "用 pointpillars 标注 KITTI 帧 000008 中的汽车", "6 Car GT",
      opts=PP_CHAT, expect=_pp_expect("000008")),
    C(105, "3d", "用 pointpillars 标注 KITTI 帧 000021 中的汽车", "6 Car GT",
      opts=PP_CHAT, expect=_pp_expect("000021")),
    C(106, "3d", "用 pointpillars 标注 KITTI 帧 000045 中的汽车", "6 Car GT",
      opts=PP_CHAT, expect=_pp_expect("000045")),
    C(107, "3d", "用 pointpillars 标注 KITTI 帧 000025 中的汽车", "5 Car GT",
      opts=PP_CHAT, expect=_pp_expect("000025")),
    C(108, "3d", "用 pointpillars 标注 KITTI 帧 000067 中的汽车", "5 Car GT（case-074 实测 6 框回归）",
      opts=PP_CHAT, expect=_pp_expect("000067")),
    C(109, "3d", "用 pointpillars 标注 KITTI 帧 000050 中的汽车", "4 Car GT",
      opts=PP_CHAT, expect=_pp_expect("000050")),
    C(110, "3d", "用 pointpillars 标注 KITTI 帧 000016 中的汽车", "4 Car GT",
      opts=PP_CHAT, expect=_pp_expect("000016")),
    C(111, "3d", "用 pointpillars 标注 KITTI 帧 000055 中的汽车", "4 Car GT",
      opts=PP_CHAT, expect=_pp_expect("000055")),
    C(112, "3d", "用 pointpillars 标注 KITTI 帧 000142 中的汽车", "9 Car GT",
      opts=PP_CHAT, expect=_pp_expect("000142")),
    C(113, "3d", "用 pointpillars 标注 KITTI 帧 000076 中的行人", "8 Ped GT",
      opts=PP_CHAT, expect=_pp_expect("000076")),
    C(114, "3d", "用 pointpillars 标注 KITTI 帧 000145 中的行人", "9 Ped GT",
      opts=PP_CHAT, expect=_pp_expect("000145")),
    C(115, "3d", "用 pointpillars 标注 KITTI 帧 000134 中的行人", "7 Ped GT",
      opts=PP_CHAT, expect=_pp_expect("000134")),
    C(116, "3d", "用 pointpillars 标注 KITTI 帧 000073 中的行人", "5 Ped GT",
      opts=PP_CHAT, expect=_pp_expect("000073")),
    C(117, "3d", "用 pointpillars 标注 KITTI 帧 000011 中的行人", "4 Ped GT",
      opts=PP_CHAT, expect=_pp_expect("000011")),
    C(118, "3d", "用 pointpillars 标注 KITTI 帧 000043 中的行人", "4 Ped GT（无 Car）",
      opts=PP_CHAT, expect=_pp_expect("000043")),
    C(119, "3d", "用 pointpillars 标注 KITTI 帧 000015 中的行人", "4 Ped GT",
      opts=PP_CHAT, expect=_pp_expect("000015")),
    C(120, "3d", "用 pointpillars 标注 KITTI 帧 000134 中骑自行车的人", "5 Cyc GT",
      opts=PP_CHAT, expect=_pp_expect("000134")),
    C(121, "3d", "用 pointpillars 标注 KITTI 帧 000073 中的自行车", "2 Cyc GT（0 框可能）",
      opts=PP_CHAT, expect=_pp_bev_only("000073")),
    C(122, "3d", "用 pointpillars 标注 KITTI 帧 000145 中的自行车", "2 Cyc GT（0 框可能）",
      opts=PP_CHAT, expect=_pp_bev_only("000145")),
    # ================= 3D 点云线：多类/阈值/特征词/显式 BEV（123-136）=================
    C(123, "3d", "用 pointpillars 标注 KITTI 帧 000049 中的汽车和行人", "点云线双类",
      opts=PP_CHAT, expect=_pp_expect("000049")),
    C(124, "3d", "用 pointpillars 标注 KITTI 帧 000142 中的汽车和行人", "点云线双类",
      opts=PP_CHAT, expect=_pp_expect("000142")),
    C(125, "3d", "用 pointpillars 标注 KITTI 帧 000134 中的汽车、行人和自行车", "点云线三类",
      opts=PP_CHAT, expect=_pp_expect("000134")),
    C(126, "3d", "用 pointpillars 标注 KITTI 帧 000049 中的汽车，置信度 0.7", "点云线高阈值",
      opts=PP_CHAT, expect=_pp_bev_only("000049")),
    C(127, "3d", "用 pointpillars 标注 KITTI 帧 000049 中的汽车，置信度 0.9", "点云线极高阈值",
      opts=PP_CHAT, expect=_pp_bev_only("000049")),
    C(128, "3d", "用 pointpillars 标注 KITTI 帧 000049 中的汽车，置信度 0.1", "点云线低阈值",
      opts=PP_CHAT, expect=_pp_expect("000049")),
    C(129, "3d", "对 KITTI 帧 000021 做 lidar 点云检测，标出汽车", "lidar 特征词",
      opts=PP_CHAT, expect=_pp_expect("000021")),
    C(130, "3d", "用雷达点云标注 KITTI 帧 000045 中的汽车", "雷达特征词", opts=PP_CHAT, expect=_pp_expect("000045")),
    C(131, "3d", "标注 KITTI 帧 000038 velodyne 点云中的汽车", "velodyne 特征词",
      opts=PP_CHAT, expect=_pp_expect("000038")),
    C(132, "3d", "use pointpillars to detect cars in KITTI frame 000049", "点云线纯英文",
      opts=PP_CHAT, expect=_pp_expect("000049")),
    C(133, "3d", "用 pointpillars 标注 KITTI 帧 000049 中的汽车，并输出 BEV 鸟瞰图", "显式 BEV（触发 visualize 工具）",
      opts=PP_CHAT, expect=_pp_expect("000049")),
    C(134, "3d", "标注 KITTI 帧 000134 中的汽车、行人和自行车，画 BEV 点云预测图", "转述式 BEV 请求",
      opts=PP_CHAT, expect=_pp_expect("000134")),
    C(135, "3d", "用 pointpillars 标注 KITTI 帧 000049 中的公交车", "3 类外类别（宁多勿漏）",
      opts=PP_CHAT, expect=_pp_bev_only("000049")),
    C(136, "3d", "用 pointpillars 标注 KITTI 帧 000049 中的摩托车", "motorcycle→Cyclist 映射",
      opts=PP_CHAT, expect=_pp_bev_only("000049")),
    # ================= 3D 点云线：边界/第二引擎（137-140）=================
    C(137, "3d", "用 pointpillars 标注 KITTI 帧 abc123 中的汽车", "非法帧号×点云线（F6 护栏）", opts=PP_CHAT),
    C(138, "3d", "用 pointpillars_nus 标注 KITTI 帧 000049 中的汽车", "nuScenes 引擎×KITTI 帧协议错配",
      opts=("--no-wait", "-d", "pointpillars_nus", "-o", OUT_PP)),
    C(139, "3d", "用 pv_rcnn 标注 KITTI 帧 000049 中的汽车", "PV-RCNN 第二引擎（14 Car GT）",
      opts=("--no-wait", "-d", "pvrcnn_kitti", "-o", OUT_PP), expect=_pp_expect("000049")),
    C(140, "3d", "用 pvrcnn 标注 KITTI 帧 000076 中的行人", "PV-RCNN 行人",
      opts=("--no-wait", "-d", "pvrcnn_kitti", "-o", OUT_PP), expect=_pp_bev_only("000076")),
    # ================= 3D 点云线：auto3dlabel run 代码级直跑/批量/跟踪（141-145）=================
    C(141, "3d", "检测汽车", "run 单帧（零 LLM，BEV 点云预测图）",
      argv_prefix=("auto3dlabel", "run", "000049"),
      opts=("-d", "pointpillars_kitti", "-o", OUT_PP_BATCH),
      expect=_batch_expect(("000049",)) + ((f"{OUT_PP_BATCH}/000049_bev.png", 500),)),
    C(142, "3d", "检测汽车", "run 批量 10 帧（一次 forward 整批）",
      argv_prefix=("auto3dlabel", "run", "000049-000058"),
      opts=("-d", "pointpillars_kitti", "--batch", "-o", OUT_PP_BATCH),
      expect=_batch_expect(("000049", "000050", "000052", "000053", "000055"))
      + ((f"{OUT_PP_BATCH}/000049_bev.png", 500),)),
    C(143, "3d", "检测汽车", "run 批量显式 batch-size 4",
      argv_prefix=("auto3dlabel", "run", "000049-000058"),
      opts=("-d", "pointpillars_kitti", "--batch", "--batch-size", "4", "-o", OUT_PP_BATCH),
      expect=_batch_expect(("000049", "000053")) + ((f"{OUT_PP_BATCH}/000053_bev.png", 500),)),
    C(144, "3d", "检测行人", "run 批量行人（000073-000078）",
      argv_prefix=("auto3dlabel", "run", "000073-000078"),
      opts=("-d", "pointpillars_kitti", "--batch", "-o", OUT_PP_BATCH),
      expect=_batch_expect(("000073", "000076")) + ((f"{OUT_PP_BATCH}/000073_bev.png", 500),)),
    C(145, "3d", "检测汽车", "run --track3d 序列（tracks.json + 逐帧 BEV）",
      argv_prefix=("auto3dlabel", "run", "000049-000058"),
      opts=("-d", "pointpillars_kitti", "--batch", "--track3d", "-o", OUT_PP_BATCH),
      expect=((f"{OUT_PP_BATCH}/tracks.json", 100),) + _batch_expect(("000049",))
      + ((f"{OUT_PP_BATCH}/000049_bev.png", 500),)),
]

assert len(CASES) == 145, len(CASES)


@pytest.fixture(scope="session")
def results() -> Generator[list[dict[str, Any]], None, None]:
    """会话级收集器：yield rows 供各用例 append，生成器收尾打印汇总。"""
    rows: list[dict[str, Any]] = []
    yield rows
    # 会话尾汇总
    n_pass = sum(1 for r in rows if r["ok"])
    n_fail = len(rows) - n_pass
    by_domain = {"2d": [0, 0], "3d": [0, 0]}
    for r in rows:
        by_domain[r["domain"]][0 if r["ok"] else 1] += 1
    total_s = sum(r["duration"] for r in rows)
    print("\n" + "=" * 60)
    print(f"汇总: {len(rows)} 条 | PASS {n_pass} / FAIL {n_fail} | "
          f"2D {by_domain['2d'][0]}/{by_domain['2d'][1]}  3D {by_domain['3d'][0]}/{by_domain['3d'][1]} | "
          f"总耗时 {total_s:.0f}s")
    for r in rows:
        if not r["ok"]:
            print(f"  FAIL case-{r['id']:03d} [{r['domain']}] {r['reason']} | {r['instruction'][:60]}")
    print("=" * 60)


@pytest.mark.parametrize("case", CASES, ids=[f"case-{c.id:03d}" for c in CASES])
def test_cli_command(case: Case, results: list[dict]) -> None:
    """真实执行 autolabel：超时内退出、exit 0、无 Traceback + expect 产物检查即 PASS。"""
    t0 = time.time()
    log_path = LOG_DIR / f"case-{case.id:03d}.log"
    cmd = [*case.argv_prefix, *case.opts, case.instruction]
    try:
        proc = subprocess.run(
            cmd, cwd=PROJ, capture_output=True, text=True, timeout=case.timeout
        )
        rc, out, err = proc.returncode, proc.stdout, proc.stderr
        log_path.write_text(
            f"$ {' '.join(cmd)}\n--- stdout ---\n{out}\n--- stderr ---\n{err}\n", errors="replace"
        )
        crashed = "Traceback" in out or "Traceback" in err
        ok = rc == 0 and not crashed
        reason = ""
        if rc != 0:
            reason = f"exit={rc}"
        if crashed:
            reason += (" / " if reason else "") + "traceback"
        if reason:
            reason = f"({reason})"
        # 后置产物检查（点云线真实检出 + BEV 点云预测图）：glob 命中且 ≥ min_bytes
        miss = _expect_misses(case.expect)
        if miss:
            ok = False
            reason = (reason[:-1] + " / " if reason.endswith(")") else "(") + (
                f"expect-miss: {miss})"
            )
        # 摘要：末几行输出（便于快速定位业务行为）
        tail = "\n".join((out + err).strip().splitlines()[-4:])
    except subprocess.TimeoutExpired as e:
        rc, ok, reason, tail = -1, False, "(timeout)", f"TIMEOUT after {case.timeout}s: {e}"
        log_path.write_text(f"$ {' '.join(cmd)}\n{tail}\n", errors="replace")
    row = {
        "id": case.id,
        "domain": case.domain,
        "instruction": case.instruction,
        "note": case.note,
        "ok": ok,
        "exit": rc,
        "reason": reason,
        "duration": round(time.time() - t0, 1),
        "tail": tail,
    }
    (RESULT_DIR / f"case-{case.id:03d}.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    results.append(row)
    assert ok, f"[{case.domain}] {case.instruction!r} {reason}\n{tail}"
