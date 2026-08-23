"""指代约束过滤层 —— 指令结构化约束解析 + 属性/空间过滤纯函数（v0.4 3a）。

统一洞察（2026-08-22 定案）：「穿红衣服的人」（属性）、「车道上的行人」
（空间 ROI）、「左边的人」（方位）本质是同一件事——类别之外的附加约束。
统一为一条过滤链，指代检测 L1 与手动 ROI（C1）复用同一实现：

    指令 → parse_referential 解析为 ReferentialConstraint
         → 检测类别框（现有 29 模型）→ filter_by_attributes（CLIP 逐框）
         → filter_by_spatial（ROI 多边形 / 方位词分位）
         → 剩余目标 → 跟踪/标注

解析是代码级的（属性/方位词表扫描，与 CN_EN_MAP 同模式）；--llm 路径
在 cli_track._llm_plan_once 输出同构 JSON（字段向后兼容）。
属性打分模型走 Protocol 注入（零真实权重单测铁律，同 ReIDModel 模式）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from auto2dlabel.schema.annotation import Bbox
from auto2dlabel.tools.prompts import extract_prompts

# 属性词表（中文颜色 → CLIP 文本候选用英文形容词）
CN_ATTR_MAP: dict[str, str] = {
    "红色": "red", "红": "red", "黄色": "yellow", "黄": "yellow",
    "蓝色": "blue", "蓝": "blue", "绿色": "green", "绿": "green",
    "黑色": "black", "黑": "black", "白色": "white", "白": "white",
    "灰色": "gray", "灰": "gray", "橙色": "orange", "橙": "orange",
    "紫色": "purple", "紫": "purple", "粉色": "pink", "粉": "pink",
    "棕色": "brown", "棕": "brown",
}

# 方位词表（→ 画面坐标分位过滤；「前/后」为深度语义无可靠 2D 映射，
# 由 ROI 多边形兜底，见 v0.4.md 3a 实现记录）
CN_POSITION_MAP: dict[str, str] = {
    "左边": "left", "左侧": "left", "左": "left", "left": "left",
    "右边": "right", "右侧": "right", "右": "right", "right": "right",
    "中间": "center", "中央": "center", "中心": "center", "center": "center",
    "上边": "top", "上方": "top", "上": "top", "top": "top",
    "下边": "bottom", "下方": "bottom", "下": "bottom", "bottom": "bottom",
}

# 属性匹配置信度阈值（CLIP 候选对 softmax 概率）
ATTR_MATCH_THRESHOLD = 0.5

# 关系词表（v0.5 指代 L2 触发）：中文关系词 → Florence-2 <OD> 英文短语词。
# 「左/右/上/下/中间」已在 CN_POSITION_MAP（L1 方位分位），此处只收纯关系语义；
# has_relation 命中即自动启用 L2 解析（Florence-2 看图文出框）。
CN_RELATION_EN_MAP: dict[str, str] = {
    "旁边": "next to", "附近": "near", "周围": "around", "之间": "between",
    "紧挨": "next to", "挨着": "next to", "靠近": "near", "邻近": "near",
    "一侧": "beside", "不远处": "near",
    "前面": "in front of", "后面": "behind",
}

# 英文关系关键词（大小写不敏感子串扫描，与中文表共用同一触发语义）
RELATION_KEYWORDS: list[str] = [
    "next to", "near", "beside", "between", "in front of", "behind",
    "above", "below", "on top of", "under", "around", "adjacent to",
]


@dataclass
class ReferentialConstraint:
    """指令结构化约束：类别 + 可选属性/空间约束（无约束时等价纯 prompts）。"""

    prompts: list[str]
    attributes: list[str] = field(default_factory=list)   # 英文形容词（如 "red"）
    roi: list[tuple[float, float]] | None = None          # ROI 多边形顶点（像素坐标）
    position: str | None = None                           # left/right/center/top/bottom
    threshold: float | None = None                        # LLM 建议阈值（沿用 CLI 时 None）

    @property
    def is_plain(self) -> bool:
        """是否无任何附加约束（纯类别检测）。"""
        return not self.attributes and self.roi is None and self.position is None


class AttributeScorer(Protocol):
    """属性打分器协议 —— duck typing 注入 Fake，零真实权重单测铁律。

    输入 PIL 裁剪图列表 + 文本候选，返回概率矩阵 scores[i][j] =
    第 i 张图对第 j 个候选的概率（**候选原序**，softmax 归一化）。
    真实现 ClipCropScorer 吃 PIL 对象不落盘（逐框裁剪性能要求）。
    """

    def score_crops(
        self,
        images: list[Any],
        candidates: list[str],
    ) -> list[list[float]]: ...


def parse_referential(instruction: str) -> ReferentialConstraint:
    """代码级解析指令 → 结构化约束（类别沿用 extract_prompts，属性/方位词表扫描）。

    属性多词 AND 关系；方位词取第一个匹配。无属性/方位时退化为纯 prompts
    （is_plain=True，向后兼容）。ROI 不来自文本（CLI --roi 单独注入）。

    Raises:
        ValueError: 无类别关键词（同 extract_prompts）。
    """
    prompts = extract_prompts(instruction)
    attributes: list[str] = []
    for cn, en in CN_ATTR_MAP.items():
        if cn in instruction and en not in attributes:
            attributes.append(en)
    position: str | None = None
    for cn, pos in CN_POSITION_MAP.items():
        if cn in instruction:
            position = pos
            break
    return ReferentialConstraint(prompts=prompts, attributes=attributes, position=position)


def has_relation(instruction: str) -> bool:
    """是否含关系指代（中英文关系词任一命中）——L2 Florence-2 自动触发判据。"""
    if any(cn in instruction for cn in CN_RELATION_EN_MAP):
        return True
    lower = instruction.lower()
    return any(kw in lower for kw in RELATION_KEYWORDS)


def _subject_after_relation(instruction: str, prompts: list[str]) -> str:
    """主体类别 = 关系词之后最近的类别词（中文指令经 CN_EN_MAP 反向查）。

    「红车旁边的行人」→ 关系词「旁边」之后最近的类别词是「行人」→ person。
    无命中回退 prompts[0]（空 prompts 时返回空串，phrase 构造层过滤）。
    """
    rel_end = -1
    for cn in CN_RELATION_EN_MAP:
        idx = instruction.find(cn)
        if idx >= 0:
            rel_end = max(rel_end, idx + len(cn))
    if rel_end < 0:
        return prompts[0] if prompts else ""

    from auto2dlabel.tools.prompts import CN_EN_MAP

    best, best_idx = prompts[0] if prompts else "", 10**9
    for p in prompts:
        for cn, en in CN_EN_MAP.items():
            if en != p:
                continue
            idx = instruction.find(cn)
            if idx >= rel_end and idx < best_idx:
                best, best_idx = p, idx
    return best


def build_referential_phrase(instruction: str, constraint: ReferentialConstraint) -> str:
    """构造 Florence-2 <OD> 英文短语（L2 解析输入）。

    「红车旁边的行人」→ "the person next to the red car"：定冠词引导主体 +
    关系词英译 + 属性 + 其余类别词（语境）。2026-08-23 实测：无冠词时
    Florence-2 把关系短语理解为定位地标（"person next to car" 出车框、
    "person near car" 出车框），主体加 "the" 后稳定命中主体
    （"the person next to the red car" 出行人框）；无关系词时退化为
    "属性 + 类别"（此时 L1 已够，L2 仅兜底）。
    """
    rel_en: str | None = None
    for cn, en in CN_RELATION_EN_MAP.items():
        if cn in instruction:
            rel_en = en
            break
    if rel_en is None:
        lower = instruction.lower()
        for kw in RELATION_KEYWORDS:
            if kw in lower:
                rel_en = kw
                break
    subject = _subject_after_relation(instruction, constraint.prompts)
    others = [p for p in constraint.prompts if p != subject]
    # 有关系词：「the 主体 关系词 the 属性+语境类别」；无地标（prompts 单类）
    # 时只用「the 主体」；无关系词退化「属性 类别」（不加冠词，保旧行为）
    if rel_en:
        landmark = " ".join(p for p in constraint.attributes + others if p)
        return f"the {subject} {rel_en} the {landmark}" if landmark else f"the {subject}"
    return " ".join(p for p in constraint.attributes + [subject] + others if p)


def parse_roi(text: str) -> list[tuple[float, float]]:
    """解析 --roi 文本 → 多边形顶点列表（像素坐标）。

    矩形: "x1,y1,x2,y2" → 4 顶点；多边形: "x1,y1;x2,y2;..."（≥3 点）。
    非法输入抛 ValueError。
    """
    parts = text.strip().split(";")
    if len(parts) == 1:
        nums = parts[0].split(",")
        if len(nums) != 4:
            raise ValueError(f"--roi 矩形需 4 个数 (x1,y1,x2,y2): {text}")
        x1, y1, x2, y2 = (float(n) for n in nums)
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    pts: list[tuple[float, float]] = []
    for part in parts:
        nums = part.split(",")
        if len(nums) != 2:
            raise ValueError(f"--roi 多边形顶点需 (x,y): {part}")
        pts.append((float(nums[0]), float(nums[1])))
    if len(pts) < 3:
        raise ValueError(f"--roi 多边形至少 3 个顶点: {text}")
    return pts


def filter_by_spatial(
    bboxes: list[Bbox],
    *,
    roi: list[tuple[float, float]] | None = None,
    position: str | None = None,
    image_size: tuple[int, int] | None = None,
) -> list[Bbox]:
    """空间过滤（纯函数）：ROI 多边形包含（框中心点）+ 方位词坐标分位。

    roi: 多边形顶点；框中心点在多边形内保留（ray casting，零 numpy 依赖）。
    position: left/right/center/top/bottom —— 框中心落在画面 1/3 分位区。
    两者同时给出时取交集（AND）。image_size 仅方位过滤需要。
    """
    if roi is None and position is None:
        return list(bboxes)

    def center(b: Bbox) -> tuple[float, float]:
        return (b.x + b.width / 2, b.y + b.height / 2)

    out: list[Bbox] = []
    for b in bboxes:
        cx, cy = center(b)
        ok_roi = True
        if roi is not None:
            ok_roi = _point_in_polygon(cx, cy, roi)
        ok_pos = True
        if position is not None:
            assert image_size is not None, "方位过滤需要 image_size"
            w, h = image_size
            ok_pos = {
                "left": cx < w / 3,
                "right": cx > 2 * w / 3,
                "center": w / 3 <= cx <= 2 * w / 3,
                "top": cy < h / 3,
                "bottom": cy > 2 * h / 3,
            }.get(position, False)
        if ok_roi and ok_pos:
            out.append(b)
    return out


def _point_in_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    """射线法点-多边形包含判定（含边界）。"""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def filter_by_attributes(
    image: Any,  # PIL Image
    bboxes: list[Bbox],
    attributes: list[str],
    scorer: AttributeScorer,
    match_threshold: float = ATTR_MATCH_THRESHOLD,
) -> list[Bbox]:
    """属性过滤：逐框裁剪 → 候选对 ["a {attr} {label}", "a {label}"] 打分。

    每属性候选对 softmax 概率 > match_threshold 保留；多属性 AND 关系。
    image 为已打开的 PIL 图（调用方负责打开，逐帧循环内复用零重复 I/O）；
    scorer 走 Protocol 注入（真实现 ClipCropScorer / 测试 Fake，零权重铁律）。
    """
    if not attributes or not bboxes:
        return list(bboxes)
    crops = [
        image.crop((int(b.x), int(b.y), int(b.x + b.width), int(b.y + b.height)))
        for b in bboxes
    ]
    keep = [True] * len(bboxes)
    for attr in attributes:
        pos_cands = [f"a {attr} {b.label}" for b in bboxes]
        neg_cands = [f"a {b.label}" for b in bboxes]
        # 唯一候选合并（CLIP 文本编码共享）；每框取自己 label 的正负候选索引
        unique = list(dict.fromkeys(pos_cands + neg_cands))
        idx_pos = [unique.index(c) for c in pos_cands]
        idx_neg = [unique.index(c) for c in neg_cands]
        scores = scorer.score_crops(crops, unique)  # 概率矩阵（候选原序）
        for i in range(len(bboxes)):
            if not keep[i]:
                continue
            s_pos = scores[i][idx_pos[i]]
            s_neg = scores[i][idx_neg[i]]
            if s_pos + s_neg <= 0.0 or s_pos / (s_pos + s_neg) < match_threshold:
                keep[i] = False
    return [b for b, k in zip(bboxes, keep) if k]
