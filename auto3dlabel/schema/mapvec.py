"""MapTR 矢量预测契约 `mapvec_pred/1` 的 AutoLabel 侧消费(纯值,不 import AutoDriveData)。

**复制自** `AutoDriveData/autodrivedata/mapvec_schema.py`(契约产出方),版本锁定:
- 依赖单向红线:AutoLabel 不 import AutoDriveData → 纯值逻辑**复制**而非 import。
- 产出方每次改契约,本文件同步(交叉验证测试锁定一致性,见 tools/mapvec_compare.py)。
- 契约字段/校验与产出方逐字段一致:不匹配的 JSON 在 `frame_from_dict` 即拒收。

上层字段:
- `schema` = `"mapvec_pred/1"`(不匹配即拒收)
- `coord` = `"ego"`(x 前/y 左,原点 = ego 后轴中心,单位米)
- `bev_range` = [-15, -30, 15, 30](x 前/y 左,米)
- `preds`/`gts` = 实例数组 `[{class, points: [[x,y]×20], score?}]`,按类序 `MAPTR_CLASSES`
- **越窗不算错误**:GT 恒在窗口内,而 pred 无裁剪(实测 x 达 19.75)——越窗只计数
  供诊断(`out_of_window`),不拒收。
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

SCHEMA_ID = "mapvec_pred/1"
COORD = "ego"
NUM_POINTS = 20
# MapTR 训练四类 + BEV 窗口(与产出方 autodrivedata/mapvec.py 同值;复制为本地常量)
MAPTR_CLASSES = ("divider", "ped_crossing", "boundary", "centerline")
BEV_RANGE = (-15.0, -30.0, 15.0, 30.0)


@dataclass(frozen=True)
class MapVecInstance:
    """一条矢量折线:类 + `num_points` 个 (x, y) ego 系点;pred 带 score,GT 为 None。"""

    cls: str
    points: tuple[tuple[float, float], ...]
    score: float | None = None


@dataclass(frozen=True)
class MapVecFramePred:
    """单帧预测 + 同帧 GT(ego 系,窗口 `BEV_RANGE`)。"""

    frame: int
    token: str
    score_thr: float
    ckpt: str
    preds: tuple[MapVecInstance, ...]
    gts: tuple[MapVecInstance, ...]


# ---------- 构造与校验 ----------


def make_instance(
    cls_name: str, points: Iterable[Sequence[float]], score: float | None = None
) -> MapVecInstance:
    """任意 [x, y] 序列 → 实例(numpy 数组也可;本模块不依赖 numpy)。"""
    return MapVecInstance(cls_name, tuple((float(p[0]), float(p[1])) for p in points), score)


def _check_geometry(inst: MapVecInstance, tag: str, errors: list[str]) -> None:
    if inst.cls not in MAPTR_CLASSES:
        errors.append(f"{tag} 未知类 {inst.cls!r}")
    if len(inst.points) != NUM_POINTS:
        errors.append(f"{tag} 点数 {len(inst.points)} != {NUM_POINTS}")
    for x, y in inst.points:
        if not (math.isfinite(x) and math.isfinite(y)):
            errors.append(f"{tag} 坐标非有限数 {x, y}")
            break


def validate_frame(rec: MapVecFramePred) -> None:
    """结构性校验;越窗(见模块 docstring)不算错误。违规即 ValueError 列表。"""
    errors: list[str] = []
    if not rec.token:
        errors.append("token 为空")
    if not 0.0 <= rec.score_thr <= 1.0:
        errors.append(f"score_thr 越界 {rec.score_thr}")
    for i, inst in enumerate(rec.preds):
        _check_geometry(inst, f"preds[{i}]", errors)
        if inst.score is None:
            errors.append(f"preds[{i}] 缺 score")
        elif not 0.0 <= inst.score <= 1.0:
            errors.append(f"preds[{i}] score 越界 {inst.score}")
    for i, inst in enumerate(rec.gts):
        _check_geometry(inst, f"gts[{i}]", errors)
    if errors:
        raise ValueError("契约校验失败: " + "; ".join(errors[:8]))


def out_of_window(rec: MapVecFramePred) -> int:
    """pred 落在 BEV 窗口外的点数(诊断用;GT 应为 0)。"""
    xmin, ymin, xmax, ymax = BEV_RANGE
    return sum(
        1 for i in rec.preds for x, y in i.points if not (xmin <= x <= xmax and ymin <= y <= ymax)
    )


def gt_out_of_window(rec: MapVecFramePred) -> int:
    """GT 越窗点数——恒应为 0(GT 经裁剪);非 0 说明上游裁剪被绕过。"""
    xmin, ymin, xmax, ymax = BEV_RANGE
    return sum(
        1 for i in rec.gts for x, y in i.points if not (xmin <= x <= xmax and ymin <= y <= ymax)
    )


# ---------- JSON 往返 ----------


def inst_to_dict(inst: MapVecInstance) -> dict:
    """实例 → dict(坐标 mm 精度,与产出方同口径)。"""
    d: dict = {"class": inst.cls, "points": [[round(x, 3), round(y, 3)] for x, y in inst.points]}
    if inst.score is not None:
        d["score"] = round(float(inst.score), 6)
    return d


def inst_from_dict(d: dict) -> MapVecInstance:
    return make_instance(d["class"], d["points"], None if "score" not in d else float(d["score"]))


def frame_to_dict(rec: MapVecFramePred) -> dict:
    return {
        "schema": SCHEMA_ID,
        "frame": rec.frame,
        "token": rec.token,
        "classes": list(MAPTR_CLASSES),
        "coord": COORD,
        "bev_range": list(BEV_RANGE),
        "num_points": NUM_POINTS,
        "score_thr": rec.score_thr,
        "ckpt": rec.ckpt,
        "preds": [inst_to_dict(i) for i in rec.preds],
        "gts": [inst_to_dict(i) for i in rec.gts],
    }


def frame_from_dict(d: dict) -> MapVecFramePred:
    """反序列化 + 硬校验(契约 ID/类序/坐标系/点数不符即拒收)。"""
    if d.get("schema") != SCHEMA_ID:
        raise ValueError(f"schema 不匹配: {d.get('schema')!r} != {SCHEMA_ID!r}")
    if tuple(d.get("classes", ())) != MAPTR_CLASSES:
        raise ValueError(f"classes 顺序不符: {d.get('classes')!r}")
    if d.get("coord") != COORD:
        raise ValueError(f"coord 不符: {d.get('coord')!r} != {COORD!r}")
    if int(d.get("num_points", -1)) != NUM_POINTS:
        raise ValueError(f"num_points 不符: {d.get('num_points')!r} != {NUM_POINTS}")
    rec = MapVecFramePred(
        frame=int(d["frame"]),
        token=str(d["token"]),
        score_thr=float(d["score_thr"]),
        ckpt=str(d["ckpt"]),
        preds=tuple(inst_from_dict(x) for x in d["preds"]),
        gts=tuple(inst_from_dict(x) for x in d["gts"]),
    )
    validate_frame(rec)
    return rec


# ---------- 落盘(逐帧一文件) ----------


def load_frame(path: str | Path) -> MapVecFramePred:
    return frame_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
