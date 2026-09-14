"""MapTR 矢量逐帧比对纯值(消费侧):Chamfer 匹配 → TP/FP/FN + CD 分布 + 越窗计数。

**复制自** `AutoDriveData/autodrivedata/chamfer_ap.py` + `mapvec_schema.py` 的
比对逻辑,版本锁定;AutoLabel 不 import AutoDriveData,纯值逻辑本地复制
(交叉验证测试锁定一致性,见 tests/functional/test_mapvec_compare.py)。

比对口径(与产出方官方评估同口径):
- `chamfer_distance` CD = 双向最近点平均距离(米),pred×GT 逐对;
- `match_greedy` 逐类贪婪一对一:每个预测按最近 GT 距离排序、近者先占位,
  未占用 GT 不重复使用;CD ≤ 阈值 → TP,余 pred → FP、余 GT → FN;
- 逐帧先匹配再计数,全局再聚(对齐官方跨帧汇聚,GT/pred 不对称帧不引入偏差)。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from auto3dlabel.schema.mapvec import (
    BEV_RANGE,
    MAPTR_CLASSES,
    MapVecFramePred,
    out_of_window,
)

# MapTR 官方 chamfer 阈值口径(米)
CHAMFER_THRESHOLDS = (0.5, 1.0, 1.5)


def chamfer_distance(pred: np.ndarray, gt: np.ndarray) -> float:
    """折线 Chamfer 距离 [米]。pred (P, 2),gt (Q, 2)。"""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.size == 0 or gt.size == 0:
        return float("inf")
    d_pq = np.linalg.norm(pred[:, None] - gt[None], axis=-1)  # (P, Q)
    p2q = d_pq.min(axis=1).sum()
    q2p = d_pq.min(axis=0).sum()
    return float((p2q + q2p) / (pred.shape[0] + gt.shape[0]))


def _pad_polylines(polys: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """折线列表 → (补齐坐标 (N, L, 2), 掩码 (N, L), 点数 (N,))。虚点放 _PAD_XY 处。"""
    max_len = max(len(p) for p in polys)
    pad = np.full((len(polys), max_len, 2), _PAD_XY)
    mask = np.ones((len(polys), max_len), dtype=bool)
    for i, p in enumerate(polys):
        pad[i, : len(p)] = p
        mask[i, : len(p)] = False
    return pad, mask, np.array([len(p) for p in polys], dtype=np.float64)


# 补齐虚点坐标:放在远大于任何真实使用距离的位置(本项目 BEV 窗口 ±30m,
# 真实点距 < 1e2m 量级),min 永不选中虚点
_PAD_XY = 1e4


def chamfer_cost_matrix(
    preds: list[np.ndarray], gts: list[np.ndarray], chunk: int = 32
) -> np.ndarray:
    """批量 Chamfer 代价矩阵 (Np, Ng)——与逐对 chamfer_distance 同口径,向量化加速。

    折线点数可变:按最长补齐,虚点放在 _PAD_XY 远处——真实点距离 < 1e3m,
    min 永不选虚点。距离用平方展开 + BLAS GEMM,分块限制中间张量大小。
    精度口径:平方距离与 min 走 float32,求和与归一化走 float64。
    """
    np_ = len(preds)
    ng = len(gts)
    cost = np.empty((np_, ng), dtype=np.float64)
    if np_ == 0 or ng == 0:
        return cost
    p_pad, p_mask, p_len = _pad_polylines(preds)
    q_pad, q_mask, q_len = _pad_polylines(gts)
    qf = q_pad.reshape(-1, 2).astype(np.float32)  # (Ng·Lq, 2)
    q_norm2 = (qf * qf).sum(axis=1)  # (Ng·Lq,)

    for c in range(0, np_, chunk):
        pc = p_pad[c : c + chunk]  # (Nc, Lp, 2)
        pm = p_mask[c : c + chunk]
        nc, lp = pc.shape[:2]
        pf = pc.reshape(-1, 2).astype(np.float32)  # (Nc·Lp, 2)
        sq = (pf * pf).sum(axis=1)[:, None] + q_norm2[None, :] - 2 * (pf @ qf.T)
        np.clip(sq, 0.0, None, out=sq)  # 浮点负零截断 + 原地省两次分配
        np.sqrt(sq, out=sq)
        d = sq.reshape(nc, lp, ng, q_pad.shape[1])
        p2q = np.where(pm[:, :, None], np.float32(0), d.min(axis=3)).sum(axis=1, dtype=np.float64)
        q2p = np.where(q_mask[None, :, :], np.float32(0), d.min(axis=1)).sum(
            axis=2, dtype=np.float64
        )
        cost[c : c + chunk] = (p2q + q2p) / (p_len[c : c + chunk][:, None] + q_len[None, :])
    return cost


def match_greedy(
    preds: list[np.ndarray], gts: list[np.ndarray], thr: float, cost: np.ndarray | None = None
) -> tuple[int, int, int]:
    """逐类贪婪一对一匹配 → (TP, FP, FN)(阈值 thr 米)。preds/gts 为折线列表。"""
    if not preds:
        return 0, 0, len(gts)
    if not gts:
        return 0, len(preds), 0
    if cost is None:
        cost = chamfer_cost_matrix(preds, gts)
    used = np.zeros(len(gts), dtype=bool)
    tp = 0
    for i in np.argsort(cost.min(axis=1)):  # 按最近 GT 距离排序,近者优先占位
        j = int(cost[i].argmin())
        if cost[i, j] <= thr and not used[j]:
            tp += 1
            used[j] = True
    fp = len(preds) - tp
    fn = len(gts) - used.sum()
    return int(tp), int(fp), int(fn)


def _by_class(
    rec: MapVecFramePred,
) -> tuple[dict[str, list[np.ndarray]], dict[str, list[np.ndarray]]]:
    """preds/gts 实例 → {cls: [points(N,2)]};类序无关,长度按 MAPTR_CLASSES 顺序聚。"""
    preds_by: dict[str, list[np.ndarray]] = {c: [] for c in MAPTR_CLASSES}
    gts_by: dict[str, list[np.ndarray]] = {c: [] for c in MAPTR_CLASSES}
    for i in rec.preds:
        preds_by[i.cls].append(np.asarray(i.points, dtype=np.float64))
    for i in rec.gts:
        gts_by[i.cls].append(np.asarray(i.points, dtype=np.float64))
    return preds_by, gts_by


@dataclass(frozen=True)
class FrameCompareResult:
    """单帧比对结果:逐类 TP/FP/FN + 全局 CD 距离(仅匹配对)+ 越窗计数。

    若一帧某类两边皆空 → 不产生 CD 样本(空 vs 空无匹配);pred 空而 gt 有 → FN。
    """

    frame: int
    token: str
    tp: dict[str, int]
    fp: dict[str, int]
    fn: dict[str, int]
    cd_dist_hist: list[float]  # 逐帧匹配对 CD 距离(米),供全局中位/分位
    out_of_window: int  # pred 越窗点数(诊断;越窗不拒收)
    gt_out_of_window: int  # GT 越窗点数——恒应为 0


def compare_frame(rec: MapVecFramePred, thr: float = 0.5) -> FrameCompareResult:
    """单帧比价:逐类 match_greedy + CD 分布 + 越窗计数(纯值)。

    阈值默认 0.5m(MapTR 官方三阈值最低档,作"是否算匹配"的判定)。CD 分布
    用匹配对的 CD 距离(≤ thr 且被贪婪选中的对),未匹配的 pred/gt 不计入。
    """
    preds_by, gts_by = _by_class(rec)
    tp: dict[str, int] = {}
    fp: dict[str, int] = {}
    fn: dict[str, int] = {}
    cd: list[float] = []
    for cls in MAPTR_CLASSES:
        p, g = preds_by[cls], gts_by[cls]
        t, f, n = match_greedy(p, g, thr)
        tp[cls], fp[cls], fn[cls] = t, f, n
        # 匹配对 CD(与 match_greedy 同序:按最近 GT 距离升序、近者先占位)
        if t > 0:
            cost = chamfer_cost_matrix(p, g)
            used = np.zeros(len(g), dtype=bool)
            for i in np.argsort(cost.min(axis=1)):
                j = int(cost[i].argmin())
                if cost[i, j] <= thr and not used[j]:
                    cd.append(float(cost[i, j]))
                    used[j] = True
    return FrameCompareResult(
        frame=rec.frame,
        token=rec.token,
        tp=tp,
        fp=fp,
        fn=fn,
        cd_dist_hist=cd,
        out_of_window=out_of_window(rec),
        gt_out_of_window=gt_out_of_window(rec),
    )


def gt_out_of_window(rec: MapVecFramePred) -> int:
    """GT 越窗点数——恒应为 0(GT 经裁剪);非 0 说明上游裁剪被绕过。"""
    xmin, ymin, xmax, ymax = BEV_RANGE
    return sum(
        1 for i in rec.gts for x, y in i.points if not (xmin <= x <= xmax and ymin <= y <= ymax)
    )


# 供 export 层使用(与 mapvec_schema 同签名别名,便于 copy-and-paste 消费侧统一)
def aggregate(
    results: Sequence[FrameCompareResult],
    frames: Sequence[MapVecFramePred] | None = None,
) -> dict[str, float]:
    """逐帧结果 → 全局统计:逐类 TP/FP/FN、CD 中位/p25/p75、AP(官方三阈值 precision 均值)。

    TP/FP/FN 用 0.5m CD 判定(compare_frame 同阈值);AP 与产出方 chamfer_ap
    同口径——三阈值 {0.5, 1.0, 1.5}m 的 precision 均值,**无 recall 项**。
    frames 提供原始折线供 AP 重算(跨帧聚合同官方);不传则退化单 0.5m 的
    precision(仅作 fallback)。绝对数字必须带 score_thr 引用(report 层强制打印)。
    """
    per_class: dict[str, dict[str, int]] = {c: {"tp": 0, "fp": 0, "fn": 0} for c in MAPTR_CLASSES}
    cds: list[float] = []
    for r in results:
        for c in MAPTR_CLASSES:
            per_class[c]["tp"] += r.tp[c]
            per_class[c]["fp"] += r.fp[c]
            per_class[c]["fn"] += r.fn[c]
        cds.extend(r.cd_dist_hist)
    cd_arr = np.asarray(cds, dtype=np.float64) if cds else np.array([], dtype=np.float64)
    out: dict[str, float] = {
        "frames": float(len(results)),
        "cd_median_m": float(np.median(cd_arr)) if len(cd_arr) else float("nan"),
        "cd_p25_m": float(np.percentile(cd_arr, 25)) if len(cd_arr) else float("nan"),
        "cd_p75_m": float(np.percentile(cd_arr, 75)) if len(cd_arr) else float("nan"),
        "cd_count": float(len(cd_arr)),
    }
    for c in MAPTR_CLASSES:
        t, f = per_class[c]["tp"], per_class[c]["fp"]
        out[f"tp_{c}"] = float(t)
        out[f"fp_{c}"] = float(f)
        out[f"fn_{c}"] = float(per_class[c]["fn"])
        if frames is not None:
            preds = [
                np.asarray(i.points, dtype=np.float64)
                for fr in frames
                for i in fr.preds
                if i.cls == c
            ]
            gts = [
                np.asarray(i.points, dtype=np.float64)
                for fr in frames
                for i in fr.gts
                if i.cls == c
            ]
            cost = chamfer_cost_matrix(preds, gts) if preds and gts else None
            precs: list[float] = []
            for thr in CHAMFER_THRESHOLDS:
                tp, fp, _ = match_greedy(preds, gts, thr, cost=cost)
                precs.append(tp / max(1, tp + fp))
            out[f"ap_{c}"] = float(np.mean(precs))
        else:
            out[f"ap_{c}"] = float(t) / max(1, t + f)
    return out
