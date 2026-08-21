"""BoT-SORT 精度档单测 —— 融合关联 / 特征 EMA / ECC / 确定性（零真实权重）。

特征全部脚本化注入（ReID 模型不参与，零权重下载铁律）；
ECC 用合成灰度图验证。关键锚点：
- test_extensions_off_matches_byte_tracker：全扩展关断时与 ByteTracker 逐位一致
- test_reid_gate_prevents_swap：纯 IoU 匈牙利会交换 ID 的交错场景被 ReID 门控阻止
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
import pytest

from auto2dlabel.models.tracking import (
    BotSORTTracker,
    ByteTracker,
    Tracklet,
    _associate_high,
    _estimate_ecc,
    _rescale_warp,
    _similarity_matrix,
    _warp_boxes,
    _warp_sane,
    ema_feature,
)
from auto2dlabel.schema.annotation import Bbox


def _det(
    cx: float, cy: float, w: float = 50.0, h: float = 50.0, conf: float = 0.9,
) -> Bbox:
    """以中心点构造检测框（测试坐标约定，与 test_byte_tracker 一致）。"""
    return Bbox(
        x=cx - w / 2, y=cy - h / 2, width=w, height=h,
        label="person", confidence=conf,
    )


# ============================================================
# 与 ByteTracker 逐位等价（无回归最强锚点）
# ============================================================

def _scripted_frames() -> list[list[Bbox]]:
    """手工混合场景帧序列：两段关联 + 遮挡恢复 + 孤立低分 + 新目标。"""
    return [
        [_det(100.0, 100.0, conf=0.9), _det(300.0, 100.0, conf=0.7)],
        [_det(105.0, 100.0, conf=0.9), _det(305.0, 100.0, conf=0.35)],  # 低分救援
        [_det(110.0, 100.0, conf=0.9)],  # 目标 2 消失
        [_det(115.0, 100.0, conf=0.9), _det(305.0, 100.0, conf=0.8),
         _det(500.0, 500.0, conf=0.3)],  # 恢复 + 孤立低分（无 ID）
        [_det(120.0, 100.0, conf=0.9), _det(310.0, 100.0, conf=0.8),
         _det(500.0, 500.0, conf=0.9)],  # 新目标新轨迹
    ]


def _run_snapshot(
    frames_fn: Callable[[], list[list[Bbox]]],
    tracker_factory: Callable[[], ByteTracker],
) -> list[tuple[object, ...]]:
    """跑一遍 tracker（每次重建帧对象），收集逐帧输出 + 轨迹状态快照。"""
    tracker = tracker_factory()
    snap: list[tuple[object, ...]] = []
    for boxes in frames_fn():
        tracker.update(boxes)
        snap.append(tuple(
            (b.track_id, b.x, b.y, b.width, b.height, b.confidence)
            for b in boxes
        ))
    for tr in tracker.tracks():
        snap.append((tr.track_id, tr.hits, tr.time_since_update,
                     tr.velocity(), tuple(tr.trajectory())))
    return snap


def test_extensions_off_matches_byte_tracker() -> None:
    """BotSORT 全扩展关断（无特征 + 无 ECC + ByteTrack 同阈值）→ 逐位等于 ByteTracker。

    锚点意义：所有新路径（Tracklet.feature / _associate_high / warp 参数）
    在默认关断时不得改变 ByteTrack 语义——本测试对 _scripted_frames 上
    逐帧输出与轨迹状态做全量逐位对比。
    """
    bot_factory = lambda: BotSORTTracker(  # noqa: E731
        use_cmc=False, track_high_thresh=0.5, new_track_thresh=0.5)
    byte_snap = _run_snapshot(_scripted_frames, ByteTracker)
    bot_snap = _run_snapshot(_scripted_frames, bot_factory)
    assert len(byte_snap) == len(bot_snap)
    for i, (a, b) in enumerate(zip(byte_snap, bot_snap)):
        assert a == b, f"快照第 {i} 项不一致:\n{a}\n!=\n{b}"


def test_botsort_defaults() -> None:
    """BoT-SORT 默认参数（与 ByteTrack 的 0.5/0.5 有意不同）。

    new_track_thresh 0.6 为标注场景定案（2026-08-22，官方 0.7 → 0.6）：
    与 track_high 对齐，[0.6,0.7) 未匹配框立即建轨迹，消除轨迹延迟输出。
    """
    tracker = BotSORTTracker()
    assert tracker.track_high_thresh == 0.6
    assert tracker.new_track_thresh == 0.6
    assert tracker.lambda_ == 0.98
    assert tracker.appearance_thresh == 0.25
    assert tracker.ema_alpha == 0.9
    assert tracker.use_cmc is True
    assert tracker.cmc_model == "euclidean"
    assert tracker.cmc_max_side == 640


def test_new_track_thresh_06_boundary() -> None:
    """方案 2 定案边界（2026-08-22）：conf 0.65 未匹配建轨迹（旧官方 0.7 不建），
    conf 0.55 低分池未匹配仍不建（只救援语义保持）。"""
    det_065 = _det(100.0, 100.0, conf=0.65)
    BotSORTTracker().update([det_065])
    assert det_065.track_id == 0, "[0.6,0.7) 未匹配框应立即建轨迹（消除延迟输出）"

    det_055 = _det(100.0, 100.0, conf=0.55)
    BotSORTTracker().update([det_055])
    assert det_055.track_id is None, "[0.5,0.6) 低分池未匹配不建轨迹（语义保持）"


# ============================================================
# 纯函数：EMA / 仿射 / 相似度
# ============================================================

def test_ema_feature_pure() -> None:
    prev = np.array([1.0, 0.0])
    new = np.array([0.0, 1.0])
    out = ema_feature(prev, new, 0.9)
    assert np.linalg.norm(out) == pytest.approx(1.0)
    assert np.allclose(out, np.array([0.9, 0.1]) / np.sqrt(0.82))
    assert np.allclose(ema_feature(prev, new, 1.0), prev), "α=1 保持旧特征"
    assert np.allclose(ema_feature(prev, new, 0.0), new), "α=0 直接取新特征"


def test_ema_feature_zero_result_preserved() -> None:
    """新旧特征抵消为零向量：原样返回（不产生 NaN）。"""
    out = ema_feature(np.array([1.0, 0.0]), np.array([-1.0, 0.0]), 0.5)
    assert np.all(out == 0.0)


def test_warp_boxes_identity_translation_rotation() -> None:
    boxes = np.array([[10.0, 20.0, 30.0, 40.0]])
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert np.allclose(_warp_boxes(boxes, identity), boxes)
    shift = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]])
    assert np.allclose(_warp_boxes(boxes, shift), [[12.0, 23.0, 32.0, 43.0]])
    rot90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
    assert np.allclose(_warp_boxes(boxes, rot90), [[-20.0, 10.0, -40.0, 30.0]])


def test_rescale_warp_exact() -> None:
    """下采样空间 → 原图空间推导精确断言：R00/R11 不变、
    R01×sx/sy、R10×sy/sx、t 逐分量×缩放。"""
    warp_small = np.array([[1.0, 0.1, 3.0], [0.2, 1.0, 4.0]])
    out = _rescale_warp(warp_small, 2.0, 4.0)
    assert np.allclose(out, [[1.0, 0.05, 6.0], [0.4, 1.0, 16.0]])
    out2 = _rescale_warp(warp_small, 2.0, 2.0)
    assert np.allclose(out2, [[1.0, 0.1, 6.0], [0.2, 1.0, 8.0]])


def test_similarity_matrix_missing_feature_nan() -> None:
    """任一侧缺特征 → NaN（走纯 IoU 通道）；双侧有特征 → 精确余弦。"""
    t0 = Tracklet(0, _det(100.0, 100.0))
    t1 = Tracklet(1, _det(200.0, 100.0))
    t0.feature = np.array([1.0, 0.0])
    sim = _similarity_matrix([None, np.array([0.0, 1.0])], [t0, t1])
    assert np.isnan(sim[0, 0]) and np.isnan(sim[0, 1]), "检测 0 缺特征整行 NaN"
    assert sim[1, 0] == 0.0, "cos([0,1], [1,0]) = 0"
    assert np.isnan(sim[1, 1]), "轨迹 1 缺特征该格 NaN"


# ============================================================
# 融合关联（_associate_high）
# ============================================================

def test_associate_high_lambda_switches_winner() -> None:
    """融合 cost 语义：λ=0.98 外观主导选 tr1（IoU 较远但外观近），
    λ=0 退化纯 IoU 选 tr0（IoU 更近）。"""
    tr0 = Tracklet(0, _det(201.0, 100.0))  # 距 det d=1，IoU 更近
    tr1 = Tracklet(1, _det(200.0, 100.0))  # 距 det d=2，外观更近（cos≈0.8）
    tr0.feature = np.array([1.0, 0.0])
    tr1.feature = np.array([-0.12, 0.992773])
    det = _det(202.0, 100.0)
    feat = np.array([0.5, np.sqrt(0.75)])  # cos(feat, tr0)=0.5, cos(feat, tr1)≈0.8

    matches, unmatched, _ = _associate_high(
        [det], [feat], [tr0, tr1], None, 0.8, 0.98, 0.25)
    assert len(matches) == 1 and matches[0][2].track_id == 1, \
        f"外观主导应选 tr1，实际 {matches}"
    assert unmatched == []

    matches_io, _, _ = _associate_high(
        [det], [feat], [tr0, tr1], None, 0.8, 0.0, 0.25)
    assert len(matches_io) == 1 and matches_io[0][2].track_id == 0, \
        "λ=0 应退化纯 IoU 选 tr0"


def test_associate_high_gate_rejection_and_iou_fallback() -> None:
    """cosine 门控拒绝外观不一致对；任一侧缺特征回退纯 IoU 通道。"""
    tr0 = Tracklet(0, _det(201.0, 100.0))
    tr1 = Tracklet(1, _det(200.0, 100.0))
    tr0.feature = np.array([1.0, 0.0])  # tr1 无特征
    det = _det(202.0, 100.0)

    # 特征与 tr0 正交（cos=0 < 0.25）→ 门控拒绝；tr1 缺特征 → 纯 IoU 匹配
    matches, _, _ = _associate_high(
        [det], [np.array([0.0, 1.0])], [tr0, tr1], None, 0.8, 0.98, 0.25)
    assert len(matches) == 1 and matches[0][2].track_id == 1, \
        "门控拒绝 tr0 后应经纯 IoU 通道匹配 tr1"

    # 唯一候选被门控拒绝 → 检测未匹配（即使 IoU 很高）
    matches2, unmatched2, _ = _associate_high(
        [det], [np.array([0.0, 1.0])], [tr0], None, 0.8, 0.98, 0.25)
    assert matches2 == []
    assert len(unmatched2) == 1 and unmatched2[0][0] is det


def test_associate_high_iou_hard_gate_blocks_far_appearance_match() -> None:
    """IoU 硬门回归（2026-08-22 轨迹跳变根因）：外观相似但 IoU=0 的框对不得匹配。

    真实案例：sportscheck 视频帧 245→246 两框相隔 1300px、IoU=0、cos=0.5，
    漏门致轨迹跨屏跳变 + 两目标间 flip-flop（轨迹线从画面一端跳到另一端）。
    """
    tr = Tracklet(7, _det(100.0, 100.0))
    tr.feature = np.array([1.0, 0.0])
    far = _det(1500.0, 500.0)  # 与 tr 预测框零重叠
    feat = np.array([0.5, np.sqrt(0.75)])  # cos(feat, tr.feature) = 0.5

    matches, unmatched, _ = _associate_high(
        [far], [feat], [tr], None, 0.8, 0.98, 0.25)
    assert matches == [], "IoU=0 的框对即使外观相似（cos=0.5）也不得匹配"
    assert len(unmatched) == 1 and unmatched[0][0] is far


def test_reid_gate_prevents_swap() -> None:
    """交错场景：纯 IoU 匈牙利总 cost 更低的配对是「交换 ID」，
    ReID 门控（外观正交对 cos=0）阻止交换——BoT-SORT 价值演示。

    构造：两轨迹预测框 200.5/200.0，检测框 202/201（互为交错）。
    IoU 矩阵 identity 总 cost 0.09747 > swap 0.09673 → 纯 IoU 必交换；
    特征正交（A=[1,0]、B=[0,1]）→ 交换对 cos=0 被门控拒绝 → 恒等匹配。
    """
    byte = ByteTracker()
    bot = BotSORTTracker(use_cmc=False)
    feats = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    byte.update([_det(100.0, 100.0), _det(300.0, 100.0)])
    bot.update([_det(100.0, 100.0), _det(300.0, 100.0)], features=feats)

    # 白盒：把两轨迹预测位置摆成交错构型（速度清零防卡尔曼漂移）
    for tracker in (byte, bot):
        tracker._tracklets[0].mean[0] = 200.5  # noqa: SLF001
        tracker._tracklets[1].mean[0] = 200.0  # noqa: SLF001
        tracker._tracklets[0].mean[4] = 0.0  # noqa: SLF001
        tracker._tracklets[1].mean[4] = 0.0  # noqa: SLF001

    det_a, det_b = _det(202.0, 100.0), _det(201.0, 100.0)
    byte.update([det_a, det_b])
    assert det_a.track_id == 1 and det_b.track_id == 0, \
        "纯 IoU 匈牙利在交错构型下交换 ID（对照组）"

    det_a2, det_b2 = _det(202.0, 100.0), _det(201.0, 100.0)
    bot.update([det_a2, det_b2], features=feats)
    assert det_a2.track_id == 0 and det_b2.track_id == 1, \
        "ReID 门控阻止交换，ID 保持正确"


# ============================================================
# 特征 EMA 生命周期
# ============================================================

def test_feature_ema_on_match_and_rescue_no_update() -> None:
    """高分命中 EMA 更新特征；低分救援（纯 IoU）不更新特征。"""
    bot = BotSORTTracker(use_cmc=False)
    f0 = np.array([1.0, 0.0])
    bot.update([_det(100.0, 100.0, conf=0.9)], features=[f0])
    feat0 = bot.tracks()[0].feature
    assert feat0 is not None
    assert np.allclose(feat0, f0), "首帧特征原样存储"

    # 相似但不同的特征（cos 20° ≈ 0.94 ≥ 0.25 过门控）
    f1 = np.array([np.cos(np.deg2rad(20.0)), np.sin(np.deg2rad(20.0))])
    bot.update([_det(105.0, 100.0, conf=0.9)], features=[f1])
    merged = 0.9 * f0 + 0.1 * f1
    expected = merged / np.linalg.norm(merged)
    feat1 = bot.tracks()[0].feature
    assert feat1 is not None
    assert np.allclose(feat1, expected), "EMA α=0.9 + 重归一化"

    bot.update([_det(110.0, 100.0, conf=0.35)], features=[f1])  # 低分救援
    feat2 = bot.tracks()[0].feature
    assert feat2 is not None
    assert np.allclose(feat2, expected), "低分救援不更新特征"


def test_new_track_feature_init() -> None:
    """新轨迹带首帧特征初始化；特征为 None 时轨迹无特征。"""
    bot = BotSORTTracker(use_cmc=False)
    bot.update([_det(100.0, 100.0)], features=[np.array([1.0, 0.0])])
    feat = bot.tracks()[0].feature
    assert feat is not None
    assert np.allclose(feat, [1.0, 0.0])

    bot2 = BotSORTTracker(use_cmc=False)
    bot2.update([_det(100.0, 100.0)], features=[None])
    assert bot2.tracks()[0].feature is None


# ============================================================
# ECC 相机运动补偿
# ============================================================

def _texture(seed: int) -> np.ndarray[Any, Any]:
    """平滑合成纹理（高斯模糊噪声——ECC 需要空间相关梯度，纯白噪声不收敛）。"""
    rng = np.random.default_rng(seed)
    noise = (rng.random((120, 160)) * 255).astype(np.uint8)
    return cv2.GaussianBlur(noise, (15, 15), 0)


def test_ecc_shift_recovery() -> None:
    """合成纹理平移 (3, -2)：ECC translation 模式恢复误差 < 1.5px。"""
    tex = _texture(0)
    m = np.array([[1, 0, 3], [0, 1, -2]], dtype=np.float32)
    shifted = cv2.warpAffine(tex, m, (160, 120))
    warp = _estimate_ecc(tex, shifted, "translation")
    assert warp is not None
    assert warp[0, 2] == pytest.approx(3.0, abs=1.5)
    assert warp[1, 2] == pytest.approx(-2.0, abs=1.5)


def test_ecc_euclidean_rotation_recovery() -> None:
    """合成纹理绕中心旋转 5°：euclidean 模式恢复线性部（cos≈0.9962），
    平移分量满足 t = c − R·c（绕图像中心 (80,60) 旋转的恒等式）。"""
    tex = _texture(1)
    rotated = cv2.warpAffine(
        tex, cv2.getRotationMatrix2D((80, 60), 5.0, 1.0), (160, 120))
    warp = _estimate_ecc(tex, rotated, "euclidean")
    assert warp is not None
    assert abs(warp[0, 0] - 0.9962) < 0.02
    assert abs(abs(warp[0, 1]) - 0.0872) < 0.02, "R01 ≈ ∓sin5°"
    c = np.array([80.0, 60.0])
    assert np.allclose(warp[:, 2], c - warp[:, :2] @ c, atol=3.0), \
        "绕中心旋转：t = c − R·c"


def test_ecc_constant_image_returns_none() -> None:
    """恒定图（无梯度）ECC 报错 → 返回 None（回退恒等）。"""
    constant = np.full((60, 80), 127, dtype=np.uint8)
    assert _estimate_ecc(constant, constant, "translation") is None


def test_ecc_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="ECC"):
        _estimate_ecc(
            np.zeros((4, 4), dtype=np.uint8),
            np.zeros((4, 4), dtype=np.uint8), "homography")


def test_warp_sane_rejections() -> None:
    """合理性三查：正常微平移通过；平移超 30% 短边 / 缩放异常拒绝。"""
    ok = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 5.0]])
    assert _warp_sane(ok, 100, 100)
    big_t = np.array([[1.0, 0.0, 40.0], [0.0, 1.0, 0.0]])  # 40 > 0.3·100
    assert not _warp_sane(big_t, 100, 100)
    scaled = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])  # det=4
    assert not _warp_sane(scaled, 100, 100)


# ============================================================
# 确定性（property 测试）：含 ECC + 特征注入的完整路径
# ============================================================

def test_botsort_deterministic_with_ecc_and_features() -> None:
    """完整 BoT-SORT 路径（ECC 开 + 特征注入）同输入必同输出。"""
    from auto2dlabel.benchmarks.tracker_time_benchmark import synthetic_frames

    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur(
        (rng.random((256, 256)) * 255).astype(np.uint8), (21, 21), 0)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def feat_fn(b: Bbox) -> np.ndarray[Any, Any]:
        v = np.array([b.x, b.y, 1.0])
        return v / float(np.linalg.norm(v))

    def run() -> list[tuple[object, ...]]:
        tracker = BotSORTTracker()
        snap: list[tuple[object, ...]] = []
        for boxes in synthetic_frames(30, 50, seed=5):
            feats = [feat_fn(b) if b.confidence >= 0.6 else None for b in boxes]
            tracker.update(boxes, image=img_bgr, features=feats)
            snap.append(tuple(
                (b.track_id, b.x, b.y, b.width, b.height, b.confidence)
                for b in boxes))
        for tr in tracker.tracks():
            feat = tr.feature
            feat_key: object = (
                None if feat is None else tuple(np.round(feat, 6)))
            snap.append((tr.track_id, tr.hits, tr.time_since_update,
                         tr.velocity(), tuple(tr.trajectory()), feat_key))
        return snap

    first, second = run(), run()
    assert len(first) == len(second)
    for i, (a, b) in enumerate(zip(first, second)):
        assert a == b, f"快照第 {i} 项不一致:\n{a}\n!=\n{b}"
