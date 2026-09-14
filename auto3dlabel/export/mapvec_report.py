"""MapTR 矢量对账报告(消费侧 CLI 主流程):契约 → 比对 → 渲染 → 报告 + 复核队列。

AutoDriveData 逐帧契约 `mapvec_pred/1`(`--pred-dir` 下 `{token}.json`,每帧
preds+gts)+ 图像数据(`--img-root` 下 `cam_front/{token}.png` + `calib.json` +
`ego_pose.json`)→ AutoLabel 侧对账产物:

- 校验:全量 `frame_from_dict` 硬校验,坏文件报告跳过(不静默)。
- 三件套比对(见 tools/mapvec_compare.py):投影 overlay(CAM_FRONT,pred 品红 /
  GT 青绿)+ BEV 面板 + 逐帧 TP/FP/FN、CD 分布、越窗计数。
- 报告:`mapvec_report.md`(逐类 AP/CD 中位/最差 5 帧)+ `mapvec_report.json`(逐帧)。
- 复核队列:`--review-out` 下每帧 `{token}_review.json`(annotations 空、
  image_path 指向 overlay,summary 带匹配统计)——`REVIEW3D_DIR=<review-out>`
  起 server 后 `/api/review-files` 自动可见、`/api/review-image` 直接看 overlay,
  `review-save` 无副作用(空 annotations)。

**AP 强制打印 score_thr**:MapTR chamfer AP 是 3 阈值 precision 均值、无 recall 项,
绝对数字必须带阈值引用(产出方红线)。CLI `--score-thr` 与契约内 `score_thr`
不一致时报告告警。比对用的 0.5m 阈值是「是否算匹配」的 CD 判定,与 score_thr
(预测置信度过滤)不同维度,报告里分开标注。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from auto3dlabel.schema.mapvec import MapVecFramePred, load_frame, out_of_window
from auto3dlabel.schema.mapvec_proj import (
    GT_COLOR,
    PRED_COLOR,
    bev_panel,
    cam_pose,
    draw_projected_lines,
    intrinsics_from_k,
)
from auto3dlabel.tools.mapvec_compare import (
    CHAMFER_THRESHOLDS,
    aggregate,
    compare_frame,
)

CAMERA_NAME = "CAM_FRONT"


def scan_pred_files(pred_dir: str | Path) -> list[Path]:
    """契约目录 → 排序后的 {token}.json 列表。"""
    return sorted(Path(pred_dir).glob("*.json"))


def load_all_frames(
    pred_dir: str | Path, progress_cb=None
) -> tuple[list[MapVecFramePred], list[tuple[str, str]]]:
    """扫描 + 全量硬校验。返回 (frames, errors=[(路径, 错误)])——坏文件报告跳过。"""
    frames: list[MapVecFramePred] = []
    errors: list[tuple[str, str]] = []
    files = scan_pred_files(pred_dir)
    for i, p in enumerate(files):
        try:
            frames.append(load_frame(p))
        except (ValueError, KeyError, TypeError, json.JSONDecodeError, OSError) as e:
            errors.append((str(p), str(e)))
        if progress_cb and (i % 50 == 0 or i == len(files) - 1):
            progress_cb(i + 1, len(files))
    return frames, errors


def _inst_lines(rec: MapVecFramePred, attr: str) -> list[np.ndarray]:
    return [np.asarray(i.points, dtype=np.float64) for i in getattr(rec, attr)]


def render_frame(
    rec: MapVecFramePred,
    image_path: Path,
    ego: list[float],
    calib_front: dict,
    out_dir: Path,
) -> tuple[Path, Path, int, int]:
    """单帧 → (overlay png, bev png, pred 段数, gt 段数)。段数 > 0 = 数值自证。"""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    pose = cam_pose(ego, calib_front["sensor2ego"])
    fx, fy, cx, cy = intrinsics_from_k(calib_front["intrinsic"])
    w, h = img.size
    n_pred = draw_projected_lines(
        draw, _inst_lines(rec, "preds"), ego, pose, fx, fy, cx, cy, w, h, PRED_COLOR
    )
    n_gt = draw_projected_lines(
        draw, _inst_lines(rec, "gts"), ego, pose, fx, fy, cx, cy, w, h, GT_COLOR
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    ov = out_dir / f"{rec.token}_overlay.png"
    img.save(ov)
    bev = bev_panel(
        [_inst_lines(rec, "preds")],
        [_inst_lines(rec, "gts")],
        title=f"{rec.token}",
        out_of_window_pts=out_of_window(rec),
    )
    bv = out_dir / f"{rec.token}_bev.png"
    bev.save(bv)
    return ov, bv, n_pred, n_gt


def _worst_frames(results, k: int = 5) -> list[tuple[str, int]]:
    """最差帧 = FN+FP 最多的前 k 帧(配对,并列按 frame 升序)。"""
    scored = sorted(
        ((sum(r.fn.values()) + sum(r.fp.values()), r.frame) for r in results),
        key=lambda x: (-x[0], x[1]),
    )[:k]
    return [(f"{fr:06d}", s) for s, fr in scored]


def write_report(
    results,
    frames: list[MapVecFramePred],
    score_thr: float,
    ckpt: str,
    contract_score_thr: float | None,
    out_dir: str | Path,
    n_pred_segs: int = 0,
    n_gt_segs: int = 0,
) -> Path:
    """汇总 → mapvec_report.md + mapvec_report.json(报告强制带 score_thr)。"""
    agg = aggregate(results, frames)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if contract_score_thr is not None and abs(contract_score_thr - score_thr) > 1e-9:
        warn = f"  ⚠ 契约内 score_thr={contract_score_thr} ≠ CLI --score-thr={score_thr}"
    else:
        warn = ""
    cd = agg
    lines = [
        "# MapTR 矢量对账报告(AutoLabel 消费侧)",
        "",
        f"- 契约: `mapvec_pred/1`,ckpt = `{ckpt}`",
        f"- score_thr = **{score_thr}**{warn}(AP 无 recall 项,绝对数字必须带阈值)",
        f"- CD 匹配判定阈值 = {CHAMFER_THRESHOLDS[0]}m(官方三阈值最低档)",
        f"- 帧数: {int(agg['frames'])}(比对) / {len(frames)}(加载)",
        "",
        "## 全局统计",
        "",
        f"- CD 中位(匹配对): {cd['cd_median_m']:.2f}m"
        f"(p25 {cd['cd_p25_m']:.2f} / p75 {cd['cd_p75_m']:.2f}, {int(cd['cd_count'])} 对)",
        "",
        "| 类 | TP | FP | FN | AP(precision) |",
        "|---|---|---|---|---|",
    ]
    for c in ("divider", "ped_crossing", "boundary", "centerline"):
        lines.append(
            f"| {c} | {int(cd[f'tp_{c}'])} | {int(cd[f'fp_{c}'])} | {int(cd[f'fn_{c}'])} | "
            f"{cd[f'ap_{c}']:.3f} |"
        )
    lines += ["", "## 最差 5 帧(按 FN+FP)", "", "| 帧 | FN+FP |", "|---|---|"]
    for f, s in _worst_frames(results):
        lines.append(f"| {f} | {s} |")
    md = out / "mapvec_report.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    js: dict = {
        "schema": "mapvec_report/1",
        "score_thr": score_thr,
        "contract_score_thr": contract_score_thr,
        "ckpt": ckpt,
        "cd_match_threshold_m": CHAMFER_THRESHOLDS[0],
        "global": agg,
        "frames": [
            {
                "frame": r.frame,
                "token": r.token,
                "tp": r.tp,
                "fp": r.fp,
                "fn": r.fn,
                "cd_median_m": float(np.median(r.cd_dist_hist)) if r.cd_dist_hist else None,
                "out_of_window": r.out_of_window,
                "gt_out_of_window": r.gt_out_of_window,
            }
            for r in results
        ],
        "overlay_segs_drawn": {"pred": n_pred_segs, "gt": n_gt_segs},
    }
    jp = out / "mapvec_report.json"
    jp.write_text(json.dumps(js, indent=2, ensure_ascii=False), encoding="utf-8")
    return md


def build_mapvec_review_queue(
    frames: list[MapVecFramePred],
    results,
    overlay_dir: str | Path,
    review_out: str | Path,
    score_thr: float,
) -> int:
    """每帧 → {token}_review.json(队列形态,annotations 空,image_path 指 overlay)。

    返回写出数量(全部写出,无 resume 语义——每次重跑覆盖,幂等)。
    """
    out = Path(review_out)
    out.mkdir(parents=True, exist_ok=True)
    by_frame = {r.frame: r for r in results}
    written = 0
    for rec in frames:
        r = by_frame.get(rec.frame)
        if r is None:
            continue
        ov = Path(overlay_dir) / f"{rec.token}_overlay.png"
        bev = Path(overlay_dir) / f"{rec.token}_bev.png"
        img_size = [1242, 375]  # CAM_FRONT 固定分辨率(契约图)
        data = {
            "dataset": "mapvec",
            "image": rec.token,
            "image_path": str(ov.resolve()),
            "image_size": img_size,
            "bev_path": str(bev.resolve()) if bev.is_file() else "",
            "description": "MapTR 矢量对账 — GT vs 预测自动比对(pred 品红 / GT 青绿)",
            "annotations": [],
            "summary": {
                "score_thr": score_thr,
                "pred_count": len(rec.preds),
                "gt_count": len(rec.gts),
                "tp": r.tp,
                "fp": r.fp,
                "fn": r.fn,
                "cd_median_m": float(np.median(r.cd_dist_hist)) if r.cd_dist_hist else None,
                "out_of_window": r.out_of_window,
            },
        }
        (out / f"{rec.token}_review.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        written += 1
    return written


def run_mapvec_report(
    pred_dir: str | Path,
    img_root: str | Path,
    out_dir: str | Path,
    review_out: str | Path,
    score_thr: float = 0.2,
    progress_cb=None,
) -> dict:
    """全链路主流程:校验 → 比对 → 渲染 → 报告 → 复核队列。返回汇总 dict。"""
    img_root = Path(img_root)
    calib = json.loads((img_root / "calib.json").read_text(encoding="utf-8"))
    ego_poses = {
        int(p["frame"]): [p["x"], p["y"], p["z"], p["yaw"], p["pitch"], p["roll"]]
        for p in json.loads((img_root / "ego_pose.json").read_text(encoding="utf-8"))
    }
    calib_front = calib[CAMERA_NAME]

    frames, errors = load_all_frames(pred_dir, progress_cb)
    results = [compare_frame(r) for r in frames]
    # 渲染(段数自证;缺图/坏图帧跳过渲染,比对仍保留)
    ov_dir = Path(out_dir) / "overlay"
    n_pred = n_gt = 0
    skipped = []
    for rec in frames:
        img = img_root / "cam_front" / f"{rec.token}.png"
        ego = ego_poses.get(rec.frame)
        if not img.is_file() or ego is None:
            skipped.append(rec.token)
            continue
        _ov, _bv, np_, ng_ = render_frame(rec, img, ego, calib_front, ov_dir)
        n_pred += np_
        n_gt += ng_
        del _ov, _bv  # 仅段数进汇总(数值自证),路径在队列构建时重新解析
    contract_score_thr = frames[0].score_thr if frames else None
    ckpt = frames[0].ckpt if frames else ""
    md = write_report(
        results,
        frames,
        score_thr,
        ckpt,
        contract_score_thr,
        out_dir,
        n_pred_segs=n_pred,
        n_gt_segs=n_gt,
    )
    n_queue = build_mapvec_review_queue(frames, results, ov_dir, review_out, score_thr)
    return {
        "frames": len(frames),
        "errors": errors,
        "skipped_render": skipped,
        "report_md": str(md),
        "overlay_segs": {"pred": n_pred, "gt": n_gt},
        "review_queue": n_queue,
    }
