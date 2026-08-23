"""Auto3dLabel Web 复核界面 — FastAPI 后端（协议照抄 auto2dlabel web 四端点）。

启动: python -m auto3dlabel.web.server
访问: http://localhost:8766

与 2D 版的差异（刻意最小化）：
- 队列文件 = 3D 格式（pcd_path/calib_path + Box3D dict annotations）
- save 3D 版：Box3D.from_dict 直通（不走 2D _bbox_from_dict，保 3D 字段）
  + 同时导出 KITTI label 到 REVIEW_DIR 的兄弟 labels/ 目录
- 安全模式继承：文件名后缀白名单 + 路径遍历拒绝 + 损坏 try/except 跳过
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from auto3dlabel.export.kitti_label import build_label_file
from auto3dlabel.schema.box3d import Box3D

# 复核队列扫描目录（CLI HITL 分流输出目录，可环境变量覆盖供测试/部署）
REVIEW_DIR = Path(os.environ.get("REVIEW3D_DIR", "outputs/kitti3d/reviews"))
# KITTI label 导出目录（Web save 后同步更新）
LABELS_DIR = REVIEW_DIR.parent / "labels"

# 复核图像读取允许的后缀（防任意文件读取）
_ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

STATIC_DIR = Path(__file__).resolve().parent / "static"
app = FastAPI(title="Auto3dLabel Web", version="0.1.0")


# ── 静态文件 ──────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


# ── 复核队列四端点（协议同 auto2dlabel server.py:522-680 安全模式）──

@app.get("/api/review-files")
async def list_review_files() -> JSONResponse:
    """扫描 REVIEW_DIR 下 *_review.json 队列文件（排除已复核 .reviewed）。

    对损坏文件容错跳过（同 2D）；annotations 为 Box3D dict 直读。
    """
    files = []
    if REVIEW_DIR.exists():
        for f in sorted(REVIEW_DIR.glob("*_review.json")):
            if ".reviewed" in f.name:
                continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue  # 损坏文件跳过
            files.append({
                "name": f.name,
                "image_path": data.get("image_path", ""),
                "image_stem": data.get("image", ""),
                "count": len(data.get("annotations", [])),
                "modified": f.stat().st_mtime,
            })
    files.sort(key=lambda x: (-x["modified"], x["name"]))

    # 已复核文件（*_reviewed.json 供重开复查，同款容错扫描）
    reviewed = []
    for f in sorted(REVIEW_DIR.glob("*_reviewed.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue  # 损坏文件跳过
        reviewed.append({
            "name": f.name,
            "image_path": data.get("image_path", ""),
            "image_stem": f.name.removesuffix("_reviewed.json"),
            "count": len(data.get("annotations", [])),
            "modified": f.stat().st_mtime,
        })
    reviewed.sort(key=lambda x: (-x["modified"], x["name"]))
    return JSONResponse({"files": files, "reviewed": reviewed, "dir": str(REVIEW_DIR.resolve())})


@app.get("/api/review-file")
async def get_review_file(name: str = "") -> JSONResponse:
    """读取单个复核队列文件内容（含已复核的 _reviewed.json 供重开）。"""
    if not name.endswith(("_review.json", "_reviewed.json")) or "/" in name or "\\" in name:
        return JSONResponse({"error": f"非法队列文件: {name}"}, status_code=400)
    src = REVIEW_DIR / name
    if not src.is_file():
        return JSONResponse({"error": f"队列文件不存在: {name}"}, status_code=404)
    try:
        return JSONResponse(json.loads(src.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return JSONResponse({"error": f"队列文件损坏: {name}"}, status_code=400)


@app.get("/api/review-image", response_model=None)
async def get_review_image(path: str = "") -> JSONResponse | FileResponse:
    """读取复核队列对应的图（相机图 / BEV 鸟瞰图；仅允许图片后缀）。"""
    p = Path(path)
    if p.suffix.lower() not in _ALLOWED_IMAGE_EXTS or not p.is_file():
        return JSONResponse({"error": f"图像不存在或类型不支持: {path}"}, status_code=404)
    return FileResponse(str(p))


@app.post("/api/review-save")
async def save_review(payload: dict[str, Any] = Body(...)) -> JSONResponse:
    """保存人工复核修正（3D 版）：过滤删除框 → 写 3D _reviewed.json + 导出 KITTI label。

    payload 支持两种模式（互斥，edited 优先）：
    - edited: 修正后的框全量重建（Box3D dict 列表，前端直通）
    - deleted_indices: 仅删除框下标（旧模式，向后兼容）

    3D 字段经 Box3D.from_dict 直通（不经 2D _bbox_from_dict），fit_points 保留；
    KITTI label 同步写入 REVIEW_DIR 兄弟 labels/ 目录（line_from_box3d 单一事实源）。
    """
    name = str(payload.get("queue_file", ""))
    is_reviewed = name.endswith("_reviewed.json")
    if not name.endswith(("_review.json", "_reviewed.json")) or "/" in name or "\\" in name:
        return JSONResponse({"error": f"非法队列文件: {name}"}, status_code=400)

    src = REVIEW_DIR / name
    if not src.is_file():
        return JSONResponse({"error": f"队列文件不存在: {name}"}, status_code=404)

    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return JSONResponse({"error": f"队列文件损坏: {name}"}, status_code=400)

    stem = name.removesuffix("_reviewed.json" if is_reviewed else "_review.json")
    annotations = list(data.get("annotations", []))

    edited = payload.get("edited")
    if edited is not None:
        kept = list(edited)  # 修正框全量重建
        deleted_count = len(annotations) - len(kept)
    else:
        deleted = {int(i) for i in payload.get("deleted_indices", [])}
        kept = [a for i, a in enumerate(annotations) if i not in deleted]
        deleted_count = len(deleted)

    boxes = [Box3D.from_dict(b) for b in kept if isinstance(b, dict)]
    # 人工修正标记（前端可逐框置位）
    for i, b in enumerate(kept):
        if isinstance(b, dict) and b.get("edited_by_human") and i < len(boxes):
            boxes[i].edited_by_human = True

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    out = REVIEW_DIR / f"{stem}_reviewed.json"
    out_data = {
        "image": stem,
        "image_path": data.get("image_path", ""),
        "image_size": data.get("image_size") or [1242, 375],
        "pcd_path": data.get("pcd_path", ""),
        "calib_path": data.get("calib_path", ""),
        "description": "已人工复核（3D）— 修正后保留框",
        "annotations": [b.to_dict() for b in boxes],
        "kept": len(boxes),
        "deleted": deleted_count,
    }
    out.write_text(json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8")

    # KITTI label 同步导出（Web 复核结果直接可用）
    if boxes:
        build_label_file(stem, boxes, LABELS_DIR)

    if not is_reviewed:
        src.rename(src.with_name(name + ".reviewed"))

    return JSONResponse({
        "ok": True,
        "saved_path": str(out),
        "kept": len(boxes),
        "deleted": deleted_count,
    })


# ── 启动 ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("AUTOLABEL3D_PORT", "8766"))
    print("  Auto3dLabel Web v0.1.0")
    print(f"  访问: http://localhost:{port}")
    print(f"  API 文档: http://localhost:{port}/docs")
    print(f"  复核队列: {REVIEW_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=port)
