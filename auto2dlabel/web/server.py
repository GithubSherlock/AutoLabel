"""AutoLabel Web 审核界面 — FastAPI 后端。

启动: python -m auto2dlabel.web.server
访问: http://localhost:8765

支持 model_catalog.py 中所有检测和分割模型。
"""

from __future__ import annotations

import base64
import io
import json
import os
import tempfile
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import Body, FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from auto2dlabel.export.coco import build_coco_dict
from auto2dlabel.models.detection import create_detection_model
from auto2dlabel.models.model_catalog import (
    ALL_DETECTION_MODELS,
    COCO_CLASSES,
    GROUNDING_DINO_MODELS,
    PYTORCH_DETECTION_MODELS,
    SEGMENTATION_MODELS,
    ULTRALYTICS_MODELS,
    WEIGHTS_DIR,
)
from auto2dlabel.schema.annotation import Annotation, Bbox, Mask
from auto2dlabel.schema.task_plan import DEFAULT_MODEL

# 复核队列扫描目录（CLI HITL 分流输出目录）
REVIEW_DIR = Path("outputs")

# 复核图像读取允许的后缀（防任意文件读取）
_ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

STATIC_DIR = Path(__file__).resolve().parent / "static"
app = FastAPI(title="AutoLabel Web", version="0.3.0")

# 确保 weights 目录存在
WEIGHTS_DIR.mkdir(exist_ok=True)

# ── 静态文件 ──────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


# ── 模型目录接口 ──────────────────────────────────────────────

def _classify_model(name: str) -> str:
    """判断模型属于哪个类别。"""
    if name in GROUNDING_DINO_MODELS:
        return "grounding_dino"
    if name in ULTRALYTICS_MODELS:
        return "ultralytics"
    if name in PYTORCH_DETECTION_MODELS:
        return "pytorch_vision"
    if "/" in name:  # HuggingFace ID 变体
        return "grounding_dino"
    if name.endswith(".pt"):
        return "ultralytics"
    if any(name.startswith(p) for p in ("fasterrcnn_", "retinanet_", "ssd", "fcos_")):
        return "pytorch_vision"
    return "unknown"


@app.get("/api/models")
async def list_detection_models():
    """列出所有可用检测模型，按类别分组。"""
    grouped: dict[str, list[dict]] = {
        "grounding_dino": [],
        "ultralytics": [],
        "pytorch_vision": [],
        "custom": [],
    }

    # 分类所有模型
    for name in ALL_DETECTION_MODELS:
        cat = _classify_model(name)
        entry = {
            "name": name,
            "type": (
                "open_vocabulary"
                if cat == "grounding_dino" or "world" in name.lower()
                else "coco_classes"
            ),
        }
        grouped[cat].append(entry)

    # 扫描自定义权重文件
    custom_files: list[str] = []
    for ext in ("*.pt", "*.pth", "*.onnx", "*.engine", "*.trt"):
        for f in WEIGHTS_DIR.glob(ext):
            if f.name not in ALL_DETECTION_MODELS:
                custom_files.append(f.name)
    for name in sorted(set(custom_files)):
        grouped["custom"].append({"name": name, "type": "custom_weights"})

    return JSONResponse({
        "models": grouped,
        "total": sum(len(v) for v in grouped.values()),
        "weights_dir": str(WEIGHTS_DIR),
        "coco_classes": COCO_CLASSES[:20] + ["..."],  # 预览前 20 个
    })


@app.get("/api/seg-models")
async def list_segmentation_models():
    """列出所有可用分割模型，按类别分组。"""
    grouped: dict[str, list[dict]] = {
        "fastsam": [],
        "sam": [],
        "sam2": [],
        "sam3": [],
        "maskrcnn": [],
        "custom": [],
    }

    for name in SEGMENTATION_MODELS:
        name_lower = name.lower()
        if name_lower.startswith("fastsam"):
            grouped["fastsam"].append({"name": name, "type": "fastsam"})
        elif "sam2." in name_lower or name_lower.startswith("sam2"):
            grouped["sam2"].append({"name": name, "type": "sam2"})
        elif name_lower.startswith("sam3"):
            grouped["sam3"].append({"name": name, "type": "sam3"})
        elif any(
            name_lower.startswith(prefix) for prefix in ("sam_t", "sam_s", "sam_b", "sam_l")
        ):
            grouped["sam"].append({"name": name, "type": "sam"})
        elif "maskrcnn" in name_lower:
            grouped["maskrcnn"].append({"name": name, "type": "maskrcnn"})

    # 扫描自定义分割权重
    seen = set(SEGMENTATION_MODELS)
    for ext in ("*.pt", "*.pth", "*.onnx"):
        for f in WEIGHTS_DIR.glob(ext):
            if f.name not in seen and f.name not in ALL_DETECTION_MODELS:
                grouped["custom"].append({"name": f.name, "type": "custom_weights"})

    return JSONResponse({
        "models": grouped,
        "total": sum(len(v) for v in grouped.values()),
        "weights_dir": str(WEIGHTS_DIR),
    })


@app.get("/api/weights-dir")
async def get_weights_dir():
    """获取权重目录路径及已有文件列表。"""
    files = []
    for ext in ("*.pt", "*.pth", "*.onnx", "*.engine", "*.trt"):
        for f in sorted(WEIGHTS_DIR.glob(ext)):
            files.append({
                "name": f.name,
                "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                "modified": f.stat().st_mtime,
            })
    return JSONResponse({
        "weights_dir": str(WEIGHTS_DIR),
        "files": files,
    })


# ── 权重上传接口 ──────────────────────────────────────────────

@app.post("/api/upload-model")
async def upload_model(file: UploadFile = File(...)):
    """上传自定义模型权重文件到 auto2dlabel/weights/。"""
    if not file.filename:
        return JSONResponse({"error": "文件名不能为空"}, status_code=400)

    # 安全检查：只允许常见权重后缀
    allowed_exts = {".pt", ".pth", ".onnx", ".engine", ".trt", ".bin", ".safetensors"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed_exts:
        return JSONResponse(
            {"error": f"不支持的文件类型: {suffix}。允许: {', '.join(allowed_exts)}"},
            status_code=400,
        )

    dest = WEIGHTS_DIR / file.filename
    content = await file.read()
    dest.write_bytes(content)

    return JSONResponse({
        "ok": True,
        "filename": file.filename,
        "path": str(dest),
        "size_mb": round(len(content) / (1024 * 1024), 2),
    })


# ── 标注接口 ──────────────────────────────────────────────────

def _mask_to_dict(mask: Mask, ann: Annotation) -> dict[str, Any]:
    """Mask 转 dict 并附加 bbox_index（指向响应 bboxes 数组下标）。

    bbox prompt 路径（SAM2/FastSAM）的 Mask.bbox 复用 ann.bboxes 原对象，
    id 即数组下标；SAM3/Mask R-CNN 自检框无 id 时按坐标回退匹配。
    """
    d = mask.to_dict()
    if mask.bbox.id is not None:
        d["bbox_index"] = mask.bbox.id
    else:
        d["bbox_index"] = -1
        for i, b in enumerate(ann.bboxes):
            if (b.x, b.y, b.width, b.height) == (
                mask.bbox.x, mask.bbox.y, mask.bbox.width, mask.bbox.height,
            ):
                d["bbox_index"] = i
                break
    return d


def _bbox_from_dict(d: dict[str, Any]) -> Bbox:
    """从前端传来的 bbox dict 重建 Bbox。"""
    return Bbox(
        x=float(d["x"]), y=float(d["y"]),
        width=float(d["width"]), height=float(d["height"]),
        label=str(d.get("label", "")),
        confidence=float(d.get("confidence", 1.0)),
    )


def _annotation_from_payload(payload: dict[str, Any]) -> Annotation:
    """从 Web 前端提交的过滤结果重建 Annotation（供导出端点复用）。"""
    ann = Annotation(
        image_path=str(payload.get("image_path", "upload.jpg")),
        image_size=(int(payload.get("image_width", 0)), int(payload.get("image_height", 0))),
    )
    for b in payload.get("bboxes", []):
        ann.add_bbox(_bbox_from_dict(b))
    for m in payload.get("masks", []):
        ann.add_mask(Mask(
            bbox=_bbox_from_dict(m["bbox"]),
            segmentation=m.get("segmentation", []),
            area=float(m.get("area", 0.0)),
        ))
    return ann

@app.post("/api/annotate")
async def annotate(
    image: UploadFile = File(...),
    instruction: str = Form("检测 car 和 person"),
    conf: float = Form(0.1),
    iou: float = Form(0.3),
    model: str = Form(DEFAULT_MODEL),
    with_seg: str = Form("false"),
    seg_model: str = Form("FastSAM-s.pt"),
    box_threshold: float = Form(0.3),
    text_threshold: float = Form(0.25),
):
    """标注接口：上传图像 + 指令 → 返回标注结果 + 可视化图。

    Args:
        image: 上传的图像文件。
        instruction: 中文逗号/英文逗号分隔的检测目标，如 "car, person, bicycle"。
        conf: 置信度阈值 (0-1)。
        iou: IoU 阈值 (0-1)。
        model: 检测模型名，支持 model_catalog 中所有模型 + 自定义权重文件名。
        with_seg: 是否同时进行实例分割 ("true" / "false")。
        seg_model: 分割模型名，支持所有 SEGMENTATION_MODELS。
        box_threshold: Grounding DINO 的 box 阈值。
        text_threshold: Grounding DINO 的文本阈值。
    """
    # ── 1. 读取并保存图像 ──
    img_bytes = await image.read()
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return JSONResponse({"error": "无法解析上传的图像文件"}, status_code=400)

    img_np = np.array(img)

    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(img_bytes)
        tmp_path = tmp.name

    try:
        # ── 2. 构建模型参数 ──
        model_kwargs: dict = {"iou_threshold": iou}

        # Grounding DINO 专属参数
        if "/" in model or model in GROUNDING_DINO_MODELS:
            model_kwargs["box_threshold"] = box_threshold
            model_kwargs["text_threshold"] = text_threshold

        # ── 3. 检测 ──
        instructions = [s.strip() for s in instruction.replace("，", ",").split(",") if s.strip()]
        det_model = create_detection_model(model, **model_kwargs)
        results = det_model.detect(tmp_path, instructions, confidence_threshold=conf)

        # 构建 Annotation
        ann = Annotation(image_path=image.filename or "upload.jpg")
        ann.image_size = (img.width, img.height)
        for r in results:
            ann.add_bbox(Bbox(
                x=r.x, y=r.y, width=r.width, height=r.height,
                label=r.label, confidence=r.confidence,
            ))

        # ── 4. 可选分割 ──
        do_seg = with_seg.lower() in ("true", "on", "1", "yes")
        masks_out: list[dict] = []

        if do_seg:
            from auto2dlabel.models.segmentation import create_segmentation_model

            seg = create_segmentation_model(seg_model)
            high_conf = [b for b in ann.bboxes if b.confidence >= 0.5]

            if high_conf:
                seg_masks = seg.generate(tmp_path, high_conf)
                for m in seg_masks:
                    ann.add_mask(m)
                masks_out = [_mask_to_dict(m, ann) for m in ann.masks]

        # ── 5. 生成可视化 ──
        import cv2

        vis_img = img_np.copy()
        vis_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

        from auto2dlabel.tools.visualize import draw_bboxes, draw_masks

        if ann.masks:
            vis_bgr = draw_masks(vis_bgr, ann.masks)
        if ann.bboxes:
            vis_bgr = draw_bboxes(vis_bgr, ann.bboxes)

        vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)

        buf = io.BytesIO()
        Image.fromarray(vis_rgb).save(buf, format="PNG")
        vis_b64 = base64.b64encode(buf.getvalue()).decode()

        # 类别统计
        classes = list({b.label for b in ann.bboxes})
        avg_conf = round(sum(b.confidence for b in ann.bboxes) / max(len(ann.bboxes), 1), 3)

        return JSONResponse({
            "image_base64": f"data:image/png;base64,{vis_b64}",
            "image_path": image.filename or "upload.jpg",
            "bboxes": [b.to_dict() for b in ann.bboxes],
            "masks": masks_out,
            "summary": {
                "total": len(ann.bboxes),
                "classes": classes,
                "avg_conf": avg_conf,
                "with_seg": do_seg and len(ann.masks) > 0,
                "model": model,
                "seg_model": seg_model if do_seg else None,
            },
        })

    except ImportError as e:
        return JSONResponse({
            "error": f"依赖缺失: {e}\n请安装所需包后重试。",
            "detail": traceback.format_exc(),
        }, status_code=500)
    except FileNotFoundError as e:
        return JSONResponse({
            "error": str(e),
            "detail": traceback.format_exc(),
        }, status_code=404)
    except Exception as e:
        return JSONResponse({
            "error": f"标注失败: {e}",
            "detail": traceback.format_exc(),
        }, status_code=500)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ── 旧接口兼容 ────────────────────────────────────────────────

@app.post("/annotate")
async def annotate_legacy(
    image: UploadFile = File(...),
    instruction: str = Form("检测 car 和 person"),
    conf: float = Form(0.1),
    iou: float = Form(0.3),
    model: str = Form(DEFAULT_MODEL),
    with_seg: str = Form("false"),
):
    """旧接口：重定向到新接口。"""
    return await annotate(
        image=image, instruction=instruction, conf=conf, iou=iou,
        model=model, with_seg=with_seg,
    )


# ── 前端导出（过滤后重建 COCO） ───────────────────────────────

@app.post("/api/export-coco")
async def export_coco_endpoint(payload: dict[str, Any] = Body(...)) -> JSONResponse:
    """由前端过滤后的 bbox/mask 重建 COCO JSON，供浏览器下载。"""
    try:
        ann = _annotation_from_payload(payload)
        coco = build_coco_dict([ann])
        return JSONResponse(coco)
    except Exception as e:
        return JSONResponse({"error": f"导出失败: {e}"}, status_code=400)


# ── 复核队列消费（HITL 人工复核闭环） ─────────────────────────

@app.get("/api/review-files")
async def list_review_files() -> JSONResponse:
    """扫描 outputs/ 下 *_review.json 队列文件（排除已复核 .reviewed）。

    对旧版缺 image_path/image_size 的文件容错（置空）。
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
    return JSONResponse({"files": files, "dir": str(REVIEW_DIR.resolve())})


@app.get("/api/review-file")
async def get_review_file(name: str = "") -> JSONResponse:
    """读取单个复核队列文件内容（含已复核的 _reviewed.json 供下载）。"""
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
    """读取复核队列对应的原图（仅允许图片后缀，防任意文件读取）。"""
    p = Path(path)
    if p.suffix.lower() not in _ALLOWED_IMAGE_EXTS or not p.is_file():
        return JSONResponse({"error": f"图像不存在或类型不支持: {path}"}, status_code=404)
    return FileResponse(str(p))


@app.post("/api/review-save")
async def save_review(payload: dict[str, Any] = Body(...)) -> JSONResponse:
    """保存人工复核修正：过滤删除框 → 写 COCO → 源队列文件标记 .reviewed。"""
    name = str(payload.get("queue_file", ""))
    if not name.endswith("_review.json") or "/" in name or "\\" in name:
        return JSONResponse({"error": f"非法队列文件: {name}"}, status_code=400)

    src = REVIEW_DIR / name
    if not src.is_file():
        return JSONResponse({"error": f"队列文件不存在: {name}"}, status_code=404)

    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return JSONResponse({"error": f"队列文件损坏: {name}"}, status_code=400)

    annotations = list(data.get("annotations", []))
    deleted = {int(i) for i in payload.get("deleted_indices", [])}
    kept = [a for i, a in enumerate(annotations) if i not in deleted]

    stem = name.removesuffix("_review.json")
    ann = Annotation(
        image_path=data.get("image_path", data.get("image", "")),
        image_size=tuple(data.get("image_size", (0, 0))),
    )
    for b in kept:
        ann.add_bbox(_bbox_from_dict(b))

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    out = REVIEW_DIR / f"{stem}_reviewed.json"
    out.write_text(
        json.dumps(build_coco_dict([ann]), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    src.rename(src.with_name(name + ".reviewed"))

    return JSONResponse({
        "ok": True,
        "saved_path": str(out),
        "kept": len(kept),
        "deleted": len(deleted),
    })


# ── 启动 ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("AUTOLABEL_PORT", "8765"))
    print("  AutoLabel Web v0.3.0")
    print(f"  访问: http://localhost:{port}")
    print(f"  API 文档: http://localhost:{port}/docs")
    print(f"  权重目录: {WEIGHTS_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=port)
