# auto2dlabel 开发指南

> 供每次会话参考的**模型选型速查**与**设计红线**。功能定义与里程碑状态见 `AutoLabel_plan.md`，各版本详细记录见 `milestone/`。

## 模型选型速查

### 检测模型（3 引擎 × 49+ 模型）

| 指令 | 推荐命令 | 理由 |
| --- | --- | --- |
| 快速扫图 | `auto2dlabel run img.jpg "检测汽车和行人" -d yolov8n.pt -t 0.3` | nano 最快，适合预览 |
| 高召回（不漏检） | `auto2dlabel run img.jpg "检测汽车和行人" -d fasterrcnn_resnet50_fpn_v2 -t 0.5` | 35 框，person 检出 16（vs YOLO 仅 6），置信度 85%+ |
| 均衡精度 | `auto2dlabel run img.jpg "检测汽车和行人" -d yolov8x.pt -t 0.5` | 15 框，car/person 各 7，精度与召回折中 |
| 开放词汇（不限类别） | `auto2dlabel run img.jpg "检测所有红色车辆" -d IDEA-Research/grounding-dino-tiny` | 文本 prompt 直出 bbox，中文 prompt 直译后用 |
| 密集场景 | `auto2dlabel run img.jpg "检测行人、汽车、自行车" -d fasterrcnn_resnet50_fpn_v2 -t 0.3 --iou 0.3` | 高召回 + 低 IoU 去重 |
| 旧数据集（VOC 类） | `-d yolo26x.pt` | VOC 20 类实测 mAP 59.2% vs FRCNN 24.5%（`tests/test-v0.1.md`） |
| 大分辨率图像 | 任一模型 + `--sahi` 或指令中提「切片/SAHI」 | SAHI 切片推理 |

### 分割模型（4 种）

| 场景 | 推荐 | 说明 |
| --- | --- | --- |
| 文本驱动的全图分割 | **SAM3**（`sam3.pt`，手动下载 ~3.4GB） | 自带检测+分割一步，开放词汇，最强但慢（~20s） |
| 精准 bbox→mask | **SAM2**（`sam_b.pt`） | 1s，只分割给它的框，轻量 |
| 均衡检测+分割 | **Mask R-CNN**（torchvision） | 2.5s，检测+分割一步，多类别覆盖 |
| 最轻量 | **FastSAM**（`FastSAM-s.pt`） | 1.1s，bbox IoU 匹配 |

分割任务自动路由：`sam3`/`maskrcnn` 自带检测无需外部检测模型；`sam`/`fastsam` 先检测后分割（分割专用模型名不能用于检测，检测兜底 `yolo26x.pt`）。

### 其他任务模型（v0.3 已实现分类 + OBB）

| 任务 | 模型 | 说明 |
| --- | --- | --- |
| 图像分类 | CLIP（openai/clip-vit-base-patch32）/ SigLIP（google/siglip-base-patch16-224） | transformers 懒加载，HF_HOME=weights/hf；多候选 softmax 排序取 top-K |
| OBB 旋转框 | yolo11n/s/m/l/x-obb.pt | 仅 YOLO-OBB；Oriented R-CNN 延后（mmrotate 依赖重） |

姿态：ViTPose / RTMPose（v0.4 规划）· 跟踪：ByteTrack / BoT-SORT（v1.0 规划）

### CLI 速查

```bash
auto2dlabel run dir/ "检测汽车" --batch --resume outputs/batch_manifest.json  # 失败隔离 + 续跑
auto2dlabel sample --top-k 5                  # 主动学习采样（聚合 outputs/*_review.json）
auto2dlabel chat "分类为猫和狗，用 clip" --no-wait        # CLIP/SigLIP 零样本分类
auto2dlabel chat "检测旋转框，用 yolo11n-obb.pt" --no-wait  # OBB → dota/yolo_obb 导出
```

## 设计原则与红线

- **Export 不暴露给 LLM**：导出由 CLI 代码直接调用（`tools/export.py`），避免 token 浪费。
- **防重复调用**：Agent Loop 跟踪 `_detect_called`，LLM 第二次调 detect 直接 skip；`max_iterations=3`（`agent/orchestrator.py`）。
- **质量评估分层**：代码级判据（0 框降阈值 ×0.5 重试一次、类别覆盖检查、>200 框警告）在 tool/编排层完成（`agent/evaluate.py`），三条检测路径（Agent Loop / chat / no-LLM baseline）共用；LLM Evaluate 节点仅在 `quality.ok == False` 时条件暴露（evaluate_quality 不进全局 registry，防跨图污染），动作 accept/flag_for_review/retry_lower_threshold（retry = conf×0.25，每图一次），迭代核算 ≤3。
- **OBB 角度约定**：`Bbox.angle` 弧度、(-π/2, π/2]、width 轴相对 x 轴（与 ultralytics xywhr 零转换）；`to_dict` 恒输出；穿透点四处（to_dict / state._annotation_from_dict / export._make_bbox / visualize）漏一处静默丢角。
- **分类结果存 `Annotation.labels`**（`list[ImageLabel]`），不是 metadata；模型名存 `metadata["model"]`。
- **默认检测模型**：`schema/task_plan.DEFAULT_MODEL`（yolo26x.pt）为单一事实源，env `DETECTION_MODEL` 可覆盖；各模块不得硬编码其他默认值。
- **中英映射兜底**：用户消息注入 `Keyword hints: 行人=person` 等，类别翻译不依赖 LLM（代码级兜底）。
- **权重统一管理**：所有权重在 `auto2dlabel/weights/`——Ultralytics 用 `settings.update({"weights_dir": ...})`，PyTorch 用 `TORCH_HOME`。
- **纯本地部署**：所有模型可用开源权重；DeepSeek API 是唯一外部依赖，可换本地 Qwen（`configs/.env`）。
- **分割红线**：SAM 系列在小目标/密集场景 mask 易粘连 → 用检测 bbox 作 prompt，必要时回退 Mask R-CNN。
- **GPU 分级**：模型分级（nano 快速扫 → x/v2 高精度），GPU 紧张时 cascade 策略。
- **测试位置**：测试在 `auto2dlabel/tests/`（`pytest auto2dlabel/tests/`）——`functional/` pytest 用例、`helpers/` 测试工具（no-LLM baseline / benchmark runner）、`data/` 样例图；实测数据文档（`test-v0.1.md` / `test-v0.2.md`）同目录保留。测试代码不进 cli.py。
