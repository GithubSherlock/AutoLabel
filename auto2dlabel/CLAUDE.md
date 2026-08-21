# auto2dlabel 开发指南

> 供每次会话参考的**模型选型速查**与**设计红线**。功能定义与里程碑状态见 `AutoLabel_plan.md`，各版本详细记录见 `milestone/`。

## 模型选型速查

### 检测模型（3 引擎 × 29 模型）

| 指令 | 推荐命令 | 理由 |
| --- | --- | --- |
| 快速扫图 | `auto2dlabel run img.jpg "检测汽车和行人" -d yolo12n.pt -t 0.3` | nano 最快，适合预览 |
| 高召回（不漏检） | `auto2dlabel run img.jpg "检测汽车和行人" -d fasterrcnn_resnet50_fpn_v2 -t 0.5` | 35 框，person 检出 16（vs YOLO 仅 6），置信度 85%+ |
| 均衡精度 | `auto2dlabel run img.jpg "检测汽车和行人" -d yolo26x.pt -t 0.5` | 15 框，car/person 各 7，精度与召回折中 |
| 开放词汇（不限类别） | `auto2dlabel run img.jpg "检测所有红色车辆" -d IDEA-Research/grounding-dino-tiny` | 文本 prompt 直出 bbox，中文 prompt 直译后用 |
| 密集场景 | `auto2dlabel run img.jpg "检测行人、汽车、自行车" -d fasterrcnn_resnet50_fpn_v2 -t 0.3 --iou 0.3` | 高召回 + 低 IoU 去重 |
| 旧数据集（VOC 类） | `-d yolo26x.pt` | VOC 20 类实测 mAP 59.2% vs FRCNN 24.5%（`tests/test-v0.1.md`） |
| 大分辨率图像 | 任一模型 + `--sahi` 或指令中提「切片/SAHI」 | SAHI 切片推理 |

### 分割模型（5 类）

| 场景 | 推荐 | 说明 |
| --- | --- | --- |
| 文本驱动的全图分割 | **SAM3**（`sam3.pt`，手动下载 ~3.4GB） | 自带检测+分割一步，开放词汇，最强但慢（~20s） |
| 默认（两段式 bbox→mask） | **SAM2**（`sam2_l.pt`）⭐ | 默认分割模型；coco_seg 实测 0.6368 vs FastSAM 0.4917，几乎不丢框 |
| 精准 bbox→mask 轻量 | **SAM2**（`sam_b.pt`） | 1s，只分割给它的框 |
| 均衡检测+分割 | **Mask R-CNN**（torchvision） | 2.5s，检测+分割一步，多类别覆盖 |
| cityscapes 域内分割 | **Mask R-CNN cityscapes**（`maskrcnn_r50_cityscapes`） | mmdet 官方 cityscapes 权重（Box 40.9 / Mask 36.4），COCO 预训练模型对 30~50px 小目标失效时的唯一出路；需先运行 `python3 -m auto2dlabel.tools.convert_mmdet_cityscapes_maskrcnn` |
| 最轻量 | **FastSAM**（`FastSAM-s.pt`） | 1.1s，bbox IoU 匹配；丢框多，仅快速预览 |
| 语义分割（全图逐类） | **FCN / DeepLabV3 / LRASPP**（torchvision） | VOC 21 类全图分割，每类一个 mask，无需 bbox |

分割任务自动路由：`sam3`/`maskrcnn`/`fcn*`/`deeplabv3*`/`lraspp*` 自带检测无需外部检测模型；`sam`/`fastsam` 先检测后分割（分割专用模型名不能用于检测，检测兜底 `yolo26x.pt`）。两段式 prompt 过滤阈值默认 0.3（`prompt-conf`，实测 0.6368 vs 0.5 的 0.5649；0.0 会 FP 洪水不可取）。

### 其他任务模型（v0.3 已实现分类 + OBB）

| 任务 | 模型 | 说明 |
| --- | --- | --- |
| 图像分类 | CLIP（openai/clip-vit-base-patch32）/ SigLIP（google/siglip-base-patch16-224） | transformers 懒加载，HF_HOME=weights/hf；多候选 softmax 排序取 top-K（零样本，中文候选可直接用） |
| 图像分类 | torchvision 14 款（convnext_large 等 6 款 + resnet18/34/50/101/152、resnext50_32x4d/101_32x8d/101_64x4d，ImageNet1K 监督 top-K） | TORCH_HOME 自动下载；candidates 子串过滤，**候选须英文**（无多语言能力，中文候选建议用 CLIP）；resnet50 通用之选 |
| OBB 旋转框 | yolo11/12/26 n/s/m/l/x-obb.pt | 仅 YOLO-OBB；Oriented R-CNN 延后（mmrotate 依赖重） |

姿态：ViTPose / RTMPose（v0.5 规划）· 跟踪：ByteTrack 默认（v0.4，纯后处理零模型依赖）+ BoT-SORT 精度档（`--bot-sort`，ReID 外观关联 CLIP/SigLIP + ECC 相机运动补偿，权重进 weights/hf/）

### CLI 速查

```bash
auto2dlabel run dir/ "检测汽车" --batch --resume outputs/batch_manifest.json  # 失败隔离 + 续跑
auto2dlabel sample --top-k 5                  # 主动学习采样（聚合 outputs/*_review.json）
auto2dlabel chat "分类为猫和狗，用 clip" --no-wait        # CLIP/SigLIP 零样本分类
auto2dlabel chat "分类为 cat 和 dog，用 convnext_large" --no-wait  # torchvision 监督分类（候选须英文）
auto2dlabel chat "检测旋转框，用 yolo11n-obb.pt" --no-wait  # OBB → dota/yolo_obb 导出
auto2dlabel chat "检测 /data/images 中的汽车" --batch-size 8 --num-workers 4  # 批量推理（未指定时交互询问，仍无则模型加载后自动实测最大 batch）
auto2dlabel run video.mp4 "检测行人" --track          # 跟踪模式：视频/帧目录逐帧检测 + ByteTrack ID + MOT 导出 + 轨迹可视化
auto2dlabel run video.mp4 "检测行人" --track --bot-sort --reid-model google/siglip-base-patch16-224  # 精度档：ReID 外观关联 + ECC（密集场景降 IDSW）
auto2dlabel run video.mp4 "跟踪穿红衣服的人" --track --llm     # 复杂指令：序列级一次性 LLM 解析（失败回退代码级）
auto2dlabel chat "跟踪 video.mp4 中的行人和车辆，使用 ByteTrack" --no-wait   # chat 跟踪入口（planner 产 tracking 步 → 与 run --track 共用 TrackingTool 管线）
auto2dlabel chat "跟踪 dir/ 中的行人、车辆和自行车，使用 BoT-SORT" --no-wait  # BoT-SORT 精度档：指令含 bot-sort 即触发（代码级扫描，LLM 不参与）
auto2dlabel run video.mp4 "检测行人" --track --no-viz   # 测试/省磁盘：跳过逐帧 PNG 可视化（vis_outputs）；MOT/逐帧 JSON/成片视频不受影响
# 视频源跟踪附带标注成片：output_<原名>.mp4 落在源视频同目录（帧率随源视频）；帧目录不产片
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
- **批量推理超参**：`batch_size`/`num_workers` 四档来源——CLI 显式 > chat 交互询问（非法重问 ≤3 次）> 执行阶段动态实测（模型加载后 `tools/device.measure_single_image_memory` 增量法测单图峰值显存 → 空闲显存×0.85÷单图峰值=最大 batch，`resolve_batch_params` 统一入口）> 规划阶段静态表（`recommend_batch_params`，LLM prompt 注入与代码兜底同源，4090 档 det/obb 8 / seg 4 / cls 16）；num_workers 按 CPU 核数 `min(4, cores//4)`（DataLoader 是 CPU 资源，与显存无关）；`cli_execute`/benchmark 按 `hasattr` 分派 `*_batch`（ultralytics/torchvision/CLIP 原生 batch，SAM 系列逐图回退；批量 OOM 时降级逐图 + empty_cache）；单图协议（`detect`/`generate`/`classify`）签名冻结；benchmark 脚本 `--batch`/`--workers` 默认 None=自动实测（1=逐图）。
- **批量 parity 两条铁律（2026-08-18 实测根因，违反即批量结果≠逐图）**：① 所有 ultralytics 推理调用必须显式 `rect=False`（ultralytics predict 默认 `rect=True`，单图走矩形 letterbox、批量混合尺寸自动退化正方形 letterbox，两种预处理不同结果不同——曾致 dota_obb 批量 110+/图 差异）；② 推理入口必须关闭 TF32（`tools/device.disable_tf32`，挂在 `resolve_batch_params` 与 `print_device`；TF32 下 batch=1 与 batch>1 走不同 cudnn kernel，经 200 层 + 角度 argmax 放大后曾致 OBB 批量 20+ 框差异）。两者修复后 dota_obb 批量/逐图 mAP 完全一致（0.8397）。
- **测试位置**：测试在 `auto2dlabel/tests/`（`pytest auto2dlabel/tests/`）——`functional/` pytest 用例、`helpers/` 测试工具（no-LLM baseline / benchmark runner）、`data/` 样例图；实测数据文档（`test-v0.1.md` / `test-v0.2.md`）同目录保留。测试代码不进 cli.py。
- **Bbox.track_id 语义（v0.4）**：`id` 是 annotation 内索引（orchestrator review_flags / labelme group_id / Web bbox_index 引用，勿混用）；跟踪 ID 用独立字段 `track_id`（None=未跟踪）。穿透点六处：`to_dict` / `state._annotation_from_dict` / `export._make_bbox` / `export.coco.build_coco_dict`（从 Bbox 对象直构 dict 绕过 to_dict，曾静默丢 track_id）/ `visualize.draw_bboxes` / `cli_common.display_results`，新增形态勿漏。
- **跟踪评测口径（v0.4 红线）**：MOT 评测必须同时报 MOTA/IDF1 与检测 recall/IDSW（统一走 `benchmarks/track_eval.py`）——recall 低是检测的锅、IDSW 高才是跟踪的锅；`mot_tracking_benchmark.py` 走完整检测→ByteTrack 管线，帧采样用连续窗口（跟踪需时序连续性，不能均匀采样）。
- **prompts 单一事实源（v0.4）**：`CN_EN_MAP`/`extract_prompts` 只在 `tools/prompts.py`；orchestrator / no-LLM baseline / cli_track 均引用（曾三副本发散，勿再复制）。
- **ReID 特征只存 Tracklet 不进 Bbox（v0.4 BoT-SORT）**：`Tracklet.feature` + `update_feature`（EMA + L2 重归一化）；Bbox 无 embedding 字段——track_id 六处穿透点之外不再加特征穿透点；特征逐帧按索引对齐经 `update(bboxes, image, features)` 注入（预提取，跟踪算法本体零权重依赖，tracker 单测可注入合成特征）。
- **BoT-SORT 对比同序列同窗口同 conf（v0.4 红线）**：MOTA/IDF1/IDSW 对比必须同一序列、同一帧窗口、同一 conf（`mot_tracking_benchmark.py --bot-sort` 与基线 ByteTrack 参数一致），且同时报检测 recall/precision——BoT-SORT 默认 track_high_thresh 0.6 / new_track_thresh 0.6（官方 0.7，2026-08-22 标注场景定案降为 0.6：与 track_high 对齐，[0.6,0.7) 未匹配框立即建轨迹，消除轨迹延迟输出；ByteTracker 仍 0.5/0.5），换 tracker 即换口径，勿混比。
- **ReID 单测零真实权重（v0.4 铁律）**：test_reid_model.py / test_bot_sort.py 只用 FakeReIDModel（duck typing ReIDModel Protocol）注入合成特征，不构造 CLIP/SigLIP 实例；CLIP/SigLIP 构造零加载（`_load()` 幂等 + HF_HOME=weights/hf + ImportError 守卫）。
- **TrackingTool 不注册 LLM registry（v0.4 chat 跟踪）**：序列级工具（视频→逐帧→MOT）orchestrator 单图循环调不动；调用方 = execute_plan track 分支 + cli_track 薄壳（共用 `tools/tracking.py` 管线，错误只抛 ValueError、typer.Exit 收敛 CLI 层）；跟踪器选择 `detect_tracker_kind`（tools/tracking.py）为单一事实源——LLM 不参与，chat 在确认后用 plan.raw_instruction 代码级扫描，勿在 planner/execute_plan 再散副本；逐帧 JSON 在 export_format=mot 时映射回 coco（MOT 恒为序列级合并导出）；**视频源附带标注成片**（源视频同目录 `output_<原名>.mp4`，帧率随源视频、mp4v 编码，与 PNG 同一 draw_bboxes+draw_trajectories 绘制管线）——帧目录不产片（防污染数据集目录）。
