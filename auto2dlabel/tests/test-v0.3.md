# v0.3 实测数据（自 `milestone/v0.3.md` 转移）

> 纯 CPU 可行性验证（2026-08-16，32 核 CPU / 无 GPU，torch 2.13.0 CPU）。功能定义与完成标记见 `../milestone/v0.3.md`。产物在仓库根 `benchmarks_outputs/`（时间戳命名），汇总 `summary_2026-08-16-20-42-29.md`。

## CPU 可行性验证

目标：证明「数据准备 → 推理 → 逐类评测 → 报告」全管线在**无 GPU** 服务器上可行（可行性结论），精度仅作 GPU 对照参考（nano 模型 + 每集 50 图小样本，不做精度结论）。

### 三模块冒烟吞吐（--max-images 2）

| 模块 | 模型 | 冒烟吞吐 | 外推依据 |
| --- | --- | --- | --- |
| 分类 | `resnet18` | 0.02s/图 | 全量 5000 图 ≈ 60s |
| 目标检测 | `yolo11n.pt` | 1.15s/图（coco2017） | 50 图 ≈ 60s |
| 实例分割 | `yolo11n.pt` + `FastSAM-s.pt` | 1.55s/图（coco_seg） | 50 图 ≈ 80s |

**结论**：三模块单集 50 图预估均 ≤ 90s，远低于 `run_all` 3600s 超时红线，全量可行。检测/分割默认模型 `yolo26x.pt` 在 CPU 上 50 图预计超时，故全量以 `yolo11n.pt` 覆盖（`--extra "--model yolo11n.pt"`）；`yolo26x` / `rtdetr-x` / `sam2_l` 等大模型的精度对比留 GPU。

### 分类全量：ImageNet100 5000 图（resnet18，监督 1000 类空间）

每类均匀抽样解压 100 类 × 50 张 = 5000 图 / 566MB（`--per-class 50`）。

| 模型 | top-1 | top-5 | 吞吐 | 总耗时 |
| --- | --- | --- | --- | --- |
| `resnet18` | **78.0%** | **94.6%** | **85.3 img/s** | ~59s |

**结论**：纯 CPU 上 5000 图分类约 1 分钟，吞吐无瓶颈；top-1 78.0% 为 resnet18 在 1000 类空间的官方口径成绩，管线评测逻辑与 torchvision 元数据 100/100 匹配（`imagenet100_resnet18_2026-08-16-20-38-55`）。

### 检测全量：5 数据集 × 50 图（yolo11n.pt，conf=0.3）

| 数据集 | mAP@0.5 | 吞吐（img/s） | 备注 |
| --- | --- | --- | --- |
| voc2007 | **0.5198** | — | 20 类；bicycle/bus/sheep/train 等小样本类 AP=1.0 |
| coco2017 | 0.2632 | — | 46 类出现在 50 图抽样 |
| kitti | 0.2610 | 11.2 | 5 类（car 0.5758 / person 0.4336） |
| mot | 0.1725 | **12.0** | 单类 person；Prec 0.9009 / Recall 0.1224（4231 GT 仅检出 575，密集行人漏检） |
| dota | 0.0303 | 6.4 | 4 类；航拍小目标几乎全漏（car GT 759 / 检出 4） |

**结论**：通用场景（VOC）可用，域差距数据集（dota 航拍、mot 密集）在 nano + CPU 下基本失效——与预期一致，GPU 上换大模型后才有精度意义；coco/voc 脚本未打印吞吐行（历史遗留手写 argparse），时间戳差估算全程 5-6s/50 图。

### 分割全量：4 数据集 × 50 图（yolo11n.pt + FastSAM-s.pt）

| 数据集 | mask mAP@0.5 | mIoU | mDice | 吞吐（img/s） | 备注 |
| --- | --- | --- | --- | --- | --- |
| d2sa | **0.8873** | **0.8966** | **0.9401** | **6.4** | 1 超类 150 实例；**box-prompted 无检测步**（GT bbox 直接作 prompt，上界性质） |
| coco_seg | 0.4917 | 0.5782 | 0.6403 | — | 59 类；两段式（检测 conf≥0.5 → box-prompt） |
| nuimages | 0.3469 | 0.7932 | 0.8780 | 2.9 | 6 类；bicycle/bus/motorcycle 小样本 |
| cityscapes | 0.0050 | 0.6367 | 0.7764 | 2.3 | 3 类；与 v0.2 历史值 0.007\* 同量级（FastSAM 域差距，需换 SAM3） |

**结论**：分割链路 CPU 全量可行（单集 ≤ ~80s）；域阶梯复现——通用（COCO）→ 驾驶（nuImages）→ Cityscapes（失效）→ D2SA（box-prompt 上界高）。cityscapes 低分与 v0.2 结论一致，非本轮回归。

## cityscapes 消融（2026-08-16，50 图，定位 mask mAP 根因）

三种模式二分定位：`--det-only` 隔离检测器（bbox mAP）、`--box-prompted` 隔离分割器（GT bbox 直接 prompt，检测零误差）、两段式 full pipeline（组合效应）。一键重跑：`bash auto2dlabel/benchmarks/run_cityscapes_ablation.sh`。

| 实验 | 模型 | mask/bbox mAP@0.5 | mIoU | Pred/GT（car） | 解读 |
| --- | --- | --- | --- | --- | --- |
| ①a det-only conf 0.3 | yolo11n.pt | bbox **0.0008** | — | 206/29 | 检测器 bbox 完全无效 |
| ①b det-only conf 0.5 | yolo11n.pt | bbox **0.0000** | — | 146/29 | 更高 conf 更差 |
| ② box-prompted | FastSAM-s.pt | mask 0.1010 | 0.6904 | 5/29 | FastSAM 大量丢框（47 GT 仅输出 9），切出的 mask 质量尚可 |
| ③ box-prompted | sam2_l.pt | mask **0.4657** | 0.6976 | 28/29 | SAM2 基本不丢框，分割器上限 ≈ 0.47 |
| ④ full prompt-conf 0 | yolo11n + FastSAM | mask 0.0077 | 0.4029 | 202/29 | 检测无效 → 噪声 prompt 全盘崩塌 |
| 基线 full conf≥0.5 | yolo11n + FastSAM | mask 0.0050 | 0.6367 | 241/29 | 历史值，与 ④ 同量级 |

**结论（根因二分）**：
1. **主因是检测步**：yolo11n 在 cityscapes 2048×1024 大图上的 bbox mAP≈0（0.0008）——YOLO resize 到 640 后小目标框位置漂移 + 域差距，检测框基本对不上 GT。分割步再强（③ SAM2 0.47）也救不回检测丢失的实例。
2. **次因是 FastSAM 丢框**：即使 GT bbox 直接 prompt（②），47 个 GT 实例仅输出 9 个 mask（输出与 GT 的 mask-IoU 重匹配后 <0.01 被丢弃）；SAM2 同样条件下输出 43 个。
3. **GPU 修正方向**：检测侧换 yolo26x/rtdetr + SAHI 切片（大图小目标），分割侧 box-prompted SAM2/SAM3；预期组合上限 ≈ ③ 的 0.47 并随检测质量向 bbox 上界逼近。

## cityscapes 域内权重（2026-08-17，域边界「唯一出路」落地）

**GT 解析修复（先行）**：`load_cityscapes_ground_truth` 原用 `inst_id % 1000` 解析类别——对 thing 类像素（label_id×1000+instance_idx，如 24000）得 0 → 全部跳过，**GT 近乎空**。修复为 `inst_id // 1000`。⚠️ 本表之前（含 v0.2 历史 0.007\*、CPU/GPU 复测 0.0050、消融矩阵）的 cityscapes 数字均系坏 GT 下测得，仅作定性参考；本节数字为修复后 GT。

**接入**：mmdet 官方 `mask_rcnn_r50_fpn_1x_cityscapes`（Box 40.9 / Mask 36.4，8 类）经 `tools/convert_mmdet_cityscapes_maskrcnn.py` 一次性键名转换 → torchvision v1 结构纯本地推理（自检+分割一步，`--seg-model maskrcnn_r50_cityscapes` 走 self-detect 模式）。转换坑：**mmdet fc_cls 背景在最后一维 → torchvision 背景在 index 0**，需行重排 [fg×8, bg]→[bg, fg×8]（初版漏此步，背景行被当作 bicycle：输出 100 框全 bicycle/score 1.0、mAP 0.0000；proposal 与 GT 的 IoU 分布验证 + 行重排后修复，回归测试见 `test_maskrcnn_cityscapes.py`）。

| 规模 | mask mAP@0.5 | mIoU | 吞吐 | 备注 |
| --- | --- | --- | --- | --- |
| 冒烟 2 图 | **0.7071** | 0.7156 | 1.7 s/img | person 0.8182 / car 0.7576 / bicycle 0.5455 |
| 全量 500 图（val 全集） | **0.5149** | **0.7552** | 110s，4.5 img/s | 7 类全检出：bus 0.7055 / car 0.6234 / train 0.5249 / person 0.4966 / bicycle 0.4192 / truck 0.4191 / motorcycle 0.4154 |

**结论**：全量 mAP 0.5149 vs 坏 GT 基线 0.0082——域内权重把 cityscapes 从「失效」拉到 7 类全检出（口径与 mmdet 官方 36.4 不同：官方为 8 类 COCO 协议 AP，本项目为 GT 推导 7 类名折叠 + mask IoU@0.5，不做逐数字对标）。消融结论「唯一出路是域微调权重」验证闭环：通用域（COCO）快速生成预标注 → 微调域内权重 → 回采难域。自检+分割一步到位的 self-detect 模式还免去了两段式检测步依赖。

## DOTA OBB 旋转框（2026-08-16）

Task1 旋转框标注（15 类全评），`rotate_iou`（shapely 四边形求交）+ `evaluate_per_class(iou_fn=rotate_iou)`，模型 `yolo11n-obb.pt`（DOTAv1 训练权重，类名与 GT 原生对齐，无 COCO 映射损耗）。难度 difficulty=1 按官方惯例剔除（`--include-difficult` 可放开）。

| 规模 | mAP@0.5 | large-vehicle | small-vehicle | 吞吐 | 备注 |
| --- | --- | --- | --- | --- | --- |
| 冒烟 2 图 | 0.6962 | 0.7583 | 0.6340 | 1.3 img/s | 对比 HBB 版 dota 同场景 0.03 |
| 50 图 | **0.7047** | **0.8001** | **0.4992** | 1.3 img/s | 13 类出现；plane 0.9046 / tennis-court 0.9091 / ship 0.5230（762 GT 检出 455） |

**结论**：OBB 链路闭环（GT 解析 → 旋转框推理 → rotate_iou 评估 → 报告）；类名映射修复后 yolo-obb 权重名（small vehicle 空格）与 GT 连字符名（small-vehicle）经 `normalize_dota_class` 统一。同数据集 OBB 路径（0.70）对比 HBB 路径（0.03）差距 20 倍——旋转框场景必须走 OBB 模型 + rotate_iou，HBB 评估仅作域差距参考。GPU 全量：`bash auto2dlabel/benchmarks/run_benchmarks.sh obb --extra "--model yolo26x-obb.pt --max-images 0"`（458 图）。

## GPU 复测（2026-08-17，RTX 4090 24GB，实测）

> benchmarks 脚本无 CUDA 硬编码（ultralytics 自动选设备），数据与权重同盘迁移后脚本零改动——**一键全跑实测成立**。
>
> **一键全跑**：`bash auto2dlabel/benchmarks/run_gpu_all.sh`（4 步串行：环境自检 → 全量 11 集 → OBB 458 全量 → cityscapes 消融；单步失败不阻断，无 GPU 环境拒绝启动）。GPU 权重预热：`bash auto2dlabel/weights/download_weights.sh`（含 obb 条目）。

### 全量 11 集（50 图/集，GPU yolo26x vs CPU yolo11n）

| 数据集 | CPU yolo11n | GPU yolo26x | 解读 |
| --- | --- | --- | --- |
| coco2017 | 0.2632 | **0.4668** | +0.20，最大受益 |
| voc2007 | 0.5198 | **0.5909** | +0.07 |
| kitti | 0.2610 | **0.4081** | +0.15 |
| dota（航拍） | 0.0303 | 0.0660 | 仍失效（域差距，大模型无效） |
| mot（密集行人） | 0.1725 | 0.0909 | **反降**（11-point AP 量化阶跃，见下「mot 反降调查」） |
| coco_seg | 0.4917 | 0.4917 | 零变化（瓶颈在分割链路，SAM2+prompt-conf 0.3 新默认 0.6368，见下） |
| cityscapes | 0.0050 | 0.0050 | 零变化（检测步域边界，见下） |
| nuimages | 0.3469 | 0.3469 | 零变化（同上） |
| d2sa | 0.8873 | 0.8873 | box-prompted 上界（无检测步） |
| imagenet100 | 78.0/94.6 | 74.0/94.0 | top-1 略降（GPU 版 torch 实现差异，协议不变） |

**吞吐**：全量 11 集 GPU 上约 4 分钟（CPU 上 yolo11n 需 30+ 分钟）——迁移 + 一键全跑流程成立。

### mot 反降调查（2026-08-17）

现象：yolo26x mAP 0.0909 vs yolo11n 0.1725（-47%）。同 50 图、同 conf 0.3、同 NMS iou 0.5 逐框重跑诊断（GT 4231 人，两轮采样一致）：

| 指标 | yolo11n | yolo26x | 解读 |
| --- | --- | --- | --- |
| 检出框 / TP / FP | 577 / 520 / 57 | 476 / 412 / 64 | 26x 检出少 17%，但 FP 未增 |
| Prec | 0.9012 | 0.8655 | 基本持平（不是「precision 暴跌」） |
| Recall | **0.1229** | **0.0974** | -3pp，但恰好跌破 0.1 |
| TP conf 中位 / P75 | 0.498 / 0.683 | **0.556 / 0.784** | 26x 正确框置信度反而更高 |
| recall ≤ 0.097 区间 max prec | 0.9867 | **1.0000** | 共同区间内 26x 质量更好 |

**根因：11-point AP 的量化阶跃，不是模型质量断崖。** recall 0.1229 → 0.0974 恰好跨越 t=0.1 插值点：yolo11n 贡献 2 个点（0.0897+0.0828=0.1725），yolo26x 只剩 1 个点（0.0909）→ AP 腰斩。在共同 recall 区间内 yolo26x 的 PR 曲线反而更优；「反降」的实质是模型在 MOT 密集域的召回略低（-3pp，检测头域差距）触发了评测指标的网格敏感性。NMS 非主因（26x 在 iou 0.5/0.0 下框数同为 476）。

**评测启示**：域差距大数据集 recall 普遍 <0.2，11-point AP 对 ±0.1 网格敏感——低 recall 场景的 mAP 横向对比应附 recall 或 PR 区间 max-prec 作辅助指标。

### cityscapes 消融（GPU 矩阵 + SAHI 扩展）

| 实验 | bbox/mask mAP@0.5 | 排除的假设 |
| --- | --- | --- |
| det-only yolo11n（CPU 基线） | 0.0008 | — |
| det-only yolo26x | 0.0060 | ❌ 模型规模无效 |
| det-only yolo26x + SAHI | 0.0066 | ❌ 尺度切片无效（recall 0.10→0.28 但 FP 洪水 527/29） |
| det-only rtdetr-x + SAHI | 0.0119 | ❌ 模型架构无效 |
| box-prompted FastSAM-s | 0.1010 | FastSAM 丢框（CPU 复现） |
| box-prompted sam2_l | **0.4657** | 分割器上限（CPU 复现） |
| box-prompted sam3 | 0.0175 | SAM3 封装无 box-prompt 语义（bboxes 仅取 label 作文本 prompt）→ 实为文本检测，同失效 |
| full sam3（det yolo26x） | 0.0014 | SAM3 文本检测在 cityscapes 失效 |

**最终结论（域边界定性）**：cityscapes 检测步失效不是模型规模（yolo26x）、尺度（SAHI）、置信度（0.5）、架构（rtdetr）任一单因素——是 **COCO 预训练模型在 30~50px 小目标域的定位极限**（框 IoU 到不了 0.5）。唯一出路是域微调权重；AutoLabel 的价值闭环恰好在这里：在通用域（COCO/VOC 可用）快速生成预标注 → 微调出域内权重 → 回采难域。分割侧升级路径 = 两段式换 box-prompted SAM2（cityscapes 上限 0.4657）。

> **通俗解释（2026-08-18 确认记录）**：通用模型在 cityscapes 上失效 = **检测视角 + 训练数据分布**双重差异。模型只在 COCO 类日常照片上训练——目标是画面主体，大而清晰（几十到几百 px）；cityscapes 是车载摄像头视角，目标小而密集（30~50px、远距、互相遮挡）。规模/SAHI/架构均无效，说明这不是「能力不够」而是「没学过这类场景」；域内权重（在 cityscapes 上训练过）把 mAP 从 0.0082 拉到 0.5149（62.8×）即是最直接的证明。

### coco_seg 分割器消融（2026-08-17，含两段式与 prompt-conf）

前提：全量 11 集中 coco_seg 0.4917 在 CPU/GPU 完全一致 + mot 反降调查证 yolo26x 检测步在通用域无问题 → 瓶颈应在分割链路。为此给 `coco_seg_benchmark.py` 加 `--box-prompted`（GT bbox 直接 prompt，隔离分割器）与 `--prompt-conf`（两段式 prompt 过滤阈值），实测：

| 配置 | mask mAP@0.5 | mIoU | Prec/Recall | 解读 |
| --- | --- | --- | --- | --- |
| 两段式 yolo26x + FastSAM-s，prompt-conf 0.5 | 0.4917 | 0.5782 | — | 原默认链路 |
| 两段式 yolo26x + sam2_l，prompt-conf 0.5 | 0.5649 | 0.6526 | 0.7043 / 0.5728 | 换分割器 +0.07 |
| **两段式 yolo26x + sam2_l，prompt-conf 0.3** | **0.6368** | **0.6902** | 0.6766 / 0.6543 | 降 prompt 阈值 recall +8pp |
| box-prompted sam2_l（2 图冒烟） | 1.0000 | 0.8019 | — | 逐框 prompt 无丢框 |
| **box-prompted sam2_l（50 图）** | **0.9278** | **0.8212** | 0.9565 / 0.9536 | **分割器上限**（无检测步） |

**结论（三因素分解）**：0.9278（上限）→ 0.6368（新默认两段式）的落差 0.29 全部来自**检测步**（prompt-conf 过滤 + 检测召回），分割器本身只占 0.07（FastSAM→SAM2）——检测步才是两段式链路的主要瓶颈（与 cityscapes 消融结论同构）。SAM2 短板在 dining table 0.43 / fork 0.50 / banana 0.55——大目标边界 + 小细长物体（与「SAM 系列大目标边界粗糙」的红线一致）。超参数调整：默认分割模型 → `sam2_l.pt`、prompt-conf → 0.3（见 `milestone/v0.3.md`）。

### DOTA OBB（GPU 补测）

| 规模 | 模型 | mAP@0.5 | 备注 |
| --- | --- | --- | --- |
| 50 图同场景 | yolo11n-obb（CPU） | 0.7047 | 上节 |
| 50 图同场景 | yolo26x-obb（GPU） | **0.7997** | +0.095，x 模型增量 |
| 458 图全量 | yolo26x-obb（GPU） | **0.6161** | 全量含难图；bridge 0.2446 / roundabout 0.4197 拉低；tennis-court 0.9029 / large-vehicle 0.7954 |

### 基础设施修复（本复测暴露，均带回归测试）

1. `run_benchmarks.sh` `--extra` 透传缺 `--extra` 标志 → argparse 报 unrecognized（修复 + obb 2 图实测回归）
2. ultralytics SAM3 与 PyPI openai-clip tokenizer 接口不兼容（`SimpleTokenizer` 不可调用）→ `segmentation.py::_patch_clip_tokenizer`（3 测试）
3. `_gt_bbox_view` 缺 `file_name` 键（SAHI KeyError）→ 修复 + cityscapes det-only 接入 `--sahi`（回归测试）
4. ghfast 镜像截断权重（68MB/121MB 损坏）→ 换 gh-proxy.com + zip 完整性校验
5. 新服务器环境：`pip install openai-clip` + `setuptools<81`（clip 依赖 pkg_resources）

## 全量可视化（2026-08-17，11 数据集 × 全量，RTX 4090）

`bash benchmarks/run_visualization.sh` 串行跑 11 集（`--max-images 0` 全量），产物镜像原相对路径写入项目同级 `Visualization/<数据集名>/`（已存在跳过，断点续跑）。渲染分派：检测/OBB 画框（quad 旋转框折线）、分割画 mask 轮廓（d2sa polygon 免解码直填）、分类画顶部文本条（GT + top-5）。

| 数据集 | 任务 / 模型 | 渲染数 | 产物大小 | 备注 |
| --- | --- | --- | --- | --- |
| coco2017 | 检测 yolo26x.pt | 4952（2 跳过） | 666M | |
| voc2007 | 检测 yolo26x.pt | 4952 | 444M | |
| kitti | 检测 yolo26x.pt | 300 | 230M | |
| dota | 检测 yolo26x.pt | 458 | 4.0G | 大图 PNG |
| mot | 检测 yolo26x.pt | 14247 | 5.1G | 视频序列逐帧 |
| dota_obb | OBB yolo11n-obb.pt | 458 | 4.0G | 旋转框折线 |
| coco_seg | 分割 sam2_l.pt | 4952 | 598M | |
| cityscapes | maskrcnn_r50_cityscapes | 500 | 1.3G | 域内权重 |
| nuimages | 分割 sam2_l.pt | 48 | 17M | val 集 |
| d2sa | box-prompted sam2_l.pt | 3600（10 跳过） | 1.4G | polygon 直填 |
| imagenet100 | 分类 resnet18 | 5000 | 481M | 文本条 |

合计约 **4.0 万张 / 18G**。耗时：10 集串行 05:12–07:55 ≈ 2.7h（含中途抽检；mot 46min、imagenet100 55min 为主），d2sa 单独重跑约 20min。

### 渲染正确性抽查

- **dota**：对全部 458 张产物做像素级 diff（原图 vs 产物），几乎全部有渲染痕迹（数千~数十万非零 diff 像素）；初查 2 张 diff=0 经核实为 0 检测框图（不是 bug）
- **d2sa polygon 路径**：冒烟 10 图每张约 68 万 diff 像素（密集 mask 填充，符合预期）
- mot / cityscapes / coco_seg 各抽查：bbox / mask 叠加正常

### d2sa OOM 修复记录（首次全量 exit 137）

首次全量运行 d2sa 被 SIGKILL：GT 15654 实例 × 1920×1440 bool mask 全量解码 ≈ 43GB，叠加预测 mask ≈ 43GB，超容器 cgroup 62GB 限制（重试时 411/3600 张处已达 51GB，9.5s/img）。修复：GT 存 COCO RLE + 预测存 polygon（压缩存储，峰值 2.5GB），评测经 `_decode_gt_view`/`_decode_pred_view` 按类解码视图（峰值 = 单类实例量级）。注意 D2SA 的 RLE counts 为特殊字符串编码，`pycocotools.mask.decode` 可解（annToMask 同路径），`frPyObjects` 不可用。修复后全量 3600 张 8.6 img/s，**mask mAP@0.5 0.9073 / mIoU 0.9081 / mDice 0.9484**（与 10 图冒烟一致）。

### chat 批量推理冒烟

4 图 × batch_size=2：日志正确输出 `批量: batch_size=2 num_workers=2`，检测结果逐图归位。`batch_size`/`num_workers` 三档来源（CLI 显式 > 交互询问 > GPU 显存推荐）与 `*_batch` 分派（ultralytics/torchvision/CLIP 原生 batch，SAM 系列逐图回退）见 `milestone/v0.3.md` 同节。

## 批量推理动态调优（2026-08-18，RTX 4090）

**需求**：旧实现静态 batch=2/workers=2（浪费 4090 显存）；benchmark 全部逐图串行（GPU 空转）。改为模型加载后动态实测最大 batch 并接入 11 个 benchmark。

### 吞吐实测（coco 100 图 yolo26x）

| 模式 | batch | 耗时 | 吞吐 | mAP@0.5 |
| --- | --- | --- | --- | --- |
| 逐图（--batch 1） | 1 | 3.6s | 27.5 img/s | 0.6145 |
| 动态实测（默认） | **64**（实测档位） | 2.0s | **50.6 img/s（1.84×）** | 0.6145 |

mAP 完全一致；`resolve_batch_params` 实测档位打印 `批量推理: batch_size=64 num_workers=4`。coco 冒烟（batch 2 / 1 / 自动）此前已验证 mAP 均 0.8333。

### OBB 批量 parity 双根因排查（dota_obb）

冒烟发现 dota_obb 批量与逐图 mAP 不一致（10 图 0.8191 vs 0.8397），逐层定位出两个独立根因：

1. **TF32**（Ampere+ 19-bit 尾数加速）：同图×4 批量 vs 单图即差 20+ 框。逐层 hook 证明首层 conv 输出已发散（8e-5），且与注意力无关（Attention 恒等后差异仍在）；cudnn.deterministic 无效、double 对照消失 → 判定 TF32 kernel 选择。关闭 `cudnn.allow_tf32`/`matmul.allow_tf32` 后 raw 差从 9.9e+2 降至 6.1e-5、结果差异 0。
2. **rect 矩形 letterbox**：同图一致但混合尺寸批量仍差（54 vs 56 框）。ultralytics `_predict` 默认 `rect=True`——单图 `same_shapes=True` 走矩形 letterbox（928×1024 最小 padding），批量混合尺寸 `same_shapes=False` 自动退化正方形（1024×1024），两种预处理结果不同。

**修复**：`tools/device.disable_tf32()`（挂在 `resolve_batch_params`/`print_device` 两入口）+ 全部 6 处 ultralytics 调用显式 `rect=False`（detection detect/detect_batch/SAHI、obb detect/detect_obb_batch、segmentation SAM/FastSAM）。修复后：

- 混合尺寸 4 图批量 vs 逐图：**总差异 0**（56/96/158/0 框逐位一致）
- dota_obb 10 图 batch 1 vs 2：mAP **完全一致 0.8397**（修复前差 0.02）

代价：单图 letterbox 统一为正方形（历史单图结果有变）+ TF32 关闭吞吐 ~10%（被批量加速覆盖）。回归：`test_disable_tf32_sets_both_flags` / `test_resolve_batch_params_disables_tf32`。

### 分割/生成路径冒烟

- coco_seg 两段式 sam2_l（--batch 2，4 图）：检测步批量生效、分割步逐图，mAP 0.7500，无降级
- coco_seg maskrcnn generate_batch（--batch 4，4 图）：0.0889（模型固有水平，批量路径跑通）
- chat 冒烟：单图 → batch=1；4 图目录 → 实测 batch_size=64 num_workers=4

## 待 GPU 复测清单（2026-08-18，CPU 环境整理，GPU 可用后执行）

原「留 GPU」标记闭环核查：① 大模型精度对比（§CPU 可行性验证）→ ✅ GPU 复测全量 11 集；② GPU 修正方向（§cityscapes 消融）→ ✅ GPU 矩阵 + SAHI 扩展；③ OBB 458 全量（§DOTA OBB）→ ✅ 0.6161。以下为尚未闭环项：

| # | 待测项 | 原因 | 验收 |
| --- | --- | --- | --- |
| 1 | cityscapes 消融矩阵按修复后 GT 重跑 | 消融表（0.0008/0.1010/0.4657/0.0077）均系坏 GT（`% 1000`）下测得，仅定性参考；GT 修复后只重测了域内权重全量 | `run_cityscapes_ablation.sh` 重跑，矩阵数字更新（定性结论「唯一出路域微调」预期不变） |
| 2 | 零样本分类路（CLIP/SigLIP）实测 | 权重未下载，协议已固定但该路从未跑过 | 下载 `weights/hf` 权重 → imagenet100 100 类空间 top-1/top-5 成行（与监督路不可横向比较） |
| 3 | SAM3 box-prompt 语义补齐后重测 | 现封装 bboxes 仅取 label 作文本 prompt（实为文本检测），box-prompted sam3 0.0175 不代表真实上限 | `segmentation.py` SAM3 封装补 box prompt → 重测 cityscapes box-prompted sam3 行（对照 SAM2 0.4657） |
| 4 | 11 benchmark 批量接入全量验证 | 批量接入后仅实测 coco 100 图（1.84×）；其余 10 集全量批量吞吐/parity 未跑 | `run_gpu_all.sh` 全量（默认自动实测档位）：各集 mAP 与逐图基线一致 + 吞吐提升 |
| 5 | nuimages 新默认链路重测 | 0.3469 为旧链路（FastSAM）；新默认 sam2_l + prompt-conf 0.3 仅实测 coco_seg（0.6368） | nuimages 两段式新默认重测；预期检测步域差距仍在（与 cityscapes 同构） |
| 6 | imagenet100 GPU top-1 下降调查（低优先级） | 74.0 vs CPU 78.0 未深究，同权重跨设备不应差 4pp | 排查 GPU 版 torch resize/预处理差异；结论记入本文 |

## 评测协议注记

- **分类双协议**：监督路径（resnet18 等）在完整 ImageNet1K **1000 类空间**取 top-K（官方协议）；零样本路径（CLIP/SigLIP）在 **100 类候选空间**取 top-K——两路 top-1/top-5 **不可直接横向比较**。本轮未跑零样本路（权重未下载），协议已由 `classification_benchmark.py` docstring 与 `docs/Benchmark_plan.md` 固定。
- **模型口径**：全量均以轻量模型覆盖（CPU 可行性），不代表管线在 GPU 上的精度上限；`run_all` 注册表默认模型（yolo26x.pt）不变，CPU 全组运行必须 `--extra "--model yolo11n.pt"`。
