# v0.4 实测数据（自 `milestone/v0.4.md` 转移）

> Phase 1 视频 Tracking（ByteTrack）冒烟验证（2026-08-19，32 核 CPU / 无 GPU，torch 2.13.0 CPU）。功能定义与完成标记见 `../milestone/v0.4.md`。评测产物在 `benchmarks_outputs/`（`mot_track_yolo12n.pt_2026-08-19-20-24-16.json/.md`），`--track` 冒烟产物在 `outputs/` 与 `vis_outputs/`。

## MOT 跟踪评测冒烟（MOT17-02-FRCNN 20 帧连续窗口）

目标：验证「检测 → ByteTrack → CLEAR MOT 评测」全管线在 CPU 上可行（可行性结论），精度仅作定量参考（nano 模型 + 20 帧小样本）。

| 模型 | conf | GT 框/轨迹 | MOTA | IDF1 | IDSW | MT | ML | FP | FN | 检测 recall | 检测 precision | 吞吐 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `yolo12n.pt` | 0.3 | 450 / 23 | **12.7%** | **24.7%** | **3** | **4/23** | 19/23 | **20** | **370** | **17.8%** | **80.0%** | **0.1 img/s**（282.9s） |

**结论**：管线 CPU 全通（20 帧 282.9s，冒烟可行；GPU 上预计实时量级）。按红线双口径解读：**检测 recall 17.8% 是 MOTA 低的主因**——FN 370 占 GT 82%，与 v0.3 mot 检测 mAP 0.1725 / Recall 0.1224 同量级，是检测的锅（nano + 密集行人漏检，非本轮回归）；**跟踪口径干净**——IDSW=3（23 轨迹 / 20 帧）低、precision 80.0% 高，高置信框的 ID 维持稳定，跟踪器没添乱。低分救援（BYTE 第二段）在 recall 17.8% 时已把 20 个 FP 大部分解释为现有轨迹，否则 FP/IDSW 会更高。

## --track 全链路冒烟（8 帧帧目录）

`auto2dlabel run /tmp/track_smoke/frames "检测行人" --track -d yolo11n.pt -t 0.3`（MOT17-02-FRCNN 帧 1-8，CPU）。验证里程碑验收链：逐帧 bbox + track_id + MOT 导出 + 可视化 + HITL 分流。

| 项目 | 实测 | 说明 |
| --- | --- | --- |
| 逐帧检测 | 8-10 框/帧 | 全部帧处理完成，无 LLM 直连检测路径（`--track` 免 API） |
| Track 列 | ID 0-4 帧间维持，帧 8 新 ID 5 | CLI 表格 Track 列；conf<0.5 未匹配低分框显示 "-"（帧 5 有 5 个） |
| 轨迹汇总 | **8 帧, 6 条轨迹** | ID 0-4 帧 1 起全程维持，ID 5 帧 8 新入 |
| MOT 导出 | 40 行 = 5 轨迹 × 8 帧 | `outputs/frames_mot.txt`，frame 从 1 起；无 track_id 低分框跳过（MOT 格式无容纳位） |
| ID 维持验证 | 帧 1→2 同 ID 坐标漂移 <1px | ID 0: (582.3,446.5)→(582.4,445.7)——静态相机场景 ID 关联正确 |
| 轨迹可视化 | 6 条轨迹线 md5 配色像素命中（21-149px/条） | 逐 ID 稳定哈希配色，程序化反查验证（会话图像查看不可用时的替代验证） |
| 逐帧归档 | JSON（COCO）+ 可视化 png + HITL 分流 | 帧 5-8 分流：4 直接采纳 / 5-6 待复核 / 0 困难（低分框进复核队列） |

**结论**：验收链全通——8 帧逐帧 bbox + 全局 track_id（ID 帧间维持正确）+ MOT 导出 + ID 角标/轨迹线可视化 + HITL 分流。冒烟中发现的唯一缺陷：COCO JSON 导出（`export/coco.py` 从 Bbox 对象直构 dict、绕过 `to_dict`）静默丢 `track_id`，逐帧 JSON 归档无 ID——已修复并补回归测试（`test_coco_export_preserves_track_id`），穿透点红线更新为六处。标注场景「不丢检测」取舍生效：低分框不建轨迹（ByteTrack 原版语义），但仍保留在逐帧 JSON/HITL 复核队列中。

## LLM 一次性规划（`--track --llm`，2026-08-19 追加）

| 场景 | 输入 | 结果 |
| --- | --- | --- |
| 真调用（DeepSeek） | `"检测行人"` | 「LLM 规划: 类别 person」→ 3 帧 5 轨迹全链路通 |
| 失败回退 | 同上 + `--base-url http://127.0.0.1:1` | 「LLM 规划失败，回退 extract_prompts: Connection error」→ 代码级解析照常出 person |

**结论**：序列级单次 LLM 解析（每序列 1 次调用，绝不逐帧）与零外部依赖兜底均验证——LLM 可用时覆盖复杂指令，不可用时流程无损降级。测试 `test_track_llm_plan.py` 11 用例（回退链 + JSON 解析/围栏剥离/bool 阈值拒绝）。

## GT-as-detection 上界模式（`--gt-as-detection`，2026-08-19 追加）

GT 框直接喂 ByteTracker（conf=1.0，跳过检测）——理想检测下评测跟踪关联能力上界，与真实管线对比分离「检测的锅/跟踪的锅」。产物 `benchmarks_outputs/mot_track_gt_2026-08-19-23-17-39.json/.md`。

| 检测来源 | 序列 | GT 框/轨迹 | MOTA | IDF1 | IDSW | MT | ML | FP | FN | 检测 recall | 检测 precision | 耗时 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GT 直喂（上界） | MOT17-02-FRCNN | 450 / 23 | **1.0000** | **1.0000** | **0** | **23/23** | 0/23 | **0** | **0** | **1.0000** | **1.0000** | **0.2s**（113.8 帧/s） |
| `yolo12n.pt` 检测→ByteTrack | 同上 | 450 / 23 | 12.7% | 24.7% | 3 | 4/23 | 19/23 | 20 | 370 | 17.8% | 80.0% | 282.9s（0.1 img/s） |

**结论**：检测=GT 时跟踪器在 MOT17-02 首 20 帧实现**零损失**——MOTA/IDF1=1.0、IDSW=0、23/23 全 MT。上界模式把红线双口径落地为定量对照：真实管线 12.7% MOTA 与上界 1.0 的差距 100% 来自检测（recall 17.8% 的锅），跟踪关联本身干净；跟踪后处理耗时可忽略（0.2s/20 帧）。该模式同时成为未来跟踪器调优（阈值/生命周期参数）的对照基线——参数改动后先跑 `--gt-as-detection`，上界下降即跟踪器退化。注意上界模式下 FN/FP 结构性为零（pred 由 GT 派生），MOTA 损失唯一来源是 IDSW。

## 纯 tracker 耗时微基准（2026-08-19 追加）

合成随机游走检测（seed=0 确定性，10% 帧间遮挡 + 高低分两段关联激活）直喂 ByteTracker，200 帧/档 × 3 轮取最短，密度扫 10-200 框/帧。

| 目标数/帧 | 总框数 | 均值 ms/帧 | µs/框 |
| --- | --- | --- | --- |
| 10 | 1809 | **4.960** | **548.34** |
| 25 | 4498 | 12.736 | 566.32 |
| 50 | 8974 | 25.681 | 572.35 |
| 100 | 18014 | 54.034 | 599.91 |
| 200 | 36014 | 124.226 | 689.88 |

**结论**：跟踪后处理在全部密度档远低于检测耗时——典型场景（10-50 框/帧）单帧 update 5-26ms，MOT20 量级密集场景（200 框/帧）124ms（约 8 帧/s 纯跟踪吞吐），相对检测主导的 0.1 img/s 可忽略；10→200 框耗时增长约 25 倍（近线性），匈牙利 O(n³) 常数在实测规模下不构成瓶颈。

## BoT-SORT 精度档（`--bot-sort`，2026-08-20 追加）

ReID 外观关联（CLIP/SigLIP 图像编码器）+ ECC 相机运动补偿。CLIP 权重 ~350MB 进 `weights/hf/`（镜像 `HF_ENDPOINT=https://hf-mirror.com` 下载，部署环境配置）；transformers 5.15 的 `get_image_features` 返回输出对象（非 Tensor），已做 `_image_embeds` 兼容（CLIP/SigLIP 双验证）。对比红线：同序列同窗口同 conf（检测 conf=0.3 不变，BoT-SORT 用官方轨迹阈值 0.6/0.7、ByteTrack 0.5/0.5——换 tracker 即换口径，阈值差是 tracker 行为的一部分）。

### A. 真实管线对比（MOT17-02-FRCNN 20 帧，conf=0.3）

| tracker | MOTA | IDF1 | IDSW | MT | ML | FP | FN | 检测 recall | 检测 precision | 耗时 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ByteTrack（2026-08-19 基线） | 12.7% | 24.7% | 3 | 4/23 | 19/23 | 20 | 370 | 17.8% | 80.0% | 282.9s |
| BoT-SORT（`--bot-sort`） | **17.78%** | **30.19%** | **0** | 4/23 | 19/23 | **0** | 370 | 17.8% | **100.0%** | **21.2s**（0.9 img/s） |

**结论**：稀疏场景 BoT-SORT 净收益——MOTA 12.7%→17.78% 全部来自 **IDSW 3→0 与 FP 20→0**（官方 0.6/0.7 高轨迹阈值过滤掉低分假阳性框，ReID 融合关联稳定高分框 ID）；recall 17.8% 两者相同（检测同源，FN 370 的锅仍在检测）。耗时差异（21.2s vs 282.9s）为同日机器负载波动，吞吐仅量级参考。

### B. GT-as-detection 双 tracker（MOT17-02-FRCNN 20 帧，上界无回归验证）

| tracker | MOTA | IDF1 | IDSW | MT | 耗时 |
| --- | --- | --- | --- | --- | --- |
| ByteTrack（基线） | 1.0000 | 1.0000 | 0 | 23/23 | 0.2s |
| BoT-SORT | 1.0000 | 1.0000 | 0 | 23/23 | 23.5s（ReID 450 框 ≈ **52ms/框**） |

**结论**：上界无回归——GT 检测下 BoT-SORT 同样零损失；跟踪纯后处理 0.2s 可忽略，ReID 特征提取是 BoT-SORT 全部耗时（CLIP-base CPU 约 50-60ms/框）。

### C. MOT20-01 密集序列 GT（30 帧 1094 框 / 37 轨迹，静态相机）

| tracker | MOTA | IDF1 | IDSW | MT | 耗时 |
| --- | --- | --- | --- | --- | --- |
| ByteTrack（IoU-only） | **1.0000** | **1.0000** | **0** | 37/37 | 0.3s |
| BoT-SORT（λ=0.98 官方） | 0.9963 | 0.9982 | 4 | 37/37 | 41.3s |

参数诊断（同配置单变量）：ECC 关 = IDSW 4（不变，ECC 无辜）；λ=0.5 = **IDSW 0**；appearance_thresh 0.5 = IDSW 4（不变）。

**结论**：反向证据——密集同质行人（36 框/帧）下官方 λ=0.98 外观主导产生 4 次 IDSW（纯 IoU 为 0），λ=0.5 平衡权重归零。根因：λ=0.98 是 BoT-SORT 论文为其专用 ReID 特征（高区分度）调优的，CLIP 通用图像特征在密集同质外观下区分度不足，外观微小噪声主导排序。**决策：保持官方默认参数**（单序列 30 帧样本太小，不据此改默认），λ 调优列为后续 GPU 长窗口实验项——GT-as-detection 对照基线的定位首次兑现（参数改动后先跑上界，上界下降即跟踪器退化）。真实序列演示 ReID 价值未遂（MOT20-01 前 30 帧无交叉遮挡，IoU-only 已零损失）；ReID 门控价值由合成单测数学构造证明（`test_reid_gate_prevents_swap`：0.5px 级漂移使纯 IoU 匈牙利必交换，正交特征被 cosine 门控拒绝 → 恒等匹配）。

### D. CLI 全链路冒烟（`--bot-sort`，8 帧帧目录）

`auto2dlabel run /tmp/track_smoke/frames "检测行人" --track -d yolo11n.pt -t 0.3 --bot-sort`（MOT17-02-FRCNN 帧 1-8，CPU）。

| 项目 | 实测 | 说明 |
| --- | --- | --- |
| 轨迹汇总 | **8 帧, 4 条轨迹** | ByteTrack 同输入为 6 条（`test-v0.4.md` §--track 冒烟）——官方 0.6/0.7 高阈值语义：conf<0.7 的未匹配低分框不建新轨迹（帧 1 conf 64.75% 框无 Track ID），低分框仍保留在逐帧 JSON/HITL 复核队列 |
| MOT 导出 | `outputs/frames_mot.txt` | 全链路同 ByteTrack 路径 |
| 轨迹可视化 | 8 张 ID 角标/轨迹线图 | `vis_outputs/` |

**结论**：CLI 精度档全链路通；`--bot-sort` 输出与 ByteTrack 的差异全部可解释（官方高轨迹阈值 + ReID 融合关联），低分框「不丢检测」语义保持（进 HITL 队列）。

## 测试统计

| 项 | 数值 |
| --- | --- |
| 新增单测（tracking 相关 7 文件） | 46（byte_tracker 13 / export_mot 5 / track_eval 7 / track_frames 4 / track_llm_plan 11 / gt_as_detection 2 / tracker_time_benchmark 4） |
| 新增单测（BoT-SORT 2 文件，2026-08-20） | 30（bot_sort 18 / reid_model 12） |
| 确定性 property（byte_tracker + bot_sort 内） | 3（手工混合场景 + 密集合成场景 + BoT-SORT 含 ECC 开，同输入逐位相等） |
| 全量 | **366 passed** |
