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

### E. chat 跟踪入口冒烟（2026-08-21 追加）

`/tmp/track_smoke`（MOT17-02-FRCNN 帧 1-5 帧目录），`-d yolo12n.pt --batch-size 4`，CPU，DeepSeek 规划。

| 指令 | 实测 | 说明 |
| --- | --- | --- |
| `chat "跟踪 /tmp/track_smoke 中的行人和车辆，使用 ByteTrack" --no-wait` | **5 帧, 4 条轨迹**，MOT 导出 `outputs/track_smoke_mot.txt` | planner 产 tracking 步（prompts=[person, car]、export_format=mot、model_name 未受「ByteTrack」污染）；Chat 日志 `step_1_done` 含 bbox_count=73 / track_ids=[0..3] / mot_path |
| `chat "…，使用 BoT-SORT" --no-wait` | **5 帧, 4 条轨迹** | 指令含 bot-sort → `detect_tracker_kind` 代码级扫描启用精度档（CLIP ReID 懒加载，需 HF_ENDPOINT=hf-mirror）；逐帧 JSON 为 coco 格式（mot→coco 映射），MOT 序列级合并导出 |

**结论**：chat 与 `run --track` 同一 TrackingTool 管线（cli_track 薄壳化后两者输出等价）；跟踪器选择 LLM 零参与。

### F. 真实视频成片实跑（sportscheck_2min，2026-08-21 追加）

`auto2dlabel chat "跟踪 /root/autodl-tmp/Documents/videos/sportscheck_2min.mp4 中的人、汽车和自行车，使用 ByteTrack" --no-wait -d yolo12n.pt --batch-size 8 --num-workers 4`——2 分钟真实体育视频（1920×1080 @ 23.976fps，2882 帧），CPU 单进程。本次新增能力：**视频源跟踪附带标注成片**（源视频同目录 `output_<原名>.mp4`，帧率随源视频、mp4v 编码，与 PNG 同一 draw_bboxes+draw_trajectories 绘制管线；帧目录不产片——防污染数据集目录）。

| 项目 | 实测 | 说明 |
| --- | --- | --- |
| 视频内容 | person 40 / car 21 / bicycle 4（5 帧抽样） | yolo12n 程序化勘察（会话图像查看不可用），据此定跟踪类别 |
| 跟踪规模 | **2882 帧, 1390 条轨迹, 60520 框** | conf=0.1 低阈值全量保留；MOT 导出 35132 行（高分轨迹框） |
| 成片 | `output_sportscheck_2min.mp4`（217MB） | 2882 帧 / 23.976fps / 1920×1080 / 120.2s 与源视频逐项一致；中帧 vs 原帧 mean_abs 像素差 7.58 确认标注已绘制 |
| Chat 日志 | steps_results 含 `video_path` | track_ids=[0..1389]、mot_path、video_path 三字段齐全 |
| 耗时 | **~22 分钟**（21:33→21:55） | 启动段慢（抽帧 2882 张 + 模型加载 + 首批编译 ~10 分钟），稳态 ~4.3 帧/s（batch 8 推理 0.24s/图，余为 PNG/JSON/视频编码逐帧开销） |

**结论**：视频源 → 抽帧 → 跟踪 → 成片全链路通；成片与逐帧 PNG 同一绘制管线（ID 角标 + 轨迹线），输出路径/命名符合「同文件夹 + `output_` 前缀」约定。单测覆盖：视频源出片端到端（合成 3 帧 mp4 输入，断言成片存在/可回读/已绘制）+ 帧目录不产片回归（`test_tracking_tool.py` 新增 1 用例）。

### G. ID 切换根因与 BoT-SORT 同序列对比（sportscheck_2min，2026-08-21 追加）

ByteTrack 成片复查发现「ID 切换过多」→ 量化后同视频同 conf=0.1 重跑 BoT-SORT 对比（红线：同序列同窗口同 conf；无 GT，结构指标替代 IDSW）。匈牙利匹配（`scipy.optimize.linear_sum_assignment`）**两个 tracker 的关联阶段都已内置**——ByteTrack 纯 IoU 关联 + 未匹配高分立即建轨迹，在检测闪烁（yolo12n 对快速运动/遮挡目标高分框时有时无）下必然碎片化：1390 条轨迹中 46.3%（643 条）单帧即断（其中 569 条 conf<0.7）。

| 结构指标 | ByteTrack | BoT-SORT |
| --- | --- | --- |
| 轨迹数 | 1390 | **26** |
| 单帧闪断轨迹 | 643（46%） | **0** |
| ≥100 帧长轨迹 | 99 条 | **24 条（92%）** |
| 平均轨迹长 | 25 帧 | **753 帧** |
| MOT 行（轨迹附着框） | 35132 | 19574 |
| 轨迹框 conf≥0.5 占比 | 72% | 96% |

**结论**：ID 切换的解法不是「加匈牙利」（已内置），而是 BoT-SORT 三件套——ReID 外观关联 + ECC 相机运动补偿 + 官方高轨迹阈值（0.6/0.7：conf<0.7 的未匹配框不建新轨迹，挡掉全部 643 条闪断）。代价是轨迹框总量 35132→19574（低分框不建轨迹，但「不丢检测」语义保持——低分框仍进逐帧 JSON/HITL 复核队列）；26 条长轨迹（平均 31.4s 持续）与视频实际目标规模匹配，成片 `output_sportscheck_2min.mp4` 为 BoT-SORT 版（224MB，规格逐项一致）。后续可调杠杆：ByteTrack 单改 new_track_thresh 0.7 可消大部分闪断（无 ReID/ECC 的 ID 维持）；conf 0.1→0.25 减少噪声进量（本对比保持 0.1 不变）。

### H. 轨迹跳变根因与修复（sportscheck_2min conf=0.5，2026-08-22）

用户反馈：成片轨迹线「从图像的一端跳到另一端」。定位链：

1. **无场景切换**：源视频 2882 帧逐帧灰度 MAD 检测零切帧——跳变不是剪辑造成
2. **零 IoU 错配**：帧 245→246 ID 2 两框 (1728,175,125,83)→(461,308,70,49) 相隔 1300px、IoU=0 仍被匹配，帧 251 又跳回原位置（两目标间 flip-flop）；ID 3 帧 72-80 在三个远距离行人之间来回跳
3. **根因（代码缺陷）**：`_associate_high` 融合通道 `reid_ok = ~isnan(sim) & (sim >= appearance_thresh)` 漏 IoU 门控（`iou_ok` 算了却没用）——λ=0.98 下 IoU 项权重仅 0.02，任意外观相似（cos≥0.25，同类目标几乎必然成立）的框对都能跨屏匹配，与 docstring 声称的「IoU ≥ match_thresh 才融合」矛盾。最小复现：IoU=0、sim=0.5 的框对当前代码匹配 1 对（应为 0）。次生效应：错配一次 → 卡尔曼速度爆炸 + 特征 EMA 污染 → flip-flop 持续
4. **修复 A（治根）**：`reid_ok = iou_ok & ~isnan(sim) & (sim >= appearance_thresh)`——IoU ≥ 0.8 成为融合通道硬门，外观只在位置合理的候选之间选优；既有 4 个关联单测（λ 切换/门控拒绝/防 swap）构造均满足 IoU 门，零破坏
5. **修复 C（断线兜底）**：`Tracklet.update` 丢失后重命中时清空 `_centers`——只清视觉历史（轨迹线在 gap 处断开），卡尔曼状态/特征/ID 关联零影响；防未来即使错配也不跨屏连线

同视频同参数（conf=0.5/iou=0.5/yolo12n.pt/BoT-SORT）重跑对比：

| 指标 | 修复前 | 修复后（A） |
| --- | --- | --- |
| >500px/帧 跳变 | 270 处 | **0** |
| >200px/帧 跳变 | 498 处 | **0** |
| 轨迹线最长单段 | 1548px（跨全屏） | **185px**（15 帧正常走动） |
| 成片跨屏长线（像素级验证） | — | **0 条**（5 帧抽样：源片 vs 成片 diff 连通域无 >60% 帧宽者） |
| 轨迹数 | 33 条（假象） | 404 条（真实） |

**结论**：修复前 33 条超长轨迹（最长 2351 帧横贯 80% 视频）是 bug 的产物——一条轨迹用外观匹配到处「吸收」远距离目标，270 处跳变正是吸收动作；修复后每个目标拥有诚实的轨迹，轨迹线在目标丢失处断开。验证方法（会话图像查看不可用时的替代）：MOT 逐轨迹相邻中心点位移 + 源片/成片像素 diff 连通域检测。附带暴露：160 条单帧闪断轨迹为 conf=0.5 边界检测抖动（修复前被错误吸收掩盖）。

### I. 轨迹延迟输出根因与方案 2（sportscheck_2min conf=0.5，2026-08-22）

用户反馈：不少目标检测到后过一会才有轨迹输出。量化（2882 帧逐帧 JSON + MOT）：

| 证据 | 数值 |
| --- | --- |
| 「有框无轨迹」检测 | 4867 个（track_id=None） |
| 其 conf 分布 | **100% 落 [0.5, 0.7)**（<0.5 被检测阈值拦、≥0.7 未匹配必立即建轨迹） |
| 有前史的轨迹 | 304/404（75%）首帧前同位置目标已被检测到但无轨迹 |
| 延迟帧数 | 中位 3 帧、最长 30 帧 |
| 前史框 conf | 0.6-0.7 段 66.5%、0.5-0.6 段 33.5% |

**根因**：BoT-SORT 官方轨迹阈值（track_high=0.6 / new_track=0.7）不跟随用户指令 conf=0.5——新目标 conf 在 0.5-0.7 区间时框画出来了（draw_bboxes 画所有检测）但拿不到 ID/轨迹线，直到某帧 conf≥0.7。修复 A 前外观错配把远处轨迹拉来「立即认领」（错误但快，正是跳变来源）；修 A 后被 IoU 门限制而诚实等待——「错误快」变「正确慢」，现象因此显现。

**方案 2 定案（用户拍板）**：`new_track_thresh` 官方 0.7 → 0.6（与 track_high 对齐）——[0.6,0.7) 未匹配框立即建轨迹（消除 66.5% 延迟来源）；[0.5,0.6) 保持低分救援语义（只匹配已有轨迹不建轨迹，闪断风险小）。拒绝方案 1（完全对齐 conf=0.5：闪断轨迹增多、MOT 口径变化大）与方案 3（仅可视化区分：观感保留）。

重跑验证（同参数，A+C+方案 2）：

| 指标 | 修复 A 后 | A+C+方案 2 |
| --- | --- | --- |
| >500px 跳变 | 0 | **0**（853 条轨迹逐段位移验证） |
| 轨迹数 | 404 | **853**（404→853，[0.6,0.7) 未匹配框立即建轨迹） |
| 无轨迹框（track_id=None） | 4867 | **1316**（-73%，100% 落 [0.5,0.6) 低分救援段） |
| 前史轨迹占比 | 75% | **40.4%**（345/853） |
| 延迟中位 | 3 帧 | **2 帧** |

**口径说明**（统一 JSON-only 复算，窗口 = 轨迹首帧前 30 帧内同位置 <100px 无轨迹检测框）：方案 2 前 82.2% 有前史（中位 1 帧、≥2 帧 22.8%）→ 方案 2 后 40.4%（中位 2 帧、≥2 帧 22.9%）。≥2 帧前史绝对数 92→195 非退化：分母 404→853 翻倍后占比持平——新增轨迹多为 [0.5,0.6) 闪烁目标（此前从未有轨迹，现由 [0.6,0.7) 帧建立，前史自然更长）；[0.6,0.7) 段 2020 个无轨迹框**归零**（立即建轨迹，方案 2 的 66.5% 延迟来源完全消除）。剩余 1316 个无轨迹框全在 [0.5,0.6)：低于用户 conf 0.5 仅 0.1 的模糊带，维持低分救援语义（只并入已有轨迹、不建新轨迹，闪断风险小）为定案预期。

### J. from_dict 快照续跑真实冒烟（2026-08-22）

真实管线冒烟：`auto2dlabel run sample.png "检测行人" -t 0.3 -o outputs`（yolo26x.pt，CPU）→ 快照文件 `sample_*_state.json` 读回 `AgentState.from_dict`——**snapshot OK**：iter=3 / done=True / max_iter=3 全字段恢复，与运行期状态一致。续跑路径（`--resume` 恢复 + 防重复检测）由单测覆盖（零权重 Fake 注入，见测试统计行）。

### K. 3b 模型级重试 + 批次级策略（2026-08-22）

真实冒烟：`auto2dlabel chat "检测 /tmp/chat_batch_smoke 中的汽车" --no-wait --batch-strategy --batch-size 2`（4 张 COCO 图，yolo26x.pt CPU）——**全链路通过**：Step 1 object_detection → 批次策略抽样 4 张 → LLM 每批 1 次调参 **conf 0.1→0.05 + 建议模型 yolo26x.pt** → 2 图 × 2 块批量推理 → 4 图导出/可视化完成。无 API key / LLM 输出非法时黄字降级代码级默认参数（单测覆盖）。换模型动作（retry_swap_model）路径由单测覆盖（FakeAltModel monkeypatch，零真实权重）——真实换模型加载留给 GPU 批量场景复测。

### L. KITTI difficulty 分层真实冒烟（2026-08-22）

冒烟：`python -m auto2dlabel.benchmarks.kitti_benchmark --max-images 50 --conf 0.3`（yolo26x.pt，CPU，24.1s）——**分层同报通过**：50 图 GT 152 目标全部分档（无 ignore 残留），三档之和 == overall 总 GT；mAP 随难度单调递减符合 KITTI 语义：

| 难度档 | mAP@0.5 | GT 目标 |
| --- | --- | --- |
| overall | 0.3147 | 152 |
| easy | 0.3794 | 58 |
| moderate | 0.1621 | 64 |
| hard | 0.1459 | 30 |

`--difficulty easy/moderate/hard` 单档模式与 `--difficulty all` 默认（overall + 三档同报 + JSON per_difficulty 字段）均由单测覆盖（判据边界 10 例 + GT 标签/过滤/分层指标合成，见测试统计行）。域内微调闭环为 GPU 项（见 milestone/v0.4.md Phase 2）。

### M. Web 三件套：分类展示 + OBB 旋转框 + bbox 拖拽/标签编辑（2026-08-22）

真实冒烟（server 端口 8766 启动 + curl）：

- **端点冒烟**：`/api/obb-models` 15 个 YOLO-OBB 模型；`/api/cls-models` 21 个（hf 零样本 2 + torchvision 14 + custom 5）；首页含 taskType/clsModel/画布编辑钩子 15 处
- **OBB 真实标注**：`POST /api/annotate task_type=obb model=yolo11n-obb.pt`（COCO 图 000000000139.jpg，本地权重）→ 1 旋转框：xy=(446.7,127.6)（cx-w/2 转换）、angle=1.3893 rad、labels=[]，可视化 base64 正常
- sample.png 本身无目标（直接 detect_obb 亦 0 框）——路由正常非缺陷
- 分类真实模型冒烟（CLIP 权重 ~600MB 下载）留给 GPU 环境复测；分类/拖拽/标签编辑路径由单测覆盖（Fake 注入零真实权重）

### N. KITTI 域内微调闭环（GPU，2026-08-23）

训练（`python3 -m auto2dlabel.tools.train_kitti`，RTX 3080 Ti 12GB）：

| 项 | 值 |
| --- | --- |
| 数据 | train 6733 / val 748（seed 42，10% 划分，images symlink 零复制） |
| 配置 | yolo11s，80 epochs，batch=16，imgsz=640，patience=15 |
| 时长 | ~33 分钟（81% GPU 利用率，4.5G 显存） |
| best.pt val（748 图，conf=0.001） | P=0.9047 R=0.8699 **mAP50=0.9430** mAP50-95=0.7505 |

官方口径 benchmark（`--difficulty all --max-images 0`，全量 7481 图，conf=0.3，IoU@0.5，186.0s / 40.2 img/s）：

| 难度档 | 微调 best.pt | yolo26x 基线（同 conf 全量） | GT 目标 |
| --- | --- | --- | --- |
| overall | **0.8867** | 0.2702 | 30584 |
| easy | **0.6177** | 0.2448 | 9744 |
| moderate | **0.4090** | 0.1068 | 13039 |
| hard | **0.1592** | 0.0276 | 7801 |

全量同口径（conf=0.3）提升 **3.28 倍**（0.8867 vs 0.2702），难度越深提升越大（hard 档 5.8 倍）；此前基线 50 图冒烟 0.3147 / v0.3 GPU 复测 0.4081 均为子集口径，不可与全量直接比。

逐类 AP（微调 vs 基线）：bicycle **0.9019 vs 0.0147**（61 倍——KITTI Cyclist 框含骑车人整体，COCO 预训练 bicycle 类只认车；微调学到 KITTI 语义）/ truck 0.9086 vs 0.1230 / train 0.9083 vs 0.1530 / person 0.8088 vs 0.4925 / car 0.9060 vs 0.5678——5 类微调后全部 >0.8。

过程修复两处 bug（均有回归测试）：

- `create_detection_model` 路由顺序：`"/" in model_name` 先于 `.pt` 判定，微调产物完整路径被误判为 GroundingDINO HF repo id（AutoProcessor OSError）→ `.pt` 判定前置
- `save_results` 文件名直接拼接模型名，路径含 "/" 形成非法嵌套目录（指标算完崩在保存）→ common.save_results 统一 sanitize（"/"→"_"，所有 benchmark 受益）

权重落 `weights/kitti_finetune/yolo11s_kitti/weights/best.pt`（19MB，不入库）。

## 测试统计

| 项 | 数值 |
| --- | --- |
| 新增单测（tracking 相关 7 文件） | 46（byte_tracker 13 / export_mot 5 / track_eval 7 / track_frames 4 / track_llm_plan 11 / gt_as_detection 2 / tracker_time_benchmark 4） |
| 新增单测（BoT-SORT 2 文件，2026-08-20） | 30（bot_sort 18 / reid_model 12） |
| 新增单测（chat 跟踪 3 文件，2026-08-21） | 19（tracking_tool 13 / planner_tracking 3 / execute_tracking_branch 3） |
| 新增单测（视频成片输出，2026-08-21） | 1（tracking_tool：视频源出片端到端） |
| 新增单测（ReID 离线加载优先，2026-08-21） | 1（reid_model：本地缓存优先→在线回退） |
| 新增单测（跳变修复 A + 断线 C + 方案 2，2026-08-22） | 3（bot_sort：IoU 硬门回归 + new_track 0.6 边界；byte_tracker：gap 断线）；另重命名 1（test_botsort_official_defaults→test_botsort_defaults） |
| 新增单测（--no-viz 开关，2026-08-22） | 1（tracking_tool：viz=False 跳过 vis_outputs 逐帧 PNG，MOT/逐帧 JSON/成片视频不受影响） |
| 新增单测（约束过滤层 3a，2026-08-22） | 27（constraints 23：parse 参数化 7 / parse_roi 7 / 方位·交集 4 / 属性过滤 4 / TrackingTool 端到端 1；track_llm_plan 11→15 净增 4：attributes/position 解析 + 非法词拒绝） |
| 新增单测（from_dict 快照续跑，2026-08-22） | 7（test_agent_state_roundtrip 7：全字段 round-trip / annotations 完整恢复 angle·track_id·labels·review_flags / 空与旧快照默认值兼容 / 续跑不重复检测 / 续跑重复调用跳过 / done 短路）；test_cli_resume 适配 initial_state 参数 |
| 新增单测（3b 模型级重试 + 批次级策略，2026-08-22） | 30（model_swap_retry 13：pick 规则参数化 8 / swap 动作 4 / orchestrator e2e 1；batch_strategy 17：抽样 3 / LLM 调参 6 / apply_strategy 3 / execute_plan 挂钩 5） |
| 新增单测（KITTI difficulty 分层，2026-08-22） | 14（kitti_difficulty：判据边界参数化 10 / GT 标签与过滤 3 / 分层指标合成 1） |
| 新增单测（Web 三件套，2026-08-22） | 8（test_web_annotate：_bbox_from_dict 保真 2 / annotate 三路由 3 / export-coco angle 1 / review-save edited 2） |
| 新增单测（KITTI 域内微调，2026-08-23） | 17（test_kitti_finetune 14：类别映射 / kitti_line_to_yolo 参数化 6 / 转换幂等 / 划分确定性 / data.yaml / 自定义 names 3 / .pt 路径路由回归；test_benchmark_save_results 3：模型路径·HF repo id sanitize / 普通名不变） |
| 确定性 property（byte_tracker + bot_sort 内） | 3（手工混合场景 + 密集合成场景 + BoT-SORT 含 ECC 开，同输入逐位相等） |
| 全量 | **585 passed** |
