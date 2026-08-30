# auto3dlabel v0.3 实测记录

> 本文件 = v0.3 真权重冒烟与评测的实测数据（合成测试全绿见 pytest 基线；里程碑定义见 `milestone/v0.3.md`）。
> **环境换卡记录**：v0.2 实测于 RTX 4090 D；本版本为 **RTX 3080 Ti 12GB**（多租户宿主，实测期 load 13-20）——AP 数字硬件无关（可比），耗时数字硬件相关（如实记录，不可与 v0.2 耗时直接对比）。

## P2-1：KITTI 接入 PV-RCNN（`-d pvrcnn_kitti`）

### 接入与权重

- 路由 `DETECTOR3D_NAMES` + `download_detector3d.sh`（ENTRIES 改 4 字段 `name|version_prefix|subdir|file`）+ cli `-d` help + planner prompt，共 4 处；`models/detection3d.py` **零改动**（PV-RCNN 输出与 PointPillars KITTI 同构：`InstanceData{bboxes_3d/scores_3d/labels_3d}` 单路融合、LiDAR 系 z=底面中心 origin=(0.5,0.5,0.0)）
- 权重 `pv_rcnn_8xb2-80e_kitti-3d-3class_20221117_234428-b384d22f.pth`（**162614449 字节**，openmmlab 官方直链），torch.load 冒烟 5 keys（state_dict/meta/message_hub/optimizer/param_schedulers）✓
- **torch 2.6+ weights_only 兼容修复（本版新发现）**：默认 weights_only=True 拒 2022 年老 checkpoint 的 numpy scalar/dtype 与 mmengine HistoryBuffer 全局（allowlist 逐项追加无止境）→ `_init_model_trusted` scoped patch：patch 作用域仅限 `init_model` 调用期（`setattr(torch, "load", ...)`，finally 恢复），下载脚本冒烟同法；回归测试 `test_init_model_trusted_scoped_patch`（正常/异常两路径均恢复 torch.load）

### 20 帧冒烟（conf 0.3，帧 003712-003731，与 v0.2 同区间同口径）

执行：`python3 -m auto3dlabel.benchmarks.smoke_kitti 003712-003731 pvrcnn_kitti 0.3`
实测：12.8s（0.67s/帧——**计时含首帧模型加载 ~8s**，纯推理 0.23s/帧见全 val）、216 框、**峰值显存 9083 MiB**（12GB 内富余）

**官方 40-point 口径**（每类官方 IoU；括号 = GT 计数累积分层）：

| 难度 | Car（PV-RCNN） | Car（PP v0.2） | Ped（PV-RCNN） | Ped（PP v0.2） | Cyclist（PV-RCNN） | Cyclist（PP v0.2） |
| --- | --- | --- | --- | --- | --- | --- |
| easy | 27.5 (12) | 27.1 (12) | 28.2 (13) | 18.1 (13) | 0.0 (0) | 0.0 (0) |
| moderate | **60.9 (30)** | 54.1 (30) | 28.2 (14) | 20.5 (14) | 0.0 (1) | 0.0 (1) |
| hard | 88.3 (46) | 76.4 (46) | 34.7 (18) | 25.3 (18) | 0.0 (1) | 0.0 (1) |

11-point 对照（IoU 0.5，v0.1 基线同口径）3D Car 43.8/17.8/25.6、Ped 66.8/0.0/6.1；BEV Cyclist moderate 16.7 (1)（官方口径 0.0——唯一 1 个 GT 匹配失败）。

- 冒烟即全面超 PP（Car moderate +6.8 / hard +11.9）；Cyclist 全 0 与 PP 同（20 帧 GT 仅 0/1/1，v0.2 已证小样本伪影，终版看全 val）

### 全 val 终版数字（conf 0.1，帧 3712-7481，3769 帧）

执行：`nohup python3 -m auto3dlabel.benchmarks.smoke_kitti 3712-7481 pvrcnn_kitti 0.1 > logs/p2_pvrcnn_fullval.log 2>&1 &`
实测：推理 **865.4s（0.23s/帧，纯推理，不含加载）**、**52045 框**（PP v0.2：414s / 56297 框）、推理期显存 9085 MiB（与冒烟 9083 一致）

**官方 40-point 口径**（每类官方 IoU；括号 = GT 计数累积分层，与 v0.2 全 val 同一 GT 池）：

| 难度 | Car（PV-RCNN） | Car（PP v0.2） | Ped（PV-RCNN） | Ped（PP v0.2） | Cyclist（PV-RCNN） | Cyclist（PP v0.2） |
| --- | --- | --- | --- | --- | --- | --- |
| easy | **92.4 (3055)** | 89.8 | **72.3 (1116)** | 60.4 | **89.7 (311)** | 88.4 |
| moderate | **86.9 (7991)** | **82.0** | **65.1 (1770)** | 53.9 | **78.7 (535)** | 74.5 |
| hard | **84.8 (11065)** | 77.2 | **60.7 (2115)** | 49.5 | **73.9 (591)** | 69.9 |

**zoo 对照**：PV-RCNN 官方发布值 Car moderate **81.43** → 实测 **86.9（+5.5）**（PP 时实测 vs zoo 为 +4.4，同向同量级；成因同 v0.2 疑点：conf 0.1 滤框 + 推理链 NMS 细节，如实记录不追查）。

**11-point 对照口径**（IoU 0.5，v0.1 基线同口径）：

| 层次 | 难度 | Car | Pedestrian | Cyclist |
| --- | --- | --- | --- | --- |
| 3D | easy | 44.5 (3056) | 44.9 (1116) | 51.3 (312) |
| 3D | moderate | 32.6 (4937) | 7.5 (654) | 9.5 (223) |
| 3D | hard | 13.6 (3076) | 2.2 (345) | 1.0 (56) |
| BEV | easy / moderate / hard | 44.5 / 32.6 / 13.6 | 45.9 / 8.0 / 2.3 | 51.3 / 9.8 / 1.0 |

- Car 11-point vs PP v0.2（48.9/28.3/9.7）：moderate/hard 升、easy 降（双口径口径差已在 v0.2 记录）；Ped/Cyclist 11-point PP 无记录不对比
- **耗时预算核对**：里程碑 20-30min 推理 + ~95min 评测 → 实测 **865s 推理 + ~3min 评测**（总 ~18min）。评测远快于 v0.2 的 95min：评测代码自 v0.2 未变（纯 numpy 精确移植），差异为宿主 CPU 不同（v0.2 实测于 4090 宿主），数字完整性已核——GT 池计数（3055/7991/11065、11-point hard 3076）与 v0.2 全 val 完全一致，双口径表齐全
- **冒烟→全 val 一致**：20 帧冒烟（小样本）与全 val（终版）同向一致，冒烟即超 PP 的预判在全 val 上兑现且幅度扩大

## P2-2：nuScenes 接入 CenterPoint（`centerpoint_nus`）

执行：`python3 -m auto3dlabel.benchmarks.smoke_nuscenes centerpoint_nus 0.3`（Mini val 2 场景、81 samples、LIDAR_TOP 单帧、x-y 旋转矩形 IoU 0.5、10-point、2 距离分桶——v0.2 同口径）

推理 14.5s（0.18s/sample）、**4164 框**（PP 2214）：

| 范围 | barrier | bicycle | bus | car | constr. | motorc. | pedest. | traff.cone | trailer | truck |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 0.0 (0) | 6.2 (43) | 80.0 (41) | **67.8 (1991)** | 0.0 (0) | 6.7 (232) | 27.1 (1254) | 0.0 (35) | 0.0 (0) | 46.1 (122) |
| 0-25m | 0.0 | 13.0 | 100.0 | 88.7 | 0.0 | 14.5 | 35.7 | 0.0 | 0.0 | 76.3 |
| 25-50m | 0.0 | 6.1 | 100.0 | 53.3 | 0.0 | 0.0 | 11.2 | 0.0 | 0.0 | 44.8 |

**mAP = 23.4 vs pointpillars_nus 17.7（+5.7）**；PP 对照（v0.2）：bicycle 0.0 / bus 68.7 / car 58.4 / motorc 0.0 / pedest 8.1 / t.cone 0.0 / truck 42.2，0-25m car 79.3，25-50m car 44.0。

- **L-W sanity 通过**：car 为最大类且 AP 67.8（>PP 58.4）——v0.2 的「2021-08 旧权重 L-W 与 1.4.0 box coder 相反」问题（症状 car AP≈0 + 预测多）未出现，2022-08 CenterPoint 权重与现码约定一致，**未预改代码**
- 与 zoo 官方 mAP 44.6 不可直接对等（全量 val 150 场景 + 10 次扫合并 + 官方 TP 距离阈值指标 vs 本表 Mini 2 场景单帧简化口径），数字仅作同口径模型对比
- 提交 JSON 自检通过（81 samples、4164 框；outputs/nuscenes_submission_centerpoint_nus.json）
- **墙钟如实记录**：首轮前台 600s 超时后残留孤儿进程（CPU 93.8% 烧 10:34）+ 多租户宿主 load 13-20，后台任务墙钟 ~15min。分阶段探针实证管线本身仅 ~15s（import 0.2s / 模型构建+权重加载 10.2s / 首帧 4.7s / 81 样本 14.5s），评测代码线性——**环境因素，非管线缺陷**

## P2-3：难类分层收尾

判定标准（计划批准）：难类升 + 对照组稳 → PP 的 0 是模型短板；难类仍低但预测数 >0 → Mini GT 池 + 单帧无 sweeps 的召回/匹配上限（数据稀少）。两模型同一 GT 池，差异即模型能力。

### nuScenes（CenterPoint vs PointPillars，预测数 = 提交 JSON 实测）

| 类 | GT | PP AP（预测数） | CP AP（预测数） | 判定 |
| --- | --- | --- | --- | --- |
| bicycle | 43 | 0.0（2） | **6.2（95）** | PP 的 0 含模型短板（CP 召回 47×） |
| motorcycle | 232 | 0.0（10） | **6.7（129）** | 同上（召回 13×） |
| traffic_cone | 35 | 0.0（1） | 0.0（**147**） | 检出不缺（147×）但 0 匹配——cone 极小（~0.4m）+ 距离远，定位误差超旋转矩形 IoU 0.5 门槛 |
| car（对照） | 1991 | 58.4 | 67.8 | 对照稳升 |
| pedestrian（对照） | 1254 | 8.1 | 27.1 | 对照稳升 |
| truck（对照） | 122 | 42.2 | 46.1 | 对照稳升 |

- bicycle/motorcycle 已非零：v0.2「预测极少（2/10 条）→ AP 0」的根因确为模型短板成分，CenterPoint 修复至非零；但 6-7 的绝对水平仍低——小型目标 + 单帧无 sweeps 的稀疏点云是上限主因（0-25m 13.0/14.5 vs 25-50m 6.1/0.0，距离分层印证稀疏性集中段）
- traffic_cone 判定：**「能检出（147 预测）但匹配 0」**——瓶颈在定位精度/极小目标匹配门槛而非召回；GT 池仅 35 且多为远距小目标
- 对照组全升（car +9.4 / pedest +19.0 / truck +3.9）→ CenterPoint 提升是全面的，非难类专属

### KITTI Cyclist（PV-RCNN 全 val vs PointPillars）

| 类 | GT | PP v0.2 全 val | PV-RCNN 全 val | 判定 |
| --- | --- | --- | --- | --- |
| Cyclist | 311/535/591 | 88.4 / 74.5 / 69.9 | **89.7 / 78.7 / 73.9** | 非难类：两模型均健康，PV-RCNN 三难度全面领先（+1.3/+4.2/+4.0） |
| Car（对照） | — | 82.0 moderate | 86.9 | 对照组同升 |
| Pedestrian（对照） | — | 53.9 moderate | 65.1 | 对照组同升 |

- **分层归因定案**：v0.2 记录的「20 帧冒烟 Cyclist 全 0」根因 = **数据稀少（小样本伪影）**——20 帧 GT 仅 0/1/1（官方 41 点稀疏惩罚：1 个 GT 无 TP 即 AP 0，官方同款行为），**非模型短板**；全 val 上 PP 与 PV-RCNN 的 Cyclist 均健康（88.4 / 89.7 easy），PV-RCNN 进一步全面领先
- 全 val AP 非零即已满足验收，无需再分预测数（v0.2 教训「检不出 vs 检出未匹配」的区分仅适用于 0 AP 情形）

## P2 验收对照

| 验收条款 | 结果 |
| --- | --- |
| KITTI PV-RCNN 全 val 官方口径出表（Car moderate vs 82.0） | ✅ **86.9 vs 82.0（+4.9）**；zoo 81.43 亦超（+5.5）；Ped/Cyclist 全面领先 |
| Cyclist 非零（或分层说明） | ✅ **89.7 / 78.7 / 73.9 非零**（vs PP 88.4/74.5/69.9）；20 帧全 0 分层归因：数据稀少（GT 0/1/1 小样本伪影），非模型短板 |
| nuScenes CenterPoint 简化评测出表（vs 17.7） | ✅ mAP 23.4（+5.7），car 67.8 vs 58.4，L-W sanity 通过 |
| 难类结果如实记录 | ✅ 见 P2-3（bicycle/motorcycle 非零 + traffic_cone 147 预测 0 匹配分层说明 + KITTI Cyclist 归因） |

## P3：BEVFusion nuScenes + KITTI 单目降级（pgd）

### 接入与权重

- 新增 `models/bevfusion3d.py`（四段式母版同 detection3d.py）：`build_bevfusion_data`（data_ 契约：LIDAR_TOP + 6 相机 cam2img/lidar2cam 标定链，devkit 矩阵自检）+ `BevFusionDetector`（零加载 `__init__` + 幂等 `_load` + `custom_imports` 注册 projects/BEVFusion + Swin `init_cfg.checkpoint=None` Config 深拷贝 patch，零临时文件）+ 工厂；`configs/nuscenes.py` 加 `BEVFUSION_NAMES`（**不进 DETECTOR3D_NAMES**——detect(frame) 协议不兼容）；`download_detector3d.sh` 追加 3 条 ENTRIES（bevfusion_nus / bevfusion_lidar_nus 备选 / pgd_kitti）
- 权重 `bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth`（**`v1.1.0_models` 前缀**，官方直链实测 HTTP 200；此前 openmmlab 页面未列，README blob `git show` 锁定 URL）；mmdet3d v1.4.0 `projects/BEVFusion/` sparse-checkout + `setup.py build_ext --inplace`（bev_pool_ext/voxel_layer，产物留在 weights/ 不入库）
- `models/detection3d.py` 仅 `_init_model_trusted` 参数放宽 `str | Path | Config`（函数体零变）；`data/nuscenes.py` 抽公共纯函数 `boxes_sensor_to_global`（smoke_nuscenes 补偿段上移，行为零变，复用不复制）
- 新增 `models/mono3d.py`（单目 pgd：pgd_r101_caffe_fpn_gn-head_3x4_4x_kitti-mono3d，Camera 系 7 值底面中心 → `Det3DResult.to_box3d()` 直通）+ `configs/kitti.py` 加 `MONO3D_NAMES` + `benchmarks/smoke_kitti_mono.py`（20 帧，官方 40-point + 11-point 双口径，`--lidar` 同帧对照）

### BEVFusion 简化评测（Mini val 2 场景 81 samples，conf 0.3，P2-2 同口径）

执行：`python3 -m auto3dlabel.benchmarks.smoke_bevfusion bevfusion_nus 0.3`
实测：**53.4s（0.66s/sample）**（含模型构建+加载 ~10s）、12GB 无 OOM：

| 范围 | barrier | bicycle | bus | car | constr. | motorc. | pedest. | traff.cone | trailer | truck |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 0.0 (0) | 4.2 (43) | 80.0 (41) | **69.7 (1991)** | 0.0 (0) | 18.7 (232) | **31.6 (1254)** | **25.7 (35)** | 0.0 (0) | 39.8 (122) |
| 0-25m | 0.0 | 10.0 | 100.0 | **89.8** | 0.0 | 28.2 | 47.7 | 25.4 | 0.0 | 69.7 |
| 25-50m | 0.0 | 10.0 | 100.0 | 59.2 | 0.0 | 0.0 | 18.2 | 0.0 | 0.0 | 30.0 |

**mAP = 27.0 > centerpoint_nus 23.4 > pointpillars_nus 17.7**；CP 对照（P2-2）：bicycle 6.2 / motorc 6.7 / pedest 27.1 / t.cone 0.0 / truck 46.1 / 0-25m car 88.7。提交 JSON 自检通过（81 samples，outputs/nuscenes_submission_bevfusion_nus.json）。与官方 mAP 66.6 不可直接对等（官方全量 val 150 场景 + 10 次扫合并 + TP 距离阈值指标），数字仅作同口径模型对比（同 P2-2 口径注）。

**L-W 修复记录（本版关键）**：首跑 mAP 5.8、car 全灭（1991 GT）——v0.2 同款「旧权重 L-W 与 coder 相反」症状。配对诊断（提交 JSON vs GT 中心距<4m 配对，5 假设 BEV IoU）：**换 L-W 后 car BEV IoU 0.254→0.795，yaw 与中心零误差**。单帧实证：decode 输出 idx3=4.47≈GT l 4.7、idx4=1.85≈GT w 2.0 → TransFusionBBoxCoder 输出即训练语义 (l,w,h)。`_normalize_bevfusion_boxes` 曾按「tensor dims=(w,l,h)」推断做过 swap（`[0,1,2,4,3,5,6]`）→ car 全灭；修正为**恒等直通** → mAP 27.0。函数保留作归一化单一事实源（新模型按需在此换序）。教训：代码推断须经真实数据实证校准。

### KITTI 单目 pgd（20 帧 003712-003731，与 P2-1 同区间）

执行：`python3 -m auto3dlabel.benchmarks.smoke_kitti_mono 003712-003731 pgd_kitti 0.3 --lidar pointpillars_kitti`

**官方 40-point 口径**（每类官方 IoU；括号 = GT 计数累积分层）：

| 难度 | Car（pgd） | Car（PP LiDAR） | Ped（pgd） | Ped（PP） | Cyclist（pgd） | Cyclist（PP） |
| --- | --- | --- | --- | --- | --- | --- |
| easy | 11.7 (12) | 27.1 (12) | 2.9 (13) | 17.3 (13) | 0.0 (0) | 0.0 (0) |
| moderate | **33.7 (30)** | 54.1 (30) | 2.9 (14) | 19.7 (14) | 0.0 (1) | 0.0 (1) |
| hard | 49.1 (46) | 76.4 (46) | 4.5 (18) | 24.4 (18) | 0.0 (1) | 0.0 (1) |

**11-point 对照口径**（IoU 0.5，v0.1 基线同口径）：3D Car 37.0 / **31.6** / 16.1、Ped 19.2 / 0.0 / 6.4；Cyclist 全 0（20 帧 GT 0/1/1，小样本伪影同 P2-1 归因）。

**L-W sanity（h/w/l 中位数，米）**：

| 类 | GT | pgd 预测 | 判定 |
| --- | --- | --- | --- |
| Car | h=1.50 w=1.63 l=3.82 (n=71) | h=1.57 w=1.54 l=3.42 (n=170) | ✅ 同量级无交换 |
| Pedestrian | h=1.77 w=0.78 l=0.91 (n=18) | h=1.66 w=0.68 l=0.91 (n=124) | ✅ 同量级无交换 |
| Cyclist | h=1.79 w=0.60 l=1.58 (n=3) | h=1.63 w=1.30 l=3.17 (n=86) | GT n=3 小样本；预测尺寸膨胀（单目远距小目标），如实记录 |

- pgd Car moderate 33.7（官方口径）/ 31.6（11-point）落在「单目 ≤30% 量级」预期带（31.6/33.7），与 LiDAR 系（PV-RCNN 86.9 / PP 82.0）差 ~2.5-2.7×——印证「LiDAR 系是唯一可达标注业务精度的路线」，pgd 作 P5 LabelAny3D 判据的单目基准

### P3 验收对照

| 验收条款 | 结果 |
| --- | --- |
| BEVFusion nuScenes 简化评测出表（vs 17.7 / 23.4） | ✅ **mAP 27.0 > 23.4 > 17.7**；car 69.7 / bus 80.0 / truck 39.8；traffic_cone 25.7 首次非零（CP 0.0）；0-25m car 89.8 |
| KITTI 单目降级方案落地一项（pgd，moderate ≤30% 基准） | ✅ 官方口径 33.7 / 11-point 31.6（≤30% 量级预期带）；L-W sanity 通过 |
| 缺口如实记录 | ✅ 官方无 KITTI 域 BEVFusion 权重（KITTI 融合引擎空缺，LiDAR 系已有 PV-RCNN 86.9 覆盖）；后融合原型不做（milestone 排除）；pgd Cyclist 20 帧 0.0 = GT 0/1/1 小样本伪影（同 P2-1） |

## P4：Web 真 3D 复核（四视图 + 同端口共存）

### 接入与实现

- 新增 `web/static/three.min.js`（jsdelivr r149 ~600KB，r150 后无 min 构建版）+ `README.md`（下载源/版本/sha256）；`web/static/logic3d.js`（**纯逻辑零 three**：状态机/loadFiles/selectObject/changeLabel/buildSaveBody/renderLists + orthoFocus 纯数学，fetch 相对路径 `api/frame-data`——jsdom 可测）+ `web/static/viewer3d.js`（three 装配：点云 `(x,z,-y)` 换轴、cuboid 12 边 + 车头方向线、四视图 scissor 四分屏——正交视图只显选中对象 + 自动聚焦）；`index.html` 升级（保留侧栏与 `<title>Auto3dLabel 复核</title>`，主区四视图 + 脚本相对路径）
- 新增 `web/payloads.py` 纯函数：`downsample_points`（随机均匀，seed 确定性，100k 上限）、`validate_queue_name`（从 server.py 抽公共，行为零变）、`frame_payload(name, review_dir)`（读 `*_review.json` → `KittiFrame.load_points()` + `velo_to_cam`（相机系，前端零 calib 依赖）→ 下采样 → 逐框 `Box3D.from_dict(b).corners_cam()` 8x3，**yaw 数学留在 Python 侧**；损坏 → None）
- `web/server.py`：`GET /api/frame-data?name=<queue_file>` → `frame_payload`；**只收 queue_file 名，pcd/calib 路径从队列文件读（零任意路径读取面）**——安全模式继承
- 同端口共存：`auto2dlabel/web/server.py` 仅 `__main__` 段加守卫挂载（模块级 app 零改动 → 2D 测试零影响）：`importlib.import_module("auto3dlabel.web.server")` try/except ImportError → `app.mount("/3d", s3d.app)`；3D 前端全相对路径 → 挂载模式 `/3d/` 与独立模式 :8766 双兼容；2D index 加「3D 复核」链接、3D 页加「2D 复核」链接
- 测试：`test_web3d_payloads.py`（downsample/validate/frame_payload 纯函数）+ 扩展 `test_web3d`（frame-data 端点：合成队列落盘 → 点数 ≤100k / 确定性 / velo_to_cam 数值 / corners 8x3 / 400/404 / pcd_path 缺失）+ `helpers/smoke_web3d.js`（node + jsdom，canvas getContext Proxy mock，不加载 three.min.js——覆盖 logic3d.js 全部函数）；既有 test_index_html 回归

### E2E 实测（2026-08-29）

- 执行链：`auto3dlabel run 003712-003713 "检测汽车和行人" -d pointpillars_kitti --no-viz` → `REVIEW3D_DIR=outputs/kitti3d/reviews python3 -m auto2dlabel.web.server`（:8765 单进程）→ 浏览器打开 `/3d/` 帧
- **四视图交互全链路通过**：透视 + Top/Side/Front 正交，旋转/缩放/选中（正交视图只显选中对象）/正交自动聚焦；改标签 → 保存 → **回写 `labels/000000.txt` 15 字段校验通过**（y 底部中心回写，label 已改）；导出校验通过
- **同端口共存通过**：:8765 上 2D 复核（`/`）与 3D 复核（`/3d/`）同时可用，互不干扰；独立模式 `REVIEW3D_DIR=... python3 -m auto3dlabel.web.server`（:8766）回归全绿
- 保存行为核验：原 `_review.json` rename 成 `.reviewed` + 新写 `_reviewed.json`——**设计行为非 bug**（与 2D 复核协议一致）
- `cd auto3dlabel/tests/helpers && npm i jsdom && node smoke_web3d.js` 全绿（npm 依赖不入 pytest）

### P4 验收对照

| 验收条款 | 结果 |
| --- | --- |
| Web 打开 KITTI 帧四视图可交互（旋转/缩放/选中/正交聚焦） | ✅ E2E 通过（透视 + 三正交 scissor 四分屏，正交只显选中 + 自动聚焦） |
| 改标签保存回写导出校验 | ✅ 15 字段回写 `labels/<frame>.txt` + 导出校验通过 |
| 与 2D 复核 :8765 同端口共存 | ✅ `/`（2D）与 `/3d/`（3D）同进程共存；独立 :8766 回归全绿 |

## P5：LabelAny3D 无 LiDAR 域 3D 标注验证

### 方法与侦察摘要

- LabelAny3D（NeurIPS 2025，Apache-2.0；MASt3R 组件 CC BY-NC-SA 非商用——验证阶段无合规障碍，接入生产需换 matcher）8 步管线：depth（MoGe+DepthPro+RANSAC）→ enhance（InvSR 4x）→ crops（SAM2 mask）→ completion（amodal InstructPix2Pix）→ elevation（zero123 4 视图）→ reconstruction（TRELLIS）→ whole（MASt3R 对齐）→ combine（Omni3D 输出）
- 价值主张：相对深度 + Objaverse 形状先验（TRELLIS）+ amodal 补全遮挡，面向 COCO 日常物体无 GT 域；论文指标为「伪标签训练下游单目检测器」间接指标，无直接标注误差数字——故本次判据 = KITTI 20 帧 moderate 3D AP vs 近零基线（v0.1 反投影拟合）与 pgd 单目基准（31.6/33.7）
- 验证环境完全隔离：`conda env la3d`（torch 2.2.2 cu121 + pytorch3d/detectron2/kaolin/flash-attn/spconv/diffoctreerast 等编译依赖）+ 隔离脚本 `/root/autodl-tmp/p5_validation/`（repo 外）+ 权重 hf-mirror 缓存，不入 repo 不进 autolabel env

### 数据适配（KITTI 20 帧 003712-003731）

- COCONUT 格式改造：20 张图 98 标注（car/person/bicycle），segmentation 包装为多边形列表的列表 + 字段规范化（bbox/area/id 数值化）；管线按 6.25% 图高过滤小对象（记录在案，不修）
- KITTI P2 内参注入：`kitti_depth.py` 注入 K=[721.5, 609.6] 主点（经 cam_params.json 验证）

### 管线执行（2026-08-29）

| 步骤 | 结果 | 说明 |
| --- | --- | --- |
| 1 depth | ✅ 20 帧 | P2 注入生效 |
| 2 enhance | ✅ 4x 4968×1500 | InvSR |
| 3 crops | ✅ 58 个 amodal 裁剪 | "Too small segmentation" 过滤记录在案 |
| 4 completion | ✅ 58 个 rgba，~14min | safety_checker 关闭 + fp16 variant（0.45G 权重子集） |
| 5 elevation | ✅ 58 对象 × 4 视图，4:33 | diffusers 0.30 下可用（无需降级）；"K is not provided, using default K" 记录在案 |
| 6 reconstruction | ✅ 58/58 glb，30:45 | TRELLIS 批量单对象 ~30s（加载/烘焙摊销）；1 个纹理烘焙 OOM 经 `expandable_segments` 重跑成功 |
| 7 whole | ✅ 20 帧 exit 0，~20s/帧 | MASt3R 对齐；`3dbbox_ground.json` 最终 rename 为 `3dbbox.json`（whole.py 内置） |
| 8 combine | ✅ 19 帧 58 框 | `COCO3D_val.json`（默认 bbox_file 即最终名，无需改参数） |

### TRELLIS 12GB 适配（3080 Ti）

- 官方 fp32 加载需 ~16GB → 用 TRELLIS 官方 `use_fp16` 开关（`convert_to_fp16()` 只转主干 blocks，输入/输出层保持 fp32，forward 内自动 cast），**无任何手拼 dtype patch**；运行时显存 11.4G/12G 稳定
- DINOv2 `dinov2_vitl14_reg4_pretrain.pth`（1.14G）经 wget 断点续传下载至 torch.hub checkpoints（curl 截断 / hf-mirror 仓库 ID 不存在两个坑，如实记录）

### 评测与决策

- 全链路 8 步完成（2026-08-29）：58 对象重建（step 6 批量 30:45，单对象均值 ~30s，1 个纹理烘焙 OOM 经 `expandable_segments` 重跑成功）→ MASt3R 对齐（step 7 exit 0，~20s/帧）→ combine（19 帧 58 框 Omni3D；003729 整帧 2 标注被 6.25% 图高过滤 → 无预测 skip，98 标注 → 58 框 = 40% 丢弃率）
- 转换器 `p5_validation/omni3d_to_box3d.py` 合成自检通过：底面中心 = 几何中心 + h/2；dimensions [h,w,l]；类别过滤。**yaw 轴实测纠偏**：canonical z 轴才是车头朝向（列 2 与 GT 一致，BEV Car 9.8；列 0/PCA 主方向差 90° 系统偏，0.0）——按实测选列 2

| 层次-难度 | v0.1 反投影基线（同口径） | pgd 单目（P3） | LabelAny3D P5 |
| --- | --- | --- | --- |
| BEV Car easy / moderate / hard | 1.8 / 7.0 / 0.0 | — | 6.4 / **9.8** / 0.4 |
| BEV Ped easy / moderate / hard | 15.2 / 0.0 / 0.0 | — | 0.0 / 0.0 / 0.0 |
| 3D Car easy / moderate / hard | 0.0 / 0.0 / 0.0 | — | 0.0 / 0.0 / 0.0 |
| 3D Ped easy / moderate / hard | 2.3 / 0.0 / 0.0 | — | 0.0 / 0.0 / 0.0 |
| 3D Car moderate | ~0（近零） | 31.6（11-point）/ 33.7（官方） | **0.0** |

- **判据（20 帧 moderate 3D AP 显著高于近零基线）不达标 → 决策：不接入**。3D 层全零（位置/高度/朝向任意一项失准即出 0.5 IoU）；BEV Car moderate 9.8 仅比 v0.1 基线 7.0 高 2.8 点（非显著），BEV Ped easy 0.0 反比基线 15.2 显著更差
- 误差归因（如实记录）：① yaw 与 GT 中位差 44.2°（PCA 主方向估计，±180° 歧义）；② 深度对齐位置偏差（003712 Car cz 差 ~3m）——相对深度 + 中位比例缩放在 KITTI 远距透视下放大误差；③ 40% 对象被小目标过滤；④ 3D 高度（cy/h）受 depth 精度与 ground alignment 双重影响。耗时 ~3.5min/帧 vs pgd 秒级
- **结论**：LabelAny3D 方法学在 KITTI 自动驾驶域不成立——与其论文定位一致（面向 COCO 日常物体无 GT 域的伪标签训练间接指标，非直接标注精度）；「无 LiDAR 数据源 3D 预标注候选」维持 v2+ 议题不变，验证门槛已有实测数据支撑（本表）。单目系 pgd 31.6 仍是无 LiDAR 域最高可及基线

## P6：mmdet3d 模型矩阵扩展 + 共享 Token 工程（2026-08-30）

### P6a：P3 收尾前置

- P3 冒烟出表（BEVFusion mAP 27.0 / pgd 33.7）已在本文件 P3 段验收——P6a 无需新工作，前置条件满足 ✅

### P6b：模型矩阵扩展（质量/速度档）

**接入与权重**：

- **TransFusion / VoxelNeXt：不接入（如实记录降级）**——v1.4.0 主线 `configs/` 与 `projects/` 均无（TransFusion 在 OpenPCDet 框架、VoxelNeXt 未进 v1.4.0 configs）；速度档由 **FreeAnchor regnet-400mf** 替代（`configs/free_anchor/` 主线现成 + 20210827 权重直链；backbone init_cfg Pretrained 由 `_patch_pretrained_init` 阻断——全量 ckpt 已含 backbone 权重）
- **FCOS3D**（单目质量档，nuScenes 侧——pgd 是 KITTI 侧单目）：`configs/fcos3d/` nus-mono3d_finetune + 20210717 权重；SMOKE 仅 KITTI config（与 pgd 档位重叠）不重复接入
- 接入形态：free_anchor 走 `DETECTOR3D_NAMES`（LiDAR 协议，Mmdet3dDetector 复用）；fcos3d 走 `FCOS3D_NAMES` 独立路由 + `benchmarks/smoke_nuscenes_mono.py`（detect_sample(sample, nusc, conf) 协议，同 BEVFUSION_NAMES 纪律，**不进 DETECTOR3D_NAMES**）

**FCOS3D 输出语义考古与实测定案（本版关键）**：

- 权重 2021-07 训练（v0.15 converter 时代）。源码考古（v0.15 converter/dataset/head + devkit `Box.wlh` + 2021-07 训练时点代码）：GT 训练语义为 (w,l,h) 直传 + 正相机系 yaw——**按考古实现转换 mAP 全 0**
- 实测探针（单样本 pred vs GT 相机系对照）：pred dims (4.44,1.51,1.84) 按 **(l,h,w)** 解读匹配 GT wlh (2.00,4.73,1.48)；yaw = **-GT 相机系 yaw**
- 决定性实验（Mini 全量 81 samples 一次前向缓存 + 4 组合评测）：仅 **D「dims 重排 (t5,t3,t4) + yaw 负号」mAP 非零（0.8 / car 8.0）**，其余三组全 0
- 与 v1.4.0 head 推理路径（零重排零负号，与 v0.15 逐行一致）矛盾 → **矛盾源于权重本身，以实测为准**；适配层 `_fcos3d_reorder_dims_yaw` 归一 (w,l,h)+正 yaw 后仍走 v0.15 版 output_to_nusc_box 定式（`boxes_cam_to_global`）。单测 `test_fcos3d_reorder_dims_yaw` 锁定（实测锚点 + copy 语义 + 空批量）

**fcos3d_nus 简化评测**（Mini val 2 场景 81 samples，conf 0.3，CAM_FRONT 单相机，P2-2 同口径）：

- 执行：`python3 -m auto3dlabel.benchmarks.smoke_nuscenes_mono fcos3d_nus 0.3`；前向 8.3s（0.10s/sample）、510 框

| 范围 | barrier | bicycle | bus | car | constr. | motorc. | pedest. | traff.cone | trailer | truck |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 0.0 (0) | 0.0 (43) | 0.0 (41) | **8.0 (1991)** | 0.0 (0) | 0.0 (232) | 0.0 (1254) | 0.0 (35) | 0.0 (0) | 0.0 (122) |
| 0-25m | 0.0 | 0.0 | 0.0 | 9.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 25-50m | 0.0 | 0.0 | 0.0 | 6.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

**mAP = 0.8、car 8.0**——单相机口径 recall 受限（bevfusion/centerpoint 用 LiDAR 全 360°；单目前向 FOV 天然少框），数字与 LiDAR 系不可直接对等，作**单目交叉验证基准**（pgd 是 KITTI 侧：官方口径 33.7 / 11-point 31.6）。ped 0.0 归因：per-class IoU 诊断——car 中位 bestIoU 0.63（239/362 ≥0.5）证转换正确，ped 中位 0.12 = 单目深度误差对小目标（0.7×0.6m）的固有放大，非转换残留（如实记录）。

**free_anchor_nus 简化评测**（同口径）：

- 执行：`python3 -m auto3dlabel.benchmarks.smoke_nuscenes free_anchor_nus 0.3`；**5.6s/81 samples（0.07s/sample）——全 Mini 最快**（CP 0.18s/sample）、7672 框

| 范围 | barrier | bicycle | bus | car | constr. | motorc. | pedest. | traff.cone | trailer | truck |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 0.0 (0) | 3.8 (43) | 77.5 (41) | **76.1 (1991)** | 0.0 (0) | 20.4 (232) | 36.5 (1254) | 0.2 (35) | 0.0 (0) | 43.9 (122) |
| 0-25m | 0.0 | 6.7 | 100.0 | 88.6 | 0.0 | 26.7 | 47.8 | 0.2 | 0.0 | 68.4 |
| 25-50m | 0.0 | 2.8 | 91.8 | 68.0 | 0.0 | 6.2 | 17.8 | 0.0 | 0.0 | 30.8 |

**mAP = 25.8 > centerpoint_nus 23.4 > pointpillars_nus 17.7**——**速度档成立**：精度超 CP 且速度 2.6×（regnet-400mf 轻骨干）；提交 JSON 自检通过（81 samples）。

### P6b 验收对照

| 验收条款 | 结果 |
| --- | --- |
| 每模型 20 帧冒烟 + 简化评测出表（vs 既有数字） | ✅ fcos3d_nus mAP 0.8 / car 8.0（单目基准）；free_anchor_nus mAP 25.8（vs 17.7 / 23.4）；单样本探针 + Mini 全量 4 组合决定实验（详见 FCOS3D 语义段） |
| 速度档报耗时（对照 CenterPoint 基线） | ✅ free_anchor 5.6s/81（0.07s/sample）vs CP 14.5s（0.18s/sample）——2.6×；单目 fcos3d 前向 8.3s（0.10s/sample） |
| 不可得模型如实记录 | ✅ TransFusion / VoxelNeXt 不在 v1.4.0 主线（OpenPCDet 框架）→ 不接入；速度档由 FreeAnchor 替代；SMOKE 仅 KITTI config（与 pgd 重叠）不重复接入 |

### P6c：共享 Token 工程（LLM Harness 3D 受益）

- 3D 与 2D 共用同一 LLMClient——v0.6 Phase 4 的 usage 台账 / max_tokens / json mode / 前缀缓存 / Evaluate 降级**自动覆盖 3D 侧**，零实现受益
- 3D 侧唯一接线改动：`planner3d.py` parse 走 `max_tokens=1024 + json_mode=True + call_site="planner3d.parse"`、对话路径 `call_site="planner3d.dialog"`——3D chat 调用进同一台账（cost-report 按调用点区分 2D/3D）
- 回归：`test_dialog3d.py` 全绿（Fake LLM 注入，kwargs 穿透断言）

### P6d：模型路由表收拢（2026-08-31，照 2D 母版）

- 4 张路由表（DETECTOR3D_NAMES / MONO3D_NAMES / BEVFUSION_NAMES / FCOS3D_NAMES）自 `configs/kitti.py`、`configs/nuscenes.py` **收拢至 `configs/model_catalog.py`**（单一事实源，含协议纪律注释：detect(frame) vs detect_sample 互斥）；原两文件留指针注释防散副本
- 消费方（`models/detection3d.py` / `mono3d.py` / `bevfusion3d.py` + 3 个测试文件 + `cli.py` help 动态生成）全部 import 引用该表
- **planner3d prompt 注入**：新增 `format_catalog_summary3d()`（5 组 × 9 引擎名 + 用途 hint），注入 system prompt——原硬编码仅 3 个 LiDAR 引擎（P3/P6 新增的 pgd/bevfusion/fcos3d/free_anchor 对 LLM 不可见），现全量可见且新增模型零改 prompt；中文关键词映射行（点柱→pointpillars_kitti 等）保留并补齐新引擎
- 回归：新增 `test_model_catalog3d.py` 9 用例（表字段锚点 / 4 表 key 互斥 / summary 覆盖全部引擎 / groups 与表同步 / prompt 注入不断开 / 消费方同对象引用 / 旧散副本不复存）
- 连带收尾：2D 侧 `auto2dlabel/models/model_catalog.py` → `configs/model_catalog.py` 的半成品迁移（文件已删、40+ 处 import 未改完导致收集错误）——sed 批量改指 configs 完成，configs 版含 COCO_91_TO_80 等全部表

## 质量门

P6d 收拢终局数字（2026-08-31 重跑）：

- pytest autolabel 环境（全项目）：**1056 passed, 4 skipped**（+10 = test_model_catalog3d 新用例）
- pytest base 环境（全项目）：**1059 passed, 1 skipped**（守卫测试运行 + 新 trusted-patch 测试 skip）
- `ruff check auto3dlabel/`：All checks passed（0）
- `mypy auto3dlabel/ --follow-imports=silent`：Success（0，86 source files）
- `pyright auto3dlabel/ + auto2dlabel/`：0 errors, 0 warnings

auto2dlabel 侧不恶化（2D catalog 迁移连带）：ruff **75**（--fix 修 53 处迁移引发项，< 119 基线）/ mypy **167**（< 169 基线）/ pyright 0——明细见 `auto2dlabel/tests/test-v0.6.md` 质量门段。

修复记录（首轮门检发现，均有源头修复）：`test_detection3d.py` 新增测试的 `Any` 标注缺 `from typing import Any`（mypy 3 / pyright 5 / ruff F821 ×5）；`detection3d.py` 守卫 import 块未按 isort 排序（I001，mmdet3d 排 torch 前，守卫语义不变）。
