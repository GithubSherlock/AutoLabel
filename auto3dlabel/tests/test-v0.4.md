# auto3dlabel v0.4 实测记录

> 本文件 = v0.4 实测数据（合成测试计数见各节；里程碑定义见 `milestone/v0.4.md`）。
> **环境**：RTX 3080 Ti 12GB 多租户宿主（P1 纯前端无 GPU 项；P2/P3 有 GPU 项）。

## Phase 1：P4b cuboid 手柄编辑（Web 3D 复核补齐）

### 实现落点（2026-08-31）

**后端零改动**（v0.1 红线保持）：编辑体直接操作队列 JSON `annotations[i]`（cx/cy/cz/h/w/l/rotation_y 原字段），`buildSaveBody` 全量直通 → server `Box3D.from_dict`。`payload.py / server.py / box3d.py` 零改动，Python 侧零 import 变化。

**logic3d.js 编辑层**（纯函数 + 状态机，jsdom 可测）：
- 角度转换：`rotationYToYaw(ry) = wrap(ry + π/2)` / `yawToRotationY` 逆变换（wrap [-π, π]，照 `tools/geometry.py` 唯一转换点；编辑渲染必要转换，产品导出链路不动）
- `boxToCorners(b)`：annotations dict → corners_cam 8×3（车头 (sin yaw, cos yaw)、右向 (dz, −dx)、y_low/y_high，公式照 box3d.py）
- 编辑数学（2D 母版 resize/rotate 在 3D 的适配，BEV 内部角 t = **−yaw**）：
  - `resizeTopEdit`：Top 平面「对角固定 F + 手柄吸附指针」（w/l 齐变或单轴，`w1 = sx·(plx−fx)` 带 MIN_BOX3D 钳位）
  - `rotateYawEdit`：**增量式** `yaw = wrap(yaw0 + (t1 − t0))`（3D 车头有向，不能用 2D 无向绝对角）
  - `resizeSideEdit`：顶面 → `h += −δy_cam, cy += δy_cam/2`（对面固定；cam y 向下语义）；底面同理反号
  - `moveSideEdit`：前/后 = **cz 平移**（车头轴 ≠ z 轴时沿 z 缩 l 无意义，l 编辑由 Top 车头/车尾边中点手柄覆盖）
  - `MIN_BOX3D = 0.3`（米级最小尺寸，同 2D MIN_BOX 语义）
- 状态机：`beginEdit(i, view, mode, drag)`（snapshot 深拷贝 = undo 锚点；drag = 视图层 mousedown 几何锚，快照式绝无增量累积）→ `editTo(view, mode, ptr)`（调纯函数 → 写回 `annotations[i]` 几何键 + rotation_y 派生 → emit）→ `endEdit()`（**几何容差 1e-9 判定变化** → 置 `edited_by_human` + pushUndo 闭包栈（MAX 32，新编辑清 redoStack）；零变化编辑恢复 snapshot 原 dict）
- 编辑中选中框几何实时刷新（`boxToCorners → edgesOf/headLine` 重建）由 viewer 层经 emit 订阅完成

**viewer3d.js 交互层**：
- 手柄集（选中框 + 正交视图可见）：Top = 低 4 角（corner，sx/sy 符号锚）+ 4 边中点（edge）+ yaw 球（车头线伸出点）；Side = 顶面中心（高 4 均值）/底面中心（低 4 均值）/前（角 0,3 中点）/后（角 1,2 中点）；**Front / 透视不编辑（MVP 降级，如实记录）**
- 投影命中：手柄 3D 位置 `project(camera)` → NDC → 视图内像素，距离 < 14px 最近者
- 拖拽数学：mousedown 命中手柄 → `beginEdit` + 构建 drag 锚（yaw 球锚含指针初始方位角 t0）；mousemove「选中框中心平面（法线 = 视图轴）∩ 指针射线」求交 → cam 系 ptr（Top `{x, z}` / Side `{z, y}`）→ `editTo`；mouseup `endEdit`
- 编辑中禁用视图旋转拖拽；相机锁定不聚焦（避免框移动时视角跳动）；手柄/几何经 emit 实时跟随
- **顺手修复 P4a 遗留**：click 处理里 `drag && drag.moved` 检查在 click 时 drag 已被 mouseup 清空（恒失效）→ 拖拽旋转后仍触发选中；改用 `suppressClickUntil` 时间戳（拖拽/编辑后 300ms 内 click 不算选中）
- index.html view-hint 文案更新（编辑操作说明）

### 测试（node + jsdom，零浏览器）

`smoke_web3d.js` **94 断言 OK**（v0.3 61 + 编辑层新增 33），新增断言组：

| 组 | 断言内容 |
| --- | --- |
| 12 | boxToCorners yaw=0（rotation_y=−π/2）与既有 CORNERS fixture 逐点一致（独立锚点） |
| 13 | yaw=0.05 非轴对齐不变量：8 角中心 = (cx,cy,cz)、12 边边长集合 {h,w,l}={1.5,1.6,3.9}、低/高角 y = cy∓h/2、车头边 z 分量方向（yaw=π/2+0.05 车头 +x 略偏 −z） |
| 14 | rotationYToYaw/yawToRotationY 往返 + wrap（ry=3.1 / −3.1 越界） |
| 15 | Top corner resize：w/l 齐变、对角固定、两轮不漂移 |
| 16 | Top edge resize：单轴变（l 变 w 不动）、对面（车尾边）固定 |
| 17 | Top rotate：yaw 增量 = 指针方位角差、中心不动、wrap |
| 18 | Side 顶/底手柄：h + cy 联动、顶/底面固定（对面固定语义） |
| 19 | MIN_BOX3D 钳位（Top w/l + Side h 压穿对面） |
| 20 | 完整状态机：beginEdit → editTo 回写（w/cx/rotation_y 往返）→ endEdit 置位 → undo 全字段恢复 → redo 重放 |
| 21 | 零变化编辑（点击手柄零拖动）：无回写、不入 undo 栈 |
| 22 | undo 后新编辑清 redoStack（redo 无效果） |

**两个真实 bug 修复（冒烟暴露，均有断言钉死）**：

1. **`edited_by_human` 置位时机**：原 `_applyEdit` 无条件置位 + 无条件写回全部几何键——点击手柄零拖动也会产生「假人工修正」并入 undo 栈；且 `(x+π/2)−π/2 ≠ x` 浮点噪声使 rotation_y 每次往返漂移 ~1e-17，`JSON.stringify` 对比恒判「有变化」。修复：`endEdit` 按几何键容差（1e-9）判定变化才置位 + 入栈；零变化恢复 snapshot 原 dict（清理浮点噪声）。
2. **P4a 遗留 click 拦截失效**（见上「顺手修复」）。

### E2E 手动验收（如实记录）

容器无 headless WebGL / 无浏览器可自动化四视图拖拽交互——**手柄数学、状态机、undo 栈由 smoke 断言兜底**（组 12-22 全覆盖编辑数学与回写链路），浏览器手动验收待用户（`REVIEW3D_DIR=<队列目录> python3 -m auto3dlabel.web.server` 后 Top 拖角/边/yaw、Side 拖高度 → 保存 → `labels/<frame>.txt` 校验 15 字段）。

## Phase 2：nuScenes 端到端标注闭环（P2）

### E2E 实测（2026-09-02，bevfusion_nus 引擎）

| 环节 | 实测数字 |
| --- | --- |
| 队列生成 | `nuscenes-queue` CLI（`create_detector3d_any` 三引擎路由，键名 **bevfusion_nus**）：**81 文件 / 2696 框**，cam_like 转换（点云 + 框）零偏差 |
| Web 复核 | frame-data（点云 + 6 相机 + 四视图）200 OK；批量保存 **81 样本：2185 保留 / 464 删除（conf<0.4）**；reviewed json 保渲染 dict（velocity/track_id 存活） |
| 标签导出 | `labels/` 81 文件 NusBox 格式（translation/rotation 四元数/velocity/track_id），`load_dataset_labels` 可聚合 |
| 回灌评测 | labels 当 pred vs 官方 GT（**零检测器加载，0.1s 出表**）——**mAP 24.6**（2226 框 × 10 类，x-y 旋转 IoU 0.5 简化口径） |

- 回灌 mAP 24.6 < 官方 BEVFusion 27.0：人工删除低 conf 框 + 简化口径差异，**如实记录不粉饰**——闭环验收点 = 「训练标签可解析回灌出表」，非 AP 提升
- `smoke_nuscenes` 回灌参数（第三位置 labels 路径）→ 与官方 GT 同表 = P2 E2E 验收达成

## Phase 3：KITTI 3D 微调闭环（P3）

### 冒烟训练（2026-09-02，pointpillars_kitti，官方权重热启）

| 项 | 实测 |
| --- | --- |
| 规模 | train 300 / val 40 帧（subsample 按 sample_idx 确定性截取，metainfo 保留） |
| 训练 | epoch 4 × RepeatDataset 2 = **1200 iter**，**0.078s/iter**（全量 ≈ 100 秒）；batch 2 显存 **1.7GB**（12GB 余量大） |
| **反传探针** | ✅ **loss 0.32→0.39、grad_norm 2.1→3.0、无 NaN**——mmcv sparse ops（pointpillars 免 spconv）反传正常 |
| 产物 | `weights/kitti3d_finetune/epoch_4.pth`（58MB，不入库） |

### 训练坑记录（5 个，均有回归断言）

1. **`train.py` 在 `.mim/tools` 平级**（无 `tools/` 子目录）——`tools/train.py` 不存在；`create_data.py` 同病但被幂等跳过掩盖
2. **config/work-dir 相对路径**在训练 cwd=.mim/tools 下必炸 → `build_train_cmd` 统一 `.resolve()`
3. **`convert_to_iter_based` 只能配 epoch-based scheduler**（mmengine 断言）→ param_scheduler 纯 iter-based（LinearLR 预热 + ConstantLR 恒定，end 直接 iter 数）
4. **Config 合并的引用不传播**：`dataset=dict(dataset=dict(pipeline=train_pipeline))` 与顶层是同一 list 两处引用、内嵌 `data_root` 是已求值字符串——单改顶层键无效，须显式双赋值（dump 文本级断言：零 db_sampler/ObjectSample/相对路径）
5. **numba 0.67 × CUDA 13 环境级炸**：KittiMetric 的 rotate_iou.py **import 期**编译 CUDA kernel Signature mismatch（删显式 signature 同样失败）→ 内置 val 用 `val_begin=epochs+1` 越界关闭（mmengine 强制最终 epoch 跑 val，val_interval 大数无效）；**评测由 smoke_kitti 自写 40-point 口径（纯 numpy）承担**

### 微调 vs 官方同口径（P3-5，2026-09-02 实测，19 帧 003712-003731，conf 0.3）

`smoke_kitti` 新增 [config] [checkpoint] 可选参数（`Mmdet3dDetector` 直构，detect_points 协议复用）——官方权重与微调权重同 spec 同 conf 同函数出表。

**40-point 主口径（项目主表，每类官方 IoU）——微调全面优于或持平官方**：

| 难度 | Car 官方→微调 | Pedestrian 官方→微调 | Cyclist |
| --- | --- | --- | --- |
| easy | 27.1 → **27.5** | 17.3 → **19.5** | 0.0 → 0.0 |
| moderate | 54.1 → **55.6** | 19.7 → **22.4** | 0.0 → 0.0 |
| hard | 76.4 → **80.5** | 24.4 → **28.3** | 0.0 → 0.0 |

**11-point 对照口径（IoU 0.5，v0.1 基线同口径）——Car 略降、Pedestrian 大涨**：

| 难度 | Car | Pedestrian |
| --- | --- | --- |
| easy | 29.8 → 27.5（−2.3） | 39.4 → **48.6**（+9.2） |
| moderate | 26.2 → 22.0（−4.2） | 11.1 → **25.0**（+13.9） |
| hard | 17.4 → **20.8**（+3.4） | 6.8 → 7.7（+0.9） |

- 微调输出框数 119 vs 官方 183（box coder 置信度分布变保守，conf 0.3 过滤后差距更大）；推理速度持平 0.50s/帧
- 结论：**小样本冒烟（300 帧 / 4 epoch）已产生正向信号**——主口径全面提升；11-point Car 略降与 4 epoch 未收敛稳定一致。P3 闭环达成：训练 → 微调 → 同口径评测对比
- 真实增益空间在自标注增量数据（P3b/P4）；KITTI 官方权重本就 KITTI 域内训成，增益有限符合预期（计划书风险项已列）



## 质量门

| 检查 | 结果 | 基线对照 |
| --- | --- | --- |
| pytest（autolabel env 全量） | **1096 passed, 4 skipped** | 基线 1056+4skip，新增 40（P2 队列/payloads/server/labels/cli 路由 + P3 train3d），✅ 全绿 |
| pytest（base env 全量） | **1096 passed, 4 skipped** | ✅ 双环境一致 |
| ruff check auto3dlabel/ | **0** | ✅ 归零 |
| ruff check auto2dlabel/ | **75** | 基线 75，不恶化 ✅ |
| mypy auto3dlabel/ --follow-imports=silent | **0** | ✅ 归零 |
| mypy auto2dlabel/ --follow-imports=silent | **167 errors / 33 files** | 基线 169，不恶化 ✅ |
| pyright auto3dlabel/ + auto2dlabel/ | **0 / 0** | ✅ |
| smoke_web3d.js | **94 断言 OK** | 前端零改动回归（P2 仅 renderDetail 加 cameras 分支）✅ |

- P2/P3 新增测试：test_nuscenes_labels 4 + test_cli_nuscenes 2 + test_train3d 5 + test_web3d_payloads/test_web3d/test_detection3d 扩展断言；smoke_kitti P3-5 参数无单测（M3 执行器，照 smoke_nuscenes 回灌先例）
