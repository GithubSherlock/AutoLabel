# Auto3dLabel v0.2 实测数据

> 零真实权重铁律：合成测试全绿 + 真实权重冒烟单独记录（本文件）。
> 冒烟执行：`python3 -m auto3dlabel.benchmarks.smoke_kitti 003712-003731 pointpillars_kitti 0.3`

## 环境与工具链（2026-08-26 打通）

- CUDA 13 工具链：conda nvidia 频道 cuda-nvcc 13.0.88 / cuda-cudart 13.0 / cuda-cccl 13.0（CUDA_HOME=/root/miniconda3/targets/x86_64-linux）
- 踩坑：thrust/cub 头文件 ln 到 include 顶层；torch 自带 cu13 头 151 个 ln 到 CUDA_HOME include；cicc/ptxas/fatbinary/nvlink ln 到 nvcc 同目录；**编译必须 `PATH=$CUDA_HOME/bin`**（nvcc 子工具经 sh 按 PATH 查找）+ `-I$CUDA_HOME/include`（nvcc 13 不自动搜 conda split 布局的 include）
- **--no-deps 红线**：mmcv/mmdet3d 安装不带 --no-deps 会把 numpy 升回 2.2.6（破坏 matplotlib/ultralytics 的 numpy 1.x ABI）；环境 numpy 固定 1.26.4

## M1：mmcv/mmdet3d 冒烟

- mmcv 2.1.0 **源码编译**成功（cu13 nvcc 工具链，配方见里程碑；`--no-deps` 红线保 numpy 1.26.4）；`from mmcv.ops import Voxelization` import OK
- mmdet3d 1.4.0 + mmdet 3.3.0 + mmengine 0.10.7 安装（均 `--no-deps`）；`LiDARPoints/LiDARInstance3DBoxes/init_model/inference_detector` import OK
- 双态：mmdet3d 未装时 `create_detector3d` 返回 None / detect 抛 ImportError 守卫（tests 全绿）

## M3：KITTI 20 帧 PointPillars 冒烟（双口径）

帧 003712–003731（19 帧）/ conf 0.3 / 19 帧 183 框 0.47s/帧

**坐标系 bug 记录（修复后 AP 0→非零）**：mmdet3d KITTI 模型输出 LiDAR 系 **z=底面中心**（origin=(0.5,0.5,0)），detection3d.py 曾误传 origin=(0.5,0.5,0.5)（中心语义）→ convert_to 多做一次底面换算 → y_cam 凭空 +h/2 → 3D AP 全 0（BEV 不受影响非零，症状指向 y）。修复：`LiDARInstance3DBoxes(boxes, origin=(0.5,0.5,0.0))`。

| 口径 | 难度 | Car | Pedestrian | Cyclist |
| --- | --- | --- | --- | --- |
| 官方 40-point（IoU 0.7/0.5/0.5） | easy | 29.8 (12) | 39.6 (13) | 0.0 (0) |
| | moderate | 19.5 (18) | 11.1 (1) | 0.0 (1) |
| | hard | 17.2 (16) | 6.4 (4) | 0.0 (0) |
| 11-point（IoU 0.5）3D | easy | 29.8 (12) | 39.4 (13) | 0.0 (0) |
| | moderate | 26.2 (18) | 11.1 (1) | 0.0 (1) |
| | hard | 17.4 (16) | 6.8 (4) | 0.0 (0) |

- 单帧核对（003712）：conf 0.94 的 Car vs GT IoU **0.824**（easy TP）；conf 0.85（44m 远车）IoU 0.679（0.7 阈值差一点 → 官方 moderate 19.5 < 11-point 26.2 的成因）；3D==BEV（IoU 0.5 下匹配集一致，y 对齐后自洽）
- 20 帧全为 20–50m 远车场景，Car moderate 19.5 未达 ≥74 验收线——zoo 77.6 为全 val 3769 帧口径（含大量近车），终版数字看 M4

## M4：全 val 3769 帧终版数字

（跑完后填：官方口径三难度 + zoo 对照表）

## M5：KITTI 20 帧跟踪冒烟

（跑完后填：IDSW / 轨迹长 / velocity vs GT 差分 MSE）

## M7：nuScenes Mini 简化评测

数据：v1.0-mini val 2 场景（Mini 单卷解压，口径差异见 milestone/v0.2.md Phase 3）

（出表后填）
