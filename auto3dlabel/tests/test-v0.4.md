# auto3dlabel v0.4 实测记录

> 本文件 = v0.4 实测数据（合成测试计数见各节；里程碑定义见 `milestone/v0.4.md`）。
> **环境**：RTX 3080 Ti 12GB 多租户宿主（本版纯前端，无 GPU 项）。

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

## 质量门

| 检查 | 结果 | 基线对照 |
| --- | --- | --- |
| pytest（autolabel env 全量） | **1056 passed, 4 skipped** | ✅ 全绿（Python 零改动确认） |
| pytest（base env 全量） | **1059 passed, 1 skipped** | ✅ 全绿 |
| ruff check auto3dlabel/ | **0** | ✅ 归零 |
| ruff check auto2dlabel/ | **75**（89 → `--fix` 14 清理项 → 75） | 上轮记录 75，基线 119，不恶化 ✅ |
| mypy auto3dlabel/ --follow-imports=silent | **0** | ✅ 归零 |
| mypy auto2dlabel/ --follow-imports=silent | **167 errors / 33 files** | 基线 169，不恶化 ✅ |
| pyright auto3dlabel/ + auto2dlabel/ | **0 / 0** | ✅ |

- ruff --fix 14 项全为无行为清理（删未用 import `json/compute_iou/DATASETS_ROOT`、去冗余 f 前缀、PIL import 排序），pytest 全量复跑仍全绿
- 新增断言计数：smoke_web3d.js 61 → **94**（编辑层 33）
- 质量门在 --fix 后复跑 pytest 全量确认（改动无行为影响）
