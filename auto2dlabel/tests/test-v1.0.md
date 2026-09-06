# v1.0 Agentic 交互化实测记录

> 版本归属：Auto2dLabel v1.0 主线（P1 → P5）。企划与执行指南：`docs/Agentic_UI_plan.md` + `docs/AutoLabel_plan.md` §Agentic 交互化执行指南。
> 环境：纯 CPU 服务器（32 核 / 377GB，无 CUDA），Python 3.10 base env，textual 8.2.8。

## P1 验收（四件套全过）

| 验收项 | 结果 |
| --- | --- |
| `autolabel` 进对话 | ✅ 无参 → ChatApp TUI（Header + 对话区 + Input + Footer）；`autolabel "指令"` → route_domain 路由一次性对话 |
| 流式输出 | ✅ `LLMClient.chat(stream=True, on_delta=)`：OpenAI（stream_options include_usage + tool_calls 按 index 增量累积）/ Anthropic（messages.stream + get_final_message 共用尾部解析）；穿透链 dialog → planner×2 → cli 调用点×2（console.print 增量）；响应仍全量返回，json_mode 解析兜底不动 |
| /cost 出台账 | ✅ 斜杠命令 /cost 输出 usage 台账（`log_llm_usage` 统一出口未动，流式/非流式同落盘，回归断言锁死） |
| 无 key 降级 | ✅ TUI 内凭据横幅红字「无 API 凭据 → 指令将走代码直跑降级（LLM 规划/评估不可用）」；有 key 绿字 |

## 路由判据矩阵（route_domain，17/17 手动 + 23 用例）

- `-d` ∈ ENGINE3D_NAMES（四名单 frozenset 并集）→ 强制 3D，优先级最高
- 3D 强特征词：点云/lidar/velodyne/3d bbox/3d框/3d检测/3d标注/nuscenes + 独立 6 位帧号（`(?<![0-9A-Za-z])\d{6}(?!\s*\.(?:png|jpe?g|tiff?|bmp|webp)\b)` 排除图像文件名）
- 「KITTI」非 3D 特征词（`kitti_finetune` 是 2D 模型）；未知模型名/其余 → 2D（宁 2D 勿错）

## 质量门（本机新工具版本基线，均不恶化）

| 项 | 数字 |
| --- | --- |
| pytest | **1129 passed, 4 skipped**（基线 1128+1 修复 referential_l3 环境依赖 → 新增 P1 33 用例：route 23 + stream 6 + TUI 4） |
| smoke_tui | **11/11 断言**（对标 smoke_web.js 模式，run_test + Fake runner 零真实执行） |
| pyright | 0 / 0 / 0 |
| mypy | 174（基线 183，stash 对比不恶化；textual py.typed partial → App/work 行级 `# type: ignore`） |
| ruff | P1 改动文件 **0**；全项目 74 = 基线 83 − 9（P1 清理） |

## 关键坑（有测试锁死）

1. **Textual 8.2.8 顶层无重导出**：组件在私有子模块（`textual.widgets._input.Input` / `_static.Static` / `_footer.Footer`）；Footer 无 `.content`、App 无 `.bindings`
2. **SVG 空格编码 `&#160;`**：export_screenshot 断言只用无空格片段（中文直接 UTF-8）
3. **长内容滚动出可视区**：SVG 断言不可靠 → 组件树 children 内容断言（smoke ④）
4. **流式 Fake 兼容**：10 处测试 Fake 覆写 `chat()`（duck-typing 旧 6 参数）→ 调用点条件传参（`on_delta is None` 时不传 stream）；`_UsageClient._chat` 签名同步
5. **Anthropic 截断规整**：stream 路径 stop_reason=max_tokens → finish_reason=length（与尾部解析同口径）

## 实测 bug 修复（2026-09-02 用户反馈「TUI 无响应」）

**现象**：`autolabel` 输入「我想对 nuscenes-mini 随机 100 张进行 bev 预测…推荐模型」→ 无任何回显。

**两层根因**：

1. **TUI 无回显**：`_subprocess_run` 的子进程 stdout 连 pipe → Python 全缓冲，输出积压到进程退出才 flush。修：cmd 加 `-u`（无缓冲，回归断言锁死）。
2. **LLM 空响应**：`.env` 配的 `DEEPSEEK_MODEL=deepseek-v4-flash` 是**推理模型**——planner 低温确定性 JSON 请求（temperature=0）下推理链吃满 max_tokens 产出空 content（实测 temp=0.1 → reasoning 1023/content 0；temp=1.0 → 569/834 仍高方差）。修：① `.env` 改回 `deepseek-chat`（v0.6 实测基线，规划任务无需推理模型；注释警示保留）；② `chat()` 统一出口钳制（推理模型 temperature→1.0 + max_tokens≥4096，非推理模型零影响，3 用例锁死）。

**修复后终态**（实测）：流式 JSON 直出（`LLM 解析中: {...}` 增量可见）→ 缺参清单「缺少参数: frame_id（KITTI 帧号）, prompts（检测类别）」exit=2 清晰可见。注：该指令属咨询类 + 批量 nuScenes，超出 3D chat 单帧 KITTI 语义——批量走 `auto3dlabel nuscenes-queue`，纯咨询类对话 P1 不支持（P2+ 议题）。

## 批量 nuScenes 扩展（v1.0 P1+，2026-09-02 用户选定方向）

上文「咨询类 + 批量」指令经用户确认扩展为 **3D chat 批量 nuScenes 直跑**（用户原指令「对 nuscenes-mini 随机 100 张做 bev 预测 + 推荐模型」在 TUI 内跑通）：

- **Plan3D 批量形态**：`dataset: "kitti"|"nuscenes"` + `sample_limit: int|None`（随机抽样 seed=42 确定性，None=全量 81）。nuscenes 任务 `missing_params == []`（类别固定 10 类、样本集全量/抽样）。sanitize 白名单守卫：dataset ∉ 白名单回 kitti（宁单帧勿误入批量）。
- **chat 分派**：`plan.dataset == "nuscenes"` → `_run_nuscenes_batch`（直调队列管线，不走单帧 run_3d_agent）；`nuscenes-queue` 子命令同步加 `--limit`。抽样先于 resume（幂等：同 seed 续跑补产同子集）。
- **prompt 规则**：nuscenes 批量意图 → dataset=nuscenes/frame_id ""/prompts []；「推荐/不知道用什么模型」→ `bevfusion_nus`（mAP 27.0 最高）。
- **CPU 守卫（纯 CPU 服务器实测）**：bevfusion 系/centerpoint 依赖 CUDA 自定义 op（bev_pool/spconv），CPU 推理炸 `CUDAGuardImpl initialized with non-CUDA DeviceType: cpu`（E2E 实测）；`_cpu_safe_nuscenes_engine` 无 CUDA 时自动降级 `pointpillars_nus`（pillar 架构无自定义 op，CPU 实测 1 样本跑通出队列文件）。`CUDA_ONLY_NUSCENES_ENGINES` 常量在 model_catalog（单一事实源）。
- **实测**（E2E 用户指令「对 nuscenes-mini 数据集随机 2 张进行 bev 预测，帮我推荐模型」，真实 LLM + 真实权重，纯 CPU）：规划 `dataset=nuscenes limit=2 conf=0.3 det=bevfusion_nus`（推荐规则命中）→ CPU 守卫降级提示 → pointpillars_nus 2 样本推理 → `reviews/` 2 个队列文件。抽样确定性佐证：limit=1 探针产物 `c59e6004…` ∈ limit=2 子集（同 seed 子集扩展）。CPU 单样本 pointpillars 推理约 1-2 分钟——**100 张全量 CPU 推理预计 2-4 小时**，建议 `-o` 指定目录 + `/quit` 后重跑 resume 幂等续跑。

## P2–P5 连续交付（2026-09-06，逐 Phase 质量门全绿才进下一 Phase）

### P2 provider 注册表

- `auto2dlabel/configs/providers.yaml`（入库，pyproject package-data 打包，`pip wheel` + zipfile 断言进包）：deepseek / openai / anthropic / ollama（本地端点示例）四条目——`api_key_env` 只引用环境变量名（绝不含密钥），`api_key_env: null` 即本地免 key（SDK 占位 "EMPTY"，`has_credentials` 恒 True）
- `llm.py`：`ProviderSpec` frozen dataclass + `load_provider_config`（损坏/缺失 → 内置表回退**绝不 raise**）+ `list_providers`（yaml ∪ 内置按 name 去重）+ `create_client` 模型四档解析（显式 > model_env 的 env > spec.model > 内置默认）；单价表留代码（账单口径不漂移），yaml `price` 为扩展查找顺序
- TUI `/model` 三态：无参列出（当前项 `*` 标记）/ `set <name>`（校验存在）/ `set-key`（.env 绝对路径 + 变量名指引，红字「密钥不入库」）；runner 契约一次升级到 5 参 `(instruction, domain, provider, on_line, on_progress)`，子进程透传 `--provider`
- 台账：`aggregate_usage` 键加 provider、行加 `"provider"` 字段，旧行 `e.get("provider", "-")` 容错；`/cost` 行格式 `call_site [provider/model]`
- **红线修复**：`auto3dlabel/configs/.env`（含真实 key）`git rm --cached` + `.gitignore` 补条目 + `.env.example` ×2（去 key 模板）；key 已进 git 历史（5b8768d 进入），**轮换与否用户决定**
- 新增 21 用例（test_provider_registry.py）；质量门 pyright 0 / mypy ≤169 / ruff P2 文件全绿 / pytest 1170 passed / smoke_tui 16/16

### P3 后台任务面板（差异化核心）

- 执行路径保持 **subprocess**（进程隔离：torch 权重不污染 TUI 进程，子进程崩 TUI 存活；取消的显存/内存由 OS 即刻回收）
- `[AL_PROGRESS] done/total` stdout 行协议 4 挂点：`cli_run.run_command` 图像循环 / `cli_commands.execute_plan` 批量单图分支 / `TrackingTool.forward` 帧循环（经 progress_cb）/ `auto3dlabel/cli.py` chat 帧循环——协议行进任务面板不进对话区
- `TrackingTool.forward` 加 `progress_cb(done,total)` / `cancel_event`（帧循环 + `_detect_frames` chunk 循环检查，置位抛 `TrackingCancelled`（ValueError 子类，收敛既有 except 链，finally video_writer.release 天然执行））；子进程 SIGTERM → KeyboardInterrupt 触发 finally 链
- `/cancel`（无参 = 全部；`<id>` = 单任务）：`_ACTIVE_PROCS` 登记表按指令匹配 → terminate → wait(5s) → TimeoutExpired → kill 两级；取消后引导行：manifest 由 TUI 倒序扫 stdout 缓存 100 行提取 `--resume <manifest>`（跟踪模式提示「重跑即可」）
- 任务面板：指令截断 40 字 + ProgressBar（total=None 不确定态 → `bar.total=N` 转确定态）+ 状态行 `3/120`；多任务并发不限；on_progress 经 `call_from_thread` 主线程更新
- 新增 14 用例（test_tui_tasks.py）+ 9 用例（test_tracking_progress.py，FakeDetector 零权重）；质量门 pyright 0 / mypy 167 / ruff 86 存量（P3 文件全绿）/ pytest 1239 passed / smoke_tui 22/22

### P4 会话管理

- `logs/chat_sessions.jsonl` 单文件逐行 append（llm_usage.jsonl 先例；logs/ 已 gitignore）：`{session_id, seq, ts, kind: user|assistant|system|task_result, text, domain, provider, markup, task:{id,status,manifest}}`
- `append(text, record=True)`——子进程回显 `record=False` 防膨胀；损坏行跳过、旧行缺字段 `.get` 默认值容错
- `/new`：新 session_id（`%Y-%m-%dT%H-%M-%S`）+ 清屏 + seq=0；`/resume` 两态：无参列最近 5 会话（首条 user 截断 40 字预览）/ `<session_id>` 逐行重放（markup 保真，task_result 且 manifest 非 null 且 status != done 附续跑引导行）；**重放不重跑任务**（subprocess 已死，重跑有重复副作用），续跑走 manifest 引导；恢复后 `_session_id/_seq` 切换续写同会话
- 新增 15 用例（test_tui_sessions.py，autouse fixture 重定向 SESSION_LOG_PATH 防写真实 logs/）；质量门 pyright 0 / mypy 167 / ruff P4 文件全绿 / pytest 1254 passed / smoke_tui 27/27

### P5 HITL 指挥台

- `/review` 三档分流统计（每次重扫零缓存）：2D `outputs/*_review.json` summary（review_count/hard_count，**无 accepted 档如实只展示两档**）+ 3D `outputs/kitti3d/reviews/*_review.json`（含 accepted 三档）；`*.reviewed` 侧车排除；损坏/缺 summary JSON 跳过（collect_review_pool 同款容错，不复用其代码——语义是评分排序 vs 计数汇总）；已复核口径 2D = reviewed_files、3D = accepted
- `/web [2d|3d]`：端口 env `AUTOLABEL_PORT`(8765)/`AUTOLABEL3D_PORT`(8766)（server.py 自身读，webctl 只检测/轮询）→ 已占用「已在运行」/ 未占用 spawn detach（DEVNULL ×3，脚本路径走 `web/server.py`——web/ 无 `__init__.py` 不能包内调用）→ 10s 轮询 → URL + `kill <pid>` 指引
- 回流链路零改动：标注 → /review → /web 复核 → 保存写 `*_reviewed.json`（server.py 既有链路）→ /review 重扫已复核 +1
- 新增 13 用例（test_tui_command_center.py，Fake Popen 记录 cmd 零真服务）；质量门 pyright 0 / mypy 168 ≤169 / ruff P5 文件全绿 / pytest 1267 passed / smoke_tui 31/31

### P2–P5 终态质量门

| 项 | 数字 |
| --- | --- |
| pytest | **1267 passed**（P1 1129 → P2 1170 → P3 1239 → P4 1254 → P5 1267） |
| smoke_tui | **31/31**（P1 11 → P2 16 → P3 22 → P4 27 → P5 31） |
| pyright | **0** |
| mypy | **168**（基线 169 不恶化，命令写全三包） |
| ruff | 全项目 86 = 存量 benchmark 文件（≤163 容忍线内），P2–P5 改动文件 **0** |

### P2–P5 关键坑（有测试锁死）

1. **`remove_children()` 延迟语义**：返回 AwaitRemove 延迟到下一次 pump——remove 后 append 的新行不受影响，但立即查询 children 看不到删除效果（/new 清屏断言需先 `await pilot.pause()`）
2. **取消置位在主线程、worker buf 不可得**：取消路径 task_result 不附 manifest（对比失败路径从 stdout 缓存提取）
3. **`_cancel_procs` 不移除登记**：出表靠 `_subprocess_run` finally（登记表是进程生命周期簿记，不是取消队列）
4. **2D review summary 无 accepted 档**：/review 口径必须按两侧 schema 如实分档（2D 两档 + 3D 三档），不能统一三档模板
5. **smoke 注入只换属性不换模块**：mypy 不允许 lambda 赋值带类型变量 → 具名函数 + 用后恢复原属性（防常驻进程）
