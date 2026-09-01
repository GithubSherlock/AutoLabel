# AutoLabel Agentic 交互化企划（v1.0 方向）

> **状态：企划文档（未来方向，未立项）**——2026-09-01 用户讨论定案：从「Python 包 + CLI 命令」进化为「以智能体驱动的自动标注软件」，形态对标 Hermes / Claude Code：输入 `autolabel` 即进入对话式终端界面，可连接多种大模型 API。
>
> 四项定案决策（2026-09-01 AskUserQuestion）：
> 1. **交互形态**：Textual TUI（全屏终端对话界面）
> 2. **版本归属**：Auto2dLabel **v1.0 主线**（与 Auto3dLabel v0.4 并行，标注能力已就位不另起炉灶）
> 3. **文档落点**：本文档（独立企划），主计划书 `AutoLabel_plan.md` 版本路线挂引用
> 4. **HITL 形态**：**TUI 指挥台 + Web 复核**——TUI 做对话/任务/结果概览与入口，复核本体仍在现有 Web（2D :8765 + 3D /3d/ 四视图 + 编辑），不重复造轮子

---

## 一、背景与目标

AutoLabel 现状：`auto2dlabel` / `auto3dlabel` 两个 typer CLI 命令，能力侧已很完整（2D 84 模型 + 3D 9 引擎 + Agentic 闭环 + HITL 三档 + Web 复核 + 成本台账），但**交互侧是「命令式」**：用户敲一条指令看一段输出，Agent 循环（planner → orchestrator → evaluate）在幕后黑盒运行，不可见、不可中断、不可对话。

**目标**：进化为类 Claude Code 的交互式智能体软件——

```
$ autolabel                    # 无参 → 进入全屏对话界面
╭─ AutoLabel ─────────────────────────────────╮
│ ┌─ 对话区 ──────────────────────────────┐   │
│ │ ✻ 已解析：检测 000860.png 中的汽车     │   │
│ │ ⟳ 调用 detect_objects(000860.png)…   │   │
│ │ ✓ 3 个对象（car 2 / person 1）        │   │
│ └───────────────────────────────────────┘   │
│ ┌─ 任务面板 ────────────────────────────┐   │
│ │ ████████░░ 批量标注 12/40 帧  2.3s/帧 │   │
│ └───────────────────────────────────────┘   │
│ > 继续检测右侧的视频…                    │   │
│ /model  /cost  /new  /resume  /help      │   │
╰──────────────────────────────────────────────╯
```

- **输入**：`autolabel`（交互）、`autolabel "指令"`（一次性对话）、`autolabel run …`（兼容保留现有子命令）
- **多模型**：openai / anthropic / deepseek + 任意 OpenAI 兼容端点（Ollama / vLLM / 本地 Qwen）可配置、会话内切换
- **智能体**：复用现有 Agent 循环，从黑盒变为**可见、可中断、可续跑**

## 二、现状资产盘点（直接复用，不重造）

| 目标能力 | 现状资产 | 差距 |
|---|---|---|
| 多模型 API | `agent/llm.py` `create_client(provider)`：openai / anthropic / deepseek 三类 + `base_url` 任意兼容端点（Ollama/vLLM 即此形态）；usage 台账 + 费用折算 + call_site + json_mode + max_tokens | **无流式输出**；provider 硬编码 dict、无配置化管理 |
| 无 key 零影响 | `has_credentials` 判定 + 全链代码级兜底（v0.6 红线，`configs/.env`） | ✅ 已内建 |
| Agent 循环 | `planner.py`（JSON 三级解析）+ `orchestrator.py`（tool-use + 质量门 + 防重复调用 + max_iterations）+ `dialog.py`（多轮对话骨架）+ `evaluate.py`（代码级判据） | 循环**不可见不可中断**；无流式/进度/取消通道 |
| 成本台账 | `logs/llm_usage.jsonl` + `cost-report` + 前缀缓存命中率（v0.6 Phase 4） | ✅ 已内建，TUI 内 `/cost` 直接消费 |
| 长任务 | batch `--resume` manifest + 失败隔离 + 动态实测 batch + `AgentState` 快照续跑 | 终端内无进度/取消/续跑 UI |
| 复核界面 | Web 复核完整（2D 编辑 + 3D 四视图手柄编辑 + 三档分流队列） | 与 TUI **未打通**（无入口/无概览） |
| 终端渲染 | `rich 15.0.0` 已装（Textual 同生态，同作者） | Textual 未装 |

**结论**：企划的工程重心不在「能力」（已 84 模型 + 9 引擎），而在**交互外壳 + 可见性 + 可中断性 + 配置化**。

## 三、技术选型论证

### 1. 交互层：Textual TUI（定案）

- Claude Code 本身即 Textual 实现——目标形态有成熟先例
- Python 原生、与已装 rich 同生态、pip install 即用、纯终端零服务端（不破纯本地红线）
- 组件齐备：对话区（ScrollableContainer/MessageLog）、输入区（Input）、状态栏（Footer）、任务面板（ProgressBar/Worker）

**本项目定制点（与 Claude Code 的本质差异）**：Claude Code 的工具是毫秒级函数调用；**我们的工具是秒~分钟级推理任务**（检测/批量/跟踪/SAM）。因此 TUI 不能只是对话——必须带**后台任务面板**：任务进 Worker 后台跑，对话区可继续输入，任务面板显示进度/可取消/可断点续跑。这是 v1.0 的差异化卖点，也是最大的工程增量。

### 2. LLM 层：流式扩展（改动最小的关键项）

- `LLMClient.chat()` 加 `stream: bool` 参数 + 增量回调（openai / anthropic SDK 原生 streaming）
- **红线**：流式只影响展示路径，**usage 台账统一出口不变**（流式结束仍走 `log_llm_usage`）；json_mode 流式仍等完整响应再解析（展示流式、解析兜底全量，防半截 JSON）
- provider 注册表：硬编码 dict → `configs/providers.yaml`（name / provider / base_url / model / api_key env 引用），`/model` 命令列出与切换

### 3. 入口层：`autolabel` 统一命令（2026-09-01 评审定案，见 §九）

- pyproject 新增 console script `autolabel = "autolabel.cli:app"`（薄壳，代码级关键词路由 2D/3D）；发行名 `name` 同步改为 `autolabel`
- `autolabel`（无参）→ TUI；`autolabel "指令" [公共选项]` → 一次性对话（route_domain 路由 → 直接调用现有 2D/3D 对话函数，零复制）
- **P1 不透传子命令**（评审修正）：run/sample/cost_report/dataset 保持 `auto2dlabel` / `auto3dlabel` 旧命令不动（防双路由复杂度 + 旧脚本零破坏），后续版本逐步吸收
- 旧入口 `auto2dlabel` / `auto3dlabel` 保留（防既有脚本 break）

### 4. HITL 层：指挥台 + Web 复核（定案）

- TUI 提供结果概览表格（帧/类/conf/三档分流统计）+ 一键启动 Web 复核（spawn server + 端口检测 + 打印 URL）
- 复核本体不重复实现——Web 已成熟（编辑/手柄/快捷键/undo）

## 四、分期里程碑（企划，未实施）

### P1：TUI 外壳 + 流式对话（最高优先）

- Textual App 骨架：对话区 / 输入区 / 状态栏；斜杠命令框架（`/model /cost /new /resume /help /quit`）
- `LLMClient.chat` 流式改造（增量渲染 + 台账兼容 + 截断可见性不回归）
- 无 key 零影响在 TUI 内保持：对话区显示「无凭据 → 代码直跑」降级提示
- **验收**：`autolabel` 进入对话、流式输出、`/cost` 显示台账、无 key 时全流程可完成

### P2：多模型连接配置化

- `configs/providers.yaml` provider 注册表 + `/model` 列出/切换（含本地 Ollama/vLLM 端点）
- 密钥管理提示（`/model set-key` → 指引编辑 `configs/.env`，密钥不入库红线不变）
- **验收**：切换任意已配置 provider 生效；无 key provider 启动即降级提示；usage 台账按 provider 区分（`estimate_cost_rmb` 扩展）

### P3：后台任务面板（本项目差异化核心）

- 工具 Worker 化：Tool 调用进 Textual Worker，检测/批量/跟踪任务显示进度（`progress_cb`）、可 Ctrl+C 取消（`cancel_event`）、断点续跑（复用 batch `--resume` manifest 与 `AgentState`）
- 工具契约扩展需过 `tools/registry.py`（同步 forward 现状 → 可注入进度/取消回调），**同步路径兼容不破坏**
- **验收**：批量标注在 TUI 内显示进度、可取消、取消后 `--resume` 续跑；结果进对话区

### P4：会话管理

- 会话 JSONL（消息历史 + 任务记录），`/new` 新会话、`/resume` 恢复上下文（对标 Claude Code `--resume`）
- **验收**：中断会话后 `/resume` 恢复对话上下文与任务状态

### P5：HITL 指挥台打通

- TUI 结果概览（三档分流统计表）+ 一键启动 Web 复核（spawn + URL 提示 + 端口占用检测）
- **验收**：标注完成 → TUI 内看分流 → 回车开 Web → 复核保存回流（走既有 `*_reviewed.json` 链路）

### 远期（不承诺，如实记录）

- MCP 客户端/服务端接入（工具注册表 → MCP 工具，Claude Code 生态互通）
- TUI 内嵌轻量三档确认（密集复核场景少开浏览器）
- 插件体系（自定义 Tool/模型/provider 热插拔）
- 跨机器 GPU 任务编排（Mac 指挥 → AutoDL 服务器执行，3D 训练/大批量）

## 五、红线（继承 + 新增）

- **继承**：纯本地（Textual 零服务端、不引入 Node）；无 key 零影响；权重与 `.env` 不入库；质量门 pyright/mypy/ruff/pytest 全绿；复用不复制（Agent 循环/Web 复核/成本台账全复用）
- **新增（草案，立项时评审）**：
  1. **流式不破坏台账统一出口**——`chat()` 签名向后兼容，usage 记账路径零改动
  2. **TUI 逻辑可测**——Textual 提供 async headless 测试（`run_test` + Pilot），零真实权重铁律沿用（Fake 注入）；前端改动必跑 Pilot 冒烟（对标 jsdom 冒烟铁律）
  3. **长任务取消必须清理**——推理中断需释放显存/终止子进程，宁失败不悬挂
  4. **双入口并存期**——`autolabel` 与旧命令透传并存，防既有脚本 break

## 六、风险与开放问题

| 项 | 说明 | 对策 |
|---|---|---|
| Textual 版本锁定 | 8.x 与 rich 15 兼容性 | ✅ 已实测（2026-09-01，§8.1）：rich 零升级；锁 `textual>=8.2.8,<9` |
| 流式 + JSON 解析 | json_mode 流式半截 JSON | 展示走流式、解析走全量兜底（现状解析路径不动） |
| 取消语义 | 推理中途取消的显存/进程清理 | P3 设计取消事件穿透（detect 层 + 进程级兜底） |
| 工具契约变更面 | registry 同步 forward 现状改动 | 进度/取消回调做成可选注入，同步路径兼容 |
| 双入口混乱 | TUI 与子命令并存 | ✅ 已定案（2026-09-01，§9.1）：P1 只做对话形态，子命令保留旧入口，零双路由 |
| 3D 侧整合 | 3D chat 已有（v0.3 P1 对话 planner） | v1.0 入口统一后按 task_type 路由 2D/3D planner，3D 零新增逻辑 |

## 七、立项前提（满足才开工）

1. Auto3dLabel v0.4（P2 nuScenes 闭环 / P3 微调）收尾或明确挂起——v1.0 与 3D v0.4 并行但资源不打架 —— **✅ 已挂起（2026-08-31 用户决策：P2 暂停、P3 待 GPU 服务器）**
2. Textual 依赖评估落盘（版本锁定 + 与现有 rich 兼容性冒烟）——**✅ 已评审通过（2026-09-01，见 §八）**
3. `autolabel` 入口命名与旧命令透传方案评审 —— **✅ 已评审通过（2026-09-01，见 §九）**

> **立项前提三项全部满足（2026-09-01）**——v1.0 P1 可立项，开工须用户显式确认。

## 九、立项评审记录：`autolabel` 统一入口方案（2026-09-01）

> 评审结论：**通过，v1.0 P1 可立项**。三项定案（AskUserQuestion）：① 路由判据 = 代码级关键词；② P1 只做对话形态（不透传子命令）；③ 发行名改名为 `autolabel`。

### 9.1 命令矩阵（P1）

| 命令 | 行为 |
|---|---|
| `autolabel`（无参） | 进入 Textual TUI（对话界面） |
| `autolabel "指令" [公共选项]` | 一次性对话：`route_domain()` 路由 → 直接调用现有对话函数（2D `cli_commands.chat_command` / 3D `auto3dlabel.cli.chat` 命令函数可编程调用），**零复制** |
| `auto2dlabel …` / `auto3dlabel …` | 旧命令原样保留（run/sample/cost_report/dataset/chat），零破坏 |

公共选项面 = 2D chat 参数超集 + 3D 独有（`--out-dir` / `--max-iterations`）；`-d` 命中 `DETECTOR3D_NAMES` 强制 3D。

### 9.2 包结构

```
autolabel/                # 新顶层包（薄壳，不入 agent/models 逻辑）
├── __init__.py
├── cli.py                # typer app：无参 → TUI；指令参数 → 一次性对话
├── route.py              # route_domain() 单一事实源（关键词表 + 3D 模型名命中）
└── tui/app.py            # P1 Textual App 骨架（对话区/输入区/状态栏/Worker）
```
pyproject：`name = "autolabel"`（发行名），`[tool.setuptools.packages.find] include` 加 `autolabel*`，`[project.scripts]` 加 `autolabel`。`pip install -e .` 重装即可；`import auto2dlabel/auto3dlabel` 不受影响（模块名与发行名独立）。

### 9.3 路由判据设计（route.py，含边界修正）

**「KITTI」不能作为 3D 强特征词**——auto2dlabel 有 KITTI 域 2D（`kitti_finetune`/`yolo11s_kitti` 是 2D 模型），「检测 KITTI 数据集的汽车」是合法 2D 任务。判据表（单一事实源，单测覆盖）：

| 判据 | 路由 |
|---|---|
| `det_model` ∈ `DETECTOR3D_NAMES`（pointpillars/pvrcnn/centerpoint/bevfusion/fcos3d 等） | 强制 3D |
| 指令含 3D 强特征词：点云 / lidar / velodyne / 3d( bbox|框|检测) + 6 位帧号（`\d{6}`） | 3D |
| 指令含 `kitti` 但无 3D 强特征、`-d` 为 2D 模型 | 2D（默认域宁 2D 勿错） |
| 其余 | 2D（默认） |

### 9.4 兼容与测试

- **兼容**：旧命令原样保留；发行名改名后重装 editable 即可，import 路径零变化；TUI 内执行链复用 `create_client` / dialog / planner（无 key 零影响不回归）
- **测试**：`route_domain()` 纯函数单测（关键词矩阵用例，含「kitti 域 2D 不误判 3D」边界用例）；TUI 骨架 Pilot 冒烟（§8.2 铁律）；一次性对话用 Fake LLM 注入（沿用零真实权重铁律）
- **对 P1 工程影响**：3D chat 命令体（auto3dlabel/cli.py:472-591）为可编程调用可直接 import，无需抽函数；2D 侧 `chat_command` 已是普通函数——**两端零重构**，P1 新增代码全在 `autolabel/` 包内

## 八、立项评审记录：Textual 版本评估（2026-09-01）

> 评审结论：**通过，v1.0 P1 可立项**。实测于 conda `sl` 环境（Python 3.11.13 / rich 15.0.0 / macOS）。

### 8.1 版本锁定与依赖

| 项 | 结论 |
|---|---|
| 版本锁定 | **textual==8.2.8**（2026-09-01 PyPI 最新；写码以实测 API 为准，见 8.3） |
| Python | `requires-python: <4.0,>=3.9` → 3.11.13 ✅ |
| rich 兼容 | `rich>=14.2.0` 已满足（15.0.0）→ **零升级、零降级**（dry-run 实证） |
| 新增依赖 | 仅 4 个纯 Python 包：textual / Pygments / linkify-it-py / mdit-py-plugins，**零编译** |
| 不装项 | `[syntax]` extra（tree-sitter 系列 15 包）不装——纯本地依赖最小化 |

### 8.2 冒烟结果（headless，`/tmp/textual_smoke.py`）

`App.run_test()` + Pilot 全通过：组件挂载（query_one）/ 聚焦 / 输入值写入 / Footer 渲染 / 动态更新（Label.update 后导出含新文本）。**测试 API 确认可用**：`run_test` / `export_screenshot` / `deliver_screenshot` / `run_worker` / `workers`（P3 任务面板的 Worker 能力实证存在）。零真实权重铁律在 TUI 侧等价物 = `run_test` + Fake 注入（Pilot 驱动）。

### 8.3 Textual 8.x API 重构实测发现（写码必读，勿凭旧版记忆）

冒烟暴露 8.x 相对旧版（0.x/6.x）的破坏性变化——写 P1 前核对以下清单，避免照旧文档写崩：

1. **Footer 无 `.content`**（旧版属性移除）——验证 Footer 改用 `export_screenshot()` 屏幕断言
2. **App 无 `.bindings`**（已内化为 `_bindings`）——键绑定查询走其他途径，勿直接访问
3. **Label 无 `.renderable`**（移除）——渲染断言统一走 `export_screenshot()`
4. **`export_screenshot()` 返回 SVG**（富终端格式）——**空格被 HTML 实体编码为 `&#160;`**，字符串断言须用无空格片段（如 `"000860.png"` 而非完整带空格句子）
5. `App.bindings` / 组件渲染属性在 8.x 频繁内化——**新增 UI 代码的每个属性访问先 `dir()` 实测**，把验证写进测试

### 8.4 对 P1 工程的影响

- requirements 锁 `textual>=8.2.8,<9`（8.x 重构中，锁上界防破坏性升级）
- P1 测试铁律落地：`tests/helpers/` 新增 Pilot 冒烟脚本（对标 jsdom 冒烟模式），断言统一用 `export_screenshot()` 无空格片段
- 真实终端兼容（macOS Terminal/iTerm2）留 P1 手工验收项：headless 已验证逻辑层，真终端配色/键盘映射需人工过一遍
