# ✦ Fix (`v1.0.0-beta`) — 自主演进的终端 AI 研发搭档

**Fix** 是一个运行在终端中的全屏交互式 AI 编程与架构分析智能体（Autonomous AI Engineering Partner），具备 **双层心智镜像（Mind Mirror）**、**跨语言零预编译语义雷达（Polyglot Semantic Radar）**、**科学复现与独立审查闭环（Observe → Hypothesize → Reproduce → Fix → Verify → Review）** 以及 **自举式可演进技能（Executable Skills & L1–L4 Autonomy）**。

无需繁琐配置，在任何项目目录下直接输入 `fix` 即可进入全屏交互终端。

---

## 🚀 一句话极速安装（macOS / Linux）

打开终端，复制并执行以下 **任意一行命令** 即可完成全局安装：

### 方式一：官方一键安装脚本（推荐，自动创建独立虚拟环境并配置全局 `fix` 命令）
```bash
curl -fsSL https://raw.githubusercontent.com/NormalFix/fix/main/install.sh | bash
```

### 方式二：使用 `pip` / `uv` / `pipx` 一句话安装
```bash
pip install git+https://github.com/NormalFix/fix.git
```
> 若使用 `uv` 或 `pipx`：
> ```bash
> uv tool install git+https://github.com/NormalFix/fix.git
> # 或
> pipx install git+https://github.com/NormalFix/fix.git
> ```

安装完成后，运行以下命令验证版本：
```bash
fix --version
# 输出: Fix v1.0.0-beta
```

---

## ⚡ 30 秒快速上手

### 1. 在任意代码仓库启动 `Fix`
```bash
cd /path/to/your/project
fix
```

### 2. 连接大模型厂商（`/connect` 交互式向导）
进入 `fix` 界面后，直接输入 `/connect` 并回车（或输入 `/` 通过上下方向键选择 `/connect`），即可在弹窗中一键连接 **7 大主流模型厂商**：

- **交互式三步配置**：输入 `/connect` → 选择厂商 → 粘贴 API Key 自动在线鉴权并拉取可用模型列表 → 选择默认模型，配置自动加密权限保存在 `~/.fix/llm_providers.json`，下次启动免配置。
- **随时切换模型**：输入 `/model` 打开模型切换弹窗，或直接执行 `/model <模型名>`。

---

## 🛠 常用交互指令与斜杠命令

在 `fix` 交互界面中，你可以直接用自然语言提问、排查 Bug、重构代码，也可以使用 `/` 快捷命令（输入 `/` 自动弹出实时补全菜单）：

| 命令 | 功能说明 |
| :--- | :--- |
| `/connect` | 打开大模型厂商连接弹窗（支持 7 大厂商 API Key 在线验证与模型自动发现） |
| `/model` | 切换当前激活模型的 `fast_model` 与 `reasoning_model` |
| `/resume` | 打开历史会话恢复弹窗（按项目/全局查看历史对话并原地恢复上下文） |
| `/new` | 开启全新对话（自动归档当前会话到 `~/.fix/sessions/`） |
| `/ask <问题>` | 强制进入只读深度架构诊断模式（物理禁用写操作，所有结论强制锚定 `(file:line)`） |
| `/bug <现象>` | 强制进入科学 Bug 修复闭环（`假设 → 测试复现 → 最小修复 → 验证 → 独立审查`） |
| `/opt <目标>` | 进入架构与性能优化轨道（支持 Git Worktree 沙箱实证对比） |
| `/diff` | 查看当前工作区 `git diff` 变更摘要 |
| `/undo` | 毫秒级回滚到上一步修改前的 Git 快照 |
| `/trace <符号>` | 跨文件追踪函数/类/组件的调用链、被调方与生命周期契约 |
| `/diag [路径]` | 运行跨语言架构热点与高耦合模块雷达扫描 |
| `/skills` | 查看或运行项目自举沉淀的可执行诊断技能（`.fix/skills/`） |
| `/rules` | 查看或管理开发者心智镜像规则（`~/.fix/persona.db`） |
| `/evolve` | 查看与触发第一性原理心智模型、架构优化建议与元认知自省 |
| `/status` | 查看当前仓库画像、模型状态、路标记忆与自治等级（L1–L4） |
| `/help` | 查看全部交互命令与快捷键说明 |
| `/exit` | 退出 Fix（或在输入框为空时按 `Ctrl+C`） |

### 命令行单次执行模式（无需进入交互界面）
```bash
fix "梳理当前项目的核心架构与模块职责"
fix ask "分析订单状态机的并发安全性"
fix bug "修复空指针异常并补充回归测试"
fix status
```

---

## 🏗️ 项目架构与模块职责划分（Codex-Style Modular Architecture）

`Fix` 采用对标 OpenAI Codex 的 **七层单向依赖领域包架构（Strict Layered Domain Packages）**，`src/fix/` 顶层零散落脚本文件，每个子包职责单一、边界清晰：

```text
src/fix/
├── __init__.py              # 包版本元数据与向后兼容模块别名加载器
├── __main__.py              # 支持 `python -m fix` 直接启动
├── py.typed                 # PEP 561 静态类型检查标记
├── protocol/                # [1. 协议与数据契约层] 全局零外部依赖的 Pydantic v2 领域模型
│   ├── enums.py             # 任务意图 (TaskIntent)、状态机阶段 (TaskPhase)、L1-L4 自治等级枚举
│   ├── contracts.py         # 工具调用/返回契约、探针决策、独立审查裁决、语义雷达与沙箱报告模型
│   ├── memory_models.py     # 仓库路标 (EvidenceFact)、心智镜像风格规则、三层心智模型与技能卡模型
│   ├── session.py           # 会话状态 (TaskSession)、Token/步数安全预算与后台进化队列切片模型
│   └── schema.py            # 协议层统一导出门面
├── llm/                     # [2. 多厂商模型网关层] 7 大厂商协议适配与在线模型发现
│   ├── catalog.py           # 7 大厂商官方规范注册表 (PROVIDER_SPECS)、退役模型过滤与零配置环境变量探测
│   ├── adapters.py          # OpenAI Chat / Responses 与 Anthropic Messages 协议双向转换器
│   ├── gateway.py           # 异步多轮工具调用网关 (LLMGateway)、截断自动续写与结构化 JSON 校验
│   └── mock.py              # 确定性可编程测试替身 (ScriptedMockLLM)
├── memory/                  # [3. 双层持久化记忆层] SQLite WAL 路标库与个人心智镜像
│   └── store.py             # DualMemoryStore (<repo>/.fix/repo.db 项目路标 + ~/.fix/persona.db 个人偏好)
├── tools/                   # [4. Agent-Computer Interface (ACI) 工具与沙箱层]
│   ├── toolkit.py           # WorkspaceToolkit (文件读取、ripgrep 搜索、AST 门禁补丁、进程组命令执行、Git 懒快照与毫秒级回滚)
│   ├── lsp.py               # SemanticRadar 跨语言 AST/LSP 符号定义、调用链追踪与括号平衡检查
│   └── profiler.py          # RuntimeProfiler 运行时 cProfile 性能热点与覆盖率剖析
├── core/                    # [5. 核心认知与状态机引擎层]
│   ├── runtime.py           # FixRuntime 六阶段科学认知状态机 (PROBE → REPRODUCE → IMPLEMENT → VERIFY → REVIEW → SYNTHESIZE)
│   ├── context.py           # ContextManager 0-Token 仓库雷达画像 (RepoProfile) 与动态上下文组装
│   ├── looper.py            # FixLooper 优先级调度器 (P0 实时用户干预 / P1 主任务 / P2 后台空闲进化)
│   ├── autonomy.py          # AutonomyRouter (L1-L4 渐进式自治评估) 与 SandboxVerifier (Git Worktree 隔离实证)
│   └── evolution.py         # EvolutionEngine 第一性原理三层心智模型蒸馏、Git 共变挖掘与可执行技能自举
├── tui/                     # [6. 全屏终端交互界面层 (Full-Screen Terminal UI)]
│   ├── theme.py             # Rich Markdown 定制渲染组件、高对比度主题配色与实时终端尺寸同步
│   ├── composer.py          # Codex 风格输入提示符排版与 `/slash` 命令元数据规范表
│   ├── tracker.py           # CollapsibleProgressTracker 实时折叠步骤流、Claude 风格动态思考指示器与流式汇总
│   ├── persistence.py       # SessionPersistenceMixin 项目级隔离输入历史、多轮会话自动快照与 `/resume` 恢复
│   ├── modals.py            # ModalControllerMixin (`/connect`、`/model`、`/resume` 交互式弹窗与二级参数补全)
│   ├── commands.py          # SlashCommandRouterMixin 斜杠命令路由分发与交互式任务执行闭环
│   ├── fullscreen.py        # FullscreenTuiMixin 基于 prompt_toolkit 的全屏交替缓冲区布局、鼠标滚轮/拖拽选区与快捷键状态机
│   └── session.py           # FixInteractiveSession 交互式会话主控制器
└── cli/                     # [7. 命令行入口层]
    └── app.py               # Typer CLI 命令行入口 (`fix`、`fix run`、`fix ask`、`fix bug`、`fix opt`、`fix status` 等)
```

### 依赖方向约束（Acyclic Dependency Rule）
```text
cli  ──▶  tui  ──▶  core  ──▶  { llm, tools, memory }  ──▶  protocol
```
底层协议（`protocol`）与基础设施（`llm`、`tools`、`memory`）完全独立于上层 UI（`tui`、`cli`），任何模块均可独立导入、单元测试与复用。

---

## 🔄 升级与卸载

- **一键升级到最新版本**：
  ```bash
  curl -fsSL https://raw.githubusercontent.com/NormalFix/fix/main/install.sh | bash
  ```
- **一键卸载**：
  ```bash
  rm -rf ~/.fix/venv ~/.local/bin/fix
  ```

---

## 📄 License

MIT License
