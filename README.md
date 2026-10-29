# ✦ Fix — 自主演进的终端 AI 研发搭档

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
