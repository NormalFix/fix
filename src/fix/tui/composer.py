"""Codex-style prompt composer formatting and slash command specifications."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.text import Text

from fix.tui.theme import PRIMARY_TEXT_STYLE, _sync_live_terminal_width

CODEX_PROMPT_CHEVRON = "›"
CODEX_PROMPT_PREFIX = "› "
CODEX_COMPOSER_PREFIX = " › "
CODEX_COMPOSER_CONT_PREFIX = "   "
CODEX_PLACEHOLDER_TEXT = "Ask Fix to do anything · 输入问题或 / 命令"


def wrap_codex_user_prompt_lines(user_prompt: str, max_body_cells: int) -> list[str]:
    """
    Wrap `user_prompt` into visual lines of at most `max_body_cells` display cells
    (matching OpenAI Codex CLI's `wrap_user_message` in `codex-rs/tui/src/history_cell/messages.rs`),
    so line 1 is prefixed with `› ` (2 cols) and all soft-wrapped / explicit-newline continuation
    lines are indented with `  ` (2 cols, `LIVE_PREFIX_COLS = 2`) without ever wrapping under `›`.
    """
    from rich.cells import cell_len

    cleaned = user_prompt.rstrip("\r\n") if user_prompt.strip() else user_prompt
    raw_lines = cleaned.split("\n") if cleaned else [""]
    budget = max(12, max_body_cells)
    out: list[str] = []
    for raw_ln in raw_lines:
        expanded = raw_ln.replace("\r", "").replace("\t", "    ")
        if not expanded:
            out.append("")
            continue
        if cell_len(expanded) <= budget:
            out.append(expanded)
            continue
        cur_chars: list[str] = []
        cur_w = 0
        for ch in expanded:
            cw = max(1, cell_len(ch))
            if cur_w + cw > budget and cur_chars:
                out.append("".join(cur_chars))
                cur_chars = [ch]
                cur_w = cw
            else:
                cur_chars.append(ch)
                cur_w += cw
        if cur_chars:
            out.append("".join(cur_chars))
    return out or [""]


def render_codex_user_prompt_to(c: Console, user_prompt: str) -> None:
    """
    Render a submitted user prompt in OpenAI Codex CLI's `UserHistoryCell` style
    (`codex-rs/tui/src/history_cell/messages.rs`):
    - Top blank spacing row separating the user turn from preceding content
    - First visual row prefixed with Codex's `› ` (`bold #0EA5E9`)
    - Every soft-wrapped continuation row and explicit `\n` row indented by `  ` (2 spaces, `LIVE_PREFIX_COLS = 2`)
    - User prompt body rendered in crisp native primary foreground (`PRIMARY_TEXT_STYLE = ""`)
    - Bottom blank spacing row separating the user prompt from the agent's execution steps / response
    """
    console_w = getattr(c, "width", None) or 80
    max_body_cells = max(12, console_w - 3)
    vis_lines = wrap_codex_user_prompt_lines(user_prompt, max_body_cells=max_body_cells)
    c.print(Text(""))
    for idx_ln, ln_str in enumerate(vis_lines):
        row_txt = Text()
        if idx_ln == 0:
            row_txt.append(CODEX_PROMPT_PREFIX, style="bold #0EA5E9")
        else:
            row_txt.append("  ", style="")
        row_txt.append(ln_str, style=PRIMARY_TEXT_STYLE)
        c.print(row_txt, no_wrap=True, overflow="ignore", crop=False)
    c.print(Text(""))



KNOWN_SUBCOMMANDS = {
    "init",
    "status",
    "rules",
    "learn-diff",
    "run",
    "ask",
    "bug",
    "opt",
    "evolve",
    "chat",
}

SLASH_COMMANDS = [
    "/help",
    "/status",
    "/steps",
    "/ask",
    "/bug",
    "/opt",
    "/trace",
    "/diag",
    "/profile",
    "/idle",
    "/evolve",
    "/insights",
    "/autonomy",
    "/sandbox",
    "/rules",
    "/landmarks",
    "/skills",
    "/learn",
    "/diff",
    "/undo",
    "/copy",
    "/new",
    "/clear",
    "/resume",
    "/model",
    "/connect",
    "/init",
    "/exit",
    "/quit",
]

SLASH_COMMAND_SPECS: list[dict[str, Any]] = [
    {
        "cmd": "/new",
        "arg_hint": "[新对话问题]",
        "category": "会话",
        "desc": "在当前目录新建对话 (原对话自动保存至 /resume)",
        "keywords": ("new", "create", "fresh", "start", "新建", "新对话", "开启", "新会话"),
        "has_stage2": True,
        "requires_arg": False,
        "canonical": True,
    },
    {
        "cmd": "/resume",
        "arg_hint": "[序号|关键词]",
        "category": "会话",
        "desc": "恢复当前目录的历史对话并保持在最新对话位置",
        "keywords": ("resume", "restore", "history", "session", "continue", "恢复", "历史", "历史对话", "继续", "会话"),
        "has_stage2": True,
        "requires_arg": False,
        "canonical": True,
    },
    {
        "cmd": "/model",
        "arg_hint": "[模型名称]",
        "category": "模型",
        "desc": "查看当前支持切换的大模型列表或切换模型",
        "keywords": ("model", "switch", "llm", "fast", "reasoning", "模型", "切换", "大模型", "切换模型"),
        "has_stage2": True,
        "requires_arg": False,
        "canonical": True,
    },
    {
        "cmd": "/connect",
        "arg_hint": "[厂商名] [API_Key]",
        "category": "模型",
        "desc": "连接其他大模型厂商 (DeepSeek/GLM/MiMo/MiniMax/Qwen/OpenAI/Anthropic)",
        "keywords": ("connect", "vendor", "provider", "api", "key", "deepseek", "glm", "mimo", "qwen", "连接", "厂商", "接入", "密钥"),
        "has_stage2": True,
        "requires_arg": False,
        "canonical": True,
    },
    {
        "cmd": "/clear",
        "arg_hint": "",
        "category": "会话",
        "desc": "清空当前会话多轮上下文",
        "keywords": ("clear", "reset", "clean", "清空", "清理", "重置"),
        "has_stage2": False,
        "requires_arg": False,
        "canonical": True,
    },
    {
        "cmd": "/steps",
        "arg_hint": "[#轮次]",
        "category": "会话",
        "desc": "展开或收起上一轮任务工具调用明细",
        "keywords": ("steps", "tools", "fold", "expand", "步骤", "展开", "收起", "工具", "过程"),
        "has_stage2": True,
        "requires_arg": False,
        "canonical": True,
    },
    {
        "cmd": "/status",
        "arg_hint": "",
        "category": "工作区",
        "desc": "查看项目技术栈、模型与地标状态",
        "keywords": ("status", "info", "repo", "project", "状态", "信息", "技术栈", "概览"),
        "has_stage2": False,
        "requires_arg": False,
        "canonical": True,
    },
    {
        "cmd": "/ask",
        "arg_hint": "<架构/代码问题>",
        "category": "分析",
        "desc": "只读架构分析与代码问答 ([file:line])",
        "keywords": ("ask", "explain", "analyze", "question", "提问", "问答", "分析", "架构", "解释"),
        "has_stage2": True,
        "requires_arg": True,
        "canonical": True,
    },
    {
        "cmd": "/bug",
        "arg_hint": "<Bug现象或报错>",
        "category": "诊断",
        "desc": "科学闭环排查与修复 Bug",
        "keywords": ("bug", "fix", "debug", "error", "crash", "报错", "修复", "排查", "调试", "缺陷"),
        "has_stage2": True,
        "requires_arg": True,
        "canonical": True,
    },
    {
        "cmd": "/opt",
        "arg_hint": "<优化或重构目标>",
        "category": "分析",
        "desc": "性能与架构优化轨道",
        "keywords": ("opt", "optimize", "refactor", "perf", "优化", "重构", "性能", "改进"),
        "has_stage2": True,
        "requires_arg": True,
        "canonical": True,
    },
    {
        "cmd": "/trace",
        "arg_hint": "<符号名> [路径]",
        "category": "分析",
        "desc": "LSP/AST 跨文件符号调用链追踪",
        "keywords": ("trace", "symbol", "lsp", "callgraph", "caller", "追踪", "符号", "调用链", "引用"),
        "has_stage2": True,
        "requires_arg": True,
        "canonical": True,
    },
    {
        "cmd": "/diag",
        "arg_hint": "[文件或目录路径]",
        "category": "诊断",
        "desc": "运行静态语法与导入诊断",
        "keywords": ("diag", "check", "lint", "syntax", "import", "诊断", "静态检查", "语法", "导入"),
        "has_stage2": True,
        "requires_arg": False,
        "canonical": True,
    },
    {
        "cmd": "/diff",
        "arg_hint": "",
        "category": "工作区",
        "desc": "查看当前工作区 Git Diff",
        "keywords": ("diff", "git", "changes", "modified", "差异", "改动", "修改"),
        "has_stage2": False,
        "requires_arg": False,
        "canonical": True,
    },
    {
        "cmd": "/undo",
        "arg_hint": "",
        "category": "工作区",
        "desc": "一键回滚上一轮任务代码修改",
        "keywords": ("undo", "rollback", "revert", "restore", "撤回", "回滚", "撤销", "还原"),
        "has_stage2": False,
        "requires_arg": False,
        "canonical": True,
    },
    {
        "cmd": "/copy",
        "arg_hint": "",
        "category": "会话",
        "desc": "复制上一轮回答到系统剪贴板",
        "keywords": ("copy", "clipboard", "pbcopy", "复制", "剪贴板"),
        "has_stage2": False,
        "requires_arg": False,
        "canonical": True,
    },
    {
        "cmd": "/rules",
        "arg_hint": "[approve|reject <id>]",
        "category": "进化",
        "desc": "管理个人开发风格规则 (Mind Mirror)",
        "keywords": ("rules", "style", "persona", "mirror", "approve", "reject", "规则", "风格", "偏好", "习惯"),
        "has_stage2": True,
        "requires_arg": False,
        "canonical": True,
    },
    {
        "cmd": "/skills",
        "arg_hint": "[bootstrap|run <技能>]",
        "category": "进化",
        "desc": "查看、生成或执行项目专属诊断技能",
        "keywords": ("skills", "bootstrap", "run", "技能", "脚本", "专属技能"),
        "has_stage2": True,
        "requires_arg": False,
        "canonical": True,
    },
    {
        "cmd": "/landmarks",
        "arg_hint": "[搜索关键词]",
        "category": "工作区",
        "desc": "查看已验证的项目架构地标",
        "keywords": ("landmarks", "facts", "architecture", "地标", "架构地标", "事实"),
        "has_stage2": True,
        "requires_arg": False,
        "canonical": True,
    },
    {
        "cmd": "/idle",
        "arg_hint": "[run]",
        "category": "进化",
        "desc": "查看或运行后台 IdleHandler 进化队列",
        "keywords": ("idle", "looper", "queue", "background", "后台", "空闲", "队列", "进化任务"),
        "has_stage2": True,
        "requires_arg": False,
        "canonical": True,
    },
    {
        "cmd": "/evolve",
        "arg_hint": "[run]",
        "category": "进化",
        "desc": "查看三层第一性原理心智模型与优化提案",
        "keywords": ("evolve", "insights", "mental", "paradigm", "进化", "心智模型", "第一性原理", "洞察", "提案"),
        "has_stage2": True,
        "requires_arg": False,
        "canonical": True,
    },
    {
        "cmd": "/insights",
        "arg_hint": "[run]",
        "category": "进化",
        "desc": "查看三层第一性原理心智模型与优化提案",
        "keywords": ("insights", "evolve", "洞察", "心智模型"),
        "has_stage2": True,
        "requires_arg": False,
        "canonical": False,
    },
    {
        "cmd": "/autonomy",
        "arg_hint": "[任务描述]",
        "category": "进化",
        "desc": "查看 L1-L4 渐进式自治评分",
        "keywords": ("autonomy", "level", "score", "l1", "l2", "l3", "l4", "自治", "评分", "等级"),
        "has_stage2": True,
        "requires_arg": False,
        "canonical": True,
    },
    {
        "cmd": "/profile",
        "arg_hint": "[运行命令]",
        "category": "诊断",
        "desc": "运行 cProfile 性能热点剖析",
        "keywords": ("profile", "cprofile", "hotspot", "benchmark", "剖析", "热点", "性能分析", "耗时"),
        "has_stage2": True,
        "requires_arg": False,
        "canonical": True,
    },
    {
        "cmd": "/sandbox",
        "arg_hint": "[验证命令]",
        "category": "诊断",
        "desc": "Git Worktree 隔离沙箱回归验证",
        "keywords": ("sandbox", "worktree", "verify", "test", "沙箱", "隔离", "验证", "回归"),
        "has_stage2": True,
        "requires_arg": False,
        "canonical": True,
    },
    {
        "cmd": "/learn",
        "arg_hint": "[Git提交引用]",
        "category": "进化",
        "desc": "从最近一次 Git 提交学习代码风格",
        "keywords": ("learn", "commit", "git", "habit", "学习", "提交", "代码风格"),
        "has_stage2": True,
        "requires_arg": False,
        "canonical": True,
    },
    {
        "cmd": "/init",
        "arg_hint": "",
        "category": "工作区",
        "desc": "初始化 .fix/ 并扫描 Git 共变关系",
        "keywords": ("init", "setup", "hook", "cochange", "初始化"),
        "has_stage2": False,
        "requires_arg": False,
        "canonical": True,
    },
    {
        "cmd": "/help",
        "arg_hint": "",
        "category": "帮助",
        "desc": "查看所有交互命令说明",
        "keywords": ("help", "commands", "usage", "manual", "帮助", "命令", "说明", "指南"),
        "has_stage2": False,
        "requires_arg": False,
        "canonical": True,
    },
    {
        "cmd": "/exit",
        "arg_hint": "",
        "category": "会话",
        "desc": "退出 Fix 交互终端",
        "keywords": ("exit", "quit", "bye", "close", "退出", "关闭", "结束"),
        "has_stage2": False,
        "requires_arg": False,
        "canonical": True,
    },
    {
        "cmd": "/quit",
        "arg_hint": "",
        "category": "会话",
        "desc": "退出 Fix 交互终端",
        "keywords": ("quit", "exit", "退出"),
        "has_stage2": False,
        "requires_arg": False,
        "canonical": False,
    },
]
SLASH_COMMAND_SPEC_MAP: dict[str, dict[str, Any]] = {spec["cmd"]: spec for spec in SLASH_COMMAND_SPECS}

