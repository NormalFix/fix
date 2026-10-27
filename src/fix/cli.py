"""Interactive Claude Code / Codex-style CLI & REPL (`fix`) for the Fix AI Engineering Partner."""

from __future__ import annotations

import asyncio
from datetime import datetime
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import typer
from rich import box
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.live import Live
from rich.markdown import CodeBlock, Heading, Markdown, Paragraph, TableElement
from rich.markup import escape
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from fix import __version__
from fix.autonomy import AutonomyRouter, SandboxVerifier
from fix.context import ContextManager, RepoProfile, resolve_project_root
from fix.evolution import EvolutionEngine
from fix.llm import LLMGateway, PROVIDER_ORDER, PROVIDER_SPECS, resolve_provider_id
from fix.looper import FixLooper
from fix.memory import DualMemoryStore
from fix.runtime import FixRuntime, format_tool_call_summary
from fix.schema import StyleRule, TaskIntent, TaskPhase, TaskSession
from fix.tools import WorkspaceToolkit

# Custom theme that eliminates solid black background (`bgcolor="black"`) on English/code spans
# Theme 2 (High-Contrast for both Light & Dark Terminals):
# Rich Mint Teal (#0D9488/#10B981) + Vibrant Coral Orange (#F97316) + Deep Sky Blue Code (#0EA5E9) + Native Default Foreground for body/targets
FIX_THEME = Theme(
    {
        "code": "bold #c87dbb",
        "markdown.code": "bold #c87dbb",
        "markdown.code_block": "#c87dbb",
        "markdown.h1": "bold #F97316",
        "markdown.h2": "bold #0D9488",
        "markdown.h3": "bold #0EA5E9",
        "markdown.h4": "bold #10B981",
        "markdown.block_quote": "#0D9488",
        "markdown.item.bullet": "bold #10B981",
        "markdown.item.number": "bold #10B981",
        "markdown.hr": "#0D9488",
        "markdown.table.border": "#0D9488",
        "markdown.table.header": "bold #0D9488",
        "markdown.link": "#0EA5E9",
        "markdown.link_url": "underline #0EA5E9",
    }
)

EXPLORE_TOOL_NAMES = {
    "read_file",
    "search_code",
    "list_dir",
    "inspect_symbols",
    "trace_symbol",
    "check_diagnostics",
}

CLAUDE_SPINNER_FRAMES = ("·", "✢", "✳", "✶", "✻", "✽", "✻", "✶", "✳", "✢")
SPINNER_FRAMES = CLAUDE_SPINNER_FRAMES


class ReflowParagraph(Paragraph):
    """Paragraph element without right-side space padding so terminals can naturally reflow text."""

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        self.text.justify = "default"
        yield self.text


class ReflowHeading(Heading):
    """Clean, soft-toned left-aligned Markdown heading (`# Title`, `## Section`)."""

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        text = self.text.copy()
        text.justify = "default"
        level = int(self.tag[1]) if len(self.tag) == 2 and self.tag[1].isdigit() else 2
        prefix = "#" * level + " "
        style_name = f"markdown.h{min(level, 4)}"
        heading_line = Text(prefix, style=style_name)
        heading_line.append_text(text)
        heading_line.stylize(style_name)
        yield heading_line


class CleanCodeBlock(CodeBlock):
    """Code block with transparent background (`default`) instead of solid black/monokai background."""

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        code = str(self.text).rstrip()
        syntax = Syntax(
            code,
            self.lexer_name,
            theme="ansi_dark",
            word_wrap=True,
            background_color="default",
            padding=(0, 2),
        )
        yield syntax


class CleanTableElement(TableElement):
    """Markdown table that wraps cell content cleanly (`overflow='fold'`) instead of truncating with `…`."""

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        table = Table(
            box=box.SIMPLE_HEAVY,
            pad_edge=False,
            style="markdown.table.border",
            show_edge=False,
            collapse_padding=False,
            padding=(0, 1),
        )
        if self.header is not None and self.header.row is not None:
            for column in self.header.row.cells:
                heading = column.content.copy()
                heading.stylize("markdown.table.header")
                table.add_column(heading, overflow="fold")
        if self.body is not None:
            for row in self.body.rows:
                row_content = [element.content for element in row.cells]
                table.add_row(*row_content)
        yield table


# Secondary color for Agent intermediate execution operation prompts (`✦`, `• Explored`, `Read/Search`, `• Ran`):
# Solid and clearly legible on both dark & light terminals, yet one step lighter than the `bold` primary tier
# (User Prompts + Agent Final Results).
PRIMARY_TEXT_STYLE = ""
SECONDARY_STEP_TEXT_STYLE = "#788596"


class CleanMarkdown(Markdown):
    """
    Reflow-friendly Markdown renderer with zero black background behind English/code terms.
    Defaults to `style="bold"` so Agent final output results have the exact same primary visual solidness
    as the user's prompts (`> ...`).
    """

    elements = dict(Markdown.elements)
    elements["paragraph_open"] = ReflowParagraph
    elements["heading_open"] = ReflowHeading
    elements["fence"] = CleanCodeBlock
    elements["code_block"] = CleanCodeBlock
    elements["table_open"] = CleanTableElement

    def __init__(self, markup: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(markup, *args, **kwargs)


def _query_os_terminal_size() -> tuple[int, int]:
    """
    Query the real OS terminal window `(columns, lines)` from un-redirected descriptors
    (`sys.__stdout__`, `sys.__stderr__`, `sys.__stdin__`, or `/dev/tty`), bypassing Rich's `_FileProxy`.
    """
    for stream in (
        getattr(sys, "__stdout__", None),
        getattr(sys, "__stderr__", None),
        getattr(sys, "__stdin__", None),
    ):
        if stream is not None and hasattr(stream, "fileno"):
            try:
                sz = os.get_terminal_size(stream.fileno())
                if sz.columns > 0:
                    return sz.columns, max(10, sz.lines)
            except (AttributeError, ValueError, OSError):
                continue

    for fd in (1, 2, 0):
        try:
            sz = os.get_terminal_size(fd)
            if sz.columns > 0:
                return sz.columns, max(10, sz.lines)
        except OSError:
            continue

    try:
        with open("/dev/tty") as tty_f:
            sz = os.get_terminal_size(tty_f.fileno())
            if sz.columns > 0:
                return sz.columns, max(10, sz.lines)
    except Exception:
        pass

    fallback = shutil.get_terminal_size(fallback=(100, 30))
    return fallback.columns, max(10, fallback.lines)


def create_fix_console(
    file: Any = None,
    width: int | None = None,
    force_terminal: bool | None = None,
) -> Console:
    """Create a Rich Console configured with FIX_THEME (no black code backgrounds) and live terminal sizing."""
    env_copy = dict(os.environ)
    env_copy.pop("COLUMNS", None)
    env_copy.pop("LINES", None)
    env_copy.pop("NO_COLOR", None)
    env_copy.pop("FORCE_COLOR", None)
    env_copy["COLORTERM"] = "truecolor"
    if width is not None and env_copy.get("TERM") in (None, "", "dumb"):
        env_copy["TERM"] = "xterm-256color"
    return Console(
        file=file,
        width=width,
        height=40 if width is not None else None,
        force_terminal=force_terminal,
        color_system="truecolor" if force_terminal else "auto",
        theme=FIX_THEME,
        highlight=False,
        _environ=env_copy,
    )


def _sync_live_terminal_width(c: Console, explicit_width: int | None = None) -> int:
    """Synchronize Console width with the current OS terminal window size."""
    if explicit_width is not None:
        c._width = explicit_width
        if c._height is None:
            c._height = 40
        return explicit_width

    # Check if console is writing to a custom StringIO test buffer (not stdout/stderr, Live's _FileProxy, or live TUI sink)
    raw_file = getattr(c, "_file", None)
    if (
        raw_file is not None
        and raw_file not in (
            sys.stdout,
            sys.stderr,
            getattr(sys, "__stdout__", None),
            getattr(sys, "__stderr__", None),
        )
        and not hasattr(raw_file, "_live")
        and not getattr(raw_file, "_is_tui_sink", False)
    ):
        return c.width

    cols, rows = _query_os_terminal_size()
    c._width = cols
    c._height = rows
    return cols


def normalize_line_range_notation(text: str) -> str:
    """
    Normalize all line-number and file-line range citations so they:
    1. Always use a standard ASCII hyphen-minus `-` (U+002D) between start and end line numbers
       (converting en-dash `–`, em-dash `—`, wave `~`/`～`, or underscore `_` between digits into `-`).
    2. Convert full-width parentheses `（...）` and full-width colons `：` containing line citations
       or code references into tight half-width ASCII `(:357-374)`, `(98-168)`, `(OrderService.py:505-518)`
       with zero leading/trailing inner spaces so digits and parentheses align on the same baseline.
    """
    if not text:
        return text

    # 1. Replace any en-dash, em-dash, wave, or underscore between two line numbers with ASCII `-`
    out = re.sub(r"(?<=\d)\s*[–—―~～_]+\s*(?=\d)", "-", text)

    # 2. Normalize full-width colon before digits (`：357-374` -> `:357-374`)
    out = re.sub(r"：\s*(?=\d)", ":", out)

    # 3. Normalize any `（...）` or `(...)` that contains a line number / range or starts with a line/code ref
    def _clean_paren(m: re.Match[str]) -> str:
        inner = m.group(1).strip()
        has_line_ref = bool(
            re.search(r"(?:^|[\s`a-zA-Z0-9_./]):\s*\d+", inner)
            or re.search(r"\b\d+-\d+\b", inner)
            or re.match(r"^约?\s*\d+-\d+", inner)
        )
        if not has_line_ref:
            return m.group(0)
        inner = re.sub(r"^：\s*", ":", inner)
        inner = re.sub(r"^`：\s*", "`:", inner)
        inner = re.sub(r"(^|[\s`]):\s+(?=\d)", r"\1:", inner)
        inner = re.sub(r"(?<=\d)\s*-\s*(?=\d)", "-", inner)
        return f"({inner})"

    out = re.sub(r"[（(]([^（）()\n]+)[）)]", _clean_paren, out)
    return out


def _truncate_middle_for_width(text: str, max_cols: int) -> str:
    """Truncate a single-line tool label cleanly in the middle so it NEVER wraps across multiple lines."""
    if max_cols < 20:
        max_cols = 20
    if len(text) <= max_cols:
        return text
    keep = max_cols - 3
    head = keep // 2
    tail = keep - head
    return f"{text[:head]}...{text[-tail:]}"


def _format_codex_explore_action(name: str, args: dict[str, Any]) -> tuple[str, str]:
    if name == "read_file":
        rel = str(args.get("rel_path", ""))
        return "Read", Path(rel).name or rel
    if name == "search_code":
        pat = str(args.get("pattern", ""))
        glob_str = str(args.get("path_glob") or ".")
        scope = Path(glob_str.rstrip("/*")).name or "."
        return "Search", f"Search {pat} in {scope}"
    if name == "list_dir":
        rel = str(args.get("rel_path", "."))
        return "List", f"List {rel}"
    if name == "inspect_symbols":
        rel = str(args.get("rel_path", ""))
        return "Read", Path(rel).name or rel
    if name == "trace_symbol":
        sym = str(args.get("symbol_name", ""))
        return "Trace", f"Trace {sym}"
    if name == "check_diagnostics":
        rel = str(args.get("rel_path") or ".")
        return "Diag", f"Diagnostics in {rel}"
    return "Other", format_tool_call_summary(name, args)


def format_elapsed_hms(elapsed_sec: float) -> str:
    """Format elapsed seconds into `Xh Ym Zs` / `Ym Zs` / `Zs` (e.g. 3798 -> `1h 3m 18s`)."""
    total_secs = int(max(0.0, elapsed_sec))
    hours, rem = divmod(total_secs, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def format_claude_working_action(name: str, args: dict[str, Any], fallback_label: str = "") -> str:
    """
    Convert a tool execution event into a Claude Code-style progressive Chinese working status (`正在xxx...`).
    """
    if name == "read_file":
        rel = str(args.get("rel_path", "")).strip()
        fname = Path(rel).name or rel or "文件"
        return f"正在读取 {fname}..."
    if name == "search_code":
        pat = str(args.get("pattern", "")).strip()
        return f"正在搜索代码 {pat}..." if pat else "正在搜索代码..."
    if name == "list_dir":
        rel = str(args.get("rel_path", ".")).strip() or "."
        return f"正在浏览目录 {rel}..."
    if name == "inspect_symbols":
        rel = str(args.get("rel_path", "")).strip()
        fname = Path(rel).name or rel or "文件"
        return f"正在解析符号结构 {fname}..."
    if name == "trace_symbol":
        sym = str(args.get("symbol_name", "")).strip()
        return f"正在追踪符号引用 {sym}..." if sym else "正在追踪符号引用..."
    if name == "check_diagnostics":
        rel = str(args.get("rel_path") or "项目").strip()
        return f"正在检查代码诊断 {rel}..."
    if name == "run_command":
        cmd = str(args.get("cmd", "")).strip()
        return f"正在执行命令 {cmd}..." if cmd else "正在执行命令..."
    if name == "profile_execution":
        cmd = str(args.get("cmd", "")).strip()
        return f"正在剖析运行性能 {cmd}..." if cmd else "正在剖析运行性能..."
    if name == "run_skill":
        slug = str(args.get("slug", "")).strip()
        return f"正在执行技能 {slug}..." if slug else "正在执行技能..."
    if name == "apply_patch":
        rel = str(args.get("rel_path", "")).strip()
        fname = Path(rel).name or rel or "代码"
        return f"正在编辑代码 {fname}..."
    if name == "write_new_file":
        rel = str(args.get("rel_path", "")).strip()
        fname = Path(rel).name or rel or "文件"
        return f"正在写入文件 {fname}..."
    clean_fb = fallback_label.strip()
    if clean_fb.startswith("正在"):
        return clean_fb
    if clean_fb:
        return f"正在执行 {clean_fb}..."
    return "正在思考与分析代码..."


def _build_claude_shimmer_fragments(text: str, elapsed: float) -> list[tuple[str, str]]:
    """
    Build Claude Code's signature left-to-right warm terracotta shimmer wave across the active `正在xxx...` status text.
    """
    if not text:
        return []
    n = len(text)
    wave_idx = int(max(0.0, elapsed) * 10) % max(1, n + 4)
    frags: list[tuple[str, str]] = []
    for idx, ch in enumerate(text):
        dist = abs(idx - wave_idx)
        if dist == 0:
            st = "bold #FEF3C7"
        elif dist == 1:
            st = "bold #FDBA74"
        else:
            st = "bold #D97757"
        if frags and frags[-1][0] == st:
            frags[-1] = (st, frags[-1][1] + ch)
        else:
            frags.append((st, ch))
    return frags


def format_explored_group_lines(raw_actions: list[tuple[str, str]]) -> list[str]:
    merged: list[str] = []
    pending_reads: list[str] = []

    def _flush_reads() -> None:
        if pending_reads:
            deduped = list(dict.fromkeys(pending_reads))
            merged.append(f"Read {', '.join(deduped)}")
            pending_reads.clear()

    for kind, detail in raw_actions:
        if kind == "Read":
            pending_reads.append(detail)
        else:
            _flush_reads()
            merged.append(detail)
    _flush_reads()

    formatted: list[str] = []
    for idx, line in enumerate(merged):
        prefix = "  └ " if idx == 0 else "    "
        formatted.append(f"{prefix}{line}")
    return formatted




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


def _render_blocks_to_pt_fragments(
    blocks: list[Callable[[Console], None]],
    width: int,
    active_tracker: CollapsibleProgressTracker | None = None,
    inflight_prompt: str | None = None,
    repo_name: str = "",
) -> list[tuple[str, str]]:
    """Render session screen blocks into prompt_toolkit (style, text) fragments for the full-screen TUI."""
    from io import StringIO

    buf = StringIO()
    temp_console = create_fix_console(file=buf, width=max(40, width - 2), force_terminal=True)
    temp_console.no_color = False

    # Use record=True or direct segment rendering: let's render via console.begin_capture() or console.render()
    # Even simpler and 100% faithful to Rich ANSI styles: use prompt_toolkit.formatted_text.ANSI!
    from prompt_toolkit.formatted_text import ANSI, to_formatted_text

    for fn in blocks:
        fn(temp_console)

    if inflight_prompt:
        render_codex_user_prompt_to(temp_console, inflight_prompt)
    if active_tracker is not None:
        active_tracker.render_all_completed_blocks_to(temp_console)
        if active_tracker._pending_explore_actions:
            hdr = Text()
            hdr.append("• ", style="#10B981")
            hdr.append("Explored", style="#0D9488")
            temp_console.print(hdr)
            for ln in format_explored_group_lines(active_tracker._pending_explore_actions)[-6:]:
                temp_console.print(active_tracker._style_explored_line(ln))

    ansi_str = buf.getvalue()
    if not ansi_str.strip():
        return [("", "")]
    raw_frags = list(to_formatted_text(ANSI(ansi_str)))
    # Coalesce adjacent 1-char fragments sharing the same style into contiguous strings
    coalesced: list[tuple[str, str]] = []
    for item in raw_frags:
        st, ch = item[0], item[1]
        if coalesced and coalesced[-1][0] == st:
            coalesced[-1] = (st, coalesced[-1][1] + ch)
        else:
            coalesced.append((st, ch))
    return coalesced


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


app = typer.Typer(
    name="fix",
    help="Fix: Interactive AI Engineering Partner (run `fix` in any project folder to launch the interactive CLI).",
    add_completion=False,
    invoke_without_command=True,
)
console = create_fix_console()


def _build_stack(
    repo: str | Path | None = None,
) -> tuple[WorkspaceToolkit, DualMemoryStore, ContextManager, LLMGateway, EvolutionEngine, FixRuntime]:
    repo_path = resolve_project_root(repo)
    toolkit = WorkspaceToolkit(repo_path)
    memory = DualMemoryStore(repo_path)
    context_mgr = ContextManager()
    llm = LLMGateway()
    evolution = EvolutionEngine(repo_path, memory=memory, llm=llm)
    runtime = FixRuntime(
        llm=llm,
        toolkit=toolkit,
        context_mgr=context_mgr,
        memory=memory,
        evolution=evolution,
    )
    return toolkit, memory, context_mgr, llm, evolution, runtime


class CollapsibleProgressTracker:
    """
    Codex-style progressive stream & live status tracker:
    - Prints intermediate reasoning narratives (`我会先从项目里定位...`, `已确认...接下来...`)
      and grouped `• Explored` (`└ Search ...` / `  Read A.kt, B.kt`) and `• Ran <cmd>` blocks
      progressively as each batch finishes.
    - While an LLM call or tool is actively running, displays a live 1-line status indicator at the bottom
      so the user always sees what Fix is currently doing (`⠋ Fix 正在执行: ...`).
    - Dynamically adapts to terminal window resizes (`SIGWINCH` / 8Hz width check) during execution.
    """

    def __init__(
        self,
        out_console: Console | None = None,
        on_resize_redraw: Callable[[int], None] | None = None,
    ) -> None:
        self.console = out_console or console
        self.on_resize_redraw = on_resize_redraw
        self.started_at = time.time()
        self.phase_line = "正在分析项目上下文与规划步骤..."
        self.current_action_label = "正在思考与分析代码..."
        self.tool_lines: list[str] = []
        self.total_tools = 0
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.llm_calls = 0
        self.current_step = 0
        self._pending_explore_actions: list[tuple[str, str]] = []
        self.completed_stream_blocks: list[dict[str, Any]] = []
        self._live: Live | None = None
        self._last_cols: int = _sync_live_terminal_width(self.console)
        self._tick_idx = 0

    def _flush_pending_explore_group(self) -> None:
        """Flush accumulated read/search actions into a Codex-style `• Explored` block."""
        if not self._pending_explore_actions:
            return
        lines = format_explored_group_lines(self._pending_explore_actions)
        self._pending_explore_actions.clear()
        block = {"type": "explored", "lines": lines}
        self.completed_stream_blocks.append(block)
        self._render_stream_block_to(self.console, block)

    @staticmethod
    def _style_explored_line(ln: str) -> Text:
        t = Text()
        if ln.startswith("  └ "):
            t.append("  └ ", style="#0D9488")
            rest = ln[4:]
        elif ln.startswith("    "):
            t.append("    ")
            rest = ln[4:]
        else:
            rest = ln

        for verb in ("Search ", "Read ", "List ", "Trace ", "Diagnostics "):
            if rest.startswith(verb):
                t.append(verb, style="#0EA5E9")
                body = rest[len(verb):]
                if verb == "Search " and " in " in body:
                    pat, scope = body.rsplit(" in ", 1)
                    # Secondary step style: solid and legible, yet slightly lighter than primary user prompt / final output
                    t.append(pat, style=SECONDARY_STEP_TEXT_STYLE)
                    t.append(" in ", style="#0D9488")
                    t.append(scope, style="#10B981")
                else:
                    t.append(body, style=SECONDARY_STEP_TEXT_STYLE)
                return t

        t.append(rest, style=SECONDARY_STEP_TEXT_STYLE)
        return t

    @classmethod
    def _render_stream_block_to(cls, c: Console, block: dict[str, Any]) -> None:
        btype = block.get("type")
        if btype == "narrative":
            raw_txt = normalize_line_range_notation(str(block.get("text", "")).lstrip("✦").strip())
            nar = Text()
            nar.append("✦ ", style="#F97316")
            # Secondary step style: solid and legible, yet slightly lighter than the user's prompt & final result
            nar.append(raw_txt, style=SECONDARY_STEP_TEXT_STYLE)
            c.print(nar)
        elif btype == "explored":
            hdr = Text()
            hdr.append("• ", style="#10B981")
            hdr.append("Explored", style="#0D9488")
            c.print(hdr)
            for ln in block.get("lines", []):
                c.print(cls._style_explored_line(ln))
        elif btype == "ran":
            cmd_str = str(block.get("cmd", ""))
            hdr = Text()
            hdr.append("• ", style="#10B981")
            hdr.append("Ran ", style="#0D9488")
            hdr.append(cmd_str, style=SECONDARY_STEP_TEXT_STYLE)
            c.print(hdr)
            for ln in block.get("preview_lines", []):
                if ln.startswith("  └ "):
                    p_txt = Text()
                    p_txt.append("  └ ", style="#0D9488")
                    p_txt.append(ln[4:], style=SECONDARY_STEP_TEXT_STYLE)
                else:
                    p_txt = Text(ln, style=SECONDARY_STEP_TEXT_STYLE)
                c.print(p_txt)
        elif btype == "edited":
            rel = str(block.get("rel_path", ""))
            hdr = Text()
            hdr.append("• ", style="#F97316")
            hdr.append("Edited ", style="#EA580C")
            hdr.append(rel, style=SECONDARY_STEP_TEXT_STYLE)
            c.print(hdr)

    def render_all_completed_blocks_to(self, c: Console) -> None:
        """Re-render all completed narrative / `• Explored` / `• Ran` blocks (used on window resize)."""
        for blk in self.completed_stream_blocks:
            self._render_stream_block_to(c, blk)

    def handle_resize(self, new_cols: int | None = None) -> int:
        """Handle a terminal window resize during active task execution."""
        cols = _sync_live_terminal_width(self.console, explicit_width=new_cols)
        if cols != self._last_cols or new_cols is not None:
            self._last_cols = cols
            if self._live is not None:
                self._live._live_render._shape = None
            if self.on_resize_redraw is not None:
                self.on_resize_redraw(cols)
        return cols

    def _get_live_renderable(self) -> Group:
        cols = _sync_live_terminal_width(self.console)
        if cols != self._last_cols:
            self.handle_resize(cols)
        return self._build_renderable(cols)

    def _build_renderable(self, cols: int | None = None) -> Group:
        width = cols or _sync_live_terminal_width(self.console)
        elapsed = time.time() - self.started_at
        self._tick_idx = (self._tick_idx + 1) % len(SPINNER_FRAMES)
        spinner = SPINNER_FRAMES[self._tick_idx]
        max_line_chars = max(24, width - 4)

        items: list[Any] = []
        if self._pending_explore_actions:
            hdr = Text()
            hdr.append("• ", style="#10B981")
            hdr.append("Explored", style="#0D9488")
            items.append(hdr)
            for ln in format_explored_group_lines(self._pending_explore_actions)[-6:]:
                fitted_ln = _truncate_middle_for_width(ln, max_line_chars)
                t_ln = self._style_explored_line(fitted_ln)
                t_ln.no_wrap = True
                t_ln.overflow = "crop"
                items.append(t_ln)

        status_txt = Text()
        status_txt.append(f"{spinner} ✦ Fix 正在执行: ", style="#0D9488")
        rem_budget = max(12, max_line_chars - 18)
        fitted_act = _truncate_middle_for_width(
            f"{self.current_action_label}  ({elapsed:.1f}s · {self.total_tools} tools)",
            rem_budget,
        )
        status_txt.append(fitted_act, style=SECONDARY_STEP_TEXT_STYLE)
        status_txt.no_wrap = True
        status_txt.overflow = "crop"
        items.append(status_txt)
        return Group(*items)

    def on_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if event_type == "narrative":
            self._flush_pending_explore_group()
            text = normalize_line_range_notation(str(payload.get("text", "")).strip())
            if text:
                blk = {"type": "narrative", "text": text}
                self.completed_stream_blocks.append(blk)
                self._render_stream_block_to(self.console, blk)
            self.current_action_label = "正在执行下一步探索与分析..."

        elif event_type == "phase":
            phase = str(payload.get("phase", "")).upper()
            note = str(payload.get("note", ""))
            intent_val = str(payload.get("intent", "")).lower()
            if phase == "PROBE":
                if intent_val == "explain":
                    self.current_action_label = "正在深入分析代码架构与逻辑链路..."
                elif intent_val == "bug_fix":
                    self.current_action_label = "正在分析异常线索与定位根因..."
                else:
                    self.current_action_label = "正在思考与分析代码..."
            elif phase == "SYNTHESIZE":
                self._flush_pending_explore_group()
                self.current_action_label = note or "正在汇总代码证据并撰写完整分析报告..."
            elif phase == "REPRODUCE":
                self.current_action_label = "正在定位问题根因并尝试复现..."
            elif phase == "IMPLEMENT":
                self._flush_pending_explore_group()
                self.current_action_label = "正在编写与修改代码..."
            elif phase == "VERIFY":
                self._flush_pending_explore_group()
                self.current_action_label = "正在运行验证..."
            elif phase == "REVIEW":
                self._flush_pending_explore_group()
                self.current_action_label = "正在进行独立代码审查..."

        elif event_type == "review_start":
            self._flush_pending_explore_group()
            self.current_action_label = "正在进行独立代码审查..."

        elif event_type == "telemetry":
            self.total_tokens = int(payload.get("total_tokens", self.total_tokens) or 0)
            self.prompt_tokens = int(payload.get("prompt_tokens", self.prompt_tokens) or 0)
            self.completion_tokens = int(payload.get("completion_tokens", self.completion_tokens) or 0)
            self.llm_calls = int(payload.get("llm_calls", self.llm_calls) or 0)
            self.current_step = int(payload.get("step", self.current_step) or 0)

        elif event_type == "tool_start":
            name = str(payload.get("name", ""))
            args = payload.get("arguments") or {}
            label = payload.get("label") or format_tool_call_summary(name, args)
            self.total_tools += 1
            self.tool_lines.append(str(label))
            self.current_action_label = format_claude_working_action(name, args, str(label))

            if name in EXPLORE_TOOL_NAMES:
                kind, detail = _format_codex_explore_action(name, args)
                self._pending_explore_actions.append((kind, detail))
            else:
                self._flush_pending_explore_group()

        elif event_type == "tool_end":
            name = str(payload.get("name", ""))
            args = payload.get("arguments") or {}
            if name in ("run_command", "profile_execution", "run_skill"):
                self._flush_pending_explore_group()
                cmd_str = str(args.get("cmd") or args.get("slug") or name)
                raw_prev = str(payload.get("output_preview", "")).splitlines()
                clean_prev = [l for l in raw_prev if not l.startswith("[exit_code=") and l.strip()][:3]
                prev_lines = [
                    ("  └ " if i == 0 else "    ") + l[:120]
                    for i, l in enumerate(clean_prev)
                ]
                blk = {"type": "ran", "cmd": cmd_str, "preview_lines": prev_lines}
                self.completed_stream_blocks.append(blk)
                self._render_stream_block_to(self.console, blk)
            elif name in ("apply_patch", "write_new_file") and payload.get("ok"):
                self._flush_pending_explore_group()
                rel = str(args.get("rel_path", ""))
                blk = {"type": "edited", "rel_path": rel}
                self.completed_stream_blocks.append(blk)
                self._render_stream_block_to(self.console, blk)
            self.current_action_label = "正在结合工具输出推理下一步..."

        elif event_type == "verify":
            self._flush_pending_explore_group()
            cmd_str = str(payload.get("cmd", "")).strip()
            self.total_tools += 1
            self.tool_lines.append(f"Verify({cmd_str})")
            self.current_action_label = f"正在运行验证 {cmd_str}..." if cmd_str else "正在运行验证..."

        elif event_type == "done":
            if "total_tokens" in payload:
                self.total_tokens = int(payload.get("total_tokens", self.total_tokens) or 0)
            self._flush_pending_explore_group()

        self.refresh()

    def refresh(self) -> None:
        if self._live is not None:
            try:
                self._live.refresh()
            except Exception:
                pass
        if getattr(self, "on_tui_invalidate", None) is not None:
            try:
                self.on_tui_invalidate()
            except Exception:
                pass

    @contextmanager
    def track(self) -> Iterator[Callable[[str, dict[str, Any]], None]]:
        if self.console.is_terminal:
            with Live(
                get_renderable=self._get_live_renderable,
                console=self.console,
                transient=True,
                refresh_per_second=8,
            ) as live:
                self._live = live
                try:
                    yield self.on_event
                finally:
                    self._flush_pending_explore_group()
                    self._live = None
        else:
            try:
                yield self.on_event
            finally:
                self._flush_pending_explore_group()


def _render_session_result(
    session: TaskSession,
    out_console: Console | None = None,
    show_telemetry: bool = True,
) -> None:
    """
    Render the final Markdown result cleanly (Codex style: NO `─── Fix 分析与执行结果 ───` banner!).
    """
    c = out_console or console
    _sync_live_terminal_width(c)

    body_text = normalize_line_range_notation(session.final_summary.strip()) if session.final_summary else "(No textual output generated)"
    c.print()
    c.print(CleanMarkdown(body_text))
    c.print()
    if not show_telemetry:
        return

    mod_files_str = ", ".join(sorted(session.modified_files)) if session.modified_files else "无"
    telemetry_line = (
        f"  [bold #10B981]✓ 搞定[/bold #10B981]  ·  "
        f"[bold #F97316]⏱ 耗时:[/bold #F97316] [bold]{session.elapsed_sec:.2f}s[/bold]  ·  "
        f"[bold #0EA5E9]🪙 Token 消耗:[/bold #0EA5E9] [bold]{session.total_tokens:,}[/bold] "
        f"(输入 {session.prompt_tokens:,} · 输出 {session.completion_tokens:,} · LLM 调用 {session.llm_calls} 次)  ·  "
        f"[bold #0D9488]🧭 统计:[/bold #0D9488] 自治=[bold #10B981]{session.autonomy.level.value}[/bold #10B981]({session.autonomy.score:.2f}) · "
        f"轨道=[bold #0EA5E9]{session.intent.value if session.intent else 'auto'}[/bold #0EA5E9] · "
        f"步数={session.budget.steps_used}/{session.budget.max_steps} · "
        f"工具={session.tool_calls_count}次 · "
        f"修改文件={escape(mod_files_str)}"
    )
    if session.last_review is not None:
        rev_status = "APPROVED" if session.last_review.approved else "REJECTED"
        telemetry_line += f" · 审查={rev_status}" 
    c.print(telemetry_line)
    c.print()


def _run_task_with_collapsible_ui(
    runtime: FixRuntime,
    user_prompt: str,
    forced_intent: TaskIntent | None = None,
    out_console: Console | None = None,
) -> TaskSession:
    c = out_console or console

    def _clear_on_resize(_cols: int) -> None:
        raw_out = getattr(sys, "__stdout__", None)
        if raw_out is not None and c.is_terminal:
            try:
                raw_out.write("\033[2J\033[H")
                raw_out.flush()
            except Exception:
                pass
        tracker.render_all_completed_blocks_to(c)

    tracker = CollapsibleProgressTracker(c, on_resize_redraw=_clear_on_resize)
    with tracker.track() as on_event:
        session = asyncio.run(
            runtime.execute_task(
                user_prompt=user_prompt,
                forced_intent=forced_intent,
                on_event=on_event,
            )
        )
    _render_session_result(session, c)
    return session


class FixInteractiveSession:
    """
    Claude Code / Codex-style interactive TUI REPL bound to the current project directory.
    - Zero black background behind English/code terms (`FIX_THEME` + `CleanMarkdown`).
    - Borderless horizontal-rule layout (`Rule`) + real-time window resize re-render engine
      that works BOTH at the idle prompt AND mid-execution while a task is running!
    """

    def __init__(
        self,
        repo: str | Path | None = None,
        stack: tuple[WorkspaceToolkit, DualMemoryStore, ContextManager, LLMGateway, EvolutionEngine, FixRuntime] | None = None,
        out_console: Console | None = None,
    ) -> None:
        self._custom_console_passed = out_console is not None
        if out_console is not None:
            out_console.push_theme(FIX_THEME)
            self.console = out_console
        else:
            self.console = console
        self.toolkit, self.memory, self.context_mgr, self.llm, self.evolution, self.runtime = (
            stack if stack is not None else _build_stack(repo)
        )
        self.looper = FixLooper(
            repo_root=self.toolkit.repo_root,
            toolkit=self.toolkit,
            memory=self.memory,
            evolution=self.evolution,
            llm=self.llm,
        )
        self.profile: RepoProfile = self._auto_bootstrap_workspace()
        self.conversation_context: list[dict[str, str]] = []
        self.snapshot_stack: list[str] = []
        self.last_session: TaskSession | None = None
        self._screen_blocks: list[Callable[[Console], None]] = []
        self._active_tracker: CollapsibleProgressTracker | None = None
        self._inflight_prompt: str | None = None
        self._last_cols: int = _sync_live_terminal_width(self.console)
        self._running_task: asyncio.Task[Any] | None = None
        self._agent_state_label: str = "就绪"
        self._task_elapsed_sec: float = 0.0
        self._toast_message: str = ""
        self._toast_expires_at: float = 0.0
        self._scroll_offset_from_bottom: int = 0
        self._tui_mode: bool = False
        self._tui_app: Any = None
        self._sel_anchor: tuple[int, int] | None = None
        self._sel_head: tuple[int, int] | None = None
        self._sel_dragging: bool = False
        self._sel_autoscrolled: bool = False
        self._sel_last_xy: tuple[int, int] = (0, 0)
        self._input_sel_anchor: int | None = None
        self._input_sel_head: int | None = None
        self._input_sel_dragging: bool = False
        self._input_sel_autoscrolled: bool = False
        self._drag_autoscroll_task: asyncio.Task[Any] | None = None
        self._cached_render_lines: list[list[tuple[str, str]]] = []
        self._last_selected_text: str = ""
        self._turn_fold_states: dict[int, dict[str, Any]] = {}
        self._turn_counter: int = 0
        self._modal_state: dict[str, Any] | None = None
        self._modal_screen_row_to_item: dict[int, int] = {}
        self._modal_doc_line_to_item: dict[int, int] = {}
        self._input_buffer_ref: Any = None
        self._shortcut_overlay_open: bool = False
        self._slash_popup_selected_idx: int = 0
        self._slash_popup_dismissed: bool = False
        self._last_slash_prefix: str = ""
        self._slash_stage2_cmd: str = ""
        self._slash_dock_row_to_item: dict[int, int] = {}

        raw_session_dir = (
            Path(repo).resolve()
            if repo is not None
            else Path(self.toolkit.repo_root).resolve()
        )
        if raw_session_dir.is_file():
            raw_session_dir = raw_session_dir.parent
        self.session_dir: Path = raw_session_dir
        self.project_state_dir: Path = self._resolve_project_state_dir(self.session_dir)
        self.history_file_path: Path = self.project_state_dir / "prompt_history"
        self.sessions_dir: Path = self.project_state_dir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        from prompt_toolkit.history import FileHistory

        self._file_history: FileHistory = FileHistory(str(self.history_file_path))
        try:
            self._file_history._loaded_strings = list(self._file_history.load_history_strings())
            self._file_history._loaded = True
        except Exception:
            self._file_history._loaded_strings = []
            self._file_history._loaded = True

        self.session_id: str = self._new_conversation_session_id()
        self.session_created_at: float = time.time()
        self._saved_turns_records: list[dict[str, Any]] = []

    _last_session_save_ts: float = 0.0

    @staticmethod
    def _new_conversation_session_id() -> str:
        return f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _resolve_project_state_dir(session_dir: Path) -> Path:
        """Resolve the isolated per-directory state folder under `~/.fix/projects/<name>_<hash>/`."""
        resolved = session_dir.resolve()
        base_root = Path(
            os.environ.get("FIX_PROJECTS_STATE_DIR")
            or (Path.home() / ".fix" / "projects")
        )
        safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", resolved.name or "root").strip("_")[:32] or "root"
        dir_hash = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:16]
        state_dir = base_root / f"{safe_name}_{dir_hash}"
        state_dir.mkdir(parents=True, exist_ok=True)
        return state_dir

    def record_input_history(self, text: str) -> None:
        """Record a prompt into the current directory's isolated `FileHistory` (deduplicating consecutive identical entries)."""
        clean = text.strip()
        if not clean:
            return
        low = clean.lower()
        if low in ("/new", "/clear", "/exit", "/quit", "/q") or low.startswith("/resume"):
            return
        hist = self._file_history
        if not getattr(hist, "_loaded", False):
            try:
                hist._loaded_strings = list(hist.load_history_strings())
                hist._loaded = True
            except Exception:
                hist._loaded_strings = []
                hist._loaded = True
        existing = hist.get_strings()
        if not existing or existing[-1] != clean:
            hist.append_string(clean)
        if self._input_buffer_ref is not None and not getattr(self, "_browsing_history", False):
            try:
                all_items = hist.get_strings()
                tail_txt = self._input_buffer_ref.text
                self._input_buffer_ref._working_lines.clear()
                for item in all_items:
                    self._input_buffer_ref._working_lines.append(item)
                self._input_buffer_ref._working_lines.append(tail_txt)
                self._input_buffer_ref.working_index = len(self._input_buffer_ref._working_lines) - 1
            except Exception:
                pass

    @staticmethod
    def _format_relative_time(ts: float, now: float | None = None) -> str:
        curr = now if now is not None else time.time()
        diff = max(0.0, curr - ts)
        if diff < 60:
            return "刚刚"
        if diff < 3600:
            return f"{int(diff // 60)}分钟前"
        if diff < 86400:
            return f"{int(diff // 3600)}小时前"
        return f"{int(diff // 86400)}天前"

    def _persist_current_conversation_session(self) -> Path | None:
        """Persist the current session's multi-turn conversation to `self.sessions_dir / <session_id>.json`."""
        if not self._saved_turns_records:
            return None
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        now_ts = max(time.time(), FixInteractiveSession._last_session_save_ts + 1e-4)
        FixInteractiveSession._last_session_save_ts = now_ts
        first_prompt = str(self._saved_turns_records[0].get("user_prompt", "")).strip()
        latest_prompt = str(self._saved_turns_records[-1].get("user_prompt", "")).strip()
        latest_summary = str(self._saved_turns_records[-1].get("final_summary", "")).strip()
        payload = {
            "session_id": self.session_id,
            "directory": str(self.session_dir),
            "repo_root": str(self.toolkit.repo_root),
            "created_at": self.session_created_at,
            "updated_at": now_ts,
            "title": " ".join(first_prompt.split())[:120],
            "latest_prompt": " ".join(latest_prompt.split())[:120],
            "latest_summary_preview": " ".join(latest_summary.split())[:160],
            "turn_count": len(self._saved_turns_records),
            "turns": self._saved_turns_records,
        }
        target_path = self.sessions_dir / f"{self.session_id}.json"
        tmp_path = self.sessions_dir / f".{self.session_id}.json.tmp"
        try:
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp_path.replace(target_path)
        except Exception:
            try:
                target_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                return None
        return target_path

    def list_saved_sessions(self) -> list[dict[str, Any]]:
        """List saved historical conversation sessions for `self.session_dir`, sorted newest (`updated_at` desc) first."""
        if not self.sessions_dir.exists():
            return []
        now_ts = time.time()
        records: list[dict[str, Any]] = []
        for fpath in self.sessions_dir.glob("*.json"):
            if fpath.name.startswith("."):
                continue
            try:
                data = json.loads(fpath.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            turns = data.get("turns")
            if not isinstance(turns, list) or not turns:
                continue
            sid = str(data.get("session_id") or fpath.stem)
            created_at = float(data.get("created_at") or fpath.stat().st_mtime)
            updated_at = float(data.get("updated_at") or created_at)
            first_q = " ".join(str(turns[0].get("user_prompt", "")).split())
            latest_q = " ".join(str(turns[-1].get("user_prompt", "")).split())
            latest_ans = " ".join(str(turns[-1].get("final_summary", "")).split())
            dt_str = datetime.fromtimestamp(updated_at).strftime("%m-%d %H:%M")
            rel_str = self._format_relative_time(updated_at, now=now_ts)
            records.append(
                {
                    "session_id": sid,
                    "directory": str(data.get("directory") or self.session_dir),
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "updated_at_display": f"{dt_str} · {rel_str}",
                    "relative_time": rel_str,
                    "title": str(data.get("title") or first_q)[:120],
                    "latest_prompt": str(data.get("latest_prompt") or latest_q)[:120],
                    "latest_summary_preview": str(data.get("latest_summary_preview") or latest_ans)[:160],
                    "turn_count": len(turns),
                    "is_current": bool(sid == self.session_id and len(self._saved_turns_records) > 0),
                    "turns": turns,
                }
            )
        records.sort(key=lambda item: (float(item.get("updated_at", 0.0)), str(item.get("session_id", ""))), reverse=True)
        for idx, item in enumerate(records, start=1):
            item["index"] = idx
        return records

    def _make_turn_render_block(
        self,
        turn_id: int,
        user_prompt: str,
        completed_sess: TaskSession,
        saved_stream_blocks: list[dict[str, Any]],
        fold_state: dict[str, Any],
    ) -> Callable[[Console], None]:
        repo_name = Path(self.profile.repo_root).name

        def _render_turn_with_prompt(c: Console) -> None:
            if self._tui_mode:
                render_codex_user_prompt_to(c, user_prompt)
                if saved_stream_blocks:
                    is_collapsed = bool(fold_state.get("collapsed", True))
                    c.print(
                        self._format_execution_fold_header(
                            collapsed=is_collapsed,
                            turn_idx=turn_id,
                            stream_blocks=saved_stream_blocks,
                            session=completed_sess,
                        )
                    )
                    if not is_collapsed:
                        for blk in saved_stream_blocks:
                            CollapsibleProgressTracker._render_stream_block_to(c, blk)
            else:
                c.print(
                    f"[bold #0D9488]fix[/bold #0D9488] [bold #10B981]({escape(repo_name)})[/bold #10B981] "
                    f"[bold #F97316]❯[/bold #F97316] {escape(user_prompt)}"
                )
                for blk in saved_stream_blocks:
                    CollapsibleProgressTracker._render_stream_block_to(c, blk)
            _render_session_result(completed_sess, c, show_telemetry=not self._tui_mode)

        return _render_turn_with_prompt

    def open_resume_modal(self) -> bool:
        """Open the Codex-style interactive `/resume` historical conversation picker modal in Full-Screen TUI mode."""
        sessions = self.list_saved_sessions()
        if not sessions:
            self.show_toast("当前目录暂无历史对话记录", duration_sec=3.0)
            return False
        self._slash_popup_dismissed = True
        self._slash_stage2_cmd = ""
        self._shortcut_overlay_open = False
        if self._input_buffer_ref is not None:
            try:
                self._input_buffer_ref.cancel_completion()
                self._input_buffer_ref.reset()
            except Exception:
                pass
        self._modal_state = {
            "mode": "resume_session",
            "sessions": sessions,
            "selected_idx": 0,
            "filter_query": "",
            "error_msg": "",
            "status_msg": "",
        }
        self._scroll_offset_from_bottom = 0
        self._set_terminal_hover_mouse_tracking(True)
        self.show_toast("↑↓/鼠标悬停选择历史对话 · Enter/鼠标点击恢复 · Esc 取消", duration_sec=3.0)
        if self._tui_app is not None:
            try:
                self._tui_app.invalidate()
            except Exception:
                pass
        return True

    def start_new_conversation_session(self) -> str:
        """
        Start a brand-new conversation session in the current directory (Codex `/new` behavior):
        - Persists the previous conversation session if it had any turns so it remains available in `/resume`.
        - Allocates a fresh `session_id` and resets the screen to a clean new conversation view.
        - Preserves the current directory's `FileHistory` (Up/Down prompt history) and active model settings.
        """
        if self._saved_turns_records:
            self._persist_current_conversation_session()

        self._modal_state = None
        self._set_terminal_hover_mouse_tracking(False)
        self._modal_screen_row_to_item.clear()
        self._modal_doc_line_to_item.clear()
        self._sel_anchor = None
        self._sel_head = None
        self._sel_dragging = False
        if self._input_buffer_ref is not None:
            try:
                self._input_buffer_ref.cancel_completion()
                self._input_buffer_ref.reset()
            except Exception:
                pass

        self.conversation_context.clear()
        self._turn_fold_states.clear()
        self._turn_counter = 0
        self._saved_turns_records.clear()
        self.last_session = None
        self._task_elapsed_sec = 0.0
        self.session_id = self._new_conversation_session_id()
        self.session_created_at = time.time()
        self._scroll_offset_from_bottom = 0
        self._cached_completed_key = None
        self._cached_lines_key = None
        self._agent_state_label = "就绪"
        self._screen_blocks = [
            self._render_minimal_header_to if self._tui_mode else self._render_banner_to
        ]
        self.show_toast("✓ 已在当前目录新建对话", duration_sec=3.0)
        if not self._tui_mode:
            self.console.print("[bold #10B981]✓ 已在当前目录新建对话 (原对话已保存，可通过 `/resume` 随时恢复)。[/bold #10B981]")
        if self._tui_app is not None:
            try:
                self._tui_app.invalidate()
            except Exception:
                pass
        return self.session_id

    def resume_conversation_session(
        self,
        session_id_or_index: str | int | None = None,
    ) -> dict[str, Any] | None:
        """
        Restore a saved conversation session for the current directory (Codex `/resume` behavior):
        - Restores all historical turns into `self._screen_blocks`, `self.conversation_context`, and `self._turn_fold_states`.
        - Keeps `self.session_id` bound to the resumed conversation so subsequent prompts append seamlessly.
        - Pins `self._scroll_offset_from_bottom = 0` so the viewport stays positioned at the latest conversation turn!
        """
        sessions = self.list_saved_sessions()
        if not sessions:
            return None

        target: dict[str, Any] | None = None
        if session_id_or_index is None:
            target = sessions[0]
        elif isinstance(session_id_or_index, int):
            if 0 <= session_id_or_index < len(sessions):
                target = sessions[session_id_or_index]
        else:
            q = str(session_id_or_index).strip()
            if not q or q.lower() in ("latest", "last", "newest"):
                target = sessions[0]
            elif q.lstrip("#").isdigit():
                idx_1 = int(q.lstrip("#")) - 1
                if 0 <= idx_1 < len(sessions):
                    target = sessions[idx_1]
            if target is None:
                for cand in sessions:
                    sid = str(cand.get("session_id", ""))
                    if sid == q or sid.startswith(q):
                        target = cand
                        break
            if target is None:
                q_low = q.lower()
                for cand in sessions:
                    if (
                        q_low in str(cand.get("title", "")).lower()
                        or q_low in str(cand.get("latest_prompt", "")).lower()
                    ):
                        target = cand
                        break

        if target is None:
            return None

        # Close any active modal and reset modal mappings
        self._modal_state = None
        self._set_terminal_hover_mouse_tracking(False)
        self._modal_screen_row_to_item.clear()
        self._modal_doc_line_to_item.clear()
        if self._input_buffer_ref is not None:
            try:
                self._input_buffer_ref.reset()
            except Exception:
                pass

        self.session_id = str(target["session_id"])
        self.session_created_at = float(target.get("created_at") or time.time())
        self._saved_turns_records = list(target.get("turns") or [])
        self.conversation_context.clear()
        self._turn_fold_states.clear()
        self._turn_counter = 0
        self._screen_blocks = [
            self._render_minimal_header_to if self._tui_mode else self._render_banner_to
        ]

        for t_rec in self._saved_turns_records:
            self._turn_counter += 1
            t_id = self._turn_counter
            u_prompt = str(t_rec.get("user_prompt", ""))
            f_summary = str(t_rec.get("final_summary", ""))
            mod_files = [str(f) for f in (t_rec.get("modified_files") or [])]
            s_blocks = list(t_rec.get("stream_blocks") or [])
            self.record_input_history(u_prompt)

            restored_sess = TaskSession(
                repo_root=str(self.profile.repo_root),
                user_prompt=u_prompt,
                final_summary=f_summary,
                modified_files=set(mod_files),
                elapsed_sec=float(t_rec.get("elapsed_sec", 0.0)),
                tool_calls_count=int(t_rec.get("tool_calls_count", 0)),
                tool_calls_log=[str(x) for x in (t_rec.get("tool_calls_log") or [])],
                total_tokens=int(t_rec.get("total_tokens", 0)),
                prompt_tokens=int(t_rec.get("prompt_tokens", 0)),
                completion_tokens=int(t_rec.get("completion_tokens", 0)),
                llm_calls=int(t_rec.get("llm_calls", 0)),
            )
            fold_state: dict[str, Any] = {
                "collapsed": True,
                "turn_id": t_id,
                "blocks_count": len(s_blocks),
            }
            if s_blocks:
                self._turn_fold_states[t_id] = fold_state

            turn_block_fn = self._make_turn_render_block(
                turn_id=t_id,
                user_prompt=u_prompt,
                completed_sess=restored_sess,
                saved_stream_blocks=s_blocks,
                fold_state=fold_state,
            )
            self._screen_blocks.append(turn_block_fn)
            self.conversation_context.append(
                {
                    "user": u_prompt,
                    "assistant": f_summary,
                    "modified_files": ", ".join(mod_files),
                }
            )
            self.last_session = restored_sess
            self._task_elapsed_sec = restored_sess.elapsed_sec

        if len(self._screen_blocks) > 13:
            self._screen_blocks = [self._screen_blocks[0], *self._screen_blocks[-12:]]

        # Keep viewport positioned at the bottom on the latest restored conversation turn (Codex style!)
        self._scroll_offset_from_bottom = 0
        self._cached_completed_key = None
        self._cached_lines_key = None
        self._agent_state_label = "就绪"
        n_turns = len(self._saved_turns_records)
        self.show_toast(f"✓ 已恢复历史对话 (共 {n_turns} 轮 · 当前位于最新对话)", duration_sec=3.0)

        if not self._tui_mode:
            for blk_fn in self._screen_blocks[1:]:
                blk_fn(self.console)
            self.console.print(
                f"[bold #10B981]✓ 已恢复历史对话 `{escape(self.session_id)}` (共 {n_turns} 轮)，当前保持在最新对话位置。[/bold #10B981]"
            )

        if self._tui_app is not None:
            try:
                self._tui_app.invalidate()
            except Exception:
                pass
        return target

    def _auto_bootstrap_workspace(self) -> RepoProfile:
        """Silently initialize `.fix/`, invalidate stale landmarks, and scan co-changes on startup."""
        self.memory.invalidate_stale_facts()
        profile = self.context_mgr.detect_repo_profile(self.toolkit.repo_root)
        try:
            self.evolution.scan_git_cochanges(max_commits=80)
            self.evolution.install_git_post_commit_hook()
            self.looper.schedule_startup_idle_events()
        except Exception:
            pass
        return profile

    def _render_banner_to(self, c: Console) -> None:
        _sync_live_terminal_width(c)
        rules_count = len(self.memory.list_style_rules())
        landmarks = self.memory.search_landmarks(query="", limit=50, include_stale=False)
        skills = self.memory.list_skills()
        model_info = (
            f"[bold #0EA5E9]{self.llm.fast_model}[/bold #0EA5E9] / [bold #10B981]{self.llm.reasoning_model}[/bold #10B981]"
            if self.llm.is_configured
            else "[bold #F97316]Not configured (set FIX_API_KEY in ~/.zshrc)[/bold #F97316]"
        )
        branch_info = (
            f"{self.profile.git_branch} ({self.profile.head_sha[:8]})"
            if self.profile.git_branch
            else "non-git directory"
        )
        autonomy_assess = AutonomyRouter(self.memory).assess_autonomy()

        c.print(
            Rule(
                f"[bold #F97316]✦ Fix Agent v{__version__}[/bold #F97316] [bold #0D9488]— Autonomous Engineering Partner[/bold #0D9488]",
                style="#0D9488",
            )
        )
        c.print(
            f"  [bold #0D9488]Project Root:[/bold #0D9488] [bold]{self.profile.repo_root}[/bold]  [bold #10B981]({branch_info})[/bold #10B981]\n"
            f"  [bold #0D9488]Tech Stack:[/bold #0D9488]   [bold]{', '.join(self.profile.languages)}[/bold]  "
            f"· Test: [bold #0EA5E9]{self.profile.test_command or 'auto'}[/bold #0EA5E9]\n"
            f"  [bold #0D9488]Models:[/bold #0D9488]       {model_info}\n"
            f"  [bold #0D9488]Mind Mirror:[/bold #0D9488]  [bold #F97316]{rules_count}[/bold #F97316] personal rules · "
            f"[bold #0EA5E9]{len(landmarks)}[/bold #0EA5E9] verified landmarks · [bold #10B981]{len(skills)}[/bold #10B981] bootstrapped skills  "
            f"· Autonomy: [bold #F97316]{autonomy_assess.level.value}[/bold #F97316] "
            f"[bold]({autonomy_assess.score:.2f})[/bold]\n"
            f"  直接输入自然语言即可自主探索与完成任务；输入框为空时按 Ctrl+C 直接退出；支持窗口大小实时自适应重排。\n"
            f"  [bold #0D9488]快捷命令:[/bold #0D9488] /help · /steps · /status · /bug · /ask · /opt · /trace · /diag · /profile · /skills · /autonomy · /undo · /exit"
        )
        c.print(Rule(style="#0D9488"))

    def render_welcome_banner(self) -> None:
        if not self._screen_blocks:
            self._screen_blocks.append(self._render_banner_to)
        self._render_banner_to(self.console)

    def _render_minimal_header_to(self, c: Console) -> None:
        _sync_live_terminal_width(c)
        repo_name = Path(self.profile.repo_root).name
        branch_info = f"{self.profile.git_branch}" if self.profile.git_branch else "local"
        model_badge = f"[bold #0EA5E9]{escape(self.llm.fast_model)}[/bold #0EA5E9]"
        if getattr(self.llm, "reasoning_model", "") and self.llm.reasoning_model != self.llm.fast_model:
            model_badge += f" [dim]/[/dim] [bold #10B981]{escape(self.llm.reasoning_model)}[/bold #10B981]"
        c.print(
            f"[bold #F97316]✦ Fix v{__version__}[/bold #F97316]  ·  "
            f"[bold #0D9488]{escape(repo_name)}[/bold #0D9488] [bold #10B981]({escape(branch_info)})[/bold #10B981]  ·  "
            f"{model_badge}"
        )

    def show_toast(self, message: str, duration_sec: float = 3.0) -> None:
        """Display a right-aligned transient notification for `duration_sec` seconds (default 3s)."""
        self._toast_message = message
        self._toast_expires_at = time.time() + duration_sec

    def get_right_toast_text(self, now: float | None = None) -> str:
        """Return the active right-aligned transient toast if within its 3-second display window."""
        curr = now if now is not None else time.time()
        if self._toast_message and curr < self._toast_expires_at:
            return self._toast_message
        return ""

    def get_claude_working_status_parts(
        self,
        now: float | None = None,
    ) -> tuple[str, str, str, str, float] | None:
        """
        Return `(spinner, action_text, elapsed_hms, meta_tail, elapsed_sec)` in Claude Code style
        when Fix is actively working (`_active_tracker` is running or `_agent_state_label == '正在执行'`).
        """
        curr = now if now is not None else time.time()
        if self._active_tracker is not None:
            elapsed = max(0.0, curr - self._active_tracker.started_at)
            spinner = CLAUDE_SPINNER_FRAMES[int(elapsed * 8) % len(CLAUDE_SPINNER_FRAMES)]
            raw_act = self._active_tracker.current_action_label or "正在思考与分析代码..."
            if raw_act == "正在思考与分析代码..." and self._active_tracker.total_tools == 0:
                if elapsed >= 12.0:
                    raw_act = "正在深入分析代码架构与逻辑链路..."
                elif elapsed >= 7.0:
                    raw_act = "正在构思执行方案与推理路径..."
                elif elapsed >= 3.0:
                    raw_act = "正在理解需求与分析项目上下文..."
            act = _truncate_middle_for_width(raw_act, 48)
            elapsed_str = format_elapsed_hms(elapsed)
            meta_parts: list[str] = ["esc/ctrl+c 中断"]
            if self._active_tracker.total_tools > 0:
                meta_parts.append(f"{self._active_tracker.total_tools} 次工具")
            tok_total = getattr(self._active_tracker, "total_tokens", 0)
            if tok_total > 0:
                tok_str = f"{tok_total / 1000:.1f}k" if tok_total >= 1000 else str(tok_total)
                meta_parts.append(f"↓ {tok_str} tokens")
            meta_tail = f"({' · '.join(meta_parts)})"
            return spinner, act, elapsed_str, meta_tail, elapsed
        if self._agent_state_label == "正在执行":
            elapsed = max(0.0, self._task_elapsed_sec)
            spinner = CLAUDE_SPINNER_FRAMES[int(curr * 8) % len(CLAUDE_SPINNER_FRAMES)]
            act = "正在思考与分析代码..."
            elapsed_str = format_elapsed_hms(elapsed)
            return spinner, act, elapsed_str, "(esc/ctrl+c 中断)", elapsed
        return None

    def get_left_status_text(self) -> str:
        """Return the bottom-left status indicator: Claude Code working status when active, or idle status."""
        working_parts = self.get_claude_working_status_parts()
        if working_parts is not None:
            spinner, act, elapsed_str, meta_tail, _ = working_parts
            return f"{spinner} {act} · {elapsed_str} {meta_tail}"
        if getattr(self.looper, "active_idle_tick", None) is not None:
            t_item = self.looper.active_idle_tick
            t_name = Path(t_item.target).name or t_item.target
            return f"✦ 自我进化中: {t_item.kind.value}({t_name}) · {format_elapsed_hms(self._task_elapsed_sec)}"
        return f"● {self._agent_state_label} · {format_elapsed_hms(self._task_elapsed_sec)}"

    def format_status_row(self, width: int, now: float | None = None) -> str:
        """Format the dual-purpose status row (Left: Agent Status · Elapsed Time | Right: 3s Toast flush-right)."""
        from rich.cells import cell_len

        left = f" {self.get_left_status_text()}"
        right = self.get_right_toast_text(now=now)
        if not right:
            return left
        right_padded = f"{right} "
        gap = max(2, width - cell_len(left) - cell_len(right_padded))
        return f"{left}{' ' * gap}{right_padded}"

    def get_context_left_percent(self) -> int:
        """Compute remaining context window percentage (Codex `100% context left` indicator)."""
        max_ctx_tokens = 128000
        used_tokens = 0
        if self.last_session is not None and getattr(self.last_session, "total_tokens", 0) > 0:
            used_tokens = max(
                int(getattr(self.last_session, "prompt_tokens", 0)),
                int(getattr(self.last_session, "total_tokens", 0)),
            )
        elif self.conversation_context:
            total_chars = sum(
                len(str(turn.get("user", ""))) + len(str(turn.get("assistant", "")))
                for turn in self.conversation_context
            )
            used_tokens = max(1, total_chars // 3)
        if used_tokens <= 0:
            return 100
        pct = round(100.0 * (1.0 - min(0.99, used_tokens / float(max_ctx_tokens))))
        return max(1, min(100, int(pct)))

    def format_codex_footer_row(self, width: int) -> str:
        """Format the Codex-style footer hint bar below the composer (`  ? for shortcuts ... model · 100% context left  `)."""
        from rich.cells import cell_len

        draft_txt = self._input_buffer_ref.text if self._input_buffer_ref is not None else ""
        if self._shortcut_overlay_open:
            left = "  ? / esc close shortcuts"
        elif self._modal_state is None and draft_txt.startswith("/") and " " not in draft_txt and "\n" not in draft_txt and not self._slash_popup_dismissed:
            left = "  ↑↓ select · enter run · tab complete · esc close"
        elif not draft_txt:
            left = "  ? for shortcuts · / commands · ↑↓ history"
        else:
            task_running = (self._running_task is not None and not self._running_task.done()) or self._active_tracker is not None
            if task_running:
                left = "  enter to steer · esc to interrupt"
            else:
                left = "  enter to send · ctrl+j newline · ctrl+c clear"
        pct = self.get_context_left_percent()
        model_label = getattr(self.llm, "fast_model", "") or "codex"
        right = f"{model_label} · {pct}% context left  "
        if cell_len(left) + cell_len(right) + 2 > width:
            short_right = f"{pct}% context left  "
            if cell_len(left) + cell_len(short_right) + 2 <= width:
                right = short_right
            else:
                return left
        gap = max(2, width - cell_len(left) - cell_len(right))
        return f"{left}{' ' * gap}{right}"

    def copy_text_to_clipboard(self, text_to_copy: str, toast_label: str = "✓ 复制成功") -> bool:
        """Copy arbitrary text to the OS clipboard (pbcopy / xclip / xsel / wl-copy + OSC 52) and show a 3s toast."""
        if not text_to_copy:
            self.show_toast("暂无可复制内容", duration_sec=3.0)
            return False
        self._last_selected_text = text_to_copy
        for clip_cmd in (
            ["pbcopy"],
            ["wl-copy"],
            ["xclip", "-selection", "clipboard"],
            ["xsel", "--clipboard", "--input"],
        ):
            if shutil.which(clip_cmd[0]):
                try:
                    subprocess.run(clip_cmd, input=text_to_copy, text=True, timeout=2, check=False)
                    break
                except Exception:
                    pass
        # Also emit OSC 52 clipboard sequence for terminal emulators (iTerm2, Terminal.app, Warp, kitty, WezTerm)
        if self._tui_app is not None and getattr(self._tui_app, "output", None) is not None:
            try:
                import base64
                from prompt_toolkit.output.defaults import DummyOutput

                if not isinstance(self._tui_app.output, DummyOutput):
                    b64 = base64.b64encode(text_to_copy.encode("utf-8")).decode("ascii")
                    if len(b64) <= 100000:
                        self._tui_app.output.write_raw(f"\033]52;c;{b64}\a")
                        self._tui_app.output.flush()
            except Exception:
                pass
        self.show_toast(toast_label, duration_sec=3.0)
        return True

    @staticmethod
    def _format_execution_fold_header(
        collapsed: bool,
        turn_idx: int,
        stream_blocks: list[dict[str, Any]],
        session: TaskSession,
    ) -> Text:
        n_narr = sum(1 for b in stream_blocks if b.get("type") == "narrative")
        n_exp = sum(len(b.get("lines", [])) for b in stream_blocks if b.get("type") == "explored")
        n_ran = sum(1 for b in stream_blocks if b.get("type") == "ran")
        n_edit = sum(1 for b in stream_blocks if b.get("type") == "edited")
        parts: list[str] = []
        if session.tool_calls_count:
            parts.append(f"{session.tool_calls_count} 次工具调用")
        if n_narr:
            parts.append(f"{n_narr} 步思考")
        if n_exp:
            parts.append(f"{n_exp} 项代码探索")
        if n_ran:
            parts.append(f"{n_ran} 次命令")
        if n_edit:
            parts.append(f"{n_edit} 处修改")
        if not parts:
            parts.append(f"{len(stream_blocks)} 步操作")
        summary_str = " · ".join(parts)
        hdr = Text()
        if collapsed:
            hdr.append("▸ ", style="bold #0D9488")
            hdr.append(
                f"已自动收起执行过程 (#{turn_idx}: {summary_str} · 耗时 {format_elapsed_hms(session.elapsed_sec)}) ",
                style=SECONDARY_STEP_TEXT_STYLE,
            )
            hdr.append("[点击此行 / Ctrl+O / /steps 展开]", style="#0EA5E9")
        else:
            hdr.append("▾ ", style="bold #F97316")
            hdr.append(
                f"已展开执行过程 (#{turn_idx}: {summary_str} · 耗时 {format_elapsed_hms(session.elapsed_sec)}) ",
                style=SECONDARY_STEP_TEXT_STYLE,
            )
            hdr.append("[点击此行 / Ctrl+O / /steps 收起]", style="#0EA5E9")
        return hdr

    def toggle_turn_steps_collapsed(self, turn_idx: int | None = None) -> bool | None:
        """
        Toggle the collapsed/expanded state of the execution steps (`✦`, `• Explored`, `• Ran`)
        for `turn_idx` (or the latest turn with execution steps if `turn_idx` is None).
        Returns the new `collapsed` boolean (`True` = collapsed, `False` = expanded), or `None` if no turn exists.
        """
        if not self._turn_fold_states:
            return None
        target_id = turn_idx if (turn_idx is not None and turn_idx in self._turn_fold_states) else max(self._turn_fold_states.keys())
        st = self._turn_fold_states.get(target_id)
        if st is None:
            return None
        new_collapsed = not bool(st.get("collapsed", True))
        st["collapsed"] = new_collapsed
        self._cached_completed_key = None
        self._cached_lines_key = None
        self.show_toast(
            f"✓ 已{'收起' if new_collapsed else '展开'} #{target_id} 执行过程",
            duration_sec=3.0,
        )
        if self._tui_app is not None:
            try:
                self._tui_app.invalidate()
            except Exception:
                pass
        return new_collapsed

    def _find_turn_fold_at_doc_line(self, doc_line: int) -> int | None:
        """Check if `doc_line` in `_cached_render_lines` is an execution-steps fold header line (`#<id>`)."""
        if not self._cached_render_lines or doc_line < 0 or doc_line >= len(self._cached_render_lines):
            return None
        for idx_check in (doc_line, doc_line - 1):
            if 0 <= idx_check < len(self._cached_render_lines):
                line_str = "".join(seg for _, seg in self._cached_render_lines[idx_check])
                if ("已自动收起执行过程 (#" in line_str) or ("已展开执行过程 (#" in line_str):
                    m = re.search(r"\(#(\d+):", line_str)
                    if m:
                        t_id = int(m.group(1))
                        if t_id in self._turn_fold_states:
                            return t_id
                    if self._turn_fold_states:
                        return max(self._turn_fold_states.keys())
        return None

    def _set_terminal_hover_mouse_tracking(self, enable_hover: bool) -> None:
        """
        Ensure SGR Any-Event Mouse Tracking (`?1003h`) is active on the terminal (just like prompt_toolkit's
        default `Vt100_Output.enable_mouse_support`) so hovering the mouse cursor over any vendor or model row
        immediately sends SGR motion events (`35;x;yM`) and highlights that option in real time!
        """
        seq = (
            "\033[?1000h\033[?1002h\033[?1003h\033[?1015h\033[?1006h"
            if enable_hover
            else "\033[?1000h\033[?1002h\033[?1003h\033[?1015h\033[?1006h"
        )
        if self._tui_app is not None and getattr(self._tui_app, "output", None) is not None:
            try:
                out_obj = self._tui_app.output
                if hasattr(out_obj, "write_raw"):
                    out_obj.write_raw(seq)
                    if hasattr(out_obj, "flush"):
                        out_obj.flush()
            except Exception:
                pass
        raw_out = getattr(sys, "__stdout__", sys.stdout)
        if raw_out is not None and hasattr(raw_out, "isatty"):
            try:
                if raw_out.isatty():
                    raw_out.write(seq)
                    raw_out.flush()
            except Exception:
                pass

    def open_connect_modal(self) -> None:
        """Open the 3-step interactive `/connect` popup modal in Full-Screen TUI mode."""
        self._slash_popup_dismissed = True
        self._slash_stage2_cmd = ""
        self._shortcut_overlay_open = False
        if self._input_buffer_ref is not None:
            try:
                self._input_buffer_ref.cancel_completion()
                self._input_buffer_ref.reset()
            except Exception:
                pass
        providers = self.llm.list_supported_providers()
        active_idx = next((i for i, p in enumerate(providers) if p.get("active")), 0)
        self._modal_state = {
            "mode": "connect_provider",
            "providers": providers,
            "selected_idx": active_idx,
            "provider_id": providers[active_idx]["provider_id"] if providers else "deepseek",
            "filter_query": "",
            "api_key_input": "",
            "error_msg": "",
            "status_msg": "",
            "models": [],
        }
        self._scroll_offset_from_bottom = 0
        self._set_terminal_hover_mouse_tracking(True)
        self.show_toast("↑↓/鼠标悬停或输入关键词选择厂商 · Enter/点击确认 · Esc 取消", duration_sec=3.0)
        if self._tui_app is not None:
            try:
                self._tui_app.invalidate()
            except Exception:
                pass

    def open_model_selector_modal(self) -> None:
        """Open the interactive `/model` model-switching popup modal in Full-Screen TUI mode."""
        self._slash_popup_dismissed = True
        self._slash_stage2_cmd = ""
        self._shortcut_overlay_open = False
        if self._input_buffer_ref is not None:
            try:
                self._input_buffer_ref.cancel_completion()
                self._input_buffer_ref.reset()
            except Exception:
                pass
        models = self.llm.list_switchable_models()
        cur_idx = next((i for i, m in enumerate(models) if m.get("is_current")), 0)
        self._modal_state = {
            "mode": "switch_model",
            "models": models,
            "selected_idx": cur_idx,
            "filter_query": "",
            "error_msg": "",
            "status_msg": "",
        }
        self._scroll_offset_from_bottom = 0
        self._set_terminal_hover_mouse_tracking(True)
        self.show_toast("↑↓/鼠标悬停或输入关键词选择模型 · Enter/点击切换 · Esc 关闭", duration_sec=3.0)
        if self._tui_app is not None:
            try:
                self._tui_app.invalidate()
            except Exception:
                pass

    def _apply_modal_filter_query(self, raw_query: str) -> int | None:
        """
        Live-filter and jump `selected_idx` in the active `/resume`, `/model`, or `/connect` modal
        as the user types in the input box (`input_window`).
        """
        if self._modal_state is None:
            return None
        mode = self._modal_state.get("mode")
        q = raw_query.strip()
        self._modal_state["filter_query"] = q
        if not q:
            return int(self._modal_state.get("selected_idx", 0))
        q_low = q.lower()

        if mode == "connect_provider":
            providers = self._modal_state.get("providers") or self.llm.list_supported_providers()
            if not providers:
                return None
            if q_low.lstrip("#").isdigit():
                idx = max(0, min(len(providers) - 1, int(q_low.lstrip("#")) - 1))
                self._modal_state["selected_idx"] = idx
                return idx
            pid = resolve_provider_id(q_low)
            for idx, p in enumerate(providers):
                if (
                    (pid and p.get("provider_id") == pid)
                    or q_low in str(p.get("provider_id", "")).lower()
                    or q_low in str(p.get("display_name", "")).lower()
                    or q_low in str(p.get("vendor_label", "")).lower()
                    or q_low in str(p.get("base_url", "")).lower()
                ):
                    self._modal_state["selected_idx"] = idx
                    return idx
            return int(self._modal_state.get("selected_idx", 0))

        if mode == "connect_select_model":
            models = self._modal_state.get("models") or []
            if not models:
                return None
            if q_low.lstrip("#").isdigit():
                idx = max(0, min(len(models) - 1, int(q_low.lstrip("#")) - 1))
                self._modal_state["selected_idx"] = idx
                return idx
            for idx, m_name in enumerate(models):
                if q_low in str(m_name).lower():
                    self._modal_state["selected_idx"] = idx
                    return idx
            return int(self._modal_state.get("selected_idx", 0))

        if mode == "switch_model":
            models = self._modal_state.get("models") or []
            total = len(models) + 1
            if q_low.lstrip("#").isdigit():
                idx = max(0, min(total - 1, int(q_low.lstrip("#")) - 1))
                self._modal_state["selected_idx"] = idx
                return idx
            if any(k in q_low for k in ("connect", "连接", "其他", "新厂商", "+")):
                self._modal_state["selected_idx"] = len(models)
                return len(models)
            for idx, entry in enumerate(models):
                if (
                    q_low in str(entry.get("model", "")).lower()
                    or q_low in str(entry.get("provider_name", "")).lower()
                    or q_low in str(entry.get("vendor_label", "")).lower()
                ):
                    self._modal_state["selected_idx"] = idx
                    return idx
            return int(self._modal_state.get("selected_idx", 0))

        if mode == "resume_session":
            sessions = self._modal_state.get("sessions") or self.list_saved_sessions()
            if not sessions:
                return None
            if q_low.lstrip("#").isdigit():
                idx = max(0, min(len(sessions) - 1, int(q_low.lstrip("#")) - 1))
                self._modal_state["selected_idx"] = idx
                return idx
            for idx, s_item in enumerate(sessions[:15]):
                if (
                    q_low in str(s_item.get("latest_prompt", "")).lower()
                    or q_low in str(s_item.get("title", "")).lower()
                    or q_low in str(s_item.get("session_id", "")).lower()
                    or q_low in str(s_item.get("updated_at_display", "")).lower()
                ):
                    self._modal_state["selected_idx"] = idx
                    return idx
            return int(self._modal_state.get("selected_idx", 0))

        return None

    def match_slash_commands(self, query_token: str = "/") -> list[dict[str, Any]]:
        """
        Claude Code style multi-dimension slash command matcher:
        Matches by command prefix (`/re`), substring (`/mod`), alias (`/q`), OR Chinese/English
        keywords (`/恢复`, `/模型`, `/新建`, `/连接`, `/回滚`, `/规则`, `/技能`), sorted by relevance.
        """
        raw = query_token.strip()
        if not raw.startswith("/"):
            return []
        q = raw[1:].strip().lower()
        if not q:
            return [dict(spec) for spec in SLASH_COMMAND_SPECS if spec.get("canonical", True)]

        scored: list[tuple[int, int, dict[str, Any]]] = []
        for idx_s, spec in enumerate(SLASH_COMMAND_SPECS):
            cmd_bare = spec["cmd"][1:].lower()
            desc_low = spec["desc"].lower()
            cat_low = spec["category"].lower()
            kws = tuple(str(k).lower() for k in spec.get("keywords", ()))
            if not spec.get("canonical", True) and not cmd_bare.startswith(q):
                continue
            tier: int | None = None
            if cmd_bare == q:
                tier = 0
            elif cmd_bare.startswith(q):
                tier = 1
            elif any(k.startswith(q) for k in kws):
                tier = 2
            elif q in cmd_bare:
                tier = 3
            elif q in cat_low or q in desc_low or any(q in k for k in kws):
                tier = 4
            if tier is not None:
                scored.append((tier, idx_s, dict(spec)))

        # Deduplicate if a non-canonical alias and its canonical target both matched
        scored.sort(key=lambda item: (item[0], item[1]))
        return [item[2] for item in scored]

    def get_slash_stage2_options(self, cmd_name: str, arg_query: str = "") -> list[dict[str, Any]]:
        """
        Claude Code style in-input interactive argument & subcommand provider:
        Returns live-filtered selectable options for `cmd_name` right inside the input box dock!
        """
        cmd = cmd_name.strip().lower()
        q = arg_query.strip()
        q_low = q.lower()
        options: list[dict[str, Any]] = []

        if cmd == "/model":
            switchable = self.llm.list_switchable_models()
            for idx_m, item in enumerate(switchable, start=1):
                roles: list[str] = []
                if item.get("is_current_fast"):
                    roles.append("Fast")
                if item.get("is_current_reasoning"):
                    roles.append("Reasoning")
                badge = f"● 当前使用 ({'/'.join(roles)})" if roles else "✓ 可直接切换"
                opt = {
                    "label": f"{idx_m}. [{item['provider_name']}] {item['model']}",
                    "badge": badge,
                    "desc": f"{item['vendor_label']} · 按 Enter 立即切换",
                    "fill_text": f"/model {item['model']}",
                    "exec_line": f"/model {item['model']}",
                    "action": "switch_model_direct",
                    "provider_id": str(item["provider_id"]),
                    "model": str(item["model"]),
                    "is_current": bool(item.get("is_current")),
                    "enter_hint": "Enter 立即切换",
                }
                if (
                    not q_low
                    or (q_low.lstrip("#").isdigit() and int(q_low.lstrip("#")) == idx_m)
                    or q_low in str(item["model"]).lower()
                    or q_low in str(item["provider_name"]).lower()
                    or q_low in str(item["vendor_label"]).lower()
                ):
                    options.append(opt)
            conn_opt = {
                "label": "+ 连接其他大模型厂商 (/connect)",
                "badge": "7大厂商",
                "desc": "DeepSeek / GLM / MiMo / MiniMax / Qwen / OpenAI / Anthropic",
                "fill_text": "/connect ",
                "exec_line": "/connect",
                "action": "transition_to_connect",
                "enter_hint": "Enter/Tab 连接新厂商",
            }
            if not q_low or any(k in q_low for k in ("connect", "连接", "厂商", "新", "+")):
                options.append(conn_opt)
            return options

        if cmd == "/resume":
            sessions = self.list_saved_sessions()
            if not sessions:
                return [
                    {
                        "label": "(当前目录暂无历史对话记录)",
                        "badge": "空",
                        "desc": "发送提问后会自动按目录保存对话，随时可通过 /resume 恢复",
                        "fill_text": "/resume",
                        "exec_line": "/resume",
                        "action": "noop_empty",
                        "enter_hint": "Esc 返回",
                    }
                ]
            for idx_s, s_item in enumerate(sessions[:15], start=1):
                t_disp = str(s_item.get("updated_at_display", ""))
                n_turns = int(s_item.get("turn_count", 1))
                latest_q = str(s_item.get("latest_prompt") or s_item.get("title") or "(空对话)")
                badge = "● 当前会话" if s_item.get("is_current") else ("★ 最新" if idx_s == 1 else f"{n_turns}轮")
                opt = {
                    "label": f"{idx_s}. [{t_disp}] ({n_turns}轮)",
                    "badge": badge,
                    "desc": latest_q,
                    "fill_text": f"/resume {idx_s}",
                    "exec_line": f"/resume {s_item['session_id']}",
                    "action": "resume_direct",
                    "session_id": str(s_item["session_id"]),
                    "enter_hint": "Enter 立即恢复",
                }
                if (
                    not q_low
                    or (q_low.lstrip("#").isdigit() and int(q_low.lstrip("#")) == idx_s)
                    or q_low in latest_q.lower()
                    or q_low in str(s_item.get("title", "")).lower()
                    or q_low in str(s_item.get("session_id", "")).lower()
                    or q_low in t_disp.lower()
                ):
                    options.append(opt)
            return options

        if cmd == "/connect":
            tokens = q.split()
            has_trailing_space = arg_query.endswith(" ") and len(tokens) >= 1
            resolved_first = resolve_provider_id(tokens[0]) if tokens else None
            if resolved_first and (len(tokens) >= 2 or has_trailing_space):
                spec = self.llm.get_provider_spec(resolved_first)
                typed_key = tokens[1] if len(tokens) >= 2 else ""
                if len(typed_key) > 8:
                    masked = typed_key[:4] + ("*" * min(16, len(typed_key) - 6)) + typed_key[-2:]
                elif typed_key:
                    masked = typed_key[:2] + ("*" * max(1, len(typed_key) - 2))
                else:
                    masked = "等待输入..."
                return [
                    {
                        "label": f"🔑 {spec['vendor_label']} API Key ({masked})",
                        "badge": spec["display_name"],
                        "desc": f"接口: {spec['base_url']} · 输入/粘贴 API Key 后按 Enter 自动验证并拉取模型列表",
                        "fill_text": f"/connect {resolved_first} {typed_key}".rstrip() + (" " if not typed_key else ""),
                        "exec_line": f"/connect {resolved_first} {typed_key}".strip(),
                        "action": "connect_verify_key" if typed_key else "noop_require_arg",
                        "ghost_hint": "" if typed_key else f"<在此粘贴 {spec['display_name']} API Key 后按 Enter 验证连接>",
                        "enter_hint": "Enter 验证并连接" if typed_key else "请先输入 API Key",
                    }
                ]
            providers = self.llm.list_supported_providers()
            for p in providers:
                badge = (
                    "● 当前使用"
                    if p.get("active")
                    else ("✓ 已连接" if p.get("connected") else "未连接")
                )
                opt = {
                    "label": f"{p['index']}. {p['display_name']}",
                    "badge": badge,
                    "desc": f"{p['vendor_label']} · 代表模型: {', '.join(p['models'][:3])}",
                    "fill_text": f"/connect {p['provider_id']} ",
                    "exec_line": f"/connect {p['provider_id']}",
                    "action": "connect_step_vendor",
                    "provider_id": str(p["provider_id"]),
                    "enter_hint": "Tab/Enter 选择厂商并输入 Key",
                }
                if (
                    not q_low
                    or (q_low.lstrip("#").isdigit() and int(q_low.lstrip("#")) == int(p["index"]))
                    or (resolved_first and p["provider_id"] == resolved_first)
                    or q_low in str(p["provider_id"]).lower()
                    or q_low in str(p["display_name"]).lower()
                    or q_low in str(p["vendor_label"]).lower()
                ):
                    options.append(opt)
            return options

        if cmd == "/rules":
            rules = self.memory.list_style_rules(include_candidate=True)
            base_opts = [
                {
                    "label": "1. 查看全部开发风格规则 (Mind Mirror)",
                    "badge": f"共 {len(rules)} 条",
                    "desc": "展示当前仓库与全局已学习的个人代码风格规则表",
                    "fill_text": "/rules",
                    "exec_line": "/rules",
                    "action": "exec_line",
                    "enter_hint": "Enter 查看规则",
                }
            ]
            for r in rules[:8]:
                base_opts.append(
                    {
                        "label": f"approve {r.rule_id}",
                        "badge": r.status.value,
                        "desc": f"批准固化规则: {r.pattern_desc} -> {r.preferred_way}",
                        "fill_text": f"/rules approve {r.rule_id}",
                        "exec_line": f"/rules approve {r.rule_id}",
                        "action": "exec_line",
                        "enter_hint": "Enter 批准规则",
                    }
                )
                base_opts.append(
                    {
                        "label": f"reject {r.rule_id}",
                        "badge": r.status.value,
                        "desc": f"拒绝归档规则: {r.pattern_desc}",
                        "fill_text": f"/rules reject {r.rule_id}",
                        "exec_line": f"/rules reject {r.rule_id}",
                        "action": "exec_line",
                        "enter_hint": "Enter 拒绝规则",
                    }
                )
            return [
                o
                for o in base_opts
                if not q_low or q_low in o["label"].lower() or q_low in o["desc"].lower()
            ]

        if cmd == "/skills":
            skills = self.memory.list_skills()
            base_opts = [
                {
                    "label": "1. 查看已注册诊断技能列表",
                    "badge": f"共 {len(skills)} 个",
                    "desc": "列出 .fix/skills/ 下所有可执行项目技能",
                    "fill_text": "/skills",
                    "exec_line": "/skills",
                    "action": "exec_line",
                    "enter_hint": "Enter 查看列表",
                },
                {
                    "label": "2. bootstrap (自动生成项目专属技能)",
                    "badge": "生成",
                    "desc": "扫描当前项目架构并自动生成 .fix/skills/<slug>/run.py 可执行技能",
                    "fill_text": "/skills bootstrap",
                    "exec_line": "/skills bootstrap",
                    "action": "exec_line",
                    "enter_hint": "Enter 立即生成",
                },
            ]
            for slug in skills:
                base_opts.append(
                    {
                        "label": f"run {slug}",
                        "badge": "可执行",
                        "desc": f"立即执行项目专属诊断技能 `{slug}`",
                        "fill_text": f"/skills run {slug}",
                        "exec_line": f"/skills run {slug}",
                        "action": "exec_line",
                        "enter_hint": "Enter 执行技能",
                    }
                )
            return [
                o
                for o in base_opts
                if not q_low or q_low in o["label"].lower() or q_low in o["desc"].lower()
            ]

        if cmd == "/idle":
            st = self.looper.get_status_dict()
            base_opts = [
                {
                    "label": "1. 查看后台进化队列状态 (status)",
                    "badge": f"排队 {st['p2_idle_queued']} 项",
                    "desc": "查看 Serial Looper 与 IdleHandler P2 队列、已完成 Tick 与预算",
                    "fill_text": "/idle",
                    "exec_line": "/idle",
                    "action": "exec_line",
                    "enter_hint": "Enter 查看状态",
                },
                {
                    "label": "2. run (立即执行后台 P2 进化队列)",
                    "badge": "立即执行",
                    "desc": "立即排空并运行后台 IdleHandler 自我进化与架构心智提炼任务",
                    "fill_text": "/idle run",
                    "exec_line": "/idle run",
                    "action": "exec_line",
                    "enter_hint": "Enter 立即运行",
                },
            ]
            return [
                o
                for o in base_opts
                if not q_low or q_low in o["label"].lower() or q_low in o["desc"].lower()
            ]

        if cmd in ("/evolve", "/insights"):
            base_opts = [
                {
                    "label": f"1. 查看三层第一性原理心智模型与优化提案 ({cmd})",
                    "badge": "洞察报告",
                    "desc": "展示现象层 -> 机制契约层 -> 第一性原理本质及主动优化提案",
                    "fill_text": cmd,
                    "exec_line": cmd,
                    "action": "exec_line",
                    "enter_hint": "Enter 查看报告",
                },
                {
                    "label": f"2. {cmd} run (立即触发后台提炼并展示)",
                    "badge": "立即提炼",
                    "desc": "先运行后台进化队列提炼最新心智模型，再展示完整三层洞察报告",
                    "fill_text": f"{cmd} run",
                    "exec_line": f"{cmd} run",
                    "action": "exec_line",
                    "enter_hint": "Enter 提炼并查看",
                },
            ]
            return [
                o
                for o in base_opts
                if not q_low or q_low in o["label"].lower() or q_low in o["desc"].lower()
            ]

        if cmd == "/steps":
            base_opts = [
                {
                    "label": "1. 展开/收起上一轮任务工具调用明细",
                    "badge": "Ctrl+O",
                    "desc": "切换最新一轮任务的探索与命令执行详情折叠状态",
                    "fill_text": "/steps",
                    "exec_line": "/steps",
                    "action": "exec_line",
                    "enter_hint": "Enter 切换折叠",
                }
            ]
            for t_id in sorted(self._turn_fold_states.keys(), reverse=True):
                f_st = self._turn_fold_states[t_id]
                is_col = bool(f_st.get("collapsed", True))
                n_b = int(f_st.get("blocks_count", 0))
                base_opts.append(
                    {
                        "label": f"#{t_id} 轮执行过程 ({n_b} 步操作)",
                        "badge": "当前已收起" if is_col else "当前已展开",
                        "desc": f"点击或按 Enter 立即{'展开' if is_col else '收起'}第 #{t_id} 轮工具调用明细",
                        "fill_text": f"/steps #{t_id}",
                        "exec_line": f"/steps #{t_id}",
                        "action": "exec_line",
                        "enter_hint": f"Enter {'展开' if is_col else '收起'}",
                    }
                )
            return [
                o
                for o in base_opts
                if not q_low or q_low in o["label"].lower() or q_low in o["desc"].lower()
            ]

        if cmd == "/diag":
            base_opts = [
                {
                    "label": "1. 全量工作区静态语法与导入诊断",
                    "badge": "全量检查",
                    "desc": "扫描并检查当前仓库所有源码文件的语法、AST 与导入完整性",
                    "fill_text": "/diag",
                    "exec_line": "/diag",
                    "action": "exec_line",
                    "enter_hint": "Enter 运行诊断",
                }
            ]
            try:
                repo_root = Path(self.toolkit.repo_root)
                py_files = sorted(
                    str(p.relative_to(repo_root))
                    for p in repo_root.rglob("*.py")
                    if ".git" not in p.parts and ".fix" not in p.parts and "__pycache__" not in p.parts
                )[:12]
                for rel_f in py_files:
                    if not q_low or q_low in rel_f.lower():
                        base_opts.append(
                            {
                                "label": f"诊断文件: {rel_f}",
                                "badge": "单文件",
                                "desc": f"对 `{rel_f}` 运行静态语法与导入诊断",
                                "fill_text": f"/diag {rel_f}",
                                "exec_line": f"/diag {rel_f}",
                                "action": "exec_line",
                                "enter_hint": "Enter 诊断该文件",
                            }
                        )
            except Exception:
                pass
            return base_opts[:10]

        if cmd == "/trace":
            landmarks = self.memory.search_landmarks(query=q, limit=6, include_stale=False)
            if q:
                options.append(
                    {
                        "label": f"追踪符号: {q}",
                        "badge": "LSP/AST",
                        "desc": f"跨文件检索符号 `{q}` 的定义位置、调用方 (Callers) 与被调用方 (Callees)",
                        "fill_text": f"/trace {q}",
                        "exec_line": f"/trace {q}",
                        "action": "exec_line",
                        "enter_hint": "Enter 立即追踪",
                    }
                )
            else:
                options.append(
                    {
                        "label": "请在输入框输入要追踪的符号名 (如函数名或类名)",
                        "badge": "必填参数",
                        "desc": "用法: /trace <symbol_name> [rel_path] — 跨文件追踪定义与调用链",
                        "fill_text": "/trace ",
                        "exec_line": "/trace",
                        "action": "noop_require_arg",
                        "ghost_hint": "<输入函数名或类名，按 Enter 追踪跨文件调用链>",
                        "enter_hint": "请输入符号名",
                    }
                )
            for lm in landmarks:
                for ev_p in lm.evidence_paths[:1]:
                    sym_cand = Path(ev_p).stem
                    if sym_cand and (not q_low or q_low in sym_cand.lower()):
                        options.append(
                            {
                                "label": f"追踪地标模块: {sym_cand} ({ev_p})",
                                "badge": lm.category,
                                "desc": lm.summary[:60],
                                "fill_text": f"/trace {sym_cand} {ev_p}",
                                "exec_line": f"/trace {sym_cand} {ev_p}",
                                "action": "exec_line",
                                "enter_hint": "Enter 追踪",
                            }
                        )
            return options[:8]

        if cmd == "/new":
            if q:
                return [
                    {
                        "label": f"新建对话并直接提问: {q}",
                        "badge": "新建+提问",
                        "desc": "在当前目录开启全新会话并立即执行该问题 (原对话自动保存至 /resume)",
                        "fill_text": f"/new {q}",
                        "exec_line": f"/new {q}",
                        "action": "exec_line",
                        "enter_hint": "Enter 新建并提问",
                    },
                    {
                        "label": "仅新建空白对话 (/new)",
                        "badge": "空白会话",
                        "desc": "清空当前屏幕并分配全新会话 ID，原对话自动保存至 /resume",
                        "fill_text": "/new",
                        "exec_line": "/new",
                        "action": "exec_line",
                        "enter_hint": "Enter 新建空白对话",
                    },
                ]
            return [
                {
                    "label": "1. 立即在当前目录新建空白对话 (/new)",
                    "badge": "推荐",
                    "desc": "原对话自动保存至 /resume；也可继续输入 `/new <问题>` 直接在新对话提问",
                    "fill_text": "/new",
                    "exec_line": "/new",
                    "action": "exec_line",
                    "ghost_hint": " [可选: 输入新对话问题，或直接按 Enter 新建空白对话]",
                    "enter_hint": "Enter 立即新建",
                }
            ]

        if cmd in ("/ask", "/bug", "/opt"):
            meta_map = {
                "/ask": (
                    "只读架构分析轨道",
                    "<在此输入架构或代码问题，按 Enter 启动只读分析 ([file:line])>",
                    "强制使用 EXPLAIN 只读轨道分析代码链路并输出精确 [file:line] 证据",
                ),
                "/bug": (
                    "科学闭环修复轨道",
                    "<在此输入 Bug 现象或报错栈，按 Enter 启动 Hypothesis->Reproduce->Fix->Verify>",
                    "强制使用 BUG_FIX 科学闭环轨道定位根因、修复代码并运行回归验证",
                ),
                "/opt": (
                    "性能与架构优化轨道",
                    "<在此输入性能或架构优化目标，按 Enter 启动优化轨道>",
                    "强制使用 FEATURE/OPT 轨道执行重构、性能优化与回归验证",
                ),
            }
            badge_t, ghost_t, desc_t = meta_map[cmd]
            if q:
                return [
                    {
                        "label": f"执行 {cmd}: {q}",
                        "badge": badge_t,
                        "desc": desc_t,
                        "fill_text": f"{cmd} {q}",
                        "exec_line": f"{cmd} {q}",
                        "action": "exec_line",
                        "enter_hint": "Enter 立即执行",
                    }
                ]
            return [
                {
                    "label": f"请在输入框 `{cmd} ` 后输入内容并按 Enter 执行",
                    "badge": badge_t,
                    "desc": desc_t,
                    "fill_text": f"{cmd} ",
                    "exec_line": cmd,
                    "action": "noop_require_arg",
                    "ghost_hint": ghost_t,
                    "enter_hint": "请输入参数内容",
                }
            ]

        if cmd in ("/profile", "/sandbox"):
            default_cmd = self.profile.test_command or "pytest -q"
            target_c = q or default_cmd
            label_prefix = "cProfile 性能剖析" if cmd == "/profile" else "Git Worktree 沙箱验证"
            return [
                {
                    "label": f"{label_prefix}: {target_c}",
                    "badge": "默认命令" if not q else "自定义命令",
                    "desc": f"运行 `{cmd} {target_c}`",
                    "fill_text": f"{cmd} {target_c}",
                    "exec_line": f"{cmd} {target_c}",
                    "action": "exec_line",
                    "ghost_hint": "" if q else f"[默认: {default_cmd} · 按 Enter 直接运行]",
                    "enter_hint": "Enter 立即运行",
                }
            ]

        if cmd in ("/landmarks", "/autonomy", "/learn"):
            default_map = {
                "/landmarks": ("查看全部架构地标", "输入关键词可过滤 .fix/repo.db 中的架构地标"),
                "/autonomy": ("评估当前仓库 L1-L4 自治等级", "可附加具体任务描述评估该任务的爆炸半径与自治分"),
                "/learn": ("从 HEAD 提交学习代码风格", "可指定 Git commit SHA 或 HEAD~1 学习开发者代码风格偏好"),
            }
            lbl_d, desc_d = default_map[cmd]
            return [
                {
                    "label": f"{lbl_d}" + (f" ({q})" if q else ""),
                    "badge": cmd[1:],
                    "desc": desc_d,
                    "fill_text": f"{cmd} {q}".strip(),
                    "exec_line": f"{cmd} {q}".strip(),
                    "action": "exec_line",
                    "enter_hint": "Enter 立即执行",
                }
            ]

        return options

    def close_modal(self, toast_msg: str = "已关闭弹窗") -> None:
        """Close any active `/connect` or `/model` popup modal."""
        self._modal_state = None
        self._set_terminal_hover_mouse_tracking(False)
        self._modal_screen_row_to_item.clear()
        self._modal_doc_line_to_item.clear()
        if self._input_buffer_ref is not None:
            try:
                self._input_buffer_ref.reset()
            except Exception:
                pass
        if toast_msg:
            self.show_toast(toast_msg, duration_sec=3.0)
        if self._tui_app is not None:
            try:
                self._tui_app.invalidate()
            except Exception:
                pass

    def move_modal_selection(self, delta: int) -> bool:
        """Move the highlighted item up (`delta < 0`) or down (`delta > 0`) in the active popup modal."""
        if self._modal_state is None:
            return False
        mode = self._modal_state.get("mode")
        if mode == "connect_provider":
            items = self._modal_state.get("providers") or []
            total = len(items)
        elif mode == "connect_select_model":
            items = self._modal_state.get("models") or []
            total = len(items)
        elif mode == "switch_model":
            items = self._modal_state.get("models") or []
            total = len(items) + 1  # +1 for `+ 连接其他大模型厂商 (/connect)`
        elif mode == "resume_session":
            items = self._modal_state.get("sessions") or []
            total = len(items)
        else:
            return False
        if total <= 0:
            return False
        cur = int(self._modal_state.get("selected_idx", 0))
        self._modal_state["selected_idx"] = (cur + delta) % total
        if self._tui_app is not None:
            try:
                self._tui_app.invalidate()
            except Exception:
                pass
        return True

    def _finalize_model_activation(self, provider_id: str, model_name: str, via_connect: bool = False) -> dict[str, Any]:
        """Activate `provider_id` + `model_name`, close modal, record session block, and show toast."""
        self.llm.activate_provider_model(provider_id, model_name)
        spec = self.llm.get_provider_spec(provider_id)
        self._modal_state = None
        self._set_terminal_hover_mouse_tracking(False)
        self._modal_screen_row_to_item.clear()
        self._modal_doc_line_to_item.clear()
        self._cached_completed_key = None
        self._cached_lines_key = None
        action_verb = "已连接并切换至" if via_connect else "已切换当前模型为"
        msg_line = (
            f"[bold #10B981]✓ {action_verb} {escape(spec['vendor_label'])} — "
            f"模型: [bold #0EA5E9]{escape(model_name)}[/bold #0EA5E9] "
            f"(Endpoint: {escape(self.llm.base_url)})[/bold #10B981]"
        )
        self._record_and_render_block(lambda c, _m=msg_line: c.print(_m))
        self.show_toast(f"✓ {spec['display_name']} · {model_name}", duration_sec=3.0)
        if self._tui_app is not None:
            try:
                self._tui_app.invalidate()
            except Exception:
                pass
        return {
            "step": "completed",
            "ok": True,
            "provider_id": provider_id,
            "model": model_name,
        }

    def confirm_modal_step_sync(self, explicit_index: int | None = None) -> dict[str, Any] | None:
        """
        Synchronously handle non-network modal confirmations (`connect_provider -> connect_api_key`,
        `connect_select_model -> completed`, `switch_model -> completed`).
        Returns None if the current modal step requires async network verification (`connect_api_key`).
        """
        if self._modal_state is None:
            return {"step": "none", "ok": False}
        mode = self._modal_state.get("mode")
        if explicit_index is not None:
            self._modal_state["selected_idx"] = explicit_index
        sel_idx = int(self._modal_state.get("selected_idx", 0))

        if mode == "connect_provider":
            providers = self._modal_state.get("providers") or self.llm.list_supported_providers()
            if not providers:
                return {"step": "error", "ok": False}
            sel_idx = max(0, min(len(providers) - 1, sel_idx))
            prov = providers[sel_idx]
            pid = prov["provider_id"]
            self._modal_state["mode"] = "connect_api_key"
            self._modal_state["provider_id"] = pid
            self._modal_state["provider_info"] = prov
            self._modal_state["api_key_input"] = ""
            self._modal_state["error_msg"] = ""
            if self._input_buffer_ref is not None:
                try:
                    self._input_buffer_ref.reset()
                except Exception:
                    pass
            self.show_toast(f"请输入 {prov['display_name']} API Key 后按回车确认", duration_sec=3.0)
            if self._tui_app is not None:
                try:
                    self._tui_app.invalidate()
                except Exception:
                    pass
            return {"step": "connect_api_key", "ok": True, "provider_id": pid}

        if mode == "connect_select_model":
            models = self._modal_state.get("models") or []
            pid = str(self._modal_state.get("provider_id") or self.llm.provider_id)
            if not models:
                models = self.llm.get_provider_models(pid)
            sel_idx = max(0, min(len(models) - 1, sel_idx))
            chosen_model = str(models[sel_idx])
            return self._finalize_model_activation(pid, chosen_model, via_connect=True)

        if mode == "switch_model":
            models = self._modal_state.get("models") or []
            if sel_idx >= len(models):
                self.open_connect_modal()
                return {"step": "connect_provider", "ok": True}
            sel_idx = max(0, min(len(models) - 1, sel_idx))
            entry = models[sel_idx]
            return self._finalize_model_activation(
                str(entry["provider_id"]),
                str(entry["model"]),
                via_connect=False,
            )

        if mode == "resume_session":
            sessions = self._modal_state.get("sessions") or self.list_saved_sessions()
            if not sessions:
                self.close_modal("当前目录暂无历史对话记录")
                return {"step": "error", "ok": False}
            sel_idx = max(0, min(len(sessions) - 1, sel_idx))
            chosen = sessions[sel_idx]
            resumed = self.resume_conversation_session(str(chosen["session_id"]))
            return {
                "step": "completed",
                "ok": resumed is not None,
                "session_id": str(chosen["session_id"]),
                "turn_count": int(chosen.get("turn_count", 0)),
            }

        return None

    async def confirm_modal_step(
        self,
        explicit_index: int | None = None,
        explicit_api_key: str | None = None,
    ) -> dict[str, Any]:
        """
        Confirm the current step in `/connect` or `/model` modal (supports both sync selection steps
        and async API Key verification against the vendor endpoint).
        """
        sync_res = self.confirm_modal_step_sync(explicit_index=explicit_index)
        if sync_res is not None:
            return sync_res
        if self._modal_state is None:
            return {"step": "none", "ok": False}

        mode = self._modal_state.get("mode")
        if mode == "connect_api_key":
            pid = str(self._modal_state.get("provider_id") or "deepseek")
            buf_txt = ""
            if self._input_buffer_ref is not None:
                buf_txt = self._input_buffer_ref.text.strip()
            raw_key = (
                explicit_api_key
                if explicit_api_key is not None
                else (buf_txt or str(self._modal_state.get("api_key_input") or ""))
            ).strip()
            if not raw_key:
                err = "API Key 不能为空，请粘贴或输入您的 API Key"
                self._modal_state["error_msg"] = err
                self.show_toast("✗ API Key 不能为空", duration_sec=3.0)
                if self._tui_app is not None:
                    try:
                        self._tui_app.invalidate()
                    except Exception:
                        pass
                return {"step": "connect_api_key", "ok": False, "error": err}

            if self._input_buffer_ref is not None:
                try:
                    self._input_buffer_ref.reset()
                except Exception:
                    pass
            self._modal_state["api_key_input"] = raw_key
            self._modal_state["mode"] = "connect_verifying"
            self._modal_state["error_msg"] = ""
            spec = self.llm.get_provider_spec(pid)
            self.show_toast(f"⠋ 正在验证 {spec['display_name']} API Key...", duration_sec=5.0)
            if self._tui_app is not None:
                try:
                    self._tui_app.invalidate()
                except Exception:
                    pass

            ok, models, msg = await self.llm.verify_and_fetch_models(pid, raw_key)
            if not ok:
                if self._modal_state is not None:
                    self._modal_state["mode"] = "connect_api_key"
                    self._modal_state["error_msg"] = msg
                self.show_toast("✗ API Key 验证失败", duration_sec=3.0)
                if self._tui_app is not None:
                    try:
                        self._tui_app.invalidate()
                    except Exception:
                        pass
                return {"step": "connect_api_key", "ok": False, "error": msg}

            if self._modal_state is not None:
                self._modal_state["mode"] = "connect_select_model"
                self._modal_state["models"] = models
                self._modal_state["selected_idx"] = 0
                self._modal_state["status_msg"] = msg
                self._modal_state["error_msg"] = ""
            self.show_toast(f"✓ {spec['display_name']} 连接成功！请选择模型", duration_sec=3.0)
            if self._tui_app is not None:
                try:
                    self._tui_app.invalidate()
                except Exception:
                    pass
            return {
                "step": "connect_select_model",
                "ok": True,
                "provider_id": pid,
                "models": models,
                "message": msg,
            }

        return {"step": str(mode), "ok": False}

    def _build_modal_tagged_lines(self, max_line_cells: int) -> list[tuple[int | None, list[tuple[str, str]]]]:
        """
        Build the visual lines for the active `/connect` or `/model` popup dialog.
        Each entry is `(selectable_item_index_or_None, [(style, text), ...])`.
        """
        if self._modal_state is None:
            return []
        mode = self._modal_state.get("mode")
        border_w = min(max(48, max_line_cells - 4), 104)
        horiz = "─" * max(10, border_w - 4)
        out: list[tuple[int | None, list[tuple[str, str]]]] = []

        if mode == "connect_provider":
            providers = self._modal_state.get("providers") or self.llm.list_supported_providers()
            sel_idx = int(self._modal_state.get("selected_idx", 0))
            out.append(
                (
                    None,
                    [
                        ("bold #0D9488", "╭── "),
                        ("bold #F97316", "🔌 连接大模型厂商 (/connect)"),
                        ("bold #0D9488", " ── "),
                        ("#0EA5E9", "↑↓ 选择 · Enter/鼠标点击确认 · Esc 取消"),
                        ("bold #0D9488", " ──╮"),
                    ],
                )
            )
            for idx, prov in enumerate(providers):
                is_sel = idx == sel_idx
                bg = ""
                marker = " ❯ " if is_sel else "   "
                badge = (
                    " [● 当前使用]"
                    if prov.get("active")
                    else (" [✓ 已连接]" if prov.get("connected") else "")
                )
                badge_style = ("bold #10B981" if (prov.get("active") or prov.get("connected")) else "#788596") + bg
                hover_hint = "  ◀ 点击确认" if is_sel else ""
                out.append(
                    (
                        idx,
                        [
                            ("#0D9488", "│"),
                            (f"bold #F97316{bg}" if is_sel else "#0D9488", f"{marker}{idx + 1}. "),
                            (f"bold #0EA5E9{bg}" if is_sel else "bold", f"{prov['display_name']:<10} "),
                            (f"bold #10B981{bg}" if is_sel else "#10B981", f"{prov['vendor_label']:<24} "),
                            (f"bold #0EA5E9{bg}" if is_sel else "#788596", f"{prov['base_url']}"),
                            (badge_style, badge),
                            (f"bold #F97316{bg}", hover_hint),
                        ],
                    )
                )
            out.append(
                (
                    None,
                    [
                        ("bold #0D9488", f"╰{horiz}╯"),
                    ],
                )
            )
            return out

        if mode in ("connect_api_key", "connect_verifying"):
            pid = str(self._modal_state.get("provider_id") or "deepseek")
            spec = self.llm.get_provider_spec(pid)
            buf_txt = self._input_buffer_ref.text if self._input_buffer_ref is not None else ""
            raw_key = buf_txt or str(self._modal_state.get("api_key_input") or "")
            if len(raw_key) > 8:
                masked_key = raw_key[:4] + ("*" * min(24, len(raw_key) - 6)) + raw_key[-2:]
            elif raw_key:
                masked_key = raw_key[:2] + ("*" * max(1, len(raw_key) - 2))
            else:
                masked_key = ""
            out.append(
                (
                    None,
                    [
                        ("bold #0D9488", "╭── "),
                        ("bold #F97316", f"🔑 输入 {spec['vendor_label']} API Key"),
                        ("bold #0D9488", " ── "),
                        ("#0EA5E9", "Enter 验证连接 · Esc 取消"),
                        ("bold #0D9488", " ──╮"),
                    ],
                )
            )
            out.append(
                (
                    None,
                    [
                        ("#0D9488", "│  "),
                        ("bold", "官方接口: "),
                        ("#0EA5E9", f"{spec['base_url']}  "),
                        ("#788596", f"({spec['api_notes']})"),
                    ],
                )
            )
            out.append(
                (
                    0,
                    [
                        ("#0D9488", "│  "),
                        ("bold #F97316", "API Key 输入框: "),
                        (
                            "bold #10B981" if masked_key else "#788596",
                            f"[ {masked_key}█ ]" if masked_key else "[ 请直接输入或粘贴 API Key 后按 Enter 确认 █ ]",
                        ),
                    ],
                )
            )
            if mode == "connect_verifying":
                out.append(
                    (
                        None,
                        [
                            ("#0D9488", "│  "),
                            ("bold #0EA5E9", f"⠋ 正在连接 {spec['display_name']} 官方接口并拉取支持的模型列表..."),
                        ],
                    )
                )
            err = str(self._modal_state.get("error_msg") or "")
            if err:
                out.append(
                    (
                        None,
                        [
                            ("#0D9488", "│  "),
                            ("bold #F97316", f"✗ {err}"),
                        ],
                    )
                )
            out.append((None, [("bold #0D9488", f"╰{horiz}╯")]))
            return out

        if mode == "connect_select_model":
            pid = str(self._modal_state.get("provider_id") or self.llm.provider_id)
            spec = self.llm.get_provider_spec(pid)
            models = self._modal_state.get("models") or self.llm.get_provider_models(pid)
            sel_idx = int(self._modal_state.get("selected_idx", 0))
            status_msg = str(self._modal_state.get("status_msg") or f"✓ {spec['vendor_label']} 连接成功")
            out.append(
                (
                    None,
                    [
                        ("bold #0D9488", "╭── "),
                        ("bold #10B981", f"{status_msg}"),
                        ("bold #0D9488", " ── "),
                        ("#0EA5E9", "请选择模型 (↑↓ · Enter/鼠标点击确认)"),
                        ("bold #0D9488", " ──╮"),
                    ],
                )
            )
            for idx, m_name in enumerate(models):
                is_sel = idx == sel_idx
                bg = ""
                marker = " ❯ " if is_sel else "   "
                tag = "  [推荐]" if idx == 0 else ""
                hover_hint = "  ◀ 点击启用" if is_sel else ""
                out.append(
                    (
                        idx,
                        [
                            ("#0D9488", "│"),
                            (f"bold #F97316{bg}" if is_sel else "#0D9488", f"{marker}{idx + 1}. "),
                            (f"bold #0EA5E9{bg}" if is_sel else "bold", f"{m_name:<28}"),
                            (f"bold #10B981{bg}" if is_sel else "#10B981", tag),
                            (f"bold #F97316{bg}", hover_hint),
                        ],
                    )
                )
            out.append((None, [("bold #0D9488", f"╰{horiz}╯")]))
            return out

        if mode == "switch_model":
            models = self._modal_state.get("models") or self.llm.list_switchable_models()
            sel_idx = int(self._modal_state.get("selected_idx", 0))
            out.append(
                (
                    None,
                    [
                        ("bold #0D9488", "╭── "),
                        ("bold #F97316", "🤖 当前支持切换的大模型列表 (/model)"),
                        ("bold #0D9488", " ── "),
                        ("#0EA5E9", "↑↓ 选择 · Enter/鼠标点击切换 · Esc 关闭"),
                        ("bold #0D9488", " ──╮"),
                    ],
                )
            )
            for idx, entry in enumerate(models):
                is_sel = idx == sel_idx
                bg = ""
                marker = " ❯ " if is_sel else "   "
                cur_badge = "  ● 当前使用" if entry.get("is_current") else ""
                hover_hint = "  ◀ 点击切换" if is_sel else ""
                out.append(
                    (
                        idx,
                        [
                            ("#0D9488", "│"),
                            (f"bold #F97316{bg}" if is_sel else "#0D9488", f"{marker}{idx + 1}. "),
                            (f"bold #10B981{bg}" if is_sel else "#10B981", f"[{entry['provider_name']}] "),
                            (f"bold #0EA5E9{bg}" if is_sel else "bold", f"{entry['model']:<28}"),
                            (f"bold #F97316{bg}", cur_badge),
                            (f"bold #F97316{bg}", hover_hint),
                        ],
                    )
                )
            connect_idx = len(models)
            is_conn_sel = sel_idx == connect_idx
            conn_bg = ""
            conn_marker = " ❯ " if is_conn_sel else "   "
            conn_hint = "  ◀ 点击打开" if is_conn_sel else ""
            out.append(
                (
                    connect_idx,
                    [
                        ("#0D9488", "│"),
                        (f"bold #F97316{conn_bg}" if is_conn_sel else "#0D9488", f"{conn_marker}+ "),
                        (
                            f"bold #0EA5E9{conn_bg}" if is_conn_sel else "#0EA5E9",
                            "连接其他大模型厂商 (/connect: DeepSeek / GLM / MiMo / MiniMax / Qwen / OpenAI / Anthropic)",
                        ),
                        (f"bold #F97316{conn_bg}", conn_hint),
                    ],
                )
            )
            out.append((None, [("bold #0D9488", f"╰{horiz}╯")]))
            return out

        if mode == "resume_session":
            sessions = self._modal_state.get("sessions") or self.list_saved_sessions()
            sel_idx = int(self._modal_state.get("selected_idx", 0))
            dir_label = self.session_dir.name or str(self.session_dir)
            out.append(
                (
                    None,
                    [
                        ("bold #0D9488", "╭── "),
                        ("bold #F97316", f"💬 恢复历史对话 (/resume · {dir_label})"),
                        ("bold #0D9488", " ── "),
                        ("#0EA5E9", "↑↓/鼠标悬停选择 · Enter/点击恢复 · Esc 取消"),
                        ("bold #0D9488", " ──╮"),
                    ],
                )
            )
            preview_budget = max(22, min(56, border_w - 42))
            for idx, sess_item in enumerate(sessions[:15]):
                is_sel = idx == sel_idx
                marker = " ❯ " if is_sel else "   "
                t_disp = str(sess_item.get("updated_at_display", ""))
                n_turns = int(sess_item.get("turn_count", 1))
                latest_q = _truncate_middle_for_width(
                    str(sess_item.get("latest_prompt") or sess_item.get("title") or "(空对话)"),
                    preview_budget,
                )
                badge = (
                    "  [● 当前会话]"
                    if sess_item.get("is_current")
                    else ("  [最新]" if idx == 0 else "")
                )
                badge_style = (
                    "bold #10B981"
                    if (sess_item.get("is_current") or idx == 0)
                    else "#788596"
                )
                hover_hint = "  ◀ 点击恢复" if is_sel else ""
                out.append(
                    (
                        idx,
                        [
                            ("#0D9488", "│"),
                            ("bold #F97316" if is_sel else "#0D9488", f"{marker}{idx + 1}. "),
                            ("bold #0EA5E9" if is_sel else "#0EA5E9", f"[{t_disp}] "),
                            ("bold #10B981" if is_sel else "#10B981", f"({n_turns}轮) "),
                            ("bold #F97316" if is_sel else "", f"{latest_q}"),
                            (badge_style, badge),
                            ("bold #F97316", hover_hint),
                        ],
                    )
                )
            out.append((None, [("bold #0D9488", f"╰{horiz}╯")]))
            return out

        return out

    def get_selected_main_text(self) -> str:
        """
        Extract the plain text currently selected in `main_content_window` between `_sel_anchor` and `_sel_head`
        across all spanned document lines (even across multiple scrolled screens!), completely free of right-edge
        scrollbar characters (`┃`/`│`) and trailing padding spaces.
        """
        from rich.cells import cell_len

        if (
            self._sel_anchor is None
            or self._sel_head is None
            or self._sel_anchor == self._sel_head
            or not self._cached_render_lines
        ):
            return ""
        (line_a, col_a), (line_b, col_b) = sorted([self._sel_anchor, self._sel_head])
        total_l = len(self._cached_render_lines)
        line_a = max(0, min(total_l - 1, line_a))
        line_b = max(0, min(total_l - 1, line_b))

        extracted_lines: list[str] = []
        for doc_idx in range(line_a, line_b + 1):
            ln_frags = self._cached_render_lines[doc_idx]
            if line_a == line_b:
                s_col, e_col = min(col_a, col_b), max(col_a, col_b)
            elif doc_idx == line_a:
                s_col, e_col = col_a, 999999
            elif doc_idx == line_b:
                s_col, e_col = 0, col_b
            else:
                s_col, e_col = 0, 999999

            cur_col = 0
            chars: list[str] = []
            for _, seg in ln_frags:
                for ch in seg:
                    cw = max(1, cell_len(ch))
                    if cur_col + cw > s_col and cur_col < e_col:
                        chars.append(ch)
                    cur_col += cw
            extracted_lines.append("".join(chars).rstrip())
        return "\n".join(extracted_lines)

    def copy_last_response_to_clipboard(self) -> bool:
        """Copy active selection (if any) or the latest Agent response to system clipboard and trigger a 3-second toast."""
        sel_txt = self.get_selected_main_text()
        if sel_txt:
            return self.copy_text_to_clipboard(sel_txt, toast_label="✓ 复制成功")
        text_to_copy = ""
        if self.last_session and self.last_session.final_summary:
            text_to_copy = self.last_session.final_summary
        elif self.conversation_context:
            text_to_copy = self.conversation_context[-1].get("assistant", "")
        return self.copy_text_to_clipboard(text_to_copy, toast_label="✓ 复制成功")

    def handle_ctrl_c_action(self, current_input_text: str) -> str:
        """
        Execute the 3-state Ctrl+C state machine:
        1. If Agent is currently executing a task (`_running_task` active or `_active_tracker` set):
           Cancel the running task ONLY -> returns `"cancel_task"`.
        2. If Agent is idle and `current_input_text` is non-empty:
           Clear the input box ONLY -> returns `"clear_input"`.
        3. If Agent is idle AND `current_input_text` is empty:
           Exit the Agent UI -> returns `"exit_agent"`.
        """
        task_in_flight = (
            (self._running_task is not None and not self._running_task.done())
            or self._active_tracker is not None
        )
        if task_in_flight:
            if self._running_task is not None and not self._running_task.done():
                self._running_task.cancel()
            if self._active_tracker is not None:
                self._task_elapsed_sec = round(max(0.0, time.time() - self._active_tracker.started_at), 1)
            self._agent_state_label = "已取消"
            self.show_toast("已取消当前任务", duration_sec=3.0)
            return "cancel_task"

        if self._modal_state is not None:
            self.close_modal("已取消弹窗操作")
            return "close_modal"

        if current_input_text:
            self.show_toast("已清空输入", duration_sec=3.0)
            return "clear_input"

        return "exit_agent" 

    def _clear_real_terminal_viewport(self) -> None:
        if getattr(self, "_tui_mode", False):
            if getattr(self, "_tui_app", None) is not None:
                try:
                    self._tui_app.invalidate()
                except Exception:
                    pass
            return
        raw_file = getattr(self.console, "_file", None)
        is_std = raw_file is None or raw_file in (
            sys.stdout,
            sys.stderr,
            getattr(sys, "__stdout__", None),
            getattr(sys, "__stderr__", None),
        ) or hasattr(raw_file, "_live")
        if is_std and self.console.is_terminal:
            raw_out = getattr(sys, "__stdout__", sys.stdout)
            try:
                raw_out.write("\033[2J\033[3J\033[H")
                raw_out.flush()
            except Exception:
                pass

    def _redraw_above_live_on_resize(self, new_cols: int) -> None:
        """
        Called when the terminal window is resized WHILE a task is executing inside `Live`.
        Clears the screen and re-renders the banner, completed turns, and the active user prompt
        at `new_cols` right before `Live` paints the updated progress block below it!
        """
        self._last_cols = new_cols
        if getattr(self, "_tui_mode", False):
            if getattr(self, "_tui_app", None) is not None:
                try:
                    self._tui_app.invalidate()
                except Exception:
                    pass
            return
        self._clear_real_terminal_viewport()
        for block_fn in self._screen_blocks:
            block_fn(self.console)
        if self._inflight_prompt:
            repo_name = Path(self.profile.repo_root).name
            self.console.print(
                f"[bold #0D9488]fix[/bold #0D9488] [bold #10B981]({escape(repo_name)})[/bold #10B981] "
                f"[bold #F97316]❯[/bold #F97316] {escape(self._inflight_prompt)}"
            )
        if self._active_tracker is not None:
            self._active_tracker.render_all_completed_blocks_to(self.console)

    def rerender_for_terminal_width(self, new_width: int | None = None, clear_terminal: bool = True) -> None:
        """
        Re-render the current screen history (`_screen_blocks` + active `CollapsibleProgressTracker` if running)
        at `new_width` when the user drags the terminal window larger or smaller (`SIGWINCH`).
        """
        cols = _sync_live_terminal_width(self.console, explicit_width=new_width)
        self._last_cols = cols

        if self._active_tracker is not None:
            if clear_terminal:
                self._clear_real_terminal_viewport()
            for block_fn in self._screen_blocks:
                block_fn(self.console)
            if self._inflight_prompt:
                repo_name = Path(self.profile.repo_root).name
                self.console.print(
                    f"[bold #0D9488]fix[/bold #0D9488] [bold #10B981]({escape(repo_name)})[/bold #10B981] "
                    f"[bold #F97316]❯[/bold #F97316] {escape(self._inflight_prompt)}"
                )
            self._active_tracker._last_cols = cols
            if self._active_tracker._live is not None:
                self._active_tracker._live._live_render._shape = None
                self._active_tracker.refresh()
            else:
                # For non-TTY / StringIO buffer tests, print the resized live progress block directly
                self.console.print(self._active_tracker._build_renderable(cols))
            return

        if clear_terminal:
            self._clear_real_terminal_viewport()

        for block_fn in self._screen_blocks:
            block_fn(self.console)

    def _record_and_render_block(self, block_fn: Callable[[Console], None]) -> None:
        self._screen_blocks.append(block_fn)
        if len(self._screen_blocks) > 13:
            self._screen_blocks = [self._screen_blocks[0], *self._screen_blocks[-12:]]
        block_fn(self.console)

    async def handle_line(
        self,
        raw_line: str,
        input_fn: Callable[[str], str] | None = None,
    ) -> bool:
        line = raw_line.strip()
        if not line:
            return True

        self.record_input_history(line)

        if line.startswith("/"):
            before_blocks = len(self._screen_blocks)
            recorded_prints: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
            orig_print = self.console.print

            def _capturing_print(*args: Any, **kwargs: Any) -> None:
                recorded_prints.append((args, dict(kwargs)))
                orig_print(*args, **kwargs)

            self.console.print = _capturing_print  # type: ignore[method-assign]
            try:
                keep = await self._handle_slash_command(line, input_fn=input_fn)
            finally:
                self.console.print = orig_print  # type: ignore[method-assign]

            if self._tui_mode and self._modal_state is None and len(self._screen_blocks) == before_blocks and recorded_prints:
                if line.lower() not in ("/new", "/clear", "/exit", "/quit", "/q") and not line.lower().startswith("/new ") and not (
                    line.lower().startswith("/resume") and (self._modal_state is not None or self._saved_turns_records)
                ):
                    cmd_str = line
                    saved_calls = list(recorded_prints)

                    def _replay_slash_turn(
                        c: Console,
                        _cmd: str = cmd_str,
                        _calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = saved_calls,
                    ) -> None:
                        _sync_live_terminal_width(c)
                        render_codex_user_prompt_to(c, _cmd)
                        for p_args, p_kwargs in _calls:
                            c.print(*p_args, **p_kwargs)

                    self._screen_blocks.append(_replay_slash_turn)
                    if len(self._screen_blocks) > 13:
                        self._screen_blocks = [self._screen_blocks[0], *self._screen_blocks[-12:]]
            return keep

        await self._execute_interactive_prompt(line, forced_intent=None, input_fn=input_fn)
        return True

    async def _handle_slash_command(
        self,
        line: str,
        input_fn: Callable[[str], str] | None = None,
    ) -> bool:
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/exit", "/quit", "/q"):
            self.console.print("[dim]Goodbye! Mind Mirror & project landmarks are saved in `.fix/`.[/dim]")
            return False

        if cmd == "/help":
            def _render_help(c: Console) -> None:
                _sync_live_terminal_width(c)
                help_table = Table(
                    title="Fix Interactive Commands",
                    header_style="bold cyan",
                    box=box.SIMPLE_HEAVY,
                    expand=False,
                )
                help_table.add_column("Command", style="bold green", no_wrap=True)
                help_table.add_column("Description", overflow="fold")
                help_table.add_row("<natural language>", "Directly ask questions, fix bugs, refactor, or add features in current repo")
                help_table.add_row("/steps", "Expand and view all collapsed tool calls (Read, Search, etc.) from the last task")
                help_table.add_row("/ask <question>", "Force read-only architectural diagnosis ([file:line] anchored)")
                help_table.add_row("/bug <symptom>", "Force scientific BUG_FIX loop (Hypothesis -> Reproduce -> Fix -> Verify -> Review)")
                help_table.add_row("/opt <goal>", "Force performance / architectural optimization track")
                help_table.add_row("/trace <symbol>", "LSP/AST cross-file call graph (definitions, callers & callees)")
                help_table.add_row("/diag [rel_path]", "Run static syntax, import resolution & type diagnostics")
                help_table.add_row("/profile <cmd>", "Run runtime cProfile hotspot & line-coverage probe")
                help_table.add_row("/idle [run]", "Show Looper + IdleHandler P2 queue or run background evolution ticks")
                help_table.add_row("/status", "Show current repository profile, active models, landmarks & skills")
                help_table.add_row("/rules [approve|reject <id>]", "Inspect or approve/reject Developer Mind Mirror rules")
                help_table.add_row("/landmarks", "List verified repository landmarks stored in .fix/repo.db")
                help_table.add_row("/skills [bootstrap|run <slug>]", "List, bootstrap, or execute diagnostic skills in .fix/skills/")
                help_table.add_row("/autonomy", "Inspect L1-L4 Progressive Autonomy score & breakdown")
                help_table.add_row("/sandbox [cmd]", "Run empirical baseline vs isolated Git worktree verification")
                help_table.add_row("/learn", "Learn developer style habits from the latest Git commit diff")
                help_table.add_row("/diff", "Show current working tree git diff")
                help_table.add_row("/undo", "Roll back working tree to the snapshot before the last edit")
                help_table.add_row("/new [prompt]", "Start a brand-new conversation in current directory (previous session saved for /resume)")
                help_table.add_row("/resume [index|id]", "Restore historical conversation for current directory and stay at latest turn")
                help_table.add_row("/model [fast] [reasoning]", "Show switchable models list or switch active model")
                help_table.add_row("/connect [vendor] [key]", "Connect LLM vendors (DeepSeek, GLM, MiMo, MiniMax, Qwen, OpenAI, Anthropic)")
                help_table.add_row("/clear", "Clear multi-turn conversation history in current session")
                help_table.add_row("/exit", "Exit Fix interactive CLI (or press Ctrl+D)")
                c.print(help_table)

            self._record_and_render_block(_render_help)
            return True

        if cmd == "/steps":
            if self._tui_mode and self._turn_fold_states:
                target_turn: int | None = None
                if arg.strip().lstrip("#").isdigit():
                    cand = int(arg.strip().lstrip("#"))
                    if cand in self._turn_fold_states:
                        target_turn = cand
                new_state = self.toggle_turn_steps_collapsed(target_turn)
                if new_state is not None:
                    return True
            if self.last_session is None or not self.last_session.tool_calls_log:
                self.console.print("[dim]暂无上一轮任务的工具调用记录。[/dim]")
                return True
            target_sess = self.last_session

            def _render_steps(c: Console) -> None:
                _sync_live_terminal_width(c)
                st_table = Table(
                    title=f"上一次任务工具调用明细 (共 {target_sess.tool_calls_count} 次 · 耗时 {target_sess.elapsed_sec:.2f}s)",
                    header_style="bold cyan",
                    box=box.SIMPLE_HEAVY,
                )
                st_table.add_column("#", justify="right", style="dim", no_wrap=True)
                st_table.add_column("Tool Call", style="white", overflow="fold")
                for idx, entry in enumerate(target_sess.tool_calls_log, start=1):
                    st_table.add_row(str(idx), escape(entry))
                c.print(st_table)

            self._record_and_render_block(_render_steps)
            return True

        if cmd == "/status":
            _do_status(repo=str(self.toolkit.repo_root), out_console=self.console)
            return True

        if cmd == "/new":
            self.start_new_conversation_session()
            if arg:
                self.record_input_history(arg)
                await self._execute_interactive_prompt(arg, forced_intent=None, input_fn=input_fn)
            return True

        if cmd == "/clear":
            self.conversation_context.clear()
            self._turn_fold_states.clear()
            self._turn_counter = 0
            self._saved_turns_records.clear()
            self.session_id = self._new_conversation_session_id()
            self.session_created_at = time.time()
            self._scroll_offset_from_bottom = 0
            self._cached_completed_key = None
            self._cached_lines_key = None
            self._screen_blocks = [self._render_minimal_header_to if self._tui_mode else self._render_banner_to]
            self.show_toast("✓ 已清空对话", duration_sec=3.0)
            self.console.print("[bold #10B981]✓ Cleared session conversation context.[/bold #10B981]")
            return True

        if cmd == "/resume":
            sessions = self.list_saved_sessions()
            if not sessions:
                self.show_toast("当前目录暂无历史对话记录", duration_sec=3.0)
                self.console.print("[dim]当前目录暂无可恢复的历史对话记录。[/dim]")
                return True

            if arg:
                resumed = self.resume_conversation_session(arg)
                if resumed is None:
                    self.show_toast(f"✗ 未找到匹配的历史对话: {arg}", duration_sec=3.0)
                    self.console.print(
                        f"[red]✗ 未找到匹配的历史对话 `{escape(arg)}`。直接输入 `/resume` 查看可选历史对话。[/red]"
                    )
                return True

            if self._tui_mode and input_fn is None:
                self.open_resume_modal()
                return True

            r_table = Table(
                title=f"当前目录历史对话记录 (/resume · {self.session_dir})",
                header_style="bold #0D9488",
                box=box.SIMPLE_HEAVY,
                expand=False,
            )
            r_table.add_column("#", style="dim", justify="right", no_wrap=True)
            r_table.add_column("更新时间", style="bold #0EA5E9", no_wrap=True)
            r_table.add_column("轮数", style="bold #10B981", justify="right", no_wrap=True)
            r_table.add_column("最新对话内容预览", overflow="fold")
            for idx_s, s_item in enumerate(sessions[:15], start=1):
                badge_s = " [当前会话]" if s_item.get("is_current") else (" [最新]" if idx_s == 1 else "")
                r_table.add_row(
                    str(idx_s),
                    escape(str(s_item.get("updated_at_display", ""))),
                    f"{s_item.get('turn_count', 1)} 轮",
                    escape(str(s_item.get("latest_prompt") or s_item.get("title") or "")) + badge_s,
                )
            self.console.print(r_table)

            if input_fn is not None:
                sel_raw = input_fn(
                    f"请选择要恢复的历史对话序号 (1-{len(sessions)}，直接回车默认恢复最新 #1，输入 q 取消): "
                ).strip()
                if sel_raw.lower() in ("q", "quit", "exit", "cancel"):
                    self.console.print("[dim]已取消恢复历史对话。[/dim]")
                    return True
                self.resume_conversation_session(sel_raw or "1")
                return True

            # Non-TUI automatic fallback: restore the latest conversation session immediately
            self.resume_conversation_session(0)
            return True

        if cmd == "/diff":
            diff_text = self.toolkit.get_git_diff()
            if diff_text.strip():
                self.console.print(Rule("[bold yellow]Current Working Tree Diff[/bold yellow]", style="yellow"))
                self.console.print(escape(diff_text[:6000]))
                self.console.print(Rule(style="dim yellow"))
            else:
                self.console.print("[dim]Working tree is clean (no uncommitted changes).[/dim]")
            return True

        if cmd == "/undo":
            if not self.snapshot_stack:
                self.console.print("[yellow]No previous task snapshot recorded in this session to undo.[/yellow]")
                return True
            snap_id = self.snapshot_stack.pop()
            ok = self.toolkit.rollback_to_snapshot(snap_id)
            if ok:
                self.show_toast("✓ 已回滚修改", duration_sec=3.0)
                self.console.print(f"[bold #10B981]✓ Rolled back working tree to snapshot {snap_id[:12]}.[/bold #10B981]")
            else:
                self.console.print(f"[red]Failed to roll back snapshot {snap_id}.[/red]")
            return True

        if cmd == "/landmarks":
            landmarks = self.memory.search_landmarks(query=arg, limit=20, include_stale=True)
            if not landmarks:
                self.console.print("[dim]No repository landmarks recorded yet.[/dim]")
                return True
            lm_table = Table(title="Repository Landmarks (.fix/repo.db)", header_style="bold magenta", box=box.SIMPLE_HEAVY)
            lm_table.add_column("ID", no_wrap=True)
            lm_table.add_column("Category", no_wrap=True)
            lm_table.add_column("Status", no_wrap=True)
            lm_table.add_column("Summary", overflow="fold")
            lm_table.add_column("Files", overflow="fold")
            for f in landmarks:
                badge = "[red]STALE[/red]" if f.stale else f"[green]{f.status.value}[/green]"
                lm_table.add_row(f.fact_id, f.category, badge, escape(f.summary[:80]), ", ".join(f.evidence_paths))
            self.console.print(lm_table)
            return True

        if cmd == "/skills":
            sub_parts = arg.split()
            if sub_parts and sub_parts[0] == "bootstrap":
                topic = sub_parts[1] if len(sub_parts) > 1 else "auto"
                card = self.evolution.bootstrap_project_skill(topic=topic)
                self.console.print(
                    f"[bold green]✓ Bootstrapped executable skill `{card.slug}`[/bold green] "
                    f"(`.fix/skills/{card.slug}/run.py`) — {card.title}"
                )
                return True
            if len(sub_parts) >= 2 and sub_parts[0] == "run":
                slug = sub_parts[1]
                target_arg = sub_parts[2] if len(sub_parts) > 2 else ""
                res = await self.toolkit.run_skill(slug=slug, target=target_arg)
                self.console.print(Rule(f"[bold green]Skill Execution: {escape(slug)}[/bold green]", style="green"))
                self.console.print(escape(res.output))
                self.console.print(Rule(style="dim green"))
                return True

            skills = self.memory.list_skills()
            if skills:
                self.console.print(
                    f"[bold green]Bootstrapped Executable Skills ({len(skills)}):[/bold green] "
                    + ", ".join(skills)
                    + "  [dim](Use `/skills run <slug>` or `/skills bootstrap`)[/dim]"
                )
            else:
                self.console.print(
                    "[dim]No bootstrapped skills in .fix/skills/ yet. Run `/skills bootstrap` to generate one![/dim]"
                )
            return True

        if cmd == "/autonomy":
            assess = AutonomyRouter(self.memory).assess_autonomy(user_prompt=arg)
            self.console.print(Rule("[bold bright_green]L1–L4 Progressive Autonomy Assessment[/bold bright_green]", style="green"))
            self.console.print(
                f"  [bold]Current Level:[/bold]       [bold green]{assess.level.value}[/bold green] (Score: {assess.score:.3f})\n"
                f"  [bold]Landmark Coverage:[/bold]   {assess.landmark_coverage:.1%}\n"
                f"  [bold]Task Win Rate:[/bold]       {assess.history_success_rate:.1%}\n"
                f"  [bold]Persona Alignment:[/bold]   {assess.persona_alignment:.1%}\n"
                f"  [bold]Blast Radius Risk:[/bold]   {assess.blast_radius_risk:.1%}\n"
                f"  [bold]Auto-Clarify Mode:[/bold]   {'Enabled (L3/L4)' if assess.auto_approve_clarifications else 'Interactive (L1/L2)'}\n"
                f"  [dim]{escape(assess.rationale)}[/dim]"
            )
            self.console.print(Rule(style="dim green"))
            return True

        if cmd == "/sandbox":
            target_cmd = arg or (self.profile.test_command or "pytest -q")
            verifier = SandboxVerifier(self.toolkit)
            rep = await verifier.verify_in_worktree(target_cmd)
            self.console.print(Rule("[bold bright_magenta]Git Worktree Sandbox Verification[/bold bright_magenta]", style="magenta"))
            self.console.print(f"  {escape(rep.summary)}")
            self.console.print(Rule(style="dim magenta"))
            return True

        if cmd == "/rules":
            sub_parts = arg.split()
            if len(sub_parts) == 2 and sub_parts[0] == "approve":
                _do_rules(repo=str(self.toolkit.repo_root), approve=sub_parts[1], reject=None, add_pattern=None, preferred=None, out_console=self.console)
            elif len(sub_parts) == 2 and sub_parts[0] == "reject":
                _do_rules(repo=str(self.toolkit.repo_root), approve=None, reject=sub_parts[1], add_pattern=None, preferred=None, out_console=self.console)
            else:
                _do_rules(repo=str(self.toolkit.repo_root), approve=None, reject=None, add_pattern=None, preferred=None, out_console=self.console)
            return True

        if cmd == "/learn":
            commit_ref = arg or "HEAD"
            msg_proc = subprocess.run(
                ["git", "log", "-1", "--pretty=%s", commit_ref],
                cwd=self.toolkit.repo_root,
                capture_output=True,
                text=True,
            )
            diff_proc = subprocess.run(
                ["git", "show", "--format=", commit_ref],
                cwd=self.toolkit.repo_root,
                capture_output=True,
                text=True,
            )
            c_msg = msg_proc.stdout.strip() if msg_proc.returncode == 0 else ""
            h_diff = diff_proc.stdout if diff_proc.returncode == 0 else ""
            learned = await self.evolution.learn_from_human_diff(human_diff=h_diff, commit_message=c_msg)
            if learned:
                self.console.print(f"[green]Learned/updated {len(learned)} Developer Mind Mirror rule(s) from {commit_ref}:[/green]")
                for r in learned:
                    self.console.print(f"  - [{r.rule_id}] ({r.status.value}, hits={r.hit_count}) {r.pattern_desc} -> {r.preferred_way}")
            else:
                self.console.print(
                    "[dim]Semantic Equivalence Filter: Commit classified as business feature/logic change; "
                    "no general style rules extracted.[/dim]"
                )
            return True

        if cmd == "/init":
            _do_init(repo=str(self.toolkit.repo_root), install_hook=True, out_console=self.console)
            return True

        if cmd == "/model":
            if arg:
                m_parts = arg.split()
                prov_name, new_fast, new_reas = self.llm.switch_model(
                    m_parts[0],
                    m_parts[1] if len(m_parts) > 1 else None,
                )
                self._cached_completed_key = None
                self._cached_lines_key = None
                self.show_toast(f"✓ 已切换模型: {new_fast}", duration_sec=3.0)
                self.console.print(
                    f"[green]✓ Switched active models to fast={new_fast}, "
                    f"reasoning={new_reas} ({prov_name} · {self.llm.base_url})[/green]"
                )
            else:
                active_spec = self.llm.get_provider_spec()
                switchable = self.llm.list_switchable_models()
                providers_info = self.llm.list_supported_providers()
                self.console.print(
                    f"[bold]Provider:[/bold] {escape(active_spec['vendor_label'])}\n"
                    f"[bold]Base URL:[/bold] {escape(self.llm.base_url)}\n"
                    f"[bold]Fast Model:[/bold] {escape(self.llm.fast_model)}\n"
                    f"[bold]Reasoning Model:[/bold] {escape(self.llm.reasoning_model)}"
                )
                m_table = Table(
                    title="当前支持切换的大模型列表 (输入 `/model <模型名>` 切换 · 输入 `/connect` 连接新厂商)",
                    header_style="bold #0D9488",
                    box=box.SIMPLE_HEAVY,
                    expand=False,
                )
                m_table.add_column("#", style="dim", justify="right", no_wrap=True)
                m_table.add_column("厂商 (Provider)", style="bold #10B981", no_wrap=True)
                m_table.add_column("模型名称 (Model)", style="bold #0EA5E9", overflow="fold")
                m_table.add_column("状态 (Status)", overflow="fold")
                for idx_m, item in enumerate(switchable, start=1):
                    roles: list[str] = []
                    if item.get("is_current_fast"):
                        roles.append("Fast")
                    if item.get("is_current_reasoning"):
                        roles.append("Reasoning")
                    status_str = (
                        f"[bold #F97316]● 当前使用 ({'/'.join(roles)})[/bold #F97316]"
                        if roles
                        else "[#10B981]✓ 可直接切换[/#10B981]"
                    )
                    m_table.add_row(
                        str(idx_m),
                        escape(item["vendor_label"]),
                        escape(item["model"]),
                        status_str,
                    )
                self.console.print(m_table)
                vendor_summary = " · ".join(
                    f"{p['display_name']}{'(已连接)' if p['connected'] else ''}" for p in providers_info
                )
                self.console.print(
                    f"[dim]支持接入的 7 大模型厂商: {escape(vendor_summary)} — 使用 `/connect` 连接更多厂商 API。[/dim]"
                )
                if self._tui_mode and input_fn is None:
                    self.open_model_selector_modal()
            return True

        if cmd == "/connect":
            sub_args = arg.split()
            if len(sub_args) >= 2:
                raw_prov, raw_key = sub_args[0], sub_args[1]
                explicit_model = sub_args[2] if len(sub_args) >= 3 else None
                pid = resolve_provider_id(raw_prov)
                if not pid:
                    self.console.print(
                        f"[red]✗ 不支持的厂商 `{escape(raw_prov)}`。支持: DeepSeek, GLM, MiMo, MiniMax, Qwen, OpenAI, Anthropic[/red]"
                    )
                    return True
                spec = self.llm.get_provider_spec(pid)
                ok, models, msg = await self.llm.verify_and_fetch_models(pid, raw_key)
                if not ok:
                    self.show_toast("✗ API Key 验证失败", duration_sec=3.0)
                    self.console.print(f"[red]✗ {escape(msg)}[/red]")
                    return True
                self.console.print(f"[bold #10B981]{escape(msg)}[/bold #10B981]")
                self.console.print(
                    f"[bold]当前 API 支持的模型列表 ({len(models)}):[/bold] "
                    + ", ".join(f"[bold #0EA5E9]{escape(m)}[/bold #0EA5E9]" for m in models)
                )
                if explicit_model:
                    self._finalize_model_activation(pid, explicit_model, via_connect=True)
                    return True
                if input_fn is not None:
                    choice = input_fn(f"请选择要使用的模型 (1-{len(models)} 或输入模型名，回车默认 {models[0]}): ").strip()
                    chosen = models[0]
                    if choice.isdigit() and 1 <= int(choice) <= len(models):
                        chosen = models[int(choice) - 1]
                    elif choice:
                        chosen = choice
                    self._finalize_model_activation(pid, chosen, via_connect=True)
                    return True
                if self._tui_mode:
                    self._modal_state = {
                        "mode": "connect_select_model",
                        "provider_id": pid,
                        "models": models,
                        "selected_idx": 0,
                        "status_msg": msg,
                        "error_msg": "",
                    }
                    self.show_toast(f"✓ {spec['display_name']} 连接成功！请选择模型", duration_sec=3.0)
                    if self._tui_app is not None:
                        self._tui_app.invalidate()
                    return True
                self._finalize_model_activation(pid, models[0], via_connect=True)
                return True

            providers_info = self.llm.list_supported_providers()
            p_table = Table(
                title="Fix 支持连接的大模型厂商 (/connect)",
                header_style="bold #0D9488",
                box=box.SIMPLE_HEAVY,
                expand=False,
            )
            p_table.add_column("#", style="dim", justify="right", no_wrap=True)
            p_table.add_column("厂商", style="bold #F97316", no_wrap=True)
            p_table.add_column("官方 API Base URL & 调用方式", overflow="fold")
            p_table.add_column("代表模型", style="#0EA5E9", overflow="fold")
            p_table.add_column("状态", no_wrap=True)
            for p in providers_info:
                st_badge = (
                    "[bold #F97316]● 当前使用[/bold #F97316]"
                    if p["active"]
                    else ("[bold #10B981]✓ 已连接[/bold #10B981]" if p["connected"] else "[dim]未连接[/dim]")
                )
                p_table.add_row(
                    str(p["index"]),
                    escape(p["vendor_label"]),
                    f"{escape(p['base_url'])}\n[dim]{escape(p['api_notes'])}[/dim]",
                    escape(", ".join(p["models"][:4])),
                    st_badge,
                )
            self.console.print(p_table)

            if self._tui_mode and input_fn is None:
                self.open_connect_modal()
                if len(sub_args) == 1:
                    pid = resolve_provider_id(sub_args[0])
                    if pid and self._modal_state is not None:
                        idx_match = next(
                            (i for i, p in enumerate(providers_info) if p["provider_id"] == pid),
                            0,
                        )
                        self.confirm_modal_step_sync(explicit_index=idx_match)
                return True

            if input_fn is not None:
                raw_sel = sub_args[0] if len(sub_args) == 1 else input_fn(
                    "请选择要连接的大模型厂商 (输入序号 1-7 或厂商名称，如 DeepSeek / GLM / MiMo / MiniMax / Qwen / OpenAI / Anthropic): "
                ).strip()
                if not raw_sel:
                    self.console.print("[dim]已取消连接。[/dim]")
                    return True
                pid = resolve_provider_id(raw_sel)
                if not pid:
                    self.console.print(f"[red]✗ 未识别的厂商: {escape(raw_sel)}[/red]")
                    return True
                spec = self.llm.get_provider_spec(pid)
                raw_key = input_fn(f"请输入 {spec['vendor_label']} 的 API Key: ").strip()
                if not raw_key:
                    self.console.print("[red]✗ API Key 不能为空，已取消连接。[/red]")
                    return True
                ok, models, msg = await self.llm.verify_and_fetch_models(pid, raw_key)
                if not ok:
                    self.console.print(f"[red]✗ {escape(msg)}[/red]")
                    return True
                self.console.print(f"[bold #10B981]{escape(msg)}[/bold #10B981]")
                for idx_m, m_name in enumerate(models, start=1):
                    self.console.print(f"  {idx_m}. [bold #0EA5E9]{escape(m_name)}[/bold #0EA5E9]")
                model_sel = input_fn(
                    f"请选择要启用的模型 (输入序号 1-{len(models)} 或模型名称，直接回车默认选择 {models[0]}): "
                ).strip()
                chosen_model = models[0]
                if model_sel.isdigit() and 1 <= int(model_sel) <= len(models):
                    chosen_model = models[int(model_sel) - 1]
                elif model_sel:
                    chosen_model = model_sel
                self._finalize_model_activation(pid, chosen_model, via_connect=True)
            return True

        if cmd == "/ask":
            if not arg:
                self.console.print("[yellow]Usage: /ask <your question>[/yellow]")
                return True
            await self._execute_interactive_prompt(arg, forced_intent=TaskIntent.EXPLAIN, input_fn=input_fn)
            return True

        if cmd == "/bug":
            if not arg:
                self.console.print("[yellow]Usage: /bug <bug symptom or traceback>[/yellow]")
                return True
            await self._execute_interactive_prompt(arg, forced_intent=TaskIntent.BUG_FIX, input_fn=input_fn)
            return True

        if cmd == "/opt":
            if not arg:
                self.console.print("[yellow]Usage: /opt <optimization or feature target>[/yellow]")
                return True
            await self._execute_interactive_prompt(arg, forced_intent=TaskIntent.FEATURE, input_fn=input_fn)
            return True

        if cmd == "/trace":
            if not arg:
                self.console.print("[yellow]Usage: /trace <symbol_name> [rel_path][/yellow]")
                return True
            t_parts = arg.split()
            res = self.toolkit.trace_symbol(t_parts[0], rel_path=t_parts[1] if len(t_parts) > 1 else None)
            self.console.print(Rule(f"[bold cyan]LSP Semantic Trace: {escape(t_parts[0])}[/bold cyan]", style="cyan"))
            self.console.print(escape(res.output))
            self.console.print(Rule(style="dim cyan"))
            return True

        if cmd == "/diag":
            res = self.toolkit.check_diagnostics(rel_path=arg or None)
            style = "green" if res.ok else "red"
            self.console.print(Rule(f"[bold {style}]Static & Import Diagnostics[/bold {style}]", style=style))
            self.console.print(escape(res.output))
            self.console.print(Rule(style=f"dim {style}"))
            return True

        if cmd == "/profile":
            target_cmd = arg or (self.profile.test_command or "pytest -q")
            res = await self.toolkit.profile_execution(target_cmd)
            self.console.print(Rule(f"[bold magenta]Runtime Profiler: {escape(target_cmd)}[/bold magenta]", style="magenta"))
            self.console.print(escape(res.output))
            self.console.print(Rule(style="dim magenta"))
            return True

        if cmd == "/idle":
            if arg.lower() in ("run", "step", "drain"):
                outcomes = await self.looper.drain_idle_queue(max_ticks=5)
                if outcomes:
                    for out in outcomes:
                        badge = "[yellow]PREEMPTED[/yellow]" if out.preempted else "[green]COMMITTED[/green]"
                        self.console.print(f"  ⚡ {badge} {out.kind.value} ({out.target}) -> {out.summary}")
                else:
                    self.console.print("[dim]IdleHandler P2 queue is empty (0 ticks pending).[/dim]")
            st = self.looper.get_status_dict()
            self.console.print(Rule("[bold bright_blue]Serial Looper + IdleHandler Status[/bold bright_blue]", style="bright_blue"))
            self.console.print(
                f"  [bold]P1 Foreground Active:[/bold] {st['p1_active']}\n"
                f"  [bold]P2 Idle Queue ({st['p2_idle_queued']}):[/bold] {', '.join(st['p2_queue_items']) or '(empty)'}\n"
                f"  [bold]Completed Idle Ticks:[/bold] {st['completed_ticks_count']}\n"
                f"  [bold]Idle Window Budget:[/bold] LLM={st['llm_budget']} · Deterministic={st['det_budget']}"
            )
            self.console.print(Rule(style="dim bright_blue"))
            return True

        if cmd in ("/evolve", "/insights"):
            if arg.lower() in ("run", "now", "sync"):
                await self.looper.drain_idle_queue(max_ticks=6)
            models = self.memory.list_mental_models(limit=8)
            proposals = self.memory.list_optimization_proposals(status=None, limit=10)
            paradigms = self.memory.list_cognitive_paradigms(limit=6)

            def _render_evolve(c: Console) -> None:
                _sync_live_terminal_width(c)
                c.print(
                    Rule(
                        "[bold #F97316]✦ Fix Self-Evolution & First-Principles Intelligence[/bold #F97316]",
                        style="#0D9488",
                    )
                )
                if models:
                    mm_table = Table(
                        title="1. 三层本质心智模型 (现象层 -> 机制契约层 -> 第一性原理本质 & UX 影响)",
                        header_style="bold #0D9488",
                        box=box.SIMPLE_HEAVY,
                    )
                    mm_table.add_column("组件/范围", style="bold #0EA5E9", no_wrap=True)
                    mm_table.add_column("现象与机制契约", overflow="fold")
                    mm_table.add_column("第一性原理本质 & UX 影响", overflow="fold")
                    for m in models:
                        mm_table.add_row(
                            m.component_or_scope,
                            f"[bold]现象:[/bold] {escape(m.phenomenon_layer)}\n[bold #0D9488]机制:[/bold #0D9488] {escape(m.mechanism_contract_layer)}",
                            f"[bold #F97316]本质:[/bold #F97316] {escape(m.essence_first_principles)}\n[bold #10B981]UX:[/bold #10B981] {escape(m.ux_and_product_impact)}",
                        )
                    c.print(mm_table)
                else:
                    c.print("  [dim]暂无已沉淀的三层心智模型（执行分析或运行 `/evolve run` 即可自动提炼）。[/dim]")

                if proposals:
                    opt_table = Table(
                        title="2. 主动项目完善与用户体验优化提案 (Architecture / UX / Reliability / Performance)",
                        header_style="bold #0D9488",
                        box=box.SIMPLE_HEAVY,
                    )
                    opt_table.add_column("优先级·维度", style="bold #F97316", no_wrap=True)
                    opt_table.add_column("提案标题 & 目标文件", style="bold #0EA5E9", overflow="fold")
                    opt_table.add_column("本质根因 -> 演进方案 & UX 收益", overflow="fold")
                    for p in proposals:
                        files_s = ", ".join(p.target_files[:3])
                        opt_table.add_row(
                            f"{p.priority} · {p.dimension}",
                            f"{escape(p.title)}\n({escape(files_s)})",
                            f"[bold]根因:[/bold] {escape(p.root_essence)}\n[bold #10B981]方案:[/bold #10B981] {escape(p.proposed_evolution)}\n[bold #0EA5E9]收益:[/bold #0EA5E9] {escape(p.ux_or_engineering_gain)}",
                        )
                    c.print(opt_table)

                if paradigms:
                    p_table = Table(
                        title="3. 元认知自我进化范式 (Meta-Cognitive Paradigms in ~/.fix/persona.db)",
                        header_style="bold #0D9488",
                        box=box.SIMPLE_HEAVY,
                    )
                    p_table.add_column("触发域 (Hits)", style="bold #10B981", overflow="fold")
                    p_table.add_column("第一性原理检查清单 & 探索效率进化", overflow="fold")
                    for pd in paradigms:
                        chk = " · ".join(pd.first_principles_checklist[:4])
                        p_table.add_row(
                            f"{escape(pd.domain_trigger)} (hits={pd.hit_count})",
                            f"{escape(chk)}\n[bold #0EA5E9]效率复盘:[/bold #0EA5E9] {escape(pd.exploration_efficiency_lesson)}",
                        )
                    c.print(p_table)

            self._record_and_render_block(_render_evolve)
            return True

        if cmd == "/copy":
            self.copy_last_response_to_clipboard()
            return True

        self.console.print(f"[yellow]Unknown command {cmd}. Type /help to see available commands.[/yellow]")
        return True

    async def _execute_interactive_prompt(
        self,
        user_prompt: str,
        forced_intent: TaskIntent | None = None,
        input_fn: Callable[[str], str] | None = None,
    ) -> TaskSession | None:
        self._inflight_prompt = user_prompt
        if self._active_tracker is not None:
            tracker = self._active_tracker
        else:
            tracker = CollapsibleProgressTracker(
                self.console,
                on_resize_redraw=self._redraw_above_live_on_resize,
            )
        if self._tui_app is not None:
            tracker.on_tui_invalidate = self._tui_app.invalidate
        self._active_tracker = tracker
        if self._tui_app is not None:
            self._tui_app.invalidate()
            await asyncio.sleep(0)

        # Also hook asyncio loop's SIGWINCH while executing so resize signal wakes immediately
        loop = asyncio.get_running_loop()
        sigwinch_registered = False
        if hasattr(signal, "SIGWINCH"):
            try:
                loop.add_signal_handler(
                    signal.SIGWINCH,
                    lambda: tracker.handle_resize(),
                )
                sigwinch_registered = True
            except Exception:
                sigwinch_registered = False

        try:
            with tracker.track() as on_event:
                session = await self.looper.run_p1_task(
                    lambda: self.runtime.execute_task(
                        user_prompt=user_prompt,
                        forced_intent=forced_intent,
                        steer_queue=self.looper.p0_steer_queue,
                        conversation_context=self.conversation_context,
                        on_event=on_event,
                    )
                )
        except (KeyboardInterrupt, asyncio.CancelledError):
            self._task_elapsed_sec = round(max(0.0, time.time() - tracker.started_at), 1)
            self._agent_state_label = "已取消"
            saved_cancel_blocks = list(tracker.completed_stream_blocks)
            self._active_tracker = None
            self._inflight_prompt = None
            if self._tui_mode:
                def _render_cancelled_turn(c: Console) -> None:
                    render_codex_user_prompt_to(c, user_prompt)
                    for blk in saved_cancel_blocks:
                        CollapsibleProgressTracker._render_stream_block_to(c, blk)
                    c.print("[#F97316]✦ 已取消当前执行中的任务[/#F97316]")
                self._screen_blocks.append(_render_cancelled_turn)
            self.show_toast("已取消当前任务", duration_sec=3.0)
            self.console.print("\n[bold #F97316]✦ 已取消当前执行中的任务[/bold #F97316]")
            return None
        except Exception as exc:
            saved_err_blocks = list(tracker.completed_stream_blocks)
            self._active_tracker = None
            self._inflight_prompt = None
            if self._tui_mode:
                err_msg = str(exc)
                def _render_error_turn(c: Console) -> None:
                    render_codex_user_prompt_to(c, user_prompt)
                    for blk in saved_err_blocks:
                        CollapsibleProgressTracker._render_stream_block_to(c, blk)
                    c.print(f"[red]Execution error: {escape(err_msg)}[/red]")
                self._screen_blocks.append(_render_error_turn)
            self.console.print(f"[red]Execution error: {exc}[/red]")
            return None
        finally:
            self._active_tracker = None
            self._inflight_prompt = None
            if sigwinch_registered and hasattr(signal, "SIGWINCH"):
                try:
                    loop.remove_signal_handler(signal.SIGWINCH)
                except Exception:
                    pass

        if session.snapshots:
            self.snapshot_stack.append(session.snapshots[0])

        if session.phase == TaskPhase.CLARIFY and session.clarification_questions and input_fn is not None:
            self.console.print(Rule("[bold yellow]Clarification Needed[/bold yellow]", style="yellow"))
            self.console.print(escape(session.final_summary))
            answers: list[str] = []
            for idx, q in enumerate(session.clarification_questions, start=1):
                ans = input_fn(f"Answer Q{idx} (Enter for recommended '{q.recommended}'): ").strip()
                answers.append(f"Q: {q.question} -> A: {ans or q.recommended}")
            combined_prompt = f"{user_prompt}\n\n[User Clarifications]:\n" + "\n".join(answers)
            self._inflight_prompt = user_prompt
            tracker2 = CollapsibleProgressTracker(
                self.console,
                on_resize_redraw=self._redraw_above_live_on_resize,
            )
            self._active_tracker = tracker2
            try:
                with tracker2.track() as on_event2:
                    session = await self.looper.run_p1_task(
                        lambda: self.runtime.execute_task(
                            user_prompt=combined_prompt,
                            forced_intent=session.intent or forced_intent,
                            steer_queue=self.looper.p0_steer_queue,
                            conversation_context=self.conversation_context,
                            on_event=on_event2,
                        )
                    )
            finally:
                self._active_tracker = None
                self._inflight_prompt = None

        self.last_session = session
        self._task_elapsed_sec = session.elapsed_sec
        self._agent_state_label = "就绪"
        completed_sess = session
        saved_stream_blocks = list(tracker.completed_stream_blocks)
        self._turn_counter += 1
        turn_id = self._turn_counter
        fold_state: dict[str, Any] = {
            "collapsed": True,
            "turn_id": turn_id,
            "blocks_count": len(saved_stream_blocks),
        }
        if saved_stream_blocks:
            self._turn_fold_states[turn_id] = fold_state

        def _render_completed_turn(c: Console) -> None:
            _render_session_result(completed_sess, c, show_telemetry=not self._tui_mode)

        _render_turn_with_prompt = self._make_turn_render_block(
            turn_id=turn_id,
            user_prompt=user_prompt,
            completed_sess=completed_sess,
            saved_stream_blocks=saved_stream_blocks,
            fold_state=fold_state,
        )

        self._screen_blocks.append(_render_turn_with_prompt)
        if len(self._screen_blocks) > 13:
            self._screen_blocks = [self._screen_blocks[0], *self._screen_blocks[-12:]]

        _render_completed_turn(self.console)
        self.conversation_context.append(
            {
                "user": user_prompt,
                "assistant": session.final_summary,
                "modified_files": ", ".join(sorted(session.modified_files)),
            }
        )
        self._saved_turns_records.append(
            {
                "turn_id": turn_id,
                "user_prompt": user_prompt,
                "final_summary": session.final_summary,
                "modified_files": sorted(session.modified_files),
                "elapsed_sec": float(session.elapsed_sec),
                "tool_calls_count": int(session.tool_calls_count),
                "tool_calls_log": list(session.tool_calls_log),
                "total_tokens": int(session.total_tokens),
                "prompt_tokens": int(session.prompt_tokens),
                "completion_tokens": int(session.completion_tokens),
                "llm_calls": int(session.llm_calls),
                "intent": session.intent.value if session.intent else "auto",
                "stream_blocks": saved_stream_blocks,
                "timestamp": time.time(),
            }
        )
        self._persist_current_conversation_session()
        return session

    def build_fullscreen_tui_app(
        self,
        pt_input: Any = None,
        pt_output: Any = None,
    ) -> Any:
        """
        Build the minimalist Full-Screen Alternate-Screen (`full_screen=True`) Application matching the user's sketch:
        ┌──────────────────────────────────────────────────────────────────┐
        │                                                                  │
        │                  1. Top Main Content Stream                      │
        │         (Conversation, ✦ Narrative, • Explored, Markdown)        │
        │                                                                  │
        │ 当前Agent的状态 · 任务耗时          临时性提示(复制成功等)仅显示3s │
        │ ──────────────────────────────────────────────────────────────── │
        │ > 输入框                                                         │
        └──────────────────────────────────────────────────────────────────┘
        """
        from prompt_toolkit.application import Application
        from prompt_toolkit.buffer import Buffer
        from prompt_toolkit.completion import Completer, Completion
        from prompt_toolkit.formatted_text import FormattedText
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.keys import Keys
        from prompt_toolkit.layout import ConditionalContainer, Float, FloatContainer, HSplit, Layout, VSplit, Window
        from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl, UIContent
        from prompt_toolkit.layout.dimension import Dimension
        from prompt_toolkit.layout.margins import Margin
        from prompt_toolkit.layout.menus import CompletionsMenu
        from prompt_toolkit.layout.processors import AfterInput, ConditionalProcessor
        from prompt_toolkit.filters import Condition
        from prompt_toolkit.utils import get_cwidth
        from io import StringIO

        self._tui_mode = True
        self._input_top_row = 0
        self._last_input_top_hit_ts = 0.0
        self._browsing_history = False
        self._cached_completed_key: Any = None
        self._cached_completed_frags: list[tuple[str, str]] = []
        self._cached_lines_key: Any = None
        self._cached_wrapped_lines: list[list[tuple[str, str]]] = []
        self._cached_total_main_lines: int = 0

        if not self._custom_console_passed:
            self._tui_Sink = StringIO()
            setattr(self._tui_Sink, "_is_tui_sink", True)
            init_cols, _ = _query_os_terminal_size()
            self.console = create_fix_console(file=self._tui_Sink, width=init_cols, force_terminal=False)
        if not self._screen_blocks:
            self._screen_blocks.append(self._render_minimal_header_to)

        self.history_file_path.parent.mkdir(parents=True, exist_ok=True)

        slash_meta = {
            "/help": "查看所有交互命令说明",
            "/status": "查看项目技术栈、模型与地标状态",
            "/steps": "展开查看上一轮任务工具调用明细",
            "/ask": "只读架构分析与代码问答 ([file:line])",
            "/bug": "科学闭环排查与修复 Bug",
            "/opt": "性能与架构优化轨道",
            "/trace": "LSP/AST 跨文件符号调用链追踪",
            "/diag": "运行静态语法与导入诊断",
            "/profile": "运行 cProfile 性能热点剖析",
            "/idle": "查看或运行后台 IdleHandler 进化队列",
            "/evolve": "查看三层第一性原理心智模型与优化提案",
            "/insights": "查看三层第一性原理心智模型与优化提案",
            "/autonomy": "查看 L1-L4 渐进式自治评分",
            "/sandbox": "Git Worktree 隔离沙箱回归验证",
            "/rules": "管理个人开发风格规则 (Mind Mirror)",
            "/landmarks": "查看已验证的项目架构地标",
            "/skills": "查看、生成或执行项目专属诊断技能",
            "/learn": "从最近一次 Git 提交学习代码风格",
            "/diff": "查看当前工作区 Git Diff",
            "/undo": "一键回滚上一轮任务代码修改",
            "/copy": "复制上一轮回答到系统剪贴板",
            "/new": "在当前目录新建对话 (原对话自动保存至 /resume)",
            "/clear": "清空当前会话多轮上下文",
            "/resume": "恢复当前目录的历史对话并保持在最新对话位置",
            "/model": "查看当前支持切换的大模型列表或切换模型",
            "/connect": "连接其他大模型厂商 (DeepSeek/GLM/MiMo/MiniMax/Qwen/OpenAI/Anthropic)",
            "/init": "初始化 .fix/ 并扫描 Git 共变关系",
            "/exit": "退出 Fix 交互终端",
            "/quit": "退出 Fix 交互终端",
        }

        class _SlashCommandCompleter(Completer):
            def get_completions(self_, document: Any, complete_event: Any) -> Iterator[Completion]:
                if self._modal_state is not None:
                    return
                txt = document.text_before_cursor
                if not txt.startswith("/") or "\n" in txt or " " in txt:
                    return
                for spec in self.match_slash_commands(txt):
                    cmd_name = spec["cmd"]
                    yield Completion(
                        cmd_name,
                        start_position=-len(txt),
                        display=cmd_name,
                        display_meta=spec.get("desc", ""),
                    )

        completer = _SlashCommandCompleter()
        normalizing_guard = False
        history_nav_guard = False

        def _extract_active_slash_context(
            text: str | None = None,
            cursor_pos: int | None = None,
        ) -> dict[str, Any] | None:
            if self._modal_state is not None or self._slash_popup_dismissed:
                return None
            raw = input_buffer.text if text is None else text
            cur = input_buffer.cursor_position if cursor_pos is None else cursor_pos
            if not raw or "\n" in raw:
                return None

            if raw.startswith("/"):
                if " " not in raw:
                    tok_low = raw.lower()
                    if (
                        self._slash_stage2_cmd
                        and tok_low == self._slash_stage2_cmd
                        and SLASH_COMMAND_SPEC_MAP.get(tok_low, {}).get("has_stage2")
                    ):
                        opts = self.get_slash_stage2_options(tok_low, "")
                        if opts:
                            return {
                                "stage": 2,
                                "cmd": tok_low,
                                "arg_query": "",
                                "options": opts,
                                "is_mid_input": False,
                            }
                    matches = self.match_slash_commands(raw)
                    if matches:
                        return {
                            "stage": 1,
                            "token": raw,
                            "matches": matches,
                            "is_mid_input": False,
                            "token_start": 0,
                            "token_end": len(raw),
                        }
                    return None

                first_tok, rest_arg = raw.split(" ", 1)
                cmd_low = first_tok.lower()
                # Check if user is typing a mid-input `/command` token after whitespace (Claude Code v2.1+ feature)
                before_cur = raw[: max(0, min(len(raw), cur))]
                last_sp = before_cur.rfind(" ")
                if last_sp >= len(first_tok):
                    mid_tok = before_cur[last_sp + 1 :]
                    if mid_tok.startswith("/") and "/" not in mid_tok[1:]:
                        mid_matches = self.match_slash_commands(mid_tok)
                        if mid_matches:
                            return {
                                "stage": 1,
                                "token": mid_tok,
                                "matches": mid_matches,
                                "is_mid_input": True,
                                "token_start": last_sp + 1,
                                "token_end": len(before_cur),
                            }

                if cmd_low in SLASH_COMMAND_SPEC_MAP and SLASH_COMMAND_SPEC_MAP[cmd_low].get("has_stage2"):
                    opts = self.get_slash_stage2_options(cmd_low, rest_arg)
                    if opts:
                        return {
                            "stage": 2,
                            "cmd": cmd_low,
                            "arg_query": rest_arg,
                            "options": opts,
                            "is_mid_input": False,
                        }
                return None

            # Mid-input `/command` discovery when prompt starts with natural language text
            before_cur = raw[: max(0, min(len(raw), cur))]
            last_sp = before_cur.rfind(" ")
            if last_sp != -1:
                mid_tok = before_cur[last_sp + 1 :]
                if mid_tok.startswith("/") and "/" not in mid_tok[1:]:
                    mid_matches = self.match_slash_commands(mid_tok)
                    if mid_matches:
                        return {
                            "stage": 1,
                            "token": mid_tok,
                            "matches": mid_matches,
                            "is_mid_input": True,
                            "token_start": last_sp + 1,
                            "token_end": len(before_cur),
                        }
            return None

        def _get_matching_slash_commands(text: str | None = None) -> list[tuple[str, str]]:
            ctx = _extract_active_slash_context(text=text)
            if ctx is None:
                return []
            if ctx["stage"] == 1:
                return [(s["cmd"], s["desc"]) for s in ctx["matches"]]
            return [(o["fill_text"], o["desc"]) for o in ctx["options"]]

        def _is_slash_popup_active() -> bool:
            return _extract_active_slash_context() is not None

        def _get_slash_popup_items_count() -> int:
            ctx = _extract_active_slash_context()
            if ctx is None:
                return 0
            return len(ctx["matches"]) if ctx["stage"] == 1 else len(ctx["options"])

        def _build_claude_command_dock_tagged_lines(
            max_line_cells: int,
        ) -> list[tuple[int | None, list[tuple[str, str]]]]:
            ctx = _extract_active_slash_context()
            if ctx is None:
                return []
            from rich.cells import cell_len

            if ctx["stage"] == 1:
                matches = ctx["matches"]
                total_m = len(matches)
                if total_m == 0:
                    return []
                sel_i = max(0, min(total_m - 1, self._slash_popup_selected_idx))
                self._slash_popup_selected_idx = sel_i
                max_popup_rows = min(7, total_m)
                half_w = max_popup_rows // 2
                win_start = max(0, min(total_m - max_popup_rows, sel_i - half_w))
                win_end = win_start + max_popup_rows
                mode_title = "行内 / 命令补全" if ctx.get("is_mid_input") else "命令交互面板 (/ Commands)"
                out_lines: list[tuple[int | None, list[tuple[str, str]]]] = [
                    (
                        None,
                        [
                            ("bold #0D9488", "  ✦ "),
                            ("bold #F97316", f"{mode_title} ({sel_i + 1}/{total_m})"),
                            ("#6B7280", " ── 输入字母或中文关键词过滤 · ↑↓ 选择 · Tab/Space 选参 · Enter 执行"),
                        ],
                    )
                ]
                for idx_m in range(win_start, win_end):
                    spec = matches[idx_m]
                    is_sel = idx_m == sel_i
                    marker = " › " if is_sel else "   "
                    cmd_name = spec["cmd"]
                    arg_h = f" {spec['arg_hint']}" if spec.get("arg_hint") else ""
                    cat_tag = f"[{spec['category']}] "
                    cmd_col_w = 12
                    pad_cmd = max(1, cmd_col_w - cell_len(cmd_name))
                    arg_str = f"{arg_h:<18}" if arg_h else (" " * 18)
                    if cell_len(arg_str) > 18:
                        arg_str = _truncate_middle_for_width(arg_h, 17) + " "
                    tail_hint = (
                        ("  ◀ Tab/Space 展开选项 · Enter 执行" if spec.get("has_stage2") else "  ◀ Enter 立即执行")
                        if is_sel
                        else ""
                    )
                    used_w = 3 + cell_len(cmd_name) + pad_cmd + cell_len(arg_str) + cell_len(cat_tag) + cell_len(tail_hint)
                    desc_budget = max(14, max_line_cells - used_w - 2)
                    trunc_desc = _truncate_middle_for_width(spec["desc"], desc_budget)
                    row_frags: list[tuple[str, str]] = [
                        ("bold #F97316" if is_sel else "", marker),
                        ("bold #0EA5E9", f"{cmd_name}{' ' * pad_cmd}"),
                        ("#38BDF8" if is_sel else "#6B7280", arg_str),
                        ("bold #0D9488" if is_sel else "#6B7280", cat_tag),
                        ("bold #10B981" if is_sel else "#788596", trunc_desc),
                    ]
                    if tail_hint and used_w + cell_len(trunc_desc) <= max_line_cells:
                        row_frags.append(("bold #F97316", tail_hint))
                    out_lines.append((idx_m, row_frags))
                return out_lines

            # Stage 2: Interactive Argument & Option Picker right below the input box
            cmd_name = ctx["cmd"]
            opts = ctx["options"]
            total_o = len(opts)
            if total_o == 0:
                return []
            sel_i = max(0, min(total_o - 1, self._slash_popup_selected_idx))
            self._slash_popup_selected_idx = sel_i
            max_popup_rows = min(7, total_o)
            half_w = max_popup_rows // 2
            win_start = max(0, min(total_o - max_popup_rows, sel_i - half_w))
            win_end = win_start + max_popup_rows
            q_str = str(ctx.get("arg_query") or "").strip()
            filter_badge = f" · 🔎 过滤: '{q_str}'" if q_str else ""
            out_lines = [
                (
                    None,
                    [
                        ("bold #0D9488", "  ✦ "),
                        ("bold #F97316", f"{cmd_name} 交互选参 ({sel_i + 1}/{total_o}){filter_badge}"),
                        ("#6B7280", " ── 在输入框打字实时过滤 · ↑↓ 选择 · Enter 确认 · Esc 返回"),
                    ],
                )
            ]
            for idx_o in range(win_start, win_end):
                opt = opts[idx_o]
                is_sel = idx_o == sel_i
                marker = " › " if is_sel else "   "
                lbl = _truncate_middle_for_width(str(opt.get("label", "")), 34)
                pad_lbl = max(1, 35 - cell_len(lbl))
                bdg = f"[{opt['badge']}] " if opt.get("badge") else ""
                tail_h = f"  ◀ {opt.get('enter_hint', 'Enter 确认')}" if is_sel else ""
                used_w = 3 + cell_len(lbl) + pad_lbl + cell_len(bdg) + cell_len(tail_h)
                desc_budget = max(12, max_line_cells - used_w - 2)
                trunc_desc = _truncate_middle_for_width(str(opt.get("desc", "")), desc_budget)
                row_frags = [
                    ("bold #F97316" if is_sel else "", marker),
                    ("bold #0EA5E9" if is_sel else "bold", f"{lbl}{' ' * pad_lbl}"),
                    ("bold #10B981" if is_sel else "#0D9488", bdg),
                    ("bold #F97316" if is_sel else "#788596", trunc_desc),
                ]
                if tail_h and used_w + cell_len(trunc_desc) <= max_line_cells:
                    row_frags.append(("bold #F97316", tail_h))
                out_lines.append((idx_o, row_frags))
            return out_lines

        def _build_codex_shortcut_overlay_lines(max_line_cells: int) -> list[list[tuple[str, str]]]:
            """Build OpenAI Codex CLI's 3-column `Keyboard shortcuts` overlay (`Compose` / `Session` / `Transcript`)."""
            return [
                [("bold #0EA5E9", "  Keyboard shortcuts")],
                [
                    ("bold", "  Compose                      "),
                    ("bold", "Session                         "),
                    ("bold", "Transcript"),
                ],
                [
                    ("bold #0EA5E9", "  /        "),
                    ("#788596", "Commands & args     "),
                    ("bold #0EA5E9", "enter    "),
                    ("#788596", "Send / steer           "),
                    ("bold #0EA5E9", "pgup / pgdn  "),
                    ("#788596", "Scroll history"),
                ],
                [
                    ("bold #0EA5E9", "  tab      "),
                    ("#788596", "Expand command args "),
                    ("bold #0EA5E9", "/new     "),
                    ("#788596", "New conversation       "),
                    ("bold #0EA5E9", "ctrl+o       "),
                    ("#788596", "Toggle steps"),
                ],
                [
                    ("bold #0EA5E9", "  ↑ / ↓    "),
                    ("#788596", "Select / history    "),
                    ("bold #0EA5E9", "/resume  "),
                    ("#788596", "Resume session         "),
                    ("bold #0EA5E9", "ctrl+y       "),
                    ("#788596", "Copy response"),
                ],
                [
                    ("bold #0EA5E9", "  ctrl+u   "),
                    ("#788596", "Clear line          "),
                    ("bold #0EA5E9", "ctrl+c   "),
                    ("#788596", "Clear / interrupt      "),
                    ("bold #0EA5E9", "mouse drag   "),
                    ("#788596", "Select & copy"),
                ],
            ]

        def _on_buf_text_changed(_buf: Buffer) -> None:
            nonlocal normalizing_guard
            if not history_nav_guard:
                self._browsing_history = False
            if self._modal_state is not None:
                mode_m = self._modal_state.get("mode")
                if mode_m == "connect_api_key":
                    self._modal_state["api_key_input"] = _buf.text
                elif mode_m in ("connect_provider", "connect_select_model", "switch_model", "resume_session"):
                    self._apply_modal_filter_query(_buf.text)
            if normalizing_guard:
                return
            raw_t = _buf.text
            if raw_t:
                self._shortcut_overlay_open = False
            if "\r" in raw_t or "\t" in raw_t:
                normalizing_guard = True
                try:
                    before = raw_t[: _buf.cursor_position]
                    norm_before = before.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
                    norm_all = raw_t.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
                    _buf.text = norm_all
                    _buf.cursor_position = min(len(norm_all), len(norm_before))
                    raw_t = norm_all
                finally:
                    normalizing_guard = False

            # If user was in Stage 2 after pressing Tab (e.g. `buf.text == "/model"`) and directly types a filter character
            # like `f` -> automatically insert the separating space (`/model f`) so live filtering works seamlessly!
            if (
                self._slash_stage2_cmd
                and raw_t.lower().startswith(self._slash_stage2_cmd)
                and len(raw_t) > len(self._slash_stage2_cmd)
                and raw_t[len(self._slash_stage2_cmd)] != " "
                and "\n" not in raw_t
            ):
                normalizing_guard = True
                try:
                    Inserted = raw_t[: len(self._slash_stage2_cmd)] + " " + raw_t[len(self._slash_stage2_cmd) :]
                    _buf.text = Inserted
                    _buf.cursor_position = len(Inserted)
                    raw_t = Inserted
                finally:
                    normalizing_guard = False

            if self._slash_stage2_cmd:
                low_r = raw_t.lower()
                if low_r != self._slash_stage2_cmd and not low_r.startswith(self._slash_stage2_cmd + " "):
                    self._slash_stage2_cmd = ""

            if raw_t and "\n" not in raw_t:
                if raw_t != self._last_slash_prefix:
                    self._last_slash_prefix = raw_t
                    self._slash_popup_selected_idx = 0
                    self._slash_popup_dismissed = False
                    if raw_t.lower().strip() == "/model":
                        sw = self.llm.list_switchable_models()
                        cur_m_idx = next((i for i, m in enumerate(sw) if m.get("is_current")), 0)
                        if self._slash_stage2_cmd == "/model" or raw_t.lower().startswith("/model "):
                            self._slash_popup_selected_idx = cur_m_idx
            else:
                self._last_slash_prefix = ""
                self._slash_stage2_cmd = ""
                self._slash_popup_selected_idx = 0
                self._slash_popup_dismissed = False

        input_buffer = Buffer(
            completer=completer,
            history=self._file_history,
            complete_while_typing=False,
            multiline=False,
            on_text_changed=_on_buf_text_changed,
        )
        self._input_buffer_ref = input_buffer

        # Patch prompt_toolkit's UIContent.get_height_for_line so CJK 2-cell characters that wrap early
        # at odd column boundaries are counted character-by-character identically to Window._copy_body -> copy_line.
        if not getattr(UIContent, "_fix_cjk_wrap_patched", False):
            def _accurate_get_height_for_line(
                ui_self: UIContent,
                lineno: int,
                width: int,
                get_line_prefix: Any,
                slice_stop: int | None = None,
            ) -> int:
                from prompt_toolkit.application.current import get_app
                from prompt_toolkit.formatted_text import fragment_list_to_text, to_formatted_text

                key = get_app().render_counter, lineno, width, slice_stop
                try:
                    return ui_self._line_heights_cache[key]
                except KeyError:
                    if width <= 0:
                        height = 10**8
                    else:
                        line_str = fragment_list_to_text(ui_self.get_line(lineno))[:slice_stop]
                        x = 0
                        height = 1
                        wrap_count = 0
                        if get_line_prefix:
                            p0 = fragment_list_to_text(to_formatted_text(get_line_prefix(lineno, 0)))
                            x = get_cwidth(p0)
                        for ch in line_str:
                            cw = get_cwidth(ch)
                            if x + cw > width:
                                height += 1
                                wrap_count += 1
                                x = 0
                                if get_line_prefix:
                                    pn = fragment_list_to_text(
                                        to_formatted_text(get_line_prefix(lineno, wrap_count))
                                    )
                                    pw = get_cwidth(pn)
                                    if pw >= width:
                                        height = 10**8
                                        break
                                    x = pw
                            x += cw
                    ui_self._line_heights_cache[key] = height
                    return height

            setattr(UIContent, "get_height_for_line", _accurate_get_height_for_line)
            setattr(UIContent, "_fix_cjk_wrap_patched", True)

        def _get_live_tui_dimensions() -> tuple[int, int]:
            cols_c = _sync_live_terminal_width(self.console)
            _, rows_c = _query_os_terminal_size()
            if (
                self._tui_app is not None
                and getattr(self._tui_app, "output", None) is not None
                and getattr(_query_os_terminal_size, "__name__", "") == "_query_os_terminal_size"
            ):
                try:
                    from prompt_toolkit.output.defaults import DummyOutput

                    if not isinstance(self._tui_app.output, DummyOutput):
                        pt_sz = self._tui_app.output.get_size()
                        if pt_sz.columns > 0 and pt_sz.rows > 0:
                            cols_c, rows_c = pt_sz.columns, pt_sz.rows
                            self.console._width = cols_c
                            self.console._height = rows_c
                except Exception:
                    pass
            self._last_cols = cols_c
            return cols_c, rows_c

        def _build_input_visual_row_spans(cols: int, right_margin_w: int = 0) -> list[tuple[int, int]]:
            """
            Build the exact list of `(start_char_idx, end_char_idx)` for each wrapped visual row
            of `input_buffer.text` at terminal width `cols`, matching `Window._copy_body -> copy_line`.
            """
            txt = input_buffer.text
            if not txt:
                return [(0, 0)]
            width = max(16, cols - right_margin_w)
            spans: list[tuple[int, int]] = []
            pos = 0
            raw_lines = txt.split("\n")
            for idx_ln, raw_ln in enumerate(raw_lines):
                line_start = pos
                x = 3  # ` > ` or `   ` prefix is 3 columns
                seg_start = line_start
                for i, ch in enumerate(raw_ln):
                    cw = get_cwidth(ch)
                    if x + cw > width:
                        spans.append((seg_start, line_start + i))
                        seg_start = line_start + i
                        x = 3 + cw
                    else:
                        x += cw
                line_end = line_start + len(raw_ln)
                # Account for the trailing cursor space " " appended by BufferControl.create_content()
                if x + 1 > width:
                    spans.append((seg_start, line_end))
                    spans.append((line_end, line_end))
                else:
                    spans.append((seg_start, line_end))
                pos = line_end + (1 if idx_ln < len(raw_lines) - 1 else 0)
            return spans or [(0, 0)]

        def _find_cursor_visual_row(spans: list[tuple[int, int]], cur_pos: int) -> int:
            cur_row = len(spans) - 1
            for r_idx, (s_idx, e_idx) in enumerate(spans):
                if s_idx <= cur_pos <= e_idx:
                    # At a soft-wrap boundary (`spans[r_idx + 1][0] == e_idx`), `cur_pos == e_idx` sits at column 0 of `r_idx + 1`!
                    if cur_pos == e_idx and r_idx + 1 < len(spans) and spans[r_idx + 1][0] == cur_pos:
                        continue
                    cur_row = r_idx
                    break
            return cur_row

        def _compute_input_total_and_visible_rows(cols: int, rows: int) -> tuple[int, int]:
            txt = input_buffer.text
            if not txt:
                self._input_top_row = 0
                return 1, 1
            max_input_rows = max(4, min(10, rows // 3))
            spans_no_margin = _build_input_visual_row_spans(cols, right_margin_w=0)
            total_rows = len(spans_no_margin)
            if total_rows > max_input_rows:
                spans_with_bar = _build_input_visual_row_spans(cols, right_margin_w=1)
                total_rows = len(spans_with_bar)
            vis_rows = max(1, min(total_rows, max_input_rows))
            return total_rows, vis_rows

        def _sync_input_viewport(cols: int, rows: int) -> tuple[int, int, int, int]:
            """Ensure `self._input_top_row` keeps the current cursor visual row visible and within `[0, total_r - vis_r]`."""
            total_r, vis_r = _compute_input_total_and_visible_rows(cols, rows)
            margin_w = 1 if total_r > vis_r else 0
            spans = _build_input_visual_row_spans(cols, right_margin_w=margin_w)
            cur_row = _find_cursor_visual_row(spans, input_buffer.cursor_position)
            if total_r <= vis_r:
                self._input_top_row = 0
                return total_r, vis_r, cur_row, 0
            max_top = max(0, total_r - vis_r)
            if cur_row < self._input_top_row:
                self._input_top_row = cur_row
            elif cur_row >= self._input_top_row + vis_r:
                self._input_top_row = cur_row - vis_r + 1
            self._input_top_row = max(0, min(max_top, self._input_top_row))
            return total_r, vis_r, cur_row, self._input_top_row

        def _compute_input_visual_rows(cols: int, rows: int) -> int:
            _, vis_rows = _compute_input_total_and_visible_rows(cols, rows)
            return vis_rows

        def _get_command_dock_tagged_lines(max_line_cells: int) -> list[tuple[int | None, list[tuple[str, str]]]]:
            if self._modal_state is not None:
                return self._build_modal_tagged_lines(max_line_cells=max_line_cells)
            if _is_slash_popup_active():
                return _build_claude_command_dock_tagged_lines(max_line_cells=max_line_cells)
            if self._shortcut_overlay_open:
                return [(None, ln) for ln in _build_codex_shortcut_overlay_lines(max_line_cells=max_line_cells)]
            return []

        def _compute_command_dock_rows(cols: int, rows: int) -> int:
            tagged = _get_command_dock_tagged_lines(max(20, cols))
            if not tagged:
                return 0
            cap = max(4, min(14, rows - 7)) if self._modal_state is not None else max(2, min(9, rows - 8))
            return min(cap, len(tagged))

        def _compute_avail_main_height(cols: int, rows: int) -> int:
            in_r = _compute_input_visual_rows(cols, rows)
            dock_r = _compute_command_dock_rows(cols, rows)
            return max(3, rows - 4 - in_r - dock_r)

        def _move_input_cursor_vertical(direction: int) -> bool:
            """
            Move cursor up (`direction = -1`) or down (`direction = +1`) by 1 visual row inside `input_buffer`
            AND shift `self._input_top_row` in lockstep when `total_r > vis_r`, so the very first tick of
            mouse wheel / trackpad / arrow scrolling immediately scrolls the input box text and scrollbar!
            """
            txt = input_buffer.text
            if not txt:
                return False
            cols_m, rows_m = _get_live_tui_dimensions()
            total_r, vis_r, cur_row, _ = _sync_input_viewport(cols_m, rows_m)
            margin_w = 1 if total_r > vis_r else 0
            spans = _build_input_visual_row_spans(cols_m, right_margin_w=margin_w)
            if len(spans) <= 1:
                return False

            cur_pos = input_buffer.cursor_position
            target_row = cur_row + direction
            if target_row < 0 or target_row >= len(spans):
                return False

            s_cur, _ = spans[cur_row]
            target_visual_col = sum(get_cwidth(ch) for ch in txt[s_cur:cur_pos])

            s_tgt, e_tgt = spans[target_row]
            acc_col = 0
            new_pos = s_tgt
            for idx_c in range(s_tgt, e_tgt):
                cw = get_cwidth(txt[idx_c])
                if acc_col + cw > target_visual_col:
                    break
                acc_col += cw
                new_pos = idx_c + 1

            # Soft-wrap boundary guard: if `target_row` soft-wraps into `target_row + 1` (`spans[target_row + 1][0] == e_tgt`),
            # cap `new_pos` at `e_tgt - 1` so the cursor lands on `target_row` instead of slipping onto column 0 of `target_row + 1`!
            if target_row + 1 < len(spans) and spans[target_row + 1][0] == e_tgt and e_tgt > s_tgt:
                new_pos = min(new_pos, e_tgt - 1)

            input_buffer.cursor_position = max(0, min(len(txt), new_pos))

            if total_r > vis_r:
                max_top = max(0, total_r - vis_r)
                self._input_top_row = max(0, min(max_top, self._input_top_row + direction))
                if target_row < self._input_top_row:
                    self._input_top_row = target_row
                elif target_row >= self._input_top_row + vis_r:
                    self._input_top_row = target_row - vis_r + 1
                self._input_top_row = max(0, min(max_top, self._input_top_row))

            if target_row == 0 and cur_row > 0:
                self._last_input_top_hit_ts = time.monotonic()

            return True

        def _wrap_fragments_into_lines(
            frags: list[tuple[str, str]],
            max_line_cells: int,
        ) -> list[list[tuple[str, str]]]:
            from rich.cells import cell_len

            raw_lines: list[list[tuple[str, str]]] = [[]]
            for style_str, txt in frags:
                parts = txt.split("\n")
                for idx_p, part in enumerate(parts):
                    if part:
                        raw_lines[-1].append((style_str, part))
                    if idx_p < len(parts) - 1:
                        raw_lines.append([])

            wrapped: list[list[tuple[str, str]]] = []
            for rline in raw_lines:
                if not rline:
                    wrapped.append([])
                    continue
                cur_line: list[tuple[str, str]] = []
                cur_cells = 0
                for st, seg in rline:
                    seg_cells = cell_len(seg)
                    if cur_cells + seg_cells <= max_line_cells:
                        cur_line.append((st, seg))
                        cur_cells += seg_cells
                    else:
                        for ch in seg:
                            ch_w = cell_len(ch)
                            if cur_cells + ch_w > max_line_cells and cur_line:
                                wrapped.append(cur_line)
                                cur_line = [(st, ch)]
                                cur_cells = ch_w
                            else:
                                if cur_line and cur_line[-1][0] == st:
                                    cur_line[-1] = (st, cur_line[-1][1] + ch)
                                else:
                                    cur_line.append((st, ch))
                                cur_cells += ch_w
                wrapped.append(cur_line)
            return wrapped

        def _get_main_content_fragments() -> list[tuple[str, str]]:
            from rich.cells import cell_len

            cols, rows = _get_live_tui_dimensions()
            avail_height = _compute_avail_main_height(cols, rows)
            repo_name = Path(self.profile.repo_root).name

            show_main_scrollbar = self._scroll_offset_from_bottom > 0
            max_line_cells = max(20, cols - (1 if show_main_scrollbar else 0))

            # Fast cache for completed `_screen_blocks` so typing / scrolling in the input box renders in <0.05ms!
            completed_key = (
                tuple(id(fn) for fn in self._screen_blocks),
                tuple((k, bool(v.get("collapsed", True))) for k, v in sorted(self._turn_fold_states.items())),
                cols,
                getattr(self.llm, "fast_model", ""),
                getattr(self.llm, "reasoning_model", ""),
            )
            if completed_key != self._cached_completed_key:
                self._cached_completed_frags = _render_blocks_to_pt_fragments(
                    blocks=self._screen_blocks,
                    width=cols,
                    active_tracker=None,
                    inflight_prompt=None,
                    repo_name=repo_name,
                )
                self._cached_completed_key = completed_key

            if self._active_tracker is None and not self._inflight_prompt:
                lines_key = (completed_key, max_line_cells)
                if lines_key != self._cached_lines_key:
                    self._cached_wrapped_lines = _wrap_fragments_into_lines(
                        self._cached_completed_frags,
                        max_line_cells=max_line_cells,
                    )
                    self._cached_lines_key = lines_key
                lines = self._cached_wrapped_lines
            else:
                active_frags = _render_blocks_to_pt_fragments(
                    blocks=[],
                    width=cols,
                    active_tracker=self._active_tracker,
                    inflight_prompt=self._inflight_prompt,
                    repo_name=repo_name,
                )
                combined_frags = list(self._cached_completed_frags)
                if combined_frags and active_frags and not combined_frags[-1][1].endswith("\n"):
                    combined_frags.append(("", "\n"))
                combined_frags.extend(active_frags)
                lines = _wrap_fragments_into_lines(combined_frags, max_line_cells=max_line_cells)

            total_lines = len(lines)
            self._cached_total_main_lines = total_lines
            self._cached_render_lines = lines
            if total_lines > avail_height:
                max_off = max(0, total_lines - avail_height)
                self._scroll_offset_from_bottom = min(max(0, self._scroll_offset_from_bottom), max_off)
                end_idx = total_lines - self._scroll_offset_from_bottom
                start_idx = max(0, end_idx - avail_height)
                visible_lines = list(lines[start_idx:end_idx])
            else:
                self._scroll_offset_from_bottom = 0
                start_idx = 0
                end_idx = total_lines
                visible_lines = list(lines)

            # Check if there is an active mouse drag selection in `main_content_window` (both single-screen and cross-screen).
            has_main_sel = (
                self._sel_anchor is not None
                and self._sel_head is not None
                and self._sel_anchor != self._sel_head
            )
            if has_main_sel:
                (sel_line_a, sel_col_a), (sel_line_b, sel_col_b) = sorted(
                    [self._sel_anchor, self._sel_head]  # type: ignore[list-item]
                )
            else:
                sel_line_a = sel_col_a = sel_line_b = sel_col_b = -1

            def _highlight_line_selection(
                doc_idx: int,
                ln_frags: list[tuple[str, str]],
            ) -> list[tuple[str, str]]:
                if not has_main_sel or doc_idx < sel_line_a or doc_idx > sel_line_b:
                    return ln_frags
                if sel_line_a == sel_line_b:
                    s_c, e_c = min(sel_col_a, sel_col_b), max(sel_col_a, sel_col_b)
                elif doc_idx == sel_line_a:
                    s_c, e_c = sel_col_a, 999999
                elif doc_idx == sel_line_b:
                    s_c, e_c = 0, sel_col_b
                else:
                    s_c, e_c = 0, 999999

                if not ln_frags:
                    return [("bg:#264F78 #ffffff", " ")] if s_c == 0 and e_c > 0 else []

                cur_c = 0
                highlighted: list[tuple[str, str]] = []
                for st, seg in ln_frags:
                    for ch in seg:
                        cw = max(1, cell_len(ch))
                        in_sel = cur_c + cw > s_c and cur_c < e_c
                        ch_style = f"{st} bg:#264F78 #ffffff".strip() if in_sel else st
                        if highlighted and highlighted[-1][0] == ch_style:
                            highlighted[-1] = (ch_style, highlighted[-1][1] + ch)
                        else:
                            highlighted.append((ch_style, ch))
                        cur_c += cw
                return highlighted

            # When scrolled (`_scroll_offset_from_bottom > 0` and `total_lines > avail_height`),
            # render a sleek right-edge scrollbar (`┃` thumb on `#0D9488`, `│` track on `#475569`).
            # When NOT scrolled (`_scroll_offset_from_bottom == 0`), render normal clean content with no scrollbar!
            render_scrollbar_now = self._scroll_offset_from_bottom > 0 and total_lines > avail_height
            if render_scrollbar_now:
                while len(visible_lines) < avail_height:
                    visible_lines.append([])
                max_scroll_range = max(1, total_lines - avail_height)
                top_line_idx = max(0, total_lines - avail_height - self._scroll_offset_from_bottom)
                thumb_h = max(1, min(avail_height - 1, round(avail_height * avail_height / max(1, total_lines))))
                max_thumb_top = max(0, avail_height - thumb_h)
                thumb_top = round((top_line_idx / max_scroll_range) * max_thumb_top)

            out_frags: list[tuple[str, str]] = []
            for idx_l, ln_frags in enumerate(visible_lines):
                doc_line_idx = start_idx + idx_l
                styled_ln_frags = _highlight_line_selection(doc_line_idx, ln_frags)
                out_frags.extend(styled_ln_frags)
                if render_scrollbar_now:
                    line_w = sum(cell_len(seg) for _, seg in styled_ln_frags)
                    pad_w = max(0, (cols - 1) - line_w)
                    if pad_w > 0:
                        out_frags.append(("", " " * pad_w))
                    is_thumb = thumb_top <= idx_l < thumb_top + thumb_h
                    out_frags.append(("#0D9488", "┃") if is_thumb else ("#475569", "│"))
                if idx_l < len(visible_lines) - 1:
                    out_frags.append(("", "\n"))
            return out_frags or [("", "")]

        def _get_status_row_fragments() -> list[tuple[str, str]]:
            from rich.cells import cell_len

            cols, rows = _get_live_tui_dimensions()
            right_toast = self.get_right_toast_text()
            right_style = "bold #F97316"
            right_text = f"{right_toast} " if right_toast else ""

            if not right_text:
                if (
                    self._sel_dragging
                    and self._sel_anchor is not None
                    and self._sel_head is not None
                    and self._sel_anchor != self._sel_head
                ):
                    n_sel_lines = abs(self._sel_head[0] - self._sel_anchor[0]) + 1
                    right_text = f"正在拖动选中 {n_sel_lines} 行 (向上/向下拖到边缘自动滚屏) · 松开鼠标自动复制 "
                    right_style = "bold #0EA5E9"
                else:
                    total_r, vis_r, cur_row, top_r = _sync_input_viewport(cols, rows)
                    if total_r > 1:
                        bot_r = min(total_r, top_r + vis_r)
                        right_text = f"输入 {top_r + 1}-{bot_r}/{total_r}行 (光标第{cur_row + 1}行) · ↑↓/滚轮滑动 · Ctrl+J换行 · Ctrl+C清空 "
                        right_style = "nobold #6b7280"
                    elif self._scroll_offset_from_bottom > 0:
                        right_text = f"↑ 历史已上滚 {self._scroll_offset_from_bottom} 行 (↓/PgDn 回到底部) "
                        right_style = "nobold #0EA5E9"

            working_parts = self.get_claude_working_status_parts()
            if working_parts is not None:
                spinner, act, elapsed_str, meta_tail, elapsed = working_parts
                left_frags: list[tuple[str, str]] = [("bold #D97757", f" {spinner} ")]
                left_frags.extend(_build_claude_shimmer_fragments(act, elapsed))
                left_frags.append(("#94A3B8", f" · {elapsed_str}"))
                core_w = sum(cell_len(seg) for _, seg in left_frags)
                tail_str = f" {meta_tail}"
                if core_w + cell_len(tail_str) <= cols:
                    left_frags.append(("#6B7280", tail_str))
                elif core_w + cell_len(" (esc/ctrl+c 中断)") <= cols:
                    left_frags.append(("#6B7280", " (esc/ctrl+c 中断)"))
            else:
                left_text = f" {self.get_left_status_text()}"
                if self._agent_state_label == "已取消":
                    left_style = "bold #F97316"
                else:
                    left_style = "bold #10B981"
                left_frags = [(left_style, left_text)]

            if not right_text:
                return left_frags

            left_w = sum(cell_len(seg) for _, seg in left_frags)
            right_w = cell_len(right_text)
            if left_w + right_w + 2 > cols:
                return left_frags
            pad = max(2, cols - left_w - right_w)
            return [
                *left_frags,
                ("", " " * pad),
                (right_style, right_text),
            ]

        def _get_divider_fragments() -> list[tuple[str, str]]:
            cols, _ = _get_live_tui_dimensions()
            return [("#0D9488", "─" * max(20, cols))]

        session_self = self

        from prompt_toolkit.layout.processors import Processor, Transformation, TransformationInput

        class _InputMouseSelectionProcessor(Processor):
            """Highlight mouse-selected character range `[s_pos, e_pos)` inside `input_buffer`."""

            def apply_transformation(self, transformation_input: TransformationInput) -> Transformation:
                s_a = session_self._input_sel_anchor
                s_h = session_self._input_sel_head
                if s_a is None or s_h is None or s_a == s_h:
                    return Transformation(transformation_input.fragments)
                sel_s, sel_e = min(s_a, s_h), max(s_a, s_h)
                doc = transformation_input.document
                lineno = transformation_input.lineno
                lines_list = doc.lines
                if lineno < 0 or lineno >= len(lines_list):
                    return Transformation(transformation_input.fragments)
                line_start = sum(len(lines_list[i]) + 1 for i in range(lineno))
                line_end = line_start + len(lines_list[lineno])
                if sel_e <= line_start or sel_s >= line_end:
                    return Transformation(transformation_input.fragments)
                local_s = max(0, sel_s - line_start)
                local_e = min(len(lines_list[lineno]), sel_e - line_start)
                idx_char = 0
                new_frags: list[tuple[str, str]] = []
                for st, seg, *_ in transformation_input.fragments:
                    for ch in seg:
                        in_sel = local_s <= idx_char < local_e
                        ch_st = f"{st} bg:#264F78 #ffffff".strip() if in_sel else st
                        if new_frags and new_frags[-1][0] == ch_st:
                            new_frags[-1] = (ch_st, new_frags[-1][1] + ch)
                        else:
                            new_frags.append((ch_st, ch))
                        idx_char += 1
                return Transformation(new_frags)

        class _CodexComposerHighlightProcessor(Processor):
            """
            Highlight leading `/command` tokens and mid-input `/command` tokens in cyan (`bold #0EA5E9`),
            and Stage 2 subcommand keywords (`approve`, `reject`, `bootstrap`, `run`) in emerald (`bold #10B981`).
            """

            def apply_transformation(self, transformation_input: TransformationInput) -> Transformation:
                if transformation_input.lineno != 0:
                    return Transformation(transformation_input.fragments)
                doc = transformation_input.document
                if not doc.lines or not doc.lines[0]:
                    return Transformation(transformation_input.fragments)
                line0 = doc.lines[0]
                cyan_ranges: list[tuple[int, int]] = []
                emerald_ranges: list[tuple[int, int]] = []
                if line0.startswith("/"):
                    space_idx = line0.find(" ")
                    cmd_end = space_idx if space_idx != -1 else len(line0)
                    if cmd_end > 0:
                        cyan_ranges.append((0, cmd_end))
                    if space_idx != -1:
                        rest_s = line0[space_idx + 1 :]
                        sub_m = re.match(r"^(approve|reject|bootstrap|run|status)\b", rest_s, flags=re.IGNORECASE)
                        if sub_m:
                            emerald_ranges.append((space_idx + 1, space_idx + 1 + sub_m.end()))
                for m_mid in re.finditer(r"(?<=\s)/[a-zA-Z0-9_-]+", line0):
                    cyan_ranges.append((m_mid.start(), m_mid.end()))
                if not cyan_ranges and not emerald_ranges:
                    return Transformation(transformation_input.fragments)
                idx_c = 0
                styled_frags: list[tuple[str, str]] = []
                for st, seg, *_ in transformation_input.fragments:
                    for ch in seg:
                        ch_st = st
                        if any(s <= idx_c < e for s, e in cyan_ranges):
                            ch_st = "bold #0EA5E9"
                        elif any(s <= idx_c < e for s, e in emerald_ranges):
                            ch_st = "bold #10B981"
                        if styled_frags and styled_frags[-1][0] == ch_st:
                            styled_frags[-1] = (ch_st, styled_frags[-1][1] + ch)
                        else:
                            styled_frags.append((ch_st, ch))
                        idx_c += 1
                return Transformation(styled_frags)

        class _ClaudeInlineGhostProcessor(Processor):
            """
            Render Claude Code style inline completion, argument hints, and live modal filter feedback
            directly inside the input box (`input_window`), strictly clamped to single-line available width
            so it never causes extra soft-wrapping!
            """

            def apply_transformation(self, transformation_input: TransformationInput) -> Transformation:
                if transformation_input.lineno != 0 or len(transformation_input.document.lines) > 1:
                    return Transformation(transformation_input.fragments)
                from rich.cells import cell_len

                cols_g, _ = _get_live_tui_dimensions()
                line0 = transformation_input.document.lines[0] if transformation_input.document.lines else ""
                line0_w = cell_len(line0)
                budget = max(0, cols_g - 3 - line0_w - 2)
                if budget < 6:
                    return Transformation(transformation_input.fragments)

                def _clamp_ghost(text_s: str, max_w: int) -> str:
                    if cell_len(text_s) <= max_w:
                        return text_s
                    out_c: list[str] = []
                    acc_w = 0
                    for ch_g in text_s:
                        cw_g = max(1, cell_len(ch_g))
                        if acc_w + cw_g > max_w - 1:
                            out_c.append("…")
                            break
                        out_c.append(ch_g)
                        acc_w += cw_g
                    return "".join(out_c)

                ghost_frags: list[tuple[str, str]] = []
                if session_self._modal_state is not None:
                    mode_m = session_self._modal_state.get("mode")
                    sel_m = int(session_self._modal_state.get("selected_idx", 0))
                    if not line0:
                        modal_placeholders = {
                            "resume_session": "[💬 /resume] 输入关键词或序号过滤历史对话 · ↑↓ 选择 · Enter 恢复 · Esc 取消",
                            "switch_model": "[🤖 /model] 输入模型名或序号实时过滤 · ↑↓ 选择 · Enter 切换 · Esc 关闭",
                            "connect_provider": "[🔌 /connect] 输入厂商名称或序号(1-7)过滤 · ↑↓ 选择 · Enter 确认 · Esc 取消",
                            "connect_api_key": "[🔑 API Key] 请在此直接输入或粘贴 API Key · Enter 验证连接 · Esc 取消",
                            "connect_select_model": "[🤖 选择模型] 输入模型名或序号过滤 · ↑↓ 选择 · Enter 启用 · Esc 取消",
                        }
                        ph = modal_placeholders.get(str(mode_m), CODEX_PLACEHOLDER_TEXT)
                        ghost_frags.append(("nobold #6b7280", _clamp_ghost(ph, budget)))
                    elif mode_m in ("resume_session", "switch_model", "connect_provider", "connect_select_model"):
                        preview_label = ""
                        if mode_m == "connect_provider":
                            provs = session_self._modal_state.get("providers") or []
                            if 0 <= sel_m < len(provs):
                                preview_label = f"#{sel_m + 1} {provs[sel_m]['display_name']}"
                        elif mode_m == "connect_select_model":
                            mds = session_self._modal_state.get("models") or []
                            if 0 <= sel_m < len(mds):
                                preview_label = f"#{sel_m + 1} {mds[sel_m]}"
                        elif mode_m == "switch_model":
                            mds = session_self._modal_state.get("models") or []
                            if 0 <= sel_m < len(mds):
                                preview_label = f"#{sel_m + 1} [{mds[sel_m]['provider_name']}] {mds[sel_m]['model']}"
                            else:
                                preview_label = "+ 连接其他大模型厂商 (/connect)"
                        elif mode_m == "resume_session":
                            s_list = session_self._modal_state.get("sessions") or []
                            if 0 <= sel_m < len(s_list):
                                preview_label = f"#{sel_m + 1} {s_list[sel_m].get('latest_prompt') or s_list[sel_m].get('title') or ''}"
                        if preview_label:
                            ghost_frags.append(("#6b7280", _clamp_ghost(f"  → 匹配: {preview_label} (Enter 确认)", budget)))
                elif not line0:
                    ghost_frags.append(("nobold #6b7280", _clamp_ghost(CODEX_PLACEHOLDER_TEXT, budget)))
                else:
                    ctx = _extract_active_slash_context(text=line0, cursor_pos=len(line0))
                    if ctx is not None:
                        if ctx["stage"] == 1:
                            matches = ctx["matches"]
                            if matches:
                                sel_i = max(0, min(len(matches) - 1, session_self._slash_popup_selected_idx))
                                spec = matches[sel_i]
                                tok = str(ctx.get("token") or "")
                                cmd_s = spec["cmd"]
                                arg_s = f" {spec['arg_hint']}" if spec.get("arg_hint") else ""
                                if cmd_s.lower().startswith(tok.lower()) and not ctx.get("is_mid_input"):
                                    rem = cmd_s[len(tok) :]
                                    tail_s = f"{arg_s}  — {spec['desc']}"
                                    if rem:
                                        rem_clamped = _clamp_ghost(rem, budget)
                                        ghost_frags.append(("#38BDF8", rem_clamped))
                                        rem_w = cell_len(rem_clamped)
                                        if budget - rem_w > 6:
                                            ghost_frags.append(("nobold #6b7280", _clamp_ghost(tail_s, budget - rem_w)))
                                    else:
                                        ghost_frags.append(("nobold #6b7280", _clamp_ghost(tail_s, budget)))
                                else:
                                    full_g = f"  → {cmd_s}{arg_s}  — {spec['desc']}"
                                    ghost_frags.append(("nobold #6b7280", _clamp_ghost(full_g, budget)))
                        elif ctx["stage"] == 2:
                            opts = ctx.get("options") or []
                            if opts:
                                sel_i = max(0, min(len(opts) - 1, session_self._slash_popup_selected_idx))
                                opt = opts[sel_i]
                                custom_g = str(opt.get("ghost_hint") or "")
                                if custom_g:
                                    prefix_sp = "" if line0.endswith(" ") or custom_g.startswith(" ") else " "
                                    ghost_frags.append(("nobold #6b7280", _clamp_ghost(f"{prefix_sp}{custom_g}", budget)))
                                else:
                                    lbl_g = f"  → {opt['label']} ({opt.get('enter_hint', 'Enter 确认')})"
                                    ghost_frags.append(("nobold #6b7280", _clamp_ghost(lbl_g, budget)))

                if not ghost_frags:
                    return Transformation(transformation_input.fragments)
                return Transformation(list(transformation_input.fragments) + ghost_frags)

        input_control = BufferControl(
            buffer=input_buffer,
            input_processors=[
                _CodexComposerHighlightProcessor(),
                _InputMouseSelectionProcessor(),
                _ClaudeInlineGhostProcessor(),
            ],
        )

        class _InputDynamicScrollbarMargin(Margin):
            """
            Right-side scrollbar for the input box:
            - Hidden (width 0, normal clean display) when all input text fits inside `input_window`.
            - Automatically appears on the rightmost column (`┃` thumb / `│` track) when the input text
              exceeds the maximum visible input box height and scrolls vertically, 100% locked to `_input_top_row`!
            """

            def get_width(self, get_ui_content: Callable[[], Any]) -> int:
                cols_m, rows_m = _get_live_tui_dimensions()
                total_r, vis_r = _compute_input_total_and_visible_rows(cols_m, rows_m)
                return 1 if total_r > vis_r else 0

            def create_margin(self, window_render_info: Any, width: int, height: int) -> list[tuple[str, str]]:
                if width <= 0 or height <= 0:
                    return []
                cols_m, rows_m = _get_live_tui_dimensions()
                total_r, vis_r, _, top_r = _sync_input_viewport(cols_m, rows_m)
                if total_r <= vis_r:
                    return []
                max_scroll = max(1, total_r - height)
                top_row = min(max_scroll, max(0, top_r))
                thumb_h = max(1, min(height - 1, round(height * height / max(1, total_r))))
                max_thumb_top = max(0, height - thumb_h)
                thumb_top = round((top_row / max_scroll) * max_thumb_top)

                frags: list[tuple[str, str]] = []
                for idx_r in range(height):
                    is_thumb = thumb_top <= idx_r < thumb_top + thumb_h
                    frags.append(("#0D9488", "┃") if is_thumb else ("#475569", "│"))
                    if idx_r < height - 1:
                        frags.append(("", "\n"))
                return frags

        class _InputScrollbarWindow(Window):
            def _get_margin_width(self, margin: Margin) -> int:
                return margin.get_width(lambda: self._get_ui_content(width=0, height=0))

            def _scroll_when_linewrapping(self, ui_content: UIContent, width: int, height: int) -> None:
                """
                Smooth 1-visual-row scrolling across both soft-wrapped CJK paragraphs and explicit `\n` lines:
                maps `session_self._input_top_row` directly to `(self.vertical_scroll, self.vertical_scroll_2)`
                so `Window._copy_body` renders the exact visual slice `[_input_top_row : _input_top_row + height]`
                with zero dead ticks and zero blank bottom rows!
                """
                self.horizontal_scroll = 0
                if width <= 0:
                    self.vertical_scroll = 0
                    self.vertical_scroll_2 = 0
                    return
                cols_m, rows_m = _get_live_tui_dimensions()
                _sync_input_viewport(cols_m, rows_m)
                line_heights = [
                    ui_content.get_height_for_line(i, width, self.get_line_prefix)
                    for i in range(ui_content.line_count)
                ]
                total_vrows = sum(line_heights)
                max_top = max(0, total_vrows - height)
                top_vrow = max(0, min(max_top, session_self._input_top_row))
                session_self._input_top_row = top_vrow
                acc = 0
                for lineno, lh in enumerate(line_heights):
                    if acc + lh > top_vrow:
                        self.vertical_scroll = lineno
                        self.vertical_scroll_2 = top_vrow - acc
                        return
                    acc += lh
                self.vertical_scroll = max(0, len(line_heights) - 1)
                self.vertical_scroll_2 = 0

        def _get_codex_composer_line_prefix(line_number: int, wrap_count: int) -> list[tuple[str, str]]:
            if line_number == 0 and wrap_count == 0:
                if self._modal_state is not None:
                    return [("bold #F97316", " ❯ ")]
                task_busy = (
                    (self._running_task is not None and not self._running_task.done())
                    or self._active_tracker is not None
                )
                return [
                    (
                        "#6b7280" if task_busy else "bold #0EA5E9",
                        CODEX_COMPOSER_PREFIX,
                    )
                ]
            return [("", CODEX_COMPOSER_CONT_PREFIX)]

        def _get_codex_footer_fragments() -> list[tuple[str, str]]:
            from rich.cells import cell_len

            cols, rows = _get_live_tui_dimensions()
            draft_txt = input_buffer.text
            if self._shortcut_overlay_open:
                left_frags: list[tuple[str, str]] = [
                    ("", "  "),
                    ("bold #94A3B8", "? / esc"),
                    ("#6B7280", " close shortcuts"),
                ]
            elif _is_slash_popup_active():
                left_frags = [
                    ("", "  "),
                    ("bold #94A3B8", "↑↓"),
                    ("#6B7280", " select · "),
                    ("bold #94A3B8", "enter"),
                    ("#6B7280", " run · "),
                    ("bold #94A3B8", "tab"),
                    ("#6B7280", " complete · "),
                    ("bold #94A3B8", "esc"),
                    ("#6B7280", " close"),
                ]
            elif not draft_txt:
                left_frags = [
                    ("", "  "),
                    ("bold #94A3B8", "?"),
                    ("#6B7280", " for shortcuts · "),
                    ("bold #94A3B8", "/"),
                    ("#6B7280", " commands · "),
                    ("bold #94A3B8", "↑↓"),
                    ("#6B7280", " history"),
                ]
            else:
                task_busy = (
                    (self._running_task is not None and not self._running_task.done())
                    or self._active_tracker is not None
                )
                if task_busy:
                    left_frags = [
                        ("", "  "),
                        ("bold #94A3B8", "enter"),
                        ("#6B7280", " to steer · "),
                        ("bold #94A3B8", "esc"),
                        ("#6B7280", " to interrupt"),
                    ]
                else:
                    total_r, vis_r, _, top_r = _sync_input_viewport(cols, rows)
                    if total_r > 1:
                        bot_r = min(total_r, top_r + vis_r)
                        left_frags = [
                            ("", "  "),
                            ("bold #94A3B8", "enter"),
                            ("#6B7280", " to send · "),
                            ("bold #94A3B8", "ctrl+j"),
                            ("#6B7280", f" newline · {top_r + 1}-{bot_r}/{total_r}行"),
                        ]
                    else:
                        left_frags = [
                            ("", "  "),
                            ("bold #94A3B8", "enter"),
                            ("#6B7280", " to send · "),
                            ("bold #94A3B8", "ctrl+j"),
                            ("#6B7280", " newline · "),
                            ("bold #94A3B8", "ctrl+c"),
                            ("#6B7280", " clear"),
                        ]

            pct = self.get_context_left_percent()
            model_label = getattr(self.llm, "fast_model", "") or "codex"
            right_str = f"{model_label} · {pct}% context left  "
            left_w = sum(cell_len(seg) for _, seg in left_frags)
            right_w = cell_len(right_str)
            if left_w + right_w + 2 > cols:
                short_right = f"{pct}% context left  "
                if left_w + cell_len(short_right) + 2 <= cols:
                    right_str = short_right
                    right_w = cell_len(short_right)
                else:
                    return left_frags
            gap = max(2, cols - left_w - right_w)
            return [
                *left_frags,
                ("", " " * gap),
                ("#6B7280", right_str),
            ]

        input_window = _InputScrollbarWindow(
            content=input_control,
            height=lambda: Dimension.exact(_compute_input_visual_rows(*_get_live_tui_dimensions())),
            get_line_prefix=_get_codex_composer_line_prefix,
            right_margins=[_InputDynamicScrollbarMargin()],
            wrap_lines=True,
        )

        class _LiveFormattedTextControl(FormattedTextControl):
            def create_content(self, width: int, height: int) -> Any:
                self._fragment_cache.clear()
                return super().create_content(width, height)

        status_row = Window(
            content=_LiveFormattedTextControl(_get_status_row_fragments),
            height=Dimension.exact(1),
            wrap_lines=False,
        )

        divider_row = Window(
            content=_LiveFormattedTextControl(_get_divider_fragments),
            height=Dimension.exact(1),
            wrap_lines=False,
        )

        composer_bottom_row = Window(
            content=_LiveFormattedTextControl(_get_divider_fragments),
            height=Dimension.exact(1),
            wrap_lines=False,
        )

        def _get_command_dock_fragments() -> list[tuple[str, str]]:
            cols_d, rows_d = _get_live_tui_dimensions()
            in_r = _compute_input_visual_rows(cols_d, rows_d)
            avail_h = _compute_avail_main_height(cols_d, rows_d)
            dock_r = _compute_command_dock_rows(cols_d, rows_d)
            self._slash_dock_row_to_item.clear()
            self._modal_screen_row_to_item.clear()
            self._modal_doc_line_to_item.clear()
            if dock_r <= 0:
                return [("", "")]
            tagged = _get_command_dock_tagged_lines(max(20, cols_d))[:dock_r]
            dock_start_y = avail_h + 3 + in_r
            out_frags: list[tuple[str, str]] = []
            for idx_r, (item_idx, ln_frags) in enumerate(tagged):
                if item_idx is not None:
                    if self._modal_state is not None:
                        self._modal_screen_row_to_item[dock_start_y + idx_r] = item_idx
                        self._modal_doc_line_to_item[idx_r] = item_idx
                    else:
                        self._slash_dock_row_to_item[dock_start_y + idx_r] = item_idx
                out_frags.extend(ln_frags)
                if idx_r < len(tagged) - 1:
                    out_frags.append(("", "\n"))
            return out_frags or [("", "")]

        command_dock_window = Window(
            content=_LiveFormattedTextControl(_get_command_dock_fragments),
            height=lambda: Dimension.exact(_compute_command_dock_rows(*_get_live_tui_dimensions())),
            wrap_lines=False,
        )

        composer_bottom_dock = HSplit(
            [
                composer_bottom_row,
                command_dock_window,
            ]
        )

        footer_row = Window(
            content=_LiveFormattedTextControl(_get_codex_footer_fragments),
            height=Dimension.exact(1),
            wrap_lines=False,
        )

        def _execute_slash_dock_selection(
            explicit_idx: int | None = None,
            via_tab: bool = False,
        ) -> str | None:
            ctx = _extract_active_slash_context()
            if ctx is None:
                return input_buffer.text.strip() if not via_tab else None

            if ctx["stage"] == 1:
                matches = ctx["matches"]
                if not matches:
                    return None
                idx = max(
                    0,
                    min(
                        len(matches) - 1,
                        explicit_idx if explicit_idx is not None else self._slash_popup_selected_idx,
                    ),
                )
                self._slash_popup_selected_idx = idx
                sel_spec = matches[idx]
                sel_cmd = str(sel_spec["cmd"])

                if ctx.get("is_mid_input"):
                    t_start = int(ctx.get("token_start", 0))
                    t_end = int(ctx.get("token_end", len(input_buffer.text)))
                    raw_cur = input_buffer.text
                    new_txt = raw_cur[:t_start] + sel_cmd + " " + raw_cur[t_end:].lstrip()
                    input_buffer.text = new_txt
                    input_buffer.cursor_position = min(len(new_txt), t_start + len(sel_cmd) + 1)
                    self._slash_popup_dismissed = True
                    return None

                if via_tab:
                    if input_buffer.text.strip().lower() != sel_cmd:
                        input_buffer.text = sel_cmd
                        input_buffer.cursor_position = len(sel_cmd)
                        if sel_spec.get("has_stage2"):
                            self._slash_stage2_cmd = sel_cmd
                            self._slash_popup_selected_idx = 0
                            if sel_cmd == "/model":
                                sw = self.llm.list_switchable_models()
                                self._slash_popup_selected_idx = next(
                                    (i for i, m in enumerate(sw) if m.get("is_current")),
                                    0,
                                )
                            self._slash_popup_dismissed = False
                        else:
                            self._slash_popup_dismissed = True
                        return None
                    if sel_spec.get("has_stage2"):
                        input_buffer.text = sel_cmd + " "
                        input_buffer.cursor_position = len(sel_cmd) + 1
                        self._slash_stage2_cmd = sel_cmd
                        self._slash_popup_dismissed = False
                        return None
                    self._slash_popup_dismissed = True
                    return None

                if sel_spec.get("requires_arg"):
                    input_buffer.text = sel_cmd + " "
                    input_buffer.cursor_position = len(sel_cmd) + 1
                    self._slash_stage2_cmd = sel_cmd
                    self._slash_popup_selected_idx = 0
                    self._slash_popup_dismissed = False
                    return None

                input_buffer.text = sel_cmd
                input_buffer.cursor_position = len(sel_cmd)
                self._slash_stage2_cmd = ""
                self._slash_popup_dismissed = True
                return sel_cmd

            opts = ctx.get("options") or []
            if not opts:
                return input_buffer.text.strip() if not via_tab else None
            idx = max(
                0,
                min(
                    len(opts) - 1,
                    explicit_idx if explicit_idx is not None else self._slash_popup_selected_idx,
                ),
            )
            self._slash_popup_selected_idx = idx
            sel_opt = opts[idx]
            action = str(sel_opt.get("action") or "exec_line")

            if action == "transition_to_connect":
                input_buffer.text = "/connect "
                input_buffer.cursor_position = len("/connect ")
                self._slash_stage2_cmd = "/connect"
                self._slash_popup_selected_idx = 0
                self._slash_popup_dismissed = False
                return None

            if via_tab:
                fill_t = str(sel_opt.get("fill_text") or input_buffer.text)
                input_buffer.text = fill_t
                input_buffer.cursor_position = len(fill_t)
                if action == "connect_step_vendor" or fill_t.endswith(" "):
                    self._slash_popup_dismissed = False
                    self._slash_popup_selected_idx = 0
                else:
                    self._slash_popup_dismissed = True
                    self._slash_stage2_cmd = ""
                return None

            if action == "noop_empty":
                self.show_toast("当前目录暂无历史对话记录", duration_sec=3.0)
                input_buffer.reset()
                self._slash_stage2_cmd = ""
                self._slash_popup_dismissed = False
                return ""

            if action == "noop_require_arg":
                self.show_toast("请在输入框补充参数内容后按 Enter 执行", duration_sec=3.0)
                return None

            if action == "switch_model_direct":
                self._slash_stage2_cmd = ""
                self._slash_popup_dismissed = False
                input_buffer.reset()
                self._finalize_model_activation(
                    str(sel_opt["provider_id"]),
                    str(sel_opt["model"]),
                    via_connect=False,
                )
                return ""

            if action == "resume_direct":
                self._slash_stage2_cmd = ""
                self._slash_popup_dismissed = False
                input_buffer.reset()
                self.resume_conversation_session(str(sel_opt["session_id"]))
                return ""

            exec_ln = str(sel_opt.get("exec_line") or input_buffer.text.strip())
            self._slash_stage2_cmd = ""
            self._slash_popup_dismissed = True
            return exec_ln

        def _launch_command_or_prompt(raw_line: str) -> None:
            if not raw_line:
                return
            self._browsing_history = False
            self._input_top_row = 0
            self._slash_stage2_cmd = ""
            self._slash_popup_dismissed = False
            if self._running_task is not None and not self._running_task.done():
                self.record_input_history(raw_line)
                input_buffer.reset()
                self.looper.p0_steer_queue.put_nowait(raw_line)
                self.show_toast("✓ 已注入实时纠偏指令", duration_sec=3.0)
                if self._tui_app is not None:
                    self._tui_app.invalidate()
                return

            self.record_input_history(raw_line)
            input_buffer.reset()
            self._scroll_offset_from_bottom = 0
            self._agent_state_label = "正在执行"
            if not raw_line.strip().startswith("/"):
                self._inflight_prompt = raw_line.strip()
                pre_tracker = CollapsibleProgressTracker(
                    self.console,
                    on_resize_redraw=self._redraw_above_live_on_resize,
                )
                if self._tui_app is not None:
                    pre_tracker.on_tui_invalidate = self._tui_app.invalidate
                self._active_tracker = pre_tracker
            if self._tui_app is not None:
                self._tui_app.invalidate()

            async def _bg_runner() -> None:
                self._agent_state_label = "正在执行"
                if self._tui_app is not None:
                    self._tui_app.invalidate()
                await asyncio.sleep(0)
                ticker_stop = False

                async def _active_spinner_ticker() -> None:
                    while not ticker_stop:
                        await asyncio.sleep(0.12)
                        if not ticker_stop and self._tui_app is not None:
                            self._tui_app.invalidate()

                ticker_task = asyncio.create_task(_active_spinner_ticker())
                try:
                    keep = await self.handle_line(raw_line)
                    if not keep and self._tui_app is not None:
                        self._tui_app.exit()
                finally:
                    ticker_stop = True
                    ticker_task.cancel()
                    if self._agent_state_label == "正在执行":
                        self._agent_state_label = "就绪"
                    self._running_task = None
                    if self._tui_app is not None:
                        self._tui_app.invalidate()

                await asyncio.sleep(0.5)
                while (
                    self._running_task is None
                    and not self.looper.p1_active
                    and self.looper.p2_idle_queue
                ):
                    if self._tui_app is not None:
                        self._tui_app.invalidate()
                    out_tick = await self.looper.run_single_idle_tick()
                    if self._tui_app is not None:
                        self._tui_app.invalidate()
                    if out_tick is None or out_tick.preempted:
                        break

            self._running_task = asyncio.create_task(_bg_runner())

        from prompt_toolkit.data_structures import Point
        from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType, MouseModifier

        def _char_idx_to_line_cell_col(doc_line: int, char_idx: int) -> int:
            """
            Convert a 0-based Python character index `char_idx` (as reported by `prompt_toolkit`'s
            `Window._write_to_screen_at_index -> yx_to_rowcol`) into a 0-based terminal display cell
            column `cell_col`, accounting for double-width CJK characters (`cell_len(ch) == 2`).
            """
            from rich.cells import cell_len

            if char_idx <= 0 or not self._cached_render_lines:
                return 0
            if doc_line < 0 or doc_line >= len(self._cached_render_lines):
                return max(0, char_idx)
            ln_frags = self._cached_render_lines[doc_line]
            full_txt = "".join(seg for _, seg in ln_frags)
            if char_idx >= len(full_txt):
                return sum(max(1, cell_len(ch)) for ch in full_txt)
            return sum(max(1, cell_len(ch)) for ch in full_txt[:char_idx])

        def _screen_xy_to_main_doc_pos(x: int, y: int, x_is_char_idx: bool = False) -> tuple[int, int]:
            if not self._cached_render_lines:
                _get_main_content_fragments()
            cols_d, rows_d = _get_live_tui_dimensions()
            in_r = _compute_input_visual_rows(cols_d, rows_d)
            avail_h = _compute_avail_main_height(cols_d, rows_d)
            lines_list = self._cached_render_lines
            total_l = len(lines_list)
            if total_l == 0:
                return (0, 0)
            if total_l > avail_h:
                max_off = max(0, total_l - avail_h)
                off = min(max(0, self._scroll_offset_from_bottom), max_off)
                e_idx = total_l - off
                s_idx = max(0, e_idx - avail_h)
            else:
                s_idx = 0
                e_idx = total_l
            vis_count = max(1, e_idx - s_idx)
            rel_y = max(0, min(vis_count - 1, y))
            doc_line = min(total_l - 1, max(0, s_idx + rel_y))
            raw_cell_x = _char_idx_to_line_cell_col(doc_line, x) if x_is_char_idx else x
            max_col = cols_d - (1 if (self._scroll_offset_from_bottom > 0 and total_l > avail_h) else 0)
            cell_col = max(0, min(max_col, raw_cell_x))
            return (doc_line, cell_col)

        def _screen_xy_to_input_char_pos(x: int, y: int) -> int:
            txt = input_buffer.text
            if not txt:
                return 0
            cols_d, rows_d = _get_live_tui_dimensions()
            total_r, vis_r, _, top_r = _sync_input_viewport(cols_d, rows_d)
            avail_h = _compute_avail_main_height(cols_d, rows_d)
            input_start_y = avail_h + 2
            local_y = max(0, min(vis_r - 1, y - input_start_y))
            margin_w = 1 if total_r > vis_r else 0
            spans = _build_input_visual_row_spans(cols_d, right_margin_w=margin_w)
            target_vrow = max(0, min(len(spans) - 1, top_r + local_y))
            s_idx, e_idx = spans[target_vrow]
            target_col = max(0, x - 3)
            acc_c = 0
            pos = s_idx
            for idx_c in range(s_idx, e_idx):
                cw = get_cwidth(txt[idx_c])
                if acc_c + cw > target_col:
                    break
                acc_c += cw
                pos = idx_c + 1
            if target_vrow + 1 < len(spans) and spans[target_vrow + 1][0] == e_idx and e_idx > s_idx:
                pos = min(pos, e_idx - 1)
            return max(0, min(len(txt), pos))

        def _step_main_drag_autoscroll(x: int, y: int) -> bool:
            """
            Scroll `_scroll_offset_from_bottom` up or down when the mouse cursor is dragged to/past the
            top edge (`y <= 1`) or bottom edge (`y >= avail_h - 2`, including into the status/input area),
            and update `self._sel_head` to include the newly revealed off-screen lines!
            """
            if not self._cached_render_lines:
                _get_main_content_fragments()
            cols_d, rows_d = _get_live_tui_dimensions()
            in_r = _compute_input_visual_rows(cols_d, rows_d)
            avail_h = _compute_avail_main_height(cols_d, rows_d)
            total_l = len(self._cached_render_lines)
            max_off = max(0, total_l - avail_h)
            old_off = self._scroll_offset_from_bottom
            old_head = self._sel_head

            if total_l > avail_h:
                if y <= 0:
                    step = 3 if y < 0 else 2
                    self._scroll_offset_from_bottom = min(max_off, self._scroll_offset_from_bottom + step)
                elif y == 1:
                    self._scroll_offset_from_bottom = min(max_off, self._scroll_offset_from_bottom + 1)
                elif y >= avail_h:
                    step = min(5, 2 + (y - avail_h))
                    self._scroll_offset_from_bottom = max(0, self._scroll_offset_from_bottom - step)
                elif y == avail_h - 1:
                    self._scroll_offset_from_bottom = max(0, self._scroll_offset_from_bottom - 2)
                elif y == avail_h - 2:
                    self._scroll_offset_from_bottom = max(0, self._scroll_offset_from_bottom - 1)

            if y < 0:
                self._sel_head = _screen_xy_to_main_doc_pos(0, 0)
            elif y >= avail_h:
                self._sel_head = _screen_xy_to_main_doc_pos(cols_d, avail_h - 1)
            else:
                self._sel_head = _screen_xy_to_main_doc_pos(x, y)

            if self._scroll_offset_from_bottom != old_off or y <= 1 or y >= avail_h - 1:
                self._sel_autoscrolled = True
            return (self._scroll_offset_from_bottom != old_off) or (self._sel_head != old_head)

        def _step_input_drag_autoscroll(x: int, y: int) -> bool:
            cols_d, rows_d = _get_live_tui_dimensions()
            total_r, vis_r, _, _ = _sync_input_viewport(cols_d, rows_d)
            avail_h = _compute_avail_main_height(cols_d, rows_d)
            input_start_y = avail_h + 2
            old_top = self._input_top_row
            old_head = self._input_sel_head

            input_end_y = input_start_y + vis_r - 1
            if total_r > vis_r:
                max_top = max(0, total_r - vis_r)
                if y <= input_start_y:
                    self._input_top_row = max(0, self._input_top_row - 1)
                elif y >= input_end_y:
                    self._input_top_row = min(max_top, self._input_top_row + 1)

            char_pos = _screen_xy_to_input_char_pos(x, max(input_start_y, min(input_end_y, y)))
            input_buffer.cursor_position = char_pos
            self._input_sel_head = char_pos
            if self._input_top_row != old_top:
                self._input_sel_autoscrolled = True
            return (self._input_top_row != old_top) or (self._input_sel_head != old_head)

        def _ensure_drag_autoscroll_ticker() -> None:
            """
            Spawn a 16Hz background loop while the mouse button is held down during a drag so holding the
            cursor stationary at the top or bottom edge continuously scrolls and selects off-screen lines!
            """
            if self._drag_autoscroll_task is not None and not self._drag_autoscroll_task.done():
                return
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return

            async def _autoscroll_loop() -> None:
                while self._sel_dragging or self._input_sel_dragging:
                    await asyncio.sleep(0.06)
                    if not (self._sel_dragging or self._input_sel_dragging):
                        break
                    lx, ly = self._sel_last_xy
                    cols_d, rows_d = _get_live_tui_dimensions()
                    in_r = _compute_input_visual_rows(cols_d, rows_d)
                    avail_h = _compute_avail_main_height(cols_d, rows_d)
                    input_start_y = avail_h + 2
                    changed = False
                    if self._sel_dragging and (ly <= 1 or ly >= avail_h - 2):
                        changed = _step_main_drag_autoscroll(lx, ly)
                    elif self._input_sel_dragging and (ly <= input_start_y or ly >= rows_d - 1):
                        changed = _step_input_drag_autoscroll(lx, ly)
                    if changed and self._tui_app is not None:
                        self._tui_app.invalidate()

            self._drag_autoscroll_task = loop.create_task(_autoscroll_loop())

        def _dispatch_tui_mouse_event(mouse_event: MouseEvent, source: str = "screen") -> Any:
            cols_d, rows_d = _get_live_tui_dimensions()
            in_r = _compute_input_visual_rows(cols_d, rows_d)
            avail_h = _compute_avail_main_height(cols_d, rows_d)
            input_start_y = avail_h + 2
            x, y = mouse_event.position.x, mouse_event.position.y
            if source == "main":
                # `Window._write_to_screen_at_index` maps `(screen_y, screen_x)` through `yx_to_rowcol`
                # before calling `FormattedTextControl.mouse_handler`, turning `mouse_event.position.x`
                # into a 0-based character index (`col`) rather than a display cell column!
                # Convert `x` back to display cell column immediately so CJK characters (`cell_len == 2`)
                # never shift the selection left!
                doc_ln_for_x, _ = _screen_xy_to_main_doc_pos(0, y, x_is_char_idx=False)
                x = _char_idx_to_line_cell_col(doc_ln_for_x, x)
            ev_type = mouse_event.event_type

            # 1. Mouse Wheel / Trackpad Two-Finger Scroll
            if ev_type in (MouseEventType.SCROLL_UP, MouseEventType.SCROLL_DOWN):
                direction = -1 if ev_type == MouseEventType.SCROLL_UP else 1
                if self._modal_state is not None and self.move_modal_selection(direction):
                    return None
                if _is_slash_popup_active():
                    n_items = _get_slash_popup_items_count()
                    if n_items > 0:
                        self._slash_popup_selected_idx = (self._slash_popup_selected_idx + direction) % n_items
                        if self._tui_app is not None:
                            self._tui_app.invalidate()
                        return None
                if self._sel_dragging:
                    if not self._cached_render_lines:
                        _get_main_content_fragments()
                    max_off = max(0, len(self._cached_render_lines) - avail_h)
                    if direction < 0:
                        self._scroll_offset_from_bottom = min(max_off, self._scroll_offset_from_bottom + 3)
                    else:
                        self._scroll_offset_from_bottom = max(0, self._scroll_offset_from_bottom - 3)
                    self._sel_autoscrolled = True
                    self._sel_head = _screen_xy_to_main_doc_pos(x, max(0, min(avail_h - 1, y)))
                    if self._tui_app is not None:
                        self._tui_app.invalidate()
                    return None
                if self._input_sel_dragging:
                    _move_input_cursor_vertical(direction)
                    self._input_sel_autoscrolled = True
                    self._input_sel_head = input_buffer.cursor_position
                    if self._tui_app is not None:
                        self._tui_app.invalidate()
                    return None
                if source != "main" and y >= avail_h + 1:
                    if not _move_input_cursor_vertical(direction):
                        if direction < 0:
                            self._scroll_offset_from_bottom += 3
                        else:
                            self._scroll_offset_from_bottom = max(0, self._scroll_offset_from_bottom - 3)
                else:
                    if direction < 0:
                        self._scroll_offset_from_bottom += 3
                    else:
                        self._scroll_offset_from_bottom = max(0, self._scroll_offset_from_bottom - 3)
                if self._tui_app is not None:
                    self._tui_app.invalidate()
                return None

            # 2. Left Mouse Button Press (Start Selection or Shift-Extend Selection)
            if ev_type == MouseEventType.MOUSE_DOWN:
                self._sel_last_xy = (x, y)
                if self._modal_state is not None:
                    if not self._modal_screen_row_to_item:
                        _get_command_dock_fragments()
                    clicked_item = self._modal_screen_row_to_item.get(y)
                    if clicked_item is not None and self._modal_state.get("mode") in (
                        "connect_provider",
                        "connect_select_model",
                        "switch_model",
                        "resume_session",
                    ):
                        self._modal_state["selected_idx"] = clicked_item
                        if self._tui_app is not None:
                            self._tui_app.invalidate()
                        return None
                if _is_slash_popup_active():
                    if not self._slash_dock_row_to_item:
                        _get_command_dock_fragments()
                    dock_item = self._slash_dock_row_to_item.get(y)
                    if dock_item is not None:
                        self._slash_popup_selected_idx = dock_item
                        if self._tui_app is not None:
                            self._tui_app.invalidate()
                        return None
                if source != "main" and y >= avail_h + 1:
                    self._sel_anchor = None
                    self._sel_head = None
                    self._sel_dragging = False
                    self._sel_autoscrolled = False
                    c_pos = _screen_xy_to_input_char_pos(x, max(input_start_y, y))
                    input_buffer.cursor_position = c_pos
                    if MouseModifier.SHIFT in mouse_event.modifiers and self._input_sel_anchor is not None:
                        self._input_sel_head = c_pos
                        self._input_sel_autoscrolled = True
                    else:
                        self._input_sel_anchor = c_pos
                        self._input_sel_head = c_pos
                        self._input_sel_autoscrolled = False
                    self._input_sel_dragging = True
                    _ensure_drag_autoscroll_ticker()
                else:
                    self._input_sel_anchor = None
                    self._input_sel_head = None
                    self._input_sel_dragging = False
                    self._input_sel_autoscrolled = False
                    doc_pos = _screen_xy_to_main_doc_pos(x, y)
                    if MouseModifier.SHIFT in mouse_event.modifiers and self._sel_anchor is not None:
                        self._sel_head = doc_pos
                        self._sel_autoscrolled = True
                    else:
                        self._sel_anchor = doc_pos
                        self._sel_head = doc_pos
                        self._sel_autoscrolled = False
                    self._sel_dragging = True
                    _ensure_drag_autoscroll_ticker()
                if self._tui_app is not None:
                    self._tui_app.invalidate()
                return None

            # 3. Mouse Move / Hover / Drag (Modal Hover Highlight + Extend Selection + Edge Auto-Scroll!)
            if ev_type == MouseEventType.MOUSE_MOVE:
                if self._modal_state is not None and not (self._sel_dragging or self._input_sel_dragging):
                    if not self._modal_screen_row_to_item:
                        _get_command_dock_fragments()
                    hovered_item = self._modal_screen_row_to_item.get(y)
                    if (
                        hovered_item is not None
                        and self._modal_state.get("mode")
                        in ("connect_provider", "connect_select_model", "switch_model", "resume_session")
                        and self._modal_state.get("selected_idx") != hovered_item
                    ):
                        self._modal_state["selected_idx"] = hovered_item
                        if self._tui_app is not None:
                            self._tui_app.invalidate()
                        return None
                    return NotImplemented
                if _is_slash_popup_active() and not (self._sel_dragging or self._input_sel_dragging):
                    if not self._slash_dock_row_to_item:
                        _get_command_dock_fragments()
                    hovered_dock = self._slash_dock_row_to_item.get(y)
                    if hovered_dock is not None and self._slash_popup_selected_idx != hovered_dock:
                        self._slash_popup_selected_idx = hovered_dock
                        if self._tui_app is not None:
                            self._tui_app.invalidate()
                        return None
                if not (self._sel_dragging or self._input_sel_dragging):
                    return NotImplemented
                self._sel_last_xy = (x, y)
                changed_drag = False
                if self._sel_dragging:
                    changed_drag = _step_main_drag_autoscroll(x, y)
                elif self._input_sel_dragging:
                    changed_drag = _step_input_drag_autoscroll(x, y)
                if changed_drag and self._tui_app is not None:
                    self._tui_app.invalidate()
                    return None
                return NotImplemented

            # 4. Mouse Button Release (Finalize Selection & Automatically Copy for both single-screen and cross-screen!)
            if ev_type == MouseEventType.MOUSE_UP:
                if self._modal_state is not None:
                    self._sel_dragging = False
                    self._input_sel_dragging = False
                    if not self._modal_screen_row_to_item:
                        _get_command_dock_fragments()
                    clicked_item = self._modal_screen_row_to_item.get(y)
                    if clicked_item is not None and self._modal_state.get("mode") in (
                        "connect_provider",
                        "connect_select_model",
                        "switch_model",
                        "resume_session",
                    ):
                        self.confirm_modal_step_sync(explicit_index=clicked_item)
                        if self._tui_app is not None:
                            self._tui_app.invalidate()
                        return None
                if _is_slash_popup_active():
                    if not self._slash_dock_row_to_item:
                        _get_command_dock_fragments()
                    clicked_dock = self._slash_dock_row_to_item.get(y)
                    if clicked_dock is not None:
                        self._sel_dragging = False
                        self._input_sel_dragging = False
                        cmd_res = _execute_slash_dock_selection(explicit_idx=clicked_dock, via_tab=False)
                        if cmd_res:
                            _launch_command_or_prompt(cmd_res)
                        elif self._tui_app is not None:
                            self._tui_app.invalidate()
                        return None
                was_main_drag = self._sel_dragging
                was_input_drag = self._input_sel_dragging
                self._sel_dragging = False
                self._input_sel_dragging = False
                if was_main_drag:
                    if (x, y) != self._sel_last_xy:
                        _step_main_drag_autoscroll(x, y)
                    if (
                        self._sel_anchor is not None
                        and self._sel_head is not None
                        and self._sel_anchor != self._sel_head
                    ):
                        sel_txt = self.get_selected_main_text()
                        if sel_txt:
                            n_ln = len(sel_txt.splitlines())
                            label_prefix = "✓ 已复制跨屏选中内容" if self._sel_autoscrolled else "✓ 已复制选中内容"
                            self.copy_text_to_clipboard(
                                sel_txt,
                                toast_label=f"{label_prefix} ({n_ln}行 / {len(sel_txt)}字)",
                            )
                    else:
                        clicked_doc_pos = _screen_xy_to_main_doc_pos(x, y)
                        if self._sel_anchor is None or self._sel_anchor == clicked_doc_pos:
                            clicked_turn = self._find_turn_fold_at_doc_line(clicked_doc_pos[0])
                            if clicked_turn is not None:
                                self.toggle_turn_steps_collapsed(clicked_turn)
                        self._sel_anchor = None
                        self._sel_head = None
                        self._sel_autoscrolled = False
                elif was_input_drag:
                    if (
                        self._input_sel_anchor is not None
                        and self._input_sel_head is not None
                        and self._input_sel_anchor != self._input_sel_head
                    ):
                        s_i, e_i = sorted([self._input_sel_anchor, self._input_sel_head])
                        sel_in = input_buffer.text[s_i:e_i]
                        if sel_in:
                            self.copy_text_to_clipboard(
                                sel_in,
                                toast_label=f"✓ 已复制输入框选中内容 ({len(sel_in)}字)",
                            )
                    else:
                        self._input_sel_anchor = None
                        self._input_sel_head = None
                        self._input_sel_autoscrolled = False
                if self._tui_app is not None:
                    self._tui_app.invalidate()
                return None

            return None

        self.handle_tui_mouse_event = _dispatch_tui_mouse_event

        class _ScrollableMainControl(FormattedTextControl):
            def create_content(self, width: int, height: int) -> Any:
                # Reset _fragment_cache so dynamic scroll offset and terminal width updates always render fresh
                self._fragment_cache.clear()
                return super().create_content(width, height)

            def mouse_handler(self, mouse_event: MouseEvent) -> Any:
                return _dispatch_tui_mouse_event(mouse_event, source="main")

        main_content_window = Window(
            content=_ScrollableMainControl(_get_main_content_fragments, focusable=False),
            height=lambda: Dimension.exact(_compute_avail_main_height(*_get_live_tui_dimensions())),
            wrap_lines=False,
        )

        root_container = FloatContainer(
            content=HSplit(
                [
                    main_content_window,
                    status_row,
                    divider_row,
                    input_window,
                    composer_bottom_dock,
                    footer_row,
                ]
            ),
            floats=[],
        )

        kb = KeyBindings()

        @kb.add(Keys.Vt100MouseEvent)
        def _on_vt100_mouse(event: Any) -> Any:
            from prompt_toolkit.key_binding.bindings.mouse import (
                UNKNOWN_BUTTON,
                UNKNOWN_MODIFIER,
                typical_mouse_events,
                urxvt_mouse_events,
                xterm_sgr_mouse_events,
            )

            if len(event.data) < 3:
                return NotImplemented
            if event.data[2] == "M":
                mouse_ev_code, mx, my = map(ord, event.data[3:])
                m_btn, m_type, m_mods = typical_mouse_events.get(
                    mouse_ev_code,
                    (UNKNOWN_BUTTON, MouseEventType.MOUSE_MOVE, UNKNOWN_MODIFIER),
                )
                if mx >= 0xDC00:
                    mx -= 0xDC00
                if my >= 0xDC00:
                    my -= 0xDC00
                mx -= 32
                my -= 32
            else:
                data_s = event.data[2:]
                if data_s[:1] == "<":
                    sgr = True
                    data_s = data_s[1:]
                else:
                    sgr = False
                try:
                    mouse_ev_code, mx, my = map(int, data_s[:-1].split(";"))
                except Exception:
                    return NotImplemented
                m_suffix = data_s[-1]
                if sgr:
                    try:
                        m_btn, m_type, m_mods = xterm_sgr_mouse_events[mouse_ev_code, m_suffix]
                    except KeyError:
                        # SGR Any-Event Tracking (`?1003h`) emits motion with no button pressed as `35;x;yM`
                        # or `35;x;ym` (or `35 + modifier_bits`).
                        if mouse_ev_code & 32:
                            m_btn = UNKNOWN_BUTTON
                            m_type = MouseEventType.MOUSE_MOVE
                            m_mods = UNKNOWN_MODIFIER
                        else:
                            return NotImplemented
                else:
                    m_btn, m_type, m_mods = urxvt_mouse_events.get(
                        mouse_ev_code,
                        (UNKNOWN_BUTTON, MouseEventType.MOUSE_MOVE, UNKNOWN_MODIFIER),
                    )
            mx -= 1
            my -= 1
            return _dispatch_tui_mouse_event(
                MouseEvent(
                    position=Point(x=mx, y=my),
                    event_type=m_type,
                    button=m_btn,
                    modifiers=m_mods,
                ),
                source="screen",
            )

        @kb.add(Keys.BracketedPaste)
        def _on_bracketed_paste(event: Any) -> None:
            self._browsing_history = False
            data = (event.data or "").replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
            if data:
                input_buffer.insert_text(data)
                _sync_input_viewport(*_get_live_tui_dimensions())
                event.app.invalidate()

        @kb.add("?")
        def _on_question_mark(event: Any) -> None:
            if self._modal_state is None and len(input_buffer.text) == 0:
                self._shortcut_overlay_open = not self._shortcut_overlay_open
                event.app.invalidate()
                return
            self._shortcut_overlay_open = False
            input_buffer.insert_text("?")
            _sync_input_viewport(*_get_live_tui_dimensions())
            event.app.invalidate()

        @kb.add("escape")
        def _on_escape(event: Any) -> None:
            if self._modal_state is not None:
                self.close_modal("已取消弹窗操作")
                event.app.invalidate()
                return
            if self._shortcut_overlay_open:
                self._shortcut_overlay_open = False
                event.app.invalidate()
                return
            if _is_slash_popup_active():
                ctx_e = _extract_active_slash_context()
                if ctx_e is not None and ctx_e.get("stage") == 2 and self._slash_stage2_cmd and " " not in input_buffer.text:
                    self._slash_stage2_cmd = ""
                    self._slash_popup_selected_idx = 0
                else:
                    self._slash_popup_dismissed = True
                    self._slash_stage2_cmd = ""
                event.app.invalidate()
                return
            if input_buffer.complete_state is not None:
                input_buffer.cancel_completion()
                event.app.invalidate()
                return
            if self._running_task is not None and not self._running_task.done():
                self.handle_ctrl_c_action(input_buffer.text)
                event.app.invalidate()

        @kb.add("c-c")
        def _on_ctrl_c(event: Any) -> None:
            action = self.handle_ctrl_c_action(input_buffer.text)
            if action in ("cancel_task", "close_modal"):
                input_buffer.reset()
                event.app.invalidate()
            elif action == "clear_input":
                self._browsing_history = False
                self._input_top_row = 0
                input_buffer.reset()
                event.app.invalidate()
            elif action == "exit_agent":
                event.app.exit()

        @kb.add("c-d")
        def _on_ctrl_d(event: Any) -> None:
            if not input_buffer.text and (self._running_task is None or self._running_task.done()):
                event.app.exit()

        @kb.add("c-y")
        def _on_ctrl_y(event: Any) -> None:
            self.copy_last_response_to_clipboard()
            event.app.invalidate()

        @kb.add("c-o")
        def _on_ctrl_o(event: Any) -> None:
            if self.toggle_turn_steps_collapsed() is None:
                self.show_toast("暂无可展开/收起的执行过程", duration_sec=3.0)
            event.app.invalidate()

        @kb.add("pageup")
        def _on_page_up(event: Any) -> None:
            self._scroll_offset_from_bottom += 10
            event.app.invalidate()

        @kb.add("pagedown")
        def _on_page_down(event: Any) -> None:
            self._scroll_offset_from_bottom = max(0, self._scroll_offset_from_bottom - 10)
            event.app.invalidate()

        def _ensure_history_loaded() -> None:
            hist = input_buffer.history
            if not getattr(hist, "_loaded", False):
                try:
                    hist._loaded_strings = list(hist.load_history_strings())
                    hist._loaded = True
                except Exception:
                    pass
            loaded_oldest_first = hist.get_strings()
            if loaded_oldest_first and not self._browsing_history and (
                len(input_buffer._working_lines) != len(loaded_oldest_first) + 1
                or list(input_buffer._working_lines)[:-1] != loaded_oldest_first
            ):
                current_tail = input_buffer._working_lines[-1] if input_buffer._working_lines else input_buffer.text
                input_buffer._working_lines.clear()
                for h_item in loaded_oldest_first:
                    input_buffer._working_lines.append(h_item)
                input_buffer._working_lines.append(current_tail)
                input_buffer.working_index = len(input_buffer._working_lines) - 1

        def _step_history(direction: int) -> bool:
            nonlocal history_nav_guard
            _ensure_history_loaded()
            if direction < 0:
                if input_buffer.working_index > 0:
                    history_nav_guard = True
                    try:
                        self._browsing_history = True
                        input_buffer.history_search_text = None
                        input_buffer.history_backward()
                    finally:
                        history_nav_guard = False
                    cols_h, rows_h = _get_live_tui_dimensions()
                    _sync_input_viewport(cols_h, rows_h)
                    return True
                return False
            else:
                if input_buffer.working_index < len(input_buffer._working_lines) - 1:
                    history_nav_guard = True
                    try:
                        input_buffer.history_search_text = None
                        input_buffer.history_forward()
                        if input_buffer.working_index >= len(input_buffer._working_lines) - 1:
                            self._browsing_history = False
                        else:
                            self._browsing_history = True
                    finally:
                        history_nav_guard = False
                    cols_h, rows_h = _get_live_tui_dimensions()
                    _sync_input_viewport(cols_h, rows_h)
                    return True
                self._browsing_history = False
                return False

        def _handle_wheel_scroll(direction: int, event: Any) -> None:
            cols_m, rows_m = _get_live_tui_dimensions()
            total_r, _, cur_row, _ = _sync_input_viewport(cols_m, rows_m)
            if direction < 0:
                if total_r > 1:
                    if _move_input_cursor_vertical(-1):
                        event.app.invalidate()
                        return
                    if time.monotonic() - self._last_input_top_hit_ts < 0.22:
                        event.app.invalidate()
                        return
                self._scroll_offset_from_bottom += 3
                event.app.invalidate()
                return

            if total_r > 1:
                if cur_row == 0 and self._scroll_offset_from_bottom > 0:
                    self._scroll_offset_from_bottom = max(0, self._scroll_offset_from_bottom - 3)
                    event.app.invalidate()
                    return
                if _move_input_cursor_vertical(1):
                    event.app.invalidate()
                    return
            if self._scroll_offset_from_bottom > 0:
                self._scroll_offset_from_bottom = max(0, self._scroll_offset_from_bottom - 3)
                event.app.invalidate()

        @kb.add("<scroll-up>")
        def _on_wheel_up(event: Any) -> None:
            _handle_wheel_scroll(-1, event)

        @kb.add("<scroll-down>")
        def _on_wheel_down(event: Any) -> None:
            _handle_wheel_scroll(1, event)

        @kb.add("up")
        def _on_up(event: Any) -> None:
            if self._modal_state is not None and self.move_modal_selection(-1):
                event.app.invalidate()
                return
            if _is_slash_popup_active():
                n_items = _get_slash_popup_items_count()
                if n_items > 0:
                    self._slash_popup_selected_idx = (self._slash_popup_selected_idx - 1) % n_items
                    event.app.invalidate()
                    return
            if input_buffer.complete_state is not None:
                input_buffer.complete_previous()
                event.app.invalidate()
                return
            cols_m, rows_m = _get_live_tui_dimensions()
            total_r, _, cur_row, _ = _sync_input_viewport(cols_m, rows_m)
            # When editing a multi-line prompt (not in history-browsing mode) and cursor is below row 0, move up 1 visual row
            if not self._browsing_history and total_r > 1 and cur_row > 0:
                if _move_input_cursor_vertical(-1):
                    event.app.invalidate()
                    return
            # Otherwise (single-line input, at row 0, or actively browsing history), recall previous history entry!
            if _step_history(-1):
                event.app.invalidate()
                return
            # If no older history entries remain, move up inside multi-line input or scroll upper chat viewport
            if total_r > 1 and _move_input_cursor_vertical(-1):
                event.app.invalidate()
                return
            self._scroll_offset_from_bottom += 3
            event.app.invalidate()

        @kb.add("down")
        def _on_down(event: Any) -> None:
            if self._modal_state is not None and self.move_modal_selection(1):
                event.app.invalidate()
                return
            if _is_slash_popup_active():
                n_items = _get_slash_popup_items_count()
                if n_items > 0:
                    self._slash_popup_selected_idx = (self._slash_popup_selected_idx + 1) % n_items
                    event.app.invalidate()
                    return
            if input_buffer.complete_state is not None:
                input_buffer.complete_next()
                event.app.invalidate()
                return
            cols_m, rows_m = _get_live_tui_dimensions()
            total_r, _, cur_row, _ = _sync_input_viewport(cols_m, rows_m)
            # When editing a multi-line prompt (not in history-browsing mode) and cursor is above bottom row, move down 1 visual row
            if not self._browsing_history and total_r > 1 and cur_row < total_r - 1:
                if _move_input_cursor_vertical(1):
                    event.app.invalidate()
                    return
            # Otherwise step forward in history if we are browsing older history entries
            if _step_history(1):
                event.app.invalidate()
                return
            if total_r > 1 and _move_input_cursor_vertical(1):
                event.app.invalidate()
                return
            if self._scroll_offset_from_bottom > 0:
                self._scroll_offset_from_bottom = max(0, self._scroll_offset_from_bottom - 3)
                event.app.invalidate()

        @kb.add("tab")
        def _on_tab(event: Any) -> None:
            if _is_slash_popup_active():
                _execute_slash_dock_selection(via_tab=True)
                _sync_input_viewport(*_get_live_tui_dimensions())
                event.app.invalidate()
                return
            if input_buffer.complete_state is not None:
                input_buffer.complete_next()
            else:
                input_buffer.insert_text("    ")
            event.app.invalidate()

        @kb.add("s-tab")
        def _on_shift_tab(event: Any) -> None:
            if _is_slash_popup_active():
                n_items = _get_slash_popup_items_count()
                if n_items > 0:
                    self._slash_popup_selected_idx = (self._slash_popup_selected_idx - 1) % n_items
                    event.app.invalidate()
                    return
            if input_buffer.complete_state is not None:
                input_buffer.complete_previous()
                event.app.invalidate()

        @kb.add("s-up")
        def _on_shift_up(event: Any) -> None:
            self._scroll_offset_from_bottom += 3
            event.app.invalidate()

        @kb.add("s-down")
        def _on_shift_down(event: Any) -> None:
            self._scroll_offset_from_bottom = max(0, self._scroll_offset_from_bottom - 3)
            event.app.invalidate()

        @kb.add("left")
        def _on_left(event: Any) -> None:
            self._browsing_history = False
            if input_buffer.cursor_position > 0:
                input_buffer.cursor_position -= 1
                _sync_input_viewport(*_get_live_tui_dimensions())
                event.app.invalidate()

        @kb.add("right")
        def _on_right(event: Any) -> None:
            self._browsing_history = False
            if input_buffer.cursor_position < len(input_buffer.text):
                input_buffer.cursor_position += 1
                _sync_input_viewport(*_get_live_tui_dimensions())
                event.app.invalidate()
                return
            if _is_slash_popup_active():
                _execute_slash_dock_selection(via_tab=True)
                _sync_input_viewport(*_get_live_tui_dimensions())
                event.app.invalidate()

        @kb.add("c-a")
        @kb.add("home")
        def _on_home(event: Any) -> None:
            txt = input_buffer.text
            cur = input_buffer.cursor_position
            if "\n" in txt and cur > 0:
                line_start = txt.rfind("\n", 0, cur) + 1
                input_buffer.cursor_position = line_start if cur > line_start else 0
            else:
                input_buffer.cursor_position = 0
            _sync_input_viewport(*_get_live_tui_dimensions())
            event.app.invalidate()

        @kb.add("c-e")
        @kb.add("end")
        def _on_end(event: Any) -> None:
            txt = input_buffer.text
            cur = input_buffer.cursor_position
            if "\n" in txt and cur < len(txt):
                nxt_nl = txt.find("\n", cur)
                line_end = nxt_nl if nxt_nl != -1 else len(txt)
                input_buffer.cursor_position = line_end if cur < line_end else len(txt)
            else:
                input_buffer.cursor_position = len(txt)
            _sync_input_viewport(*_get_live_tui_dimensions())
            event.app.invalidate()

        @kb.add("c-u")
        def _on_kill_line_back(event: Any) -> None:
            txt = input_buffer.text
            cur = input_buffer.cursor_position
            if not txt or cur == 0:
                input_buffer.reset()
                self._input_top_row = 0
            else:
                line_start = txt.rfind("\n", 0, cur) + 1
                del_start = line_start if cur > line_start else 0
                input_buffer.text = txt[:del_start] + txt[cur:]
                input_buffer.cursor_position = del_start
            _sync_input_viewport(*_get_live_tui_dimensions())
            event.app.invalidate()

        @kb.add("c-k")
        def _on_kill_line_fwd(event: Any) -> None:
            txt = input_buffer.text
            cur = input_buffer.cursor_position
            if txt and cur < len(txt):
                nxt_nl = txt.find("\n", cur)
                del_end = (nxt_nl + 1) if nxt_nl == cur else (nxt_nl if nxt_nl != -1 else len(txt))
                input_buffer.text = txt[:cur] + txt[del_end:]
                _sync_input_viewport(*_get_live_tui_dimensions())
                event.app.invalidate()

        @kb.add("escape", "b")
        @kb.add("c-left")
        def _on_word_left(event: Any) -> None:
            pos = input_buffer.document.find_previous_word_beginning(count=event.arg)
            if pos:
                input_buffer.cursor_position += pos
            else:
                input_buffer.cursor_position = 0
            _sync_input_viewport(*_get_live_tui_dimensions())
            event.app.invalidate()

        @kb.add("escape", "f")
        @kb.add("c-right")
        def _on_word_right(event: Any) -> None:
            pos = input_buffer.document.find_next_word_ending(count=event.arg)
            if pos:
                input_buffer.cursor_position += pos
            else:
                input_buffer.cursor_position = len(input_buffer.text)
            _sync_input_viewport(*_get_live_tui_dimensions())
            event.app.invalidate()

        @kb.add("escape", "enter")
        @kb.add("c-j")
        def _on_insert_newline(event: Any) -> None:
            input_buffer.insert_text("\n")
            _sync_input_viewport(*_get_live_tui_dimensions())
            event.app.invalidate()

        @kb.add("c-p")
        def _on_prev_hist(event: Any) -> None:
            _step_history(-1)
            event.app.invalidate()

        @kb.add("c-n")
        def _on_next_hist(event: Any) -> None:
            _step_history(1)
            event.app.invalidate()

        @kb.add("enter")
        def _on_enter(event: Any) -> None:
            if self._modal_state is not None:
                mode = self._modal_state.get("mode")
                if mode in ("connect_provider", "connect_select_model", "switch_model", "resume_session"):
                    raw_typed = input_buffer.text.strip()
                    if raw_typed.isdigit():
                        idx_typed = int(raw_typed) - 1
                        input_buffer.reset()
                        self.confirm_modal_step_sync(explicit_index=idx_typed)
                    else:
                        if raw_typed:
                            self._apply_modal_filter_query(raw_typed)
                        input_buffer.reset()
                        self.confirm_modal_step_sync()
                    event.app.invalidate()
                    return
                if mode == "connect_api_key":
                    typed_key = input_buffer.text.strip() or str(self._modal_state.get("api_key_input") or "").strip()
                    input_buffer.reset()

                    async def _verify_runner() -> None:
                        await self.confirm_modal_step(explicit_api_key=typed_key)
                        event.app.invalidate()

                    asyncio.create_task(_verify_runner())
                    event.app.invalidate()
                    return
                return

            self._shortcut_overlay_open = False
            if _is_slash_popup_active():
                sel_res = _execute_slash_dock_selection(via_tab=False)
                if sel_res is None or sel_res == "":
                    _sync_input_viewport(*_get_live_tui_dimensions())
                    event.app.invalidate()
                    return
                raw_line = sel_res.strip()
            else:
                if (
                    input_buffer.complete_state is not None
                    and input_buffer.complete_state.current_completion is not None
                ):
                    comp = input_buffer.complete_state.current_completion
                    input_buffer.apply_completion(comp)
                    input_buffer.cancel_completion()
                raw_line = input_buffer.text.strip()

            if not raw_line:
                return
            _launch_command_or_prompt(raw_line)

        from prompt_toolkit.output import ColorDepth
        from prompt_toolkit.styles import Style

        tui_style = Style.from_dict(
            {
                "completion-menu": "bg:default #0EA5E9",
                "completion-menu.completion": "bg:default #0EA5E9",
                "completion-menu.completion.current": "bg:default bold #F97316",
                "completion-menu.meta.completion": "bg:default #788596",
                "completion-menu.meta.completion.current": "bg:default bold #10B981",
                "scrollbar.background": "bg:default",
                "scrollbar.button": "bg:default #0D9488",
            }
        )

        app_instance: Application[Any] = Application(
            layout=Layout(root_container, focused_element=input_window),
            key_bindings=kb,
            style=tui_style,
            full_screen=True,
            mouse_support=True,
            color_depth=ColorDepth.TRUE_COLOR,
            refresh_interval=0.5,
            input=pt_input,
            output=pt_output,
        )
        # Enable SGR Button-Event Mouse Tracking (`?1000h?1002h?1015h?1006h`) instead of Any-Event (`?1003h`)
        # so mouse click, drag-to-select (with automatic edge scrolling across multiple screens!), and wheel/trackpad
        # scrolling work with 0% idle hover CPU!
        orig_enter_alt = getattr(app_instance.output, "enter_alternate_screen", None)
        orig_write_raw = getattr(app_instance.output, "write_raw", None)
        if callable(orig_enter_alt) and callable(orig_write_raw):
            def _enhanced_enter_alt() -> None:
                orig_enter_alt()
                orig_write_raw("\033[?2004h\033[?1007h\033[?1h\033[?1000h\033[?1002h\033[?1003h\033[?1015h\033[?1006h\033[2J\033[3J\033[H")

            def _enable_button_drag_mouse() -> None:
                orig_write_raw("\033[?1000h\033[?1002h\033[?1003h\033[?1015h\033[?1006h")

            def _keep_app_cursor_keys() -> None:
                orig_write_raw("\033[?2004h\033[?1007h\033[?1h")

            try:
                setattr(app_instance.output, "enter_alternate_screen", _enhanced_enter_alt)
                setattr(app_instance.output, "enable_mouse_support", _enable_button_drag_mouse)
                setattr(app_instance.output, "reset_cursor_key_mode", _keep_app_cursor_keys)
            except Exception:
                pass

        self._tui_app = app_instance
        return app_instance

    def run_repl(self) -> None:
        """
        Launch the Fix interactive UI:
        - When attached to a real terminal (`sys.stdin.isatty()`), wipes any pre-existing terminal scrollback
          (`\033[2J\033[3J\033[H`) and enters the isolated Alternate Screen Buffer (`\033[?1049h`)
          with Button-Event SGR Mouse Tracking (`mouse_support=True`), guaranteeing:
          (1) Previous terminal lines (`user@host % fix`) are 100% gone and can NEVER appear when scrolling up/down;
          (2) Left-click & drag text selection automatically scrolls up/down when dragging to the top or bottom edge,
              selecting across multiple screens of output and copying cleanly (without scrollbar `┃`/`│` chars) to clipboard;
          (3) Trackpad / mouse wheel scrolling smoothly scrolls either the input box or the conversation history.
        - When stdin is a non-TTY pipe (e.g. automated pipe tests), runs the line loop cleanly.
        """
        if sys.stdin.isatty():
            raw_out = getattr(sys, "__stdout__", sys.stdout)
            try:
                if raw_out is not None:
                    raw_out.write(
                        "\033[2J\033[3J\033[H\033[?1049h\033[?2004h\033[?1007h\033[?1h"
                        "\033[?1000h\033[?1002h\033[?1003h\033[?1015h\033[?1006h\033[2J\033[3J\033[H"
                    )
                    raw_out.flush()
                tui_app = self.build_fullscreen_tui_app()
                asyncio.run(tui_app.run_async())
                return
            except (EOFError, KeyboardInterrupt):
                return
            finally:
                if raw_out is not None:
                    try:
                        raw_out.write("\033[?1006l\033[?1015l\033[?1003l\033[?1002l\033[?1000l\033[?1l\033[?2004l\033[?1007l\033[?1049l")
                        raw_out.flush()
                    except Exception:
                        pass

        # Fallback for non-TTY piped stdin (`printf "...\n/exit\n" | fix`)
        self.render_welcome_banner()
        while True:
            try:
                repo_name = Path(self.profile.repo_root).name
                raw = input(f"fix ({repo_name}) > ")
            except (EOFError, KeyboardInterrupt):
                self.console.print("\n[dim]Session closed.[/dim]")
                break

            try:
                keep_going = asyncio.run(self.handle_line(raw, input_fn=input))
            except KeyboardInterrupt:
                self.handle_ctrl_c_action("")
                continue
            if not keep_going:
                break


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    repo: str = typer.Option(".", "--repo", "-r", help="Target repository root path (defaults to current directory)."),
    version: bool = typer.Option(False, "--version", "-v", help="Show Fix version and exit."),
) -> None:
    """Launch interactive Fix CLI in the current repository when called without subcommands."""
    if version:
        console.print(f"Fix v{__version__}")
        raise typer.Exit(0)
    if ctx.invoked_subcommand is None:
        session = FixInteractiveSession(repo=repo)
        session.run_repl()


@app.command("chat")
def cmd_chat(
    repo: str = typer.Option(".", "--repo", "-r", help="Target repository root path."),
) -> None:
    """Launch the interactive Fix REPL console in the target repository."""
    session = FixInteractiveSession(repo=repo)
    session.run_repl()


def _do_init(
    repo: str = ".",
    install_hook: bool = True,
    out_console: Optional[Console] = None,
) -> None:
    c = out_console if isinstance(out_console, Console) else console
    _sync_live_terminal_width(c)
    toolkit, memory, context_mgr, _, evolution, _ = _build_stack(repo)
    profile = context_mgr.detect_repo_profile(toolkit.repo_root)
    invalidated = memory.invalidate_stale_facts()
    edges = evolution.scan_git_cochanges()

    hook_path = evolution.install_git_post_commit_hook() if install_hook else None

    table = Table(title=f"Fix Initialized — {profile.repo_root}", show_header=True, header_style="bold cyan", box=box.SIMPLE_HEAVY)
    table.add_column("Dimension", style="bold", no_wrap=True)
    table.add_column("Details", overflow="fold")
    table.add_row("Languages", ", ".join(profile.languages))
    table.add_row("Default Test Command", profile.test_command or "(not detected)")
    table.add_row("Git Branch / HEAD", f"{profile.git_branch or 'N/A'} ({profile.head_sha[:8] if profile.head_sha else 'no commits'})")
    table.add_row("Co-Change Coupled Pairs", str(len(edges)))
    table.add_row("Stale Landmarks Invalidated", str(invalidated))
    table.add_row("Persona DB (Global)", str(memory.persona_db_path))
    table.add_row("Repo DB (Project)", str(memory.repo_db_path))
    table.add_row("Git post-commit Hook", str(hook_path) if hook_path else "Not installed")
    c.print(table)


@app.command("init")
def cmd_init(
    repo: str = typer.Option(".", "--repo", "-r", help="Target repository root path."),
    install_hook: bool = typer.Option(True, "--install-hook/--no-hook", help="Install git post-commit diff-learning hook."),
) -> None:
    """Initialize `.fix/` in a repository, run 0-token radar + Git co-change scan, and install hooks."""
    _do_init(repo=repo, install_hook=install_hook, out_console=console)


def _do_status(
    repo: str = ".",
    out_console: Optional[Console] = None,
) -> None:
    c = out_console if isinstance(out_console, Console) else console
    _sync_live_terminal_width(c)
    toolkit, memory, context_mgr, llm, _, _ = _build_stack(repo)
    memory.invalidate_stale_facts()
    profile = context_mgr.detect_repo_profile(toolkit.repo_root)
    rules = memory.list_style_rules()
    landmarks = memory.search_landmarks(query="", limit=10, include_stale=True)
    skills = memory.list_skills()

    c.print(Rule("[bold green]Fix Agent Status[/bold green]", style="green"))
    c.print(
        f"  [bold]Repo Root:[/bold] {profile.repo_root}\n"
        f"  [bold]Languages:[/bold] {', '.join(profile.languages)}\n"
        f"  [bold]Test Command:[/bold] {profile.test_command or 'N/A'}\n"
        f"  [bold]LLM Configured:[/bold] {'Yes (' + llm.fast_model + ' / ' + llm.reasoning_model + ')' if llm.is_configured else 'No (Set FIX_API_KEY or OPENAI_API_KEY)'}\n"
        f"  [bold]Active Persona Rules:[/bold] {len(rules)}\n"
        f"  [bold]Repo Landmarks:[/bold] {len(landmarks)} ({sum(1 for f in landmarks if not f.stale)} verified, {sum(1 for f in landmarks if f.stale)} stale)\n"
        f"  [bold]Bootstrapped Skills:[/bold] {', '.join(skills) if skills else '(none yet)'}"
    )
    c.print(Rule(style="dim green"))

    if landmarks:
        lm_table = Table(title="Repository Landmarks (.fix/repo.db)", header_style="bold magenta", box=box.SIMPLE_HEAVY)
        lm_table.add_column("ID", no_wrap=True)
        lm_table.add_column("Category", no_wrap=True)
        lm_table.add_column("Status", no_wrap=True)
        lm_table.add_column("Summary", overflow="fold")
        lm_table.add_column("Files", overflow="fold")
        for f in landmarks:
            state_badge = "[red]STALE[/red]" if f.stale else f"[green]{f.status.value}[/green]"
            lm_table.add_row(f.fact_id, f.category, state_badge, escape(f.summary[:80]), ", ".join(f.evidence_paths))
        c.print(lm_table)


@app.command("status")
def cmd_status(
    repo: str = typer.Option(".", "--repo", "-r", help="Target repository root path."),
) -> None:
    """Display repository radar, verified landmarks, Developer Mind Mirror rules, and bootstrapped skills."""
    _do_status(repo=repo, out_console=console)


def _do_rules(
    repo: str = ".",
    approve: Optional[str] = None,
    reject: Optional[str] = None,
    add_pattern: Optional[str] = None,
    preferred: Optional[str] = None,
    category: str = "architecture",
    out_console: Optional[Console] = None,
) -> None:
    c = out_console if isinstance(out_console, Console) else console
    _sync_live_terminal_width(c)
    _, memory, _, _, _, _ = _build_stack(repo)

    if approve:
        ok = memory.approve_rule(approve)
        c.print(f"[green]Rule {approve} promoted to hard gate![/green]" if ok else f"[red]Rule {approve} not found.[/red]")
    if reject:
        ok = memory.reject_rule(reject)
        c.print(f"[yellow]Rule {reject} rejected.[/yellow]" if ok else f"[red]Rule {reject} not found.[/red]")
    if add_pattern and preferred:
        rule = memory.upsert_style_rule(
            StyleRule(pattern_desc=add_pattern, preferred_way=preferred, category=category)  # type: ignore[arg-type]
        )
        c.print(f"[green]Upserted rule {rule.rule_id} (hits={rule.hit_count}, status={rule.status.value})[/green]")

    rules = memory.list_style_rules(include_rejected=True)
    table = Table(title=f"Developer Mind Mirror ({memory.persona_db_path})", header_style="bold yellow", box=box.SIMPLE_HEAVY)
    table.add_column("Rule ID", style="cyan", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Hits", justify="right", no_wrap=True)
    table.add_column("Category", no_wrap=True)
    table.add_column("When Pattern", overflow="fold")
    table.add_column("Preferred Way", overflow="fold")

    for r in rules:
        table.add_row(
            r.rule_id,
            r.status.value,
            str(r.hit_count),
            r.category,
            r.pattern_desc,
            r.preferred_way,
        )
    c.print(table)


@app.command("rules")
def cmd_rules(
    repo: str = typer.Option(".", "--repo", "-r", help="Target repository root path."),
    approve: Optional[str] = typer.Option(None, "--approve", help="Approve a rule_id to PROMOTED status."),
    reject: Optional[str] = typer.Option(None, "--reject", help="Reject/disable a rule_id."),
    add_pattern: Optional[str] = typer.Option(None, "--add-pattern", help="Manually register a pattern trigger."),
    preferred: Optional[str] = typer.Option(None, "--preferred", help="Preferred engineering way for --add-pattern."),
    category: str = typer.Option("architecture", "--category", help="Category for manually added rule."),
) -> None:
    """Inspect and manage Developer Mind Mirror rules in `~/.fix/persona.db`."""
    _do_rules(
        repo=repo,
        approve=approve,
        reject=reject,
        add_pattern=add_pattern,
        preferred=preferred,
        category=category,
        out_console=console,
    )


@app.command("learn-diff")
def cmd_learn_diff(
    repo: str = typer.Option(".", "--repo", "-r", help="Target repository root path."),
    commit: str = typer.Option("HEAD", "--commit", "-c", help="Git commit revision to inspect."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress console output (for git hooks)."),
) -> None:
    """Analyze a human git commit diff with the Semantic Equivalence Filter to learn developer habits."""
    toolkit, _, _, _, evolution, _ = _build_stack(repo)
    msg_proc = subprocess.run(
        ["git", "log", "-1", "--pretty=%s", commit],
        cwd=toolkit.repo_root,
        capture_output=True,
        text=True,
    )
    diff_proc = subprocess.run(
        ["git", "show", "--format=", commit],
        cwd=toolkit.repo_root,
        capture_output=True,
        text=True,
    )
    commit_msg = msg_proc.stdout.strip() if msg_proc.returncode == 0 else ""
    human_diff = diff_proc.stdout if diff_proc.returncode == 0 else ""

    learned = asyncio.run(
        evolution.learn_from_human_diff(human_diff=human_diff, commit_message=commit_msg)
    )
    if not quiet:
        if learned:
            console.print(f"[green]Learned/updated {len(learned)} Developer Mind Mirror rule(s) from {commit}:[/green]")
            for r in learned:
                console.print(f"  - [{r.rule_id}] ({r.status.value}, hits={r.hit_count}) {r.pattern_desc} -> {r.preferred_way}")
        else:
            console.print(
                "[dim]Semantic Equivalence Filter: Commit classified as business feature/logic change; "
                "no general style rules extracted.[/dim]"
            )


@app.command("run")
def cmd_run(
    prompt: str = typer.Argument(..., help="Task description or instruction."),
    repo: str = typer.Option(".", "--repo", "-r", help="Target repository root path."),
) -> None:
    """Execute the full Fix cognitive loop (`PROBE -> CLARIFY -> REPRODUCE -> IMPLEMENT -> VERIFY -> REVIEW`)."""
    _, _, _, _, _, runtime = _build_stack(repo)
    _run_task_with_collapsible_ui(runtime, user_prompt=prompt)


@app.command("ask")
def cmd_ask(
    question: str = typer.Argument(..., help="Architectural or root-cause diagnostic question."),
    repo: str = typer.Option(".", "--repo", "-r", help="Target repository root path."),
) -> None:
    """Run Fix in read-only EXPLAIN mode (write tools physically disabled, `[file:line]` anchored)."""
    _, _, _, _, _, runtime = _build_stack(repo)
    _run_task_with_collapsible_ui(runtime, user_prompt=question, forced_intent=TaskIntent.EXPLAIN)


@app.command("bug")
def cmd_bug(
    symptom: str = typer.Argument(..., help="Bug symptom, traceback, or failing behavior."),
    repo: str = typer.Option(".", "--repo", "-r", help="Target repository root path."),
) -> None:
    """Run scientific BUG_FIX workflow (`Hypothesis -> Reproduce -> Fix -> Verify -> Independent Review`)."""
    _, _, _, _, _, runtime = _build_stack(repo)
    _run_task_with_collapsible_ui(runtime, user_prompt=symptom, forced_intent=TaskIntent.BUG_FIX)


@app.command("opt")
def cmd_opt(
    target: str = typer.Argument(..., help="Performance, refactoring, or feature optimization goal."),
    repo: str = typer.Option(".", "--repo", "-r", help="Target repository root path."),
) -> None:
    """Run FEATURE / architecture & performance optimization workflow."""
    _, _, _, _, _, runtime = _build_stack(repo)
    _run_task_with_collapsible_ui(runtime, user_prompt=target, forced_intent=TaskIntent.FEATURE)


@app.command("evolve")
def cmd_evolve(
    repo: str = typer.Option(".", "--repo", "-r", help="Target repository root path."),
    run_idle: bool = typer.Option(True, "--run/--no-run", help="Drain pending P2 self-evolution ticks before displaying insights."),
) -> None:
    """Inspect and trigger Fix's 3-layer First-Principles Mental Models, UX/Arch Optimization Proposals, and Meta-Cognitive Paradigms."""
    session = FixInteractiveSession(repo=repo)
    asyncio.run(session.handle_line("/evolve run" if run_idle else "/evolve"))


def main() -> None:
    if len(sys.argv) > 1:
        first_arg = sys.argv[1]
        if not first_arg.startswith("-") and first_arg not in KNOWN_SUBCOMMANDS:
            sys.argv.insert(1, "run")
    app()


if __name__ == "__main__":
    main()
