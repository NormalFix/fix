"""Rich Markdown rendering elements, color palette, console factory, and live OS terminal sizing."""

from __future__ import annotations

import os
import re
import shutil
import sys
from typing import Any, Callable

from rich import box
from rich.console import Console, ConsoleOptions, RenderResult
from rich.markdown import CodeBlock, Heading, Markdown, Paragraph, TableElement
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

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



def _get_query_os_terminal_size_fn() -> Callable[[], tuple[int, int]]:
    """Resolve `_query_os_terminal_size`, respecting any runtime monkeypatch on `fix.cli` or `fix.tui.theme`."""
    cli_mod = sys.modules.get("fix.cli")
    if cli_mod is not None:
        patched = getattr(cli_mod, "_query_os_terminal_size", None)
        if callable(patched) and patched is not _query_os_terminal_size:
            return patched
    return _query_os_terminal_size


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

    cols, rows = _get_query_os_terminal_size_fn()()
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


def _render_blocks_to_pt_fragments(
    blocks: list[Callable[[Console], None]],
    width: int,
    active_tracker: Any = None,
    inflight_prompt: str | None = None,
    repo_name: str = "",
) -> list[tuple[str, str]]:
    """Render session screen blocks into prompt_toolkit (style, text) fragments for the full-screen TUI."""
    from io import StringIO
    from fix.tui.composer import render_codex_user_prompt_to
    from fix.tui.tracker import format_explored_group_lines

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


console = create_fix_console()
