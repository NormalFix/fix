"""Live execution progress tracker (`CollapsibleProgressTracker`) and step group formatters."""

from __future__ import annotations

import re
import signal
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from rich.console import Console, Group
from rich.live import Live
from rich.markup import escape
from rich.text import Text

from fix.core.runtime import format_tool_call_summary
from fix.protocol.schema import TaskSession
from fix.tui.theme import (
    CleanMarkdown,
    PRIMARY_TEXT_STYLE,
    SECONDARY_STEP_TEXT_STYLE,
    _sync_live_terminal_width,
    _truncate_middle_for_width,
    console,
    normalize_line_range_notation,
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

