"""Full-screen `prompt_toolkit` alternate-screen TUI application, viewport scrolling, mouse & keybindings."""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterator

from rich.console import Console
from rich.markup import escape

from fix.protocol.schema import TaskIntent
from fix.tui.composer import (
    CODEX_COMPOSER_CONT_PREFIX,
    CODEX_COMPOSER_PREFIX,
    CODEX_PLACEHOLDER_TEXT,
    SLASH_COMMAND_SPEC_MAP,
    SLASH_COMMANDS,
    render_codex_user_prompt_to,
)
from fix.tui.theme import (
    CleanMarkdown,
    _get_query_os_terminal_size_fn,
    _render_blocks_to_pt_fragments,
    _sync_live_terminal_width,
    _truncate_middle_for_width,
    create_fix_console,
)
from fix.tui.tracker import (
    CLAUDE_SPINNER_FRAMES,
    CollapsibleProgressTracker,
    _build_claude_shimmer_fragments,
    format_elapsed_hms,
    format_explored_group_lines,
)


class FullscreenTuiMixin:
    """Mixin building and running the full-screen `prompt_toolkit` alternate-screen TUI."""

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
        self._cached_active_key: Any = None
        self._cached_active_lines: list[list[tuple[str, str]]] = []
        self._cached_total_main_lines: int = 0
        _input_spans_cache: dict[tuple[str, int, int], list[tuple[int, int]]] = {}
        _per_block_lines_cache: dict[tuple[Any, ...], list[list[tuple[str, str]]]] = {}

        if not self._custom_console_passed:
            self._tui_Sink = StringIO()
            setattr(self._tui_Sink, "_is_tui_sink", True)
            init_cols, _ = _get_query_os_terminal_size_fn()()
            self.console = create_fix_console(file=self._tui_Sink, width=init_cols, force_terminal=False)
        if not self._screen_blocks:
            self._screen_blocks.append(self._render_minimal_header_to)

        self.history_file_path.parent.mkdir(parents=True, exist_ok=True)

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
            if self._sel_anchor is not None or self._sel_head is not None:
                self._sel_anchor = None
                self._sel_head = None
                self._sel_dragging = False
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
            _q_fn = _get_query_os_terminal_size_fn()
            _, rows_c = _q_fn()
            if (
                self._tui_app is not None
                and getattr(self._tui_app, "output", None) is not None
                and getattr(_q_fn, "__name__", "") == "_query_os_terminal_size"
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
            cache_k = (txt, cols, right_margin_w)
            cached_sp = _input_spans_cache.get(cache_k)
            if cached_sp is not None:
                return cached_sp
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
            res_spans = spans or [(0, 0)]
            if len(_input_spans_cache) > 16:
                _input_spans_cache.clear()
            _input_spans_cache[cache_k] = res_spans
            return res_spans

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

            lines_key = (completed_key, max_line_cells)
            if lines_key != self._cached_lines_key:
                from fix.tui.theme import _get_block_cache_key, _render_single_block_to_pt_fragments

                wrapped_blocks: list[list[tuple[str, str]]] = []
                n_blks = len(self._screen_blocks)
                for idx_b, fn_b in enumerate(self._screen_blocks):
                    b_ckey = _get_block_cache_key(fn_b, cols)
                    l_ckey = (b_ckey, max_line_cells) if b_ckey is not None else None
                    b_frags = _render_single_block_to_pt_fragments(fn_b, cols)
                    if b_frags == [("", "")]:
                        continue
                    b_lines = _per_block_lines_cache.get(l_ckey) if l_ckey is not None else None
                    if b_lines is None:
                        b_lines = _wrap_fragments_into_lines(b_frags, max_line_cells=max_line_cells)
                        if l_ckey is not None:
                            if len(_per_block_lines_cache) > 256:
                                _per_block_lines_cache.clear()
                            _per_block_lines_cache[l_ckey] = b_lines
                    if (
                        idx_b < n_blks - 1
                        and b_lines
                        and b_lines[-1] == []
                        and b_frags
                        and b_frags[-1][1].endswith("\n")
                    ):
                        wrapped_blocks.extend(b_lines[:-1])
                    else:
                        wrapped_blocks.extend(b_lines)
                self._cached_wrapped_lines = wrapped_blocks or [[]]
                self._cached_lines_key = lines_key

            if self._active_tracker is None and not self._inflight_prompt:
                lines = self._cached_wrapped_lines
            else:
                tr_obj = self._active_tracker
                active_key = (
                    cols,
                    max_line_cells,
                    self._inflight_prompt,
                    id(tr_obj),
                    len(tr_obj.completed_stream_blocks) if tr_obj is not None else 0,
                    tuple(tr_obj._pending_explore_actions) if tr_obj is not None else (),
                )
                if active_key != self._cached_active_key:
                    active_frags = _render_blocks_to_pt_fragments(
                        blocks=[],
                        width=cols,
                        active_tracker=tr_obj,
                        inflight_prompt=self._inflight_prompt,
                        repo_name=repo_name,
                    )
                    self._cached_active_lines = _wrap_fragments_into_lines(
                        active_frags,
                        max_line_cells=max_line_cells,
                    )
                    self._cached_active_key = active_key
                base_lines = self._cached_wrapped_lines
                if (
                    base_lines
                    and base_lines[-1] == []
                    and self._cached_completed_frags
                    and self._cached_completed_frags[-1][1].endswith("\n")
                ):
                    lines = base_lines[:-1] + self._cached_active_lines
                else:
                    lines = base_lines + self._cached_active_lines

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
                self._content_cache.clear()
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

            self._sel_anchor = None
            self._sel_head = None
            self._sel_dragging = False
            self._input_sel_anchor = None
            self._input_sel_head = None
            self._input_sel_dragging = False
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
                if getattr(self._tui_app, "_is_running", False):
                    try:
                        self._tui_app._redraw()
                    except Exception:
                        pass
            self.record_input_history(raw_line)

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
                    if mouse_event.button == MouseButton.LEFT:
                        if source != "main" and y >= avail_h + 1:
                            self._sel_anchor = None
                            self._sel_head = None
                            c_pos = _screen_xy_to_input_char_pos(x, max(input_start_y, y))
                            self._input_sel_anchor = c_pos
                            self._input_sel_head = c_pos
                            self._input_sel_dragging = True
                        else:
                            self._input_sel_anchor = None
                            self._input_sel_head = None
                            doc_pos = _screen_xy_to_main_doc_pos(x, y)
                            self._sel_anchor = doc_pos
                            self._sel_head = doc_pos
                            self._sel_dragging = True
                        _ensure_drag_autoscroll_ticker()
                    else:
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
                # Reset both _fragment_cache and _content_cache so selection highlights and scroll updates always render fresh
                self._fragment_cache.clear()
                self._content_cache.clear()
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
                    if m_suffix == "m":
                        m_btn = MouseButton.LEFT
                        m_type = MouseEventType.MOUSE_UP
                        m_mods = (
                            frozenset({MouseModifier.SHIFT})
                            if (mouse_ev_code & 4)
                            else frozenset()
                        )
                    else:
                        try:
                            m_btn, m_type, m_mods = xterm_sgr_mouse_events[mouse_ev_code, m_suffix]
                        except KeyError:
                            if (mouse_ev_code & 3) == 3 and not (mouse_ev_code & 32):
                                m_btn = MouseButton.LEFT
                                m_type = MouseEventType.MOUSE_UP
                                m_mods = frozenset()
                            elif mouse_ev_code & 32:
                                m_btn = MouseButton.NONE if (mouse_ev_code & 3) == 3 else MouseButton.LEFT
                                m_type = MouseEventType.MOUSE_MOVE
                                m_mods = UNKNOWN_MODIFIER
                            else:
                                return NotImplemented
                    # If button was released outside window and terminal now sends hover motion (code & 3 == 3), finalize drag
                    if (
                        m_type == MouseEventType.MOUSE_MOVE
                        and (mouse_ev_code & 3) == 3
                        and (self._sel_dragging or self._input_sel_dragging)
                    ):
                        m_type = MouseEventType.MOUSE_UP
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

        @kb.add("escape", eager=True)
        def _on_escape(event: Any) -> None:
            if self._sel_anchor is not None or self._sel_head is not None:
                self._sel_anchor = None
                self._sel_head = None
                self._sel_dragging = False
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
            max_render_postpone_time=None,
            refresh_interval=0.5,
            input=pt_input,
            output=pt_output,
        )
        app_instance.ttimeoutlen = 0.05
        app_instance.timeoutlen = 0.25
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
