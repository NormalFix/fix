"""Interactive session controller (`FixInteractiveSession`) composing persistence, modals, commands, and full-screen TUI."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Callable

from rich.console import Console
from rich.markup import escape
from rich.rule import Rule
from rich.text import Text

from fix import __version__
from fix.core.autonomy import AutonomyRouter
from fix.core.context import ContextManager, RepoProfile, resolve_project_root
from fix.core.evolution import EvolutionEngine
from fix.core.looper import FixLooper
from fix.core.runtime import FixRuntime
from fix.llm.gateway import LLMGateway
from fix.memory.store import DualMemoryStore
from fix.protocol.schema import TaskSession
from fix.tools.toolkit import WorkspaceToolkit
from fix.tui.commands import SlashCommandRouterMixin, _build_stack
from fix.tui.fullscreen import FullscreenTuiMixin
from fix.tui.modals import ModalControllerMixin
from fix.tui.persistence import SessionPersistenceMixin
from fix.tui.theme import FIX_THEME, SECONDARY_STEP_TEXT_STYLE, _sync_live_terminal_width, _truncate_middle_for_width, console
from fix.tui.tracker import (
    CLAUDE_SPINNER_FRAMES,
    CollapsibleProgressTracker,
    format_claude_working_action,
    format_elapsed_hms,
)


class FixInteractiveSession(
    SessionPersistenceMixin,
    ModalControllerMixin,
    SlashCommandRouterMixin,
    FullscreenTuiMixin,
):
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
        """Copy arbitrary text to the OS clipboard (OSC 52 + silent pbcopy / wl-copy / xclip / xsel) and show a 3s toast."""
        if not text_to_copy:
            self.show_toast("暂无可复制内容", duration_sec=3.0)
            return False
        self._last_selected_text = text_to_copy

        # 1. Emit OSC 52 clipboard escape sequence first (<0.05ms, works over SSH/tmux/headless & modern terminals)
        try:
            import base64
            from prompt_toolkit.output.defaults import DummyOutput

            b64 = base64.b64encode(text_to_copy.encode("utf-8")).decode("ascii")
            if len(b64) <= 100000:
                osc_seq = (
                    f"\033Ptmux;\033\033]52;c;{b64}\a\033\\"
                    if os.environ.get("TMUX")
                    else f"\033]52;c;{b64}\a"
                )
                if self._tui_app is not None and getattr(self._tui_app, "output", None) is not None:
                    if not isinstance(self._tui_app.output, DummyOutput):
                        self._tui_app.output.write_raw(osc_seq)
                        self._tui_app.output.flush()
                else:
                    raw_out = getattr(sys, "__stdout__", sys.stdout)
                    if raw_out is not None and hasattr(raw_out, "isatty") and raw_out.isatty():
                        raw_out.write(osc_seq)
                        raw_out.flush()
        except Exception:
            pass

        # 2. Try native OS clipboard helpers ONLY when their display server is actually active,
        #    and ALWAYS silence stdout/stderr (DEVNULL) so failed tools (like wl-copy without Wayland)
        #    NEVER print error messages into the TUI input box or scroll/corrupt the terminal screen!
        candidate_cmds: list[list[str]] = []
        if sys.platform == "darwin":
            candidate_cmds.append(["pbcopy"])
        else:
            wayland_disp = os.environ.get("WAYLAND_DISPLAY", "").strip()
            if wayland_disp:
                xdg_dir = os.environ.get("XDG_RUNTIME_DIR", "").strip() or (
                    f"/run/user/{os.getuid()}" if hasattr(os, "getuid") else ""
                )
                wl_sock = (Path(xdg_dir) / wayland_disp) if xdg_dir else Path(wayland_disp)
                if wl_sock.exists():
                    candidate_cmds.append(["wl-copy"])
            if os.environ.get("DISPLAY", "").strip():
                candidate_cmds.append(["xclip", "-selection", "clipboard"])
                candidate_cmds.append(["xsel", "--clipboard", "--input"])

        for clip_cmd in candidate_cmds:
            if shutil.which(clip_cmd[0]):
                try:
                    proc = subprocess.Popen(
                        clip_cmd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                        close_fds=True,
                    )
                    if proc.stdin is not None:
                        proc.stdin.write(text_to_copy.encode("utf-8"))
                        proc.stdin.close()
                    try:
                        ret = proc.wait(timeout=0.15)
                        if ret == 0:
                            break
                    except subprocess.TimeoutExpired:
                        # xclip / wl-copy may stay alive in background serving clipboard requests; detach cleanly
                        break
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


    def _record_and_render_block(self, block_fn: Callable[[Console], None]) -> None:
        setattr(block_fn, "_cacheable_block", True)
        self._screen_blocks.append(block_fn)
        if len(self._screen_blocks) > 13:
            self._screen_blocks = [self._screen_blocks[0], *self._screen_blocks[-12:]]
        if not (self._tui_mode and not self._custom_console_passed):
            block_fn(self.console)

