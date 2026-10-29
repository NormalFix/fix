"""Slash command router and interactive prompt execution mixin for `FixInteractiveSession`."""

from __future__ import annotations

import asyncio
from datetime import datetime
import os
from pathlib import Path
import signal
import subprocess
from typing import Any, Callable, Optional
import time

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table

from fix import __version__
from fix.core.autonomy import AutonomyRouter, SandboxVerifier
from fix.core.context import ContextManager, resolve_project_root
from fix.core.evolution import EvolutionEngine
from fix.core.runtime import FixRuntime
from fix.llm.gateway import LLMGateway
from fix.memory.store import DualMemoryStore
from fix.tools.toolkit import WorkspaceToolkit
from fix.llm.catalog import PROVIDER_ORDER, PROVIDER_SPECS, resolve_provider_id
from fix.protocol.schema import StyleRule, TaskIntent, TaskPhase, TaskSession
from fix.tui.composer import render_codex_user_prompt_to
from fix.tui.theme import (
    console,
    CleanMarkdown,
    PRIMARY_TEXT_STYLE,
    SECONDARY_STEP_TEXT_STYLE,
    _sync_live_terminal_width,
    normalize_line_range_notation,
)
from fix.tui.tracker import CollapsibleProgressTracker, _render_session_result


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



class SlashCommandRouterMixin:
    """Mixin handling all interactive `/slash` commands and task execution turns."""

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

                    setattr(_replay_slash_turn, "_cacheable_block", True)
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
                        defer_digest=self._tui_mode,
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

        self._scroll_offset_from_bottom = 0
        self._screen_blocks.append(_render_turn_with_prompt)
        if len(self._screen_blocks) > 13:
            self._screen_blocks = [self._screen_blocks[0], *self._screen_blocks[-12:]]

        if self._tui_app is not None:
            try:
                self._tui_app.invalidate()
                if getattr(self._tui_app, "_is_running", False):
                    self._tui_app._redraw()
            except Exception:
                pass
            await asyncio.sleep(0)

        if not (self._tui_mode and not self._custom_console_passed):
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
        deferred_sha = getattr(session, "_deferred_digest_sha", None)
        if deferred_sha is not None and self.evolution is not None:
            delattr(session, "_deferred_digest_sha")
            self.evolution.digest_completed_task(
                session=session,
                head_sha=str(deferred_sha),
            )
        self._persist_current_conversation_session()
        return session

