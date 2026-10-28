"""Typer CLI entrypoints and subcommands (`fix`, `fix run`, `fix bug`, `fix ask`, `fix opt`, etc.)."""

from __future__ import annotations

import asyncio
from pathlib import Path
import subprocess
import sys
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.rule import Rule
from rich.table import Table

from fix import __version__
from fix.core.autonomy import AutonomyRouter
from fix.core.context import ContextManager, resolve_project_root
from fix.core.evolution import EvolutionEngine
from fix.core.runtime import FixRuntime
from fix.llm.gateway import LLMGateway
from fix.memory.store import DualMemoryStore
from fix.protocol.schema import StyleRule, TaskIntent, TaskSession
from fix.tools.toolkit import WorkspaceToolkit
from fix.tui.composer import KNOWN_SUBCOMMANDS
from fix.tui.session import FixInteractiveSession
from fix.tui.theme import _sync_live_terminal_width, console
from fix.tui.tracker import CollapsibleProgressTracker, _render_session_result

app = typer.Typer(
    name="fix",
    help="Fix: Interactive AI Engineering Partner (run `fix` in any project folder to launch the interactive CLI).",
    add_completion=False,
    invoke_without_command=True,
)


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


@app.callback()
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
