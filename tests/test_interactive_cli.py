"""Tests for the Claude Code / Codex-style Interactive CLI REPL (`FixInteractiveSession`)."""

from __future__ import annotations

import subprocess
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from fix.cli import FixInteractiveSession
from fix.context import ContextManager, resolve_project_root
from fix.evolution import EvolutionEngine
from fix.llm import LLMResponse, ScriptedMockLLM
from fix.memory import DualMemoryStore
from fix.runtime import FixRuntime
from fix.schema import ClarifyOption, ProbeDecision, ReviewVerdict, TaskIntent, ToolCallRequest
from fix.tools import WorkspaceToolkit


def _init_sample_project(tmp_path: Path) -> Path:
    repo = tmp_path / "my_app"
    sub = repo / "src" / "sub"
    sub.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "dev@fix.local"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Fix Tester"], cwd=repo, check=True)
    (repo / "app.py").write_text("def greet(name: str) -> str:\n    return f'Hello, {name}'\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init app"], cwd=repo, check=True, capture_output=True)
    return repo


def test_resolve_project_root_from_nested_subdirectory(tmp_path: Path) -> None:
    repo = _init_sample_project(tmp_path)
    nested_dir = repo / "src" / "sub"
    # Even when launched inside `my_app/src/sub`, Fix automatically resolves `my_app` as root!
    resolved = resolve_project_root(nested_dir)
    assert resolved == repo.resolve()


@pytest.mark.asyncio
async def test_interactive_session_slash_commands_and_multi_turn_continuity(tmp_path: Path) -> None:
    repo = _init_sample_project(tmp_path)
    buf = StringIO()
    out_console = Console(file=buf, force_terminal=False, width=240)

    toolkit = WorkspaceToolkit(repo)
    memory = DualMemoryStore(repo, persona_db_path=tmp_path / "persona.db")
    context_mgr = ContextManager()
    mock_llm = ScriptedMockLLM(
        tool_Queue=[
            # Turn 1: user asks what the project does -> reads app.py
            LLMResponse(
                content="Reading app.py",
                tool_calls=[ToolCallRequest(name="read_file", arguments={"rel_path": "app.py"})],
            ),
            LLMResponse(
                content="`app.py:1-2` defines `greet(name: str) -> str`.",
                tool_calls=[],
            ),
            # Turn 2: user asks to update greet -> edits app.py
            LLMResponse(
                content="Updating greet in app.py",
                tool_calls=[
                    ToolCallRequest(
                        name="apply_patch",
                        arguments={
                            "rel_path": "app.py",
                            "search_block": "    return f'Hello, {name}'",
                            "replace_block": "    return f'Hi, {name}!'",
                        },
                    )
                ],
            ),
        ],
        structured_handlers={
            "ProbeDecision": [
                ProbeDecision(intent=TaskIntent.EXPLAIN, summary="Explain app.py"),
                ProbeDecision(intent=TaskIntent.DIRECT_EDIT, summary="Update greet in app.py"),
            ],
            "ReviewVerdict": ReviewVerdict(approved=True, confidence="HIGH"),
        },
    )
    evolution = EvolutionEngine(repo, memory=memory, llm=mock_llm)
    runtime = FixRuntime(llm=mock_llm, toolkit=toolkit, context_mgr=context_mgr, memory=memory, evolution=evolution)

    session = FixInteractiveSession(
        repo=repo / "src" / "sub",
        stack=(toolkit, memory, context_mgr, mock_llm, evolution, runtime),
        out_console=out_console,
    )
    session.render_welcome_banner()
    assert str(repo) in buf.getvalue()

    # 1. Turn 1: Natural language question
    keep = await session.handle_line("帮我看看这个项目是干嘛的")
    assert keep is True
    assert len(session.conversation_context) == 1
    assert "greet" in session.conversation_context[0]["assistant"]
    # Verify live tool event `Read(app.py:...)` was printed
    assert "• Explored" in buf.getvalue() and "Read app.py" in buf.getvalue()

    # 2. Turn 2: Natural language edit that relies on multi-turn continuity
    keep = await session.handle_line("把打招呼改成 Hi, {name}!")
    assert keep is True
    assert len(session.conversation_context) == 2
    assert "Hi, {name}!" in (repo / "app.py").read_text(encoding="utf-8")

    # Verify Turn 2 prompt included Turn 1 in `Recent Session Conversation Context`
    second_turn_sys = mock_llm.recorded_messages[-1][0]["content"]
    assert "Recent Session Conversation Context" in second_turn_sys
    assert "帮我看看这个项目是干嘛的" in second_turn_sys

    # 3. Slash command `/diff` shows modified app.py
    await session.handle_line("/diff")
    assert "+    return f'Hi, {name}!'" in buf.getvalue()

    # 4. Slash command `/undo` reverts app.py back to `Hello, {name}`!
    await session.handle_line("/undo")
    assert "Hello, {name}" in (repo / "app.py").read_text(encoding="utf-8")

    # 5. Slash commands `/help`, `/model`, `/clear`, `/exit`
    await session.handle_line("/help")
    await session.handle_line("/model mimo-v2.6-flash mimo-v2.6-pro")
    await session.handle_line("/clear")
    assert len(session.conversation_context) == 0

    should_continue = await session.handle_line("/exit")
    assert should_continue is False


@pytest.mark.asyncio
async def test_interactive_clarification_flow_in_repl(tmp_path: Path) -> None:
    repo = _init_sample_project(tmp_path)
    buf = StringIO()
    out_console = Console(file=buf, force_terminal=False, width=240)

    toolkit = WorkspaceToolkit(repo)
    memory = DualMemoryStore(repo, persona_db_path=tmp_path / "persona.db")
    context_mgr = ContextManager()

    mock_llm = ScriptedMockLLM(
        tool_Queue=[
            LLMResponse(
                content="Implemented JWT auth as selected.",
                tool_calls=[
                    ToolCallRequest(
                        name="write_new_file",
                        arguments={"rel_path": "auth.py", "content": "TOKEN_FORMAT = 'JWT'\n"},
                    )
                ],
            )
        ],
        structured_handlers={
            "ProbeDecision": ProbeDecision(
                intent=TaskIntent.FEATURE,
                summary="Add auth module",
                needs_clarification=True,
                clarifications=[
                    ClarifyOption(
                        question="Which auth token format should we use?",
                        options=["JWT", "Opaque Session Token"],
                        recommended="JWT",
                    )
                ],
            ),
            "ReviewVerdict": ReviewVerdict(approved=True, confidence="HIGH"),
        },
    )
    evolution = EvolutionEngine(repo, memory=memory, llm=mock_llm)
    runtime = FixRuntime(llm=mock_llm, toolkit=toolkit, context_mgr=context_mgr, memory=memory, evolution=evolution)

    session = FixInteractiveSession(
        repo=repo,
        stack=(toolkit, memory, context_mgr, mock_llm, evolution, runtime),
        out_console=out_console,
    )

    # Simulate user pressing Enter to accept the recommended option ("JWT")
    await session.handle_line("给项目加上登录鉴权", input_fn=lambda _prompt: "")
    assert len(session.conversation_context) == 1
    assert "Implemented JWT auth" in session.conversation_context[0]["assistant"]


@pytest.mark.asyncio
async def test_collapsed_tool_summary_telemetry_footer_and_steps_command(tmp_path: Path) -> None:
    repo = _init_sample_project(tmp_path)
    buf = StringIO()
    out_console = Console(file=buf, force_terminal=False, width=240)

    toolkit = WorkspaceToolkit(repo)
    memory = DualMemoryStore(repo, persona_db_path=tmp_path / "persona.db")
    context_mgr = ContextManager()

    mock_llm = ScriptedMockLLM(
        tool_Queue=[
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(name="search_code", arguments={"pattern": "greet"}),
                    ToolCallRequest(name="read_file", arguments={"rel_path": "app.py"}),
                ],
                prompt_tokens=500,
                completion_tokens=60,
                total_tokens=560,
            ),
            LLMResponse(
                content="### 功能地图\n- `greet` 函数位于 `[app.py:1-2]`。",
                tool_calls=[],
                prompt_tokens=650,
                completion_tokens=120,
                total_tokens=770,
            ),
        ],
        structured_handlers={
            "ProbeDecision": ProbeDecision(intent=TaskIntent.EXPLAIN, summary="Analyze greet"),
        },
    )
    evolution = EvolutionEngine(repo, memory=memory, llm=mock_llm)
    runtime = FixRuntime(llm=mock_llm, toolkit=toolkit, context_mgr=context_mgr, memory=memory, evolution=evolution)

    session = FixInteractiveSession(
        repo=repo,
        stack=(toolkit, memory, context_mgr, mock_llm, evolution, runtime),
        out_console=out_console,
    )

    await session.handle_line("分析 greet 的工作机制")
    rendered = buf.getvalue()

    # 1. Verify Codex-style `• Explored` tree is rendered and NO redundant `Fix 分析与执行结果` banner exists
    assert "• Explored" in rendered
    assert "└ Search greet in ." in rendered
    assert "Read app.py" in rendered
    assert "Fix 分析与执行结果" not in rendered

    # 2. Verify full analysis Markdown is rendered cleanly with `#` / `###` heading
    assert "### 功能地图" in rendered
    assert "[app.py:1-2]" in rendered

    # 3. Verify telemetry footer (duration & token breakdown) is rendered
    assert "⏱ 耗时:" in rendered
    assert "🪙 Token 消耗:" in rendered
    assert "LLM 调用" in rendered

    # 4. Verify `/steps` expands the full tool call log on demand
    await session.handle_line("/steps")
    after_steps = buf.getvalue()
    assert "上一次任务工具调用明细" in after_steps
    assert "Search('greet')" in after_steps
    assert "Read(app.py:1-500)" in after_steps


@pytest.mark.asyncio
async def test_no_black_background_on_code_and_dynamic_window_resize_rerender(tmp_path: Path) -> None:
    """
    Verifies the two UX requirements:
    1. Inline English/code spans (`OrderStateView`, `OrderStateCoordinator.kt:53`) and fenced code blocks
       are styled using foreground colors ONLY, with ZERO black background escape codes (`;40m`, `[40m`, `48;2;`).
    2. Output uses borderless horizontal rules (no vertical `│` box walls that break line reflow) and
       `rerender_for_terminal_width` dynamically re-renders the entire screen history & Markdown tables
       to match the new terminal width when the window is dragged larger or smaller!
    """
    from fix.cli import create_fix_console

    repo = _init_sample_project(tmp_path)
    buf = StringIO()
    out_console = create_fix_console(file=buf, width=80, force_terminal=True)
    out_console.no_color = False

    toolkit = WorkspaceToolkit(repo)
    memory = DualMemoryStore(repo, persona_db_path=tmp_path / "persona.db")
    context_mgr = ContextManager()

    mock_llm = ScriptedMockLLM(
        tool_Queue=[
            LLMResponse(
                content=(
                    "### 一、架构定位\n"
                    "`OrderStateView` 实现了 `OrderStateCoordinator.Host` 接口 `[OrderStateView.kt:70]`。\n\n"
                    "```kotlin\n"
                    "override fun onVoiceShow(bean: VoiceBean, animateEntry: Boolean)\n"
                    "```\n\n"
                    "| 层级 | 类名 | 核心职责 |\n"
                    "| :--- | :--- | :--- |\n"
                    "| 状态机 | `OrderStateCoordinator` | 唯一权威：三来源优先级、展示计时与跨宿主同步 |\n"
                ),
                tool_calls=[],
                prompt_tokens=800,
                completion_tokens=200,
                total_tokens=1000,
            ),
        ],
        structured_handlers={
            "ProbeDecision": ProbeDecision(intent=TaskIntent.EXPLAIN, summary="Analyze OrderStateView"),
        },
    )
    evolution = EvolutionEngine(repo, memory=memory, llm=mock_llm)
    runtime = FixRuntime(llm=mock_llm, toolkit=toolkit, context_mgr=context_mgr, memory=memory, evolution=evolution)

    session = FixInteractiveSession(
        repo=repo,
        stack=(toolkit, memory, context_mgr, mock_llm, evolution, runtime),
        out_console=out_console,
    )
    session.render_welcome_banner()
    await session.handle_line("深入分析 OrderStateView")

    ansi_out_80 = buf.getvalue()
    # 1. Verify ZERO black background ANSI codes (`;40m`, `\x1b[40m`, `48;2;`)
    assert ";40m" not in ansi_out_80
    assert "\x1b[40m" not in ansi_out_80
    assert "48;2;" not in ansi_out_80
    # Verify soft muted foreground (`#8ab4f8` -> `38;5;111m` or `38;2;138;180;248m`, NOT glaring `\x1b[1;96m`) IS applied
    assert "\x1b[1;96m" not in ansi_out_80
    assert "OrderStateView\x1b[0m" in ansi_out_80
    # Verify no vertical Panel side walls `│`
    assert "╭─" not in ansi_out_80
    assert "╰─" not in ansi_out_80

    # 2. Simulate dragging the terminal window wider to 130 columns -> verify dynamic re-render!
    buf.truncate(0)
    buf.seek(0)
    session.rerender_for_terminal_width(new_width=130, clear_terminal=False)
    ansi_out_130 = buf.getvalue()
    assert out_console.width == 130
    assert "OrderStateView" in ansi_out_130
    assert "唯一权威：三来源优先级、展示计时与跨宿主同步" in ansi_out_130


@pytest.mark.asyncio
async def test_live_resize_adaptation_during_active_command_execution(tmp_path: Path) -> None:
    """
    Verifies that WHILE a command is actively executing (with tool calls in flight):
    1. Resizing the terminal window from 120 -> 60 -> 140 columns triggers `_redraw_above_live_on_resize`
       and `CollapsibleProgressTracker._build_renderable(new_cols)`.
    2. Both the startup banner + in-flight `fix (my_app) ❯ <prompt>` header AND the active
       `CollapsibleProgressTracker` lines (including long file paths like
       `Read(app/src/main/java/com/acme/commerce/ui/widget/OrderStateView.kt:549-668)`)
       dynamically adapt to `60` cols (middle-truncated so no line wraps) and `140` cols (full path shown)!
    """
    from fix.cli import create_fix_console

    repo = _init_sample_project(tmp_path)
    buf = StringIO()
    out_console = create_fix_console(file=buf, width=120, force_terminal=False)

    toolkit = WorkspaceToolkit(repo)
    memory = DualMemoryStore(repo, persona_db_path=tmp_path / "persona.db")
    context_mgr = ContextManager()

    session_ref: FixInteractiveSession | None = None
    captured_60 = ""
    captured_140 = ""

    def step_cb(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> LLMResponse:
        nonlocal captured_60, captured_140
        assert session_ref is not None
        # Step 1: emit a long Read tool call, then on Step 2 simulate terminal resize mid-execution!
        if len(mock_llm.recorded_messages) == 1:
            return LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        name="read_file",
                        arguments={
                            "rel_path": "app/src/main/java/com/acme/commerce/ui/widget/OrderStateView.kt",
                            "start_line": 549,
                            "end_line": 668,
                        },
                    )
                ],
            )
        # Now we are mid-execution with 1 long tool call recorded on `_active_tracker`!
        assert session_ref._active_tracker is not None
        buf.truncate(0)
        buf.seek(0)
        session_ref.rerender_for_terminal_width(new_width=60, clear_terminal=False)
        captured_60 = buf.getvalue()

        buf.truncate(0)
        buf.seek(0)
        session_ref.rerender_for_terminal_width(new_width=140, clear_terminal=False)
        captured_140 = buf.getvalue()

        return LLMResponse(
            content="Completed analysis after live window resize.",
            tool_calls=[],
        )

    mock_llm = ScriptedMockLLM(
        step_callback=step_cb,
        structured_handlers={
            "ProbeDecision": ProbeDecision(intent=TaskIntent.EXPLAIN, summary="Analyze OrderStateView"),
        },
    )
    evolution = EvolutionEngine(repo, memory=memory, llm=mock_llm)
    runtime = FixRuntime(llm=mock_llm, toolkit=toolkit, context_mgr=context_mgr, memory=memory, evolution=evolution)

    session_ref = FixInteractiveSession(
        repo=repo,
        stack=(toolkit, memory, context_mgr, mock_llm, evolution, runtime),
        out_console=out_console,
    )
    session_ref.render_welcome_banner()
    await session_ref.handle_line("深入分析 OrderStateView")

    # Verify at width=60: banner, in-flight prompt, and middle-truncated tool call were rendered
    assert "fix (my_app) ❯ 深入分析 OrderStateView" in captured_60
    assert "Fix 正在执行" in captured_60 and "• Explored" in captured_60
    assert "..." in captured_60  # Long path was middle-truncated to fit 60 columns!
    for line in captured_60.splitlines():
        assert len(line) <= 60

    # Verify at width=140: full path is visible without truncation
    assert "fix (my_app) ❯ 深入分析 OrderStateView" in captured_140
    assert "• Explored" in captured_140 and "└ Read OrderStateView.kt" in captured_140 and "Fix 正在执行:" in captured_140


@pytest.mark.asyncio
async def test_codex_style_narrative_explored_ran_grouping_and_ctrl_c_exit(tmp_path: Path) -> None:
    """
    Verifies the Codex-style UX flow:
    1. Intermediate reasoning narratives (`我会先从项目里定位...`, `已确认...接下来...`) are printed before tool batches.
    2. Consecutive `Read` calls in a batch are merged onto a single line (`Read OrderStateView.kt, OrderStateViewV2.kt, attrs.xml`)
       inside `• Explored` alongside `Search ... in ...`.
    3. Command execution is rendered as `• Ran <cmd>` with indented preview lines (`  └ ...`).
    4. Final output starts directly with `# Title` Markdown without any `Fix 分析与执行结果 (Fix Response)` banner.
    """
    from fix.cli import create_fix_console, format_explored_group_lines

    # Direct unit check of format_explored_group_lines
    grouped = format_explored_group_lines(
        [
            ("Search", "Search OrderStateView in ."),
            ("Read", "OrderStateView.kt"),
            ("Read", "OrderStateViewV2.kt"),
            ("Read", "attrs.xml"),
            ("Search", "Search ThinkingListener in java"),
        ]
    )
    assert grouped == [
        "  └ Search OrderStateView in .",
        "    Read OrderStateView.kt, OrderStateViewV2.kt, attrs.xml",
        "    Search ThinkingListener in java",
    ]

    repo = _init_sample_project(tmp_path)
    buf = StringIO()
    out_console = create_fix_console(file=buf, width=120, force_terminal=False)

    toolkit = WorkspaceToolkit(repo)
    memory = DualMemoryStore(repo, persona_db_path=tmp_path / "persona.db")
    context_mgr = ContextManager()

    mock_llm = ScriptedMockLLM(
        tool_Queue=[
            LLMResponse(
                content="我会先从项目里定位 greet 的实现与引用关系，再整理它的功能与逻辑地图。",
                tool_calls=[
                    ToolCallRequest(name="search_code", arguments={"pattern": "greet"}),
                    ToolCallRequest(name="read_file", arguments={"rel_path": "app.py"}),
                    ToolCallRequest(name="read_file", arguments={"rel_path": "app.py", "start_line": 1, "end_line": 10}),
                ],
            ),
            LLMResponse(
                content="已确认 greet 的核心签名，接下来运行命令检查文件状态。",
                tool_calls=[
                    ToolCallRequest(name="run_command", arguments={"cmd": "echo 'hello_from_ran'"}),
                ],
            ),
            LLMResponse(
                content="# greet 深度分析\n\n## 结论\n\n`greet` 位于 `(app.py:1)`。",
                tool_calls=[],
            ),
        ],
        structured_handlers={
            "ProbeDecision": ProbeDecision(intent=TaskIntent.EXPLAIN, summary="Analyze greet"),
        },
    )
    evolution = EvolutionEngine(repo, memory=memory, llm=mock_llm)
    runtime = FixRuntime(llm=mock_llm, toolkit=toolkit, context_mgr=context_mgr, memory=memory, evolution=evolution)

    session = FixInteractiveSession(
        repo=repo,
        stack=(toolkit, memory, context_mgr, mock_llm, evolution, runtime),
        out_console=out_console,
    )
    await session.handle_line("分析 greet")
    out = buf.getvalue()

    assert "✦ 我会先从项目里定位 greet 的实现与引用关系" in out and "✓ 搞定" in out
    assert "• Explored\n  └ Search greet in .\n    Read app.py" in out
    assert "已确认 greet 的核心签名，接下来运行命令检查文件状态。" in out
    assert "• Ran echo 'hello_from_ran'\n  └ hello_from_ran" in out
    assert "# greet 深度分析" in out
    assert "## 结论" in out
    assert "Fix 分析与执行结果" not in out


@pytest.mark.asyncio
async def test_fullscreen_tui_layout_three_state_ctrl_c_and_3s_toast(tmp_path: Path) -> None:
    """
    Verifies the 3 new requirements from the user's UI sketch:
    1. 3-State Ctrl+C machine (`handle_ctrl_c_action`):
       - State 1: When a task is actively running -> cancels ONLY the running task (`cancel_task`),
         sets status to `已取消` and shows a 3s toast (`已取消当前任务`), NEVER exiting the Agent!
       - State 2: When idle and input box has text -> clears ONLY the input box (`clear_input`)
         and shows a 3s toast (`已清空输入`), NEVER exiting the Agent!
       - State 3: When idle AND input box is empty -> exits the Agent (`exit_agent`).
    2. Dual-Purpose Status Row (`format_status_row`):
       - Left side shows `当前Agent的状态 · 任务耗时` (`● 就绪 · X.Xs` or `⠋ ... · X.Xs`).
       - Right side shows transient user operation toast (`✓ 复制成功`, `已取消当前任务`, etc.) for ONLY 3.0 seconds,
         automatically disappearing once `now >= toast_expires_at`!
    3. Full-Screen Alternate Screen TUI (`build_fullscreen_tui_app`):
       - Uses `full_screen=True` (entering Alternate Screen Buffer `\033[?1049h` and restoring original terminal on exit).
       - Minimalist main content stream without noisy telemetry walls when `_tui_mode` is active.
    """
    import asyncio
    import time
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    repo = _init_sample_project(tmp_path)
    toolkit = WorkspaceToolkit(repo)
    memory = DualMemoryStore(repo, persona_db_path=tmp_path / "persona.db")
    context_mgr = ContextManager()

    # Create a slow mock LLM step so we can test pressing Ctrl+C while the task is actively running
    async def slow_task_coro() -> None:
        await asyncio.sleep(10.0)

    mock_llm = ScriptedMockLLM(
        tool_Queue=[
            LLMResponse(
                content="# Minimalist Report\n\nClean output.",
                tool_calls=[],
            )
        ],
        structured_handlers={
            "ProbeDecision": ProbeDecision(intent=TaskIntent.EXPLAIN, summary="Quick check"),
        },
    )
    evolution = EvolutionEngine(repo, memory=memory, llm=mock_llm)
    runtime = FixRuntime(llm=mock_llm, toolkit=toolkit, context_mgr=context_mgr, memory=memory, evolution=evolution)

    session = FixInteractiveSession(
        repo=repo,
        stack=(toolkit, memory, context_mgr, mock_llm, evolution, runtime),
    )

    with create_pipe_input() as pipe_in:
        tui_app = session.build_fullscreen_tui_app(pt_input=pipe_in, pt_output=DummyOutput())
        assert tui_app.full_screen is True  # Uses Alternate Screen Buffer!

        # --- Test State 1: Task is running -> Ctrl+C cancels task ONLY, does NOT exit Agent ---
        session._running_task = asyncio.create_task(slow_task_coro())
        await asyncio.sleep(0.02)
        action_running = session.handle_ctrl_c_action("")
        assert action_running == "cancel_task"
        assert session._agent_state_label == "已取消"
        assert session.get_right_toast_text() == "已取消当前任务"
        try:
            await session._running_task
        except asyncio.CancelledError:
            pass
        session._running_task = None

        # --- Test State 2: Idle, input box has text -> Ctrl+C clears input ONLY, does NOT exit Agent ---
        action_non_empty = session.handle_ctrl_c_action("unfinished prompt text")
        assert action_non_empty == "clear_input"
        assert session.get_right_toast_text() == "已清空输入"

        # --- Test State 3: Idle, input box is empty -> Ctrl+C exits Agent ---
        action_empty_idle = session.handle_ctrl_c_action("")
        assert action_empty_idle == "exit_agent"

        # --- Test 3-Second Transient Toast Expiration & Status Row Flush-Right Alignment ---
        from rich.cells import cell_len

        t0 = time.time()
        session.show_toast("✓ 复制成功", duration_sec=3.0)
        row_active = session.format_status_row(width=80, now=t0 + 1.0)
        assert "● 已取消" in row_active
        assert row_active.endswith("✓ 复制成功 ")
        assert cell_len(row_active) == 80  # Strictly flush-right at terminal column 80!

        # After 3.1 seconds, the right-side toast automatically disappears!
        row_expired = session.format_status_row(width=80, now=t0 + 3.1)
        assert "● 已取消" in row_expired
        assert "✓ 复制成功" not in row_expired

        # Run a normal turn in TUI mode and verify status updates to `● 就绪 · X.Xs`
        await session.handle_line("分析 app.py")
        assert session._agent_state_label == "就绪"
        assert "● 就绪 · " in session.get_left_status_text()


def test_normalize_line_range_notation_aligns_parens_and_hyphen() -> None:
    """
    Verifies that `normalize_line_range_notation` converts full-width parentheses/colons and
    en-dash/wave/underscore separators in line ranges into tight ASCII `(:357-374)` and `(98-168)`.
    """
    from fix.cli import normalize_line_range_notation

    assert normalize_line_range_notation("（：357-374）") == "(:357-374)"
    assert normalize_line_range_notation("（ 98_168）") == "(98-168)"
    assert normalize_line_range_notation("(: 357~374 )") == "(:357-374)"
    assert normalize_line_range_notation("还有两段被截断（98–168 属性声明段、541–689 核心动画逻辑段）") == (
        "还有两段被截断(98-168 属性声明段、541-689 核心动画逻辑段)"
    )
    assert normalize_line_range_notation("行尾省略（`:357-374`）") == "行尾省略(`:357-374`)"
    assert normalize_line_range_notation("OrderStateView.kt:98–168") == "OrderStateView.kt:98-168"


def test_fullscreen_tui_divider_and_layout_dynamically_fill_live_terminal_width(tmp_path: Path, monkeypatch) -> None:
    """Verify that default TUI session (with _tui_Sink) dynamically fills 100% of live OS terminal width on resize."""
    import fix.cli as cli_mod
    from rich.cells import cell_len

    live_dims = {"cols": 148, "rows": 36}
    monkeypatch.setattr(cli_mod, "_query_os_terminal_size", lambda: (live_dims["cols"], live_dims["rows"]))

    sess = FixInteractiveSession(repo=tmp_path)
    app = sess.build_fullscreen_tui_app()

    # Extract divider_row and status_row windows from HSplit
    hsplit = app.layout.container.content
    status_win = hsplit.children[1]
    divider_win = hsplit.children[2]

    div_frags_148 = divider_win.content.text()
    assert div_frags_148 == [("#0D9488", "─" * 148)]

    sess.show_toast("✓ 复制成功", duration_sec=3.0)
    status_frags_148 = status_win.content.text()
    total_cells_148 = sum(cell_len(t) for _, t in status_frags_148)
    assert total_cells_148 == 148

    # Now simulate resizing terminal window to 186 columns
    live_dims["cols"] = 186
    div_frags_186 = divider_win.content.text()
    assert div_frags_186 == [("#0D9488", "─" * 186)]
    status_frags_186 = status_win.content.text()
    assert sum(cell_len(t) for _, t in status_frags_186) == 186


def test_two_tier_visual_hierarchy_primary_prompt_and_result_vs_secondary_steps() -> None:
    """
    Verifies the user's visual hierarchy design (`t30u`):
    1. Primary Tier (User Prompt `> ...` + Agent Final Result `CleanMarkdown` body/lists/tables):
       Share the exact same solid `bold` primary foreground (`style == "bold"`).
    2. Secondary Tier (Agent Execution Operation Prompts: `✦` narrative, `• Explored`, `Search/Read/List`, `• Ran`):
       Rendered in non-bold secondary tone (`#788596` for step text/targets, non-bold `#0d9488`/`#0ea5e9`/`#10b981`/`#f97316` for step labels),
       so they remain crisp and readable while being visibly one step lighter than the User Prompt and Final Result.
    """
    from fix.cli import CleanMarkdown, CollapsibleProgressTracker, _render_blocks_to_pt_fragments

    def render_turn(c):
        c.print("[bold #F97316]>[/bold #F97316] 请深入分析 OrderStateView")
        CollapsibleProgressTracker._render_stream_block_to(
            c,
            {"type": "narrative", "text": "我会先定位 OrderStateView 的主类与引用关系"},
        )
        CollapsibleProgressTracker._render_stream_block_to(
            c,
            {"type": "explored", "lines": ["  └ Search OrderStateView in .", "    Read OrderStateView.kt"]},
        )
        CollapsibleProgressTracker._render_stream_block_to(
            c,
            {"type": "ran", "cmd": "pytest -q", "preview_lines": ["  └ 32 passed"]},
        )
        c.print(CleanMarkdown("## 结论\n\n这是最终输出的分析结论正文。\n\n- 列表结论项一"))

    frags = _render_blocks_to_pt_fragments([render_turn], width=120)
    frag_map = {txt.strip(): st for st, txt in frags if txt.strip()}

    # 1. Primary Tier: User prompt and Agent final result body/list items share exact same non-bold native primary foreground (`""`)
    assert frag_map["请深入分析 OrderStateView"] == ""
    assert frag_map["这是最终输出的分析结论正文。"] == ""
    assert frag_map["列表结论项一"] == ""

    # 2. Secondary Tier: Agent execution operation prompts are non-bold and use softer #788596 slate for text/targets
    assert frag_map["我会先定位 OrderStateView 的主类与引用关系"] == "#788596"
    assert "bold" not in frag_map["我会先定位 OrderStateView 的主类与引用关系"]
    assert frag_map["OrderStateView"] == "#788596"
    assert frag_map["OrderStateView.kt"] == "#788596"
    assert frag_map["pytest -q"] == "#788596"
    assert frag_map["Explored"] == "#0d9488"
    assert "bold" not in frag_map["Explored"]


def test_fullscreen_tui_isolated_alternate_screen_mouse_scroll_and_no_raw_clear(tmp_path: Path, monkeypatch) -> None:
    """
    Verifies (`t31u`):
    1. `build_fullscreen_tui_app` enables both `full_screen=True` AND `mouse_support=True`,
       so trackpad/mouse wheel events scroll Fix's own main content viewport (`_scroll_offset_from_bottom`)
       instead of scrolling the outer macOS terminal scrollback (`user@host % fix`).
    2. Mouse wheel `SCROLL_UP` / `SCROLL_DOWN` on the main content control updates `_scroll_offset_from_bottom`.
    3. While `_tui_mode` is active, `_clear_real_terminal_viewport()` and `_redraw_above_live_on_resize()`
       never write raw `\033[2J` to stdout (preventing any terminal buffer bleed).
    """
    from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType, MouseModifier
    from prompt_toolkit.layout.containers import Point

    sess = FixInteractiveSession(repo=tmp_path)
    app = sess.build_fullscreen_tui_app()
    assert app.full_screen is True
    assert app.mouse_support() is True  # Button-event SGR mouse tracking enabled for multi-screen drag selection!

    # Add enough lines so scrolling has room
    sess._record_and_render_block(lambda c: [c.print(f"Line {i}") for i in range(60)])
    hsplit = app.layout.container.content
    main_win = hsplit.children[0]
    ctrl = main_win.content

    assert sess._scroll_offset_from_bottom == 0
    ctrl.mouse_handler(
        MouseEvent(
            position=Point(x=5, y=5),
            event_type=MouseEventType.SCROLL_UP,
            button=MouseButton.NONE,
            modifiers=frozenset(),
        )
    )
    assert sess._scroll_offset_from_bottom == 3

    ctrl.mouse_handler(
        MouseEvent(
            position=Point(x=5, y=5),
            event_type=MouseEventType.SCROLL_DOWN,
            button=MouseButton.NONE,
            modifiers=frozenset(),
        )
    )
    assert sess._scroll_offset_from_bottom == 0


def test_fullscreen_tui_pasted_long_and_multiline_prompt_auto_expands_input_box(tmp_path: Path, monkeypatch) -> None:
    """
    Verifies (`t35u`):
    1. When the user pastes a long single-line or multi-line prompt into the TUI input box,
       `input_window` dynamically expands its height (`1 -> N` rows) so the full pasted text is visible
       instead of being clipped to a fixed 1-row window (`Dimension.exact(1)`).
    2. `main_content_window` correspondingly reduces `avail_height` (`rows - 3 - in_rows`) so the total
       layout height remains strictly equal to `rows` (zero vertical overflow or regression).
    """
    import fix.cli as cli_mod

    monkeypatch.setattr(cli_mod, "_query_os_terminal_size", lambda: (80, 30))
    sess = FixInteractiveSession(repo=tmp_path)
    app = sess.build_fullscreen_tui_app()

    hsplit = app.layout.container.content
    input_win = hsplit.children[3]
    buf = input_win.content.buffer

    # 1. Empty input -> height is 1 row
    dim_empty = input_win.height() if callable(input_win.height) else input_win.height
    assert dim_empty.preferred == 1

    # 2. Paste a long multi-line prompt -> input box automatically expands to show all lines!
    pasted = (
        "请深入分析 OrderStateView，不仅要输出他的工作机制、功能地图、业务逻辑地图、代码逻辑地图，"
        "还要透过现象看本质，帮我找出所有架构腐化点和用户体验优化点\n"
        "第二行补充要求：必须包含准确的代码行号区间(:start-end)"
    )
    buf.text = pasted
    dim_pasted = input_win.height() if callable(input_win.height) else input_win.height
    assert dim_pasted.preferred >= 3

    # 3. Clearing input (Ctrl+C state 2) shrinks input box back to 1 row
    buf.reset()
    dim_after_clear = input_win.height() if callable(input_win.height) else input_win.height
    assert dim_after_clear.preferred == 1


def test_fullscreen_tui_very_long_cjk_prompt_renders_100_percent_without_truncation(tmp_path: Path, monkeypatch) -> None:
    """
    Verifies (`t36u`):
    1. Pasting or typing a very long CJK + English prompt (e.g. 400-600 chars / 10-18 visual rows)
       expands `input_window` up to `rows - 5` rows and renders 100% of the characters on screen
       without clipping either the head or the tail (`【末尾标记END】`), even on odd/even column widths
       where 2-cell CJK characters wrap early at column `cols - 1`.
    2. Even when an ultra-long prompt exceeds `rows - 5` rows and scrolls inside `input_window`,
       our patched `UIContent.get_height_for_line` calculates CJK wrap height character-by-character
       so the tail where the cursor sits is NEVER cut off.
    """
    import asyncio
    from prompt_toolkit.layout.mouse_handlers import MouseHandlers
    from prompt_toolkit.layout.screen import Screen, WritePosition
    import fix.cli as cli_mod

    async def _run_check() -> None:
        for cols in (80, 81, 95, 100):
            rows = 28
            monkeypatch.setattr(cli_mod, "_query_os_terminal_size", lambda c=cols, r=rows: (c, r))
            sess = FixInteractiveSession(repo=tmp_path)
            app = sess.build_fullscreen_tui_app()
            hsplit = app.layout.container.content
            input_win = hsplit.children[3]
            buf = input_win.content.buffer

            # Case 1: 12-row CJK+English prompt -> 100% of text from start to END must be visible at once!
            long_prompt = (
                "【开头START】"
                + "".join(f"第{i:02d}句：请深入分析 OrderStateView.kt 的工作机制与架构腐化点；" for i in range(1, 7))
                + "【末尾标记END】"
            )
            buf.text = long_prompt
            buf.cursor_position = len(long_prompt)

            screen = Screen()
            app.layout.container.write_to_screen(
                screen,
                MouseHandlers(),
                WritePosition(0, 0, cols, rows),
                "",
                erase_bg=True,
                z_index=None,
            )
            in_h = input_win.height().preferred
            start_y = rows - 2 - in_h
            rendered = "".join(
                "".join(screen.data_buffer[y][x].char for x in range(3, cols)).rstrip()
                for y in range(start_y, rows - 2)
            )
            assert long_prompt.replace(" ", "") == rendered.replace(" ", "")
            assert "【开头START】" in rendered
            assert "【末尾标记END】" in rendered

            # Case 2: Ultra-long 35-row prompt (> rows - 5) -> scrolls cleanly inside input_window and never cuts off tail!
            ultra_long = (
                "".join(f"段落{i:02d}：请深入分析 OrderStateView.kt 的工作机制、功能地图、业务逻辑地图；" for i in range(1, 35))
                + "【超长末尾TAIL_OK】"
            )
            buf.text = ultra_long
            buf.cursor_position = len(ultra_long)
            screen2 = Screen()
            app.layout.container.write_to_screen(
                screen2,
                MouseHandlers(),
                WritePosition(0, 0, cols, rows),
                "",
                erase_bg=True,
                z_index=None,
            )
            in_h2 = input_win.height().preferred
            rendered2 = "".join(
                "".join(screen2.data_buffer[y][x].char for x in range(3, cols - 1)).rstrip()
                for y in range(rows - 2 - in_h2, rows - 2)
            )
            assert "【超长末尾TAIL_OK】" in rendered2.replace(" ", "")

    asyncio.run(_run_check())


def test_fullscreen_tui_pasted_text_cursor_editing_and_dynamic_right_scrollbar(tmp_path: Path, monkeypatch) -> None:
    """
    Verifies (`t37u`):
    1. Pasted text in `input_buffer` supports full cursor movement (`left`, `right`, `up`, `down`, `home`, `end`,
       word-left/right, newline insertion, character insertion/deletion in the middle of pasted text).
    2. Right-side scrollbar (`┃` thumb / `│` track on the rightmost column):
       - When NOT scrolled (main viewport at bottom `_scroll_offset_from_bottom == 0`, and input box fits within
         `max_input_rows`), NO scrollbar is rendered on the right edge (100% clean normal display).
       - When scrolled (either the main viewport is scrolled up OR an ultra-long pasted input scrolls inside
         `input_window`), the rightmost column (`col = cols - 1`) displays the scrollbar (`┃` / `│`).
    """
    import asyncio
    from prompt_toolkit.layout.mouse_handlers import MouseHandlers
    from prompt_toolkit.layout.screen import Screen, WritePosition
    import fix.cli as cli_mod

    async def _run_verify() -> None:
        cols, rows = 80, 24
        monkeypatch.setattr(cli_mod, "_query_os_terminal_size", lambda: (cols, rows))
        sess = FixInteractiveSession(repo=tmp_path)

        # Add 40 lines of history to main viewport so it CAN be scrolled
        def _history_block(c):
            for i in range(40):
                c.print(f"History output line {i:02d}")
        sess._screen_blocks.append(_history_block)

        app = sess.build_fullscreen_tui_app()
        hsplit = app.layout.container.content
        input_win = hsplit.children[3]
        buf = input_win.content.buffer

        # 1. Paste a 3-row prompt into input box (fits without scrolling input box, main viewport at bottom)
        pasted = (
            "第一段：请深入分析 OrderStateView 的工作机制与业务逻辑地图；"
            "第二段：帮我找出架构腐化点和用户体验优化点；"
            "第三段：输出完整行号引用。"
        )
        buf.text = pasted
        buf.cursor_position = len(pasted)

        screen_normal = Screen()
        app.layout.container.write_to_screen(
            screen_normal,
            MouseHandlers(),
            WritePosition(0, 0, cols, rows),
            "",
            erase_bg=True,
            z_index=None,
        )
        # Verify NO scrollbar in rightmost column when not scrolled
        right_edge_normal = "".join(screen_normal.data_buffer[y][cols - 1].char for y in range(rows))
        assert "┃" not in right_edge_normal and "│" not in right_edge_normal

        # 2. Verify cursor movement & in-place editing inside pasted text!
        kb_map = {tuple(b.keys): b.handler for b in app.key_bindings.bindings}

        class _DummyKeyEvent:
            def __init__(self) -> None:
                self.app = app
                self.arg = 1
                self.current_buffer = buf

        ev = _DummyKeyEvent()
        # Move cursor up 1 visual row inside the wrapped pasted prompt
        kb_map[("up",)](ev)
        pos_after_up = buf.cursor_position
        assert 0 < pos_after_up < len(pasted)

        # Move cursor left 2 chars and insert "【新增修改】" right in the middle of the pasted prompt!
        kb_map[("left",)](ev)
        kb_map[("left",)](ev)
        pos_after_left = buf.cursor_position
        assert pos_after_left == pos_after_up - 2
        buf.insert_text("【新增修改】")
        assert "【新增修改】" in buf.text

        # Move cursor down 1 visual row
        kb_map[("down",)](ev)
        assert buf.cursor_position > pos_after_left

        # Home / End jump to start and end of pasted prompt
        kb_map[("home",)](ev)
        assert buf.cursor_position == 0
        kb_map[("end",)](ev)
        assert buf.cursor_position == len(buf.text)

        # 3. Scroll main viewport up -> rightmost column of main viewport now shows scrollbar (`┃` and `│`)!
        sess._scroll_offset_from_bottom = 6
        screen_main_scrolled = Screen()
        app.layout.container.write_to_screen(
            screen_main_scrolled,
            MouseHandlers(),
            WritePosition(0, 0, cols, rows),
            "",
            erase_bg=True,
            z_index=None,
        )
        in_h = input_win.height().preferred
        main_h = rows - 4 - in_h
        main_right_edge = "".join(screen_main_scrolled.data_buffer[y][cols - 1].char for y in range(main_h))
        assert "┃" in main_right_edge and "│" in main_right_edge

        # Reset main scroll back to 0 -> main scrollbar disappears again
        sess._scroll_offset_from_bottom = 0

        # 4. Paste an ultra-long prompt (35 visual rows > max_input_rows) -> input_window right margin shows scrollbar!
        ultra_pasted = "".join(
            f"第{i:02d}行：请深入分析 OrderStateView.kt 的工作机制、功能地图、业务逻辑地图；"
            for i in range(1, 30)
        )
        buf.text = ultra_pasted
        buf.cursor_position = len(ultra_pasted)
        screen_input_scrolled = Screen()
        app.layout.container.write_to_screen(
            screen_input_scrolled,
            MouseHandlers(),
            WritePosition(0, 0, cols, rows),
            "",
            erase_bg=True,
            z_index=None,
        )
        in_h_ultra = input_win.height().preferred
        input_right_edge = "".join(
            screen_input_scrolled.data_buffer[y][cols - 1].char for y in range(rows - 2 - in_h_ultra, rows - 2)
        )
        assert "┃" in input_right_edge and "│" in input_right_edge

        # 5. Verify mouse wheel `<scroll-up>` / `<scroll-down>` scrolls inside the multi-line input box!
        pos_before_wheel = buf.cursor_position
        kb_map[("<scroll-up>",)](ev)
        assert buf.cursor_position < pos_before_wheel
        kb_map[("<scroll-down>",)](ev)
        assert buf.cursor_position == pos_before_wheel

    asyncio.run(_run_verify())


def test_fullscreen_tui_input_box_first_tick_scroll_and_multiline_experience(tmp_path: Path, monkeypatch) -> None:
    """
    Verifies (`t39u`):
    1. Pasting a multi-line (`\n`) or soft-wrapped prompt into `input_window` fills all visible rows cleanly
       with zero dead blank rows at the bottom.
    2. On the VERY FIRST tick of `<scroll-up>` / `up` (or `<scroll-down>` / `down`), the visible viewport
       (`_input_top_row`) and the right-side scrollbar thumb immediately shift by 1 visual row (no 8-tick dead zone!).
    3. Even if `_scroll_offset_from_bottom > 0`, moving `up` / `down` inside a multi-line input box when `cur_row > 0`
       is never locked out, and reaching the bottom of the input box on `down` never overwrites user text with history!
    4. Slash commands like `/status` and `/model` record Rich-styled blocks into `_screen_blocks` in TUI mode.
    """
    import asyncio
    from prompt_toolkit.layout.mouse_handlers import MouseHandlers
    from prompt_toolkit.layout.screen import Screen, WritePosition
    import fix.cli as cli_mod

    async def _run_verify() -> None:
        cols, rows = 90, 28
        monkeypatch.setattr(cli_mod, "_query_os_terminal_size", lambda: (cols, rows))
        sess = FixInteractiveSession(repo=tmp_path)
        app = sess.build_fullscreen_tui_app()
        hsplit = app.layout.container.content
        input_win = hsplit.children[3]
        buf = input_win.content.buffer

        # 1. Paste a 16-line snippet with explicit newlines
        pasted = "\n".join([f"第{i:02d}行：OrderStateView 与 WidgetManager 状态联动检查项" for i in range(1, 17)])
        buf.text = pasted
        buf.cursor_position = len(pasted)

        def _render_input_lines() -> list[str]:
            sc = Screen()
            app.layout.container.write_to_screen(
                sc,
                MouseHandlers(),
                WritePosition(0, 0, cols, rows),
                "",
                erase_bg=True,
                z_index=None,
            )
            in_h = input_win.height().preferred
            return [
                "".join(sc.data_buffer[y][x].char for x in range(cols)).rstrip()
                for y in range(rows - 2 - in_h, rows - 2)
            ]

        vis_initial = _render_input_lines()
        assert len(vis_initial) == 9  # rows // 3 == 9
        assert "第08行" in vis_initial[0]
        assert "第16行" in vis_initial[-1]
        assert sess._input_top_row == 7

        kb_map = {tuple(b.keys): b.handler for b in app.key_bindings.bindings}

        class _Ev:
            def __init__(self) -> None:
                self.app = app
                self.arg = 1
                self.current_buffer = buf

        ev = _Ev()

        # 2. First tick of `<scroll-up>` immediately scrolls visible text from lines 8..16 to lines 7..15!
        kb_map[("<scroll-up>",)](ev)
        vis_after_1_up = _render_input_lines()
        assert sess._input_top_row == 6
        assert "第07行" in vis_after_1_up[0]
        assert "第15行" in vis_after_1_up[-1]

        # 3. First tick of `<scroll-down>` immediately scrolls visible text back to lines 8..16!
        kb_map[("<scroll-down>",)](ev)
        vis_after_1_down = _render_input_lines()
        assert sess._input_top_row == 7
        assert "第08行" in vis_after_1_down[0]
        assert "第16行" in vis_after_1_down[-1]

        # 4. Extra `<scroll-down>` / `down` at the very bottom NEVER wipes out the pasted text!
        kb_map[("down",)](ev)
        kb_map[("<scroll-down>",)](ev)
        assert buf.text == pasted

        # 5. `/status` in TUI mode appends a Rich block to `_screen_blocks`
        before_cnt = len(sess._screen_blocks)
        await sess.handle_line("/status")
        assert len(sess._screen_blocks) == before_cnt + 1

    asyncio.run(_run_verify())


def test_typer_cli_entrypoint_and_all_subcommands_register_cleanly(tmp_path: Path) -> None:
    """Ensure `fix` Typer app and `main()` register all commands without Unsupported Type errors (e.g. Python 3.9 Typer)."""
    from typer.testing import CliRunner
    from fix.cli import app

    runner = CliRunner()
    res_help = runner.invoke(app, ["--help"])
    assert res_help.exit_code == 0, res_help.output
    assert "Fix" in res_help.output

    res_status = runner.invoke(app, ["status", "--repo", str(tmp_path)])
    assert res_status.exit_code == 0, res_status.output


def test_llm_auto_continues_on_finish_reason_length_without_truncation() -> None:
    """
    Verifies (`t41u` Issue 1):
    When an LLM response hits the per-call token budget (`finish_reason == "length"`) mid-report
    (e.g. cut off at `└─ 退` because `reasoning_tokens + content_tokens` reached `max_tokens`),
    `complete_with_tools` automatically continues generation from the exact cutoff point and
    concatenates the continuation seamlessly without losing a single character!
    Also verifies recovery when `reasoning_content` alone hits `finish_reason == "length"` before emitting `content`.
    """
    import asyncio
    from fix.llm import LLMResponse, ScriptedMockLLM

    async def _run_check() -> None:
        mock_llm = ScriptedMockLLM(
            tool_Queue=[
                LLMResponse(
                    content="## 业务逻辑地图\n  展示后按 bean.data.displaySeconds 起倒计时\n    ├─ 到期：回落 wam\n    └─ 退",
                    reasoning_content="Deep chain of thought consuming 5500 tokens...",
                    finish_reason="length",
                    reasoning_tokens=5500,
                    prompt_tokens=1200,
                    completion_tokens=8192,
                    total_tokens=9392,
                ),
                LLMResponse(
                    content="场原因为 EXIT_VOICE 或 CLICKED ───► 清空状态并发出 Dismissed 事件\n\n## 代码问题与风险清单\n1. ThinkManager 缺少 @Inject 构造器。",
                    finish_reason="stop",
                    prompt_tokens=1500,
                    completion_tokens=600,
                    total_tokens=2100,
                ),
            ]
        )
        resp = await mock_llm.complete_with_tools(
            messages=[{"role": "user", "content": "分析当前修改"}],
            tools=None,
        )
        assert resp.continuation_rounds == 1
        assert resp.finish_reason == "stop"
        assert "└─ 退场原因为 EXIT_VOICE 或 CLICKED" in resp.content
        assert "## 代码问题与风险清单" in resp.content

    asyncio.run(_run_check())


def test_fullscreen_tui_mouse_drag_selection_autoscrolls_up_and_down_and_copies_cleanly(
    tmp_path: Path, monkeypatch
) -> None:
    """
    Verifies (`t41u` Issue 2):
    1. Dragging the mouse cursor to the TOP edge (`y <= 1`) while selecting text in `main_content_window`
       automatically scrolls `_scroll_offset_from_bottom` UP across multiple screens and extends the selection!
    2. Dragging the mouse cursor to the BOTTOM edge (`y >= avail_h - 1` or into the status/input area)
       automatically scrolls `_scroll_offset_from_bottom` DOWN across multiple screens and extends the selection!
    3. Releasing the mouse button (`MOUSE_UP`) automatically extracts all spanned lines across multiple screens
       WITHOUT any right-edge scrollbar characters (`┃` or `│`) and copies them to the clipboard (`_last_selected_text`).
    """
    import asyncio
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
    import fix.cli as cli_mod

    async def _run_verify() -> None:
        cols, rows = 80, 20
        monkeypatch.setattr(cli_mod, "_query_os_terminal_size", lambda: (cols, rows))
        sess = FixInteractiveSession(repo=tmp_path)

        # Populate 70 lines of output (~4 screens at height=17)
        def _big_report(c):
            for idx in range(1, 71):
                c.print(f"Report_Line_{idx:02d}: 车机心声状态机重构分析内容第{idx:02d}行")

        sess._screen_blocks = [_big_report]
        app = sess.build_fullscreen_tui_app()
        hsplit = app.layout.container.content
        main_win = hsplit.children[0]

        # Force initial render so `_cached_render_lines` is populated
        _ = main_win.content.text()
        assert len(sess._cached_render_lines) >= 70
        assert sess._scroll_offset_from_bottom == 0

        # --- Part A: Drag UP from bottom of screen (`y=15`) to top edge (`y=0`) across multiple screens ---
        sess.handle_tui_mouse_event(
            MouseEvent(
                position=Point(x=60, y=15),
                event_type=MouseEventType.MOUSE_DOWN,
                button=MouseButton.LEFT,
                modifiers=frozenset(),
            ),
            source="screen",
        )
        # Drag to top edge `y=0` 25 times -> auto-scrolls up 50 lines!
        for _ in range(25):
            sess.handle_tui_mouse_event(
                MouseEvent(
                    position=Point(x=0, y=0),
                    event_type=MouseEventType.MOUSE_MOVE,
                    button=MouseButton.LEFT,
                    modifiers=frozenset(),
                ),
                source="screen",
            )
        assert sess._scroll_offset_from_bottom >= 40

        # Release mouse -> automatically copies multi-screen selection cleanly without `┃`/`│`!
        sess.handle_tui_mouse_event(
            MouseEvent(
                position=Point(x=0, y=0),
                event_type=MouseEventType.MOUSE_UP,
                button=MouseButton.LEFT,
                modifiers=frozenset(),
            ),
            source="screen",
        )
        copied_up = sess._last_selected_text
        assert len(copied_up.splitlines()) >= 45
        assert "Report_Line_68" in copied_up
        assert "Report_Line_15" in copied_up
        assert "┃" not in copied_up and "│" not in copied_up
        assert "✓ 已复制跨屏选中内容" in sess.get_right_toast_text()

        # --- Part B: Now at top of report (`_scroll_offset_from_bottom` high), drag DOWN to bottom edge (`y=18`) ---
        sess.handle_tui_mouse_event(
            MouseEvent(
                position=Point(x=0, y=1),
                event_type=MouseEventType.MOUSE_DOWN,
                button=MouseButton.LEFT,
                modifiers=frozenset(),
            ),
            source="screen",
        )
        for _ in range(25):
            sess.handle_tui_mouse_event(
                MouseEvent(
                    position=Point(x=70, y=18),
                    event_type=MouseEventType.MOUSE_MOVE,
                    button=MouseButton.LEFT,
                    modifiers=frozenset(),
                ),
                source="screen",
            )
        assert sess._scroll_offset_from_bottom == 0
        sess.handle_tui_mouse_event(
            MouseEvent(
                position=Point(x=70, y=18),
                event_type=MouseEventType.MOUSE_UP,
                button=MouseButton.LEFT,
                modifiers=frozenset(),
            ),
            source="screen",
        )
        copied_down = sess._last_selected_text
        assert len(copied_down.splitlines()) >= 45
        assert "Report_Line_15" in copied_down
        assert "Report_Line_70" in copied_down
        assert "┃" not in copied_down and "│" not in copied_down

    asyncio.run(_run_verify())


def test_up_down_arrow_keys_recall_previous_input_history_and_edit_multiline(tmp_path: Path) -> None:
    """Verify pressing `up` and `down` arrow keys recalls previous input history entries cleanly while preserving multi-line cursor movement."""
    from types import SimpleNamespace
    from fix.cli import FixInteractiveSession
    from fix.llm import LLMResponse, ScriptedMockLLM

    session = FixInteractiveSession(repo=str(tmp_path))
    tui_app = session.build_fullscreen_tui_app()
    input_win = tui_app.layout.current_window
    buf = input_win.content.buffer
    kb_map = {b.keys: b.handler for b in tui_app.key_bindings.bindings}
    ev = SimpleNamespace(app=tui_app, arg=1)

    # Seed two historical prompts into the buffer history (as if previously entered)
    buf.history.append_string("第一条历史指令：查看 git status")
    buf.history.append_string("你是一名资深的android车机launcher开发，现在需要你分析下当前项目的修改，看看他到底是要做什么，仅做分析，不对代码做任何的改动")
    buf.reset()
    assert buf.text == ""

    # 1. Press Up once -> recalls most recent prompt
    kb_map[("up",)](ev)
    assert "资深的android车机launcher开发" in buf.text

    # 2. Press Up a second time -> recalls older prompt immediately (even though the recent prompt wrapped to >1 row)
    kb_map[("up",)](ev)
    assert buf.text == "第一条历史指令：查看 git status"

    # 3. Press Down once -> steps forward to most recent prompt
    kb_map[("down",)](ev)
    assert "资深的android车机launcher开发" in buf.text

    # 4. Press Down a second time -> returns to blank draft
    kb_map[("down",)](ev)
    assert buf.text == ""


def test_fullscreen_tui_inflight_rendering_and_single_status_bar_no_duplicate(tmp_path: Path, monkeypatch: Any) -> None:
    """Verify full-screen TUI renders inflight prompt + active_tracker without errors and displays status ONLY in bottom status row."""
    import asyncio
    import asyncio
    from prompt_toolkit.application.current import set_app
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.output.defaults import DummyOutput
    import fix.cli as cli_mod
    from fix.cli import CollapsibleProgressTracker, FixInteractiveSession

    cols, rows = 120, 30
    monkeypatch.setattr(cli_mod, "_query_os_terminal_size", lambda: (cols, rows))

    session = FixInteractiveSession(repo=str(tmp_path))
    with create_pipe_input() as pipe_in:
        tui_app = session.build_fullscreen_tui_app(pt_input=pipe_in, pt_output=DummyOutput())
        # Simulate inflight multi-line prompt + active tracker emitting narrative, tool_start, tool_end
        session._inflight_prompt = "第一行问题：分析当前项目的修改\n第二行补充：仅做分析不改动代码"
        tracker = CollapsibleProgressTracker(session.console)
        tracker.on_event("narrative", {"text": "我先查看项目的 Git 状态与改动清单。"})
        tracker.on_event("tool_start", {"name": "read_file", "arguments": {"rel_path": "ThinkManager.kt"}, "label": "Read(ThinkManager.kt)"})
        tracker.on_event("tool_end", {"name": "run_command", "arguments": {"cmd": "git status --short"}, "output_preview": " M MainActivity.kt"})
        session._active_tracker = tracker

        async def _render_async() -> None:
            with set_app(tui_app):
                tui_app.renderer.render(tui_app, tui_app.layout)

        asyncio.run(_render_async())
        screen = tui_app.renderer._last_screen
        assert screen is not None
        rendered_rows = [
            "".join(screen.data_buffer[y][x].char for x in range(cols)).rstrip()
            for y in range(rows)
        ]
        full_screen_text = "\n".join(rendered_rows)
        assert "第一行问题：分析当前项目的修改" in full_screen_text
        assert "第二行补充：仅做分析不改动代码" in full_screen_text
        assert "正在结合工具输出推理下一步..." in full_screen_text
        # Crucial check: the active status text must appear ONLY ONCE (on the bottom status row), never duplicated above!
        occurrences = sum(1 for r in rendered_rows if "正在结合工具输出推理下一步..." in r)
        assert occurrences == 1, f"Expected status text once, found {occurrences} times in:\n{full_screen_text}"


def test_user_submitted_prompt_preserves_exact_input_box_color_never_orange(tmp_path: Path) -> None:
    """
    Verifies (`t46u`):
    When the user types a prompt in the bottom input box, the prompt text has style `""`
    (native primary foreground, non-bold), and only the `>` symbol is orange (`#f97316 bold`).
    After pressing Enter and moving to the upper main viewport (both during `inflight_prompt`
    and after completion in `_screen_blocks`), the prompt text must keep the exact same `""`
    color and NEVER inherit `#f97316` orange from `>`!
    """
    import asyncio
    from fix.cli import FixInteractiveSession, _render_blocks_to_pt_fragments
    from fix.llm import LLMResponse, ScriptedMockLLM

    user_prompt = "你是一名资深的android车机launcher开发，分析当前项目的修改\n第二行补充说明"

    # 1. Verify inflight prompt rendering (right after pressing Enter while task is running)
    inflight_frags = _render_blocks_to_pt_fragments([], width=120, inflight_prompt=user_prompt)
    assert any(st == "#0ea5e9 bold" and txt.strip() == "›" for st, txt in inflight_frags)
    prompt_inflight_spans = [(st, txt) for st, txt in inflight_frags if "资深" in txt or "第二行" in txt]
    assert prompt_inflight_spans
    for st, txt in prompt_inflight_spans:
        assert st == "", f"Expected exact input-box style '' for inflight prompt, got {(st, txt)}"

    # 2. Verify completed turn rendering in TUI mode (`_screen_blocks` after task finishes)
    mock_llm = ScriptedMockLLM(
        [
            LLMResponse(content="分析完成：`OrderStateView` 负责心声展示。", tool_calls=[]),
        ]
    )
    session = FixInteractiveSession(repo=str(tmp_path))
    session.llm = mock_llm
    session.runtime.llm = mock_llm
    session._tui_mode = True

    asyncio.run(session.handle_line(user_prompt))
    completed_frags = _render_blocks_to_pt_fragments(session._screen_blocks, width=120)
    assert any(st == "#0ea5e9 bold" and txt.strip() == "›" for st, txt in completed_frags)
    prompt_completed_spans = [(st, txt) for st, txt in completed_frags if "资深" in txt or "第二行" in txt]
    assert prompt_completed_spans
    for st, txt in prompt_completed_spans:
        assert st == "", f"Expected exact input-box style '' for completed prompt, got {(st, txt)}"
    # Verify Agent output inline code font color is #c87dbb
    from fix.cli import CleanMarkdown
    md_frags = _render_blocks_to_pt_fragments([lambda c: c.print(CleanMarkdown("分析完成：`OrderStateView` 负责心声展示。"))], width=120)
    code_spans = [(st, txt) for st, txt in md_frags if "OrderStateView" in txt]
    assert any("#c87dbb" in st.lower() for st, _ in code_spans), f"Expected #c87dbb on code span, got {code_spans}"


def test_cjk_mouse_selection_no_duplicate_white_background_and_exact_char_to_cell_mapping(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """
    Regression test for `t53u`:
    1. When the user selects `"深入分析"` on a single screen in the terminal:
       `"你是一名资深的android车机launcher开发，现在需要你深入分析下桌面上的任务卡片在接收到语音卡片后，他是如何显示的，为什么用户语音输入“介绍一下周杰伦”后，语音大卡没有显示语音"`,
       the terminal emulator already shows its native blue selection highlight.
       TUI must NOT draw a second white `reverse` highlight at `"r开发"` and must NOT overwrite the clipboard!
    2. When `source="main"` (where `prompt_toolkit` passes character index `x=32..36` instead of display cell
       column `x=49..57`) is used with Shift-extend or cross-screen auto-scroll, `_char_idx_to_line_cell_col`
       maps character index `32..36` to display cell column `49..57`, extracting `"深入分析"` (and NEVER `"r开发"`!).
    """
    import asyncio
    import asyncio
    from prompt_toolkit.application.current import set_app
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType, MouseModifier
    from prompt_toolkit.output.defaults import DummyOutput
    import fix.cli as cli_mod
    from fix.cli import FixInteractiveSession

    cols, rows = 174, 24
    monkeypatch.setattr(cli_mod, "_query_os_terminal_size", lambda: (cols, rows))

    target_line = (
        "你是一名资深的android车机launcher开发，现在需要你深入分析下桌面上的任务卡片在接收到语音卡片后，"
        "他是如何显示的，为什么用户语音输入“介绍一下周杰伦”后，语音大卡没有显示语音"
    )
    async def _verify() -> None:
        session = FixInteractiveSession(repo=str(tmp_path))
        session._inflight_prompt = target_line
        with create_pipe_input() as pipe_in:
            tui_app = session.build_fullscreen_tui_app(pt_input=pipe_in, pt_output=DummyOutput())
            with set_app(tui_app):
                tui_app.renderer.render(tui_app, tui_app.layout)

                # Find the document line index containing "深入分析"
                doc_idx = -1
                s_char = e_char = 0
                for idx_l, ln_frags in enumerate(session._cached_render_lines):
                    full_s = "".join(seg for _, seg in ln_frags)
                    if "深入分析" in full_s:
                        doc_idx = idx_l
                        s_char = full_s.index("深入分析")
                        e_char = s_char + len("深入分析")
                        break
                assert doc_idx >= 0

                # Case 1: Normal single-screen drag across "深入分析" using terminal cell columns (49..57)
                # Must highlight with `bg:#264F78 #ffffff` (never white `reverse`) and copy `"深入分析"` (never `"r开发"`)!
                from rich.cells import cell_len

                full_line_str = "".join(seg for _, seg in session._cached_render_lines[doc_idx])
                s_cell = sum(max(1, cell_len(ch)) for ch in full_line_str[:s_char])
                e_cell = sum(max(1, cell_len(ch)) for ch in full_line_str[:e_char])
                for ev_t, cx in [
                    (MouseEventType.MOUSE_DOWN, s_cell),
                    (MouseEventType.MOUSE_MOVE, e_cell),
                ]:
                    session.handle_tui_mouse_event(
                        MouseEvent(
                            position=Point(x=cx, y=doc_idx),
                            event_type=ev_t,
                            button=MouseButton.LEFT,
                            modifiers=frozenset(),
                        ),
                        source="screen",
                    )
                assert session.get_selected_main_text() == "深入分析"
                session.handle_tui_mouse_event(
                    MouseEvent(
                        position=Point(x=e_cell, y=doc_idx),
                        event_type=MouseEventType.MOUSE_UP,
                        button=MouseButton.LEFT,
                        modifiers=frozenset(),
                    ),
                    source="screen",
                )
                assert session._last_selected_text == "深入分析"
                assert "r开发" not in session._last_selected_text

                # Case 2: Selection via `source="main"` (character index `s_char..e_char`) with Shift-extend / cross-screen
                session.handle_tui_mouse_event(
                    MouseEvent(
                        position=Point(x=s_char, y=doc_idx),
                        event_type=MouseEventType.MOUSE_DOWN,
                        button=MouseButton.LEFT,
                        modifiers=frozenset(),
                    ),
                    source="main",
                )
                session.handle_tui_mouse_event(
                    MouseEvent(
                        position=Point(x=e_char, y=doc_idx),
                        event_type=MouseEventType.MOUSE_DOWN,
                        button=MouseButton.LEFT,
                        modifiers=frozenset({MouseModifier.SHIFT}),
                    ),
                    source="main",
                )
                assert session.get_selected_main_text() == "深入分析"
                session.handle_tui_mouse_event(
                    MouseEvent(
                        position=Point(x=e_char, y=doc_idx),
                        event_type=MouseEventType.MOUSE_UP,
                        button=MouseButton.LEFT,
                        modifiers=frozenset({MouseModifier.SHIFT}),
                    ),
                    source="main",
                )
                assert session._last_selected_text == "深入分析"
                assert "r开发" not in session._last_selected_text

    asyncio.run(_verify())


def test_bug_fix_or_explain_hitting_max_steps_always_triggers_final_synthesis_not_intermediate_narrative(
    tmp_path: Path,
) -> None:
    """
    Regression test for `t55u`:
    1. When a task (even with `forced_intent=TaskIntent.BUG_FIX`) emits intermediate `✦` narrative text
       alongside tool calls on every step until `max_steps` is reached, `session.final_summary` must NOT
       be overwritten with the intermediate `✦` narrative text.
    2. Instead, when the loop exits with `last_step_had_tool_calls=True`, `_force_final_synthesis` MUST
       always execute (regardless of `session.intent`) and produce the complete final analysis report.
    3. Furthermore, when the user asks a pure analysis/diagnosis prompt ("深入分析下...他是如何显示的，为什么...没有显示语音信息"),
       even if `_run_probe_phase` returns `TaskIntent.BUG_FIX`, `FixRuntime` routes it to `TaskIntent.EXPLAIN`.
    """
    import asyncio
    from fix.llm import LLMResponse, ScriptedMockLLM
    from fix.runtime import FixRuntime
    from fix.schema import ProbeDecision, TaskIntent, ToolCallRequest
    from fix.tools import WorkspaceToolkit

    (tmp_path / "sample.py").write_text("x = 1\n", encoding="utf-8")
    toolkit = WorkspaceToolkit(tmp_path)

    # Mock LLM that emits 2 tool-calling steps with intermediate narrative text, followed by final synthesis
    mock_llm = ScriptedMockLLM(
        [
            LLMResponse(
                content="startNextTransition 拿出的是旧的 pending，我查整个日志里 TaskView 队列的积压成因与最终是否消化。",
                tool_calls=[
                    ToolCallRequest(
                        call_id="c1",
                        name="read_file",
                        arguments={"rel_path": "sample.py", "start_line": 1, "end_line": 10},
                    )
                ],
            ),
            LLMResponse(
                content="## 最终根因分析报告\n\n1. 桌面任务卡片通过 `ControlledRemoteAppTaskView` 承载 `AppTaskView`。\n2. 20:33:27 因 `TaskViewTransitions` 队列积压导致未挂载。",
                tool_calls=[],
            ),
        ]
    )
    runtime = FixRuntime(llm=mock_llm, toolkit=toolkit)

    async def _verify() -> None:
        # Force max_steps=1 via monkeypatching TaskBudget default or modifying session budget
        from fix.schema import TaskBudget
        orig_max = TaskBudget.model_fields["max_steps"].default
        TaskBudget.model_fields["max_steps"].default = 1
        TaskBudget.model_rebuild(force=True)
        try:
            sess = await runtime.execute_task(
                user_prompt="深入分析为什么语音大卡没有显示语音信息",
                forced_intent=TaskIntent.BUG_FIX,
            )
            assert "## 最终根因分析报告" in sess.final_summary
            assert sess.final_summary != "startNextTransition 拿出的是旧的 pending，我查整个日志里 TaskView 队列的积压成因与最终是否消化。"
        finally:
            TaskBudget.model_fields["max_steps"].default = orig_max
            TaskBudget.model_rebuild(force=True)

    asyncio.run(_verify())


def test_incomplete_transition_note_without_tool_calls_still_triggers_final_synthesis(
    tmp_path: Path,
) -> None:
    """
    Regression test for `t56u`:
    1. If the LLM outputs a short transitional sentence (e.g. 'startNextTransition 拿出的是旧的 pending... 我查整个日志里 TaskView 队列的积压成因与最终是否消化。')
       WITHOUT tool_calls (or if Stage 1 synthesis returns a transitional sentence or fails),
       `FixRuntime` must detect via `_looks_like_incomplete_transition_note` that it is NOT a final conclusion
       and invoke `_force_final_synthesis` (falling back to Stage 2 fast model or Stage 3 digest if needed).
    2. `_slice_valid_recent_history` never starts a sliced history on an orphan `role == "tool"` message.
    """
    import asyncio
    from fix.llm import LLMResponse, ScriptedMockLLM
    from fix.runtime import FixRuntime
    from fix.schema import TaskIntent, ToolCallRequest
    from fix.tools import WorkspaceToolkit

    (tmp_path / "sample.py").write_text("x = 1\n", encoding="utf-8")
    toolkit = WorkspaceToolkit(tmp_path)

    transition_sentence = (
        "startNextTransition 拿出的是旧的 pending(mTaskView=116986696, mType=3),"
        "我们新的 AppTaskView 启动(133056948)排在 18 深队列尾部。我查整个日志里 TaskView 队列的积压成因与最终是否消化。"
    )
    assert FixRuntime._looks_like_incomplete_transition_note(transition_sentence) is True
    assert FixRuntime._looks_like_incomplete_transition_note("## 最终根因结论\n\n队列积压导致未挂载。") is False

    # Case A: Step 1 calls a tool, Step 2 returns a transition sentence without tool_calls -> must still synthesize!
    mock_llm = ScriptedMockLLM(
        [
            LLMResponse(
                content="我先读取代码和日志。",
                tool_calls=[
                    ToolCallRequest(
                        call_id="c1",
                        name="read_file",
                        arguments={"rel_path": "sample.py", "start_line": 1, "end_line": 5},
                    )
                ],
            ),
            LLMResponse(
                content=transition_sentence,
                tool_calls=[],
            ),
            LLMResponse(
                content="## 最终分析结论\n\n1. 语音大卡由 `ControlledRemoteAppTaskView` 托管 `AppTaskView`。\n2. `TaskViewTransitions` 队列积压 18 项导致未挂载。",
                tool_calls=[],
            ),
        ]
    )
    runtime = FixRuntime(llm=mock_llm, toolkit=toolkit)

    async def _verify() -> None:
        sess = await runtime.execute_task(
            user_prompt="深入分析为什么语音大卡没有显示语音信息",
            forced_intent=TaskIntent.EXPLAIN,
        )
        assert "## 最终分析结论" in sess.final_summary
        assert sess.final_summary != transition_sentence

        # Verify _slice_valid_recent_history never starts on role == 'tool'
        hist = [
            {"role": "assistant", "content": "step1", "tool_calls": [{"id": "1"}, {"id": "2"}]},
            {"role": "tool", "tool_call_id": "1", "content": "out1"},
            {"role": "tool", "tool_call_id": "2", "content": "out2"},
            {"role": "assistant", "content": "step2", "tool_calls": [{"id": "3"}]},
            {"role": "tool", "tool_call_id": "3", "content": "out3"},
        ]
        sliced = FixRuntime._slice_valid_recent_history(hist, min_messages=3)
        assert sliced[0]["role"] == "assistant"

    asyncio.run(_verify())


@pytest.mark.asyncio
async def test_auto_collapse_execution_steps_on_result_and_manual_expand_collapse(tmp_path: Path) -> None:
    """
    Regression test for `t57u`:
    1. While a task is executing (`_active_tracker` is active), execution steps (`✦` narrative, `• Explored`, `• Ran`)
       are rendered live in full detail.
    2. Once the Agent outputs the final result (`session.final_summary`), the preceding execution operation steps
       are automatically collapsed (`collapsed = True`) into a compact summary line:
       `▸ 已自动收起执行过程 (#1: ...)` while hiding the raw `✦` / `• Explored` / `• Ran` blocks above the final result.
    3. The user can manually expand and collapse the execution steps via:
       - `session.toggle_turn_steps_collapsed()` / `Ctrl+O`
       - `/steps` slash command in TUI mode
       - Mouse click on the `▸ / ▾` fold header line in `_cached_render_lines`
    """
    from fix.cli import _render_blocks_to_pt_fragments, create_fix_console
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
    from prompt_toolkit.output import DummyOutput

    repo = _init_sample_project(tmp_path)
    buf = StringIO()
    out_console = create_fix_console(file=buf, width=120, force_terminal=False)

    toolkit = WorkspaceToolkit(repo)
    memory = DualMemoryStore(repo, persona_db_path=tmp_path / "persona.db")
    context_mgr = ContextManager()

    session_ref: FixInteractiveSession | None = None
    live_text_during_run = ""

    def step_cb(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> LLMResponse:
        nonlocal live_text_during_run
        assert session_ref is not None
        if len(mock_llm.recorded_messages) == 1:
            return LLMResponse(
                content="我先搜索并读取 greet 函数的实现。",
                tool_calls=[
                    ToolCallRequest(name="search_code", arguments={"pattern": "greet"}),
                    ToolCallRequest(name="read_file", arguments={"rel_path": "app.py"}),
                ],
            )
        if len(mock_llm.recorded_messages) == 2:
            # Capture the live TUI render while the task is still in-flight!
            assert session_ref._active_tracker is not None
            frags_live = _render_blocks_to_pt_fragments(
                session_ref._screen_blocks,
                width=120,
                active_tracker=session_ref._active_tracker,
                inflight_prompt=session_ref._inflight_prompt,
                repo_name=repo.name,
            )
            live_text_during_run = "".join(seg for _, seg in frags_live)
            return LLMResponse(
                content="# greet 最终分析结果\n\n`greet` 位于 `app.py` 第 1 行。",
                tool_calls=[],
            )
        return LLMResponse(content="# greet 最终分析结果", tool_calls=[])

    mock_llm = ScriptedMockLLM(
        step_callback=step_cb,
        structured_handlers={
            "ProbeDecision": ProbeDecision(intent=TaskIntent.EXPLAIN, summary="Analyze greet"),
        },
    )
    evolution = EvolutionEngine(repo, memory=memory, llm=mock_llm)
    runtime = FixRuntime(llm=mock_llm, toolkit=toolkit, context_mgr=context_mgr, memory=memory, evolution=evolution)

    session_ref = FixInteractiveSession(
        repo=repo,
        stack=(toolkit, memory, context_mgr, mock_llm, evolution, runtime),
        out_console=out_console,
    )
    session_ref._tui_mode = True
    session_ref.render_welcome_banner()

    with create_pipe_input() as pipe_in:
        app = session_ref.build_fullscreen_tui_app(pt_input=pipe_in, pt_output=DummyOutput())
        await session_ref.handle_line("分析 greet 的实现")

        # 1. Verify that DURING execution, the live stream showed the full intermediate steps
        assert "✦ 我先搜索并读取 greet 函数的实现。" in live_text_during_run
        assert "• Explored" in live_text_during_run
        assert "└ Search greet in ." in live_text_during_run

        # 2. Verify that AFTER outputting the result, the execution steps are automatically collapsed!
        frags_collapsed = _render_blocks_to_pt_fragments(
            session_ref._screen_blocks,
            width=120,
            repo_name=repo.name,
        )
        text_collapsed = "".join(seg for _, seg in frags_collapsed)
        assert "▸ 已自动收起执行过程 (#1:" in text_collapsed
        assert "2 次工具调用" in text_collapsed
        assert "# greet 最终分析结果" in text_collapsed
        assert "✦ 我先搜索并读取 greet 函数的实现。" not in text_collapsed
        assert "└ Search greet in ." not in text_collapsed

        # 3. Verify manual expand via `/steps` command in TUI mode
        await session_ref.handle_line("/steps")
        frags_expanded = _render_blocks_to_pt_fragments(
            session_ref._screen_blocks,
            width=120,
            repo_name=repo.name,
        )
        text_expanded = "".join(seg for _, seg in frags_expanded)
        assert "▾ 已展开执行过程 (#1:" in text_expanded
        assert "✦ 我先搜索并读取 greet 函数的实现。" in text_expanded
        assert "• Explored" in text_expanded
        assert "└ Search greet in ." in text_expanded
        assert "Read app.py" in text_expanded
        assert "# greet 最终分析结果" in text_expanded

        # 4. Verify manual collapse via `toggle_turn_steps_collapsed()` (bound to Ctrl+O)
        new_state = session_ref.toggle_turn_steps_collapsed()
        assert new_state is True
        frags_collapsed_again = _render_blocks_to_pt_fragments(
            session_ref._screen_blocks,
            width=120,
            repo_name=repo.name,
        )
        text_collapsed_again = "".join(seg for _, seg in frags_collapsed_again)
        assert "▸ 已自动收起执行过程 (#1:" in text_collapsed_again
        assert "└ Search greet in ." not in text_collapsed_again

        # 5. Verify mouse click on the fold header line toggles it back to expanded!
        # Populate `_cached_render_lines` via main content control
        from prompt_toolkit.layout.controls import FormattedTextControl
        main_ctrl = None
        for w in app.layout.find_all_windows():
            if isinstance(w.content, FormattedTextControl):
                res = w.content.text() if callable(w.content.text) else []
                txt = "".join(s for _, s in res)
                if "已自动收起执行过程" in txt:
                    main_ctrl = w.content
                    break
        assert main_ctrl is not None
        fold_doc_line = None
        for idx_line, ln_frags in enumerate(session_ref._cached_render_lines):
            if "已自动收起执行过程 (#1:" in "".join(s for _, s in ln_frags):
                fold_doc_line = idx_line
                break
        assert fold_doc_line is not None
        assert session_ref._find_turn_fold_at_doc_line(fold_doc_line) == 1
        # Clicking that line expands turn #1
        session_ref.toggle_turn_steps_collapsed(session_ref._find_turn_fold_at_doc_line(fold_doc_line))
        assert session_ref._turn_fold_states[1]["collapsed"] is False


def test_bottom_left_status_time_format_hms(tmp_path: Path) -> None:
    """
    Regression test for `t58u`:
    Verifies that the bottom-left status bar (`get_left_status_text`) and `format_elapsed_hms`
    display elapsed time in `Xh Ym Zs` / `Ym Zs` / `Zs` format (e.g., 1 hour 3 minutes 18 seconds -> `1h 3m 18s`).
    """
    import time
    from fix.cli import CollapsibleProgressTracker, FixInteractiveSession, format_elapsed_hms

    assert format_elapsed_hms(0.0) == "0s"
    assert format_elapsed_hms(18.4) == "18s"
    assert format_elapsed_hms(198.0) == "3m 18s"
    assert format_elapsed_hms(3798.0) == "1h 3m 18s"
    assert format_elapsed_hms(3605.0) == "1h 0m 5s"

    repo = _init_sample_project(tmp_path)
    session = FixInteractiveSession(repo=repo)

    session._task_elapsed_sec = 3798.0
    assert session.get_left_status_text() == "● 就绪 · 1h 3m 18s"

    session._task_elapsed_sec = 198.0
    assert session.get_left_status_text() == "● 就绪 · 3m 18s"

    tracker = CollapsibleProgressTracker(session.console)
    tracker.started_at = time.time() - 3798.2
    tracker.current_action_label = "正在分析代码"
    session._active_tracker = tracker
    try:
        live_status = session.get_left_status_text()
        assert "正在分析代码 · 1h 3m 18s" in live_status
    finally:
        session._active_tracker = None


@pytest.mark.asyncio
async def test_multi_provider_api_adapters_all_seven_vendors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Regression test for `t59u` (Part 1):
    Verifies the 7 LLM provider specifications (DeepSeek, GLM, MiMo, MiniMax, Qwen, OpenAI, Anthropic)
    and their tailored API calling conventions inside `LLMGateway`:
    - Anthropic native `/v1/messages` adapter (`x-api-key`, `anthropic-version`, `system`, `tool_use`, `tool_result`, `input_schema`).
    - MiniMax temperature clamping (`temperature=0.0` -> `0.01`) and `base_resp.status_code` error detection + `/chat/completions` probe fallback.
    - DeepSeek `deepseek-reasoner` omitting `temperature` and stripping historical `reasoning_content` on non-tool turns.
    - OpenAI `o3` / `o4-mini` converting `max_tokens` to `max_completion_tokens`.
    - GLM & Qwen & MiMo verification and model discovery.
    """
    import httpx
    from fix.llm import LLMGateway, PROVIDER_ORDER, PROVIDER_SPECS

    assert PROVIDER_ORDER == ["deepseek", "glm", "mimo", "minimax", "qwen", "openai", "anthropic"]
    assert set(PROVIDER_SPECS.keys()) == set(PROVIDER_ORDER)

    captured_requests: list[tuple[str, str, dict[str, str], dict[str, object]]] = []

    class _FakeResponse:
        def __init__(self, status_code: int, json_data: dict[str, object]) -> None:
            self.status_code = status_code
            self._json_data = json_data

        def json(self) -> dict[str, object]:
            return self._json_data

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise httpx.HTTPStatusError("HTTP error", request=None, response=None)  # type: ignore[arg-type]

    class _MockAsyncClient:
        def __init__(self, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> "_MockAsyncClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

        async def get(self, url: str, headers: dict[str, str] | None = None) -> _FakeResponse:
            captured_requests.append(("GET", url, dict(headers or {}), {}))
            if "api.minimax.chat/v1/models" in url:
                # Simulate MiniMax not exposing /v1/models (404), triggering /chat/completions probe fallback
                return _FakeResponse(404, {"error": "not found"})
            if "api.anthropic.com/v1/models" in url:
                if (headers or {}).get("x-api-key") != "sk-ant-valid":
                    return _FakeResponse(401, {"error": "invalid x-api-key"})
                return _FakeResponse(
                    200,
                    {
                        "data": [
                            {"id": "claude-opus-5-5"},
                            {"id": "claude-sonnet-5-5"},
                            {"id": "claude-3-5-haiku-latest"},
                        ]
                    },
                )
            if "api.deepseek.com" in url:
                return _FakeResponse(
                    200,
                    {
                        "data": [
                            {"id": "deepseek-flash"},
                            {"id": "deepseek-v4-pro"},
                            {"id": "deepseek-chat"},
                            {"id": "deepseek-reasoner"},
                        ]
                    },
                )
            return _FakeResponse(200, {"data": [{"id": "custom-live-model-v1"}]})

        async def post(
            self,
            url: str,
            headers: dict[str, str] | None = None,
            json: dict[str, object] | None = None,
        ) -> _FakeResponse:
            payload = dict(json or {})
            captured_requests.append(("POST", url, dict(headers or {}), payload))
            if "api.anthropic.com/v1/messages" in url:
                return _FakeResponse(
                    200,
                    {
                        "id": "msg_01",
                        "model": payload.get("model", "claude-sonnet-5-5"),
                        "stop_reason": "tool_use",
                        "content": [
                            {"type": "thinking", "thinking": "Inspecting app.py first."},
                            {"type": "text", "text": "我将先读取 `app.py` 确认实现。"},
                            {
                                "type": "tool_use",
                                "id": "toolu_123",
                                "name": "read_file",
                                "input": {"rel_path": "app.py"},
                            },
                        ],
                        "usage": {"input_tokens": 120, "output_tokens": 45},
                    },
                )
            return _FakeResponse(
                200,
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": "Verified response.",
                                "reasoning_content": "Step-by-step chain of thought.",
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42},
                },
            )

    monkeypatch.setattr(httpx, "AsyncClient", _MockAsyncClient)

    cfg_path = tmp_path / "llm_providers.json"
    gw = LLMGateway(api_key="tp-init", providers_config_path=cfg_path)

    # 1. Verify Anthropic native connection (`/v1/models` with `x-api-key` + `anthropic-version`)
    ok_bad, _, err_bad = await gw.verify_and_fetch_models("anthropic", "bad-key")
    assert ok_bad is False
    assert "认证失败" in err_bad

    ok_ant, ant_models, _ = await gw.verify_and_fetch_models("anthropic", "sk-ant-valid")
    assert ok_ant is True
    assert "claude-sonnet-5-5" in ant_models
    assert "claude-3-5-haiku-latest" not in ant_models
    gw.activate_provider_model("anthropic", "claude-sonnet-5-5")

    # Test Anthropic native `/v1/messages` completion with system, assistant tool_calls, tool_result, and tools schema
    resp_ant = await gw.complete_with_tools(
        messages=[
            {"role": "system", "content": "You are Fix Agent."},
            {"role": "user", "content": "请检查 app.py"},
            {
                "role": "assistant",
                "content": "Checking directory",
                "tool_calls": [
                    {
                        "id": "call_prev",
                        "type": "function",
                        "function": {"name": "list_dir", "arguments": '{"rel_path": "."}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_prev", "content": "app.py"},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {"rel_path": {"type": "string"}}},
                },
            }
        ],
    )
    assert resp_ant.content == "我将先读取 `app.py` 确认实现。"
    assert resp_ant.reasoning_content == "Inspecting app.py first."
    assert len(resp_ant.tool_calls) == 1
    assert resp_ant.tool_calls[0].name == "read_file"
    assert resp_ant.tool_calls[0].arguments == {"rel_path": "app.py"}

    last_method, last_url, last_headers, last_payload = captured_requests[-1]
    assert last_method == "POST"
    assert last_url == "https://api.anthropic.com/v1/messages"
    assert last_headers["x-api-key"] == "sk-ant-valid"
    assert last_headers["anthropic-version"] == "2023-06-01"
    assert last_payload["system"] == "You are Fix Agent."
    assert last_payload["tools"][0]["input_schema"]["type"] == "object"  # type: ignore[index]

    # 2. Verify MiniMax connection fallback (404 on /models -> 200 on /chat/completions probe) and temperature clamping
    ok_mm, mm_models, _ = await gw.verify_and_fetch_models("minimax", "mm-valid-key")
    assert ok_mm is True
    assert "MiniMax-M3.1-Flash" in mm_models
    assert "MiniMax-M3" in mm_models
    assert "MiniMax-M1" not in mm_models
    gw.activate_provider_model("minimax", "MiniMax-M3.1-Flash")
    await gw.complete_with_tools(
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.0,  # Must be clamped to >= 0.01 for MiniMax!
    )
    _, mm_url, _, mm_payload = captured_requests[-1]
    assert mm_url == "https://api.minimax.chat/v1/chat/completions"
    assert float(mm_payload["temperature"]) >= 0.01  # type: ignore[arg-type]

    # 3. Verify DeepSeek `deepseek-v4-pro` / `deepseek-flash` filters retired `deepseek-chat`/`deepseek-reasoner`,
    #    omits temperature on reasoning turns, and strips non-tool reasoning_content when tools=None
    ok_ds, ds_models, _ = await gw.verify_and_fetch_models("deepseek", "ds-valid-key")
    assert ok_ds is True
    assert ds_models == ["deepseek-flash", "deepseek-v4-pro"]
    assert "deepseek-chat" not in ds_models
    assert "deepseek-reasoner" not in ds_models
    gw.activate_provider_model("deepseek", "deepseek-v4-pro")
    await gw.complete_with_tools(
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "prev", "reasoning_content": "old thought"},
            {"role": "user", "content": "next"},
        ],
        use_reasoning=True,
    )
    _, ds_url, _, ds_payload = captured_requests[-1]
    assert ds_url == "https://api.deepseek.com/v1/chat/completions"
    assert "temperature" not in ds_payload
    sent_msgs = ds_payload["messages"]  # type: ignore[assignment]
    assert "reasoning_content" not in sent_msgs[1]  # type: ignore[index]

    # 4. Verify OpenAI `gpt-6-astra` / `gpt-6.1-sol` uses `max_completion_tokens` and omits `temperature`
    await gw.verify_and_fetch_models("openai", "sk-openai-valid")
    gw.activate_provider_model("openai", "gpt-6-astra")
    await gw.complete_with_tools(messages=[{"role": "user", "content": "solve"}])
    _, oa_url, _, oa_payload = captured_requests[-1]
    assert oa_url == "https://api.openai.com/v1/chat/completions"
    assert "max_completion_tokens" in oa_payload
    assert "temperature" not in oa_payload


@pytest.mark.asyncio
async def test_connect_and_model_interactive_tui_popup_and_mouse_click_flow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Regression test for `t59u` (Part 2):
    Verifies the end-to-end interactive `/connect` and `/model` experience in Full-Screen TUI mode:
    1. Entering `/model` outputs Base URL, Fast Model, Reasoning Model, AND the switchable models table,
       and opens the interactive model switcher popup (`switch_model`).
    2. Entering `/connect` lists all 7 supported vendors (DeepSeek, GLM, MiMo, MiniMax, Qwen, OpenAI, Anthropic)
       and opens the Step 1 vendor selection popup (`connect_provider`).
    3. Selecting a vendor (via keyboard `↑`/`↓` + `Enter` OR mouse click) opens the Step 2 API Key input popup (`connect_api_key`).
    4. Entering the API Key and pressing `Enter` verifies the connection and pops up the Step 3 model selection list (`connect_select_model`).
    5. Selecting a model (via `Enter` OR mouse click) activates that vendor & model immediately so the user can start using it.
    """
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.output.defaults import DummyOutput
    from fix.cli import create_fix_console

    repo = _init_sample_project(tmp_path)
    buf = StringIO()
    out_console = create_fix_console(file=buf, width=120, force_terminal=False)

    toolkit = WorkspaceToolkit(repo)
    memory = DualMemoryStore(repo, persona_db_path=tmp_path / "persona.db")
    context_mgr = ContextManager()
    mock_llm = ScriptedMockLLM()
    evolution = EvolutionEngine(repo, memory=memory, llm=mock_llm)
    runtime = FixRuntime(llm=mock_llm, toolkit=toolkit, context_mgr=context_mgr, memory=memory, evolution=evolution)

    async def fake_verify(provider_id: str, api_key: str, base_url: str | None = None) -> tuple[bool, list[str], str]:
        if api_key == "invalid-key":
            return False, [], "API Key 认证失败 (HTTP 401)"
        models_map = {
            "deepseek": ["deepseek-flash", "deepseek-v4-pro"],
            "qwen": ["qwen3.8-max", "qwen3-coder-plus", "qwen3.7-plus"],
        }
        models = models_map.get(provider_id, ["model-a", "model-b"])
        mock_llm.connected_providers[provider_id] = {
            "provider_id": provider_id,
            "api_key": api_key,
            "base_url": f"https://api.{provider_id}.example/v1",
            "models": models,
        }
        return True, models, f"✓ 已成功连接 {provider_id.upper()} (共 {len(models)} 个可用模型)"

    monkeypatch.setattr(mock_llm, "verify_and_fetch_models", fake_verify)

    session = FixInteractiveSession(
        repo=repo,
        stack=(toolkit, memory, context_mgr, mock_llm, evolution, runtime),
        out_console=out_console,
    )
    session._tui_mode = True
    session.render_welcome_banner()

    with create_pipe_input() as pipe_in:
        app = session.build_fullscreen_tui_app(pt_input=pipe_in, pt_output=DummyOutput())

        # 1. User runs `/model` -> shows Base URL, Fast Model, Reasoning Model, and switchable models list + popup
        await session.handle_line("/model")
        printed = buf.getvalue()
        assert "Base URL:" in printed
        assert "Fast Model:" in printed
        assert "Reasoning Model:" in printed
        assert "当前支持切换的大模型列表" in printed
        assert session._modal_state is not None
        assert session._modal_state["mode"] == "switch_model"

        # Close `/model` modal with Ctrl+C (`handle_ctrl_c_action`)
        assert session.handle_ctrl_c_action("") == "close_modal"
        assert session._modal_state is None

        # 2. User runs `/connect` -> opens Step 1 (`connect_provider`) with all 7 vendors
        await session.handle_line("/connect")
        assert session._modal_state is not None
        assert session._modal_state["mode"] == "connect_provider"
        provider_names = [p["display_name"] for p in session._modal_state["providers"]]
        assert provider_names == ["DeepSeek", "GLM", "MiMo", "MiniMax", "Qwen", "OpenAI", "Anthropic"]

        # Render main content control to populate `_modal_screen_row_to_item` for mouse clicks
        from prompt_toolkit.layout.controls import FormattedTextControl
        main_ctrl = None
        for w in app.layout.find_all_windows():
            if isinstance(w.content, FormattedTextControl):
                txt = "".join(s for _, s in (w.content.text() if callable(w.content.text) else []))
                if "连接大模型厂商 (/connect)" in txt:
                    main_ctrl = w.content
                    break
        assert main_ctrl is not None
        assert len(session._modal_screen_row_to_item) == 7

        # 3. Mouse-click on vendor #5 (`Qwen`, index 4) in the TUI popup!
        qwen_screen_y = next(y for y, idx in session._modal_screen_row_to_item.items() if idx == 4)
        session.handle_tui_mouse_event(
            MouseEvent(
                position=Point(x=10, y=qwen_screen_y),
                event_type=MouseEventType.MOUSE_UP,
                button=MouseButton.LEFT,
                modifiers=frozenset(),
            ),
            source="screen",
        )
        # Immediately transitions to Step 2 (`connect_api_key`) for Qwen!
        assert session._modal_state is not None
        assert session._modal_state["mode"] == "connect_api_key"
        assert session._modal_state["provider_id"] == "qwen"

        # 4. First test invalid API key -> stays on `connect_api_key` with error message
        res_bad = await session.confirm_modal_step(explicit_api_key="invalid-key")
        assert res_bad["ok"] is False
        assert session._modal_state["mode"] == "connect_api_key"
        assert "认证失败" in session._modal_state["error_msg"]

        # 5. Now enter valid API key and press Enter -> verifies and pops up Step 3 (`connect_select_model`)!
        res_ok = await session.confirm_modal_step(explicit_api_key="sk-qwen-live-1234567890")
        assert res_ok["ok"] is True
        assert session._modal_state is not None
        assert session._modal_state["mode"] == "connect_select_model"
        assert session._modal_state["models"] == ["qwen3.8-max", "qwen3-coder-plus", "qwen3.7-plus"]

        # Navigate down 1 row (`qwen3-coder-plus`) with Down arrow and confirm with Enter!
        session.move_modal_selection(1)
        assert session._modal_state["selected_idx"] == 1
        res_final = await session.confirm_modal_step()
        assert res_final["step"] == "completed"
        assert session._modal_state is None
        assert mock_llm.provider_id == "qwen"
        assert mock_llm.fast_model == "qwen3-coder-plus"
        assert mock_llm.reasoning_model == "qwen3-coder-plus"


@pytest.mark.asyncio
async def test_modal_mouse_hover_updates_selected_provider_and_model_with_visual_highlight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Regression test for `t60u`:
    1. When the mouse hovers (`MouseEventType.MOUSE_MOVE` without pressing any button) over a vendor row
       in `/connect` (`connect_provider`) or a model row in `/model` (`switch_model`) or `connect_select_model`,
       `selected_idx` immediately updates to the hovered row.
    2. The hovered row visually changes style (`bg:#1E293B`, `❯` pointer, and `◀ 点击确认` / `◀ 点击切换` / `◀ 点击启用` badge)
       so the user clearly sees which vendor or model the mouse is currently pointing at before clicking.
    3. Opening a modal dynamically writes `[?1003h` (Any-Event Mouse Tracking) to the terminal output,
       and closing/completing the modal restores `[?1003l` so normal chat keeps 0% idle hover CPU.
    """
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
    from prompt_toolkit.output.defaults import DummyOutput
    from fix.cli import create_fix_console

    repo = _init_sample_project(tmp_path)
    buf = StringIO()
    out_console = create_fix_console(file=buf, width=120, force_terminal=False)

    toolkit = WorkspaceToolkit(repo)
    memory = DualMemoryStore(repo, persona_db_path=tmp_path / "persona.db")
    context_mgr = ContextManager()
    mock_llm = ScriptedMockLLM()
    evolution = EvolutionEngine(repo, memory=memory, llm=mock_llm)
    runtime = FixRuntime(llm=mock_llm, toolkit=toolkit, context_mgr=context_mgr, memory=memory, evolution=evolution)

    session = FixInteractiveSession(
        repo=repo,
        stack=(toolkit, memory, context_mgr, mock_llm, evolution, runtime),
        out_console=out_console,
    )
    session._tui_mode = True
    session.render_welcome_banner()

    class _SpyOutput(DummyOutput):
        def __init__(self) -> None:
            super().__init__()
            self.raw_writes: list[str] = []

        def write_raw(self, data: str) -> None:
            self.raw_writes.append(data)

    spy_out = _SpyOutput()
    with create_pipe_input() as pipe_in:
        app = session.build_fullscreen_tui_app(pt_input=pipe_in, pt_output=spy_out)

        # 1. Open `/connect` modal -> verify `?1003h` hover tracking is enabled
        await session.handle_line("/connect")
        assert session._modal_state is not None
        assert session._modal_state["mode"] == "connect_provider"

        main_ctrl = None
        for w in app.layout.find_all_windows():
            if isinstance(w.content, FormattedTextControl):
                txt = "".join(s for _, s in (w.content.text() if callable(w.content.text) else []))
                if "连接大模型厂商 (/connect)" in txt:
                    main_ctrl = w.content
                    break
        assert main_ctrl is not None

        # Hover over vendor #7 (`Anthropic`, index 6) using `MOUSE_MOVE` (no button pressed)
        anthropic_y = next(y for y, idx in session._modal_screen_row_to_item.items() if idx == 6)
        session.handle_tui_mouse_event(
            MouseEvent(
                position=Point(x=12, y=anthropic_y),
                event_type=MouseEventType.MOUSE_MOVE,
                button=MouseButton.NONE,
                modifiers=frozenset(),
            ),
            source="screen",
        )
        assert session._modal_state["selected_idx"] == 6

        # Re-render fragments and verify the hovered Anthropic line has `bg:#1E293B`, `❯ 7.`, and `◀ 点击确认`
        frags_after_hover = main_ctrl.text() if callable(main_ctrl.text) else []
        hovered_anthropic_frags = [
            (st, seg)
            for idx_item, line_frags in session._build_modal_tagged_lines(max_line_cells=110)
            if idx_item == 6
            for st, seg in line_frags
        ]
        hovered_line_text = "".join(seg for _, seg in hovered_anthropic_frags)
        hovered_styles = " ".join(st for st, _ in hovered_anthropic_frags)
        assert "❯ 7." in hovered_line_text
        assert "Anthropic" in hovered_line_text
        assert "◀ 点击确认" in hovered_line_text
        assert "bold #F97316" in hovered_styles and "bg:#1E293B" not in hovered_styles
        assert len(frags_after_hover) > 0

        # 2. Open `/model` modal and verify hovering over model #2 (`idx == 1`) updates `selected_idx` and shows `◀ 点击切换`
        await session.handle_line("/model")
        assert session._modal_state is not None
        assert session._modal_state["mode"] == "switch_model"
        _ = main_ctrl.text() if callable(main_ctrl.text) else []
        model_1_y = next(y for y, idx in session._modal_screen_row_to_item.items() if idx == 1)
        session.handle_tui_mouse_event(
            MouseEvent(
                position=Point(x=15, y=model_1_y),
                event_type=MouseEventType.MOUSE_MOVE,
                button=MouseButton.NONE,
                modifiers=frozenset(),
            ),
            source="screen",
        )
        assert session._modal_state["selected_idx"] == 1
        hovered_model_frags = [
            (st, seg)
            for idx_item, line_frags in session._build_modal_tagged_lines(max_line_cells=110)
            if idx_item == 1
            for st, seg in line_frags
        ]
        assert "◀ 点击切换" in "".join(seg for _, seg in hovered_model_frags)

    # 3. Also verify end-to-end with real `Vt100_Output` and raw SGR hover escape sequences (`[<35;x;yM`)
    # fed through `pipe_in` without any mouse click!
    import asyncio
    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.output.vt100 import Vt100_Output

    session.close_modal()
    term_buf = StringIO()
    vt_out = Vt100_Output(term_buf, lambda: Size(rows=36, columns=110))
    with create_pipe_input() as live_pipe:
        live_app = session.build_fullscreen_tui_app(pt_input=live_pipe, pt_output=vt_out)
        run_task = asyncio.create_task(live_app.run_async())
        try:
            await asyncio.sleep(0.06)
            assert "[?1003h" in term_buf.getvalue()

            live_pipe.send_text("/connect\r")
            await asyncio.sleep(0.08)
            assert session._modal_state is not None
            assert session._modal_state["mode"] == "connect_provider"

            target_row = next(r for r, idx in session._modal_screen_row_to_item.items() if idx == 6)
            before_len = len(term_buf.getvalue())
            live_pipe.send_text(f"\x1b[<35;18;{target_row + 1}M")
            await asyncio.sleep(0.06)
            assert session._modal_state["selected_idx"] == 6
            assert "点击确认" in term_buf.getvalue()[before_len:]
        finally:
            live_app.exit()
            await asyncio.wait_for(run_task, timeout=2.0)
        assert "bold #F97316" in " ".join(st for st, _ in hovered_model_frags) and "bg:#1E293B" not in " ".join(st for st, _ in hovered_model_frags)
        session.close_modal()


def test_per_directory_isolated_up_down_input_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Regression test for `t63u` (Part 1):
    Verifies that Up/Down (`↑`/`↓`) input history is strictly isolated per directory:
    - In directory A (`dir_a`), the user inputs `a`, `b`, `d`.
    - In directory B (`dir_b`), the user inputs `c`, `e`, `f`, `g`.
    - When switching history via Up/Down keys in directory A (even in a brand-new `FixInteractiveSession`),
      it ONLY cycles through `d` <-> `b` <-> `a` (never `c`, `e`, `f`, `g`).
    - When switching history via Up/Down keys in directory B,
      it ONLY cycles through `g` <-> `f` <-> `e` <-> `c` (never `a`, `b`, `d`).
    """
    from types import SimpleNamespace
    from fix.cli import FixInteractiveSession

    monkeypatch.setenv("FIX_PROJECTS_STATE_DIR", str(tmp_path / "projects_state"))

    dir_a = tmp_path / "dir_A"
    dir_b = tmp_path / "dir_B"
    dir_a.mkdir(parents=True)
    dir_b.mkdir(parents=True)

    # 1. Enter `a`, `b`, `d` in directory A
    sess_a1 = FixInteractiveSession(repo=dir_a)
    for item in ("a", "b", "d"):
        sess_a1.record_input_history(item)

    # 2. Enter `c`, `e`, `f`, `g` in directory B
    sess_b1 = FixInteractiveSession(repo=dir_b)
    for item in ("c", "e", "f", "g"):
        sess_b1.record_input_history(item)

    assert sess_a1.history_file_path != sess_b1.history_file_path

    # 3. Open a fresh FixInteractiveSession in directory A and test Up/Down arrow keys
    sess_a2 = FixInteractiveSession(repo=dir_a)
    app_a = sess_a2.build_fullscreen_tui_app()
    buf_a = app_a.layout.current_window.content.buffer
    kb_a = {b.keys: b.handler for b in app_a.key_bindings.bindings}
    ev_a = SimpleNamespace(app=app_a, arg=1)

    assert buf_a.text == ""
    kb_a[("up",)](ev_a)
    assert buf_a.text == "d"
    kb_a[("up",)](ev_a)
    assert buf_a.text == "b"
    kb_a[("up",)](ev_a)
    assert buf_a.text == "a"
    # Pressing Up again at the oldest entry in directory A stays at "a" (never leaks B's history!)
    kb_a[("up",)](ev_a)
    assert buf_a.text == "a"
    assert buf_a.text not in ("c", "e", "f", "g")

    # Pressing Down steps forward through directory A's history back to empty draft
    kb_a[("down",)](ev_a)
    assert buf_a.text == "b"
    kb_a[("down",)](ev_a)
    assert buf_a.text == "d"
    kb_a[("down",)](ev_a)
    assert buf_a.text == ""

    # 4. Open a fresh FixInteractiveSession in directory B and test Up/Down arrow keys
    sess_b2 = FixInteractiveSession(repo=dir_b)
    app_b = sess_b2.build_fullscreen_tui_app()
    buf_b = app_b.layout.current_window.content.buffer
    kb_b = {b.keys: b.handler for b in app_b.key_bindings.bindings}
    ev_b = SimpleNamespace(app=app_b, arg=1)

    assert buf_b.text == ""
    kb_b[("up",)](ev_b)
    assert buf_b.text == "g"
    kb_b[("up",)](ev_b)
    assert buf_b.text == "f"
    kb_b[("up",)](ev_b)
    assert buf_b.text == "e"
    kb_b[("up",)](ev_b)
    assert buf_b.text == "c"
    # Pressing Up again at the oldest entry in directory B stays at "c" (never leaks A's history!)
    kb_b[("up",)](ev_b)
    assert buf_b.text == "c"
    assert buf_b.text not in ("a", "b", "d")

    kb_b[("down",)](ev_b)
    assert buf_b.text == "e"
    kb_b[("down",)](ev_b)
    assert buf_b.text == "f"
    kb_b[("down",)](ev_b)
    assert buf_b.text == "g"
    kb_b[("down",)](ev_b)
    assert buf_b.text == ""


@pytest.mark.asyncio
async def test_codex_style_resume_command_restores_conversation_and_stays_on_latest_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Regression test for `t63u` (Part 2):
    Verifies the Codex-style `/resume` feature:
    1. Automatically saves multi-turn conversations scoped per directory (`dir_a` vs `dir_b`).
    2. In a new `FixInteractiveSession` in `dir_a`, `/resume` opens the interactive session picker modal
       (`mode == "resume_session"`) listing ONLY `dir_a`'s historical sessions (sorted newest first),
       supporting mouse hover (`MOUSE_MOVE`), mouse click (`MOUSE_UP`), and keyboard `↑`/`↓`/`Enter`.
    3. Restoring a conversation via `/resume`:
       - Rebuilds all historical turns (`_screen_blocks`, `conversation_context`, `_turn_fold_states`).
       - Keeps `_scroll_offset_from_bottom == 0` so the viewport stays pinned on the **latest turn**
         of the restored conversation (just like Codex!).
       - Preserves `session_id` so sending a new follow-up message continues the restored conversation
         with full prior context and updates the same session record.
    """
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
    from prompt_toolkit.output.defaults import DummyOutput
    import fix.cli as cli_mod
    from fix.cli import FixInteractiveSession, create_fix_console

    monkeypatch.setenv("FIX_PROJECTS_STATE_DIR", str(tmp_path / "projects_state"))
    monkeypatch.setattr(cli_mod, "_query_os_terminal_size", lambda: (110, 20))

    dir_a = _init_sample_project(tmp_path / "workspace_a")
    dir_b = _init_sample_project(tmp_path / "workspace_b")

    def _make_session_for(repo_dir: Path, responses: list[LLMResponse]) -> tuple[FixInteractiveSession, ScriptedMockLLM, StringIO]:
        buf = StringIO()
        out_c = create_fix_console(file=buf, width=110, force_terminal=False)
        tk = WorkspaceToolkit(repo_dir)
        mem = DualMemoryStore(repo_dir, persona_db_path=tmp_path / "persona.db")
        cm = ContextManager()
        m_llm = ScriptedMockLLM(
            tool_Queue=list(responses),
            structured_handlers={
                "ProbeDecision": ProbeDecision(intent=TaskIntent.EXPLAIN, summary="Analyze repo"),
                "ReviewVerdict": ReviewVerdict(approved=True, confidence="HIGH"),
            },
        )
        evo = EvolutionEngine(repo_dir, memory=mem, llm=m_llm)
        rt = FixRuntime(llm=m_llm, toolkit=tk, context_mgr=cm, memory=mem, evolution=evo)
        sess = FixInteractiveSession(
            repo=repo_dir,
            stack=(tk, mem, cm, m_llm, evo, rt),
            out_console=out_c,
        )
        return sess, m_llm, buf

    # 1. Run an older conversation (Session 1) in dir_a
    sess_a_old, _, _ = _make_session_for(
        dir_a,
        [LLMResponse(content="旧会话回答：这是第一版架构概览。", tool_calls=[])],
    )
    await sess_a_old.handle_line("旧会话问题：梳理第一版架构")
    old_session_id = sess_a_old.session_id

    # 2. Run a newer 2-turn conversation (Session 2) in dir_a with multi-line output
    long_turn1_answer = "\n".join(
        [f"第一轮分析详情第 {i:02d} 行：`app.py` 中的 `greet` 函数基础逻辑说明。" for i in range(1, 25)]
    )
    sess_a_latest, _, _ = _make_session_for(
        dir_a,
        [
            LLMResponse(
                content="我先读取 app.py 文件。",
                tool_calls=[ToolCallRequest(name="read_file", arguments={"rel_path": "app.py"})],
            ),
            LLMResponse(content=long_turn1_answer, tool_calls=[]),
            LLMResponse(
                content="## 最新第二轮结论 (Latest Turn 2)\n\n`greet` 的参数类型是 `str`，返回值格式为 `Hello, {name}`。",
                tool_calls=[],
            ),
        ],
    )
    await sess_a_latest.handle_line("最新会话第1问：详细分析 app.py 的实现")
    await sess_a_latest.handle_line("最新会话第2问：确认 greet 的返回值格式")
    latest_session_id = sess_a_latest.session_id
    assert latest_session_id != old_session_id

    # 3. Also run a conversation in dir_b (must NOT appear in dir_a's /resume list!)
    sess_b, _, _ = _make_session_for(
        dir_b,
        [LLMResponse(content="B目录回答：只属于B目录。", tool_calls=[])],
    )
    await sess_b.handle_line("B目录问题：查看B项目")

    # 4. Start a brand-new FixInteractiveSession in dir_a and open Full-Screen TUI
    sess_a_resumed, mock_llm_resumed, _ = _make_session_for(
        dir_a,
        [
            LLMResponse(
                content="## 第三轮跟进回答 (Turn 3 Follow-up)\n\n已结合前两轮关于 `greet` 的上下文完成扩展分析。",
                tool_calls=[],
            )
        ],
    )
    with create_pipe_input() as pipe_in:
        app = sess_a_resumed.build_fullscreen_tui_app(pt_input=pipe_in, pt_output=DummyOutput())

        saved_list = sess_a_resumed.list_saved_sessions()
        assert len(saved_list) == 2
        assert saved_list[0]["session_id"] == latest_session_id
        assert saved_list[0]["turn_count"] == 2
        assert saved_list[1]["session_id"] == old_session_id
        assert saved_list[1]["turn_count"] == 1
        assert all("B目录" not in s["latest_prompt"] for s in saved_list)

        # Simulate user having scrolled up before running `/resume`
        sess_a_resumed._scroll_offset_from_bottom = 18

        # Run `/resume` (no args) in TUI mode -> opens interactive `resume_session` modal
        await sess_a_resumed.handle_line("/resume")
        assert sess_a_resumed._modal_state is not None
        assert sess_a_resumed._modal_state["mode"] == "resume_session"
        assert sess_a_resumed._modal_state["selected_idx"] == 0

        # Render main content control to populate `_modal_screen_row_to_item`
        main_ctrl = None
        for w in app.layout.find_all_windows():
            if isinstance(w.content, FormattedTextControl):
                txt = "".join(s for _, s in (w.content.text() if callable(w.content.text) else []))
                if "恢复历史对话 (/resume" in txt:
                    main_ctrl = w.content
                    break
        assert main_ctrl is not None
        assert len(sess_a_resumed._modal_screen_row_to_item) == 2

        # Verify mouse hover (`MOUSE_MOVE`) over session #2 (index 1) highlights it with `bg:#1E293B` and `◀ 点击恢复`
        row_y_1 = next(y for y, idx in sess_a_resumed._modal_screen_row_to_item.items() if idx == 1)
        sess_a_resumed.handle_tui_mouse_event(
            MouseEvent(
                position=Point(x=12, y=row_y_1),
                event_type=MouseEventType.MOUSE_MOVE,
                button=MouseButton.NONE,
                modifiers=frozenset(),
            ),
            source="screen",
        )
        assert sess_a_resumed._modal_state["selected_idx"] == 1
        hovered_frags = [
            (st, seg)
            for idx_item, line_frags in sess_a_resumed._build_modal_tagged_lines(max_line_cells=106)
            if idx_item == 1
            for st, seg in line_frags
        ]
        assert "◀ 点击恢复" in "".join(seg for _, seg in hovered_frags)
        assert "bold #F97316" in " ".join(st for st, _ in hovered_frags) and "bg:#1E293B" not in " ".join(st for st, _ in hovered_frags)

        # Hover back to session #1 (latest, index 0) and click (`MOUSE_UP`) to restore it!
        row_y_0 = next(y for y, idx in sess_a_resumed._modal_screen_row_to_item.items() if idx == 0)
        sess_a_resumed.handle_tui_mouse_event(
            MouseEvent(
                position=Point(x=12, y=row_y_0),
                event_type=MouseEventType.MOUSE_MOVE,
                button=MouseButton.NONE,
                modifiers=frozenset(),
            ),
            source="screen",
        )
        assert sess_a_resumed._modal_state["selected_idx"] == 0
        sess_a_resumed.handle_tui_mouse_event(
            MouseEvent(
                position=Point(x=12, y=row_y_0),
                event_type=MouseEventType.MOUSE_UP,
                button=MouseButton.LEFT,
                modifiers=frozenset(),
            ),
            source="screen",
        )

        # Modal closes and Session 2 (2 turns) is restored!
        assert sess_a_resumed._modal_state is None
        assert sess_a_resumed.session_id == latest_session_id
        assert len(sess_a_resumed.conversation_context) == 2
        assert sess_a_resumed.conversation_context[0]["user"] == "最新会话第1问：详细分析 app.py 的实现"
        assert sess_a_resumed.conversation_context[1]["user"] == "最新会话第2问：确认 greet 的返回值格式"

        # Verify Codex-style positioning: `_scroll_offset_from_bottom` is 0 and visible screen shows the LATEST turn!
        assert sess_a_resumed._scroll_offset_from_bottom == 0
        transcript_ctrl = app.layout.container.content.children[0].content
        visible_text_at_bottom = "".join(
            s for _, s in (transcript_ctrl.text() if callable(transcript_ctrl.text) else [])
        )
        assert "最新会话第2问：确认 greet 的返回值格式" in visible_text_at_bottom
        assert "最新第二轮结论 (Latest Turn 2)" in visible_text_at_bottom
        # Earlier Turn 1 is also in `_cached_render_lines` (accessible by scrolling up!)
        all_doc_text = "\n".join("".join(seg for _, seg in ln) for ln in sess_a_resumed._cached_render_lines)
        assert "最新会话第1问：详细分析 app.py 的实现" in all_doc_text
        assert "▸ 已自动收起执行过程 (#1:" in all_doc_text

        # 5. Send a 3rd follow-up prompt in the resumed session -> continues with prior context and updates session on disk!
        await sess_a_resumed.handle_line("最新会话第3问：基于前面的分析继续扩展")
        assert len(sess_a_resumed.conversation_context) == 3
        sys_prompt_sent = mock_llm_resumed.recorded_messages[-1][0]["content"]
        assert "最新会话第1问：详细分析 app.py 的实现" in sys_prompt_sent
        assert "最新会话第2问：确认 greet 的返回值格式" in sys_prompt_sent

        updated_sessions = sess_a_resumed.list_saved_sessions()
        assert updated_sessions[0]["session_id"] == latest_session_id
        assert updated_sessions[0]["turn_count"] == 3
        assert updated_sessions[0]["latest_prompt"] == "最新会话第3问：基于前面的分析继续扩展"


@pytest.mark.asyncio
async def test_claude_code_style_bottom_left_working_status_display(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Regression test for `t64u`:
    Verifies that when Fix starts working (`开始干活时`), the bottom-left status indicator
    follows Claude Code's working status display:
    1. Uses Claude Code's animated starburst/sparkle spinner sequence (`CLAUDE_SPINNER_FRAMES`:
       `·`, `✢`, `✳`, `✶`, `✻`, `✽`, `✻`, `✶`, `✳`, `✢`) styled in warm terracotta (`bold #D97757`).
    2. Displays context-aware `正在xxx...` working phrases across all phases & tools
       (`正在思考与分析代码...`, `正在理解需求与分析项目上下文...`, `正在读取 ...`, `正在搜索代码 ...`,
       `正在执行命令 ...`, `正在编辑代码 ...`, `正在运行验证 ...`, `正在结合工具输出推理下一步...`,
       `正在汇总代码证据并撰写完整分析报告...`) with a live left-to-right shimmer wave (`#D97757` / `#FDBA74` / `#FEF3C7`).
    3. Appends Claude Code's dim secondary telemetry tail `(esc/ctrl+c 中断 · N 次工具 · ↓ X.Xk tokens)` (`#6B7280`)
       after the elapsed time (`· Xs`).
    4. Supports pressing `Esc` (as well as `Ctrl+C`) to interrupt/cancel an in-flight task.
    """
    import asyncio
    import time
    import asyncio
    from prompt_toolkit.application.current import set_app
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.output.defaults import DummyOutput
    import fix.cli as cli_mod
    from fix.cli import (
        CLAUDE_SPINNER_FRAMES,
        CollapsibleProgressTracker,
        FixInteractiveSession,
        _build_claude_shimmer_fragments,
        format_claude_working_action,
    )

    assert CLAUDE_SPINNER_FRAMES == ("·", "✢", "✳", "✶", "✻", "✽", "✻", "✶", "✳", "✢")

    # 1. Verify tool-specific `正在xxx...` working status formatting
    assert format_claude_working_action("read_file", {"rel_path": "src/fix/cli.py"}) == "正在读取 cli.py..."
    assert format_claude_working_action("search_code", {"pattern": "CollapsibleProgressTracker"}) == "正在搜索代码 CollapsibleProgressTracker..."
    assert format_claude_working_action("list_dir", {"rel_path": "src/fix"}) == "正在浏览目录 src/fix..."
    assert format_claude_working_action("run_command", {"cmd": "pytest -q"}) == "正在执行命令 pytest -q..."
    assert format_claude_working_action("apply_patch", {"rel_path": "src/fix/runtime.py"}) == "正在编辑代码 runtime.py..."
    assert format_claude_working_action("write_new_file", {"rel_path": "tests/new_test.py"}) == "正在写入文件 new_test.py..."

    # 2. Verify shimmer wave fragments preserve exact text while applying Claude Code's terracotta + highlight styles
    shimmer = _build_claude_shimmer_fragments("正在思考与分析代码...", elapsed=0.3)
    assert "".join(seg for _, seg in shimmer) == "正在思考与分析代码..."
    shimmer_styles = {st for st, _ in shimmer}
    assert "bold #D97757" in shimmer_styles
    assert ("bold #FEF3C7" in shimmer_styles) or ("bold #FDBA74" in shimmer_styles)

    # 3. Verify progressive initial thinking stages & live tool/token telemetry in `get_left_status_text` and TUI status row
    cols, rows = 120, 30
    monkeypatch.setattr(cli_mod, "_query_os_terminal_size", lambda: (cols, rows))

    repo = _init_sample_project(tmp_path)
    session = FixInteractiveSession(repo=repo)
    with create_pipe_input() as pipe_in:
        tui_app = session.build_fullscreen_tui_app(pt_input=pipe_in, pt_output=DummyOutput())
        tracker = CollapsibleProgressTracker(session.console)
        t_now = time.time()
        tracker.started_at = t_now - 1.0
        session._active_tracker = tracker

        # At 1s: initial thinking
        st_1s = session.get_left_status_text()
        assert st_1s[0] in CLAUDE_SPINNER_FRAMES
        assert "正在思考与分析代码... · 1s (esc/ctrl+c 中断)" in st_1s

        # At 4.5s (still waiting for first LLM response): progresses to understanding & context analysis
        tracker.started_at = t_now - 4.5
        st_4s = session.get_left_status_text()
        assert "正在理解需求与分析项目上下文... · 4s (esc/ctrl+c 中断)" in st_4s

        # Emit tool_start + telemetry -> shows `正在读取 ...` + tool count + token count
        tracker.on_event("telemetry", {"total_tokens": 1450, "prompt_tokens": 1200, "completion_tokens": 250, "llm_calls": 1, "step": 1})
        tracker.on_event("tool_start", {"name": "read_file", "arguments": {"rel_path": "src/fix/cli.py"}})
        st_tool = session.get_left_status_text()
        assert "正在读取 cli.py... · 4s (esc/ctrl+c 中断 · 1 次工具 · ↓ 1.4k tokens)" in st_tool

        # Render TUI status row and verify multi-styled fragments (terracotta spinner + shimmer action + slate elapsed + dim metadata)
        with set_app(tui_app):
            tui_app.renderer.render(tui_app, tui_app.layout)
        screen = tui_app.renderer._last_screen
        assert screen is not None
        rendered_rows = [
            "".join(screen.data_buffer[y][x].char for x in range(cols)).rstrip()
            for y in range(rows)
        ]
        full_screen_text = "\n".join(rendered_rows)
        assert "正在读取 cli.py... · 4s (esc/ctrl+c 中断 · 1 次工具 · ↓ 1.4k tokens)" in full_screen_text

        session._active_tracker = None
        assert "● 就绪 · " in session.get_left_status_text()


@pytest.mark.asyncio
async def test_codex_style_new_command_starts_fresh_conversation_in_current_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verifies the Codex-style `/new` command (`t67u`):
    1. Typing `/new` in an active session immediately starts a brand-new conversation session in the current directory:
       - Allocates a fresh `session_id`
       - Clears `conversation_context`, `_saved_turns_records`, `_turn_fold_states`, and resets `_screen_blocks` to the header
       - Preserves the previous conversation session on disk so `/resume` can restore it anytime
       - Preserves the current directory's Up/Down prompt input history (`FileHistory`)
    2. Also supports `/new <prompt>` to start a fresh conversation and immediately execute `<prompt>` in that new session.
    """
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output.defaults import DummyOutput
    import fix.cli as cli_mod
    from fix.cli import FixInteractiveSession, create_fix_console

    monkeypatch.setenv("FIX_PROJECTS_STATE_DIR", str(tmp_path / "projects_state_new"))
    monkeypatch.setattr(cli_mod, "_query_os_terminal_size", lambda: (110, 20))

    dir_a = _init_sample_project(tmp_path / "workspace_new")
    buf = StringIO()
    out_c = create_fix_console(file=buf, width=110, force_terminal=False)
    tk = WorkspaceToolkit(dir_a)
    mem = DualMemoryStore(dir_a, persona_db_path=tmp_path / "persona_new.db")
    cm = ContextManager()
    m_llm = ScriptedMockLLM(
        tool_Queue=[
            LLMResponse(content="旧会话第一轮回答 `[src/app.py:1]`。", tool_calls=[], total_tokens=40),
            LLMResponse(content="新会话第一轮回答 `[src/app.py:2]`。", tool_calls=[], total_tokens=45),
            LLMResponse(content="带参数 /new 直接开启的新会话回答 `[src/app.py:3]`。", tool_calls=[], total_tokens=50),
        ],
        structured_handlers={
            "ProbeDecision": ProbeDecision(intent=TaskIntent.EXPLAIN, summary="Analyze repo"),
            "ReviewVerdict": ReviewVerdict(approved=True, confidence="HIGH"),
        },
    )
    evo = EvolutionEngine(dir_a, memory=mem, llm=m_llm)
    rt = FixRuntime(llm=m_llm, toolkit=tk, context_mgr=cm, memory=mem, evolution=evo)
    sess = FixInteractiveSession(
        repo=dir_a,
        stack=(tk, mem, cm, m_llm, evo, rt),
        out_console=out_c,
    )

    with create_pipe_input() as pipe_in:
        _ = sess.build_fullscreen_tui_app(pt_input=pipe_in, pt_output=DummyOutput())

        # 1. Run a turn in the first session
        first_sid = sess.session_id
        await sess.handle_line("旧会话问题：检查 app.py")
        assert len(sess.conversation_context) == 1
        assert len(sess._saved_turns_records) == 1
        assert len(sess._screen_blocks) == 2

        # 2. Execute `/new` -> starts a brand-new session in the same directory while keeping the first session saved
        sess._scroll_offset_from_bottom = 12
        await sess.handle_line("/new")
        second_sid = sess.session_id
        assert second_sid != first_sid
        assert len(sess.conversation_context) == 0
        assert len(sess._saved_turns_records) == 0
        assert len(sess._turn_fold_states) == 0
        assert len(sess._screen_blocks) == 1
        assert sess._scroll_offset_from_bottom == 0
        assert "新建对话" in sess._toast_message

        # Previous session is still saved and visible in `list_saved_sessions()`
        saved_after_new = sess.list_saved_sessions()
        assert len(saved_after_new) == 1
        assert saved_after_new[0]["session_id"] == first_sid

        # Prompt history for the current directory is still preserved and `/new` itself is not recorded as a prompt
        assert sess._file_history.get_strings() == ["旧会话问题：检查 app.py"]

        # 3. Ask a question in the new session -> does NOT include old session's conversation_context!
        await sess.handle_line("新会话问题：全新的独立话题")
        assert len(sess.conversation_context) == 1
        assert sess.conversation_context[0]["user"] == "新会话问题：全新的独立话题"
        sys_prompt_new = m_llm.recorded_messages[-1][0]["content"]
        assert "旧会话问题：检查 app.py" not in sys_prompt_new

        # Both sessions now exist in `/resume` for this directory
        saved_two = sess.list_saved_sessions()
        assert len(saved_two) == 2
        assert {s["session_id"] for s in saved_two} == {first_sid, second_sid}

        # 4. Also test `/new <prompt>` (start a 3rd new conversation and immediately run the prompt)
        await sess.handle_line("/new 第三个会话直接提问")
        third_sid = sess.session_id
        assert third_sid not in {first_sid, second_sid}
        assert len(sess.conversation_context) == 1
        assert sess.conversation_context[0]["user"] == "第三个会话直接提问"
        assert len(sess.list_saved_sessions()) == 3



@pytest.mark.asyncio
async def test_codex_style_input_composer_and_on_enter_submitted_user_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verifies the OpenAI Codex CLI style input box (`ChatComposer`) and on-Enter submitted
    user prompt (`UserHistoryCell`) (`t68u`):
    1. Composer box structure & styling:
       - Framed symmetrically by top border (`divider_row`) and bottom border (`composer_bottom_row`)
       - Uses Codex chevron ` › ` (`bold #0EA5E9` when idle, `#6b7280` when task running) on line 0
         and `   ` (3 spaces) on wrapped/continuation lines
       - Displays Codex placeholder `Ask Fix to do anything · 输入问题或 / 命令` when empty
       - Highlights leading `/command` tokens in `bold #0EA5E9` inside the input box
    2. Codex footer hint bar (`footer_row`) below the composer:
       - Empty composer: `  ? for shortcuts · / commands · ↑↓ history` (left) + `<model> · 100% context left  ` (right)
       - Draft composer: `  enter to send · ctrl+j newline · ctrl+c clear` (left) + `<model> · 100% context left  ` (right)
       - Pressing `?` on empty composer toggles the 3-column `Keyboard shortcuts` (`Compose` / `Session` / `Transcript`)
         overlay above the composer; pressing `?` when text is non-empty inserts `?` into the draft.
       - Typing `/re` opens the Codex Slash Command Popup (` › /resume ...`) above the composer with index 0
         pre-selected so pressing `Tab` or `Enter` immediately completes/executes `/resume`.
    3. On-Enter submitted user prompt (`UserHistoryCell` style in `render_codex_user_prompt_to`):
       - Renders with a top blank spacing line, `› ` (`bold #0EA5E9`) on line 1, `  ` (2 spaces) indent on
         all soft-wrapped continuation lines and explicit `\n` lines (never wrapping under `›` at column 0),
         crisp native primary foreground (`""`) on the prompt text, and a bottom blank spacing line before the answer.
    """
    from types import SimpleNamespace
    import asyncio
    from prompt_toolkit.application.current import set_app
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.output.defaults import DummyOutput
    import fix.cli as cli_mod
    from fix.cli import (
        CODEX_PLACEHOLDER_TEXT,
        FixInteractiveSession,
        create_fix_console,
        wrap_codex_user_prompt_lines,
    )

    cols, rows = 96, 26
    monkeypatch.setattr(cli_mod, "_query_os_terminal_size", lambda: (cols, rows))

    repo = _init_sample_project(tmp_path / "codex_ui_repo")
    buf_out = StringIO()
    out_c = create_fix_console(file=buf_out, width=cols, force_terminal=False)
    tk = WorkspaceToolkit(repo)
    mem = DualMemoryStore(repo, persona_db_path=tmp_path / "persona_codex_ui.db")
    cm = ContextManager()
    m_llm = ScriptedMockLLM(
        tool_Queue=[
            LLMResponse(content="已完成 Codex 风格输入框与回车上屏样式验证 `[app.py:1]`。", tool_calls=[], total_tokens=1200),
        ],
        structured_handlers={
            "ProbeDecision": ProbeDecision(intent=TaskIntent.EXPLAIN, summary="Verify Codex UI"),
            "ReviewVerdict": ReviewVerdict(approved=True, confidence="HIGH"),
        },
    )
    evo = EvolutionEngine(repo, memory=mem, llm=m_llm)
    rt = FixRuntime(llm=m_llm, toolkit=tk, context_mgr=cm, memory=mem, evolution=evo)
    sess = FixInteractiveSession(
        repo=repo,
        stack=(tk, mem, cm, m_llm, evo, rt),
        out_console=out_c,
    )

    from prompt_toolkit.data_structures import Size

    dummy_out = DummyOutput()
    dummy_out.get_size = lambda: Size(rows=rows, columns=cols)  # type: ignore[method-assign]
    with create_pipe_input() as pipe_in:
        tui_app = sess.build_fullscreen_tui_app(pt_input=pipe_in, pt_output=dummy_out)
        hsplit = tui_app.layout.container.content
        assert len(hsplit.children) == 6
        input_win = hsplit.children[3]
        buf = input_win.content.buffer
        kb_map = {tuple(b.keys): b.handler for b in tui_app.key_bindings.bindings}
        ev = SimpleNamespace(app=tui_app, arg=1, current_buffer=buf)

        def _render_screen_rows() -> list[str]:
            with set_app(tui_app):
                tui_app.renderer.render(tui_app, tui_app.layout)
            sc = tui_app.renderer._last_screen
            assert sc is not None
            return [
                "".join(sc.data_buffer[y][x].char for x in range(cols)).rstrip()
                for y in range(rows)
            ]

        # 1. Verify initial empty Codex composer state & footer bar
        rows_empty = _render_screen_rows()
        screen_empty = "\n".join(rows_empty)
        assert f" › {CODEX_PLACEHOLDER_TEXT}" in screen_empty
        assert "? for shortcuts · / commands · ↑↓ history" in rows_empty[-1]
        assert "100% context left" in rows_empty[-1]

        # 2. Press `?` when composer is empty -> toggles Codex `Keyboard shortcuts` overlay above composer
        kb_map[("?",)](ev)
        assert sess._shortcut_overlay_open is True
        rows_shortcuts = _render_screen_rows()
        screen_shortcuts = "\n".join(rows_shortcuts)
        assert "Keyboard shortcuts" in screen_shortcuts
        assert "Compose" in screen_shortcuts and "Session" in screen_shortcuts and "Transcript" in screen_shortcuts
        assert "? / esc close shortcuts" in rows_shortcuts[-1]

        # Press `escape` -> closes shortcut overlay
        kb_map[("escape",)](ev)
        assert sess._shortcut_overlay_open is False

        # 3. Type `/n` -> opens Codex Slash Command Popup above composer with ` › /new` pre-selected!
        buf.text = "/n"
        buf.cursor_position = len(buf.text)
        rows_slash = _render_screen_rows()
        screen_slash = "\n".join(rows_slash)
        assert " › /new" in screen_slash
        assert "↑↓ select · enter run · tab complete · esc close" in rows_slash[-1]

        # Press `tab` -> completes `/new` into the composer buffer and closes popup
        kb_map[("c-i",)](ev)
        assert buf.text == "/new"
        buf.reset()

        # 4. Verify On-Enter Submitted User Prompt (`UserHistoryCell` style: `› ` on line 1, `  ` on wrapped & newline lines)
        long_multiline_prompt = (
            "你是一名资深的android车机launcher开发，现在需要你深入分析下桌面上的任务卡片在接收到语音卡片后，他是如何显示的，为什么语音大卡没有显示\n"
            "第二行补充要求：必须包含准确的代码行号区间(:start-end)"
        )
        wrapped_preview = wrap_codex_user_prompt_lines(long_multiline_prompt, max_body_cells=60)
        assert len(wrapped_preview) >= 3
        assert wrapped_preview[-1] == "第二行补充要求：必须包含准确的代码行号区间(:start-end)"

        await sess.handle_line(long_multiline_prompt)
        rows_after_turn = _render_screen_rows()

        # Verify in `_cached_render_lines` that the first prompt line starts with `› ` and wrapped/second lines start with `  `
        all_doc_lines = ["".join(seg for _, seg in ln) for ln in sess._cached_render_lines]
        chevron_idx = next(i for i, ln in enumerate(all_doc_lines) if ln.startswith("› 你是一名资深的android"))
        # Blank spacing line immediately above `› ...`
        assert all_doc_lines[chevron_idx - 1].strip() == ""
        # Wrapped continuation line & explicit second line are indented by 2 spaces (`  `), never starting at column 0!
        assert all_doc_lines[chevron_idx + 1].startswith("  ")
        # Find the explicit second line and verify it also starts with 2 spaces (`  `)
        sec_idx = next(i for i, ln in enumerate(all_doc_lines) if "第二行补充要求" in ln)
        assert all_doc_lines[sec_idx].startswith("  ")
        # Blank spacing line immediately after the user prompt block (before `▸` or final markdown)
        assert all_doc_lines[sec_idx + 1].strip() == ""
        assert any("已完成 Codex 风格输入框与回车上屏样式验证" in ln for ln in rows_after_turn)



@pytest.mark.asyncio
async def test_claude_code_style_in_input_command_and_argument_interaction_system(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verifies the Claude Code style command system & in-input command interaction (`t70u`):
    1. Bottom Input Box Command Dock (`command_dock_window` inside `composer_bottom_dock`):
       - Typing `/` or `?` or opening `/resume`, `/model`, `/connect` renders the interactive panel
         DIRECTLY BELOW the input box (`command_dock_window`), NEVER polluting the upper conversation
         transcript (`main_content_window` / 上屏界面).
    2. Multi-dimension fuzzy & Chinese/English keyword search in the input box:
       - `/恢复` or `/历史` -> `/resume`
       - `/模型` or `/切换` -> `/model`
       - `/新建` -> `/new`
       - `/连接` or `/厂商` -> `/connect`
       - `/撤回` or `/回滚` -> `/undo`
       - `/规则` -> `/rules`
       - `/技能` -> `/skills`
    3. In-Input Live Ghost Completion & Preview (`_ClaudeInlineGhostProcessor`):
       - Typing `/re` shows inline ghost completion `sume [序号|关键词]  — 恢复当前目录的历史对话...`
         right inside the input box, and pressing `Right` arrow (`→`) or `Tab` accepts the completion
         and transitions straight into Stage 2 (In-Input Argument & Option Picker).
    4. Stage 2 In-Input Interactive Option Picker (for `/model`, `/resume`, `/connect`, `/rules`, `/skills`, `/idle`, `/steps`, `/diag`, `/ask`, `/bug`, `/opt`, `/trace`):
       - Completing `/model` via `Tab` (or typing `/model `) lists switchable models + `+ 连接其他大模型厂商 (/connect)`
         below the input box; typing a filter word (e.g. `flash`) live-filters the options; pressing `Enter`
         immediately switches the active model right from the input box!
       - Completing `/resume` via `Tab` (or typing `/resume `) lists saved conversations for the current directory;
         typing a keyword filters conversations; pressing `Enter` immediately restores the conversation!
       - Pressing `Enter` on `/ask`, `/bug`, `/opt`, or `/trace` from Stage 1 when no argument is typed yet
         completes `/ask ` into the input box and opens Stage 2 inline prompt guidance instead of failing!
    5. Mid-input `/` command autocomplete (Claude Code v2.1+ feature):
       - Typing `请帮我检查 /tr` opens the mid-input command dock for `/trace`, and pressing `Tab` or `Enter`
         completes `请帮我检查 /trace ` in-place without prematurely submitting the prompt.
    6. Live input box filtering when `/resume`, `/model`, or `/connect` modal (`_modal_state`) is open:
       - Typing a keyword (e.g. `anthropic` or `架构`) in the input box while the modal is open live-filters
         and jumps `selected_idx` to the matching item in real time!
    """
    from types import SimpleNamespace
    import asyncio
    from prompt_toolkit.application.current import set_app
    from prompt_toolkit.data_structures import Point, Size
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
    from prompt_toolkit.output.defaults import DummyOutput
    import fix.cli as cli_mod
    from fix.cli import FixInteractiveSession, create_fix_console

    monkeypatch.setenv("FIX_PROJECTS_STATE_DIR", str(tmp_path / "projects_state_cc"))
    cols, rows = 104, 28
    monkeypatch.setattr(cli_mod, "_query_os_terminal_size", lambda: (cols, rows))

    repo = _init_sample_project(tmp_path / "cc_cmd_repo")
    buf_out = StringIO()
    out_c = create_fix_console(file=buf_out, width=cols, force_terminal=False)
    tk = WorkspaceToolkit(repo)
    mem = DualMemoryStore(repo, persona_db_path=tmp_path / "persona_cc.db")
    cm = ContextManager()
    m_llm = ScriptedMockLLM(
        tool_Queue=[
            LLMResponse(content="第一轮历史会话回答 `[app.py:1]`。", tool_calls=[], total_tokens=100),
            LLMResponse(content="第二轮新会话回答 `[app.py:2]`。", tool_calls=[], total_tokens=120),
        ],
        structured_handlers={
            "ProbeDecision": ProbeDecision(intent=TaskIntent.EXPLAIN, summary="Verify Claude Code UX"),
            "ReviewVerdict": ReviewVerdict(approved=True, confidence="HIGH"),
        },
    )
    evo = EvolutionEngine(repo, memory=mem, llm=m_llm)
    rt = FixRuntime(llm=m_llm, toolkit=tk, context_mgr=cm, memory=mem, evolution=evo)
    sess = FixInteractiveSession(
        repo=repo,
        stack=(tk, mem, cm, m_llm, evo, rt),
        out_console=out_c,
    )

    # Seed two historical sessions in this directory so `/resume` has multiple sessions to filter & pick
    await sess.handle_line("历史会话一：梳理车机Launcher卡片架构")
    first_sid = sess.session_id
    await sess.handle_line("/new 历史会话二：排查语音大卡不显示问题")
    second_sid = sess.session_id
    await sess.handle_line("/new")
    assert len(sess.list_saved_sessions()) == 2

    dummy_out = DummyOutput()
    dummy_out.get_size = lambda: Size(rows=rows, columns=cols)  # type: ignore[method-assign]
    with create_pipe_input() as pipe_in:
        tui_app = sess.build_fullscreen_tui_app(pt_input=pipe_in, pt_output=dummy_out)
        hsplit = tui_app.layout.container.content
        main_win = hsplit.children[0]
        input_win = hsplit.children[3]
        bottom_dock = hsplit.children[4]
        dock_win = bottom_dock.children[1]
        buf = input_win.content.buffer
        kb_map = {tuple(b.keys): b.handler for b in tui_app.key_bindings.bindings}
        ev = SimpleNamespace(app=tui_app, arg=1, current_buffer=buf)

        def _render_screen_rows() -> list[str]:
            with set_app(tui_app):
                tui_app.renderer.render(tui_app, tui_app.layout)
            sc = tui_app.renderer._last_screen
            assert sc is not None
            return [
                "".join(sc.data_buffer[y][x].char for x in range(cols)).rstrip()
                for y in range(rows)
            ]

        def _get_main_text() -> str:
            return "".join(s for _, s in (main_win.content.text() if callable(main_win.content.text) else []))

        def _get_dock_text() -> str:
            return "".join(s for _, s in (dock_win.content.text() if callable(dock_win.content.text) else []))

        # 1. Type `/` -> command menu opens in `dock_win` (BELOW the input box), NEVER in `main_win` (上屏界面)!
        buf.text = "/"
        buf.cursor_position = 1
        screen_rows = _render_screen_rows()
        main_txt = _get_main_text()
        dock_txt = _get_dock_text()
        assert "命令交互面板 (/ Commands)" not in main_txt
        assert " › /new" not in main_txt
        assert "命令交互面板 (/ Commands)" in dock_txt
        assert " › /new" in dock_txt
        assert "/resume" in dock_txt and "/model" in dock_txt and "/connect" in dock_txt
        # Verify on the rendered screen that ` › /new` appears on a row BELOW the ` › /` input box row!
        input_row_y = next(i for i, r in enumerate(screen_rows) if r.startswith(" › /"))
        cmd_row_y = next(i for i, r in enumerate(screen_rows) if i > input_row_y and r.startswith(" › /new"))
        assert cmd_row_y > input_row_y

        # Verify inline ghost completion is rendered right inside the input box row (` › /new [新对话问题]...`)
        assert " › /new" in screen_rows[input_row_y]

        # 2. Verify Chinese keyword filtering in the input box (`/恢复` -> `/resume`, `/模型` -> `/model`, `/回滚` -> `/undo`)
        assert sess.match_slash_commands("/恢复")[0]["cmd"] == "/resume"
        assert sess.match_slash_commands("/模型")[0]["cmd"] == "/model"
        assert sess.match_slash_commands("/回滚")[0]["cmd"] == "/undo"
        assert sess.match_slash_commands("/规则")[0]["cmd"] == "/rules"
        assert sess.match_slash_commands("/技能")[0]["cmd"] == "/skills"

        # 3. Type `/m` -> matches `/model`; press `Tab` (`c-i`) -> completes `/model` and transitions `dock_win` to Stage 2 model picker!
        buf.text = "/m"
        buf.cursor_position = len(buf.text)
        _render_screen_rows()
        assert " › /model" in _get_dock_text()

        kb_map[("c-i",)](ev)
        assert buf.text == "/model"
        _render_screen_rows()
        dock_stage2_model = _get_dock_text()
        assert "/model 交互选参" in dock_stage2_model
        assert "mimo-v2.6-pro" in dock_stage2_model
        assert "mimo-v2.6-flash" in dock_stage2_model
        assert "连接其他大模型厂商 (/connect)" in dock_stage2_model

        # Type `flash` directly after pressing Tab -> auto-inserts space (`/model flash`) and live-filters to `mimo-v2.6-flash`!
        buf.insert_text("flash")
        assert buf.text == "/model flash"
        _render_screen_rows()
        dock_filtered_model = _get_dock_text()
        assert "mimo-v2.6-flash" in dock_filtered_model
        assert "mimo-v2.6-pro" not in dock_filtered_model

        # Press `Enter` (`c-m`) -> immediately switches active model to `mimo-v2.6-flash` right from the input box!
        kb_map[("c-m",)](ev)
        assert buf.text == ""
        assert m_llm.fast_model == "mimo-v2.6-flash"

        # 4. Type `/re` and press `Right` arrow (`→`) at end of buffer -> completes `/resume` and opens Stage 2 session picker!
        buf.text = "/re"
        buf.cursor_position = len(buf.text)
        _render_screen_rows()
        kb_map[("right",)](ev)
        assert buf.text == "/resume"
        _render_screen_rows()
        dock_stage2_resume = _get_dock_text()
        assert "/resume 交互选参" in dock_stage2_resume
        assert "历史会话二：排查语音大卡不显示问题" in dock_stage2_resume
        assert "历史会话一：梳理车机Launcher卡片架构" in dock_stage2_resume

        # Type `Launcher` in the input box -> auto-inserts space (`/resume Launcher`) and live-filters to Session 1!
        buf.insert_text("Launcher")
        assert buf.text == "/resume Launcher"
        _render_screen_rows()
        dock_filtered_resume = _get_dock_text()
        assert "历史会话一：梳理车机Launcher卡片架构" in dock_filtered_resume
        assert "历史会话二：排查语音大卡不显示问题" not in dock_filtered_resume

        # Press `Enter` -> immediately restores Session 1 (`first_sid`) right from the input box!
        kb_map[("c-m",)](ev)
        assert buf.text == ""
        assert sess.session_id == first_sid
        assert len(sess.conversation_context) == 1
        assert sess.conversation_context[0]["user"] == "历史会话一：梳理车机Launcher卡片架构"

        # 5. Select `/ask` (requires argument) and press `Enter` with no argument yet -> fills `/ask ` in input box instead of erroring!
        buf.text = "/ask"
        buf.cursor_position = len(buf.text)
        _render_screen_rows()
        kb_map[("c-m",)](ev)
        assert buf.text == "/ask "
        _render_screen_rows()
        assert "/ask 交互选参" in _get_dock_text()
        buf.reset()

        # 6. Mid-input `/` command autocomplete (Claude Code v2.1+ feature):
        buf.text = "请先帮我分析 /tr"
        buf.cursor_position = len(buf.text)
        _render_screen_rows()
        assert "行内 / 命令补全" in _get_dock_text()
        assert " › /trace" in _get_dock_text()
        kb_map[("c-i",)](ev)
        assert buf.text == "请先帮我分析 /trace "
        buf.reset()

        # 7. Open `/connect` modal and verify live keyword filtering inside the input box + mouse click on `command_dock_window`
        await sess.handle_line("/connect")
        assert sess._modal_state is not None
        assert sess._modal_state["mode"] == "connect_provider"
        _render_screen_rows()
        # Verify `/connect` modal is rendered in `dock_win` (below input box), NOT in `main_win`!
        assert "连接大模型厂商 (/connect)" in _get_dock_text()
        assert "连接大模型厂商 (/connect)" not in _get_main_text()

        # Type `anthropic` in the input box while `/connect` modal is open -> live-jumps `selected_idx` to index 6 (Anthropic)!
        buf.text = "anthropic"
        buf.cursor_position = len(buf.text)
        assert sess._modal_state["selected_idx"] == 6
        sess.close_modal()


@pytest.mark.asyncio
async def test_instant_enter_ui_feedback_and_fast_conversational_path(tmp_path: Path, monkeypatch: Any) -> None:
    """
    Verify that when the user types `hello` and presses `Enter`:
    1. `_on_enter` -> `_launch_command_or_prompt` IMMEDIATELY (<1ms, before the background task even finishes its first await)
       clears `input_buffer`, sets `_inflight_prompt = "hello"`, sets `_agent_state_label = "正在执行"`, creates `_active_tracker`,
       and renders `› hello` in the upper conversation area plus the active working status on the bottom status bar.
    2. `FixRuntime.is_conversational_prompt("hello")` is True and `FixRuntime.execute_task("hello")` takes the fast
       conversational path: 0 git snapshots, 0 pre-flight Probe LLM calls, and 1 single fast-model completion call
       (`tools=None`, `use_reasoning=False`).
    """
    import asyncio
    from prompt_toolkit.application.current import set_app
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.output.defaults import DummyOutput
    import fix.cli as cli_mod
    from fix.cli import FixInteractiveSession
    from fix.context import ContextManager
    from fix.evolution import EvolutionEngine
    from fix.llm import LLMGateway, LLMResponse
    from fix.memory import DualMemoryStore
    from fix.runtime import FixRuntime
    from fix.tools import WorkspaceToolkit

    cols, rows = 100, 26
    monkeypatch.setattr(cli_mod, "_query_os_terminal_size", lambda: (cols, rows))

    (tmp_path / "app.py").write_text("print('hi')\n", encoding="utf-8")
    toolkit = WorkspaceToolkit(tmp_path)
    memory = DualMemoryStore(tmp_path)
    context_mgr = ContextManager()

    recorded_calls: list[dict[str, object]] = []

    class FastLiveSimGateway(LLMGateway):
        def __init__(self) -> None:
            super().__init__(api_key="test-key", base_url="https://token-plan-cn.xiaomimimo.com/v1")

        async def _single_completion_call(
            self,
            messages,
            tools=None,
            use_reasoning=False,
            temperature=0.1,
            max_tokens=32768,
        ) -> LLMResponse:
            recorded_calls.append(
                {
                    "messages": messages,
                    "tools": tools,
                    "use_reasoning": use_reasoning,
                    "max_tokens": max_tokens,
                }
            )
            await asyncio.sleep(0.05)
            return LLMResponse(
                content="Hello! I am Fix, your interactive AI engineering partner.",
                tool_calls=[],
                model=self.fast_model,
                finish_reason="stop",
                prompt_tokens=24,
                completion_tokens=12,
                total_tokens=36,
            )

    llm = FastLiveSimGateway()
    evolution = EvolutionEngine(tmp_path, memory=memory, llm=llm)
    runtime = FixRuntime(llm=llm, toolkit=toolkit, context_mgr=context_mgr, memory=memory, evolution=evolution)
    ui = FixInteractiveSession(repo=str(tmp_path))
    ui.llm = llm
    ui.runtime = runtime

    assert FixRuntime.is_conversational_prompt("hello") is True
    assert FixRuntime.is_conversational_prompt("你好！") is True
    assert FixRuntime.is_conversational_prompt("帮我分析一下 app.py 的代码结构") is False

    # Execute `hello` through runtime and verify single fast LLM call + zero snapshots
    sess = await runtime.execute_task("hello")
    assert sess.final_summary == "Hello! I am Fix, your interactive AI engineering partner."
    assert len(recorded_calls) == 1
    assert recorded_calls[0]["tools"] is None
    assert recorded_calls[0]["use_reasoning"] is False
    assert len(sess.snapshots) == 0


def test_version_v1_0_0_beta_and_clean_codebase() -> None:
    """Verify Fix version is v1.0.0-beta, `fix --version` works, and no machine-specific paths exist in src/."""
    from typer.testing import CliRunner
    from fix import __version__
    from fix.cli import app

    assert __version__ == "1.0.0-beta"
    runner = CliRunner()
    res = runner.invoke(app, ["--version"])
    assert res.exit_code == 0
    assert "Fix v1.0.0-beta" in res.stdout

    src_root = Path(__file__).resolve().parent.parent / "src" / "fix"
    for py_file in src_root.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        assert ("/" + "Users/") not in content, f"Found hardcoded macOS home path in {py_file}"
        assert ("/" + "home/") not in content, f"Found hardcoded Linux home path in {py_file}"


def test_latest_2026_official_vendor_models_and_retired_model_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Regression test for `t77u`:
    1. Verifies all 7 vendors in `PROVIDER_SPECS` match the latest 2026 official API docs:
       - DeepSeek: `deepseek-flash`, `deepseek-v4-pro` (no `deepseek-chat` or `deepseek-reasoner`)
       - GLM: `glm-5.3`, `glm-5.3-flash`, `glm-5.3-flashx`, `glm-5.2`, `glm-4.7` (no `glm-4.5` or `glm-4-flash`)
       - MiMo: `mimo-v2.6-pro`, `mimo-v2.6-flash`, `mimo-v2.6-pro-ultraspeed` (no `mimo-v2-pro` or `mimo-v2.5-pro`)
       - MiniMax: `MiniMax-M3.1-Flash`, `MiniMax-M3` (no `MiniMax-M1`, `MiniMax-Text-01`, `abab6.5s-chat`)
       - Qwen: `qwen3.8-max`, `qwen3.7-plus`, `qwen3.8-flash`, `qwen3-coder-plus` (no `qwen3-max`, `qwq-plus`)
       - OpenAI: `gpt-6-astra`, `gpt-6.1-sol`, `gpt-6.1-terra`, `gpt-6-luna` (no `gpt-4.1`, `gpt-4o`, `o3`, `o4-mini`)
       - Anthropic: `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-fable-5-1`, `claude-haiku-5-5` (no `claude-3-5-haiku-latest`)
    2. Verifies that an existing `~/.fix/llm_providers.json` containing legacy `deepseek-chat` / `deepseek-reasoner`
       is automatically migrated to `deepseek-flash` / `deepseek-v4-pro` on startup.
    3. Verifies that OpenAI `/v1/responses` fallback works seamlessly when a GPT-6 model is called with tools.
    """
    import asyncio
    import json
    import httpx
    from fix.llm import LLMGateway, PROVIDER_SPECS

    # 1. Verify official model catalogs and absence of retired models
    assert PROVIDER_SPECS["deepseek"]["default_fast_model"] == "deepseek-flash"
    assert PROVIDER_SPECS["deepseek"]["default_reasoning_model"] == "deepseek-v4-pro"
    assert PROVIDER_SPECS["deepseek"]["curated_models"] == ["deepseek-flash", "deepseek-v4-pro"]
    assert "deepseek-chat" not in PROVIDER_SPECS["deepseek"]["curated_models"]
    assert "deepseek-reasoner" not in PROVIDER_SPECS["deepseek"]["curated_models"]

    assert PROVIDER_SPECS["glm"]["default_fast_model"] == "glm-5.3-flash"
    assert PROVIDER_SPECS["glm"]["default_reasoning_model"] == "glm-5.3"
    assert "glm-5.3" in PROVIDER_SPECS["glm"]["curated_models"]
    assert "glm-4.5" not in PROVIDER_SPECS["glm"]["curated_models"]

    assert PROVIDER_SPECS["mimo"]["default_fast_model"] == "mimo-v2.6-flash"
    assert PROVIDER_SPECS["mimo"]["default_reasoning_model"] == "mimo-v2.6-pro"
    assert PROVIDER_SPECS["mimo"]["curated_models"] == [
        "mimo-v2.6-pro",
        "mimo-v2.6-flash",
        "mimo-v2.6-pro-ultraspeed",
    ]

    assert PROVIDER_SPECS["minimax"]["default_fast_model"] == "MiniMax-M3.1-Flash"
    assert PROVIDER_SPECS["minimax"]["default_reasoning_model"] == "MiniMax-M3"
    assert PROVIDER_SPECS["minimax"]["curated_models"] == ["MiniMax-M3.1-Flash", "MiniMax-M3"]

    assert PROVIDER_SPECS["qwen"]["default_fast_model"] == "qwen3.7-plus"
    assert PROVIDER_SPECS["qwen"]["default_reasoning_model"] == "qwen3.8-max"
    assert "qwen3.8-max" in PROVIDER_SPECS["qwen"]["curated_models"]
    assert "qwen3.7-plus" in PROVIDER_SPECS["qwen"]["curated_models"]

    assert PROVIDER_SPECS["openai"]["default_fast_model"] == "gpt-6.1-sol"
    assert PROVIDER_SPECS["openai"]["default_reasoning_model"] == "gpt-6-astra"
    assert "gpt-6-astra" in PROVIDER_SPECS["openai"]["curated_models"]
    assert "gpt-6.1-sol" in PROVIDER_SPECS["openai"]["curated_models"]

    assert PROVIDER_SPECS["anthropic"]["default_fast_model"] == "claude-sonnet-5-5"
    assert PROVIDER_SPECS["anthropic"]["default_reasoning_model"] == "claude-opus-5-5"
    assert "claude-opus-5-5" in PROVIDER_SPECS["anthropic"]["curated_models"]
    assert "claude-sonnet-5-5" in PROVIDER_SPECS["anthropic"]["curated_models"]

    # 2. Verify automatic migration of legacy `~/.fix/llm_providers.json`
    cfg_path = tmp_path / "llm_providers.json"
    cfg_path.write_text(
        json.dumps(
            {
                "active": {
                    "provider_id": "deepseek",
                    "api_key": "sk-ds-test",
                    "base_url": "https://api.deepseek.com/v1",
                    "fast_model": "deepseek-chat",
                    "reasoning_model": "deepseek-reasoner",
                },
                "connected_providers": {
                    "deepseek": {
                        "provider_id": "deepseek",
                        "api_key": "sk-ds-test",
                        "base_url": "https://api.deepseek.com/v1",
                        "models": ["deepseek-chat", "deepseek-reasoner", "deepseek-flash", "deepseek-v4-pro"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    gw = LLMGateway(providers_config_path=cfg_path)
    assert gw.provider_id == "deepseek"
    assert gw.fast_model == "deepseek-flash"
    assert gw.reasoning_model == "deepseek-v4-pro"
    assert gw.get_provider_models("deepseek") == ["deepseek-flash", "deepseek-v4-pro"]
    switchable_names = [m["model"] for m in gw.list_switchable_models()]
    assert "deepseek-chat" not in switchable_names
    assert "deepseek-reasoner" not in switchable_names
    assert switchable_names == ["deepseek-flash", "deepseek-v4-pro"]

    # 3. Verify OpenAI `/v1/responses` fallback when GPT-6 Astra is called with tools
    class _FakeResp:
        def __init__(self, status_code: int, data: dict[str, object]) -> None:
            self.status_code = status_code
            self._data = data

        def json(self) -> dict[str, object]:
            return self._data

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise httpx.HTTPStatusError("err", request=None, response=None)  # type: ignore[arg-type]

    class _MockClient:
        def __init__(self, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> "_MockClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

        async def post(self, url: str, headers: dict[str, str] | None = None, json: dict[str, object] | None = None) -> _FakeResp:
            if url.endswith("/chat/completions"):
                return _FakeResp(400, {"error": {"message": "Use Responses API for tools with gpt-6-astra"}})
            if url.endswith("/responses"):
                return _FakeResp(
                    200,
                    {
                        "id": "resp_01",
                        "model": "gpt-6-astra",
                        "status": "completed",
                        "output": [
                            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Checking files."}]},
                            {
                                "type": "function_call",
                                "call_id": "call_r1",
                                "name": "read_file",
                                "arguments": '{"rel_path": "main.py"}',
                            },
                        ],
                        "usage": {"input_tokens": 50, "output_tokens": 20, "output_tokens_details": {"reasoning_tokens": 8}},
                    },
                )
            return _FakeResp(404, {})

    monkeypatch.setattr(httpx, "AsyncClient", _MockClient)

    async def _check_openai_responses() -> None:
        gw.activate_provider_model("openai", "gpt-6-astra", api_key="sk-oa", base_url="https://api.openai.com/v1")
        r = await gw.complete_with_tools(
            messages=[{"role": "user", "content": "inspect main.py"}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "read_file", "description": "Read", "parameters": {"type": "object", "properties": {}}},
                }
            ],
        )
        assert r.reasoning_content == "Checking files."
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].name == "read_file"
        assert r.tool_calls[0].arguments == {"rel_path": "main.py"}

    asyncio.run(_check_openai_responses())
