"""End-to-End tests for FixRuntime cognitive state machine, physical tool gates, Reviewer & live steering."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Any

import pytest

from fix.context import ContextManager
from fix.evolution import EvolutionEngine
from fix.llm import LLMResponse, ScriptedMockLLM
from fix.memory import DualMemoryStore
from fix.runtime import FixRuntime
from fix.schema import (
    ProbeDecision,
    ReviewVerdict,
    TaskIntent,
    TaskPhase,
    ToolCallRequest,
)
from fix.tools import WorkspaceToolkit


def _create_buggy_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "order_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "dev@fix.local"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Fix Tester"], cwd=repo, check=True)

    # Production file with a bug: deducts amount twice!
    (repo / "order.py").write_text(
        "def deduct_balance(balance: int, price: int) -> int:\n"
        "    if price < 0:\n"
        "        raise ValueError('negative price')\n"
        "    return balance - price - price\n",
        encoding="utf-8",
    )
    (repo / "test_order.py").write_text(
        "from order import deduct_balance\n\n"
        "def test_deduct_balance():\n"
        "    assert deduct_balance(100, 30) == 70\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "Initial buggy order service"], cwd=repo, check=True, capture_output=True)
    return repo


@pytest.mark.asyncio
async def test_e2e_bug_fix_loop_with_reproduce_gate_and_reviewer_self_heal(tmp_path: Path) -> None:
    """
    Verifies the complete lifecycle:
    1. PROBE selects BUG_FIX track.
    2. In REPRODUCE phase, attempting to edit `order.py` directly is physically blocked (`ReproduceGateViolation`).
    3. Running `pytest -q` reproduces the failure (`reproduction_confirmed=True`) -> unlocks `IMPLEMENT`.
    4. First patch fixes `order.py`, `pytest -q` passes, but Independent Reviewer rejects once asking for a docstring/guard.
    5. Agent self-heals in `IMPLEMENT`, Reviewer approves on round 2 -> `DONE` -> `EvolutionEngine` persists landmark & skill.
    """
    repo = _create_buggy_repo(tmp_path)
    toolkit = WorkspaceToolkit(repo)
    memory = DualMemoryStore(repo, persona_db_path=tmp_path / "persona.db")
    evolution = EvolutionEngine(repo, memory=memory)

    review_verdicts = [
        ReviewVerdict(
            approved=False,
            confidence="MEDIUM",
            blocking_issues=["Please ensure insufficient balance raises ValueError when balance < price."],
        ),
        ReviewVerdict(
            approved=True,
            confidence="HIGH",
            blocking_issues=[],
        ),
    ]

    mock_llm = ScriptedMockLLM(
        tool_Queue=[
            # Step 1 (in REPRODUCE): LLM tries to prematurely patch order.py -> blocked by physical gate!
            LLMResponse(
                content="Trying to edit order.py before reproducing.",
                tool_calls=[
                    ToolCallRequest(
                        name="apply_patch",
                        arguments={
                            "rel_path": "order.py",
                            "search_block": "    return balance - price - price",
                            "replace_block": "    return balance - price",
                        },
                    )
                ],
            ),
            # Step 2 (in REPRODUCE): LLM runs pytest to reproduce the bug -> fails -> transitions to IMPLEMENT
            LLMResponse(
                content="Running pytest to reproduce the double deduction failure.",
                tool_calls=[
                    ToolCallRequest(
                        name="run_command",
                        arguments={"cmd": "pytest -q"},
                    )
                ],
            ),
            # Step 3 (in IMPLEMENT): LLM fixes the double deduction in order.py -> VERIFY passes -> Reviewer rejects (round 1)
            LLMResponse(
                content="Fixing double deduction in order.py.",
                tool_calls=[
                    ToolCallRequest(
                        name="apply_patch",
                        arguments={
                            "rel_path": "order.py",
                            "search_block": "    return balance - price - price",
                            "replace_block": "    return balance - price",
                        },
                    )
                ],
            ),
            # Step 4 (in IMPLEMENT after Reviewer feedback): LLM adds balance < price check -> VERIFY passes -> Reviewer approves!
            LLMResponse(
                content="Addressing Independent Reviewer feedback by guarding balance < price.",
                tool_calls=[
                    ToolCallRequest(
                        name="apply_patch",
                        arguments={
                            "rel_path": "order.py",
                            "search_block": "    if price < 0:\n        raise ValueError('negative price')\n    return balance - price",
                            "replace_block": (
                                "    if price < 0 or balance < price:\n"
                                "        raise ValueError('invalid or insufficient balance')\n"
                                "    return balance - price"
                            ),
                        },
                    )
                ],
            ),
        ],
        structured_handlers={
            "ProbeDecision": ProbeDecision(
                intent=TaskIntent.BUG_FIX,
                summary="Investigate double deduction in deduct_balance",
                root_cause_hypothesis="deduct_balance subtracts price twice",
                target_files=["order.py"],
                reproduction_cmd="pytest -q",
            ),
            "ReviewVerdict": review_verdicts,
        },
    )

    runtime = FixRuntime(
        llm=mock_llm,
        toolkit=toolkit,
        context_mgr=ContextManager(),
        memory=memory,
        evolution=evolution,
    )

    session = await runtime.execute_task("Fix bug where deduct_balance deducts price twice")

    assert session.phase == TaskPhase.DONE
    assert session.intent == TaskIntent.BUG_FIX
    assert session.reproduction_confirmed is True
    assert session.degraded_without_repro is False
    assert session.confidence == "HIGH"
    assert session.last_review is not None and session.last_review.approved is True
    assert "order.py" in session.modified_files
    assert "balance < price" in (repo / "order.py").read_text(encoding="utf-8")

    # Verify Post-Task Digest saved a verified landmark & bootstrapped skill
    landmarks = memory.search_landmarks("deduct_balance")
    assert len(landmarks) == 1
    assert landmarks[0].status.value == "VERIFIED"
    assert len(memory.list_skills()) == 1


@pytest.mark.asyncio
async def test_explain_track_physically_blocks_write_tools(tmp_path: Path) -> None:
    repo = _create_buggy_repo(tmp_path)
    toolkit = WorkspaceToolkit(repo)

    mock_llm = ScriptedMockLLM(
        tool_Queue=[
            LLMResponse(
                content="Attempting an unauthorized write in EXPLAIN mode.",
                tool_calls=[
                    ToolCallRequest(
                        name="write_new_file",
                        arguments={"rel_path": "hacked.py", "content": "x = 1\n"},
                    )
                ],
            ),
            LLMResponse(
                content="Diagnosis: `deduct_balance` at `[order.py:4]` subtracts `price` twice.",
                tool_calls=[],
            ),
        ]
    )

    runtime = FixRuntime(llm=mock_llm, toolkit=toolkit)
    session = await runtime.execute_task(
        "Explain why balance is deducted twice",
        forced_intent=TaskIntent.EXPLAIN,
    )

    assert session.phase == TaskPhase.DONE
    assert session.intent == TaskIntent.EXPLAIN
    assert not (repo / "hacked.py").exists()
    assert "[order.py:4]" in session.final_summary


@pytest.mark.asyncio
async def test_elastic_tdd_escape_valve_when_reproduction_unavailable(tmp_path: Path) -> None:
    repo = _create_buggy_repo(tmp_path)
    toolkit = WorkspaceToolkit(repo)

    mock_llm = ScriptedMockLLM(
        tool_Queue=[
            # Attempt 1 in REPRODUCE: command passes (cannot reproduce failure)
            LLMResponse(
                content="Try repro 1",
                tool_calls=[ToolCallRequest(name="run_command", arguments={"cmd": "true"})],
            ),
            # Attempt 2 in REPRODUCE: command still passes -> triggers Elastic Escape Valve -> enters IMPLEMENT!
            LLMResponse(
                content="Try repro 2",
                tool_calls=[ToolCallRequest(name="run_command", arguments={"cmd": "true"})],
            ),
            # Step 3 (now in degraded IMPLEMENT): patch order.py
            LLMResponse(
                content="Applying fix in degraded mode.",
                tool_calls=[
                    ToolCallRequest(
                        name="apply_patch",
                        arguments={
                            "rel_path": "order.py",
                            "search_block": "    return balance - price - price",
                            "replace_block": "    return balance - price",
                        },
                    )
                ],
            ),
        ]
    )

    runtime = FixRuntime(llm=mock_llm, toolkit=toolkit)
    session = await runtime.execute_task(
        "Fix intermittent production bug",
        forced_intent=TaskIntent.BUG_FIX,
    )

    assert session.phase == TaskPhase.DONE
    assert session.degraded_without_repro is True
    assert session.confidence == "LOW"
    assert "order.py" in session.modified_files


@pytest.mark.asyncio
async def test_mid_flight_steering_queue_and_snapshot_rollback(tmp_path: Path) -> None:
    repo = _create_buggy_repo(tmp_path)
    toolkit = WorkspaceToolkit(repo)
    steer_queue: asyncio.Queue[str] = asyncio.Queue()

    def step_callback(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> LLMResponse:
        # Check if steering note has arrived
        joined = "\n".join(str(m.get("content", "")) for m in messages)
        if "[LIVE STEERING + ROLLBACK]" not in joined:
            # First step: create a temporary file, then user injects a rollback steering command!
            steer_queue.put_nowait("停一下，回滚刚才的改动，直接把 order.py 里的双倍扣款改好")
            return LLMResponse(
                content="Creating wrong experimental file.",
                tool_calls=[
                    ToolCallRequest(
                        name="write_new_file",
                        arguments={"rel_path": "wrong_approach.py", "content": "BAD = True\n"},
                    )
                ],
            )
        # Second step (after rollback): apply the clean fix to order.py
        return LLMResponse(
            content="Rolled back wrong_approach.py and fixed order.py cleanly.",
            tool_calls=[
                ToolCallRequest(
                    name="apply_patch",
                    arguments={
                        "rel_path": "order.py",
                        "search_block": "    return balance - price - price",
                        "replace_block": "    return balance - price",
                    },
                )
            ],
        )

    mock_llm = ScriptedMockLLM(step_callback=step_callback)
    runtime = FixRuntime(llm=mock_llm, toolkit=toolkit)

    session = await runtime.execute_task(
        "Update order deduction",
        forced_intent=TaskIntent.DIRECT_EDIT,
        steer_queue=steer_queue,
    )

    assert session.phase == TaskPhase.DONE
    # Verify wrong_approach.py was rolled back and deleted!
    assert not (repo / "wrong_approach.py").exists()
    assert "order.py" in session.modified_files


@pytest.mark.asyncio
async def test_guaranteed_final_synthesis_when_max_steps_reached_and_tool_dedup(tmp_path: Path) -> None:
    """
    Verifies that:
    1. If the model keeps calling read/search tools until `max_steps` is reached with empty content,
       `FixRuntime` automatically executes `_force_final_synthesis` (`tools=None`) so `final_summary`
       contains the full analysis report instead of `(Task completed)`.
    2. Duplicate read-only tool calls in the same session are intercepted by the deduplication cache.
    3. Elapsed time and token usage (`prompt_tokens`, `completion_tokens`, `total_tokens`, `llm_calls`) are tracked.
    """
    repo = _create_buggy_repo(tmp_path)
    toolkit = WorkspaceToolkit(repo)

    call_idx = 0

    def step_cb(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> LLMResponse:
        nonlocal call_idx
        call_idx += 1
        # When `_force_final_synthesis` calls `complete_with_tools(..., tools=None)`, return the full report!
        if tools is None:
            return LLMResponse(
                content="### Deep Analysis Report\n- Working Mechanism: `deduct_balance` at `[order.py:1-4]`.\n- Root Cause: subtracts `price` twice.",
                tool_calls=[],
                prompt_tokens=1200,
                completion_tokens=350,
                total_tokens=1550,
            )
        # During regular steps, keep calling the exact same search_code tool with empty content
        return LLMResponse(
            content="",
            tool_calls=[
                ToolCallRequest(
                    name="search_code",
                    arguments={"pattern": "deduct_balance"},
                )
            ],
            prompt_tokens=400,
            completion_tokens=40,
            total_tokens=440,
        )

    mock_llm = ScriptedMockLLM(step_callback=step_cb)
    runtime = FixRuntime(llm=mock_llm, toolkit=toolkit)

    session = await runtime.execute_task(
        "深入分析 deduct_balance 的工作机制与逻辑地图",
        forced_intent=TaskIntent.EXPLAIN,
    )

    assert session.phase == TaskPhase.DONE
    assert "### Deep Analysis Report" in session.final_summary
    assert "[order.py:1-4]" in session.final_summary
    assert session.tool_calls_count == session.budget.max_steps
    assert session.tool_counts_by_name.get("Search") == session.budget.max_steps
    assert session.total_tokens > 1550
    assert session.llm_calls == session.budget.max_steps + 1
    assert session.elapsed_sec > 0.0

    # Verify that the duplicate search call on step 2 received `[Duplicate Tool Call Skipped]`
    synthesis_call_messages = mock_llm.recorded_messages[-1]
    joined_history = "\n".join(str(m.get("content", "")) for m in synthesis_call_messages)
    assert "[Duplicate Tool Call Skipped]" in joined_history
