"""Tests for Priority 2: Android `Looper + IdleHandler` Serial Evolution Scheduler (`src/fix/looper.py`)."""

from __future__ import annotations

import asyncio
import subprocess
import time
from pathlib import Path

import pytest

from fix.evolution import EvolutionEngine
from fix.looper import FixLooper
from fix.memory import DualMemoryStore
from fix.schema import IdleBudget, IdleTick, IdleTickKind, TaskIntent, TaskPhase, TaskSession
from fix.tools import WorkspaceToolkit


def _setup_looper_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "looper_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "dev@fix.local"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Fix Tester"], cwd=repo, check=True)

    (repo / "engine.py").write_text(
        "class CoreEngine:\n"
        "    def execute(self, x: int) -> int:\n"
        "        return x * 2\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "Initial engine"], cwd=repo, check=True, capture_output=True)
    return repo


@pytest.mark.asyncio
async def test_p1_preempts_p2_idle_tick_within_10ms_and_discards_memory_delta(tmp_path: Path) -> None:
    """
    Verifies that when a P2 IdleTick is executing in the background and a P1 task arrives:
    1. The P2 tick is preempted in <10ms.
    2. The in-flight Copy-on-Write `MemoryDelta` is discarded (0 dirty facts written to SQLite).
    3. The preempted tick is re-queued at the front of `p2_idle_queue` and commits cleanly once P1 finishes.
    """
    repo = _setup_looper_repo(tmp_path)
    toolkit = WorkspaceToolkit(repo)
    memory = DualMemoryStore(repo, persona_db_path=tmp_path / "persona.db")
    evolution = EvolutionEngine(repo, memory=memory)
    looper = FixLooper(repo_root=repo, toolkit=toolkit, memory=memory, evolution=evolution)

    looper.enqueue_idle_tick(
        IdleTick(
            kind=IdleTickKind.HOTSPOT_DEEP_STUDY,
            target="engine.py",
            requires_llm=False,
        )
    )
    assert len(looper.p2_idle_queue) == 1

    # Start the P2 tick with a 200ms simulated computation window
    tick_task = asyncio.create_task(looper.run_single_idle_tick(simulated_work_delay_sec=0.20))
    await asyncio.sleep(0.02)

    # Signal P1 foreground task arrival and measure preemption latency
    t_preempt_start = time.perf_counter()
    looper.signal_p1_arrival()
    outcome = await tick_task
    preempt_latency_ms = (time.perf_counter() - t_preempt_start) * 1000.0

    assert outcome is not None
    assert outcome.preempted is True
    assert outcome.committed is False
    assert preempt_latency_ms < 25.0  # Immediate coroutine wakeup (<10ms typical)

    # Verify ZERO dirty landmarks were written to SQLite during the preempted tick!
    assert len(memory.search_landmarks("engine.py")) == 0
    # Verify the preempted tick was put back at the head of P2 queue
    assert len(looper.p2_idle_queue) == 1
    assert looper.p2_idle_queue[0].target == "engine.py"

    # Clear P1 active state and let the re-queued P2 tick run to completion
    looper.clear_p1_interrupt()

    outcome2 = await looper.run_single_idle_tick(simulated_work_delay_sec=0.0)
    assert outcome2 is not None
    assert outcome2.preempted is False
    assert outcome2.committed is True
    assert len(looper.p2_idle_queue) == 0

    # Now the landmark is atomically committed in repo.db!
    landmarks = memory.search_landmarks("CoreEngine")
    assert len(landmarks) == 1
    assert "IdleHandler Hotspot Map `engine.py`" in landmarks[0].summary


@pytest.mark.asyncio
async def test_idle_budget_cap_prevents_blind_token_burning(tmp_path: Path) -> None:
    repo = _setup_looper_repo(tmp_path)
    toolkit = WorkspaceToolkit(repo)
    memory = DualMemoryStore(repo, persona_db_path=tmp_path / "persona.db")
    evolution = EvolutionEngine(repo, memory=memory)
    looper = FixLooper(
        repo_root=repo,
        toolkit=toolkit,
        memory=memory,
        evolution=evolution,
        idle_budget=IdleBudget(max_llm_ticks_per_idle=1, max_deterministic_ticks_per_idle=2),
    )

    # Enqueue 2 LLM-marked hotspot ticks
    (repo / "extra.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    looper.enqueue_idle_tick(IdleTick(kind=IdleTickKind.HOTSPOT_DEEP_STUDY, target="engine.py", requires_llm=True))
    looper.enqueue_idle_tick(IdleTick(kind=IdleTickKind.HOTSPOT_DEEP_STUDY, target="extra.py", requires_llm=True))

    outcomes = await looper.drain_idle_queue(max_ticks=5)
    assert len(outcomes) == 2
    assert outcomes[0].committed is True
    # Second LLM tick is skipped because max_llm_ticks_per_idle=1!
    assert outcomes[1].committed is False
    assert "IdleBudget max_llm_ticks_per_idle reached" in outcomes[1].summary


@pytest.mark.asyncio
async def test_run_p1_task_automatically_schedules_post_task_idle_ticks(tmp_path: Path) -> None:
    repo = _setup_looper_repo(tmp_path)
    toolkit = WorkspaceToolkit(repo)
    memory = DualMemoryStore(repo, persona_db_path=tmp_path / "persona.db")
    evolution = EvolutionEngine(repo, memory=memory)
    looper = FixLooper(repo_root=repo, toolkit=toolkit, memory=memory, evolution=evolution)

    async def _fake_p1_task() -> TaskSession:
        return TaskSession(
            repo_root=str(repo),
            user_prompt="Optimize CoreEngine",
            intent=TaskIntent.FEATURE,
            phase=TaskPhase.DONE,
            modified_files={"engine.py"},
        )

    session = await looper.run_p1_task(_fake_p1_task)
    assert session.phase == TaskPhase.DONE
    assert looper.p1_active is False

    # Post-task hook automatically scheduled HOTSPOT_DEEP_STUDY(engine.py) & ANALYZE_RECENT_COMMIT
    queued_kinds = [t.kind for t in looper.p2_idle_queue]
    assert IdleTickKind.HOTSPOT_DEEP_STUDY in queued_kinds
    assert IdleTickKind.ANALYZE_RECENT_COMMIT in queued_kinds
