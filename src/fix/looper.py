"""Android `Looper + IdleHandler` Serial Dual-State Evolution Scheduler (`src/fix/looper.py`)."""

from __future__ import annotations

import asyncio
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Any, Awaitable, Callable

from fix.evolution import EvolutionEngine
from fix.llm import LLMGateway
from fix.lsp import SemanticRadar
from fix.memory import DualMemoryStore
from fix.schema import (
    EvidenceFact,
    FactStatus,
    IdleBudget,
    IdleTick,
    IdleTickKind,
    IdleTickOutcome,
    MemoryDelta,
    TaskSession,
)
from fix.tools import WorkspaceToolkit


class FixLooper:
    """
    Implements the Android `Looper + IdleHandler` serial dual-state scheduling kernel:
    - P0 (`steer_queue`): Real-time mid-flight user steering & rollback interrupts.
    - P1 (Main Thread Task): Foreground user tasks (`execute_p1_task`).
    - P2 (`IdleHandler` Queue): Event-driven, budget-capped background evolution micro-ticks
      that write exclusively to an in-memory Copy-on-Write `MemoryDelta` buffer and
      yield within <10ms if preempted by a P1 task.
    """

    def __init__(
        self,
        repo_root: str | Path,
        toolkit: WorkspaceToolkit,
        memory: DualMemoryStore,
        evolution: EvolutionEngine,
        llm: LLMGateway | None = None,
        idle_budget: IdleBudget | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.toolkit = toolkit
        self.memory = memory
        self.evolution = evolution
        self.llm = llm
        self.radar = SemanticRadar(self.repo_root)
        self.idle_budget = idle_budget or IdleBudget()

        try:
            self.p0_steer_queue: asyncio.Queue[str] = asyncio.Queue()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
            self.p0_steer_queue = asyncio.Queue()
        self.p2_idle_queue: deque[IdleTick] = deque()
        self._p1_interrupt_flag: bool = False
        self._p1_waiters: list[asyncio.Future[bool]] = []
        self.p1_active: bool = False

        self.completed_ticks: list[IdleTickOutcome] = []
        self.unread_notifications: list[str] = []
        self._analyzed_commits: set[str] = set()
        self._studied_hotspots: set[str] = set()
        self._synthesized_essence_files: set[str] = set()
        self.active_idle_tick: IdleTick | None = None

    # ------------------------------------------------------------------
    # 1. Event-Driven P2 Tick Enqueueing (Zero Blind Token Burning)
    # ------------------------------------------------------------------

    def enqueue_idle_tick(self, tick: IdleTick, front: bool = False) -> None:
        """Enqueue a bounded P2 IdleTick if an identical (kind, target) tick isn't already queued."""
        for existing in self.p2_idle_queue:
            if existing.kind == tick.kind and existing.target == tick.target:
                return
        if front:
            self.p2_idle_queue.appendleft(tick)
        else:
            self.p2_idle_queue.append(tick)

    def schedule_startup_idle_events(self) -> int:
        """Schedule deterministic refresh + unanalyzed commit + top hotspot study on startup."""
        before = len(self.p2_idle_queue)
        self.enqueue_idle_tick(
            IdleTick(
                kind=IdleTickKind.REFRESH_STALE_LANDMARKS,
                target=str(self.repo_root),
                requires_llm=False,
            )
        )
        head_sha = self.toolkit.get_head_sha()
        if head_sha and head_sha not in self._analyzed_commits:
            self.enqueue_idle_tick(
                IdleTick(
                    kind=IdleTickKind.ANALYZE_RECENT_COMMIT,
                    target=head_sha,
                    requires_llm=False,
                )
            )
        return len(self.p2_idle_queue) - before

    def schedule_post_task_idle_events(self, session: TaskSession) -> int:
        """Schedule targeted P2 evolution ticks after a P1 task completes (modified or inspected files)."""
        before = len(self.p2_idle_queue)
        candidate_files = sorted(session.modified_files) or self.evolution._extract_inspected_paths(session)
        for mod_file in candidate_files[:2]:
            if mod_file not in self._studied_hotspots:
                self.enqueue_idle_tick(
                    IdleTick(
                        kind=IdleTickKind.HOTSPOT_DEEP_STUDY,
                        target=mod_file,
                        requires_llm=bool(self.llm and self.llm.is_configured),
                    )
                )
            if mod_file not in self._synthesized_essence_files:
                self.enqueue_idle_tick(
                    IdleTick(
                        kind=IdleTickKind.ARCHITECTURAL_ESSENCE_SYNTHESIS,
                        target=mod_file,
                        requires_llm=False,
                    )
                )
        head_sha = self.toolkit.get_head_sha()
        if head_sha and head_sha not in self._analyzed_commits:
            self.enqueue_idle_tick(
                IdleTick(
                    kind=IdleTickKind.ANALYZE_RECENT_COMMIT,
                    target=head_sha,
                    requires_llm=False,
                )
            )
        return len(self.p2_idle_queue) - before

    # ------------------------------------------------------------------
    # 2. P1 Main Thread Execution (Preempts P2 Immediately)
    # ------------------------------------------------------------------

    def signal_p1_arrival(self) -> None:
        """Signal any in-flight P2 IdleTick to abort within <10ms and discard uncommitted MemoryDelta."""
        self.p1_active = True
        self._p1_interrupt_flag = True
        for fut in list(self._p1_waiters):
            if not fut.done():
                fut.set_result(True)
        self._p1_waiters.clear()

    def clear_p1_interrupt(self) -> None:
        self.p1_active = False
        self._p1_interrupt_flag = False
        self._p1_waiters.clear()

    async def _wait_for_p1_interrupt(self) -> bool:
        if self._p1_interrupt_flag or self.p1_active:
            return True
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[bool] = loop.create_future()
        self._p1_waiters.append(fut)
        try:
            return await fut
        finally:
            if fut in self._p1_waiters:
                self._p1_waiters.remove(fut)

    async def run_p1_task(
        self,
        coro_factory: Callable[[], Awaitable[TaskSession]],
    ) -> TaskSession:
        """Preempt P2, run P1 foreground task to completion, then schedule post-task P2 events."""
        self.signal_p1_arrival()
        try:
            session = await coro_factory()
            self.schedule_post_task_idle_events(session)
            return session
        finally:
            self.clear_p1_interrupt()

    # ------------------------------------------------------------------
    # 3. P2 Serial IdleHandler Execution with Copy-on-Write & <10ms Preemption
    # ------------------------------------------------------------------

    async def run_single_idle_tick(
        self,
        simulated_work_delay_sec: float = 0.0,
    ) -> IdleTickOutcome | None:
        """
        Execute a single P2 IdleTick if P1 is idle and within IdleBudget.
        Uses `asyncio.wait(..., return_when=FIRST_COMPLETED)` against `_wait_for_p1_interrupt`
        so P1 arrival preempts the tick immediately without writing dirty data to SQLite.
        """
        if self.p1_active or self._p1_interrupt_flag or not self.p2_idle_queue:
            return None

        tick = self.p2_idle_queue.popleft()

        # Enforce hard IdleBudget caps (Zero Blind Token Burning)
        if tick.requires_llm and self.idle_budget.llm_ticks_used >= self.idle_budget.max_llm_ticks_per_idle:
            return IdleTickOutcome(
                tick_id=tick.tick_id,
                kind=tick.kind,
                target=tick.target,
                preempted=False,
                committed=False,
                summary="Skipped: IdleBudget max_llm_ticks_per_idle reached.",
            )
        if not tick.requires_llm and self.idle_budget.deterministic_ticks_used >= self.idle_budget.max_deterministic_ticks_per_idle:
            return IdleTickOutcome(
                tick_id=tick.tick_id,
                kind=tick.kind,
                target=tick.target,
                preempted=False,
                committed=False,
                summary="Skipped: IdleBudget max_deterministic_ticks_per_idle reached.",
            )

        t0 = time.perf_counter()
        self.active_idle_tick = tick
        work_task = asyncio.create_task(
            self._compute_tick_delta(tick, simulated_work_delay_sec=simulated_work_delay_sec)
        )
        interrupt_task = asyncio.create_task(self._wait_for_p1_interrupt())

        done, _pending = await asyncio.wait(
            {work_task, interrupt_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Case A: Preempted by P1! Cancel work, discard uncommitted MemoryDelta, re-queue tick at front.
        if interrupt_task in done or self._p1_interrupt_flag:
            if not work_task.done():
                work_task.cancel()
                try:
                    await work_task
                except (asyncio.CancelledError, Exception):
                    pass
            self.enqueue_idle_tick(tick, front=True)
            elapsed = round((time.perf_counter() - t0) * 1000.0, 2)
            outcome = IdleTickOutcome(
                tick_id=tick.tick_id,
                kind=tick.kind,
                target=tick.target,
                preempted=True,
                committed=False,
                summary=f"Preempted by P1 in {elapsed}ms; uncommitted MemoryDelta discarded & tick re-queued.",
                elapsed_ms=elapsed,
            )
            self.active_idle_tick = None
            self.completed_ticks.append(outcome)
            return outcome

        # Case B: Tick completed cleanly without P1 interruption -> Atomic Copy-on-Write commit!
        interrupt_task.cancel()
        try:
            await interrupt_task
        except asyncio.CancelledError:
            pass

        delta, summary = work_task.result()

        # Double-check P1 did not arrive in the final microsecond before atomic commit
        if self._p1_interrupt_flag or self.p1_active:
            self.enqueue_idle_tick(tick, front=True)
            elapsed = round((time.perf_counter() - t0) * 1000.0, 2)
            outcome = IdleTickOutcome(
                tick_id=tick.tick_id,
                kind=tick.kind,
                target=tick.target,
                preempted=True,
                committed=False,
                summary="Preempted right before commit; MemoryDelta discarded.",
                elapsed_ms=elapsed,
            )
            self.active_idle_tick = None
            self.completed_ticks.append(outcome)
            return outcome

        # Perform single atomic SQLite commit (<1ms)
        self.memory.commit_delta(delta)
        if tick.requires_llm:
            self.idle_budget.llm_ticks_used += 1
        else:
            self.idle_budget.deterministic_ticks_used += 1

        elapsed = round((time.perf_counter() - t0) * 1000.0, 2)
        outcome = IdleTickOutcome(
            tick_id=tick.tick_id,
            kind=tick.kind,
            target=tick.target,
            preempted=False,
            committed=True,
            summary=summary,
            elapsed_ms=elapsed,
        )
        self.active_idle_tick = None
        self.completed_ticks.append(outcome)
        self.unread_notifications.append(f"[{tick.kind.value}] {summary} ({elapsed}ms)")
        return outcome

    async def drain_idle_queue(
        self,
        max_ticks: int = 5,
        reset_window_budget: bool = True,
    ) -> list[IdleTickOutcome]:
        """Run queued P2 IdleTicks serially until queue is empty, budget is reached, or P1 preempts."""
        if reset_window_budget:
            self.idle_budget.llm_ticks_used = 0
            self.idle_budget.deterministic_ticks_used = 0

        outcomes: list[IdleTickOutcome] = []
        for _ in range(max_ticks):
            if not self.p2_idle_queue or self.p1_active or self._p1_interrupt_flag:
                break
            out = await self.run_single_idle_tick()
            if out is None:
                break
            outcomes.append(out)
            if out.preempted:
                break
        return outcomes

    async def _compute_tick_delta(
        self,
        tick: IdleTick,
        simulated_work_delay_sec: float = 0.0,
    ) -> tuple[MemoryDelta, str]:
        """
        Execute tick analysis purely in memory and return `(MemoryDelta, summary)`.
        NEVER writes directly to `memory.repo_db_path` or `memory.persona_db_path` mid-execution!
        """
        if simulated_work_delay_sec > 0:
            await asyncio.sleep(simulated_work_delay_sec)

        delta = MemoryDelta()

        if tick.kind == IdleTickKind.REFRESH_STALE_LANDMARKS:
            invalidated = self.memory.invalidate_stale_facts()
            edges = self.evolution._compute_cochange_edges_in_memory(max_commits=80)
            delta.co_changes_to_upsert.extend(edges)
            delta.stale_invalidated_count = invalidated
            # If top hotspot hasn't been studied yet, enqueue a HOTSPOT_DEEP_STUDY tick
            if edges:
                top_file = edges[0].file_a
                if top_file not in self._studied_hotspots:
                    self.enqueue_idle_tick(
                        IdleTick(
                            kind=IdleTickKind.HOTSPOT_DEEP_STUDY,
                            target=top_file,
                            requires_llm=False,
                        )
                    )
            return delta, f"Refreshed {len(edges)} co-change pairs & invalidated {invalidated} stale landmarks"

        if tick.kind == IdleTickKind.ANALYZE_RECENT_COMMIT:
            commit_ref = tick.target or "HEAD"
            msg_proc = subprocess.run(
                ["git", "log", "-1", "--pretty=%s", commit_ref],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
            )
            diff_proc = subprocess.run(
                ["git", "show", "--format=", commit_ref],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
            )
            commit_msg = msg_proc.stdout.strip() if msg_proc.returncode == 0 else ""
            human_diff = diff_proc.stdout if diff_proc.returncode == 0 else ""
            if tick.requires_llm:
                analysis = await self.evolution.analyze_human_diff(
                    human_diff=human_diff,
                    commit_message=commit_msg,
                )
            else:
                analysis = self.evolution._deterministic_semantic_diff_filter(
                    human_diff=human_diff,
                    commit_message=commit_msg,
                )
            self._analyzed_commits.add(commit_ref)
            if analysis.is_semantically_equivalent_refactor and analysis.extracted_rules:
                delta.rules_to_upsert.extend(analysis.extracted_rules)
                return delta, f"Extracted {len(analysis.extracted_rules)} style rule(s) from commit {commit_ref[:8]}"
            return delta, f"Inspected commit {commit_ref[:8]} (classified as {analysis.change_classification})"

        if tick.kind == IdleTickKind.HOTSPOT_DEEP_STUDY:
            rel_path = tick.target
            self._studied_hotspots.add(rel_path)
            sym_res = self.toolkit.inspect_symbols(rel_path)
            diags = self.radar.check_diagnostics(rel_path)
            head_sha = self.toolkit.get_head_sha()

            summary_lines = [sym_res.output.splitlines()[0] if sym_res.output else rel_path]
            if len(sym_res.output.splitlines()) > 1:
                summary_lines.append("; ".join(s.strip() for s in sym_res.output.splitlines()[1:5]))
            diag_note = f"{len(diags)} diagnostics" if diags else "0 static errors"

            fact = EvidenceFact(
                summary=f"[IdleHandler Hotspot Map `{rel_path}` ({diag_note})]: {' | '.join(summary_lines)[:260]}",
                category="architecture",
                evidence_paths=[rel_path],
                commit_sha=head_sha,
                confidence=0.9,
                status=FactStatus.VERIFIED,
            )
            delta.facts_to_upsert.append(fact)
            mm, proposals = self.evolution.synthesize_architectural_essence(
                rel_path=rel_path,
                head_sha=head_sha,
            )
            delta.mental_models_to_upsert.append(mm)
            delta.optimization_proposals_to_upsert.extend(proposals)
            return delta, f"Indexed hotspot symbol map, 3-layer MentalModel & diagnostics for `{rel_path}`"

        if tick.kind == IdleTickKind.ARCHITECTURAL_ESSENCE_SYNTHESIS:
            rel_path = tick.target
            self._synthesized_essence_files.add(rel_path)
            head_sha = self.toolkit.get_head_sha()
            mm, proposals = self.evolution.synthesize_architectural_essence(
                rel_path=rel_path,
                head_sha=head_sha,
            )
            delta.mental_models_to_upsert.append(mm)
            delta.optimization_proposals_to_upsert.extend(proposals)
            return (
                delta,
                f"Synthesized 3-layer MentalModel & {len(proposals)} optimization proposal(s) for `{rel_path}`",
            )

        return delta, "No-op tick"

    def pop_notifications(self) -> list[str]:
        notes = list(self.unread_notifications)
        self.unread_notifications.clear()
        return notes

    def get_status_dict(self) -> dict[str, Any]:
        return {
            "p1_active": self.p1_active,
            "p0_steer_queued": self.p0_steer_queue.qsize(),
            "p2_idle_queued": len(self.p2_idle_queue),
            "p2_queue_items": [f"{t.kind.value}({t.target})" for t in self.p2_idle_queue],
            "completed_ticks_count": len(self.completed_ticks),
            "llm_budget": f"{self.idle_budget.llm_ticks_used}/{self.idle_budget.max_llm_ticks_per_idle}",
            "det_budget": f"{self.idle_budget.deterministic_ticks_used}/{self.idle_budget.max_deterministic_ticks_per_idle}",
        }
