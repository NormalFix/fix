"""L1-L4 Progressive Autonomy Router & Git Worktree Empirical Sandbox Verifier (`src/fix/autonomy.py`)."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Callable, Awaitable

from fix.lsp import SemanticRadar
from fix.memory import DualMemoryStore
from fix.profiler import RuntimeProfiler
from fix.schema import (
    AutonomyAssessment,
    AutonomyLevel,
    RuleStatus,
    SandboxComparisonReport,
    TaskIntent,
)
from fix.tools import WorkspaceToolkit


class AutonomyRouter:
    """
    Computes the explicit L1-L4 progressive autonomy score for a task:
      Score = 0.35 * LandmarkCoverage + 0.30 * HistorySuccessRate + 0.20 * PersonaAlignment - 0.35 * BlastRadiusRisk
    Automatically upgrades Fix from L1_APPRENTICE -> L2_COLLABORATOR -> L3_SEMI_AUTONOMOUS -> L4_DIGITAL_TWIN
    as verified project landmarks, personal rules, and successful task runs accumulate.
    """

    def __init__(self, memory: DualMemoryStore) -> None:
        self.memory = memory

    def assess_autonomy(
        self,
        user_prompt: str = "",
        intent: TaskIntent | None = None,
        estimated_files_touched: int = 1,
    ) -> AutonomyAssessment:
        """Compute quantitative L1-L4 autonomy level and component scores."""
        # 1. LandmarkCoverage (0.0 .. 1.0): based on verified non-stale landmarks in repo.db
        all_landmarks = self.memory.search_landmarks(query="", limit=50, include_stale=False)
        relevant = self.memory.search_landmarks(query=user_prompt, limit=10, include_stale=False) if user_prompt else []
        landmark_cov = min(1.0, (len(all_landmarks) * 0.12) + (len(relevant) * 0.16))

        # 2. HistorySuccessRate (0.0 .. 1.0): from task_history table in repo.db
        hist_score = self._compute_history_score()

        # 3. PersonaAlignment (0.0 .. 1.0): based on learned & promoted StyleRules in persona.db
        rules = self.memory.list_style_rules(include_rejected=False)
        promoted_cnt = sum(1 for r in rules if r.status == RuleStatus.PROMOTED)
        active_cnt = len(rules)
        persona_align = min(1.0, (promoted_cnt * 0.30) + (active_cnt * 0.15))

        # 4. BlastRadiusRisk (0.0 .. 1.0)
        if intent == TaskIntent.EXPLAIN:
            blast_risk = 0.0
        else:
            prompt_lower = user_prompt.lower()
            high_risk_terms = ("schema", "migration", "database", "auth", "delete", "global", "全库", "重构所有")
            term_penalty = 0.25 if any(t in prompt_lower for t in high_risk_terms) else 0.0
            file_penalty = min(0.60, max(0, estimated_files_touched - 1) * 0.12)
            blast_risk = min(1.0, term_penalty + file_penalty)

        raw_score = (
            (0.35 * landmark_cov)
            + (0.30 * hist_score)
            + (0.20 * persona_align)
            - (0.35 * blast_risk)
        )
        # Baseline boost so even fresh repos start around 0.20-0.25
        score = round(max(0.0, min(1.0, raw_score + 0.15)), 3)

        if score >= 0.80 and blast_risk <= 0.25:
            level = AutonomyLevel.L4_DIGITAL_TWIN
            auto_clarify = True
            desc = "Digital Twin (Full autonomous execution & sandbox verification; uses recommended defaults without interrupting)"
        elif score >= 0.60:
            level = AutonomyLevel.L3_SEMI_AUTONOMOUS
            auto_clarify = True
            desc = "Semi-Autonomous (Auto-selects recommended defaults on minor ambiguities; full verification & independent review)"
        elif score >= 0.35:
            level = AutonomyLevel.L2_COLLABORATOR
            auto_clarify = False
            desc = "Trusted Collaborator (Standard scientific reproduce-implement-verify-review loop)"
        else:
            level = AutonomyLevel.L1_APPRENTICE
            auto_clarify = False
            desc = "Apprentice (Proactive probing & clarification before high-impact changes)"

        return AutonomyAssessment(
            level=level,
            score=score,
            landmark_coverage=round(landmark_cov, 3),
            history_success_rate=round(hist_score, 3),
            persona_alignment=round(persona_align, 3),
            blast_radius_risk=round(blast_risk, 3),
            auto_approve_clarifications=auto_clarify,
            rationale=desc,
        )

    def _compute_history_score(self) -> float:
        try:
            with sqlite3.connect(self.memory.repo_db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT confidence FROM task_history ORDER BY completed_at DESC LIMIT 20"
                ).fetchall()
        except Exception:
            return 0.0

        if not rows:
            return 0.0
        high_cnt = sum(1 for r in rows if r["confidence"] in ("HIGH", "MEDIUM"))
        win_rate = high_cnt / len(rows)
        volume_factor = min(1.0, len(rows) / 5.0)
        return round(win_rate * volume_factor, 3)


class SandboxVerifier:
    """
    Executes candidate code modifications or benchmark commands inside an isolated
    Git worktree (`worktree_sandbox`) and compares runtime & static metrics against the baseline.
    """

    def __init__(self, toolkit: WorkspaceToolkit) -> None:
        self.toolkit = toolkit

    async def verify_in_worktree(
        self,
        benchmark_or_test_cmd: str,
        candidate_mutator: Callable[[WorkspaceToolkit], Awaitable[None] | None] | None = None,
    ) -> SandboxComparisonReport:
        """
        1. Measure baseline diagnostics & execution time on the current repo.
        2. Spin up an isolated Git worktree, apply `candidate_mutator` (if provided),
           and measure candidate diagnostics & execution time.
        3. Return a structured `SandboxComparisonReport`.
        """
        base_radar = SemanticRadar(self.toolkit.repo_root)
        base_prof = RuntimeProfiler(self.toolkit.repo_root)
        base_diags = len(base_radar.check_diagnostics())
        base_rep = await base_prof.profile_execution(benchmark_or_test_cmd, timeout_sec=45.0)

        with self.toolkit.worktree_sandbox(prefix="fix_verify_") as wt_path:
            wt_toolkit = WorkspaceToolkit(wt_path)
            if candidate_mutator is not None:
                res = candidate_mutator(wt_toolkit)
                if res is not None:
                    await res
            cand_radar = SemanticRadar(wt_path)
            cand_prof = RuntimeProfiler(wt_path)
            cand_diags = len(cand_radar.check_diagnostics())
            cand_rep = await cand_prof.profile_execution(benchmark_or_test_cmd, timeout_sec=45.0)

        base_ms = max(0.1, base_rep.wall_time_ms)
        cand_ms = max(0.1, cand_rep.wall_time_ms)
        speedup = round(((base_ms - cand_ms) / base_ms) * 100.0, 2)
        passed = cand_rep.exit_code == 0 and cand_diags <= base_diags

        summary = (
            f"Worktree Sandbox Verification (`{benchmark_or_test_cmd}`): "
            f"baseline={base_ms:.1f}ms (diags={base_diags}) -> "
            f"candidate={cand_ms:.1f}ms (diags={cand_diags}, speedup={speedup:+.1f}%, "
            f"tests={'PASS' if passed else 'FAIL'})"
        )
        return SandboxComparisonReport(
            command=benchmark_or_test_cmd,
            baseline_exit_code=base_rep.exit_code,
            candidate_exit_code=cand_rep.exit_code,
            baseline_wall_ms=base_ms,
            candidate_wall_ms=cand_ms,
            speedup_pct=speedup,
            baseline_diagnostics=base_diags,
            candidate_diagnostics=cand_diags,
            tests_passed=passed,
            summary=summary,
        )
