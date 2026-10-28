"""Task session state, token/step budgets, and background P2 idle scheduler models."""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field

from fix.protocol.contracts import AutonomyAssessment, ClarifyOption, ReviewVerdict
from fix.protocol.enums import IdleTickKind, TaskIntent, TaskPhase
from fix.protocol.memory_models import MetaReflection


class TaskBudget(BaseModel):
    """Hard execution & token safety bounds for a single task session."""

    max_steps: int = 35
    max_repair_attempts: int = 4
    max_reproduce_attempts: int = 2
    max_review_rounds: int = 2
    steps_used: int = 0
    repairs_used: int = 0
    reproduce_attempts: int = 0
    review_rounds: int = 0


class IdleBudget(BaseModel):
    """Hard token/tick budget for P2 IdleHandler background evolution per idle window."""

    max_llm_ticks_per_idle: int = 2
    max_deterministic_ticks_per_idle: int = 10
    llm_ticks_used: int = 0
    deterministic_ticks_used: int = 0


class IdleTick(BaseModel):
    """A single bounded unit of background evolution work in the P2 IdleHandler queue."""

    tick_id: str = Field(default_factory=lambda: f"tick_{uuid.uuid4().hex[:8]}")
    kind: IdleTickKind
    target: str = ""
    requires_llm: bool = False
    priority: int = 2
    created_at: float = Field(default_factory=time.time)


class IdleTickOutcome(BaseModel):
    """Result of executing one P2 IdleTick."""

    tick_id: str
    kind: IdleTickKind
    target: str = ""
    preempted: bool = False
    committed: bool = False
    summary: str = ""
    elapsed_ms: float = 0.0



class TaskSession(BaseModel):
    """Complete runtime state of a single Fix task execution."""

    task_id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:10]}")
    repo_root: str
    user_prompt: str
    intent: TaskIntent | None = None
    phase: TaskPhase = TaskPhase.PROBE
    autonomy: AutonomyAssessment = Field(default_factory=AutonomyAssessment)
    budget: TaskBudget = Field(default_factory=TaskBudget)
    snapshots: list[str] = Field(default_factory=list)
    modified_files: set[str] = Field(default_factory=set)
    reproduction_cmd: str | None = None
    reproduction_confirmed: bool = False
    degraded_without_repro: bool = False
    root_cause_hypothesis: str = ""
    final_summary: str = ""
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = "HIGH"
    clarification_questions: list[ClarifyOption] = Field(default_factory=list)
    last_review: ReviewVerdict | None = None
    last_test_output: str = ""
    fix_diff_snapshot: str = ""
    steering_notes: list[str] = Field(default_factory=list)
    started_at: float = Field(default_factory=time.time)
    completed_at: float | None = None
    elapsed_sec: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    tool_calls_count: int = 0
    tool_calls_log: list[str] = Field(default_factory=list)
    tool_counts_by_name: dict[str, int] = Field(default_factory=dict)
    meta_reflection: MetaReflection | None = None
