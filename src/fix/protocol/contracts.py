"""Execution, tool-call, semantic radar, profiler, sandbox, and review contracts."""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from fix.protocol.enums import AutonomyLevel, TaskIntent


class ClarifyOption(BaseModel):
    """A high-value clarification question when critical business ambiguity exists."""

    question: str
    options: list[str] = Field(default_factory=list)
    recommended: str = ""
    reason: str = ""


class ProbeDecision(BaseModel):
    """Structured decision output at the conclusion of the PROBE phase."""

    intent: TaskIntent
    summary: str
    root_cause_hypothesis: str = ""
    target_files: list[str] = Field(default_factory=list)
    reproduction_cmd: str | None = None
    needs_clarification: bool = False
    clarifications: list[ClarifyOption] = Field(default_factory=list)


class ToolCallRequest(BaseModel):
    """Normalized tool invocation emitted by LLMGateway."""

    call_id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:8]}")
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Standardized output from WorkspaceToolkit with 8KB truncation support."""

    ok: bool
    output: str
    full_log_path: str | None = None
    error_type: str | None = None
    truncated: bool = False


class SymbolLocation(BaseModel):
    """Exact location and signature of a symbol definition or reference across any language."""

    symbol: str
    kind: Literal[
        "function",
        "async_function",
        "method",
        "class",
        "struct",
        "trait",
        "interface",
        "object",
        "enum",
        "variable",
        "reference",
    ]
    rel_path: str
    line: int
    end_line: int
    signature: str = ""
    docstring: str = ""
    context_line: str = ""
    module_scope: str = ""


class CallHierarchyReport(BaseModel):
    """Universal polyglot cross-file semantic call graph, type hierarchy & declarative resource report."""

    symbol: str
    definitions: list[SymbolLocation] = Field(default_factory=list)
    supertypes: list[str] = Field(default_factory=list)
    implementations: list[SymbolLocation] = Field(default_factory=list)
    wiring_and_decorators: list[str] = Field(default_factory=list)
    declarative_config_links: list[str] = Field(default_factory=list)
    callers: list[SymbolLocation] = Field(default_factory=list)
    callees: list[str] = Field(default_factory=list)
    lsp_backend_used: str = "polyglot_ast_graph" 


class DiagnosticItem(BaseModel):
    """Static analysis / syntax / import / XML / type diagnostic item."""

    rel_path: str
    line: int
    col: int = 1
    severity: Literal["ERROR", "WARNING"] = "ERROR"
    code: str = ""
    message: str


class HotspotEntry(BaseModel):
    """Single function hotspot from runtime profiling (`profile_execution`)."""

    func_name: str
    rel_path: str
    line: int
    call_count: int
    total_time_ms: float
    cumulative_time_ms: float
    per_call_ms: float


class ProfileReport(BaseModel):
    """Structured runtime performance & line-coverage report (`profile_execution`)."""

    command: str
    exit_code: int
    wall_time_ms: float
    top_hotspots: list[HotspotEntry] = Field(default_factory=list)
    executed_lines_summary: dict[str, int] = Field(default_factory=dict)
    missed_lines: dict[str, list[str]] = Field(default_factory=dict)
    bottleneck_diagnosis_hint: str = ""
    stdout_tail: str = ""


class AutonomyAssessment(BaseModel):
    """Quantitative L1-L4 progressive autonomy assessment for the current project & task."""

    level: AutonomyLevel = AutonomyLevel.L1_APPRENTICE
    score: float = 0.0
    landmark_coverage: float = 0.0
    history_success_rate: float = 0.0
    persona_alignment: float = 0.0
    blast_radius_risk: float = 0.0
    auto_approve_clarifications: bool = False
    rationale: str = ""


class SandboxComparisonReport(BaseModel):
    """Empirical comparison between baseline workspace and isolated Git worktree sandbox."""

    command: str
    baseline_exit_code: int = 0
    candidate_exit_code: int = 0
    baseline_wall_ms: float = 0.0
    candidate_wall_ms: float = 0.0
    speedup_pct: float = 0.0
    baseline_diagnostics: int = 0
    candidate_diagnostics: int = 0
    tests_passed: bool = True
    summary: str = ""


class ReviewVerdict(BaseModel):
    """Structured verdict from the context-isolated Independent Reviewer."""

    approved: bool
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = "HIGH"
    blocking_issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    root_cause_verified: bool = True

