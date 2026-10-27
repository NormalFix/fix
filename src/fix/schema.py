"""Global single source of truth for Fix data contracts (Zero-DTO Pydantic v2 schemas)."""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class TaskIntent(str, Enum):
    """Four execution tracks routed during the initial PROBE phase."""

    EXPLAIN = "explain"        # Deep read-only diagnosis & architectural Q&A (write tools disabled)
    BUG_FIX = "bug_fix"        # Scientific bug-fixing (Hypothesis -> Reproduce -> Implement -> Verify -> Review)
    FEATURE = "feature"        # Feature engineering, performance & architectural optimization
    DIRECT_EDIT = "direct"     # Targeted localized edit with clear user instructions


class TaskPhase(str, Enum):
    """Explicit Finite State Machine phases for FixRuntime."""

    PROBE = "probe"            # Fast 0-token/read-only radar to understand relevant code & intent
    CLARIFY = "clarify"        # Minimal high-leverage clarification (max 3 questions with defaults)
    REPRODUCE = "reproduce"    # Write/run failing test or reproduction command before touching prod code
    IMPLEMENT = "implement"    # Apply syntax-gated code patches
    VERIFY = "verify"          # Mechanical verification: run test suite, linter, or benchmark probe
    REVIEW = "review"          # Context-isolated independent Reviewer check
    SYNTHESIZE = "synthesize"  # Guaranteed final synthesis when tool exploration concludes
    DONE = "done"              # Task completed and ready for delivery


class AutonomyLevel(str, Enum):
    """Progressive L1-L4 Autonomy Levels computed per repository and task."""

    L1_APPRENTICE = "L1_APPRENTICE"
    L2_COLLABORATOR = "L2_COLLABORATOR"
    L3_SEMI_AUTONOMOUS = "L3_SEMI_AUTONOMOUS"
    L4_DIGITAL_TWIN = "L4_DIGITAL_TWIN"


class FactStatus(str, Enum):
    """Lifecycle status of a repository landmark / evidence fact."""

    HYPOTHESIS = "HYPOTHESIS"
    VERIFIED = "VERIFIED"


class RuleStatus(str, Enum):
    """Lifecycle status of a learned developer style / architectural rule."""

    CANDIDATE = "CANDIDATE"
    PENDING_APPROVAL = "PENDING_APPROVAL"
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"


class IdleTickKind(str, Enum):
    """Event-driven background P2 IdleHandler micro-tick types."""

    REFRESH_STALE_LANDMARKS = "refresh_stale_landmarks"  # 0-token deterministic SHA & co-change refresh
    ANALYZE_RECENT_COMMIT = "analyze_recent_commit"      # Semantic Equivalence Diff Filter on latest commit
    HOTSPOT_DEEP_STUDY = "hotspot_deep_study"            # Targeted study of high-coupling or high-repair file
    ARCHITECTURAL_ESSENCE_SYNTHESIS = "architectural_essence_synthesis"  # 3-layer Essence MentalModel & UX/Arch Optimization synthesis


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


class EvidenceFact(BaseModel):
    """Project-specific landmark stored in `<repo>/.fix/repo.db`."""

    fact_id: str = Field(default_factory=lambda: f"fact_{uuid.uuid4().hex[:10]}")
    summary: str
    category: Literal["architecture", "root_cause", "invariant", "gotcha", "performance"] = "architecture"
    evidence_paths: list[str] = Field(default_factory=list)
    commit_sha: str = ""
    confidence: float = 0.9
    status: FactStatus = FactStatus.VERIFIED
    stale: bool = False
    updated_at: float = Field(default_factory=time.time)


class StyleRule(BaseModel):
    """Developer mind-mirror rule stored in global `~/.fix/persona.db`."""

    rule_id: str = Field(default_factory=lambda: f"rule_{uuid.uuid4().hex[:10]}")
    pattern_desc: str
    preferred_way: str
    category: Literal["naming", "error_handling", "architecture", "testing", "performance", "general"] = "general"
    hit_count: int = 1
    status: RuleStatus = RuleStatus.CANDIDATE
    source_example: str = ""
    updated_at: float = Field(default_factory=time.time)


class CoChangeEdge(BaseModel):
    """Deterministic git co-modification coupling between two files."""

    file_a: str
    file_b: str
    co_commits: int
    confidence: float = 0.0


class SkillCard(BaseModel):
    """Reusable self-bootstrapped diagnostic or optimization skill stored in `.fix/skills/`."""

    slug: str
    title: str
    trigger_when: str
    steps: list[str] = Field(default_factory=list)
    verification_cmd: str = ""
    script_body: str | None = None
    created_at: float = Field(default_factory=time.time)


class MentalModel(BaseModel):
    """
    Three-Layer First-Principles Mental Model (`透过现象看本质`) stored in `<repo>/.fix/repo.db`:
    - Layer 1 (`phenomenon_layer`): Surface behavior, UI/API entry points, `[file:line]` anchors.
    - Layer 2 (`mechanism_contract_layer`): State machine, lifecycle ownership, concurrency & cross-module contract.
    - Layer 3 (`essence_first_principles`): Core domain invariant, single-source-of-truth vs dual-state tension, root essence.
    - UX Impact (`ux_and_product_impact`): User-perceived latency, animation continuity, edge-case feedback & ergonomics.
    """

    model_id: str = Field(default_factory=lambda: f"mm_{uuid.uuid4().hex[:10]}")
    component_or_scope: str
    phenomenon_layer: str
    mechanism_contract_layer: str
    essence_first_principles: str
    ux_and_product_impact: str = ""
    evidence_paths: list[str] = Field(default_factory=list)
    commit_sha: str = ""
    confidence: float = 0.92
    updated_at: float = Field(default_factory=time.time)


class OptimizationProposal(BaseModel):
    """
    Proactive Project & User-Experience Optimization Proposal stored in `<repo>/.fix/repo.db`.
    Covers 4 higher-order dimensions: `architecture`, `user_experience`, `reliability`, `performance`.
    """

    proposal_id: str = Field(default_factory=lambda: f"opt_{uuid.uuid4().hex[:10]}")
    title: str
    dimension: Literal["architecture", "user_experience", "reliability", "performance"] = "architecture"
    target_files: list[str] = Field(default_factory=list)
    phenomenon_symptom: str
    root_essence: str
    proposed_evolution: str
    ux_or_engineering_gain: str
    priority: Literal["P0", "P1", "P2"] = "P1"
    status: Literal["OPEN", "VERIFIED_IN_SANDBOX", "APPLIED", "DISMISSED"] = "OPEN"
    updated_at: float = Field(default_factory=time.time)


class CognitiveParadigm(BaseModel):
    """
    Self-Evolved Higher-Order Thinking Paradigm stored in `~/.fix/persona.db`.
    Guides Fix to penetrate surface phenomena into architectural essence and UX optimization.
    """

    paradigm_id: str = Field(default_factory=lambda: f"para_{uuid.uuid4().hex[:10]}")
    domain_trigger: str
    first_principles_checklist: list[str] = Field(default_factory=list)
    exploration_efficiency_lesson: str = ""
    ux_lens: str = ""
    hit_count: int = 1
    updated_at: float = Field(default_factory=time.time)


class MetaReflection(BaseModel):
    """Post-task meta-cognitive self-reflection & efficiency audit."""

    task_id: str
    efficiency_score: float = 1.0
    redundant_reads_detected: int = 0
    efficiency_lesson: str = ""
    paradigm_upgraded: CognitiveParadigm | None = None
    mental_models_distilled: int = 0
    optimization_proposals_generated: int = 0


class DiffAnalysisResult(BaseModel):
    """Output of Semantic Equivalence Filter when analyzing human edits after Fix."""

    is_semantically_equivalent_refactor: bool
    change_classification: Literal["style_refactor", "business_feature", "bugfix_adjustment", "mixed"]
    extracted_rules: list[StyleRule] = Field(default_factory=list)
    rationale: str = ""


class MemoryDelta(BaseModel):
    """Copy-on-Write buffer for atomic memory updates at the end of a digest or IdleHandler slice."""

    facts_to_upsert: list[EvidenceFact] = Field(default_factory=list)
    rules_to_upsert: list[StyleRule] = Field(default_factory=list)
    co_changes_to_upsert: list[CoChangeEdge] = Field(default_factory=list)
    skills_to_save: list[SkillCard] = Field(default_factory=list)
    mental_models_to_upsert: list[MentalModel] = Field(default_factory=list)
    optimization_proposals_to_upsert: list[OptimizationProposal] = Field(default_factory=list)
    paradigms_to_upsert: list[CognitiveParadigm] = Field(default_factory=list)
    stale_invalidated_count: int = 0

    def is_empty(self) -> bool:
        return not (
            self.facts_to_upsert
            or self.rules_to_upsert
            or self.co_changes_to_upsert
            or self.skills_to_save
            or self.mental_models_to_upsert
            or self.optimization_proposals_to_upsert
            or self.paradigms_to_upsert
            or self.stale_invalidated_count > 0
        )


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
