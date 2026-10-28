"""Persistent memory, Mind Mirror style rules, co-change graph, 3-layer mental models, and skill schemas."""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field

from fix.protocol.enums import FactStatus, RuleStatus


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

