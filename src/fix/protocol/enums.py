"""Core enumeration types for Fix task intents, state-machine phases, autonomy levels, and memory lifecycle."""

from __future__ import annotations

from enum import Enum


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

