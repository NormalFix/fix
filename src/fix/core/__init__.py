"""Core cognitive engine: FixRuntime, ContextManager, FixLooper, AutonomyRouter, and EvolutionEngine."""

from __future__ import annotations

from fix.core.autonomy import AutonomyRouter, SandboxVerifier
from fix.core.context import ContextManager, RepoProfile, resolve_project_root
from fix.core.evolution import EvolutionEngine
from fix.core.looper import FixLooper
from fix.core.runtime import FixRuntime, format_tool_call_summary

__all__ = [
    "AutonomyRouter",
    "ContextManager",
    "EvolutionEngine",
    "FixLooper",
    "FixRuntime",
    "RepoProfile",
    "SandboxVerifier",
    "format_tool_call_summary",
    "resolve_project_root",
]
