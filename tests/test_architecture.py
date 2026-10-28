"""Architectural boundary and modular package structure verification tests."""

from __future__ import annotations

import importlib
from pathlib import Path


def test_codex_style_modular_package_layout_and_zero_flat_dumps() -> None:
    """Verify `src/fix/` is organized into crisp domain subpackages with no loose flat `.py` module dumps."""
    src_fix = Path(__file__).resolve().parent.parent / "src" / "fix"
    assert src_fix.is_dir()

    # Only __init__.py and __main__.py are allowed at the top level of src/fix/
    root_py_files = sorted(p.name for p in src_fix.glob("*.py"))
    assert root_py_files == ["__init__.py", "__main__.py"]

    # All 7 domain packages must exist with __init__.py
    expected_packages = ("protocol", "core", "llm", "tools", "memory", "tui", "cli")
    for pkg_name in expected_packages:
        pkg_dir = src_fix / pkg_name
        assert pkg_dir.is_dir(), f"Missing domain package directory: {pkg_name}"
        assert (pkg_dir / "__init__.py").is_file(), f"Missing __init__.py in {pkg_name}"

    # Verify no single module exceeds 3,000 lines (preventing the old 7,300-line cli.py monolith)
    for py_file in src_fix.rglob("*.py"):
        line_count = len(py_file.read_text(encoding="utf-8").splitlines())
        assert line_count < 3000, f"{py_file.relative_to(src_fix)} is too large ({line_count} lines)"


def test_canonical_domain_imports_and_backward_compatible_aliases() -> None:
    """Verify canonical subpackage imports and legacy flat imports resolve to identical symbols."""
    from fix.cli import FixInteractiveSession as CliSession, app, main
    from fix.core import AutonomyRouter, ContextManager, EvolutionEngine, FixLooper, FixRuntime
    from fix.llm import LLMGateway, LLMResponse, PROVIDER_SPECS, ScriptedMockLLM
    from fix.memory import DualMemoryStore
    from fix.protocol import ProbeDecision, ReviewVerdict, TaskIntent, TaskPhase, TaskSession, ToolCallRequest
    from fix.tools import RuntimeProfiler, SemanticRadar, WorkspaceToolkit
    from fix.tui import CleanMarkdown, CollapsibleProgressTracker, FixInteractiveSession as TuiSession

    assert CliSession is TuiSession
    assert callable(main)
    assert app is not None
    assert FixRuntime is not None
    assert ContextManager is not None
    assert FixLooper is not None
    assert AutonomyRouter is not None
    assert EvolutionEngine is not None
    assert LLMGateway is not None
    assert LLMResponse is not None
    assert ScriptedMockLLM is not None
    assert "deepseek" in PROVIDER_SPECS
    assert DualMemoryStore is not None
    assert WorkspaceToolkit is not None
    assert SemanticRadar is not None
    assert RuntimeProfiler is not None
    assert TaskSession is not None
    assert TaskIntent is not None
    assert TaskPhase is not None
    assert ProbeDecision is not None
    assert ReviewVerdict is not None
    assert ToolCallRequest is not None
    assert CleanMarkdown is not None
    assert CollapsibleProgressTracker is not None

    # Verify legacy flat module aliases resolve to the exact canonical submodules
    alias_pairs = [
        ("fix.schema", "fix.protocol.schema"),
        ("fix.runtime", "fix.core.runtime"),
        ("fix.context", "fix.core.context"),
        ("fix.looper", "fix.core.looper"),
        ("fix.autonomy", "fix.core.autonomy"),
        ("fix.evolution", "fix.core.evolution"),
        ("fix.lsp", "fix.tools.lsp"),
        ("fix.profiler", "fix.tools.profiler"),
    ]
    for legacy_mod_name, canonical_mod_name in alias_pairs:
        legacy_mod = importlib.import_module(legacy_mod_name)
        canonical_mod = importlib.import_module(canonical_mod_name)
        assert legacy_mod is canonical_mod, f"{legacy_mod_name} did not alias {canonical_mod_name}"
