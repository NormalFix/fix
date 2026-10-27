"""Tests for Priority 1: Cross-File Semantic Call Graph (`src/fix/lsp.py`) & Runtime Profiler (`src/fix/profiler.py`)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from fix.tools import WorkspaceToolkit


def _create_multi_file_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "shop_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "dev@fix.local"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Fix Tester"], cwd=repo, check=True)

    (repo / "pricing.py").write_text(
        '"""Pricing engine module."""\n\n'
        "def compute_tax(subtotal: float) -> float:\n"
        '    """Compute standard 10% tax."""\n'
        "    return round(subtotal * 0.10, 2)\n\n"
        "def slow_bottleneck_calc(n: int = 120000) -> int:\n"
        '    """Deliberate CPU hotspot for profiler test."""\n'
        "    total = 0\n"
        "    for i in range(n):\n"
        "        total += (i * i) % 97\n"
        "    if n < 0:\n"
        "        total = -1\n"
        "    return total\n",
        encoding="utf-8",
    )

    (repo / "checkout.py").write_text(
        "from pricing import compute_tax, slow_bottleneck_calc\n\n"
        "class CheckoutService:\n"
        "    def finalize_order(self, amount: float) -> float:\n"
        '        """Finalize order total with tax and audit hash."""\n'
        "        tax = compute_tax(amount)\n"
        "        _ = slow_bottleneck_calc(80000)\n"
        "        return amount + tax\n",
        encoding="utf-8",
    )

    (repo / "test_checkout.py").write_text(
        "from checkout import CheckoutService\n\n"
        "def test_finalize_order():\n"
        "    svc = CheckoutService()\n"
        "    assert svc.finalize_order(100.0) == 110.0\n",
        encoding="utf-8",
    )

    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "Initial shop repo"], cwd=repo, check=True, capture_output=True)
    return repo


def test_cross_file_trace_symbol_definitions_callers_and_callees(tmp_path: Path) -> None:
    repo = _create_multi_file_repo(tmp_path)
    toolkit = WorkspaceToolkit(repo)

    # 1. Trace `compute_tax` -> finds definition in `pricing.py:3-5` and caller in `checkout.py:6`
    trace_tax = toolkit.trace_symbol("compute_tax")
    assert trace_tax.ok
    assert "[pricing.py:3-5]" in trace_tax.output
    assert "def compute_tax(subtotal) -> float" in trace_tax.output
    assert "[checkout.py:6]" in trace_tax.output

    # 2. Trace `CheckoutService.finalize_order` -> finds method definition and outgoing calls (`compute_tax`, `slow_bottleneck_calc`)
    trace_method = toolkit.trace_symbol("CheckoutService.finalize_order")
    assert trace_method.ok
    assert "[checkout.py:4-8]" in trace_method.output
    assert "compute_tax" in trace_method.output
    assert "slow_bottleneck_calc" in trace_method.output


def test_check_diagnostics_detects_unresolved_imports_and_undefined_names(tmp_path: Path) -> None:
    repo = _create_multi_file_repo(tmp_path)
    toolkit = WorkspaceToolkit(repo)

    # Clean repository has 0 diagnostics
    clean_res = toolkit.check_diagnostics()
    assert clean_res.ok
    assert "0 diagnostics" in clean_res.output

    # Introduce a broken import and an undefined variable in a new file
    (repo / "broken_module.py").write_text(
        "from pricing import non_existent_helper\n\n"
        "def run_bad():\n"
        "    return mystery_undefined_var + 1\n",
        encoding="utf-8",
    )

    diag_res = toolkit.check_diagnostics("broken_module.py")
    assert not diag_res.ok
    assert diag_res.error_type == "StaticDiagnosticsFound"
    assert "UnresolvedImportSymbol" in diag_res.output
    assert "non_existent_helper" in diag_res.output
    assert "UndefinedName" in diag_res.output
    assert "mystery_undefined_var" in diag_res.output


@pytest.mark.asyncio
async def test_runtime_profiler_identifies_hotspot_and_missed_coverage_lines(tmp_path: Path) -> None:
    repo = _create_multi_file_repo(tmp_path)
    toolkit = WorkspaceToolkit(repo)

    prof_res = await toolkit.profile_execution("pytest -q", focus_path="pricing.py")
    assert prof_res.ok
    # Verify cProfile captured `slow_bottleneck_calc` as a top bottleneck in pricing.py
    assert "slow_bottleneck_calc" in prof_res.output
    assert "pricing.py" in prof_res.output
    # Verify line coverage identified the unexecuted `if n < 0: total = -1` branch (line 13) in pricing.py
    assert "Missed / Unexecuted Statement Lines" in prof_res.output
    assert "[pricing.py] missed lines: 13" in prof_res.output
