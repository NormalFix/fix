"""Tests for WorkspaceToolkit: AST syntax-gated patching, 8KB truncation, process-group timeout & snapshots."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from fix.tools import MAX_OUTPUT_CHARS, WorkspaceToolkit


def _init_git_repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "dev@fix.local"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Fix Tester"], cwd=tmp_path, check=True)
    sample = tmp_path / "calc.py"
    sample.write_text(
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n\n"
        "def multiply(a: int, b: int) -> int:\n"
        "    return a * b\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=tmp_path, check=True, capture_output=True)
    return tmp_path


def test_apply_patch_uniqueness_and_ast_syntax_gate(tmp_path: Path) -> None:
    repo = _init_git_repo(tmp_path)
    toolkit = WorkspaceToolkit(repo)

    # 1. Missing search_block is rejected
    res_missing = toolkit.apply_patch("calc.py", "def nonexistent():", "def foo():")
    assert not res_missing.ok
    assert res_missing.error_type == "SearchBlockNotFound"

    # 2. Ambiguous search_block (matches twice) is rejected
    res_ambig = toolkit.apply_patch("calc.py", "a: int, b: int", "x: int, y: int")
    assert not res_ambig.ok
    assert res_ambig.error_type == "AmbiguousSearchBlock"

    # 3. Syntax error in replacement is blocked by AST gate BEFORE touching disk
    original_disk = (repo / "calc.py").read_text(encoding="utf-8")
    res_syntax = toolkit.apply_patch(
        "calc.py",
        "def add(a: int, b: int) -> int:\n    return a + b",
        "def add(a: int, b: int) -> int\n    return a + b",  # missing colon!
    )
    assert not res_syntax.ok
    assert res_syntax.error_type == "SyntaxError"
    assert (repo / "calc.py").read_text(encoding="utf-8") == original_disk

    # 4. Valid unique patch succeeds
    res_ok = toolkit.apply_patch(
        "calc.py",
        "def add(a: int, b: int) -> int:\n    return a + b",
        "def add(a: int, b: int) -> int:\n    return int(a) + int(b)",
    )
    assert res_ok.ok
    assert "return int(a) + int(b)" in (repo / "calc.py").read_text(encoding="utf-8")


def test_inspect_symbols_and_search_code(tmp_path: Path) -> None:
    repo = _init_git_repo(tmp_path)
    toolkit = WorkspaceToolkit(repo)

    sym_res = toolkit.inspect_symbols("calc.py")
    assert sym_res.ok
    assert "def add(a, b) [calc.py:1-2]" in sym_res.output
    assert "def multiply(a, b) [calc.py:4-5]" in sym_res.output

    search_res = toolkit.search_code("def multiply")
    assert search_res.ok
    assert "calc.py:4:" in search_res.output


@pytest.mark.asyncio
async def test_8kb_output_truncation_and_timeout_kill(tmp_path: Path) -> None:
    repo = _init_git_repo(tmp_path)
    toolkit = WorkspaceToolkit(repo)

    # 1. Command generating > 20KB output is truncated to <= ~8.5KB and logged to .fix/logs/
    cmd = "python3 -c \"print('HEAD_MARKER\\n' + ('x' * 20000) + '\\nTAIL_MARKER')\""
    res = await toolkit.run_command(cmd)
    assert res.ok
    assert res.truncated
    assert res.full_log_path is not None
    assert (repo / res.full_log_path).exists()
    assert "HEAD_MARKER" in res.output
    assert "TAIL_MARKER" in res.output
    assert len(res.output) < MAX_OUTPUT_CHARS + 300

    # 2. Hanging command in process group is killed cleanly on timeout
    res_timeout = await toolkit.run_command("python3 -c 'import time; time.sleep(10)'", timeout_sec=0.4)
    assert not res_timeout.ok
    assert res_timeout.error_type == "CommandTimeout"


def test_step_snapshot_and_millisecond_rollback(tmp_path: Path) -> None:
    repo = _init_git_repo(tmp_path)
    toolkit = WorkspaceToolkit(repo)

    snap_id = toolkit.create_step_snapshot()
    assert snap_id

    # Modify tracked file and create a new untracked file
    toolkit.apply_patch(
        "calc.py",
        "def add(a: int, b: int) -> int:\n    return a + b",
        "def add(a: int, b: int) -> int:\n    return a - b",
    )
    toolkit.write_new_file("unwanted_helper.py", "X = 42\n")
    assert (repo / "unwanted_helper.py").exists()
    assert "return a - b" in (repo / "calc.py").read_text(encoding="utf-8")

    # Rollback to snapshot restores tracked file and deletes untracked file
    rolled_back = toolkit.rollback_to_snapshot(snap_id)
    assert rolled_back
    assert not (repo / "unwanted_helper.py").exists()
    assert "return a + b" in (repo / "calc.py").read_text(encoding="utf-8")
