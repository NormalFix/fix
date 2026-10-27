"""Native Agent-Computer Interface (ACI): safe perception, syntax-gated patching, process-group sandbox & Git snapshots."""

from __future__ import annotations

import ast
import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
try:
    import tomllib
except ImportError:
    tomllib = None  # type: ignore[assignment]
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import xml.etree.ElementTree as ET
from fix.lsp import SemanticRadar, check_brace_balance_in_source
from fix.profiler import RuntimeProfiler
from fix.schema import ToolCallRequest, ToolResult

MAX_OUTPUT_CHARS = 8192
HEAD_KEEP_CHARS = 2048
TAIL_KEEP_CHARS = 6144

IGNORE_DIRS = {
    ".git",
    ".fix",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
    "target",
    "out",
    ".gradle",
    ".idea",
    ".kotlin",
    ".qoder",
    ".cursor",
    "captures",
    ".next",
}


def is_test_file_path(rel_path: str) -> bool:
    """Return True if rel_path points to a test file or sits inside a tests/ directory."""
    norm = rel_path.replace("\\", "/").strip("/")
    parts = norm.split("/")
    filename = parts[-1]
    if any(p in ("test", "tests", "__tests__", "spec", "specs") for p in parts[:-1]):
        return True
    return (
        filename.startswith("test_")
        or filename.endswith("_test.py")
        or filename.endswith(".test.ts")
        or filename.endswith(".spec.ts")
        or filename.endswith("_test.go")
    )


class WorkspaceToolkit:
    """High-safety native workspace toolkit with physical write gates and 8KB output truncation."""

    def __init__(self, repo_root: str | Path) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.fix_dir = self.repo_root / ".fix"
        self.logs_dir = self.fix_dir / "logs"
        self.snapshots_dir = self.fix_dir / "snapshots"
        self.radar = SemanticRadar(self.repo_root)
        self.profiler = RuntimeProfiler(self.repo_root)

    def _resolve_safe_path(self, rel_path: str) -> Path:
        target = (self.repo_root / rel_path).resolve()
        if target != self.repo_root and self.repo_root not in target.parents:
            raise ValueError(f"Path escapes repository root: {rel_path}")
        return target

    # ------------------------------------------------------------------
    # 1. Read-Only Perception Tools
    # ------------------------------------------------------------------

    def list_dir(self, rel_path: str = ".", max_depth: int = 2) -> ToolResult:
        """List directory hierarchy up to max_depth, ignoring noisy build/cache dirs."""
        try:
            base = self._resolve_safe_path(rel_path)
        except ValueError as exc:
            return ToolResult(ok=False, output=str(exc), error_type="PathSecurityError")

        if not base.exists():
            return ToolResult(ok=False, output=f"Path does not exist: {rel_path}", error_type="FileNotFound")

        lines: list[str] = []

        def _walk(curr: Path, depth: int) -> None:
            if depth > max_depth:
                return
            try:
                entries = sorted(curr.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except PermissionError:
                return
            for entry in entries:
                if entry.name in IGNORE_DIRS:
                    continue
                rel = entry.relative_to(self.repo_root)
                indent = "  " * depth
                suffix = "/" if entry.is_dir() else ""
                lines.append(f"{indent}{rel}{suffix}")
                if entry.is_dir() and depth < max_depth:
                    _walk(entry, depth + 1)

        _walk(base, 0)
        output = "\n".join(lines) if lines else "(empty directory)"
        return self._truncate_result(output, ok=True)

    def search_code(self, pattern: str, path_glob: str | None = None) -> ToolResult:
        """Search code using ripgrep (`rg -n`) with a deterministic Python regex fallback."""
        rg_bin = shutil.which("rg")
        if rg_bin:
            cmd = [
                rg_bin,
                "-n",
                "--no-heading",
                "--color=never",
                "--max-columns=300",
                "--max-columns-preview",
                "--glob=!.git/**",
                "--glob=!.fix/**",
                "--glob=!.venv/**",
                "--glob=!node_modules/**",
                "--glob=!__pycache__/**",
                "--glob=!.gradle/**",
                "--glob=!.idea/**",
                "--glob=!.kotlin/**",
                "--glob=!.qoder/**",
                "--glob=!.cursor/**",
                "--glob=!captures/**",
                "--glob=!**/build/**",
                "--glob=!*.li",
                "--glob=!*.min.js",
                "--glob=!*.map",
            ]
            if path_glob:
                cmd.append(f"--glob={path_glob}")
            cmd.extend([pattern, "."])
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=self.repo_root,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if proc.returncode in (0, 1):
                    out = proc.stdout.strip() or f"No matches found for pattern: {pattern}"
                    return self._truncate_result(out, ok=True)
            except Exception:
                pass

        # Pure-Python fallback
        try:
            regex = re.compile(pattern)
        except re.error:
            regex = re.compile(re.escape(pattern))

        matches: list[str] = []
        for root, dirs, files in os.walk(self.repo_root):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            for fname in sorted(files):
                fpath = Path(root) / fname
                rel = str(fpath.relative_to(self.repo_root))
                if path_glob and not Path(rel).match(path_glob):
                    continue
                try:
                    text = fpath.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                if fname.endswith((".li", ".min.js", ".map")):
                    continue
                for idx, line in enumerate(text.splitlines(), start=1):
                    if regex.search(line):
                        clipped_ln = line[:300] + " ... [line truncated]" if len(line) > 300 else line
                        matches.append(f"{rel}:{idx}:{clipped_ln}")
                        if len(matches) >= 200:
                            break
        out = "\n".join(matches) if matches else f"No matches found for pattern: {pattern}"
        return self._truncate_result(out, ok=True)

    def read_file(self, rel_path: str, start_line: int = 1, end_line: int = 500) -> ToolResult:
        """Read a file slice with 1-based line numbers anchored as `line_no| content`."""
        try:
            target = self._resolve_safe_path(rel_path)
        except ValueError as exc:
            return ToolResult(ok=False, output=str(exc), error_type="PathSecurityError")

        if not target.exists() or not target.is_file():
            return ToolResult(ok=False, output=f"File not found: {rel_path}", error_type="FileNotFound")

        if start_line > end_line and end_line > 0:
            start_line, end_line = end_line, start_line

        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        total = len(lines)
        s_idx = max(1, min(start_line, max(1, total)))
        # Adaptive full-file window: if starting at line 1 with default end_line=500 and file <= 950 lines,
        # read the entire file in one shot so 600-950 line classes never require fragmented re-reads!
        if s_idx == 1 and end_line == 500 and total <= 950:
            e_idx = total
        else:
            e_idx = min(total, max(s_idx, end_line))
        numbered = [f"{i:4d}| {lines[i - 1]}" for i in range(s_idx, e_idx + 1)]
        header = f"[{rel_path} lines {s_idx}-{e_idx} of {total}]"
        return self._truncate_result(header + "\n" + "\n".join(numbered), ok=True, max_chars=36000)

    def inspect_symbols(self, rel_path: str) -> ToolResult:
        """Extract structural AST symbols (classes, functions, signatures, line numbers)."""
        try:
            target = self._resolve_safe_path(rel_path)
        except ValueError as exc:
            return ToolResult(ok=False, output=str(exc), error_type="PathSecurityError")

        if not target.exists() or not target.is_file():
            return ToolResult(ok=False, output=f"File not found: {rel_path}", error_type="FileNotFound")

        source = target.read_text(encoding="utf-8", errors="replace")
        if target.suffix == ".py":
            try:
                tree = ast.parse(source, filename=rel_path)
            except SyntaxError as exc:
                return ToolResult(ok=False, output=f"SyntaxError in {rel_path}: {exc}", error_type="SyntaxError")

            items: list[str] = []
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    end_ln = getattr(node, "end_lineno", node.lineno)
                    items.append(f"class {node.name} [{rel_path}:{node.lineno}-{end_ln}]")
                    for sub in node.body:
                        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            args = [a.arg for a in sub.args.args]
                            s_end = getattr(sub, "end_lineno", sub.lineno)
                            prefix = "async def" if isinstance(sub, ast.AsyncFunctionDef) else "def"
                            items.append(f"  {prefix} {sub.name}({', '.join(args)}) [{rel_path}:{sub.lineno}-{s_end}]")
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    args = [a.arg for a in node.args.args]
                    end_ln = getattr(node, "end_lineno", node.lineno)
                    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                    items.append(f"{prefix} {node.name}({', '.join(args)}) [{rel_path}:{node.lineno}-{end_ln}]")
            out = "\n".join(items) if items else f"No top-level classes/functions in {rel_path}"
            return ToolResult(ok=True, output=out)

        # Generic regex symbol outline for TS/JS/Go/Rust/Java
        sym_pattern = re.compile(
            r"^\s*(?:export\s+)?(?:async\s+)?(?:function|class|interface|type|func|fn|struct|enum|def)\s+([A-Za-z0-9_]+)"
        )
        items = []
        for idx, line in enumerate(source.splitlines(), start=1):
            m = sym_pattern.match(line)
            if m:
                items.append(f"{line.strip()} [{rel_path}:{idx}]")
        out = "\n".join(items) if items else f"No symbols detected in {rel_path}"
        return ToolResult(ok=True, output=out)

    def trace_symbol(self, symbol_name: str, rel_path: str | None = None) -> ToolResult:
        """Trace cross-file definitions, callers, and outgoing callees for a symbol."""
        if not symbol_name.strip():
            return ToolResult(ok=False, output="symbol_name cannot be empty", error_type="InvalidArgument")
        report = self.radar.trace_symbol(symbol_name=symbol_name.strip(), rel_path=rel_path)
        formatted = self.radar.format_trace_report(report)
        return self._truncate_result(formatted, ok=True)

    def check_diagnostics(self, rel_path: str | None = None) -> ToolResult:
        """Run fast static syntax, import resolution, undefined name, and LSP/pyright diagnostics."""
        diags = self.radar.check_diagnostics(rel_path=rel_path)
        formatted = self.radar.format_diagnostics(diags)
        return self._truncate_result(
            formatted,
            ok=(len(diags) == 0),
            error_type=None if len(diags) == 0 else "StaticDiagnosticsFound",
        )

    async def profile_execution(
        self,
        cmd: str,
        focus_path: str | None = None,
        timeout_sec: float = 60.0,
    ) -> ToolResult:
        """Run command/test under runtime hotspot (`cProfile`) & line-coverage (`sys.settrace`) probe."""
        report = await self.profiler.profile_execution(
            target_cmd=cmd,
            focus_path=focus_path,
            timeout_sec=timeout_sec,
        )
        formatted = self.profiler.format_profile_report(report)
        return self._truncate_result(
            formatted,
            ok=(report.exit_code == 0),
            error_type=None if report.exit_code == 0 else f"ExitCode_{report.exit_code}",
        )

    async def run_skill(self, slug: str, target: str = "") -> ToolResult:
        """Execute a self-bootstrapped skill script from `.fix/skills/<slug>/run.py`."""
        clean_slug = Path(slug).name
        skill_dir = self.fix_dir / "skills" / clean_slug
        script_path = skill_dir / "run.py"
        if not script_path.exists():
            script_path = skill_dir / "probe.py"
        if not script_path.exists():
            md_path = skill_dir / "SKILL.md"
            if md_path.exists():
                return ToolResult(ok=True, output=md_path.read_text(encoding="utf-8"))
            return ToolResult(
                ok=False,
                output=f"Skill `{clean_slug}` not found in `.fix/skills/`.",
                error_type="SkillNotFound",
            )
        target_arg = f' --target "{target}"' if target else ""
        cmd = f'"{sys.executable}" "{script_path}" --repo "{self.repo_root}"{target_arg}'
        return await self.run_command(cmd, timeout_sec=30.0)

    # ------------------------------------------------------------------
    # 2. Syntax-Gated Write Tools
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_syntax_in_memory(rel_path: str, content: str) -> str | None:
        """Validate syntax in memory before touching disk. Returns error message or None."""
        suffix = Path(rel_path).suffix.lower()
        if suffix == ".py":
            try:
                ast.parse(content, filename=rel_path)
            except SyntaxError as exc:
                return f"Python AST SyntaxError at line {exc.lineno}, col {exc.offset}: {exc.msg}"
        elif suffix == ".json":
            try:
                json.loads(content)
            except json.JSONDecodeError as exc:
                return f"Invalid JSON at line {exc.lineno}, col {exc.colno}: {exc.msg}"
        elif suffix == ".toml" and tomllib is not None:
            try:
                tomllib.loads(content)
            except Exception as exc:
                return f"Invalid TOML: {exc}"
        elif suffix == ".xml":
            try:
                ET.fromstring(content)
            except ET.ParseError as exc:
                return f"Invalid XML syntax: {exc}"
        elif suffix in (".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java", ".kt", ".c", ".cpp", ".cc", ".h", ".hpp", ".cs", ".swift"):
            ok, err_ln, msg = check_brace_balance_in_source(content)
            if not ok:
                return f"Delimiter syntax error in {rel_path} at line {err_ln}: {msg}" 
        return None

    def apply_patch(self, rel_path: str, search_block: str, replace_block: str) -> ToolResult:
        """Replace a unique search_block with replace_block after in-memory AST syntax validation."""
        try:
            target = self._resolve_safe_path(rel_path)
        except ValueError as exc:
            return ToolResult(ok=False, output=str(exc), error_type="PathSecurityError")

        if not target.exists() or not target.is_file():
            return ToolResult(ok=False, output=f"File not found: {rel_path}", error_type="FileNotFound")

        original = target.read_text(encoding="utf-8")
        if not search_block:
            return ToolResult(ok=False, output="search_block cannot be empty", error_type="InvalidPatch")

        occurrences = original.count(search_block)
        if occurrences == 0:
            return ToolResult(
                ok=False,
                output=(
                    f"search_block not found in {rel_path}. Read the target lines with `read_file` "
                    "and ensure exact whitespace/indentation match."
                ),
                error_type="SearchBlockNotFound",
            )
        if occurrences > 1:
            return ToolResult(
                ok=False,
                output=(
                    f"search_block matched {occurrences} locations in {rel_path}. "
                    "Include more surrounding context lines to make the match unique."
                ),
                error_type="AmbiguousSearchBlock",
            )

        updated = original.replace(search_block, replace_block, 1)
        syntax_err = self._validate_syntax_in_memory(rel_path, updated)
        if syntax_err is not None:
            return ToolResult(
                ok=False,
                output=f"Patch rejected by syntax gate for {rel_path}: {syntax_err}",
                error_type="SyntaxError",
            )

        target.write_text(updated, encoding="utf-8")
        return ToolResult(ok=True, output=f"Successfully patched {rel_path}")

    def write_new_file(self, rel_path: str, content: str, overwrite: bool = True) -> ToolResult:
        """Create or overwrite a file after in-memory syntax pre-validation."""
        try:
            target = self._resolve_safe_path(rel_path)
        except ValueError as exc:
            return ToolResult(ok=False, output=str(exc), error_type="PathSecurityError")

        if target.exists() and not overwrite:
            return ToolResult(ok=False, output=f"File already exists: {rel_path}", error_type="FileExists")

        syntax_err = self._validate_syntax_in_memory(rel_path, content)
        if syntax_err is not None:
            return ToolResult(
                ok=False,
                output=f"Write rejected by syntax gate for {rel_path}: {syntax_err}",
                error_type="SyntaxError",
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return ToolResult(ok=True, output=f"Successfully wrote {rel_path} ({len(content)} bytes)")

    # ------------------------------------------------------------------
    # 3. Controlled Execution Sandbox (Process Group + 8KB Truncation)
    # ------------------------------------------------------------------

    async def run_command(self, cmd: str, timeout_sec: float = 60.0) -> ToolResult:
        """Run shell command in an isolated process group with timeout tree-kill & 8KB truncation."""
        import sys
        env = dict(os.environ)
        venv_bin = str((Path(sys.prefix) / 'bin').absolute())
        curr_path = env.get("PATH", "")
        if venv_bin not in curr_path.split(os.pathsep):
            env["PATH"] = f"{venv_bin}{os.pathsep}{curr_path}" if curr_path else venv_bin
        curr_pypath = env.get("PYTHONPATH", "")
        repo_str = str(self.repo_root)
        env["PYTHONPATH"] = f"{repo_str}{os.pathsep}{curr_pypath}" if curr_pypath else repo_str

        proc = await asyncio.create_subprocess_shell(
            cmd,
            cwd=self.repo_root,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
        except asyncio.TimeoutError:
            self._kill_process_group(proc.pid)
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
            return ToolResult(
                ok=False,
                output=f"Command timed out after {timeout_sec}s and process group was terminated: {cmd}",
                error_type="CommandTimeout",
            )
        except asyncio.CancelledError:
            self._kill_process_group(proc.pid)
            raise

        raw_output = stdout_bytes.decode("utf-8", errors="replace")
        exit_code = proc.returncode or 0
        header = f"[exit_code={exit_code}]\n"
        return self._truncate_result(
            header + raw_output,
            ok=(exit_code == 0),
            error_type=None if exit_code == 0 else f"ExitCode_{exit_code}",
        )

    @staticmethod
    def _kill_process_group(pid: int) -> None:
        """Terminate an entire process group to prevent orphan test/compiler processes."""
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
            time.sleep(0.05)
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def _truncate_result(
        self,
        text: str,
        ok: bool,
        error_type: str | None = None,
        max_chars: int = MAX_OUTPUT_CHARS,
    ) -> ToolResult:
        """Enforce the context firewall: keep head + tail and persist full output when exceeding `max_chars`."""
        if len(text) <= max_chars:
            return ToolResult(ok=ok, output=text, error_type=error_type, truncated=False)

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log_name = f"tool_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}.log"
        log_path = self.logs_dir / log_name
        log_path.write_text(text, encoding="utf-8")
        rel_log = str(log_path.relative_to(self.repo_root))

        omitted = len(text) - (HEAD_KEEP_CHARS + TAIL_KEEP_CHARS)
        compact = (
            text[:HEAD_KEEP_CHARS]
            + f"\n\n... [TRUNCATED {omitted} chars; full output saved to {rel_log}] ...\n\n"
            + text[-TAIL_KEEP_CHARS:]
        )
        return ToolResult(
            ok=ok,
            output=compact,
            full_log_path=rel_log,
            error_type=error_type,
            truncated=True,
        )

    # ------------------------------------------------------------------
    # 4. Step-Level Git Snapshots & Worktree Isolation
    # ------------------------------------------------------------------

    def get_head_sha(self) -> str:
        """Return current HEAD commit SHA or empty string if not a git repo."""
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip() if proc.returncode == 0 else ""

    def get_git_diff(self, staged: bool = False) -> str:
        """Return git diff of the working tree (including newly tracked/modified files)."""
        cmd = ["git", "diff", "--cached"] if staged else ["git", "diff", "HEAD"]
        proc = subprocess.run(cmd, cwd=self.repo_root, capture_output=True, text=True)
        if proc.returncode != 0:
            proc = subprocess.run(["git", "diff"], cwd=self.repo_root, capture_output=True, text=True)
        return proc.stdout

    def create_step_snapshot(self) -> str:
        """Create a fast step-level snapshot (`git stash create` + untracked state backup)."""
        stash_proc = subprocess.run(
            ["git", "stash", "create"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
        )
        git_stash_sha = stash_proc.stdout.strip() if stash_proc.returncode == 0 else ""
        head_sha = self.get_head_sha()
        snap_id = git_stash_sha or f"snap_{head_sha[:8]}_{uuid.uuid4().hex[:8]}"

        files_state: dict[str, str] = {}
        status_proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
        )
        # If git repo has < 400 total files, snapshot full text tree; for large repos (>400 files),
        # combine `git stash create` tree SHA with dirty/untracked files from `git status --porcelain`!
        all_files_proc = subprocess.run(
            ["git", "ls-files"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
        )
        tracked_list = all_files_proc.stdout.splitlines() if all_files_proc.returncode == 0 else []
        snap_mode = "full_tree"
        if len(tracked_list) <= 400:
            extra_ignore = IGNORE_DIRS | {"target", "out", "captures", "libs", "vendor", "Pods"}
            files_seen = 0
            for root, dirs, files in os.walk(self.repo_root):
                dirs[:] = [d for d in dirs if d not in extra_ignore and not d.startswith(".")]
                for fname in files:
                    if fname.startswith("."):
                        continue
                    files_seen += 1
                    if files_seen > 400:
                        break
                    fpath = Path(root) / fname
                    try:
                        if fpath.stat().st_size > 262144:
                            continue
                        rel = str(fpath.relative_to(self.repo_root))
                        files_state[rel] = fpath.read_text(encoding="utf-8")
                    except (UnicodeDecodeError, OSError):
                        continue
                if files_seen > 400:
                    break
        else:
            snap_mode = "git_delta"
            if status_proc.returncode == 0:
                for line in status_proc.stdout.splitlines():
                    if len(line) > 3:
                        rel = line[3:].strip().strip('"')
                        if rel.startswith(".fix"):
                            continue
                        fpath = self.repo_root / rel
                        if fpath.is_file():
                            try:
                                files_state[rel] = fpath.read_text(encoding="utf-8")
                            except (UnicodeDecodeError, OSError):
                                continue

        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        snap_file = self.snapshots_dir / f"{snap_id}.json"
        snap_file.write_text(
            json.dumps(
                {
                    "snapshot_id": snap_id,
                    "mode": snap_mode,
                    "git_stash_sha": git_stash_sha,
                    "head_sha": head_sha,
                    "files": files_state,
                    "created_at": time.time(),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return snap_id

    def rollback_to_snapshot(self, snapshot_id: str) -> bool:
        """Millisecond rollback of the working tree to a previously captured step snapshot."""
        snap_file = self.snapshots_dir / f"{snapshot_id}.json"
        if not snap_file.exists():
            if re.fullmatch(r"[0-9a-f]{40}", snapshot_id):
                subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=self.repo_root, capture_output=True)
                res = subprocess.run(["git", "stash", "apply", snapshot_id], cwd=self.repo_root, capture_output=True)
                return res.returncode == 0
            return False

        data = json.loads(snap_file.read_text(encoding="utf-8"))
        saved_files: dict[str, str] = data.get("files", {})
        snap_mode = data.get("mode", "full_tree")

        if snap_mode == "git_delta":
            # In large git repositories (>400 files), restore tracked files via git checkout / stash
            # and remove only newly created untracked files that were not present in saved_files.
            subprocess.run(["git", "checkout", "--", "."], cwd=self.repo_root, capture_output=True)
            git_stash_sha = data.get("git_stash_sha", "")
            if git_stash_sha:
                subprocess.run(["git", "stash", "apply", git_stash_sha], cwd=self.repo_root, capture_output=True)
            st_proc = subprocess.run(
                ["git", "status", "--porcelain", "-uall"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
            )
            if st_proc.returncode == 0:
                for line in st_proc.stdout.splitlines():
                    if line.startswith("?? ") and len(line) > 3:
                        rel = line[3:].strip().strip('"')
                        if rel.startswith(".fix") or rel in saved_files:
                            continue
                        fpath = self.repo_root / rel
                        if fpath.is_file():
                            try:
                                fpath.unlink()
                            except OSError:
                                pass
        else:
            # Remove untracked/newly created files that did not exist in the full-tree snapshot
            for root, dirs, files in os.walk(self.repo_root):
                dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
                for fname in files:
                    fpath = Path(root) / fname
                    rel = str(fpath.relative_to(self.repo_root))
                    if rel not in saved_files:
                        try:
                            fpath.unlink()
                        except OSError:
                            pass

        # Restore exact file contents from snapshot
        for rel, content in saved_files.items():
            target = self._resolve_safe_path(rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        return True

    @contextmanager
    def worktree_sandbox(self, prefix: str = "fix_sandbox_") -> Iterator[Path]:
        """Isolated git worktree context manager for background benchmarks & refactoring."""
        tmp_dir = Path(tempfile.mkdtemp(prefix=prefix))
        added = False
        try:
            proc = subprocess.run(
                ["git", "worktree", "add", "--detach", str(tmp_dir), "HEAD"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
            )
            if proc.returncode == 0:
                added = True
            else:
                shutil.copytree(
                    self.repo_root,
                    tmp_dir,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(".git", ".fix", ".venv", "node_modules", "__pycache__"),
                )
            yield tmp_dir
        finally:
            if added:
                subprocess.run(
                    ["git", "worktree", "remove", "--force", str(tmp_dir)],
                    cwd=self.repo_root,
                    capture_output=True,
                )
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # 5. OpenAI Tool Schemas & Physical Gate Dispatcher
    # ------------------------------------------------------------------

    def get_tool_schemas(self, allow_write: bool = True) -> list[dict[str, Any]]:
        """Return OpenAI tool definitions filtered by physical write permission."""
        schemas: list[dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": "list_dir",
                    "description": "List files and directories under rel_path.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "rel_path": {"type": "string", "default": "."},
                            "max_depth": {"type": "integer", "default": 2},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_code",
                    "description": "Search codebase with regex pattern and return file:line matches.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "path_glob": {"type": "string"},
                        },
                        "required": ["pattern"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read line-numbered file content between start_line and end_line.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "rel_path": {"type": "string"},
                            "start_line": {"type": "integer", "default": 1},
                            "end_line": {"type": "integer", "default": 500},
                        },
                        "required": ["rel_path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "inspect_symbols",
                    "description": "Extract AST classes, functions, signatures and line spans for a file.",
                    "parameters": {
                        "type": "object",
                        "properties": {"rel_path": {"type": "string"}},
                        "required": ["rel_path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "Run test suite, linter, or diagnostic script in a process-group sandbox.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "cmd": {"type": "string"},
                            "timeout_sec": {"type": "number", "default": 60},
                        },
                        "required": ["cmd"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "trace_symbol",
                    "description": "Cross-file LSP/AST call-graph radar: find exact definition(s), callers across all files, and outgoing callees.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol_name": {"type": "string"},
                            "rel_path": {"type": "string"},
                        },
                        "required": ["symbol_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "check_diagnostics",
                    "description": "Run static syntax, import resolution, undefined symbol, and type diagnostics.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "rel_path": {"type": "string"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "profile_execution",
                    "description": "Run pytest or python script with cProfile hotspot profiling (Top 10 bottlenecks) and line-level execution coverage.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "cmd": {"type": "string"},
                            "focus_path": {"type": "string"},
                            "timeout_sec": {"type": "number", "default": 60},
                        },
                        "required": ["cmd"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_skill",
                    "description": "Execute a bootstrapped diagnostic skill from `.fix/skills/<slug>/run.py`.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "slug": {"type": "string"},
                            "target": {"type": "string", "default": ""},
                        },
                        "required": ["slug"],
                    },
                },
            },
        ]
        if allow_write:
            schemas.extend(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "apply_patch",
                            "description": "Replace a unique search_block with replace_block (AST syntax-checked).",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "rel_path": {"type": "string"},
                                    "search_block": {"type": "string"},
                                    "replace_block": {"type": "string"},
                                },
                                "required": ["rel_path", "search_block", "replace_block"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "write_new_file",
                            "description": "Create or overwrite a file with in-memory syntax validation.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "rel_path": {"type": "string"},
                                    "content": {"type": "string"},
                                    "overwrite": {"type": "boolean", "default": True},
                                },
                                "required": ["rel_path", "content"],
                            },
                        },
                    },
                ]
            )
        return schemas

    async def execute_tool_call(
        self,
        call: ToolCallRequest,
        allow_write: bool = True,
        restrict_write_to_tests: bool = False,
    ) -> ToolResult:
        """Dispatch a tool call while enforcing track/phase physical permissions."""
        name = call.name
        args = call.arguments

        if name in ("apply_patch", "write_new_file"):
            if not allow_write:
                return ToolResult(
                    ok=False,
                    output=f"Tool `{name}` is physically disabled in read-only/probe mode.",
                    error_type="WritePermissionDenied",
                )
            rel_path = str(args.get("rel_path", ""))
            if restrict_write_to_tests and not is_test_file_path(rel_path):
                return ToolResult(
                    ok=False,
                    output=(
                        f"REPRODUCE phase physical gate: cannot modify production file `{rel_path}` "
                        "before reproducing the bug. Only test files (`test_*.py`, `tests/**`) may be edited."
                    ),
                    error_type="ReproduceGateViolation",
                )

        if name == "list_dir":
            return self.list_dir(
                rel_path=str(args.get("rel_path", ".")),
                max_depth=int(args.get("max_depth", 2)),
            )
        if name == "search_code":
            return self.search_code(
                pattern=str(args.get("pattern", "")),
                path_glob=args.get("path_glob"),
            )
        if name == "read_file":
            return self.read_file(
                rel_path=str(args.get("rel_path", "")),
                start_line=int(args.get("start_line", 1)),
                end_line=int(args.get("end_line", 500)),
            )
        if name == "inspect_symbols":
            return self.inspect_symbols(rel_path=str(args.get("rel_path", "")))
        if name == "trace_symbol":
            return self.trace_symbol(
                symbol_name=str(args.get("symbol_name", "")),
                rel_path=args.get("rel_path"),
            )
        if name == "check_diagnostics":
            return self.check_diagnostics(rel_path=args.get("rel_path"))
        if name == "profile_execution":
            return await self.profile_execution(
                cmd=str(args.get("cmd", "")),
                focus_path=args.get("focus_path"),
                timeout_sec=float(args.get("timeout_sec", 60.0)),
            )
        if name == "run_skill":
            return await self.run_skill(
                slug=str(args.get("slug", "")),
                target=str(args.get("target", "")),
            )
        if name == "apply_patch":
            return self.apply_patch(
                rel_path=str(args.get("rel_path", "")),
                search_block=str(args.get("search_block", "")),
                replace_block=str(args.get("replace_block", "")),
            )
        if name == "write_new_file":
            return self.write_new_file(
                rel_path=str(args.get("rel_path", "")),
                content=str(args.get("content", "")),
                overwrite=bool(args.get("overwrite", True)),
            )
        if name == "run_command":
            return await self.run_command(
                cmd=str(args.get("cmd", "")),
                timeout_sec=float(args.get("timeout_sec", 60.0)),
            )
        return ToolResult(ok=False, output=f"Unknown tool: {name}", error_type="UnknownTool")
