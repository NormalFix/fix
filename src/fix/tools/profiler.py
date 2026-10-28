"""Runtime Performance Hotspot & Line-Coverage Profiler Probe (`src/fix/profiler.py`)."""

from __future__ import annotations

import ast
import asyncio
import json
import os
import shlex
import sys
import tempfile
import time
from pathlib import Path

from fix.protocol.schema import HotspotEntry, ProfileReport

_HARNESS_SCRIPT = r"""
import cProfile
import json
import os
import pstats
import runpy
import sys

out_json = sys.argv[1]
repo_root = os.path.abspath(sys.argv[2])
mode = sys.argv[3]
target = sys.argv[4]
target_args = sys.argv[5:]

executed_lines = {}

def _tracer(frame, event, arg):
    if event == "line":
        fname = os.path.abspath(frame.f_code.co_filename)
        if fname.startswith(repo_root) and "/.venv/" not in fname and "/.fix/" not in fname:
            rel = os.path.relpath(fname, repo_root)
            executed_lines.setdefault(rel, set()).add(frame.f_lineno)
    return _tracer

prof = cProfile.Profile()
sys.argv = [target] + target_args
exit_code = 0

sys.settrace(_tracer)
prof.enable()
try:
    if mode == "module":
        runpy.run_module(target, run_name="__main__", alter_sys=True)
    else:
        runpy.run_path(target, run_name="__main__")
except SystemExit as exc:
    if isinstance(exc.code, int):
        exit_code = exc.code
    elif exc.code:
        exit_code = 1
except Exception:
    import traceback
    traceback.print_exc()
    exit_code = 1
finally:
    prof.disable()
    sys.settrace(None)

stats = pstats.Stats(prof)
hotspots = []
for (fname, lineno, func_name), (cc, nc, tt, ct, _callers) in stats.stats.items():
    abs_f = os.path.abspath(fname) if os.path.exists(fname) else fname
    if abs_f.startswith(repo_root) and "/.venv/" not in abs_f and "/.fix/" not in abs_f:
        rel_f = os.path.relpath(abs_f, repo_root)
        per_call = (ct * 1000.0 / nc) if nc else 0.0
        hotspots.append({
            "func_name": func_name,
            "rel_path": rel_f,
            "line": int(lineno),
            "call_count": int(nc),
            "total_time_ms": round(tt * 1000.0, 3),
            "cumulative_time_ms": round(ct * 1000.0, 3),
            "per_call_ms": round(per_call, 3),
        })

hotspots.sort(key=lambda h: (h["cumulative_time_ms"], h["total_time_ms"]), reverse=True)

payload = {
    "exit_code": exit_code,
    "hotspots": hotspots[:15],
    "executed_lines": {k: sorted(v) for k, v in executed_lines.items()},
}
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(payload, f)
sys.exit(exit_code)
"""


class RuntimeProfiler:
    """
    Executes commands/tests under a deterministic performance & line-coverage probe.
    Extracts Top 10 bottleneck functions (`cProfile`) and executed vs missed lines (`sys.settrace`).
    """

    def __init__(self, repo_root: str | Path) -> None:
        self.repo_root = Path(repo_root).resolve()

    async def profile_execution(
        self,
        target_cmd: str,
        focus_path: str | None = None,
        timeout_sec: float = 60.0,
    ) -> ProfileReport:
        """Run `target_cmd` with runtime hotspot & line coverage instrumentation."""
        mode, target, args = self._parse_python_command(target_cmd)

        env = dict(os.environ)
        venv_bin = str((Path(sys.prefix) / "bin").absolute())
        curr_path = env.get("PATH", "")
        if venv_bin not in curr_path.split(os.pathsep):
            env["PATH"] = f"{venv_bin}{os.pathsep}{curr_path}" if curr_path else venv_bin
        curr_pypath = env.get("PYTHONPATH", "")
        repo_str = str(self.repo_root)
        env["PYTHONPATH"] = f"{repo_str}{os.pathsep}{curr_pypath}" if curr_pypath else repo_str

        with tempfile.TemporaryDirectory(prefix="fix_prof_") as tmp_dir:
            tmp_p = Path(tmp_dir)
            harness_py = tmp_p / "_fix_harness.py"
            out_json = tmp_p / "profile_out.json"
            harness_py.write_text(_HARNESS_SCRIPT, encoding="utf-8")

            if mode is not None:
                cmd_list = [
                    sys.executable,
                    str(harness_py),
                    str(out_json),
                    str(self.repo_root),
                    mode,
                    target,
                    *args,
                ]
                t0 = time.perf_counter()
                proc = await asyncio.create_subprocess_exec(
                    *cmd_list,
                    cwd=self.repo_root,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True,
                )
            else:
                t0 = time.perf_counter()
                proc = await asyncio.create_subprocess_shell(
                    target_cmd,
                    cwd=self.repo_root,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True,
                )

            try:
                stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
            except asyncio.TimeoutError:
                try:
                    os.killpg(os.getpgid(proc.pid), 9)
                except Exception:
                    pass
                return ProfileReport(
                    command=target_cmd,
                    exit_code=124,
                    wall_time_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                    bottleneck_diagnosis_hint=f"Command timed out after {timeout_sec}s.",
                )

            wall_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            raw_out = stdout_bytes.decode("utf-8", errors="replace")
            exit_code = proc.returncode or 0

            hotspots: list[HotspotEntry] = []
            executed_map: dict[str, list[int]] = {}
            if out_json.exists():
                try:
                    pdata = json.loads(out_json.read_text(encoding="utf-8"))
                    exit_code = int(pdata.get("exit_code", exit_code))
                    for h in pdata.get("hotspots", [])[:10]:
                        hotspots.append(HotspotEntry(**h))
                    executed_map = {k: list(v) for k, v in pdata.get("executed_lines", {}).items()}
                except Exception:
                    pass

        exec_summary: dict[str, int] = {k: len(v) for k, v in executed_map.items()}
        missed_lines = self._compute_missed_lines(executed_map, focus_path=focus_path)
        hint = self._build_diagnosis_hint(hotspots, missed_lines, wall_ms)

        return ProfileReport(
            command=target_cmd,
            exit_code=exit_code,
            wall_time_ms=wall_ms,
            top_hotspots=hotspots,
            executed_lines_summary=exec_summary,
            missed_lines=missed_lines,
            bottleneck_diagnosis_hint=hint,
            stdout_tail=raw_out[-2000:],
        )

    @staticmethod
    def _parse_python_command(cmd: str) -> tuple[str | None, str, list[str]]:
        """Parse `pytest ...`, `python -m mod ...`, or `python script.py ...` for in-process profiling."""
        try:
            tokens = shlex.split(cmd.strip())
        except ValueError:
            return None, "", []
        if not tokens:
            return None, "", []

        head = Path(tokens[0]).name
        if head == "pytest":
            return "module", "pytest", tokens[1:]
        if head in ("python", "python3"):
            if len(tokens) >= 3 and tokens[1] == "-m":
                return "module", tokens[2], tokens[3:]
            if len(tokens) >= 2 and not tokens[1].startswith("-"):
                return "script", tokens[1], tokens[2:]
        return None, "", []

    def _compute_missed_lines(
        self,
        executed_map: dict[str, list[int]],
        focus_path: str | None = None,
    ) -> dict[str, list[str]]:
        """Compare AST executable statement lines against executed line numbers to identify missed branches."""
        candidates: list[str] = []
        if focus_path:
            candidates.append(focus_path)
        else:
            for rel in executed_map:
                if not rel.startswith("tests/") and not Path(rel).name.startswith("test_"):
                    candidates.append(rel)

        missed_report: dict[str, list[str]] = {}
        for rel in candidates[:8]:
            fpath = self.repo_root / rel
            if not fpath.exists() or fpath.suffix.lower() != ".py":
                continue
            try:
                tree = ast.parse(fpath.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                continue

            statement_lines: set[int] = set()
            for node in ast.walk(tree):
                if isinstance(
                    node,
                    (
                        ast.Return,
                        ast.Assign,
                        ast.AugAssign,
                        ast.AnnAssign,
                        ast.Expr,
                        ast.Raise,
                        ast.If,
                        ast.For,
                        ast.While,
                    ),
                ):
                    # Ignore module/class/function docstring expressions
                    if isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant) and isinstance(node.value.value, str):
                        continue
                    if hasattr(node, "lineno"):
                        statement_lines.add(node.lineno)

            ran_set = set(executed_map.get(rel, []))
            missed = sorted(statement_lines - ran_set)
            if missed:
                missed_report[rel] = self._compact_line_ranges(missed)
        return missed_report

    @staticmethod
    def _compact_line_ranges(lines: list[int]) -> list[str]:
        if not lines:
            return []
        ranges: list[str] = []
        start = prev = lines[0]
        for ln in lines[1:]:
            if ln == prev + 1:
                prev = ln
            else:
                ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
                start = prev = ln
        ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
        return ranges

    @staticmethod
    def _build_diagnosis_hint(
        hotspots: list[HotspotEntry],
        missed_lines: dict[str, list[str]],
        wall_ms: float,
    ) -> str:
        non_module = [h for h in hotspots if h.func_name != "<module>"]
        top = non_module[0] if non_module else (hotspots[0] if hotspots else None)
        parts: list[str] = []
        if top:
            parts.append(
                f"Primary CPU/time hotspot: `{top.func_name}` at [{top.rel_path}:{top.line}] "
                f"(cumulative={top.cumulative_time_ms}ms, self={top.total_time_ms}ms, calls={top.call_count})."
            )
        if missed_lines:
            missed_str = "; ".join(f"{k}: lines {', '.join(v)}" for k, v in missed_lines.items())
            parts.append(f"Uncovered/unexecuted branches: {missed_str}.")
        if not parts:
            parts.append(f"Execution completed in {wall_ms}ms with no dominant project-level bottleneck.")
        return " ".join(parts)

    @staticmethod
    def format_profile_report(report: ProfileReport) -> str:
        lines = [
            f"=== Runtime Performance & Coverage Profile ===",
            f"Command: `{report.command}` | Exit Code: {report.exit_code} | Wall Time: {report.wall_time_ms} ms",
            f"Diagnosis Hint: {report.bottleneck_diagnosis_hint}",
        ]
        if report.top_hotspots:
            lines.append("\n[Top Project Bottleneck Functions (sorted by cumulative_time_ms)]")
            for idx, h in enumerate(report.top_hotspots, start=1):
                lines.append(
                    f"  {idx}. [{h.rel_path}:{h.line}] `{h.func_name}` — "
                    f"cum={h.cumulative_time_ms}ms | self={h.total_time_ms}ms | "
                    f"calls={h.call_count} | per_call={h.per_call_ms}ms"
                )
        if report.executed_lines_summary:
            lines.append("\n[Executed Lines Count by File]")
            for rel, cnt in sorted(report.executed_lines_summary.items()):
                lines.append(f"  • {rel}: {cnt} lines executed")
        if report.missed_lines:
            lines.append("\n[Missed / Unexecuted Statement Lines]")
            for rel, rngs in sorted(report.missed_lines.items()):
                lines.append(f"  • [{rel}] missed lines: {', '.join(rngs)}")
        if report.stdout_tail.strip():
            lines.append(f"\n[Command Output Tail]\n{report.stdout_tail.strip()}")
        return "\n".join(lines)
