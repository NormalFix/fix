"""Zero-Token Repository Radar (`RepoProfile`), Auto-Root Detection & Sliding-Window Context Micro-Compressor."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from fix.schema import TaskIntent, TaskPhase, TaskSession
from fix.tools import IGNORE_DIRS

EXT_TO_LANG = {
    ".py": "Python",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".go": "Go",
    ".rs": "Rust",
    ".java": "Java",
    ".kt": "Kotlin",
    ".cpp": "C++",
    ".cc": "C++",
    ".hpp": "C++",
    ".c": "C",
    ".h": "C/C++",
    ".cs": "C#",
    ".swift": "Swift",
    ".rb": "Ruby",
    ".php": "PHP",
}


def resolve_project_root(start_path: str | Path | None = None) -> Path:
    """
    Automatically resolve the enclosing Git repository root from `start_path` (or `os.getcwd()`).
    Falls back to the resolved directory itself if not inside a Git repository.
    """
    base = Path(start_path or os.getcwd()).resolve()
    if base.is_file():
        base = base.parent
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=base,
            capture_output=True,
            text=True,
            timeout=3,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return Path(proc.stdout.strip()).resolve()
    except Exception:
        pass
    return base


class RepoProfile(BaseModel):
    """Deterministic 0-token repository snapshot generated in <50ms."""

    repo_root: str
    languages: list[str] = Field(default_factory=list)
    test_command: str | None = None
    lint_command: str | None = None
    top_level_layout: list[str] = Field(default_factory=list)
    git_branch: str = ""
    head_sha: str = ""


class ContextManager:
    """Builds high-signal prompts and applies sliding-window micro-compression to tool history."""

    def __init__(self, keep_recent_messages: int = 12, fold_threshold_chars: int = 1400) -> None:
        self.keep_recent_messages = keep_recent_messages
        self.fold_threshold_chars = fold_threshold_chars

    def detect_repo_profile(self, repo_root: str | Path) -> RepoProfile:
        """Pure deterministic 0-token scan of repository languages, test commands, and layout."""
        root = Path(repo_root).resolve()
        lang_counts: dict[str, int] = {}
        top_layout: list[str] = []

        if root.exists():
            for entry in sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                if entry.name in IGNORE_DIRS or entry.name.startswith("."):
                    continue
                top_layout.append(f"{entry.name}/" if entry.is_dir() else entry.name)

            extra_skip = IGNORE_DIRS | {"build", "captures", "assets", "res", "libs", "out", "target"}
            files_scanned = 0
            for dirpath, dirs, files in os.walk(root):
                dirs[:] = [d for d in dirs if d not in extra_skip and not d.startswith(".")]
                for fname in files:
                    files_scanned += 1
                    if files_scanned > 3000:
                        break
                if files_scanned > 3000:
                    break
                for fname in files:
                    ext = Path(fname).suffix.lower()
                    if ext in EXT_TO_LANG:
                        lang = EXT_TO_LANG[ext]
                        lang_counts[lang] = lang_counts.get(lang, 0) + 1

        languages = [k for k, _ in sorted(lang_counts.items(), key=lambda kv: kv[1], reverse=True)]

        test_cmd: str | None = None
        lint_cmd: str | None = None

        has_py_tests = (
            (root / "pytest.ini").exists()
            or (root / "tests").exists()
            or any(root.glob("test_*.py"))
            or any(root.glob("*_test.py"))
            or (
                (root / "pyproject.toml").exists()
                and "pytest" in (root / "pyproject.toml").read_text(encoding="utf-8", errors="ignore")
            )
        )
        if has_py_tests:
            test_cmd = "pytest -q"
            lint_cmd = "python3 -m compileall -q ."
        elif "Python" in languages:
            test_cmd = None
            lint_cmd = "python3 -m compileall -q ." 
        elif (root / "package.json").exists():
            try:
                pkg = json.loads((root / "package.json").read_text(encoding="utf-8"))
                scripts = pkg.get("scripts", {})
                if "test" in scripts:
                    test_cmd = "npm test"
                if "lint" in scripts:
                    lint_cmd = "npm run lint"
            except Exception:
                test_cmd = "npm test"
        elif (root / "Cargo.toml").exists():
            test_cmd = "cargo test"
            lint_cmd = "cargo check"
        elif (root / "go.mod").exists():
            test_cmd = "go test ./..."
            lint_cmd = "go vet ./..."
        elif (root / "CMakeLists.txt").exists():
            test_cmd = "ctest --output-on-failure"
            lint_cmd = None
        elif (root / "Makefile").exists():
            test_cmd = "make test"
            lint_cmd = None
        elif (root / "pom.xml").exists():
            test_cmd = "mvn test"
            lint_cmd = "mvn checkstyle:check"
        elif (root / "gradlew").exists() or (root / "build.gradle.kts").exists() or (root / "build.gradle").exists():
            test_cmd = "./gradlew test" if (root / "gradlew").exists() else "gradle test"
            lint_cmd = "./gradlew check" if (root / "gradlew").exists() else None

        branch_proc = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
        )
        head_proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
        )

        return RepoProfile(
            repo_root=str(root),
            languages=languages or ["Unknown"],
            test_command=test_cmd,
            lint_command=lint_cmd,
            top_level_layout=top_layout[:30],
            git_branch=branch_proc.stdout.strip() if branch_proc.returncode == 0 else "",
            head_sha=head_proc.stdout.strip() if head_proc.returncode == 0 else "",
        )

    def build_messages(
        self,
        session: TaskSession,
        profile: RepoProfile,
        history: list[dict[str, Any]],
        persona_rules: list[str] | None = None,
        repo_landmarks: list[str] | None = None,
        active_skills: list[str] | None = None,
        conversation_context: list[dict[str, str]] | None = None,
        mental_models: list[str] | None = None,
        cognitive_paradigms: list[str] | None = None,
        optimization_proposals: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Assemble system instructions, mind-mirror rules, repo landmarks, prior session turns, and history."""
        sys_sections: list[str] = [
            "You are Fix, an elite autonomous software engineering partner with a persistent Mind Mirror.",
            f"Current Task ID: {session.task_id}",
            f"Active Intent Track: {session.intent.value if session.intent else 'UNDECIDED'}",
            f"Current Phase: {session.phase.value.upper()}",
            "",
            "## Repository Profile (0-Token Radar)",
            f"- Root: {profile.repo_root}",
            f"- Languages: {', '.join(profile.languages)}",
            f"- Default Test Command: {profile.test_command or 'N/A'}",
            f"- Top-Level Layout: {', '.join(profile.top_level_layout) or '(empty)'}",
        ]

        if persona_rules:
            sys_sections.append("\n## Developer Mind Mirror (Personal Taste & Hard Rules)")
            sys_sections.extend(f"- {rule}" for rule in persona_rules)

        if repo_landmarks:
            sys_sections.append("\n## Verified Repository Landmarks (Prior Evidence)")
            sys_sections.extend(f"- {lm}" for lm in repo_landmarks)

        if active_skills:
            sys_sections.append("\n## Bootstrapped Skills Available in `.fix/skills/`")
            sys_sections.extend(f"- {sk}" for sk in active_skills)

        if cognitive_paradigms:
            sys_sections.append("\n## Self-Evolved Cognitive Paradigms (3-Layer First-Principles Lens)")
            sys_sections.extend(f"- {cp}" for cp in cognitive_paradigms)

        if mental_models:
            sys_sections.append("\n## Distilled 3-Layer Mental Models (现象 -> 机制契约 -> 本质与UX)")
            sys_sections.extend(f"- {mm}" for mm in mental_models)

        if optimization_proposals:
            sys_sections.append("\n## Proactive Project & UX Optimization Radar")
            sys_sections.extend(f"- {op}" for op in optimization_proposals)

        if conversation_context:
            sys_sections.append("\n## Recent Session Conversation Context (Prior Turns in This CLI Session)")
            for turn in conversation_context[-6:]:
                u_msg = turn.get("user", "").strip()
                a_msg = turn.get("assistant", "").strip()
                files_str = turn.get("modified_files", "")
                sys_sections.append(f"- User: {u_msg[:300]}")
                if files_str:
                    sys_sections.append(f"  Modified Files: {files_str}")
                sys_sections.append(f"  Fix Result: {a_msg[:500]}")

        if session.steering_notes:
            sys_sections.append("\n## High-Priority Live User Steering Notes (Follow Strictly)")
            sys_sections.extend(f"- {note}" for note in session.steering_notes)

        sys_sections.append("\n## Phase Execution & Narrative Discipline (Codex Style)")
        sys_sections.append(
            "- PROGRESSIVE NARRATIVE BEFORE TOOL CALLS: Whenever you invoke one or more tools (`search_code`, "
            "`read_file`, `run_command`, `trace_symbol`, `apply_patch`, etc.), ALWAYS write 1-2 concise sentences "
            "of natural-language explanation in `content` alongside your `tool_calls`! "
            "On the first step, briefly state your exploration plan (e.g., '我会先从项目里定位 X 的实现、引用关系和相关资源，再沿调用链整理它的生命周期与功能/逻辑地图。'). "
            "On subsequent steps, briefly state what you just confirmed from the previous tool outputs and what you are checking next "
            "(e.g., '已确认 X 的核心职责与事件流向；接下来我会继续追踪它与 Y 的交互细节。'). "
            "Do NOT use Markdown headings (`#`) in these short intermediate progress notes. "
            "CRITICAL LINE RANGE FORMAT: Whenever you mention line numbers or ranges (in narratives or reports), "
            "ALWAYS use half-width ASCII parentheses `(...)`, half-width colon `:`, and standard ASCII hyphen-minus `-` "
            "with NO spaces inside the parentheses — e.g. `(98-168)`, `(:357-374)`, `(OrderService.py:505-518)`. "
            "NEVER use full-width `（...）`, `：`, en-dash `–`, `~`, or `_` in line ranges."
        )
        if session.intent == TaskIntent.EXPLAIN:
            sys_sections.append(
                "- READ-ONLY EXPLAIN TRACK: Write tools are physically disabled. "
                "Every architectural or root-cause claim MUST cite `(path/to/file:line)` or `[file:line]`.\n"
                "- FIRST-PRINCIPLES ESSENCE (`透过现象看本质` & UX OPTIMIZATION): Do not stop at surface code descriptions (`现象层`). Always penetrate to the Mechanism & State Contract (`机制与契约层`), the First-Principles Domain Invariant (`本质层`: why it is designed this way, where dual-source-of-truth or lifecycle asymmetry exists), and concrete Project & User Experience Optimizations (`完善优化项目与用户体验`).\n"
                "- EFFICIENCY DISCIPLINE: `read_file` reads up to 900 lines (36KB) at once by default without middle truncation. Batch independent `search_code` and `read_file` calls in parallel in a single turn. "
                "NEVER repeat identical `search_code` queries or re-read overlapping line ranges of the same file. "
                "Once you have inspected the core class/files and its direct callers/collaborators, "
                "STOP calling tools and output the complete, structured Markdown report starting with `# <Title>` (e.g. `# 核心模块深度分析`), "
                "including `## 结论`, `## 功能地图`, tables, ASCII/box flowcharts (`┌───┐`), and key code logic points."
            )
            if session.budget.steps_used >= 8:
                sys_sections.append(
                    f"- [BUDGET WARNING]: Step {session.budget.steps_used}/{session.budget.max_steps} "
                    f"({session.tool_calls_count} tool calls executed). Do NOT over-search; "
                    "finish any last critical file read now and output the final comprehensive analysis."
                )
        elif session.phase == TaskPhase.REPRODUCE:
            sys_sections.append(
                "- SCIENTIFIC REPRODUCTION PHASE: Production code edits are blocked. "
                "First run or write a failing test (`test_*.py` / `tests/**`) to prove the bug exists."
            )
        elif session.phase == TaskPhase.IMPLEMENT:
            sys_sections.append(
                "- IMPLEMENTATION PHASE: Apply minimal, root-cause-targeted patches using `apply_patch`. "
                "When finished editing, stop calling write tools so mechanical verification can run."
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "\n".join(sys_sections)},
            {"role": "user", "content": session.user_prompt},
        ]

        compressed_history = self._compress_history(history)
        messages.extend(compressed_history)
        return messages

    def _compress_history(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Fold older large tool outputs while preserving the most recent N turns verbatim."""
        total = len(history)
        if total <= self.keep_recent_messages:
            return [dict(m) for m in history]

        cutoff = total - self.keep_recent_messages
        result: list[dict[str, Any]] = []
        for idx, msg in enumerate(history):
            copy_msg = dict(msg)
            content = str(copy_msg.get("content") or "")
            if idx < cutoff and copy_msg.get("role") in ("tool", "user") and len(content) > self.fold_threshold_chars:
                lines = content.splitlines()
                if len(lines) > 8:
                    head = "\n".join(lines[:3])
                    tail = "\n".join(lines[-3:])
                    copy_msg["content"] = (
                        f"{head}\n"
                        f"... [folded {len(lines) - 6} lines / {len(content)} chars from earlier step] ...\n"
                        f"{tail}"
                    )
            result.append(copy_msg)
        return result
