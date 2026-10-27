"""Event-Driven Bounded Evolution Engine: Git Co-Change Radar, Semantic Equivalence Diff Filter & Skill Crystallizer."""

from __future__ import annotations

import ast

import re
import stat
import subprocess
from collections import Counter
from itertools import combinations
from pathlib import Path

from fix.llm import LLMGateway
from fix.memory import DualMemoryStore
from fix.schema import (
    CoChangeEdge,
    CognitiveParadigm,
    DiffAnalysisResult,
    EvidenceFact,
    FactStatus,
    MemoryDelta,
    MentalModel,
    MetaReflection,
    OptimizationProposal,
    RuleStatus,
    SkillCard,
    StyleRule,
    TaskIntent,
    TaskSession,
)


class EvolutionEngine:
    """Handles post-task digest, 0-token git co-change extraction, human diff learning & skill crystallization."""

    def __init__(
        self,
        repo_root: str | Path,
        memory: DualMemoryStore,
        llm: LLMGateway | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.memory = memory
        self.llm = llm

    # ------------------------------------------------------------------
    # 1. Zero-Token Git Co-Change Coupling Scanner
    # ------------------------------------------------------------------

    def _compute_cochange_edges_in_memory(
        self,
        max_commits: int = 200,
        min_co_commits: int = 2,
    ) -> list[CoChangeEdge]:
        """Pure in-memory calculation of Git co-change edges without writing to SQLite."""
        proc = subprocess.run(
            ["git", "log", f"-n{max_commits}", "--name-only", "--pretty=format:__COMMIT__"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return []

        file_commits: Counter[str] = Counter()
        pair_commits: Counter[tuple[str, str]] = Counter()

        raw_blocks = proc.stdout.split("__COMMIT__")
        for block in raw_blocks:
            files = sorted(
                {
                    line.strip()
                    for line in block.splitlines()
                    if line.strip() and not line.strip().startswith(".fix/")
                }
            )
            if not files or len(files) > 25:
                continue
            for f in files:
                file_commits[f] += 1
            for a, b in combinations(files, 2):
                pair_commits[(a, b)] += 1

        edges: list[CoChangeEdge] = []
        for (a, b), count in pair_commits.most_common(100):
            if count < min_co_commits:
                continue
            denom = max(file_commits[a], file_commits[b], 1)
            conf = round(count / denom, 3)
            edges.append(CoChangeEdge(file_a=a, file_b=b, co_commits=count, confidence=conf))
        return edges

    def scan_git_cochanges(
        self,
        max_commits: int = 200,
        min_co_commits: int = 2,
    ) -> list[CoChangeEdge]:
        """Deterministically mine git history for files that frequently change together (0 Tokens)."""
        edges = self._compute_cochange_edges_in_memory(max_commits=max_commits, min_co_commits=min_co_commits)
        delta = MemoryDelta(co_changes_to_upsert=edges)
        self.memory.commit_delta(delta)
        return edges

    # ------------------------------------------------------------------
    # 2. Semantic Equivalence Filter for Human Diff Learning
    # ------------------------------------------------------------------

    async def analyze_human_diff(
        self,
        human_diff: str,
        fix_diff: str = "",
        commit_message: str = "",
    ) -> DiffAnalysisResult:
        """
        Extract developer StyleRules ONLY when the human diff represents a semantically equivalent
        refactoring/style improvement, filtering out business logic/feature additions.
        """
        if not human_diff.strip():
            return DiffAnalysisResult(
                is_semantically_equivalent_refactor=False,
                change_classification="business_feature",
                extracted_rules=[],
                rationale="Empty diff.",
            )

        # If LLM has a structured handler or live API configured, try LLM-backed semantic equivalence analysis
        if self.llm and self.llm.is_configured:
            try:
                prompt = (
                    "Analyze the following developer git diff (made after or during coding). "
                    "Determine whether this change is a SEMANTICALLY EQUIVALENT refactoring/style/architecture "
                    "preference (same business input/output, better idiom/typing/error-handling/layering) "
                    "OR a business feature/requirement change.\n"
                    "CRITICAL RULE: Only extract `StyleRule` items if `is_semantically_equivalent_refactor` is true.\n\n"
                    f"Commit Message: {commit_message}\n"
                    f"Prior Fix Diff:\n{fix_diff[:2000]}\n\n"
                    f"Human Diff:\n{human_diff[:4000]}"
                )
                result = await self.llm.complete_structured(
                    messages=[{"role": "user", "content": prompt}],
                    response_model=DiffAnalysisResult,
                    use_reasoning=True,
                )
                if not result.is_semantically_equivalent_refactor:
                    result.extracted_rules = []
                return result
            except Exception:
                # Fall back to deterministic AST/pattern semantic equivalence filter
                pass

        return self._deterministic_semantic_diff_filter(human_diff, commit_message)

    def _deterministic_semantic_diff_filter(
        self,
        human_diff: str,
        commit_message: str = "",
    ) -> DiffAnalysisResult:
        """Deterministic heuristic filter distinguishing style/idiom refactors from business logic additions."""
        removed_lines: list[str] = []
        added_lines: list[str] = []
        for line in human_diff.splitlines():
            if line.startswith("---") or line.startswith("+++"):
                continue
            if line.startswith("-"):
                removed_lines.append(line[1:].strip())
            elif line.startswith("+"):
                added_lines.append(line[1:].strip())

        removed_blob = "\n".join(removed_lines)
        added_blob = "\n".join(added_lines)

        rules: list[StyleRule] = []

        # Pattern 1: os.path -> pathlib.Path
        if ("os.path" in removed_blob and "Path(" in added_blob) or (
            "os.path.join" in removed_blob and "pathlib" in added_blob
        ):
            rules.append(
                StyleRule(
                    pattern_desc="manipulating filesystem paths",
                    preferred_way="Use `pathlib.Path` instead of `os.path` string manipulation",
                    category="architecture",
                    source_example="os.path -> pathlib.Path",
                )
            )

        # Pattern 2: bare except Exception -> specific custom exception / logging
        if re.search(r"except\s+Exception\s*:", removed_blob) and not re.search(
            r"except\s+Exception\s*:", added_blob
        ):
            rules.append(
                StyleRule(
                    pattern_desc="catching exceptions in domain/service logic",
                    preferred_way="Catch specific typed exceptions instead of bare `except Exception:`",
                    category="error_handling",
                    source_example="Replaced broad `except Exception:` with specific exception class",
                )
            )

        # Pattern 3: print(...) -> logger / logging
        if "print(" in removed_blob and ("logger." in added_blob or "logging." in added_blob):
            rules.append(
                StyleRule(
                    pattern_desc="emitting runtime diagnostics or status output",
                    preferred_way="Use structured `logger` calls instead of `print()`",
                    category="general",
                    source_example="print(...) -> logger.info/debug(...)",
                )
            )

        # Pattern 4: raw dict -> Pydantic BaseModel / dataclass
        if ("dict[" in removed_blob or "{}" in removed_blob) and (
            "BaseModel" in added_blob or "@dataclass" in added_blob
        ):
            rules.append(
                StyleRule(
                    pattern_desc="passing structured data contracts across functions",
                    preferred_way="Use typed `Pydantic BaseModel` or `@dataclass` instead of raw `dict`",
                    category="architecture",
                    source_example="dict -> BaseModel/dataclass",
                )
            )

        if rules:
            return DiffAnalysisResult(
                is_semantically_equivalent_refactor=True,
                change_classification="style_refactor",
                extracted_rules=rules,
                rationale="Detected semantically equivalent idiom/architecture refactoring.",
            )

        return DiffAnalysisResult(
            is_semantically_equivalent_refactor=False,
            change_classification="business_feature",
            extracted_rules=[],
            rationale="Diff modifies business logic or adds new functionality; skipped style extraction.",
        )

    async def learn_from_human_diff(
        self,
        human_diff: str,
        fix_diff: str = "",
        commit_message: str = "",
    ) -> list[StyleRule]:
        """Run semantic equivalence filter and atomically commit any extracted StyleRules."""
        analysis = await self.analyze_human_diff(
            human_diff=human_diff,
            fix_diff=fix_diff,
            commit_message=commit_message,
        )
        if not analysis.is_semantically_equivalent_refactor or not analysis.extracted_rules:
            return []

        persisted: list[StyleRule] = []
        for rule in analysis.extracted_rules:
            updated = self.memory.upsert_style_rule(rule)
            persisted.append(updated)
        return persisted

    # ------------------------------------------------------------------
    # 3. Post-Task Digest & Skill Crystallization
    # ------------------------------------------------------------------

    def synthesize_architectural_essence(
        self,
        rel_path: str,
        head_sha: str = "",
        task_summary: str = "",
    ) -> tuple[MentalModel, list[OptimizationProposal]]:
        """
        First-Principles Essence Synthesizer (`透过现象看本质`):
        Analyzes a module/component across 3 cognitive layers:
        1. Phenomenon Layer (`现象层`): Surface classes, entry points, line span, and UI/API responsibilities.
        2. Mechanism & Contract Layer (`机制与契约层`): Lifecycle hooks, state machines, timers/async callbacks, and cross-module delegation.
        3. Essence & First-Principles Layer (`本质与第一性原理层`): Core domain invariant, single-source-of-truth vs dual-state tensions, and root architectural trade-offs.
        4. UX & Product Impact (`用户体验与产品感知影响`): Latency, animation smoothness, edge-case feedback, and user ergonomics.
        Also generates proactive `OptimizationProposal` items across `architecture`, `user_experience`, `reliability`, and `performance`.
        """
        fpath = self.repo_root / rel_path
        code_text = ""
        line_count = 0
        if fpath.is_file():
            try:
                code_text = fpath.read_text(encoding="utf-8", errors="ignore")
                line_count = len(code_text.splitlines())
            except OSError:
                pass

        comp_name = Path(rel_path).stem or rel_path
        classes = re.findall(r"\b(?:class|interface|object|struct|enum)\s+([A-Za-z0-9_]+)", code_text)
        funcs = re.findall(r"\b(?:fun|def|func|fn|async\s+def|async\s+function)\s+([A-Za-z0-9_]+)", code_text)
        acquires = re.findall(
            r"\b(Handler|ValueAnimator|ObjectAnimator|AnimationDrawable|postDelayed|create_task|setInterval|subscribe|register[A-Za-z0-9_]*)",
            code_text,
        )
        releases = re.findall(
            r"\b(onDetachedFromWindow|onDestroy|removeCallbacksAndMessages|cancel|unRegister[A-Za-z0-9_]*|unsubscribe|clearInterval|close)",
            code_text,
        )

        # Layer 1: Phenomenon
        cls_str = ", ".join(dict.fromkeys(classes[:5])) or comp_name
        fn_str = ", ".join(dict.fromkeys(funcs[:8])) or "core entry points"
        phenomenon = (
            f"[{rel_path} ({line_count} lines)] Defines `{cls_str}` with key entry points ({fn_str})."
        )
        if task_summary:
            first_para = next(
                (ln.strip(" >-*#") for ln in task_summary.splitlines() if len(ln.strip(" >-*#")) > 20),
                "",
            )
            if first_para:
                phenomenon += f" Surface role: {first_para[:180]}"

        # Layer 2: Mechanism & State Contract
        acq_set = sorted(set(acquires))
        rel_set = sorted(set(releases))
        if acq_set or rel_set:
            mechanism = (
                f"Stateful lifecycle & async/animation orchestration via `{', '.join(acq_set[:4]) or 'state fields'}` "
                f"paired with cleanup/release hooks `{', '.join(rel_set[:4]) or 'none detected'}`."
            )
        else:
            mechanism = (
                f"Module contract coordinating `{cls_str}` state transitions and callers across `{rel_path}`."
            )

        # Layer 3: Essence & First Principles
        essence_points: list[str] = []
        if acq_set and not rel_set:
            essence_points.append(
                "Asymmetric resource/timer lifecycle invariant: stateful handles are acquired without a symmetric teardown gate."
            )
        elif acq_set and rel_set:
            essence_points.append(
                "Core invariant is strict state-machine & lifecycle symmetry between active execution/animation phases and teardown."
            )
        if line_count > 450:
            essence_points.append(
                f"High cognitive density ({line_count} lines): combines state arbitration, layout/data transformation, and I/O/rendering in one unit, creating dual-source-of-truth tension."
            )
        if not essence_points:
            essence_points.append(
                f"Single-source-of-truth domain contract for `{comp_name}`: callers must preserve input/state invariants across transitions."
            )
        essence = " ".join(essence_points)

        # UX & Product Impact
        if any(k in code_text.lower() for k in ("anim", "draw", "view", "toast", "click", "tts", "ui", "prompt", "console")):
            ux_impact = (
                "Directly governs user-perceived responsiveness, visual transition continuity, and deterministic feedback; "
                "any state desync or silent callback drop manifests as UI flicker, frozen state, or unresponsive interaction."
            )
        else:
            ux_impact = (
                "Impacts end-to-end task reliability, error recovery clarity, and execution latency experienced by the user."
            )

        model = MentalModel(
            component_or_scope=comp_name,
            phenomenon_layer=phenomenon[:400],
            mechanism_contract_layer=mechanism[:400],
            essence_first_principles=essence[:450],
            ux_and_product_impact=ux_impact[:350],
            evidence_paths=[rel_path],
            commit_sha=head_sha,
            confidence=0.93,
        )

        # Generate proactive OptimizationProposals (Architecture, UX, Reliability, Performance)
        proposals: list[OptimizationProposal] = []
        if line_count > 450:
            proposals.append(
                OptimizationProposal(
                    title=f"Decouple State Arbitration & Rendering/IO Concerns in `{comp_name}`",
                    dimension="architecture",
                    target_files=[rel_path],
                    phenomenon_symptom=f"`{rel_path}` has grown to {line_count} lines mixing state arbitration, orchestration, and execution.",
                    root_essence="Violation of Single Source of Truth & Separation of Concerns increases state coupling and regression risk when extending variants.",
                    proposed_evolution=f"Extract pure state-arbitration policy / controller from `{comp_name}` so rendering/IO delegates consume an immutable state contract.",
                    ux_or_engineering_gain="Eliminates cross-variant state drift and makes state transitions 100% unit-testable in isolation.",
                    priority="P1",
                )
            )
        if acq_set and not rel_set:
            proposals.append(
                OptimizationProposal(
                    title=f"Enforce Symmetric Lifecycle Teardown for `{comp_name}` Async/Timer Handles",
                    dimension="reliability",
                    target_files=[rel_path],
                    phenomenon_symptom=f"`{rel_path}` acquires `{', '.join(acq_set[:3])}` without an explicit deterministic release hook.",
                    root_essence="Unbounded timer/callback lifetime outliving component scope causes memory leaks or stale callbacks mutating dead state.",
                    proposed_evolution="Add a unified idempotent `dispose()/stopAll()` teardown hook invoked on detach/cancel.",
                    ux_or_engineering_gain="Prevents ghost animations, stale state mutations, and background resource drain.",
                    priority="P0",
                )
            )

        # Also harvest any explicit risk / hazard items from task_summary (e.g. `## 设计要点与潜在隐患` / `关注点`)
        if task_summary:
            in_risk_section = False
            for ln in task_summary.splitlines():
                stripped = ln.strip()
                if re.match(r"^#{1,4}\s+.*(隐患|风险|关注点|优化|Risk|Bottleneck|Issue)", stripped, re.I):
                    in_risk_section = True
                    continue
                if in_risk_section and stripped.startswith("#"):
                    in_risk_section = False
                if in_risk_section:
                    m_item = re.match(r"^(?:\d+\.|[-*])\s*\*\*([^*]+)\*\*[：:]?\s*(.+)", stripped)
                    if m_item:
                        r_title = m_item.group(1).strip()
                        r_desc = m_item.group(2).strip()
                        dim = (
                            "user_experience"
                            if any(w in (r_title + r_desc).lower() for w in ("动画", "显示", "淡出", "错位", "闪烁", "点击", "交互", "ui", "ux", "延迟"))
                            else "architecture"
                            if any(w in (r_title + r_desc).lower() for w in ("耦合", "分叉", "继承", "模块", "不对称", "硬编码"))
                            else "reliability"
                        )
                        proposals.append(
                            OptimizationProposal(
                                title=f"[{comp_name}] {r_title[:70]}",
                                dimension=dim,  # type: ignore[arg-type]
                                target_files=[rel_path],
                                phenomenon_symptom=r_desc[:240],
                                root_essence=f"Inconsistency between declared contract and runtime state transition in `{comp_name}`.",
                                proposed_evolution=f"Align `{r_title}` implementation and state guard in `{rel_path}` with a single verified invariant.",
                                ux_or_engineering_gain="Improves user-perceived interaction consistency and eliminates latent edge-case bugs.",
                                priority="P1",
                            )
                        )
                        if len(proposals) >= 5:
                            break

        return model, proposals

    def reflect_and_evolve_paradigms(
        self,
        session: TaskSession,
        mental_models_count: int,
        proposals_count: int,
    ) -> tuple[MetaReflection, CognitiveParadigm, list[StyleRule]]:
        """
        Meta-Cognitive Self-Reflection (`元认知自我进化`):
        1. Audits tool call efficiency (detecting fragmented reads of the same file).
        2. Evolves/reinforces a higher-order `CognitiveParadigm` in `persona.db`.
        3. Extracts user interaction/analytical preferences (`StyleRule`) from the user's prompt.
        """
        read_files: list[str] = []
        for entry in session.tool_calls_log:
            m = re.match(r"^Read\(([^:)]+)", entry)
            if m:
                read_files.append(m.group(1).strip())
        counts = Counter(read_files)
        redundant_reads = sum(c - 1 for c in counts.values() if c > 1)
        total_calls = max(1, session.tool_calls_count)
        eff_score = round(max(0.2, 1.0 - (redundant_reads / (total_calls + 2))), 2)

        if redundant_reads > 0:
            top_rep, rep_cnt = counts.most_common(1)[0]
            eff_lesson = (
                f"Detected {redundant_reads} fragmented re-reads (e.g. `{Path(top_rep).name}` read {rep_cnt}x). "
                "Upgrade strategy: read up to 1000 lines in a single `read_file(start_line=1, end_line=1000)` call "
                "and batch independent collaborator reads in parallel."
            )
        else:
            eff_lesson = (
                "High-efficiency exploration: batched parallel `search_code`/`read_file` calls with zero redundant re-reads."
            )

        # Determine domain paradigm to evolve
        prompt_low = session.user_prompt.lower()
        if any(k in prompt_low for k in ("view", "ui", "widget", "界面", "气泡", "动画", "交互")):
            paradigm = CognitiveParadigm(
                domain_trigger="analyzing or optimizing UI/View components and interactive state machines",
                first_principles_checklist=[
                    "Layer 1 (现象层): Map visible lifecycle (`onAttached` -> `onDraw` -> `onDetached`) & host wiring (`[file:line]`).",
                    "Layer 2 (机制与契约层): Dissect multi-channel priority arbitration (`realtime > normal > character`) & delegation callbacks.",
                    "Layer 3 (本质层): Audit Single-Source-of-Truth vs duplicated subclass state and asset geometry invariants.",
                    "Layer 4 (UX层): Evaluate animation continuity, cancel/exit responsiveness, and silent-failure fallbacks.",
                ],
                exploration_efficiency_lesson=eff_lesson,
                ux_lens="Ensure smooth entry/loop/exit transitions, crisp typography, and zero dead/unresponsive states.",
            )
        elif session.intent == TaskIntent.BUG_FIX:
            paradigm = CognitiveParadigm(
                domain_trigger="diagnosing and fixing functional or concurrency bugs",
                first_principles_checklist=[
                    "Layer 1 (现象层): Anchor the exact failing symptom and reproduction trace.",
                    "Layer 2 (机制层): Trace state mutation across callers, callbacks, and error boundaries.",
                    "Layer 3 (本质层): Identify the broken domain invariant (stale state, race condition, asymmetric cleanup) rather than patching symptoms.",
                    "Layer 4 (UX/系统影响): Verify edge-case recovery and zero regression across coupled co-change files.",
                ],
                exploration_efficiency_lesson=eff_lesson,
                ux_lens="Fail gracefully with clear user feedback and deterministic state recovery.",
            )
        else:
            paradigm = CognitiveParadigm(
                domain_trigger="architectural analysis, module topology & project optimization",
                first_principles_checklist=[
                    "Layer 1 (现象层): Map module boundaries, entry points, and build/DI topology (`[file:line]`).",
                    "Layer 2 (机制与契约层): Trace cross-module/IPC contracts, startup orchestration (`CoreStartable`), and flavor/device branching.",
                    "Layer 3 (本质层): Expose architectural trade-offs, hidden API coupling pressure, and monolithic hotspots.",
                    "Layer 4 (演进与UX层): Propose actionable decoupling, reliability, and user-experience optimizations.",
                ],
                exploration_efficiency_lesson=eff_lesson,
                ux_lens="Optimize developer ergonomics and end-user runtime responsiveness.",
            )

        learned_prompt_rules: list[StyleRule] = []
        if any(k in session.user_prompt for k in ("功能地图", "逻辑地图", "工作机制", "业务逻辑")):
            learned_prompt_rules.append(
                StyleRule(
                    pattern_desc="conducting deep component or project architectural analysis",
                    preferred_way=(
                        "Structure report with `# Title`, `## 结论` (TL;DR essence), `## 工作机制` (lifecycle & delegation), "
                        "`## 功能地图` (table), `## 业务逻辑地图` & `## 代码逻辑地图` (ASCII flowcharts + `[file:line]`), and `## 潜在隐患与优化建议`"
                    ),
                    category="architecture",
                    source_example=session.user_prompt[:120],
                )
            )
        if any(k in session.user_prompt for k in ("透过现象看本质", "用户体验", "自我进化", "更高层级")):
            learned_prompt_rules.append(
                StyleRule(
                    pattern_desc="analyzing code, diagnosing problems, or proposing improvements",
                    preferred_way=(
                        "Penetrate from surface phenomena (`现象层`) -> mechanism/contract (`机制契约层`) -> "
                        "first-principles essence (`本质层`), and proactively surface project & UX optimizations"
                    ),
                    category="architecture",
                    source_example=session.user_prompt[:120],
                )
            )

        reflection = MetaReflection(
            task_id=session.task_id,
            efficiency_score=eff_score,
            redundant_reads_detected=redundant_reads,
            efficiency_lesson=eff_lesson,
            paradigm_upgraded=paradigm,
            mental_models_distilled=mental_models_count,
            optimization_proposals_generated=proposals_count,
        )
        return reflection, paradigm, learned_prompt_rules

    def digest_completed_task(self, session: TaskSession, head_sha: str = "") -> MemoryDelta:
        """
        Build and commit an atomic Copy-on-Write `MemoryDelta` when a task completes:
        1. Multi-granularity verified `EvidenceFact`s (architecture, root_cause, invariant, gotcha).
        2. Three-Layer First-Principles `MentalModel` (`现象 -> 机制契约 -> 本质与UX影响`).
        3. Proactive Project & UX `OptimizationProposal`s.
        4. Meta-Cognitive Self-Reflection (`MetaReflection` + `CognitiveParadigm` + `StyleRule`).
        5. Reusable `SkillCard` crystallization when warranted.
        """
        delta = MemoryDelta()
        ev_paths = sorted(session.modified_files) or self._extract_inspected_paths(session)

        if session.final_summary or session.root_cause_hypothesis:
            summary_text = session.root_cause_hypothesis or session.final_summary
            category = "root_cause" if session.intent == TaskIntent.BUG_FIX else "architecture"
            status = (
                FactStatus.HYPOTHESIS
                if session.degraded_without_repro
                else FactStatus.VERIFIED
            )
            confidence = 0.6 if session.degraded_without_repro else 0.92
            clean_first = re.sub(r"\s+", " ", summary_text).strip()
            fact = EvidenceFact(
                summary=f"[{session.intent.value if session.intent else 'task'}] {clean_first[:360]}",
                category=category,  # type: ignore[arg-type]
                evidence_paths=ev_paths[:8],
                commit_sha=head_sha,
                confidence=confidence,
                status=status,
            )
            delta.facts_to_upsert.append(fact)

        # Distill 3-Layer First-Principles MentalModel & Proactive OptimizationProposals
        primary_path = ev_paths[0] if ev_paths else ""
        if primary_path or session.final_summary:
            mm, proposals = self.synthesize_architectural_essence(
                rel_path=primary_path or "repository_root",
                head_sha=head_sha,
                task_summary=session.final_summary or session.root_cause_hypothesis,
            )
            if ev_paths:
                mm.evidence_paths = ev_paths[:8]
            delta.mental_models_to_upsert.append(mm)
            delta.optimization_proposals_to_upsert.extend(proposals[:5])

        # Meta-Cognitive Self-Reflection & Paradigm Evolution
        reflection, paradigm, prompt_rules = self.reflect_and_evolve_paradigms(
            session=session,
            mental_models_count=len(delta.mental_models_to_upsert),
            proposals_count=len(delta.optimization_proposals_to_upsert),
        )
        session.meta_reflection = reflection
        delta.paradigms_to_upsert.append(paradigm)
        delta.rules_to_upsert.extend(prompt_rules)

        # Crystallize a reusable SkillCard when a bug fix or optimization succeeds with verified reproduction
        if (
            session.intent in (TaskIntent.BUG_FIX, TaskIntent.FEATURE)
            and session.modified_files
            and not session.degraded_without_repro
            and (session.reproduction_confirmed or session.budget.repairs_used >= 1)
        ):
            slug = self._make_slug(session.user_prompt)
            skill = SkillCard(
                slug=slug,
                title=f"Playbook: {session.user_prompt[:60]}",
                trigger_when=f"Issues or tasks involving {', '.join(sorted(session.modified_files))}",
                steps=[
                    f"Inspect target files: {', '.join(sorted(session.modified_files))}",
                    f"Root cause / invariant: {session.root_cause_hypothesis or session.final_summary}",
                    f"Run verification command: `{session.reproduction_cmd or 'pytest -q'}`",
                ],
                verification_cmd=session.reproduction_cmd or "pytest -q",
            )
            delta.skills_to_save.append(skill)

        self.memory.commit_delta(delta)
        self.memory.record_task_session(session)
        return delta

    @staticmethod
    def _make_slug(text: str) -> str:
        words = re.findall(r"[a-z0-9]+", text.lower())
        slug = "-".join(words[:5]) if words else "custom-skill"
        return slug[:48]

    def install_git_post_commit_hook(self) -> Path | None:
        """Install a lightweight git post-commit hook that triggers `fix learn-diff --quiet`."""
        hooks_dir = self.repo_root / ".git" / "hooks"
        if not hooks_dir.exists():
            return None
        hook_path = hooks_dir / "post-commit"
        hook_script = (
            "#!/bin/sh\n"
            "# Auto-generated by Fix Agent: learn developer habits from post-commit diffs\n"
            "if command -v fix >/dev/null 2>&1; then\n"
            "  fix learn-diff --quiet >/dev/null 2>&1 &\n"
            "fi\n"
        )
        hook_path.write_text(hook_script, encoding="utf-8")
        hook_path.chmod(hook_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return hook_path


    def crystallize_executable_skill(
        self,
        slug: str,
        title: str,
        trigger_when: str,
        script_body: str,
        steps: list[str] | None = None,
        verification_cmd: str = "pytest -q",
    ) -> SkillCard:
        """Validate `script_body` via `ast.parse` and persist an executable skill into `.fix/skills/<slug>/`."""
        ast.parse(script_body, filename=f"{slug}/run.py")
        skill = SkillCard(
            slug=slug,
            title=title,
            trigger_when=trigger_when,
            steps=steps or [f"Execute `run_skill(slug='{slug}')` for automated repository diagnosis."],
            verification_cmd=verification_cmd,
            script_body=script_body,
        )
        self.memory.commit_delta(MemoryDelta(skills_to_save=[skill]))
        return skill

    def _extract_inspected_paths(self, session: TaskSession) -> list[str]:
        """Extract relative file paths inspected via Read/Symbols/LSPTrace during a task session."""
        paths: list[str] = []
        for entry in session.tool_calls_log:
            m = re.match(r"^(?:Read|Symbols|Edit|Write)\(([^:)]+)", entry)
            if m:
                cand = m.group(1).strip()
                if cand and (self.repo_root / cand).is_file() and cand not in paths:
                    paths.append(cand)
        return paths

    def bootstrap_project_skill(self, topic: str = "auto") -> SkillCard:
        """
        Synthesize a universal, domain-agnostic executable Python diagnostic skill tailored
        to the current repository's codebase and architecture (works across Python, TS/JS, Go,
        Rust, C/C++, Java, Kotlin, etc.).
        """
        if topic and topic not in ("auto", "default"):
            slug = self._make_slug(f"{topic}-audit")
            title = f"Project Diagnostic Skill: {topic}"
        else:
            slug = "project-architecture-and-risk-radar"
            title = "Universal Project Architecture, Lifecycle & Hotspot Radar"

        trigger_when = (
            "Auditing module complexity, resource/async/timer lifecycle cleanup invariants, "
            "and high-coupling architectural hotspots across the repository"
        )
        script_code = r"""#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path

IGNORE_DIRS = {
    ".git", ".fix", ".venv", "venv", "node_modules", "__pycache__",
    ".pytest_cache", "build", "dist", "target", "out", ".gradle", ".idea",
}
CODE_EXTS = (".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java", ".kt", ".c", ".cpp", ".cs", ".swift")

# Universal cross-language patterns for stateful resources/timers/tasks/subscriptions
RESOURCE_ACQUIRE_RE = re.compile(
    r"\b(setInterval|addEventListener|subscribe|create_task|go\s+func|thread::spawn|"
    r"Handler|ValueAnimator|ObjectAnimator|AnimationDrawable|postDelayed|open\(|sqlite3\.connect)\b"
)
RESOURCE_RELEASE_RE = re.compile(
    r"\b(clearInterval|removeEventListener|unsubscribe|cancel|close|defer\s+|"
    r"onDetachedFromWindow|onDestroy|dispose|__exit__|with\s+)\b"
)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--target", default="")
    args = parser.parse_args()
    root = Path(args.repo).resolve()

    hotspots = []
    lifecycle_warnings = []

    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in sorted(dirs) if d not in IGNORE_DIRS and not d.startswith(".")]
        for fname in sorted(files):
            if not fname.endswith(CODE_EXTS):
                continue
            fpath = Path(dirpath) / fname
            rel = str(fpath.relative_to(root))
            if args.target and args.target not in rel and args.target not in fname:
                continue
            try:
                text = fpath.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            lines = text.splitlines()
            hotspots.append((len(lines), rel))

            acquires = RESOURCE_ACQUIRE_RE.findall(text)
            if acquires:
                has_release = bool(RESOURCE_RELEASE_RE.search(text))
                status = (
                    "OK (cleanup/release hook present)"
                    if has_release
                    else "WARN: stateful resource/timer/task acquired without obvious cleanup/release hook"
                )
                uniq_acq = ", ".join(sorted(set(acquires))[:4])
                lifecycle_warnings.append(f"- [{rel}] ({len(lines)} lines) uses `{uniq_acq}` -> {status}")

    hotspots.sort(reverse=True)
    print("=== Universal Skill Report: Architecture & Lifecycle Risk Radar ===")
    if lifecycle_warnings:
        print("\n[Resource / Async / Lifecycle Invariant Audit]")
        print("\n".join(lifecycle_warnings[:25]))
    if hotspots:
        print("\n[Top Module Size & Complexity Hotspots]")
        for cnt, rel in hotspots[:12]:
            badge = "[LARGE]" if cnt > 450 else "[OK]"
            print(f"- {badge} {rel}: {cnt} lines")

if __name__ == "__main__":
    main()
"""
        return self.crystallize_executable_skill(
            slug=slug,
            title=title,
            trigger_when=trigger_when,
            script_body=script_code,
        )
