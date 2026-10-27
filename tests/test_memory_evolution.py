"""Tests for DualMemoryStore (SQLite WAL + FTS5 + SHA staleness) & EvolutionEngine (Semantic Diff Filter)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from fix.evolution import EvolutionEngine
from fix.memory import DualMemoryStore
from fix.schema import EvidenceFact, FactStatus, MemoryDelta, RuleStatus, SkillCard, StyleRule


def _setup_git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "dev@fix.local"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Fix Tester"], cwd=repo, check=True)

    # Create 3 co-changing commits between service.py and schema.py
    for i in range(1, 4):
        (repo / "service.py").write_text(f"VERSION = {i}\n", encoding="utf-8")
        (repo / "schema.py").write_text(f"SCHEMA_VER = {i}\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-m", f"Sync v{i}"], cwd=repo, check=True, capture_output=True)
    return repo


def test_fts5_landmarks_and_git_sha_staleness(tmp_path: Path) -> None:
    repo = _setup_git_repo(tmp_path)
    persona_db = tmp_path / "persona.db"
    memory = DualMemoryStore(repo_root=repo, persona_db_path=persona_db)

    head_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()

    fact = EvidenceFact(
        summary="Order idempotency lock is enforced in service.py before payment deduction",
        category="invariant",
        evidence_paths=["service.py"],
        commit_sha=head_sha,
        status=FactStatus.VERIFIED,
    )
    memory.upsert_fact(fact)

    # 1. FTS5 search retrieves the verified landmark
    matches = memory.search_landmarks("idempotency payment")
    assert len(matches) == 1
    assert matches[0].fact_id == fact.fact_id
    assert not matches[0].stale

    # 2. Modify service.py -> invalidate_stale_facts marks the landmark stale
    (repo / "service.py").write_text("VERSION = 999\n", encoding="utf-8")
    invalidated_count = memory.invalidate_stale_facts()
    assert invalidated_count == 1

    fresh_matches = memory.search_landmarks("idempotency payment", include_stale=False)
    assert len(fresh_matches) == 0

    stale_matches = memory.search_landmarks("idempotency payment", include_stale=True)
    assert len(stale_matches) == 1
    assert stale_matches[0].stale


@pytest.mark.asyncio
async def test_semantic_equivalence_diff_filter_and_rule_promotion(tmp_path: Path) -> None:
    repo = _setup_git_repo(tmp_path)
    persona_db = tmp_path / "persona.db"
    memory = DualMemoryStore(repo_root=repo, persona_db_path=persona_db)
    evolution = EvolutionEngine(repo_root=repo, memory=memory, llm=None)

    # 1. Business feature diff (adding a new discount rate) must NOT be extracted as a StyleRule
    business_diff = (
        "--- a/service.py\n"
        "+++ b/service.py\n"
        "@@ -1,2 +1,4 @@\n"
        " VERSION = 3\n"
        "+VIP_DISCOUNT_RATE = 0.85\n"
        "+def calculate_vip_price(amount: float) -> float:\n"
        "+    return amount * VIP_DISCOUNT_RATE\n"
    )
    learned_from_biz = await evolution.learn_from_human_diff(business_diff, commit_message="feat: add VIP discount")
    assert learned_from_biz == []
    assert len(memory.list_style_rules()) == 0

    # 2. Semantically equivalent refactor (os.path -> pathlib.Path) IS extracted
    style_diff = (
        "--- a/service.py\n"
        "+++ b/service.py\n"
        "@@ -1,3 +1,3 @@\n"
        "-import os.path\n"
        "-cfg = os.path.join(base_dir, 'config.json')\n"
        "+from pathlib import Path\n"
        "+cfg = Path(base_dir) / 'config.json'\n"
    )

    # Hit 1 -> CANDIDATE
    r1 = (await evolution.learn_from_human_diff(style_diff))[0]
    assert r1.hit_count == 1
    assert r1.status == RuleStatus.CANDIDATE

    # Hit 2 -> CANDIDATE
    r2 = (await evolution.learn_from_human_diff(style_diff))[0]
    assert r2.hit_count == 2
    assert r2.status == RuleStatus.CANDIDATE

    # Hit 3 -> Automatically transitions to PENDING_APPROVAL (requiring human confirmation before hard gate)
    r3 = (await evolution.learn_from_human_diff(style_diff))[0]
    assert r3.hit_count == 3
    assert r3.status == RuleStatus.PENDING_APPROVAL

    # Approve rule -> PROMOTED
    assert memory.approve_rule(r3.rule_id)
    active_rules = memory.get_active_persona_rules()
    assert any("[HARD GATE]" in line and "pathlib.Path" in line for line in active_rules)


def test_git_cochange_extraction_and_atomic_memory_delta(tmp_path: Path) -> None:
    repo = _setup_git_repo(tmp_path)
    persona_db = tmp_path / "persona.db"
    memory = DualMemoryStore(repo_root=repo, persona_db_path=persona_db)
    evolution = EvolutionEngine(repo_root=repo, memory=memory)

    edges = evolution.scan_git_cochanges()
    assert len(edges) == 1
    assert {edges[0].file_a, edges[0].file_b} == {"schema.py", "service.py"}
    assert edges[0].co_commits == 3
    assert edges[0].confidence == 1.0

    # Modifying only service.py triggers a co-change coupling alert for schema.py
    partners = memory.get_cochange_partners(["service.py"])
    assert len(partners) == 1
    assert {partners[0].file_a, partners[0].file_b} == {"schema.py", "service.py"}

    # Test Copy-on-Write atomic commit of a SkillCard
    delta = MemoryDelta(
        skills_to_save=[
            SkillCard(
                slug="fix-double-deduction",
                title="Playbook: Order Double Deduction",
                trigger_when="Concurrent order state updates",
                steps=["Check row lock", "Run concurrency pytest"],
                verification_cmd="pytest -q",
            )
        ]
    )
    memory.commit_delta(delta)
    assert "fix-double-deduction" in memory.list_skills()
    assert (repo / ".fix" / "skills" / "fix-double-deduction" / "SKILL.md").exists()


@pytest.mark.asyncio
async def test_five_pillars_higher_order_self_evolution_and_first_principles_essence(tmp_path: Path) -> None:
    """
    Verifies the 5 Pillars of Higher-Order Cognitive Self-Evolution:
    1. 3-Layer First-Principles `MentalModel` (`现象层 -> 机制与契约层 -> 本质与第一性原理层 -> UX 影响`).
    2. Proactive Project & UX `OptimizationProposal` extraction (`architecture`, `user_experience`, `reliability`).
    3. Meta-Cognitive Self-Reflection (`MetaReflection` + `CognitiveParadigm` + user prompt `StyleRule` learning).
    4. Background `FixLooper` `ARCHITECTURAL_ESSENCE_SYNTHESIS` P2 idle tick & `/evolve` command.
    5. Adaptive 950-line full-file `read_file` without 8KB middle truncation.
    """
    from io import StringIO
    from fix.cli import FixInteractiveSession, create_fix_console
    from fix.context import ContextManager
    from fix.looper import FixLooper
    from fix.schema import IdleTickKind, TaskIntent, TaskSession
    from fix.tools import WorkspaceToolkit

    repo = _setup_git_repo(tmp_path)
    persona_db = tmp_path / "persona.db"
    memory = DualMemoryStore(repo_root=repo, persona_db_path=persona_db)
    toolkit = WorkspaceToolkit(repo)
    evolution = EvolutionEngine(repo_root=repo, memory=memory, llm=None)
    looper = FixLooper(repo_root=repo, toolkit=toolkit, memory=memory, evolution=evolution)

    # Create an 800-line stateful UI component file that acquires a Handler/ValueAnimator without a release hook
    lines = ["class BubbleVoiceView:"]
    lines.append("    def start_entry_anim(self):")
    lines.append("        self.anim = ValueAnimator()")
    lines.append("        self.timer = postDelayed(666)")
    for idx in range(5, 801):
        lines.append(f"    def helper_{idx}(self): return {idx}")
    (repo / "BubbleVoiceView.py").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Pillar 5: Verify read_file reads all 800 lines in a SINGLE default call (start_line=1, end_line=500) without truncation!
    read_res = toolkit.read_file("BubbleVoiceView.py", start_line=1, end_line=500)
    assert read_res.ok
    assert not read_res.truncated
    assert "[BubbleVoiceView.py lines 1-800 of 800]" in read_res.output
    assert "def helper_800(self): return 800" in read_res.output

    # Simulate a completed analysis session where the file was read twice and a report with risks was produced
    sess = TaskSession(
        repo_root=str(repo),
        user_prompt="深入分析BubbleVoiceView，整理出工作机制、功能地图、业务逻辑地图、代码逻辑地图，透过现象看本质并优化用户体验",
        intent=TaskIntent.EXPLAIN,
        final_summary=(
            "# BubbleVoiceView 深度分析\n\n"
            "## 结论\n气泡动画与文字绘制双层组件。\n\n"
            "## 设计要点与潜在隐患\n"
            "1. **退场动画淡出未生效**：白天模式下气泡与文字透明动画不同步，导致界面闪烁。\n"
            "2. **状态字段双份拷贝耦合**：子类复制了基类状态，容易导致状态失步。\n"
        ),
        tool_calls_count=2,
        tool_calls_log=["Read(BubbleVoiceView.py:1-400)", "Read(BubbleVoiceView.py:401-800)"],
    )

    delta = evolution.digest_completed_task(sess, head_sha=toolkit.get_head_sha())

    # Pillar 1: 3-Layer First-Principles MentalModel distilled & persisted
    models = memory.list_mental_models(query="BubbleVoiceView")
    assert len(models) >= 1
    mm = models[0]
    assert mm.component_or_scope == "BubbleVoiceView"
    assert "800 lines" in mm.phenomenon_layer
    assert "ValueAnimator" in mm.mechanism_contract_layer
    assert "invariant" in mm.essence_first_principles.lower() or "density" in mm.essence_first_principles.lower()
    assert mm.ux_and_product_impact

    # Pillar 2: Proactive Project & UX OptimizationProposals generated & persisted
    proposals = memory.list_optimization_proposals(status="OPEN")
    assert len(proposals) >= 2
    dims = {p.dimension for p in proposals}
    assert "architecture" in dims or "reliability" in dims
    assert "user_experience" in dims

    # Pillar 3: Meta-Cognitive Self-Reflection & Paradigm Evolution
    assert sess.meta_reflection is not None
    assert sess.meta_reflection.redundant_reads_detected == 1
    assert "1000 lines" in sess.meta_reflection.efficiency_lesson
    paradigms = memory.list_cognitive_paradigms()
    assert len(paradigms) >= 1
    assert any("现象层" in c for c in paradigms[0].first_principles_checklist)
    # Also learned developer StyleRules from user prompt expectations
    rules = memory.list_style_rules()
    assert len(rules) >= 1

    # Verify ContextManager injects evolved Paradigms, 3-Layer MentalModels, and UX Optimization Proposals
    ctx_mgr = ContextManager()
    profile = ctx_mgr.detect_repo_profile(repo)
    msgs = ctx_mgr.build_messages(
        session=sess,
        profile=profile,
        history=[],
        mental_models=[f"[{mm.component_or_scope}] {mm.essence_first_principles}"],
        cognitive_paradigms=memory.get_active_cognitive_paradigms(),
        optimization_proposals=[p.title for p in proposals],
    )
    sys_prompt = msgs[0]["content"]
    assert "Self-Evolved Cognitive Paradigms" in sys_prompt
    assert "Distilled 3-Layer Mental Models" in sys_prompt
    assert "Proactive Project & UX Optimization Radar" in sys_prompt

    # Pillar 4: Looper schedules & executes ARCHITECTURAL_ESSENCE_SYNTHESIS in background P2 queue
    looper.schedule_post_task_idle_events(sess)
    assert any(t.kind == IdleTickKind.ARCHITECTURAL_ESSENCE_SYNTHESIS for t in looper.p2_idle_queue)
    outcomes = await looper.drain_idle_queue(max_ticks=5)
    assert any(o.kind == IdleTickKind.ARCHITECTURAL_ESSENCE_SYNTHESIS and o.committed for o in outcomes)

    # Verify `/evolve` command renders all 3 evolution tables cleanly
    buf = StringIO()
    out_c = create_fix_console(file=buf, width=120, force_terminal=False)
    cli_sess = FixInteractiveSession(
        repo=repo,
        stack=(toolkit, memory, ctx_mgr, None, evolution, None),  # type: ignore[arg-type]
        out_console=out_c,
    )
    await cli_sess.handle_line("/evolve")
    evolve_out = buf.getvalue()
    assert "三层本质心智模型" in evolve_out
    assert "主动项目完善与用户体验优化提案" in evolve_out
    assert "元认知自我进化范式" in evolve_out
