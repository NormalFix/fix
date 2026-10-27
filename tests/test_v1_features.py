"""Tests for Fix v1.0.0-beta: Universal Polyglot Semantic & Declarative Radar, Executable Skill Bootstrapping & L1-L4 Autonomy."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from fix.autonomy import AutonomyRouter, SandboxVerifier
from fix.evolution import EvolutionEngine
from fix.memory import DualMemoryStore
from fix.schema import (
    AutonomyLevel,
    EvidenceFact,
    FactStatus,
    RuleStatus,
    StyleRule,
    TaskIntent,
    TaskPhase,
    TaskSession,
)
from fix.tools import WorkspaceToolkit


def _create_polyglot_microservice_repo(tmp_path: Path) -> Path:
    """Create a realistic polyglot repository with TypeScript, Rust, Kotlin, YAML, and XML configs."""
    repo = tmp_path / "polyglot_platform"
    ts_dir = repo / "services/gateway/src"
    rs_dir = repo / "crates/engine/src"
    kt_dir = repo / "clients/app/src/main"
    cfg_dir = repo / "deploy/k8s"

    for d in (ts_dir, rs_dir, kt_dir, cfg_dir):
        d.mkdir(parents=True)

    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "dev@fix.local"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Fix Tester"], cwd=repo, check=True)

    # 1. TypeScript service with interface, implementation, decorator & unclosed setInterval
    (ts_dir / "payment.ts").write_text(
        "export interface PaymentGateway {\n"
        "    processCharge(amount: number): Promise<boolean>;\n"
        "}\n\n"
        "@Injectable()\n"
        "export class StripeGateway extends BaseClient implements PaymentGateway {\n"
        "    private poller = setInterval(() => {}, 5000);\n"
        "    async processCharge(amount: number): Promise<boolean> {\n"
        "        return this.validateAmount(amount);\n"
        "    }\n"
        "    private validateAmount(amount: number): boolean { return amount > 0; }\n"
        "}\n",
        encoding="utf-8",
    )

    # 2. Rust trait & struct implementation
    (rs_dir / "storage.rs").write_text(
        "pub trait StorageBackend {\n"
        "    fn persist(&self, key: &str) -> bool;\n"
        "}\n\n"
        "#[derive(Debug, Clone)]\n"
        "pub struct RocksEngine {\n"
        "    pub path: String,\n"
        "}\n\n"
        "impl StorageBackend for RocksEngine {\n"
        "    fn persist(&self, key: &str) -> bool {\n"
        "        validate_key(key)\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )

    # 3. Declarative YAML & XML config referencing `StripeGateway` and `RocksEngine`
    (cfg_dir / "services.yaml").write_text(
        "apiVersion: v1\n"
        "kind: ServiceConfig\n"
        "spec:\n"
        "  paymentProvider: StripeGateway\n"
        "  storageEngine: RocksEngine\n",
        encoding="utf-8",
    )
    (cfg_dir / "pipeline.xml").write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        "<pipeline>\n"
        '    <node id="pay_node" handler="StripeGateway" />\n'
        "</pipeline>\n",
        encoding="utf-8",
    )

    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "Initial polyglot platform"], cwd=repo, check=True, capture_output=True)
    return repo


def test_universal_polyglot_semantic_radar_and_syntax_gates(tmp_path: Path) -> None:
    repo = _create_polyglot_microservice_repo(tmp_path)
    toolkit = WorkspaceToolkit(repo)

    # 1. Trace TypeScript class `StripeGateway`:
    #    - Resolves supertypes (`BaseClient`, `PaymentGateway`)
    #    - Resolves `@Injectable()` decorator wiring
    #    - Cross-references declarative `services.yaml` and `pipeline.xml`!
    trace_ts = toolkit.trace_symbol("StripeGateway")
    assert trace_ts.ok
    assert "payment.ts:6-12" in trace_ts.output
    assert "BaseClient" in trace_ts.output
    assert "PaymentGateway" in trace_ts.output
    assert "@Injectable()" in trace_ts.output
    assert "services.yaml" in trace_ts.output
    assert "pipeline.xml" in trace_ts.output

    # 2. Trace TypeScript interface `PaymentGateway` -> finds implementing class `StripeGateway`
    trace_iface = toolkit.trace_symbol("PaymentGateway")
    assert trace_iface.ok
    assert "Subclasses / Interface & Trait Implementations" in trace_iface.output
    assert "StripeGateway" in trace_iface.output

    # 3. Trace Rust trait `StorageBackend` -> finds `impl StorageBackend for RocksEngine`!
    trace_rs = toolkit.trace_symbol("StorageBackend")
    assert trace_rs.ok
    assert "RocksEngine" in trace_rs.output

    # 4. Verify XML and C-family/TypeScript/Rust brace balance syntax gates in `apply_patch`
    bad_xml = toolkit.apply_patch(
        "deploy/k8s/pipeline.xml",
        '<node id="pay_node" handler="StripeGateway" />',
        '<node id="pay_node"><unclosed>',
    )
    assert not bad_xml.ok
    assert bad_xml.error_type == "SyntaxError"

    bad_ts = toolkit.apply_patch(
        "services/gateway/src/payment.ts",
        "    private validateAmount(amount: number): boolean { return amount > 0; }",
        "    private validateAmount(amount: number): boolean { if (amount > 0) { return true; }",
    )
    assert not bad_ts.ok
    assert bad_ts.error_type == "SyntaxError"


@pytest.mark.asyncio
async def test_universal_skill_bootstrapping_and_progressive_project_mastery(tmp_path: Path) -> None:
    repo = _create_polyglot_microservice_repo(tmp_path)
    toolkit = WorkspaceToolkit(repo)
    memory = DualMemoryStore(repo, persona_db_path=tmp_path / "persona.db")
    evolution = EvolutionEngine(repo, memory=memory)

    # 1. Bootstrap universal project diagnostic skill (`project-architecture-and-risk-radar`)
    card = evolution.bootstrap_project_skill(topic="auto")
    assert card.slug == "project-architecture-and-risk-radar"
    assert "project-architecture-and-risk-radar" in memory.list_skills()
    assert (repo / ".fix/skills/project-architecture-and-risk-radar/run.py").exists()

    # 2. Execute the bootstrapped skill via `toolkit.run_skill` -> detects unclosed `setInterval` in `payment.ts`!
    res = await toolkit.run_skill("project-architecture-and-risk-radar", target="payment.ts")
    assert res.ok
    assert "payment.ts" in res.output
    assert "setInterval" in res.output
    assert "WARN: stateful resource/timer/task acquired without obvious cleanup/release hook" in res.output

    # 3. Verify that even a read-only `EXPLAIN` task records inspected files into verified `EvidenceFact` landmarks
    explain_session = TaskSession(
        repo_root=str(repo),
        user_prompt="Analyze StripeGateway and RocksEngine architecture",
        intent=TaskIntent.EXPLAIN,
        phase=TaskPhase.DONE,
        final_summary="StripeGateway implements PaymentGateway and persists via RocksEngine.",
        tool_calls_log=["Read(services/gateway/src/payment.ts:1-500)", "Read(crates/engine/src/storage.rs:1-500)"],
    )
    delta = evolution.digest_completed_task(explain_session, head_sha=toolkit.get_head_sha())
    assert len(delta.facts_to_upsert) == 1
    assert "services/gateway/src/payment.ts" in delta.facts_to_upsert[0].evidence_paths
    assert "crates/engine/src/storage.rs" in delta.facts_to_upsert[0].evidence_paths


@pytest.mark.asyncio
async def test_l1_to_l4_progressive_autonomy_router_and_worktree_sandbox_verifier(tmp_path: Path) -> None:
    repo = _create_polyglot_microservice_repo(tmp_path)
    (repo / "test_dummy.py").write_text("def test_ok():\n    assert 1 + 1 == 2\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "add pytest check"], cwd=repo, check=True, capture_output=True)

    toolkit = WorkspaceToolkit(repo)
    memory = DualMemoryStore(repo, persona_db_path=tmp_path / "persona.db")
    router = AutonomyRouter(memory)

    # 1. Fresh repository with 0 landmarks, 0 rules, 0 task history starts at L1_APPRENTICE
    initial = router.assess_autonomy("Refactor database schema across modules", intent=TaskIntent.FEATURE, estimated_files_touched=4)
    assert initial.level == AutonomyLevel.L1_APPRENTICE
    assert not initial.auto_approve_clarifications

    # 2. Accumulate verified landmarks, promoted developer rules, and successful task runs -> upgrades to L4_DIGITAL_TWIN!
    for i in range(8):
        memory.upsert_fact(
            EvidenceFact(
                summary=f"StripeGateway invariant #{i}: validateAmount guards positive charge",
                category="invariant",
                evidence_paths=["services/gateway/src/payment.ts"],
                status=FactStatus.VERIFIED,
            )
        )
    for i in range(3):
        r = memory.upsert_style_rule(
            StyleRule(
                pattern_desc=f"service architecture rule {i}",
                preferred_way="Always release timers and subscriptions on service shutdown",
                status=RuleStatus.PROMOTED,
                hit_count=5,
            )
        )
        memory.approve_rule(r.rule_id)

    for i in range(6):
        memory.record_task_session(
            TaskSession(
                task_id=f"hist_{i}",
                repo_root=str(repo),
                user_prompt="Analyze StripeGateway",
                intent=TaskIntent.BUG_FIX,
                phase=TaskPhase.DONE,
                confidence="HIGH",
                final_summary="Done",
            )
        )

    upgraded = router.assess_autonomy("Fix StripeGateway validation", intent=TaskIntent.BUG_FIX, estimated_files_touched=1)
    assert upgraded.level == AutonomyLevel.L4_DIGITAL_TWIN
    assert upgraded.score >= 0.80
    assert upgraded.auto_approve_clarifications is True

    # 3. Verify `SandboxVerifier` runs baseline vs isolated Git worktree comparison cleanly
    verifier = SandboxVerifier(toolkit)
    comp = await verifier.verify_in_worktree("pytest -q")
    assert comp.tests_passed is True
    assert comp.baseline_exit_code == 0
    assert comp.candidate_exit_code == 0
    assert "Worktree Sandbox Verification" in comp.summary
