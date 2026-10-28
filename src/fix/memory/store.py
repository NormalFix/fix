"""Dual-Layer Memory Store: Global Persona Mind Mirror (`persona.db`) + Repo Landmarks (`repo.db` with WAL & FTS5)."""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Iterable

from fix.protocol.schema import (
    CoChangeEdge,
    CognitiveParadigm,
    EvidenceFact,
    FactStatus,
    MemoryDelta,
    MentalModel,
    OptimizationProposal,
    RuleStatus,
    SkillCard,
    StyleRule,
    TaskSession,
)


class DualMemoryStore:
    """Manages `~/.fix/persona.db` (cross-project developer taste) and `<repo>/.fix/repo.db` (project landmarks)."""

    def __init__(
        self,
        repo_root: str | Path,
        persona_db_path: str | Path | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.fix_dir = self.repo_root / ".fix"
        self.fix_dir.mkdir(parents=True, exist_ok=True)
        self.skills_dir = self.fix_dir / "skills"
        self.repo_db_path = self.fix_dir / "repo.db"

        if persona_db_path is not None:
            self.persona_db_path = Path(persona_db_path).resolve()
        else:
            self.persona_db_path = Path.home() / ".fix" / "persona.db"
        self.persona_db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_persona_db()
        self._init_repo_db()

    def _connect_persona(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.persona_db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _connect_repo(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.repo_db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_persona_db(self) -> None:
        with self._connect_persona() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS style_rules (
                    rule_id TEXT PRIMARY KEY,
                    pattern_desc TEXT NOT NULL,
                    preferred_way TEXT NOT NULL,
                    category TEXT NOT NULL,
                    hit_count INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'CANDIDATE',
                    source_example TEXT NOT NULL DEFAULT '',
                    updated_at REAL NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cognitive_paradigms (
                    paradigm_id TEXT PRIMARY KEY,
                    domain_trigger TEXT NOT NULL,
                    first_principles_checklist_json TEXT NOT NULL,
                    exploration_efficiency_lesson TEXT NOT NULL DEFAULT '',
                    ux_lens TEXT NOT NULL DEFAULT '',
                    hit_count INTEGER NOT NULL DEFAULT 1,
                    updated_at REAL NOT NULL
                );
                """
            )

    def _init_repo_db(self) -> None:
        with self._connect_repo() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS evidence_facts (
                    fact_id TEXT PRIMARY KEY,
                    summary TEXT NOT NULL,
                    category TEXT NOT NULL,
                    evidence_paths_json TEXT NOT NULL,
                    commit_sha TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL DEFAULT 0.9,
                    status TEXT NOT NULL DEFAULT 'VERIFIED',
                    stale INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS evidence_facts_fts USING fts5(
                    fact_id UNINDEXED,
                    summary,
                    category,
                    evidence_paths
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS co_changes (
                    file_a TEXT NOT NULL,
                    file_b TEXT NOT NULL,
                    co_commits INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    PRIMARY KEY (file_a, file_b)
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_history (
                    task_id TEXT PRIMARY KEY,
                    user_prompt TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    final_summary TEXT NOT NULL,
                    confidence TEXT NOT NULL,
                    modified_files_json TEXT NOT NULL,
                    completed_at REAL NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mental_models (
                    model_id TEXT PRIMARY KEY,
                    component_or_scope TEXT NOT NULL,
                    phenomenon_layer TEXT NOT NULL,
                    mechanism_contract_layer TEXT NOT NULL,
                    essence_first_principles TEXT NOT NULL,
                    ux_and_product_impact TEXT NOT NULL DEFAULT '',
                    evidence_paths_json TEXT NOT NULL,
                    commit_sha TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL DEFAULT 0.92,
                    updated_at REAL NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS optimization_proposals (
                    proposal_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    dimension TEXT NOT NULL,
                    target_files_json TEXT NOT NULL,
                    phenomenon_symptom TEXT NOT NULL,
                    root_essence TEXT NOT NULL,
                    proposed_evolution TEXT NOT NULL,
                    ux_or_engineering_gain TEXT NOT NULL,
                    priority TEXT NOT NULL DEFAULT 'P1',
                    status TEXT NOT NULL DEFAULT 'OPEN',
                    updated_at REAL NOT NULL
                );
                """
            )

    # ------------------------------------------------------------------
    # 1. Global Persona Mind Mirror (`persona.db`)
    # ------------------------------------------------------------------

    def upsert_style_rule(self, rule: StyleRule) -> StyleRule:
        """Insert or increment hit_count of a developer StyleRule; auto-promote at hit_count >= 3."""
        with self._connect_persona() as conn:
            existing = conn.execute(
                """
                SELECT * FROM style_rules
                WHERE rule_id = ? OR (category = ? AND lower(pattern_desc) = lower(?))
                """,
                (rule.rule_id, rule.category, rule.pattern_desc),
            ).fetchone()

            now = time.time()
            if existing:
                rule_id = existing["rule_id"]
                new_hits = int(existing["hit_count"]) + max(1, rule.hit_count)
                curr_status = RuleStatus(existing["status"])
                if curr_status == RuleStatus.CANDIDATE and new_hits >= 3:
                    curr_status = RuleStatus.PENDING_APPROVAL
                updated = StyleRule(
                    rule_id=rule_id,
                    pattern_desc=existing["pattern_desc"],
                    preferred_way=rule.preferred_way or existing["preferred_way"],
                    category=existing["category"],
                    hit_count=new_hits,
                    status=curr_status,
                    source_example=rule.source_example or existing["source_example"],
                    updated_at=now,
                )
                conn.execute(
                    """
                    UPDATE style_rules
                    SET preferred_way = ?, hit_count = ?, status = ?, source_example = ?, updated_at = ?
                    WHERE rule_id = ?
                    """,
                    (
                        updated.preferred_way,
                        updated.hit_count,
                        updated.status.value,
                        updated.source_example,
                        updated.updated_at,
                        updated.rule_id,
                    ),
                )
                return updated

            initial_status = rule.status
            if initial_status == RuleStatus.CANDIDATE and rule.hit_count >= 3:
                initial_status = RuleStatus.PENDING_APPROVAL
            inserted = rule.model_copy(update={"status": initial_status, "updated_at": now})
            conn.execute(
                """
                INSERT INTO style_rules (
                    rule_id, pattern_desc, preferred_way, category, hit_count, status, source_example, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    inserted.rule_id,
                    inserted.pattern_desc,
                    inserted.preferred_way,
                    inserted.category,
                    inserted.hit_count,
                    inserted.status.value,
                    inserted.source_example,
                    inserted.updated_at,
                ),
            )
            return inserted

    def approve_rule(self, rule_id: str) -> bool:
        """Promote a PENDING_APPROVAL or CANDIDATE rule to PROMOTED (hard review gate)."""
        with self._connect_persona() as conn:
            cur = conn.execute(
                "UPDATE style_rules SET status = ?, updated_at = ? WHERE rule_id = ?",
                (RuleStatus.PROMOTED.value, time.time(), rule_id),
            )
            return cur.rowcount > 0

    def reject_rule(self, rule_id: str) -> bool:
        """Mark a rule as REJECTED so it is never injected or enforced."""
        with self._connect_persona() as conn:
            cur = conn.execute(
                "UPDATE style_rules SET status = ?, updated_at = ? WHERE rule_id = ?",
                (RuleStatus.REJECTED.value, time.time(), rule_id),
            )
            return cur.rowcount > 0

    def list_style_rules(self, include_rejected: bool = False) -> list[StyleRule]:
        with self._connect_persona() as conn:
            if include_rejected:
                rows = conn.execute("SELECT * FROM style_rules ORDER BY hit_count DESC, updated_at DESC").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM style_rules WHERE status != 'REJECTED' ORDER BY hit_count DESC, updated_at DESC"
                ).fetchall()
        return [
            StyleRule(
                rule_id=r["rule_id"],
                pattern_desc=r["pattern_desc"],
                preferred_way=r["preferred_way"],
                category=r["category"],
                hit_count=r["hit_count"],
                status=RuleStatus(r["status"]),
                source_example=r["source_example"],
                updated_at=r["updated_at"],
            )
            for r in rows
        ]

    def get_active_persona_rules(self) -> list[str]:
        """Return formatted developer preferences for prompt & Reviewer injection."""
        rules = self.list_style_rules(include_rejected=False)
        formatted: list[str] = []
        for r in rules:
            badge = (
                "[HARD GATE]"
                if r.status == RuleStatus.PROMOTED
                else "[PENDING APPROVAL]"
                if r.status == RuleStatus.PENDING_APPROVAL
                else f"[CANDIDATE hits={r.hit_count}]"
            )
            formatted.append(f"{badge} ({r.category}) When {r.pattern_desc} -> Prefer: {r.preferred_way}")
        return formatted

    # ------------------------------------------------------------------
    # 2. Project Landmarks & FTS5 Search (`repo.db`)
    # ------------------------------------------------------------------

    def upsert_fact(self, fact: EvidenceFact) -> EvidenceFact:
        """Upsert a project EvidenceFact and synchronize the FTS5 index."""
        with self._connect_repo() as conn:
            self._upsert_fact_in_tx(conn, fact)
        return fact

    @staticmethod
    def _upsert_fact_in_tx(conn: sqlite3.Connection, fact: EvidenceFact) -> None:
        paths_json = json.dumps(fact.evidence_paths, ensure_ascii=False)
        conn.execute(
            """
            INSERT INTO evidence_facts (
                fact_id, summary, category, evidence_paths_json, commit_sha, confidence, status, stale, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(fact_id) DO UPDATE SET
                summary = excluded.summary,
                category = excluded.category,
                evidence_paths_json = excluded.evidence_paths_json,
                commit_sha = excluded.commit_sha,
                confidence = excluded.confidence,
                status = excluded.status,
                stale = excluded.stale,
                updated_at = excluded.updated_at
            """,
            (
                fact.fact_id,
                fact.summary,
                fact.category,
                paths_json,
                fact.commit_sha,
                fact.confidence,
                fact.status.value,
                1 if fact.stale else 0,
                fact.updated_at,
            ),
        )
        conn.execute("DELETE FROM evidence_facts_fts WHERE fact_id = ?", (fact.fact_id,))
        conn.execute(
            """
            INSERT INTO evidence_facts_fts (fact_id, summary, category, evidence_paths)
            VALUES (?, ?, ?, ?)
            """,
            (fact.fact_id, fact.summary, fact.category, " ".join(fact.evidence_paths)),
        )

    def search_landmarks(
        self,
        query: str = "",
        limit: int = 8,
        include_stale: bool = False,
    ) -> list[EvidenceFact]:
        """Search verified repository landmarks via SQLite FTS5 with ranking fallback."""
        stale_clause = "" if include_stale else "AND f.stale = 0"
        tokens = [t for t in re.findall(r"[A-Za-z0-9_]+", query) if len(t) >= 2]

        with self._connect_repo() as conn:
            rows = []
            if tokens:
                fts_query = " OR ".join(f'"{t}"' for t in tokens[:8])
                try:
                    rows = conn.execute(
                        f"""
                        SELECT f.*
                        FROM evidence_facts_fts fts
                        JOIN evidence_facts f ON f.fact_id = fts.fact_id
                        WHERE evidence_facts_fts MATCH ? {stale_clause}
                        ORDER BY f.stale ASC, f.confidence DESC, f.updated_at DESC
                        LIMIT ?
                        """,
                        (fts_query, limit),
                    ).fetchall()
                except sqlite3.OperationalError:
                    rows = []

            if not rows:
                where_stale = "" if include_stale else "WHERE stale = 0"
                rows = conn.execute(
                    f"""
                    SELECT * FROM evidence_facts
                    {where_stale}
                    ORDER BY stale ASC, confidence DESC, updated_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()

        return [self._row_to_fact(r) for r in rows]

    @staticmethod
    def _row_to_fact(row: sqlite3.Row) -> EvidenceFact:
        return EvidenceFact(
            fact_id=row["fact_id"],
            summary=row["summary"],
            category=row["category"],
            evidence_paths=json.loads(row["evidence_paths_json"]),
            commit_sha=row["commit_sha"],
            confidence=float(row["confidence"]),
            status=FactStatus(row["status"]),
            stale=bool(row["stale"]),
            updated_at=float(row["updated_at"]),
        )

    def invalidate_stale_facts(self) -> int:
        """Mark facts as `stale=True` if any of their evidence_paths changed since `commit_sha`."""
        dirty_proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
        )
        dirty_files: set[str] = set()
        if dirty_proc.returncode == 0:
            for line in dirty_proc.stdout.splitlines():
                if len(line) > 3:
                    dirty_files.add(line[3:].strip().strip('"'))

        head_proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
        )
        head_sha = head_proc.stdout.strip() if head_proc.returncode == 0 else ""

        facts = self.search_landmarks(query="", limit=500, include_stale=False)
        invalidated = 0

        with self._connect_repo() as conn:
            for fact in facts:
                if not fact.evidence_paths:
                    continue
                changed_since_fact = set(dirty_files)
                if fact.commit_sha and head_sha and fact.commit_sha != head_sha:
                    diff_proc = subprocess.run(
                        ["git", "diff", "--name-only", fact.commit_sha, head_sha],
                        cwd=self.repo_root,
                        capture_output=True,
                        text=True,
                    )
                    if diff_proc.returncode == 0:
                        changed_since_fact.update(
                            line.strip() for line in diff_proc.stdout.splitlines() if line.strip()
                        )
                    else:
                        # Commit SHA unreachable/rewritten -> treat as stale if file moved/deleted
                        changed_since_fact.update(fact.evidence_paths)

                for ev_path in fact.evidence_paths:
                    full_path = self.repo_root / ev_path
                    if not full_path.exists() or ev_path in changed_since_fact:
                        conn.execute(
                            "UPDATE evidence_facts SET stale = 1, updated_at = ? WHERE fact_id = ?",
                            (time.time(), fact.fact_id),
                        )
                        invalidated += 1
                        break

        return invalidated

    # ------------------------------------------------------------------
    # 3. Co-Change Coupling & Copy-on-Write Atomic Commit (`MemoryDelta`)
    # ------------------------------------------------------------------

    def get_cochange_partners(
        self,
        modified_files: Iterable[str],
        min_confidence: float = 0.25,
    ) -> list[CoChangeEdge]:
        """Return historically coupled files that were NOT modified alongside `modified_files`."""
        mod_set = set(modified_files)
        if not mod_set:
            return []

        placeholders = ",".join("?" for _ in mod_set)
        params = list(mod_set) + list(mod_set) + [min_confidence]
        with self._connect_repo() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM co_changes
                WHERE (file_a IN ({placeholders}) OR file_b IN ({placeholders}))
                  AND confidence >= ?
                ORDER BY confidence DESC, co_commits DESC
                LIMIT 10
                """,
                params,
            ).fetchall()

        warnings: list[CoChangeEdge] = []
        for r in rows:
            fa, fb = r["file_a"], r["file_b"]
            if (fa in mod_set) ^ (fb in mod_set):
                warnings.append(
                    CoChangeEdge(
                        file_a=fa,
                        file_b=fb,
                        co_commits=int(r["co_commits"]),
                        confidence=float(r["confidence"]),
                    )
                )
        return warnings

    def record_task_session(self, session: TaskSession) -> None:
        """Persist completed task summary into `repo.db`."""
        with self._connect_repo() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO task_history (
                    task_id, user_prompt, intent, final_summary, confidence, modified_files_json, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.task_id,
                    session.user_prompt,
                    session.intent.value if session.intent else "unknown",
                    session.final_summary,
                    session.confidence,
                    json.dumps(sorted(session.modified_files)),
                    session.completed_at or time.time(),
                ),
            )

    def upsert_mental_model(self, model: MentalModel) -> MentalModel:
        """Upsert a 3-layer First-Principles MentalModel (`现象 -> 机制契约 -> 本质与UX`) in `repo.db`."""
        with self._connect_repo() as conn:
            self._upsert_mental_model_in_tx(conn, model)
        return model

    @staticmethod
    def _upsert_mental_model_in_tx(conn: sqlite3.Connection, model: MentalModel) -> None:
        existing = conn.execute(
            "SELECT model_id FROM mental_models WHERE lower(component_or_scope) = lower(?)",
            (model.component_or_scope,),
        ).fetchone()
        target_id = existing["model_id"] if existing else model.model_id
        conn.execute(
            """
            INSERT INTO mental_models (
                model_id, component_or_scope, phenomenon_layer, mechanism_contract_layer,
                essence_first_principles, ux_and_product_impact, evidence_paths_json,
                commit_sha, confidence, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_id) DO UPDATE SET
                component_or_scope = excluded.component_or_scope,
                phenomenon_layer = excluded.phenomenon_layer,
                mechanism_contract_layer = excluded.mechanism_contract_layer,
                essence_first_principles = excluded.essence_first_principles,
                ux_and_product_impact = excluded.ux_and_product_impact,
                evidence_paths_json = excluded.evidence_paths_json,
                commit_sha = excluded.commit_sha,
                confidence = excluded.confidence,
                updated_at = excluded.updated_at
            """,
            (
                target_id,
                model.component_or_scope,
                model.phenomenon_layer,
                model.mechanism_contract_layer,
                model.essence_first_principles,
                model.ux_and_product_impact,
                json.dumps(model.evidence_paths, ensure_ascii=False),
                model.commit_sha,
                model.confidence,
                model.updated_at,
            ),
        )

    def list_mental_models(self, query: str = "", limit: int = 8) -> list[MentalModel]:
        """List distilled 3-layer MentalModels ordered by relevance and recency."""
        with self._connect_repo() as conn:
            rows = conn.execute(
                "SELECT * FROM mental_models ORDER BY confidence DESC, updated_at DESC LIMIT ?",
                (max(limit * 2, 16),),
            ).fetchall()
        models = [
            MentalModel(
                model_id=r["model_id"],
                component_or_scope=r["component_or_scope"],
                phenomenon_layer=r["phenomenon_layer"],
                mechanism_contract_layer=r["mechanism_contract_layer"],
                essence_first_principles=r["essence_first_principles"],
                ux_and_product_impact=r["ux_and_product_impact"],
                evidence_paths=json.loads(r["evidence_paths_json"]),
                commit_sha=r["commit_sha"],
                confidence=float(r["confidence"]),
                updated_at=float(r["updated_at"]),
            )
            for r in rows
        ]
        if not query.strip():
            return models[:limit]
        q_low = query.lower()
        scored = sorted(
            models,
            key=lambda m: (
                1 if m.component_or_scope.lower() in q_low or q_low in m.component_or_scope.lower() else 0,
                sum(1 for p in m.evidence_paths if Path(p).stem.lower() in q_low),
                m.updated_at,
            ),
            reverse=True,
        )
        return scored[:limit]

    def upsert_optimization_proposal(self, prop: OptimizationProposal) -> OptimizationProposal:
        """Upsert an architectural or user-experience OptimizationProposal in `repo.db`."""
        with self._connect_repo() as conn:
            self._upsert_optimization_proposal_in_tx(conn, prop)
        return prop

    @staticmethod
    def _upsert_optimization_proposal_in_tx(conn: sqlite3.Connection, prop: OptimizationProposal) -> None:
        existing = conn.execute(
            "SELECT proposal_id FROM optimization_proposals WHERE lower(title) = lower(?)",
            (prop.title,),
        ).fetchone()
        target_id = existing["proposal_id"] if existing else prop.proposal_id
        conn.execute(
            """
            INSERT INTO optimization_proposals (
                proposal_id, title, dimension, target_files_json, phenomenon_symptom,
                root_essence, proposed_evolution, ux_or_engineering_gain, priority, status, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(proposal_id) DO UPDATE SET
                dimension = excluded.dimension,
                target_files_json = excluded.target_files_json,
                phenomenon_symptom = excluded.phenomenon_symptom,
                root_essence = excluded.root_essence,
                proposed_evolution = excluded.proposed_evolution,
                ux_or_engineering_gain = excluded.ux_or_engineering_gain,
                priority = excluded.priority,
                status = excluded.status,
                updated_at = excluded.updated_at
            """,
            (
                target_id,
                prop.title,
                prop.dimension,
                json.dumps(prop.target_files, ensure_ascii=False),
                prop.phenomenon_symptom,
                prop.root_essence,
                prop.proposed_evolution,
                prop.ux_or_engineering_gain,
                prop.priority,
                prop.status,
                prop.updated_at,
            ),
        )

    def list_optimization_proposals(
        self,
        dimension: str | None = None,
        status: str | None = "OPEN",
        limit: int = 15,
    ) -> list[OptimizationProposal]:
        with self._connect_repo() as conn:
            clauses: list[str] = []
            params: list[ object ] = []
            if dimension:
                clauses.append("dimension = ?")
                params.append(dimension)
            if status:
                clauses.append("status = ?")
                params.append(status)
            where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            params.append(limit)
            rows = conn.execute(
                f"SELECT * FROM optimization_proposals {where_sql} ORDER BY priority ASC, updated_at DESC LIMIT ?",
                params,
            ).fetchall()
        return [
            OptimizationProposal(
                proposal_id=r["proposal_id"],
                title=r["title"],
                dimension=r["dimension"],
                target_files=json.loads(r["target_files_json"]),
                phenomenon_symptom=r["phenomenon_symptom"],
                root_essence=r["root_essence"],
                proposed_evolution=r["proposed_evolution"],
                ux_or_engineering_gain=r["ux_or_engineering_gain"],
                priority=r["priority"],
                status=r["status"],
                updated_at=float(r["updated_at"]),
            )
            for r in rows
        ]

    def upsert_cognitive_paradigm(self, paradigm: CognitiveParadigm) -> CognitiveParadigm:
        """Upsert or reinforce a self-evolved higher-order CognitiveParadigm in `~/.fix/persona.db`."""
        with self._connect_persona() as conn:
            existing = conn.execute(
                "SELECT * FROM cognitive_paradigms WHERE paradigm_id = ? OR lower(domain_trigger) = lower(?)",
                (paradigm.paradigm_id, paradigm.domain_trigger),
            ).fetchone()
            now = time.time()
            if existing:
                pid = existing["paradigm_id"]
                new_hits = int(existing["hit_count"]) + max(1, paradigm.hit_count)
                merged_checks = list(
                    dict.fromkeys(
                        json.loads(existing["first_principles_checklist_json"])
                        + list(paradigm.first_principles_checklist)
                    )
                )[:8]
                eff_lesson = paradigm.exploration_efficiency_lesson or existing["exploration_efficiency_lesson"]
                ux_lens = paradigm.ux_lens or existing["ux_lens"]
                conn.execute(
                    """
                    UPDATE cognitive_paradigms
                    SET first_principles_checklist_json = ?, exploration_efficiency_lesson = ?,
                        ux_lens = ?, hit_count = ?, updated_at = ?
                    WHERE paradigm_id = ?
                    """,
                    (
                        json.dumps(merged_checks, ensure_ascii=False),
                        eff_lesson,
                        ux_lens,
                        new_hits,
                        now,
                        pid,
                    ),
                )
                return CognitiveParadigm(
                    paradigm_id=pid,
                    domain_trigger=existing["domain_trigger"],
                    first_principles_checklist=merged_checks,
                    exploration_efficiency_lesson=eff_lesson,
                    ux_lens=ux_lens,
                    hit_count=new_hits,
                    updated_at=now,
                )

            conn.execute(
                """
                INSERT INTO cognitive_paradigms (
                    paradigm_id, domain_trigger, first_principles_checklist_json,
                    exploration_efficiency_lesson, ux_lens, hit_count, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    paradigm.paradigm_id,
                    paradigm.domain_trigger,
                    json.dumps(paradigm.first_principles_checklist, ensure_ascii=False),
                    paradigm.exploration_efficiency_lesson,
                    paradigm.ux_lens,
                    paradigm.hit_count,
                    now,
                ),
            )
            return paradigm.model_copy(update={"updated_at": now})

    def list_cognitive_paradigms(self, limit: int = 10) -> list[CognitiveParadigm]:
        with self._connect_persona() as conn:
            rows = conn.execute(
                "SELECT * FROM cognitive_paradigms ORDER BY hit_count DESC, updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            CognitiveParadigm(
                paradigm_id=r["paradigm_id"],
                domain_trigger=r["domain_trigger"],
                first_principles_checklist=json.loads(r["first_principles_checklist_json"]),
                exploration_efficiency_lesson=r["exploration_efficiency_lesson"],
                ux_lens=r["ux_lens"],
                hit_count=int(r["hit_count"]),
                updated_at=float(r["updated_at"]),
            )
            for r in rows
        ]

    def get_active_cognitive_paradigms(self, limit: int = 5) -> list[str]:
        """Format self-evolved higher-order thinking paradigms for prompt injection."""
        paradigms = self.list_cognitive_paradigms(limit=limit)
        lines: list[str] = []
        for p in paradigms:
            checks = "; ".join(p.first_principles_checklist[:4])
            line = f"[PARADIGM hits={p.hit_count}] When {p.domain_trigger}: First-Principles Lenses=[{checks}]"
            if p.ux_lens:
                line += f" | UX Lens: {p.ux_lens}"
            if p.exploration_efficiency_lesson:
                line += f" | Efficiency: {p.exploration_efficiency_lesson}"
            lines.append(line)
        return lines

    def commit_delta(self, delta: MemoryDelta) -> None:
        """Atomically commit a Copy-on-Write MemoryDelta across repo.db, persona.db, and .fix/skills/."""
        if delta.is_empty():
            return

        if (
            delta.facts_to_upsert
            or delta.co_changes_to_upsert
            or delta.mental_models_to_upsert
            or delta.optimization_proposals_to_upsert
        ):
            with self._connect_repo() as conn:
                for fact in delta.facts_to_upsert:
                    self._upsert_fact_in_tx(conn, fact)
                for edge in delta.co_changes_to_upsert:
                    a, b = sorted([edge.file_a, edge.file_b])
                    conn.execute(
                        """
                        INSERT INTO co_changes (file_a, file_b, co_commits, confidence)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(file_a, file_b) DO UPDATE SET
                            co_commits = excluded.co_commits,
                            confidence = excluded.confidence
                        """,
                        (a, b, edge.co_commits, edge.confidence),
                    )
                for mm in delta.mental_models_to_upsert:
                    self._upsert_mental_model_in_tx(conn, mm)
                for opt in delta.optimization_proposals_to_upsert:
                    self._upsert_optimization_proposal_in_tx(conn, opt)

        for rule in delta.rules_to_upsert:
            self.upsert_style_rule(rule)

        for paradigm in delta.paradigms_to_upsert:
            self.upsert_cognitive_paradigm(paradigm)

        for skill in delta.skills_to_save:
            self._write_skill_card(skill)

    def _write_skill_card(self, skill: SkillCard) -> Path:
        skill_folder = self.skills_dir / skill.slug
        skill_folder.mkdir(parents=True, exist_ok=True)
        md_path = skill_folder / "SKILL.md"
        steps_md = "\n".join(f"{idx}. {step}" for idx, step in enumerate(skill.steps, start=1))
        body = (
            f"---\n"
            f"name: {skill.slug}\n"
            f"title: {skill.title}\n"
            f"trigger_when: {skill.trigger_when}\n"
            f"verification_cmd: {skill.verification_cmd}\n"
            f"---\n\n"
            f"# {skill.title}\n\n"
            f"**When to Trigger:** {skill.trigger_when}\n\n"
            f"## Execution Playbook\n{steps_md}\n"
        )
        if skill.script_body:
            for script_name in ("run.py", "probe.py"):
                script_path = skill_folder / script_name
                script_path.write_text(skill.script_body, encoding="utf-8")
                script_path.chmod(0o755)
            body += f"\n## Bundled Executable Diagnostic Script\nRun `python3 .fix/skills/{skill.slug}/run.py --repo <repo_root> [--target <path>]`\n" 
        md_path.write_text(body, encoding="utf-8")
        return md_path

    def list_skills(self) -> list[str]:
        if not self.skills_dir.exists():
            return []
        return sorted(
            p.parent.name for p in self.skills_dir.glob("*/SKILL.md")
        )
