# Fix Agent — Engineering Constitution & Architecture Guide

## 1. Core Identity & Mission
**Fix** is not a disposable prompt wrapper. It is a **persistent, self-evolving technical partner** built around an **Event-Driven Micro-Kernel** and **Three Core Flywheels**:
1. **Just-in-Time Deep Probe (`tools.py` + `context.py`)**: Zero-precompile repository entry, scientific `Observe -> Hypothesize -> Reproduce -> Fix -> Verify` loop, AST/LSP symbol radar, and 8KB output truncation with process-group isolation.
2. **Developer Mind Mirror (`memory.py`)**: Dual-layer SQLite memory (`~/.fix/persona.db` for cross-project developer taste/style rules + `<repo>/.fix/repo.db` for project-specific verified landmarks with Git SHA staleness tracking and FTS5 retrieval).
3. **Event-Driven Evolution & Bootstrapping (`evolution.py`)**: Zero-token deterministic Git co-change scanning, post-task reflection, semantic-equivalence diff learning from human commits, and skill crystallization into `.fix/skills/<slug>/SKILL.md`.

## 2. Architectural Invariants (Non-Negotiable Rules)
1. **Single Data Contract (`schema.py`)**:
   - All modules share the Pydantic v2 models in `src/fix/schema.py`.
   - Never introduce redundant `Domain <-> ORM <-> DTO` conversion layers.
2. **Flat, High-Cohesion Module Boundaries**:
   - Keep each core module focused on a single orthogonal responsibility (200–400 lines per file).
   - Lower layers (`schema -> llm/tools/memory -> context/evolution -> runtime -> cli`) never import higher layers.
3. **Physical Tool Gating by Track & Phase**:
   - `TaskIntent.EXPLAIN`: Physically blocks all file-mutation tools (`apply_patch`, `write_new_file`). Every diagnostic claim must cite `[file:line]`.
   - `TaskIntent.BUG_FIX` in `TaskPhase.REPRODUCE`: Only test files (`test_*.py`, `*_test.py`, `tests/**`) may be modified until a failing reproduction is observed or the elastic escape valve (`max_reproduce_attempts = 2`) triggers `degraded_without_repro = True`.
4. **Syntax-Gated Patching & Step-Level Git Snapshots**:
   - Every code modification in `apply_patch` or `write_new_file` must pass an in-memory syntax check (`ast.parse` for Python, JSON/TOML parser, or delimiter check) before touching disk.
   - Before the first write in a step, record a non-destructive tree snapshot via `git stash create` so mid-flight user steering (`steer_queue`) or destructive failures can roll back in milliseconds.
5. **Copy-on-Write Memory Commits (`MemoryDelta`)**:
   - Digest and background tasks never write partial state directly to SQLite mid-flight. They accumulate changes in a `MemoryDelta` buffer and commit atomically at the end of the slice.
6. **Semantic Equivalence Filter for Human Diff Learning**:
   - Only human diffs that preserve business inputs/outputs (refactoring, naming, error handling, architecture layering, performance idioms) are extracted as candidate `StyleRule`s.
   - Rules reaching `hit_count >= 3` transition to `PENDING_APPROVAL` and require human confirmation (`fix rules --approve <id>`) before becoming hard review gates.
7. **Context-Isolated Independent Reviewer**:
   - `run_independent_review` runs in a clean LLM context with only `[user_prompt, git_diff, test_output, persona_rules, co_change_warnings]`, eliminating self-confirmation bias.
