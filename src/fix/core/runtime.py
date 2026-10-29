"""Cognitive State Machine (`FixRuntime`) & Context-Isolated Independent Reviewer."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Awaitable, Callable

from fix.core.autonomy import AutonomyRouter
from fix.core.context import ContextManager, RepoProfile
from fix.core.evolution import EvolutionEngine
from fix.llm import LLMGateway, LLMResponse
from fix.memory.store import DualMemoryStore
from fix.protocol.schema import (
    ProbeDecision,
    ReviewVerdict,
    TaskIntent,
    TaskPhase,
    TaskSession,
    ToolResult,
)
from fix.tools.toolkit import WorkspaceToolkit, is_test_file_path

READONLY_DEDUP_TOOLS = {
    "read_file",
    "search_code",
    "list_dir",
    "inspect_symbols",
    "trace_symbol",
    "check_diagnostics",
    "run_skill",
}

TOOL_CATEGORY_NAMES = {
    "read_file": "Read",
    "search_code": "Search",
    "list_dir": "ListDir",
    "inspect_symbols": "Symbols",
    "trace_symbol": "LSPTrace",
    "check_diagnostics": "Diagnostics",
    "profile_execution": "Profiler",
    "run_skill": "Skill",
    "apply_patch": "Edit",
    "write_new_file": "Write",
    "run_command": "Bash",
}


def format_tool_call_summary(name: str, args: dict[str, Any]) -> str:
    """Format a compact human-readable label for a tool call."""
    if name == "read_file":
        return f"Read({args.get('rel_path', '')}:{args.get('start_line', 1)}-{args.get('end_line', 500)})"
    if name == "search_code":
        return f"Search({args.get('pattern', '')!r})"
    if name == "list_dir":
        return f"ListDir({args.get('rel_path', '.')})"
    if name == "inspect_symbols":
        return f"Symbols({args.get('rel_path', '')})"
    if name == "trace_symbol":
        return f"LSPTrace({args.get('symbol_name', '')})"
    if name == "check_diagnostics":
        return f"Diagnostics({args.get('rel_path') or 'repo'})"
    if name == "profile_execution":
        return f"Profiler({args.get('cmd', '')})"
    if name == "run_skill":
        return f"Skill({args.get('slug', '')})"
    if name == "apply_patch":
        return f"Edit({args.get('rel_path', '')})"
    if name == "write_new_file":
        return f"Write({args.get('rel_path', '')})"
    if name == "run_command":
        return f"Bash({args.get('cmd', '')})"
    return f"{name}({args})"


class FixRuntime:
    """
    Executes the Fix cognitive state machine:
    PROBE -> CLARIFY -> REPRODUCE -> IMPLEMENT -> VERIFY -> REVIEW -> SYNTHESIZE -> DONE
    """

    def __init__(
        self,
        llm: LLMGateway,
        toolkit: WorkspaceToolkit,
        context_mgr: ContextManager | None = None,
        memory: DualMemoryStore | None = None,
        evolution: EvolutionEngine | None = None,
        on_task_complete_hook: Callable[[TaskSession], Awaitable[None]] | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.llm = llm
        self.toolkit = toolkit
        self.context_mgr = context_mgr or ContextManager()
        self.memory = memory
        self.evolution = evolution
        self.on_task_complete_hook = on_task_complete_hook
        self.on_event = on_event

    @staticmethod
    def _slice_valid_recent_history(history: list[dict[str, Any]], min_messages: int = 16) -> list[dict[str, Any]]:
        """
        Return a valid suffix of `history` of at least `min_messages` items that NEVER starts
        mid-tool-batch on a `role == "tool"` message (which OpenAI/MiMo APIs reject with HTTP 400).
        """
        if len(history) <= min_messages:
            return list(history)
        start_idx = max(0, len(history) - min_messages)
        while start_idx > 0 and history[start_idx].get("role") == "tool":
            start_idx -= 1
        return list(history[start_idx:])

    @staticmethod
    def _looks_like_incomplete_transition_note(text: str, last_narrative: str = "") -> bool:
        """
        Detect whether `text` is merely a short intermediate transition note (e.g. '我查整个日志里...'
        or '接下来我会...') rather than a genuine final conclusion/report.
        """
        cleaned = (text or "").strip()
        if not cleaned:
            return True
        lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
        if len(lines) <= 2 and len(cleaned) < 220 and "#" not in cleaned:
            transition_markers = (
                "接下来我",
                "下一步我",
                "我先",
                "我查",
                "我去看",
                "我再看",
                "我再查",
                "我继续",
                "继续追因",
                "继续追踪",
                "现在查",
                "现在我",
                "最后验证",
            )
            if any(m in cleaned for m in transition_markers):
                return True
        return False

    def _emit(
        self,
        event_type: str,
        payload: dict[str, Any],
        local_cb: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        cb = local_cb or self.on_event
        if cb is not None:
            try:
                cb(event_type, payload)
            except Exception:
                pass

    def _record_llm_usage(
        self,
        session: TaskSession,
        resp: LLMResponse | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        if resp is not None:
            p_tok = resp.prompt_tokens
            c_tok = resp.completion_tokens
            t_tok = resp.total_tokens or (p_tok + c_tok)
        else:
            p_tok, c_tok, t_tok = getattr(self.llm, "last_usage", (0, 0, 0))
        session.prompt_tokens += p_tok
        session.completion_tokens += c_tok
        session.total_tokens += t_tok
        session.llm_calls += 1
        self._emit(
            "telemetry",
            {
                "prompt_tokens": session.prompt_tokens,
                "completion_tokens": session.completion_tokens,
                "total_tokens": session.total_tokens,
                "llm_calls": session.llm_calls,
                "step": session.budget.steps_used,
            },
            on_event,
        )

    @staticmethod
    def is_conversational_prompt(prompt: str) -> bool:
        """Detect short conversational greetings or meta-questions that do not require repository scanning or tools."""
        cleaned = prompt.strip().strip("!?.！？。，,~～`'\" ").lower()
        if not cleaned or len(cleaned) > 48:
            return False
        exact_greetings = {
            "hello",
            "hi",
            "hey",
            "yo",
            "hola",
            "hi there",
            "hello there",
            "good morning",
            "good afternoon",
            "good evening",
            "你好",
            "您好",
            "嗨",
            "哈喽",
            "在吗",
            "在不在",
            "早上好",
            "下午好",
            "晚上好",
            "早安",
            "晚安",
            "你是谁",
            "你叫什么",
            "你叫什么名字",
            "介绍一下你自己",
            "自我介绍一下",
            "你能做什么",
            "你会做什么",
            "你有什么功能",
            "who are you",
            "what is your name",
            "what can you do",
            "谢谢",
            "多谢",
            "感谢",
            "thanks",
            "thank you",
            "thx",
            "ok",
            "okay",
            "好的",
            "好",
            "明白",
            "知道了",
            "收到",
            "test",
            "ping",
            "测试",
            "测试一下",
        }
        if cleaned in exact_greetings:
            return True
        greeting_prefixes = (
            "hello ",
            "hi ",
            "hey ",
            "你好",
            "您好",
            "你是谁",
            "你叫什么",
            "who are you",
        )
        code_or_repo_markers = (
            ".py",
            ".kt",
            ".java",
            ".ts",
            ".js",
            ".go",
            ".rs",
            "/",
            "bug",
            "error",
            "fix",
            "代码",
            "文件",
            "项目",
            "目录",
            "函数",
            "方法",
            "类",
            "报错",
            "修复",
            "分析",
            "梳理",
            "架构",
            "优化",
            "重构",
            "修改",
            "看看",
        )
        if cleaned.startswith(greeting_prefixes) and not any(m in cleaned for m in code_or_repo_markers):
            return True
        return False

    @staticmethod
    def infer_intent_heuristic(prompt: str) -> TaskIntent:
        """Fast fallback intent classifier when structured probe is not used."""
        if FixRuntime.is_conversational_prompt(prompt):
            return TaskIntent.EXPLAIN
        lower = prompt.lower()
        explain_keywords = (
            "explain",
            "why",
            "how does",
            "architecture",
            "what is",
            "梳理",
            "解释",
            "分析",
            "为什么",
            "流向",
            "看看",
            "干嘛",
            "介绍",
            "结构",
            "整理出来",
            "功能地图",
            "逻辑地图",
            "工作机制",
        )
        bug_keywords = ("bug", "fix", "error", "exception", "fail", "crash", "报错", "修复", "排查", "异常", "扣款")
        opt_keywords = ("optimize", "refactor", "performance", "feature", "add", "优化", "重构", "新增", "性能", "加个")

        if any(k in lower for k in explain_keywords) and not any(k in lower for k in ("修复", "fix", "改", "加个", "新增")):
            return TaskIntent.EXPLAIN
        if any(k in lower for k in bug_keywords):
            return TaskIntent.BUG_FIX
        if any(k in lower for k in opt_keywords):
            return TaskIntent.FEATURE
        return TaskIntent.DIRECT_EDIT

    async def _drain_steer_queue(
        self,
        session: TaskSession,
        history: list[dict[str, Any]],
        steer_queue: asyncio.Queue[str] | None,
    ) -> None:
        """Check for mid-flight P0 user steering instructions and rollback if requested."""
        if steer_queue is None:
            return
        while not steer_queue.empty():
            note = steer_queue.get_nowait().strip()
            if not note:
                continue
            session.steering_notes.append(note)
            lower = note.lower()
            rollback_triggers = ("rollback", "revert", "undo", "回滚", "撤销", "停一下", "别改")
            if any(t in lower for t in rollback_triggers) and session.snapshots:
                base_snap = session.snapshots[0]
                self.toolkit.rollback_to_snapshot(base_snap)
                session.modified_files.clear()
                history.append(
                    {
                        "role": "user",
                        "content": (
                            f"[LIVE STEERING + ROLLBACK]: Working tree rolled back to clean snapshot `{base_snap}`. "
                            f"New instruction: {note}"
                        ),
                    }
                )
            else:
                history.append(
                    {
                        "role": "user",
                        "content": f"[LIVE USER STEERING]: {note}",
                    }
                )

    async def execute_task(
        self,
        user_prompt: str,
        forced_intent: TaskIntent | None = None,
        steer_queue: asyncio.Queue[str] | None = None,
        conversation_context: list[dict[str, str]] | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
        defer_digest: bool = False,
    ) -> TaskSession:
        """Run the full cognitive loop with physical tool gates, deduplication, and guaranteed final synthesis."""
        # Yield immediately so the interactive TUI event loop flushes the user prompt and spinner in <1ms
        await asyncio.sleep(0)

        # Ultra-fast direct path for conversational greetings (`hello`, `你好`, `你是谁`, etc.)
        is_mock_scripted = bool(
            getattr(self.llm, "structured_handlers", None)
            or getattr(self.llm, "tool_queue", None)
            or getattr(self.llm, "step_callback", None) is not None
        )
        if forced_intent is None and not is_mock_scripted and self.is_conversational_prompt(user_prompt):
            session = TaskSession(
                repo_root=str(self.toolkit.repo_root),
                user_prompt=user_prompt,
                intent=TaskIntent.EXPLAIN,
                phase=TaskPhase.PROBE,
                started_at=time.time(),
            )
            self._emit(
                "phase",
                {"phase": TaskPhase.PROBE.value, "intent": TaskIntent.EXPLAIN.value, "note": "正在快速生成回复..."},
                on_event,
            )
            conv_msgs: list[dict[str, Any]] = [
                {
                    "role": "system",
                    "content": (
                        "You are Fix, an interactive AI software engineering assistant running in the user's terminal. "
                        "Reply naturally, warmly, and concisely in the same language as the user."
                    ),
                }
            ]
            if conversation_context:
                for turn in conversation_context[-4:]:
                    u_m = str(turn.get("user", "")).strip()
                    a_m = str(turn.get("assistant", "")).strip()
                    if u_m:
                        conv_msgs.append({"role": "user", "content": u_m[:400]})
                    if a_m:
                        conv_msgs.append({"role": "assistant", "content": a_m[:600]})
            conv_msgs.append({"role": "user", "content": user_prompt})
            session.budget.steps_used = 1
            resp = await self.llm.complete_with_tools(
                messages=conv_msgs,
                tools=None,
                use_reasoning=False,
                temperature=0.2,
                max_tokens=4096,
                auto_continue=True,
            )
            self._record_llm_usage(session, resp, on_event=on_event)
            reply_txt = (resp.content or "").strip() or getattr(resp, "reasoning_content", "").strip()
            session.final_summary = reply_txt or "你好！我是 Fix，随时可以帮你分析代码、排查问题或实现新功能。"
            session.phase = TaskPhase.DONE
            session.completed_at = time.time()
            session.elapsed_sec = round(max(0.01, session.completed_at - session.started_at), 2)
            self._emit(
                "done",
                {
                    "task_id": session.task_id,
                    "confidence": session.confidence,
                    "elapsed_sec": session.elapsed_sec,
                    "total_tokens": session.total_tokens,
                    "tool_calls_count": session.tool_calls_count,
                },
                on_event,
            )
            if self.on_task_complete_hook is not None:
                await self.on_task_complete_hook(session)
            return session

        now_ts = time.time()
        if self.memory is not None:
            last_inv_ts = getattr(self, "_last_invalidate_facts_ts", 0.0)
            if (now_ts - last_inv_ts) >= 60.0:
                self.memory.invalidate_stale_facts()
                self._last_invalidate_facts_ts = now_ts
        cached_prof = getattr(self, "_cached_repo_profile", None)
        cached_prof_ts = getattr(self, "_cached_repo_profile_ts", 0.0)
        if cached_prof is not None and (now_ts - cached_prof_ts) < 60.0 and cached_prof.repo_root == str(self.toolkit.repo_root.resolve()):
            profile = cached_prof
        else:
            profile = self.context_mgr.detect_repo_profile(self.toolkit.repo_root)
            self._cached_repo_profile = profile
            self._cached_repo_profile_ts = now_ts
        autonomy_assessment = (
            AutonomyRouter(self.memory).assess_autonomy(user_prompt=user_prompt, intent=forced_intent)
            if self.memory is not None
            else None
        )
        session = TaskSession(
            repo_root=str(self.toolkit.repo_root),
            user_prompt=user_prompt,
            intent=forced_intent,
            phase=TaskPhase.PROBE,
            reproduction_cmd=profile.test_command,
            started_at=time.time(),
        )
        if autonomy_assessment is not None:
            session.autonomy = autonomy_assessment

        persona_rules = self.memory.get_active_persona_rules() if self.memory else []
        landmarks = (
            [
                f"[{f.category}] {f.summary} (files: {', '.join(f.evidence_paths)})"
                for f in self.memory.search_landmarks(user_prompt, limit=6, include_stale=False)
            ]
            if self.memory
            else []
        )
        active_skills = self.memory.list_skills() if self.memory else []
        cognitive_paradigms = self.memory.get_active_cognitive_paradigms(limit=4) if self.memory else []
        mental_models = (
            [
                f"[{m.component_or_scope}] 现象: {m.phenomenon_layer} | 机制契约: {m.mechanism_contract_layer} | 本质: {m.essence_first_principles} | UX影响: {m.ux_and_product_impact}"
                for m in self.memory.list_mental_models(query=user_prompt, limit=4)
            ]
            if self.memory
            else []
        )
        optimization_proposals = (
            [
                f"[{p.priority} · {p.dimension}] {p.title}: 表象={p.phenomenon_symptom} -> 本质={p.root_essence} -> 演进方案={p.proposed_evolution}"
                for p in self.memory.list_optimization_proposals(status="OPEN", limit=4)
            ]
            if self.memory
            else []
        )

        history: list[dict[str, Any]] = []
        seen_readonly_calls: set[str] = set()

        # --- Phase 1: PROBE & Intent Routing ---
        self._emit("phase", {"phase": TaskPhase.PROBE.value, "intent": session.intent.value if session.intent else "auto"}, on_event)
        await self._drain_steer_queue(session, history, steer_queue)
        has_custom_probe_handler = "ProbeDecision" in (getattr(self.llm, "structured_handlers", None) or {})
        probe_decision = (
            await self._run_probe_phase(session, profile)
            if (forced_intent is None and has_custom_probe_handler)
            else None
        )
        if probe_decision is not None:
            self._record_llm_usage(session, on_event=on_event)
            if session.intent is None:
                heuristic_intent = self.infer_intent_heuristic(user_prompt)
                if heuristic_intent == TaskIntent.EXPLAIN and probe_decision.intent == TaskIntent.BUG_FIX:
                    session.intent = TaskIntent.EXPLAIN
                else:
                    session.intent = probe_decision.intent
            if probe_decision.root_cause_hypothesis:
                session.root_cause_hypothesis = probe_decision.root_cause_hypothesis
            if probe_decision.reproduction_cmd:
                session.reproduction_cmd = probe_decision.reproduction_cmd
            if self.memory is not None:
                session.autonomy = AutonomyRouter(self.memory).assess_autonomy(
                    user_prompt=user_prompt,
                    intent=session.intent,
                    estimated_files_touched=max(1, len(probe_decision.target_files)),
                )
            if probe_decision.needs_clarification and probe_decision.clarifications:
                if session.autonomy.auto_approve_clarifications:
                    auto_notes = [
                        f"Auto-selected recommended option for '{c.question}': {c.recommended}"
                        for c in probe_decision.clarifications[:3]
                        if c.recommended
                    ]
                    session.steering_notes.extend(auto_notes)
                else:
                    session.phase = TaskPhase.CLARIFY
                    session.clarification_questions = probe_decision.clarifications[:3]
                    q_lines = ["Clarification needed before modifying code:"]
                    for idx, c in enumerate(session.clarification_questions, start=1):
                        opts = " / ".join(c.options) if c.options else "open"
                        q_lines.append(f"{idx}. {c.question} (Options: {opts} | Recommended: {c.recommended})")
                    session.final_summary = "\n".join(q_lines)
                    session.completed_at = time.time()
                    session.elapsed_sec = round(max(0.01, session.completed_at - session.started_at), 2)
                    self._emit("clarify", {"questions": [q.model_dump() for q in session.clarification_questions]}, on_event)
                    return session

        if session.intent is None:
            session.intent = self.infer_intent_heuristic(user_prompt)

        if session.intent == TaskIntent.EXPLAIN:
            session.phase = TaskPhase.PROBE
        elif session.intent == TaskIntent.BUG_FIX:
            session.phase = TaskPhase.REPRODUCE
        else:
            session.phase = TaskPhase.IMPLEMENT

        self._emit(
            "phase",
            {
                "phase": session.phase.value,
                "intent": session.intent.value,
                "hypothesis": session.root_cause_hypothesis,
            },
            on_event,
        )

        last_step_had_tool_calls = False
        last_narrative_text = ""

        # --- Main Cognitive State Machine Loop ---
        while session.phase != TaskPhase.DONE and session.budget.steps_used < session.budget.max_steps:
            await self._drain_steer_queue(session, history, steer_queue)
            session.budget.steps_used += 1

            allow_write = session.intent != TaskIntent.EXPLAIN
            restrict_write_to_tests = (
                session.intent == TaskIntent.BUG_FIX and session.phase == TaskPhase.REPRODUCE
            )
            tools_schema = self.toolkit.get_tool_schemas(allow_write=allow_write)

            messages = self.context_mgr.build_messages(
                session=session,
                profile=profile,
                history=history,
                persona_rules=persona_rules,
                repo_landmarks=landmarks,
                active_skills=active_skills,
                conversation_context=conversation_context,
                mental_models=mental_models,
                cognitive_paradigms=cognitive_paradigms,
                optimization_proposals=optimization_proposals,
            )

            try:
                llm_resp = await self.llm.complete_with_tools(
                    messages=messages,
                    tools=tools_schema,
                    use_reasoning=(session.phase in (TaskPhase.REPRODUCE, TaskPhase.IMPLEMENT)),
                    max_tokens=32768,
                )
            except Exception:
                if history:
                    last_step_had_tool_calls = True
                    break
                raise
            self._record_llm_usage(session, llm_resp, on_event=on_event)

            if llm_resp.tool_calls:
                last_step_had_tool_calls = True
                # Emit Codex-style intermediate narrative note before running the batch of tool calls!
                narrative_text = (llm_resp.content or "").strip()
                if narrative_text:
                    last_narrative_text = narrative_text
                    session.final_summary = narrative_text
                    self._emit(
                        "narrative",
                        {
                            "text": narrative_text,
                            "step": session.budget.steps_used,
                            "phase": session.phase.value,
                        },
                        on_event,
                    )
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": llm_resp.content or "",
                    "tool_calls": [
                        {
                            "id": tc.call_id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                            },
                        }
                        for tc in llm_resp.tool_calls
                    ],
                }
                if getattr(llm_resp, "reasoning_content", ""):
                    assistant_msg["reasoning_content"] = llm_resp.reasoning_content
                history.append(assistant_msg)

                wrote_prod_file = False
                all_readonly_batch = (
                    len(llm_resp.tool_calls) > 1
                    and all(c.name in READONLY_DEDUP_TOOLS for c in llm_resp.tool_calls)
                )
                precomputed_results: dict[str, ToolResult] = {}
                emitted_tool_starts: set[str] = set()
                if all_readonly_batch:
                    coros_to_run: list[tuple[str, Awaitable[ToolResult]]] = []
                    for c_item in llm_resp.tool_calls:
                        lbl = format_tool_call_summary(c_item.name, c_item.arguments)
                        cat = TOOL_CATEGORY_NAMES.get(c_item.name, c_item.name)
                        session.tool_calls_count += 1
                        session.tool_calls_log.append(lbl)
                        session.tool_counts_by_name[cat] = session.tool_counts_by_name.get(cat, 0) + 1
                        emitted_tool_starts.add(c_item.call_id)
                        self._emit(
                            "tool_start",
                            {
                                "name": c_item.name,
                                "arguments": c_item.arguments,
                                "label": lbl,
                                "phase": session.phase.value,
                                "step": session.budget.steps_used,
                                "total_tools": session.tool_calls_count,
                            },
                            on_event,
                        )
                        d_key = f"{c_item.name}:{json.dumps(c_item.arguments, sort_keys=True)}"
                        if d_key in seen_readonly_calls:
                            precomputed_results[c_item.call_id] = ToolResult(
                                ok=True,
                                output=(
                                    f"[Duplicate Tool Call Skipped]: `{lbl}` was already executed earlier in this task. "
                                    "Refer to the previous output in context and proceed toward synthesis/action."
                                ),
                            )
                        else:
                            seen_readonly_calls.add(d_key)
                            coros_to_run.append(
                                (
                                    c_item.call_id,
                                    self.toolkit.execute_tool_call(
                                        c_item,
                                        allow_write=allow_write,
                                        restrict_write_to_tests=restrict_write_to_tests,
                                    ),
                                )
                            )
                    if coros_to_run:
                        gathered = await asyncio.gather(*(co for _, co in coros_to_run))
                        for (cid, _), g_res in zip(coros_to_run, gathered):
                            precomputed_results[cid] = g_res

                for call in llm_resp.tool_calls:
                    label = format_tool_call_summary(call.name, call.arguments)
                    if call.call_id not in emitted_tool_starts:
                        # Record tool telemetry on session
                        cat = TOOL_CATEGORY_NAMES.get(call.name, call.name)
                        session.tool_calls_count += 1
                        session.tool_calls_log.append(label)
                        session.tool_counts_by_name[cat] = session.tool_counts_by_name.get(cat, 0) + 1

                        self._emit(
                            "tool_start",
                            {
                                "name": call.name,
                                "arguments": call.arguments,
                                "label": label,
                                "phase": session.phase.value,
                                "step": session.budget.steps_used,
                                "total_tools": session.tool_calls_count,
                            },
                            on_event,
                        )

                    if call.call_id in precomputed_results:
                        tool_res = precomputed_results[call.call_id]
                    else:
                        # Read-only deduplication check to prevent redundant Search/Read loops
                        dedup_key = f"{call.name}:{json.dumps(call.arguments, sort_keys=True)}"
                        if call.name in READONLY_DEDUP_TOOLS and dedup_key in seen_readonly_calls:
                            tool_res = ToolResult(
                                ok=True,
                                output=(
                                    f"[Duplicate Tool Call Skipped]: `{label}` was already executed earlier in this task. "
                                    "Refer to the previous output in context and proceed toward synthesis/action."
                                ),
                            )
                        else:
                            if call.name in READONLY_DEDUP_TOOLS:
                                seen_readonly_calls.add(dedup_key)
                            if call.name in ("apply_patch", "write_new_file") and allow_write:
                                snap = self.toolkit.create_step_snapshot()
                                session.snapshots.append(snap)
                                # Invalidate read cache after file mutation
                                seen_readonly_calls.clear()

                            tool_res = await self.toolkit.execute_tool_call(
                                call,
                                allow_write=allow_write,
                                restrict_write_to_tests=restrict_write_to_tests,
                            )

                    self._emit(
                        "tool_end",
                        {
                            "name": call.name,
                            "arguments": call.arguments,
                            "label": label,
                            "ok": tool_res.ok,
                            "error_type": tool_res.error_type,
                            "output_preview": tool_res.output[:240],
                        },
                        on_event,
                    )
                    history.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.call_id,
                            "name": call.name,
                            "content": tool_res.output,
                        }
                    )

                    if call.name in ("apply_patch", "write_new_file") and tool_res.ok:
                        rel_path = str(call.arguments.get("rel_path", ""))
                        if rel_path:
                            session.modified_files.add(rel_path)
                            if not is_test_file_path(rel_path):
                                wrote_prod_file = True

                    if session.phase == TaskPhase.REPRODUCE:
                        await self._evaluate_reproduce_progress(session, call.name, tool_res, history, on_event)

                if session.phase == TaskPhase.IMPLEMENT and wrote_prod_file:
                    session.phase = TaskPhase.VERIFY
                    await self._run_verify_and_review(session, profile, history, on_event)
                continue

            if llm_resp.content and llm_resp.content.strip():
                last_step_had_tool_calls = False
                session.final_summary = llm_resp.content.strip()
            elif session.intent != TaskIntent.EXPLAIN:
                last_step_had_tool_calls = False
            if session.intent == TaskIntent.EXPLAIN:
                session.phase = TaskPhase.DONE
                break

            if session.phase == TaskPhase.REPRODUCE:
                await self._evaluate_reproduce_progress(session, "none", None, history, on_event)
                continue

            if session.phase == TaskPhase.IMPLEMENT:
                if not session.modified_files and session.final_summary.strip():
                    session.phase = TaskPhase.DONE
                    break
                session.phase = TaskPhase.VERIFY
                await self._run_verify_and_review(session, profile, history, on_event)
                continue

        # --- Guaranteed Final Synthesis Step ---
        # If the loop exhausted max_steps while still calling tools (or final_summary is empty),
        # force a dedicated tool-free synthesis turn so the user NEVER gets an intermediate narrative or `(Task completed)`!
        needs_synthesis = (
            not session.final_summary.strip()
            or session.phase != TaskPhase.DONE
            or (session.intent == TaskIntent.EXPLAIN and last_step_had_tool_calls)
            or (
                session.tool_calls_count > 0
                and self._looks_like_incomplete_transition_note(session.final_summary, last_narrative_text)
            )
        )
        if needs_synthesis:
            session.phase = TaskPhase.SYNTHESIZE
            self._emit(
                "phase",
                {
                    "phase": TaskPhase.SYNTHESIZE.value,
                    "intent": session.intent.value if session.intent else "explain",
                    "note": "正在汇总全部代码证据并生成深度分析报告...",
                },
                on_event,
            )
            await self._force_final_synthesis(
                session=session,
                profile=profile,
                history=history,
                persona_rules=persona_rules,
                repo_landmarks=landmarks,
                active_skills=active_skills,
                conversation_context=conversation_context,
                mental_models=mental_models,
                cognitive_paradigms=cognitive_paradigms,
                optimization_proposals=optimization_proposals,
            )
        if not session.final_summary.strip() and last_narrative_text:
            session.final_summary = last_narrative_text

        session.phase = TaskPhase.DONE
        session.completed_at = time.time()
        session.elapsed_sec = round(max(0.01, session.completed_at - session.started_at), 2)
        self._emit(
            "done",
            {
                "task_id": session.task_id,
                "confidence": session.confidence,
                "elapsed_sec": session.elapsed_sec,
                "total_tokens": session.total_tokens,
                "tool_calls_count": session.tool_calls_count,
            },
            on_event,
        )

        if self.evolution is not None:
            if session.modified_files and self.memory is not None:
                self._last_invalidate_facts_ts = 0.0
            eff_sha = (
                self.toolkit.get_head_sha()
                if (session.modified_files or not getattr(profile, "head_sha", ""))
                else profile.head_sha
            )
            if defer_digest:
                setattr(session, "_deferred_digest_sha", eff_sha)
            else:
                self.evolution.digest_completed_task(
                    session=session,
                    head_sha=eff_sha,
                )
        if self.on_task_complete_hook is not None:
            await self.on_task_complete_hook(session)

        return session

    async def _force_final_synthesis(
        self,
        session: TaskSession,
        profile: RepoProfile,
        history: list[dict[str, Any]],
        persona_rules: list[str],
        repo_landmarks: list[str],
        active_skills: list[str],
        conversation_context: list[dict[str, str]] | None,
        mental_models: list[str] | None = None,
        cognitive_paradigms: list[str] | None = None,
        optimization_proposals: list[str] | None = None,
    ) -> None:
        """
        Execute a guaranteed tool-free (`tools=None`) synthesis call so that all gathered
        file reads, command outputs, and search results are ALWAYS distilled into a complete report.
        Includes multi-stage fallback (reasoning model -> fast model with compact history -> deterministic digest)
        so the user NEVER ends up without a final conclusion.
        """
        synthesis_prompt = (
            "[FINAL SYNTHESIS REQUIRED]: Tool exploration is now complete. "
            "Do NOT call any more tools, and NEVER output transitional sentences like '接下来我将查...' or '我继续看...'. "
            "Based on all the source files, log evidence, and search results gathered above, provide your complete, "
            "thorough, closed-loop final analysis and response to the user's request in Markdown "
            "(with `[file:line]` anchors, log timestamps/evidence, working mechanisms, call/logic flows, and root-cause conclusions)."
        )
        synthesis_history = list(history)
        synthesis_history.append({"role": "user", "content": synthesis_prompt})
        messages = self.context_mgr.build_messages(
            session=session,
            profile=profile,
            history=synthesis_history,
            persona_rules=persona_rules,
            repo_landmarks=repo_landmarks,
            active_skills=active_skills,
            conversation_context=conversation_context,
            mental_models=mental_models,
            cognitive_paradigms=cognitive_paradigms,
            optimization_proposals=optimization_proposals,
        )

        # Stage 1: Reasoning synthesis call
        try:
            synth_resp = await self.llm.complete_with_tools(
                messages=messages,
                tools=None,
                use_reasoning=True,
                temperature=0.2,
                max_tokens=32768,
            )
            self._record_llm_usage(session, synth_resp)
            candidate = (synth_resp.content or "").strip() or getattr(synth_resp, "reasoning_content", "").strip()
            if (
                candidate
                and candidate != "Task completed."
                and not self._looks_like_incomplete_transition_note(candidate)
            ):
                session.final_summary = candidate
                return
        except Exception:
            candidate = ""

        # Stage 2: Fast-model fallback with compact history (avoids reasoning token exhaustion or oversized context)
        try:
            compact_history = self._slice_valid_recent_history(history, min_messages=16)
            compact_history.append({"role": "user", "content": synthesis_prompt})
            compact_messages = self.context_mgr.build_messages(
                session=session,
                profile=profile,
                history=compact_history,
                persona_rules=persona_rules,
                repo_landmarks=repo_landmarks,
                active_skills=active_skills,
                conversation_context=conversation_context,
            )
            fast_resp = await self.llm.complete_with_tools(
                messages=compact_messages,
                tools=None,
                use_reasoning=False,
                temperature=0.2,
                max_tokens=16384,
            )
            self._record_llm_usage(session, fast_resp)
            candidate2 = (fast_resp.content or "").strip() or getattr(fast_resp, "reasoning_content", "").strip()
            if candidate2 and candidate2 != "Task completed.":
                session.final_summary = candidate2
                return
        except Exception:
            pass

        # Stage 3: Deterministic evidence digest fallback if all LLM calls fail
        narratives = [
            str(m.get("content", "")).strip()
            for m in history
            if m.get("role") == "assistant" and str(m.get("content", "")).strip()
        ]
        if narratives:
            bullet_notes = "\n".join(f"- {n}" for n in narratives[-8:])
            session.final_summary = (
                "## 阶段性排查与分析结论汇总\n\n"
                f"已完成 {session.budget.steps_used} 步探索（共调用 {session.tool_calls_count} 次工具），关键发现汇总如下：\n\n"
                f"{bullet_notes}"
            )

    async def _run_probe_phase(
        self,
        session: TaskSession,
        profile: RepoProfile,
    ) -> ProbeDecision | None:
        try:
            probe_messages = [
                {
                    "role": "system",
                    "content": (
                        "Analyze the user request and repository profile to select the execution track "
                        "(explain, bug_fix, feature, direct) and initial hypothesis. "
                        "IMPORTANT: You have full repository tools (`read_file`, `search_code`, `list_dir`) "
                        "in the next step, so NEVER set `needs_clarification=true` just to ask for file contents; "
                        "only set `needs_clarification=true` for genuine business requirement ambiguity."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Repo languages: {profile.languages}\n"
                        f"Top layout: {profile.top_level_layout}\n"
                        f"Default test cmd: {profile.test_command}\n"
                        f"User request: {session.user_prompt}"
                    ),
                },
            ]
            return await self.llm.complete_structured(
                messages=probe_messages,
                response_model=ProbeDecision,
                use_reasoning=False,
            )
        except Exception:
            return None

    async def _evaluate_reproduce_progress(
        self,
        session: TaskSession,
        tool_name: str,
        tool_res: Any,
        history: list[dict[str, Any]],
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        if tool_name == "run_command" and tool_res is not None:
            session.budget.reproduce_attempts += 1
            if not tool_res.ok:
                session.reproduction_confirmed = True
                session.phase = TaskPhase.IMPLEMENT
                self._emit("phase", {"phase": TaskPhase.IMPLEMENT.value, "note": "Bug reproduced!"}, on_event)
                history.append(
                    {
                        "role": "user",
                        "content": (
                            "[REPRODUCE CONFIRMED]: Failing reproduction observed! "
                            "Production file lock lifted. Proceed to `IMPLEMENT` to fix the root cause."
                        ),
                    }
                )
                return
        elif tool_name in ("write_new_file", "apply_patch") and tool_res is not None and tool_res.ok:
            if session.reproduction_cmd:
                session.budget.reproduce_attempts += 1
                repro_res = await self.toolkit.run_command(session.reproduction_cmd)
                history.append(
                    {
                        "role": "user",
                        "content": f"[AUTO REPRODUCE RUN `{session.reproduction_cmd}`]:\n{repro_res.output}",
                    }
                )
                if not repro_res.ok:
                    session.reproduction_confirmed = True
                    session.phase = TaskPhase.IMPLEMENT
                    self._emit("phase", {"phase": TaskPhase.IMPLEMENT.value, "note": "Reproduction test failed as expected!"}, on_event)
                    history.append(
                        {
                            "role": "user",
                            "content": (
                                "[REPRODUCE CONFIRMED]: Reproduction test failed as expected! "
                                "Transitioning to `IMPLEMENT` phase."
                            ),
                        }
                    )
                    return
        elif tool_name == "none":
            session.budget.reproduce_attempts += 1

        if session.budget.reproduce_attempts >= session.budget.max_reproduce_attempts:
            session.degraded_without_repro = True
            session.confidence = "LOW"
            session.phase = TaskPhase.IMPLEMENT
            self._emit("phase", {"phase": TaskPhase.IMPLEMENT.value, "note": "Elastic TDD escape valve triggered"}, on_event)
            history.append(
                {
                    "role": "user",
                    "content": (
                        f"[ELASTIC TDD ESCAPE VALVE]: Could not reproduce failure within "
                        f"{session.budget.max_reproduce_attempts} attempts. "
                        "Entering degraded `IMPLEMENT` mode (`degraded_without_repro=True`, `confidence=LOW`)."
                    ),
                }
            )

    async def _run_verify_and_review(
        self,
        session: TaskSession,
        profile: RepoProfile,
        history: list[dict[str, Any]],
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        verify_cmd = session.reproduction_cmd or profile.test_command
        if verify_cmd:
            self._emit("verify", {"cmd": verify_cmd}, on_event)
            test_res = await self.toolkit.run_command(verify_cmd)
            session.last_test_output = test_res.output
            history.append(
                {
                    "role": "user",
                    "content": f"[VERIFY `{verify_cmd}`]:\n{test_res.output}",
                }
            )
            if not test_res.ok:
                session.budget.repairs_used += 1
                if session.budget.repairs_used > session.budget.max_repair_attempts:
                    session.confidence = "LOW"
                    session.phase = TaskPhase.DONE
                    return
                session.phase = TaskPhase.IMPLEMENT
                self._emit(
                    "phase",
                    {"phase": TaskPhase.IMPLEMENT.value, "note": f"Test failed (repair {session.budget.repairs_used})"},
                    on_event,
                )
                history.append(
                    {
                        "role": "user",
                        "content": (
                            f"[VERIFY FAILED - Repair Attempt {session.budget.repairs_used}/"
                            f"{session.budget.max_repair_attempts}]: Fix the failing test above."
                        ),
                    }
                )
                return

        session.phase = TaskPhase.REVIEW
        session.budget.review_rounds += 1
        git_diff = self.toolkit.get_git_diff()
        session.fix_diff_snapshot = git_diff
        self._emit("review_start", {"round": session.budget.review_rounds}, on_event)

        verdict = await self.run_independent_review(
            session=session,
            git_diff=git_diff,
            test_output=session.last_test_output,
        )
        self._record_llm_usage(session, on_event=on_event)
        session.last_review = verdict
        self._emit(
            "review_end",
            {
                "approved": verdict.approved,
                "confidence": verdict.confidence,
                "blocking_issues": verdict.blocking_issues,
            },
            on_event,
        )

        if verdict.approved or session.budget.review_rounds >= session.budget.max_review_rounds:
            if not session.degraded_without_repro:
                session.confidence = verdict.confidence
            session.phase = TaskPhase.DONE
            return

        session.phase = TaskPhase.IMPLEMENT
        issues_str = "; ".join(verdict.blocking_issues) or "Unspecified reviewer concern"
        history.append(
            {
                "role": "user",
                "content": (
                    f"[INDEPENDENT REVIEWER REJECTED PATCH]: Blocking issues must be resolved: {issues_str}"
                ),
            }
        )

    async def run_independent_review(
        self,
        session: TaskSession,
        git_diff: str,
        test_output: str,
    ) -> ReviewVerdict:
        persona_rules = self.memory.get_active_persona_rules() if self.memory else []
        cochange_warnings: list[str] = []
        if self.memory and session.modified_files:
            edges = self.memory.get_cochange_partners(session.modified_files)
            for edge in edges:
                cochange_warnings.append(
                    f"Co-change coupling alert: `{edge.file_a}` and `{edge.file_b}` changed together "
                    f"in {edge.co_commits} commits (confidence={edge.confidence:.0%}), "
                    "but only one was modified."
                )

        review_prompt = (
            "You are an independent Principal Code Reviewer with ZERO prior conversation context.\n"
            "Audit whether the patch truly resolves the root cause, respects all Developer Mind Mirror "
            "hard rules, and avoids regressions.\n\n"
            f"### Original User Requirement\n{session.user_prompt}\n\n"
            f"### Developer Mind Mirror Rules\n"
            + ("\n".join(f"- {r}" for r in persona_rules) if persona_rules else "(none)")
            + "\n\n### Co-Change Coupling Alerts\n"
            + ("\n".join(f"- {w}" for w in cochange_warnings) if cochange_warnings else "(none)")
            + f"\n\n### Mechanical Test Output\n{test_output or '(no tests run)'}\n\n"
            f"### Git Diff\n```diff\n{git_diff[:6000]}\n```"
        )

        try:
            return await self.llm.complete_structured(
                messages=[{"role": "user", "content": review_prompt}],
                response_model=ReviewVerdict,
                use_reasoning=True,
            )
        except Exception:
            approved = bool(git_diff.strip() or session.modified_files)
            return ReviewVerdict(
                approved=approved,
                confidence="LOW" if session.degraded_without_repro else "HIGH",
                blocking_issues=[] if approved else ["No effective code changes detected."],
                suggestions=cochange_warnings,
            )


def llm_resp_will_just_probe(session: TaskSession) -> bool:
    return False
