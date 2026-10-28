"""Interactive TUI modals (`/connect`, `/model`, `/resume`) and stage-2 slash option builders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fix.core.autonomy import AutonomyRouter
from fix.llm.catalog import PROVIDER_ORDER, PROVIDER_SPECS, resolve_provider_id
from rich.markup import escape

from fix.tui.composer import SLASH_COMMAND_SPEC_MAP, SLASH_COMMAND_SPECS
from fix.tui.theme import _truncate_middle_for_width


class ModalControllerMixin:
    """Mixin providing interactive keyboard/mouse modals and two-stage slash command pickers."""

    def open_connect_modal(self) -> None:
        """Open the 3-step interactive `/connect` popup modal in Full-Screen TUI mode."""
        self._slash_popup_dismissed = True
        self._slash_stage2_cmd = ""
        self._shortcut_overlay_open = False
        if self._input_buffer_ref is not None:
            try:
                self._input_buffer_ref.cancel_completion()
                self._input_buffer_ref.reset()
            except Exception:
                pass
        providers = self.llm.list_supported_providers()
        active_idx = next((i for i, p in enumerate(providers) if p.get("active")), 0)
        self._modal_state = {
            "mode": "connect_provider",
            "providers": providers,
            "selected_idx": active_idx,
            "provider_id": providers[active_idx]["provider_id"] if providers else "deepseek",
            "filter_query": "",
            "api_key_input": "",
            "error_msg": "",
            "status_msg": "",
            "models": [],
        }
        self._scroll_offset_from_bottom = 0
        self._set_terminal_hover_mouse_tracking(True)
        self.show_toast("↑↓/鼠标悬停或输入关键词选择厂商 · Enter/点击确认 · Esc 取消", duration_sec=3.0)
        if self._tui_app is not None:
            try:
                self._tui_app.invalidate()
            except Exception:
                pass

    def open_model_selector_modal(self) -> None:
        """Open the interactive `/model` model-switching popup modal in Full-Screen TUI mode."""
        self._slash_popup_dismissed = True
        self._slash_stage2_cmd = ""
        self._shortcut_overlay_open = False
        if self._input_buffer_ref is not None:
            try:
                self._input_buffer_ref.cancel_completion()
                self._input_buffer_ref.reset()
            except Exception:
                pass
        models = self.llm.list_switchable_models()
        cur_idx = next((i for i, m in enumerate(models) if m.get("is_current")), 0)
        self._modal_state = {
            "mode": "switch_model",
            "models": models,
            "selected_idx": cur_idx,
            "filter_query": "",
            "error_msg": "",
            "status_msg": "",
        }
        self._scroll_offset_from_bottom = 0
        self._set_terminal_hover_mouse_tracking(True)
        self.show_toast("↑↓/鼠标悬停或输入关键词选择模型 · Enter/点击切换 · Esc 关闭", duration_sec=3.0)
        if self._tui_app is not None:
            try:
                self._tui_app.invalidate()
            except Exception:
                pass

    def _apply_modal_filter_query(self, raw_query: str) -> int | None:
        """
        Live-filter and jump `selected_idx` in the active `/resume`, `/model`, or `/connect` modal
        as the user types in the input box (`input_window`).
        """
        if self._modal_state is None:
            return None
        mode = self._modal_state.get("mode")
        q = raw_query.strip()
        self._modal_state["filter_query"] = q
        if not q:
            return int(self._modal_state.get("selected_idx", 0))
        q_low = q.lower()

        if mode == "connect_provider":
            providers = self._modal_state.get("providers") or self.llm.list_supported_providers()
            if not providers:
                return None
            if q_low.lstrip("#").isdigit():
                idx = max(0, min(len(providers) - 1, int(q_low.lstrip("#")) - 1))
                self._modal_state["selected_idx"] = idx
                return idx
            pid = resolve_provider_id(q_low)
            for idx, p in enumerate(providers):
                if (
                    (pid and p.get("provider_id") == pid)
                    or q_low in str(p.get("provider_id", "")).lower()
                    or q_low in str(p.get("display_name", "")).lower()
                    or q_low in str(p.get("vendor_label", "")).lower()
                    or q_low in str(p.get("base_url", "")).lower()
                ):
                    self._modal_state["selected_idx"] = idx
                    return idx
            return int(self._modal_state.get("selected_idx", 0))

        if mode == "connect_select_model":
            models = self._modal_state.get("models") or []
            if not models:
                return None
            if q_low.lstrip("#").isdigit():
                idx = max(0, min(len(models) - 1, int(q_low.lstrip("#")) - 1))
                self._modal_state["selected_idx"] = idx
                return idx
            for idx, m_name in enumerate(models):
                if q_low in str(m_name).lower():
                    self._modal_state["selected_idx"] = idx
                    return idx
            return int(self._modal_state.get("selected_idx", 0))

        if mode == "switch_model":
            models = self._modal_state.get("models") or []
            total = len(models) + 1
            if q_low.lstrip("#").isdigit():
                idx = max(0, min(total - 1, int(q_low.lstrip("#")) - 1))
                self._modal_state["selected_idx"] = idx
                return idx
            if any(k in q_low for k in ("connect", "连接", "其他", "新厂商", "+")):
                self._modal_state["selected_idx"] = len(models)
                return len(models)
            for idx, entry in enumerate(models):
                if (
                    q_low in str(entry.get("model", "")).lower()
                    or q_low in str(entry.get("provider_name", "")).lower()
                    or q_low in str(entry.get("vendor_label", "")).lower()
                ):
                    self._modal_state["selected_idx"] = idx
                    return idx
            return int(self._modal_state.get("selected_idx", 0))

        if mode == "resume_session":
            sessions = self._modal_state.get("sessions") or self.list_saved_sessions()
            if not sessions:
                return None
            if q_low.lstrip("#").isdigit():
                idx = max(0, min(len(sessions) - 1, int(q_low.lstrip("#")) - 1))
                self._modal_state["selected_idx"] = idx
                return idx
            for idx, s_item in enumerate(sessions[:15]):
                if (
                    q_low in str(s_item.get("latest_prompt", "")).lower()
                    or q_low in str(s_item.get("title", "")).lower()
                    or q_low in str(s_item.get("session_id", "")).lower()
                    or q_low in str(s_item.get("updated_at_display", "")).lower()
                ):
                    self._modal_state["selected_idx"] = idx
                    return idx
            return int(self._modal_state.get("selected_idx", 0))

        return None

    def match_slash_commands(self, query_token: str = "/") -> list[dict[str, Any]]:
        """
        Claude Code style multi-dimension slash command matcher:
        Matches by command prefix (`/re`), substring (`/mod`), alias (`/q`), OR Chinese/English
        keywords (`/恢复`, `/模型`, `/新建`, `/连接`, `/回滚`, `/规则`, `/技能`), sorted by relevance.
        """
        raw = query_token.strip()
        if not raw.startswith("/"):
            return []
        q = raw[1:].strip().lower()
        if not q:
            return [dict(spec) for spec in SLASH_COMMAND_SPECS if spec.get("canonical", True)]

        scored: list[tuple[int, int, dict[str, Any]]] = []
        for idx_s, spec in enumerate(SLASH_COMMAND_SPECS):
            cmd_bare = spec["cmd"][1:].lower()
            desc_low = spec["desc"].lower()
            cat_low = spec["category"].lower()
            kws = tuple(str(k).lower() for k in spec.get("keywords", ()))
            if not spec.get("canonical", True) and not cmd_bare.startswith(q):
                continue
            tier: int | None = None
            if cmd_bare == q:
                tier = 0
            elif cmd_bare.startswith(q):
                tier = 1
            elif any(k.startswith(q) for k in kws):
                tier = 2
            elif q in cmd_bare:
                tier = 3
            elif q in cat_low or q in desc_low or any(q in k for k in kws):
                tier = 4
            if tier is not None:
                scored.append((tier, idx_s, dict(spec)))

        # Deduplicate if a non-canonical alias and its canonical target both matched
        scored.sort(key=lambda item: (item[0], item[1]))
        return [item[2] for item in scored]

    def get_slash_stage2_options(self, cmd_name: str, arg_query: str = "") -> list[dict[str, Any]]:
        """
        Claude Code style in-input interactive argument & subcommand provider:
        Returns live-filtered selectable options for `cmd_name` right inside the input box dock!
        """
        cmd = cmd_name.strip().lower()
        q = arg_query.strip()
        q_low = q.lower()
        options: list[dict[str, Any]] = []

        if cmd == "/model":
            switchable = self.llm.list_switchable_models()
            for idx_m, item in enumerate(switchable, start=1):
                roles: list[str] = []
                if item.get("is_current_fast"):
                    roles.append("Fast")
                if item.get("is_current_reasoning"):
                    roles.append("Reasoning")
                badge = f"● 当前使用 ({'/'.join(roles)})" if roles else "✓ 可直接切换"
                opt = {
                    "label": f"{idx_m}. [{item['provider_name']}] {item['model']}",
                    "badge": badge,
                    "desc": f"{item['vendor_label']} · 按 Enter 立即切换",
                    "fill_text": f"/model {item['model']}",
                    "exec_line": f"/model {item['model']}",
                    "action": "switch_model_direct",
                    "provider_id": str(item["provider_id"]),
                    "model": str(item["model"]),
                    "is_current": bool(item.get("is_current")),
                    "enter_hint": "Enter 立即切换",
                }
                if (
                    not q_low
                    or (q_low.lstrip("#").isdigit() and int(q_low.lstrip("#")) == idx_m)
                    or q_low in str(item["model"]).lower()
                    or q_low in str(item["provider_name"]).lower()
                    or q_low in str(item["vendor_label"]).lower()
                ):
                    options.append(opt)
            conn_opt = {
                "label": "+ 连接其他大模型厂商 (/connect)",
                "badge": "7大厂商",
                "desc": "DeepSeek / GLM / MiMo / MiniMax / Qwen / OpenAI / Anthropic",
                "fill_text": "/connect ",
                "exec_line": "/connect",
                "action": "transition_to_connect",
                "enter_hint": "Enter/Tab 连接新厂商",
            }
            if not q_low or any(k in q_low for k in ("connect", "连接", "厂商", "新", "+")):
                options.append(conn_opt)
            return options

        if cmd == "/resume":
            sessions = self.list_saved_sessions()
            if not sessions:
                return [
                    {
                        "label": "(当前目录暂无历史对话记录)",
                        "badge": "空",
                        "desc": "发送提问后会自动按目录保存对话，随时可通过 /resume 恢复",
                        "fill_text": "/resume",
                        "exec_line": "/resume",
                        "action": "noop_empty",
                        "enter_hint": "Esc 返回",
                    }
                ]
            for idx_s, s_item in enumerate(sessions[:15], start=1):
                t_disp = str(s_item.get("updated_at_display", ""))
                n_turns = int(s_item.get("turn_count", 1))
                latest_q = str(s_item.get("latest_prompt") or s_item.get("title") or "(空对话)")
                badge = "● 当前会话" if s_item.get("is_current") else ("★ 最新" if idx_s == 1 else f"{n_turns}轮")
                opt = {
                    "label": f"{idx_s}. [{t_disp}] ({n_turns}轮)",
                    "badge": badge,
                    "desc": latest_q,
                    "fill_text": f"/resume {idx_s}",
                    "exec_line": f"/resume {s_item['session_id']}",
                    "action": "resume_direct",
                    "session_id": str(s_item["session_id"]),
                    "enter_hint": "Enter 立即恢复",
                }
                if (
                    not q_low
                    or (q_low.lstrip("#").isdigit() and int(q_low.lstrip("#")) == idx_s)
                    or q_low in latest_q.lower()
                    or q_low in str(s_item.get("title", "")).lower()
                    or q_low in str(s_item.get("session_id", "")).lower()
                    or q_low in t_disp.lower()
                ):
                    options.append(opt)
            return options

        if cmd == "/connect":
            tokens = q.split()
            has_trailing_space = arg_query.endswith(" ") and len(tokens) >= 1
            resolved_first = resolve_provider_id(tokens[0]) if tokens else None
            if resolved_first and (len(tokens) >= 2 or has_trailing_space):
                spec = self.llm.get_provider_spec(resolved_first)
                typed_key = tokens[1] if len(tokens) >= 2 else ""
                if len(typed_key) > 8:
                    masked = typed_key[:4] + ("*" * min(16, len(typed_key) - 6)) + typed_key[-2:]
                elif typed_key:
                    masked = typed_key[:2] + ("*" * max(1, len(typed_key) - 2))
                else:
                    masked = "等待输入..."
                return [
                    {
                        "label": f"🔑 {spec['vendor_label']} API Key ({masked})",
                        "badge": spec["display_name"],
                        "desc": f"接口: {spec['base_url']} · 输入/粘贴 API Key 后按 Enter 自动验证并拉取模型列表",
                        "fill_text": f"/connect {resolved_first} {typed_key}".rstrip() + (" " if not typed_key else ""),
                        "exec_line": f"/connect {resolved_first} {typed_key}".strip(),
                        "action": "connect_verify_key" if typed_key else "noop_require_arg",
                        "ghost_hint": "" if typed_key else f"<在此粘贴 {spec['display_name']} API Key 后按 Enter 验证连接>",
                        "enter_hint": "Enter 验证并连接" if typed_key else "请先输入 API Key",
                    }
                ]
            providers = self.llm.list_supported_providers()
            for p in providers:
                badge = (
                    "● 当前使用"
                    if p.get("active")
                    else ("✓ 已连接" if p.get("connected") else "未连接")
                )
                opt = {
                    "label": f"{p['index']}. {p['display_name']}",
                    "badge": badge,
                    "desc": f"{p['vendor_label']} · 代表模型: {', '.join(p['models'][:3])}",
                    "fill_text": f"/connect {p['provider_id']} ",
                    "exec_line": f"/connect {p['provider_id']}",
                    "action": "connect_step_vendor",
                    "provider_id": str(p["provider_id"]),
                    "enter_hint": "Tab/Enter 选择厂商并输入 Key",
                }
                if (
                    not q_low
                    or (q_low.lstrip("#").isdigit() and int(q_low.lstrip("#")) == int(p["index"]))
                    or (resolved_first and p["provider_id"] == resolved_first)
                    or q_low in str(p["provider_id"]).lower()
                    or q_low in str(p["display_name"]).lower()
                    or q_low in str(p["vendor_label"]).lower()
                ):
                    options.append(opt)
            return options

        if cmd == "/rules":
            rules = self.memory.list_style_rules(include_candidate=True)
            base_opts = [
                {
                    "label": "1. 查看全部开发风格规则 (Mind Mirror)",
                    "badge": f"共 {len(rules)} 条",
                    "desc": "展示当前仓库与全局已学习的个人代码风格规则表",
                    "fill_text": "/rules",
                    "exec_line": "/rules",
                    "action": "exec_line",
                    "enter_hint": "Enter 查看规则",
                }
            ]
            for r in rules[:8]:
                base_opts.append(
                    {
                        "label": f"approve {r.rule_id}",
                        "badge": r.status.value,
                        "desc": f"批准固化规则: {r.pattern_desc} -> {r.preferred_way}",
                        "fill_text": f"/rules approve {r.rule_id}",
                        "exec_line": f"/rules approve {r.rule_id}",
                        "action": "exec_line",
                        "enter_hint": "Enter 批准规则",
                    }
                )
                base_opts.append(
                    {
                        "label": f"reject {r.rule_id}",
                        "badge": r.status.value,
                        "desc": f"拒绝归档规则: {r.pattern_desc}",
                        "fill_text": f"/rules reject {r.rule_id}",
                        "exec_line": f"/rules reject {r.rule_id}",
                        "action": "exec_line",
                        "enter_hint": "Enter 拒绝规则",
                    }
                )
            return [
                o
                for o in base_opts
                if not q_low or q_low in o["label"].lower() or q_low in o["desc"].lower()
            ]

        if cmd == "/skills":
            skills = self.memory.list_skills()
            base_opts = [
                {
                    "label": "1. 查看已注册诊断技能列表",
                    "badge": f"共 {len(skills)} 个",
                    "desc": "列出 .fix/skills/ 下所有可执行项目技能",
                    "fill_text": "/skills",
                    "exec_line": "/skills",
                    "action": "exec_line",
                    "enter_hint": "Enter 查看列表",
                },
                {
                    "label": "2. bootstrap (自动生成项目专属技能)",
                    "badge": "生成",
                    "desc": "扫描当前项目架构并自动生成 .fix/skills/<slug>/run.py 可执行技能",
                    "fill_text": "/skills bootstrap",
                    "exec_line": "/skills bootstrap",
                    "action": "exec_line",
                    "enter_hint": "Enter 立即生成",
                },
            ]
            for slug in skills:
                base_opts.append(
                    {
                        "label": f"run {slug}",
                        "badge": "可执行",
                        "desc": f"立即执行项目专属诊断技能 `{slug}`",
                        "fill_text": f"/skills run {slug}",
                        "exec_line": f"/skills run {slug}",
                        "action": "exec_line",
                        "enter_hint": "Enter 执行技能",
                    }
                )
            return [
                o
                for o in base_opts
                if not q_low or q_low in o["label"].lower() or q_low in o["desc"].lower()
            ]

        if cmd == "/idle":
            st = self.looper.get_status_dict()
            base_opts = [
                {
                    "label": "1. 查看后台进化队列状态 (status)",
                    "badge": f"排队 {st['p2_idle_queued']} 项",
                    "desc": "查看 Serial Looper 与 IdleHandler P2 队列、已完成 Tick 与预算",
                    "fill_text": "/idle",
                    "exec_line": "/idle",
                    "action": "exec_line",
                    "enter_hint": "Enter 查看状态",
                },
                {
                    "label": "2. run (立即执行后台 P2 进化队列)",
                    "badge": "立即执行",
                    "desc": "立即排空并运行后台 IdleHandler 自我进化与架构心智提炼任务",
                    "fill_text": "/idle run",
                    "exec_line": "/idle run",
                    "action": "exec_line",
                    "enter_hint": "Enter 立即运行",
                },
            ]
            return [
                o
                for o in base_opts
                if not q_low or q_low in o["label"].lower() or q_low in o["desc"].lower()
            ]

        if cmd in ("/evolve", "/insights"):
            base_opts = [
                {
                    "label": f"1. 查看三层第一性原理心智模型与优化提案 ({cmd})",
                    "badge": "洞察报告",
                    "desc": "展示现象层 -> 机制契约层 -> 第一性原理本质及主动优化提案",
                    "fill_text": cmd,
                    "exec_line": cmd,
                    "action": "exec_line",
                    "enter_hint": "Enter 查看报告",
                },
                {
                    "label": f"2. {cmd} run (立即触发后台提炼并展示)",
                    "badge": "立即提炼",
                    "desc": "先运行后台进化队列提炼最新心智模型，再展示完整三层洞察报告",
                    "fill_text": f"{cmd} run",
                    "exec_line": f"{cmd} run",
                    "action": "exec_line",
                    "enter_hint": "Enter 提炼并查看",
                },
            ]
            return [
                o
                for o in base_opts
                if not q_low or q_low in o["label"].lower() or q_low in o["desc"].lower()
            ]

        if cmd == "/steps":
            base_opts = [
                {
                    "label": "1. 展开/收起上一轮任务工具调用明细",
                    "badge": "Ctrl+O",
                    "desc": "切换最新一轮任务的探索与命令执行详情折叠状态",
                    "fill_text": "/steps",
                    "exec_line": "/steps",
                    "action": "exec_line",
                    "enter_hint": "Enter 切换折叠",
                }
            ]
            for t_id in sorted(self._turn_fold_states.keys(), reverse=True):
                f_st = self._turn_fold_states[t_id]
                is_col = bool(f_st.get("collapsed", True))
                n_b = int(f_st.get("blocks_count", 0))
                base_opts.append(
                    {
                        "label": f"#{t_id} 轮执行过程 ({n_b} 步操作)",
                        "badge": "当前已收起" if is_col else "当前已展开",
                        "desc": f"点击或按 Enter 立即{'展开' if is_col else '收起'}第 #{t_id} 轮工具调用明细",
                        "fill_text": f"/steps #{t_id}",
                        "exec_line": f"/steps #{t_id}",
                        "action": "exec_line",
                        "enter_hint": f"Enter {'展开' if is_col else '收起'}",
                    }
                )
            return [
                o
                for o in base_opts
                if not q_low or q_low in o["label"].lower() or q_low in o["desc"].lower()
            ]

        if cmd == "/diag":
            base_opts = [
                {
                    "label": "1. 全量工作区静态语法与导入诊断",
                    "badge": "全量检查",
                    "desc": "扫描并检查当前仓库所有源码文件的语法、AST 与导入完整性",
                    "fill_text": "/diag",
                    "exec_line": "/diag",
                    "action": "exec_line",
                    "enter_hint": "Enter 运行诊断",
                }
            ]
            try:
                repo_root = Path(self.toolkit.repo_root)
                py_files = sorted(
                    str(p.relative_to(repo_root))
                    for p in repo_root.rglob("*.py")
                    if ".git" not in p.parts and ".fix" not in p.parts and "__pycache__" not in p.parts
                )[:12]
                for rel_f in py_files:
                    if not q_low or q_low in rel_f.lower():
                        base_opts.append(
                            {
                                "label": f"诊断文件: {rel_f}",
                                "badge": "单文件",
                                "desc": f"对 `{rel_f}` 运行静态语法与导入诊断",
                                "fill_text": f"/diag {rel_f}",
                                "exec_line": f"/diag {rel_f}",
                                "action": "exec_line",
                                "enter_hint": "Enter 诊断该文件",
                            }
                        )
            except Exception:
                pass
            return base_opts[:10]

        if cmd == "/trace":
            landmarks = self.memory.search_landmarks(query=q, limit=6, include_stale=False)
            if q:
                options.append(
                    {
                        "label": f"追踪符号: {q}",
                        "badge": "LSP/AST",
                        "desc": f"跨文件检索符号 `{q}` 的定义位置、调用方 (Callers) 与被调用方 (Callees)",
                        "fill_text": f"/trace {q}",
                        "exec_line": f"/trace {q}",
                        "action": "exec_line",
                        "enter_hint": "Enter 立即追踪",
                    }
                )
            else:
                options.append(
                    {
                        "label": "请在输入框输入要追踪的符号名 (如函数名或类名)",
                        "badge": "必填参数",
                        "desc": "用法: /trace <symbol_name> [rel_path] — 跨文件追踪定义与调用链",
                        "fill_text": "/trace ",
                        "exec_line": "/trace",
                        "action": "noop_require_arg",
                        "ghost_hint": "<输入函数名或类名，按 Enter 追踪跨文件调用链>",
                        "enter_hint": "请输入符号名",
                    }
                )
            for lm in landmarks:
                for ev_p in lm.evidence_paths[:1]:
                    sym_cand = Path(ev_p).stem
                    if sym_cand and (not q_low or q_low in sym_cand.lower()):
                        options.append(
                            {
                                "label": f"追踪地标模块: {sym_cand} ({ev_p})",
                                "badge": lm.category,
                                "desc": lm.summary[:60],
                                "fill_text": f"/trace {sym_cand} {ev_p}",
                                "exec_line": f"/trace {sym_cand} {ev_p}",
                                "action": "exec_line",
                                "enter_hint": "Enter 追踪",
                            }
                        )
            return options[:8]

        if cmd == "/new":
            if q:
                return [
                    {
                        "label": f"新建对话并直接提问: {q}",
                        "badge": "新建+提问",
                        "desc": "在当前目录开启全新会话并立即执行该问题 (原对话自动保存至 /resume)",
                        "fill_text": f"/new {q}",
                        "exec_line": f"/new {q}",
                        "action": "exec_line",
                        "enter_hint": "Enter 新建并提问",
                    },
                    {
                        "label": "仅新建空白对话 (/new)",
                        "badge": "空白会话",
                        "desc": "清空当前屏幕并分配全新会话 ID，原对话自动保存至 /resume",
                        "fill_text": "/new",
                        "exec_line": "/new",
                        "action": "exec_line",
                        "enter_hint": "Enter 新建空白对话",
                    },
                ]
            return [
                {
                    "label": "1. 立即在当前目录新建空白对话 (/new)",
                    "badge": "推荐",
                    "desc": "原对话自动保存至 /resume；也可继续输入 `/new <问题>` 直接在新对话提问",
                    "fill_text": "/new",
                    "exec_line": "/new",
                    "action": "exec_line",
                    "ghost_hint": " [可选: 输入新对话问题，或直接按 Enter 新建空白对话]",
                    "enter_hint": "Enter 立即新建",
                }
            ]

        if cmd in ("/ask", "/bug", "/opt"):
            meta_map = {
                "/ask": (
                    "只读架构分析轨道",
                    "<在此输入架构或代码问题，按 Enter 启动只读分析 ([file:line])>",
                    "强制使用 EXPLAIN 只读轨道分析代码链路并输出精确 [file:line] 证据",
                ),
                "/bug": (
                    "科学闭环修复轨道",
                    "<在此输入 Bug 现象或报错栈，按 Enter 启动 Hypothesis->Reproduce->Fix->Verify>",
                    "强制使用 BUG_FIX 科学闭环轨道定位根因、修复代码并运行回归验证",
                ),
                "/opt": (
                    "性能与架构优化轨道",
                    "<在此输入性能或架构优化目标，按 Enter 启动优化轨道>",
                    "强制使用 FEATURE/OPT 轨道执行重构、性能优化与回归验证",
                ),
            }
            badge_t, ghost_t, desc_t = meta_map[cmd]
            if q:
                return [
                    {
                        "label": f"执行 {cmd}: {q}",
                        "badge": badge_t,
                        "desc": desc_t,
                        "fill_text": f"{cmd} {q}",
                        "exec_line": f"{cmd} {q}",
                        "action": "exec_line",
                        "enter_hint": "Enter 立即执行",
                    }
                ]
            return [
                {
                    "label": f"请在输入框 `{cmd} ` 后输入内容并按 Enter 执行",
                    "badge": badge_t,
                    "desc": desc_t,
                    "fill_text": f"{cmd} ",
                    "exec_line": cmd,
                    "action": "noop_require_arg",
                    "ghost_hint": ghost_t,
                    "enter_hint": "请输入参数内容",
                }
            ]

        if cmd in ("/profile", "/sandbox"):
            default_cmd = self.profile.test_command or "pytest -q"
            target_c = q or default_cmd
            label_prefix = "cProfile 性能剖析" if cmd == "/profile" else "Git Worktree 沙箱验证"
            return [
                {
                    "label": f"{label_prefix}: {target_c}",
                    "badge": "默认命令" if not q else "自定义命令",
                    "desc": f"运行 `{cmd} {target_c}`",
                    "fill_text": f"{cmd} {target_c}",
                    "exec_line": f"{cmd} {target_c}",
                    "action": "exec_line",
                    "ghost_hint": "" if q else f"[默认: {default_cmd} · 按 Enter 直接运行]",
                    "enter_hint": "Enter 立即运行",
                }
            ]

        if cmd in ("/landmarks", "/autonomy", "/learn"):
            default_map = {
                "/landmarks": ("查看全部架构地标", "输入关键词可过滤 .fix/repo.db 中的架构地标"),
                "/autonomy": ("评估当前仓库 L1-L4 自治等级", "可附加具体任务描述评估该任务的爆炸半径与自治分"),
                "/learn": ("从 HEAD 提交学习代码风格", "可指定 Git commit SHA 或 HEAD~1 学习开发者代码风格偏好"),
            }
            lbl_d, desc_d = default_map[cmd]
            return [
                {
                    "label": f"{lbl_d}" + (f" ({q})" if q else ""),
                    "badge": cmd[1:],
                    "desc": desc_d,
                    "fill_text": f"{cmd} {q}".strip(),
                    "exec_line": f"{cmd} {q}".strip(),
                    "action": "exec_line",
                    "enter_hint": "Enter 立即执行",
                }
            ]

        return options

    def close_modal(self, toast_msg: str = "已关闭弹窗") -> None:
        """Close any active `/connect` or `/model` popup modal."""
        self._modal_state = None
        self._set_terminal_hover_mouse_tracking(False)
        self._modal_screen_row_to_item.clear()
        self._modal_doc_line_to_item.clear()
        if self._input_buffer_ref is not None:
            try:
                self._input_buffer_ref.reset()
            except Exception:
                pass
        if toast_msg:
            self.show_toast(toast_msg, duration_sec=3.0)
        if self._tui_app is not None:
            try:
                self._tui_app.invalidate()
            except Exception:
                pass

    def move_modal_selection(self, delta: int) -> bool:
        """Move the highlighted item up (`delta < 0`) or down (`delta > 0`) in the active popup modal."""
        if self._modal_state is None:
            return False
        mode = self._modal_state.get("mode")
        if mode == "connect_provider":
            items = self._modal_state.get("providers") or []
            total = len(items)
        elif mode == "connect_select_model":
            items = self._modal_state.get("models") or []
            total = len(items)
        elif mode == "switch_model":
            items = self._modal_state.get("models") or []
            total = len(items) + 1  # +1 for `+ 连接其他大模型厂商 (/connect)`
        elif mode == "resume_session":
            items = self._modal_state.get("sessions") or []
            total = len(items)
        else:
            return False
        if total <= 0:
            return False
        cur = int(self._modal_state.get("selected_idx", 0))
        self._modal_state["selected_idx"] = (cur + delta) % total
        if self._tui_app is not None:
            try:
                self._tui_app.invalidate()
            except Exception:
                pass
        return True

    def _finalize_model_activation(self, provider_id: str, model_name: str, via_connect: bool = False) -> dict[str, Any]:
        """Activate `provider_id` + `model_name`, close modal, record session block, and show toast."""
        self.llm.activate_provider_model(provider_id, model_name)
        spec = self.llm.get_provider_spec(provider_id)
        self._modal_state = None
        self._set_terminal_hover_mouse_tracking(False)
        self._modal_screen_row_to_item.clear()
        self._modal_doc_line_to_item.clear()
        self._cached_completed_key = None
        self._cached_lines_key = None
        action_verb = "已连接并切换至" if via_connect else "已切换当前模型为"
        msg_line = (
            f"[bold #10B981]✓ {action_verb} {escape(spec['vendor_label'])} — "
            f"模型: [bold #0EA5E9]{escape(model_name)}[/bold #0EA5E9] "
            f"(Endpoint: {escape(self.llm.base_url)})[/bold #10B981]"
        )
        self._record_and_render_block(lambda c, _m=msg_line: c.print(_m))
        self.show_toast(f"✓ {spec['display_name']} · {model_name}", duration_sec=3.0)
        if self._tui_app is not None:
            try:
                self._tui_app.invalidate()
            except Exception:
                pass
        return {
            "step": "completed",
            "ok": True,
            "provider_id": provider_id,
            "model": model_name,
        }

    def confirm_modal_step_sync(self, explicit_index: int | None = None) -> dict[str, Any] | None:
        """
        Synchronously handle non-network modal confirmations (`connect_provider -> connect_api_key`,
        `connect_select_model -> completed`, `switch_model -> completed`).
        Returns None if the current modal step requires async network verification (`connect_api_key`).
        """
        if self._modal_state is None:
            return {"step": "none", "ok": False}
        mode = self._modal_state.get("mode")
        if explicit_index is not None:
            self._modal_state["selected_idx"] = explicit_index
        sel_idx = int(self._modal_state.get("selected_idx", 0))

        if mode == "connect_provider":
            providers = self._modal_state.get("providers") or self.llm.list_supported_providers()
            if not providers:
                return {"step": "error", "ok": False}
            sel_idx = max(0, min(len(providers) - 1, sel_idx))
            prov = providers[sel_idx]
            pid = prov["provider_id"]
            self._modal_state["mode"] = "connect_api_key"
            self._modal_state["provider_id"] = pid
            self._modal_state["provider_info"] = prov
            self._modal_state["api_key_input"] = ""
            self._modal_state["error_msg"] = ""
            if self._input_buffer_ref is not None:
                try:
                    self._input_buffer_ref.reset()
                except Exception:
                    pass
            self.show_toast(f"请输入 {prov['display_name']} API Key 后按回车确认", duration_sec=3.0)
            if self._tui_app is not None:
                try:
                    self._tui_app.invalidate()
                except Exception:
                    pass
            return {"step": "connect_api_key", "ok": True, "provider_id": pid}

        if mode == "connect_select_model":
            models = self._modal_state.get("models") or []
            pid = str(self._modal_state.get("provider_id") or self.llm.provider_id)
            if not models:
                models = self.llm.get_provider_models(pid)
            sel_idx = max(0, min(len(models) - 1, sel_idx))
            chosen_model = str(models[sel_idx])
            return self._finalize_model_activation(pid, chosen_model, via_connect=True)

        if mode == "switch_model":
            models = self._modal_state.get("models") or []
            if sel_idx >= len(models):
                self.open_connect_modal()
                return {"step": "connect_provider", "ok": True}
            sel_idx = max(0, min(len(models) - 1, sel_idx))
            entry = models[sel_idx]
            return self._finalize_model_activation(
                str(entry["provider_id"]),
                str(entry["model"]),
                via_connect=False,
            )

        if mode == "resume_session":
            sessions = self._modal_state.get("sessions") or self.list_saved_sessions()
            if not sessions:
                self.close_modal("当前目录暂无历史对话记录")
                return {"step": "error", "ok": False}
            sel_idx = max(0, min(len(sessions) - 1, sel_idx))
            chosen = sessions[sel_idx]
            resumed = self.resume_conversation_session(str(chosen["session_id"]))
            return {
                "step": "completed",
                "ok": resumed is not None,
                "session_id": str(chosen["session_id"]),
                "turn_count": int(chosen.get("turn_count", 0)),
            }

        return None

    async def confirm_modal_step(
        self,
        explicit_index: int | None = None,
        explicit_api_key: str | None = None,
    ) -> dict[str, Any]:
        """
        Confirm the current step in `/connect` or `/model` modal (supports both sync selection steps
        and async API Key verification against the vendor endpoint).
        """
        sync_res = self.confirm_modal_step_sync(explicit_index=explicit_index)
        if sync_res is not None:
            return sync_res
        if self._modal_state is None:
            return {"step": "none", "ok": False}

        mode = self._modal_state.get("mode")
        if mode == "connect_api_key":
            pid = str(self._modal_state.get("provider_id") or "deepseek")
            buf_txt = ""
            if self._input_buffer_ref is not None:
                buf_txt = self._input_buffer_ref.text.strip()
            raw_key = (
                explicit_api_key
                if explicit_api_key is not None
                else (buf_txt or str(self._modal_state.get("api_key_input") or ""))
            ).strip()
            if not raw_key:
                err = "API Key 不能为空，请粘贴或输入您的 API Key"
                self._modal_state["error_msg"] = err
                self.show_toast("✗ API Key 不能为空", duration_sec=3.0)
                if self._tui_app is not None:
                    try:
                        self._tui_app.invalidate()
                    except Exception:
                        pass
                return {"step": "connect_api_key", "ok": False, "error": err}

            if self._input_buffer_ref is not None:
                try:
                    self._input_buffer_ref.reset()
                except Exception:
                    pass
            self._modal_state["api_key_input"] = raw_key
            self._modal_state["mode"] = "connect_verifying"
            self._modal_state["error_msg"] = ""
            spec = self.llm.get_provider_spec(pid)
            self.show_toast(f"⠋ 正在验证 {spec['display_name']} API Key...", duration_sec=5.0)
            if self._tui_app is not None:
                try:
                    self._tui_app.invalidate()
                except Exception:
                    pass

            ok, models, msg = await self.llm.verify_and_fetch_models(pid, raw_key)
            if not ok:
                if self._modal_state is not None:
                    self._modal_state["mode"] = "connect_api_key"
                    self._modal_state["error_msg"] = msg
                self.show_toast("✗ API Key 验证失败", duration_sec=3.0)
                if self._tui_app is not None:
                    try:
                        self._tui_app.invalidate()
                    except Exception:
                        pass
                return {"step": "connect_api_key", "ok": False, "error": msg}

            if self._modal_state is not None:
                self._modal_state["mode"] = "connect_select_model"
                self._modal_state["models"] = models
                self._modal_state["selected_idx"] = 0
                self._modal_state["status_msg"] = msg
                self._modal_state["error_msg"] = ""
            self.show_toast(f"✓ {spec['display_name']} 连接成功！请选择模型", duration_sec=3.0)
            if self._tui_app is not None:
                try:
                    self._tui_app.invalidate()
                except Exception:
                    pass
            return {
                "step": "connect_select_model",
                "ok": True,
                "provider_id": pid,
                "models": models,
                "message": msg,
            }

        return {"step": str(mode), "ok": False}

    def _build_modal_tagged_lines(self, max_line_cells: int) -> list[tuple[int | None, list[tuple[str, str]]]]:
        """
        Build the visual lines for the active `/connect` or `/model` popup dialog.
        Each entry is `(selectable_item_index_or_None, [(style, text), ...])`.
        """
        if self._modal_state is None:
            return []
        mode = self._modal_state.get("mode")
        border_w = min(max(48, max_line_cells - 4), 104)
        horiz = "─" * max(10, border_w - 4)
        out: list[tuple[int | None, list[tuple[str, str]]]] = []

        if mode == "connect_provider":
            providers = self._modal_state.get("providers") or self.llm.list_supported_providers()
            sel_idx = int(self._modal_state.get("selected_idx", 0))
            out.append(
                (
                    None,
                    [
                        ("bold #0D9488", "╭── "),
                        ("bold #F97316", "🔌 连接大模型厂商 (/connect)"),
                        ("bold #0D9488", " ── "),
                        ("#0EA5E9", "↑↓ 选择 · Enter/鼠标点击确认 · Esc 取消"),
                        ("bold #0D9488", " ──╮"),
                    ],
                )
            )
            for idx, prov in enumerate(providers):
                is_sel = idx == sel_idx
                bg = ""
                marker = " ❯ " if is_sel else "   "
                badge = (
                    " [● 当前使用]"
                    if prov.get("active")
                    else (" [✓ 已连接]" if prov.get("connected") else "")
                )
                badge_style = ("bold #10B981" if (prov.get("active") or prov.get("connected")) else "#788596") + bg
                hover_hint = "  ◀ 点击确认" if is_sel else ""
                out.append(
                    (
                        idx,
                        [
                            ("#0D9488", "│"),
                            (f"bold #F97316{bg}" if is_sel else "#0D9488", f"{marker}{idx + 1}. "),
                            (f"bold #0EA5E9{bg}" if is_sel else "bold", f"{prov['display_name']:<10} "),
                            (f"bold #10B981{bg}" if is_sel else "#10B981", f"{prov['vendor_label']:<24} "),
                            (f"bold #0EA5E9{bg}" if is_sel else "#788596", f"{prov['base_url']}"),
                            (badge_style, badge),
                            (f"bold #F97316{bg}", hover_hint),
                        ],
                    )
                )
            out.append(
                (
                    None,
                    [
                        ("bold #0D9488", f"╰{horiz}╯"),
                    ],
                )
            )
            return out

        if mode in ("connect_api_key", "connect_verifying"):
            pid = str(self._modal_state.get("provider_id") or "deepseek")
            spec = self.llm.get_provider_spec(pid)
            buf_txt = self._input_buffer_ref.text if self._input_buffer_ref is not None else ""
            raw_key = buf_txt or str(self._modal_state.get("api_key_input") or "")
            if len(raw_key) > 8:
                masked_key = raw_key[:4] + ("*" * min(24, len(raw_key) - 6)) + raw_key[-2:]
            elif raw_key:
                masked_key = raw_key[:2] + ("*" * max(1, len(raw_key) - 2))
            else:
                masked_key = ""
            out.append(
                (
                    None,
                    [
                        ("bold #0D9488", "╭── "),
                        ("bold #F97316", f"🔑 输入 {spec['vendor_label']} API Key"),
                        ("bold #0D9488", " ── "),
                        ("#0EA5E9", "Enter 验证连接 · Esc 取消"),
                        ("bold #0D9488", " ──╮"),
                    ],
                )
            )
            out.append(
                (
                    None,
                    [
                        ("#0D9488", "│  "),
                        ("bold", "官方接口: "),
                        ("#0EA5E9", f"{spec['base_url']}  "),
                        ("#788596", f"({spec['api_notes']})"),
                    ],
                )
            )
            out.append(
                (
                    0,
                    [
                        ("#0D9488", "│  "),
                        ("bold #F97316", "API Key 输入框: "),
                        (
                            "bold #10B981" if masked_key else "#788596",
                            f"[ {masked_key}█ ]" if masked_key else "[ 请直接输入或粘贴 API Key 后按 Enter 确认 █ ]",
                        ),
                    ],
                )
            )
            if mode == "connect_verifying":
                out.append(
                    (
                        None,
                        [
                            ("#0D9488", "│  "),
                            ("bold #0EA5E9", f"⠋ 正在连接 {spec['display_name']} 官方接口并拉取支持的模型列表..."),
                        ],
                    )
                )
            err = str(self._modal_state.get("error_msg") or "")
            if err:
                out.append(
                    (
                        None,
                        [
                            ("#0D9488", "│  "),
                            ("bold #F97316", f"✗ {err}"),
                        ],
                    )
                )
            out.append((None, [("bold #0D9488", f"╰{horiz}╯")]))
            return out

        if mode == "connect_select_model":
            pid = str(self._modal_state.get("provider_id") or self.llm.provider_id)
            spec = self.llm.get_provider_spec(pid)
            models = self._modal_state.get("models") or self.llm.get_provider_models(pid)
            sel_idx = int(self._modal_state.get("selected_idx", 0))
            status_msg = str(self._modal_state.get("status_msg") or f"✓ {spec['vendor_label']} 连接成功")
            out.append(
                (
                    None,
                    [
                        ("bold #0D9488", "╭── "),
                        ("bold #10B981", f"{status_msg}"),
                        ("bold #0D9488", " ── "),
                        ("#0EA5E9", "请选择模型 (↑↓ · Enter/鼠标点击确认)"),
                        ("bold #0D9488", " ──╮"),
                    ],
                )
            )
            for idx, m_name in enumerate(models):
                is_sel = idx == sel_idx
                bg = ""
                marker = " ❯ " if is_sel else "   "
                tag = "  [推荐]" if idx == 0 else ""
                hover_hint = "  ◀ 点击启用" if is_sel else ""
                out.append(
                    (
                        idx,
                        [
                            ("#0D9488", "│"),
                            (f"bold #F97316{bg}" if is_sel else "#0D9488", f"{marker}{idx + 1}. "),
                            (f"bold #0EA5E9{bg}" if is_sel else "bold", f"{m_name:<28}"),
                            (f"bold #10B981{bg}" if is_sel else "#10B981", tag),
                            (f"bold #F97316{bg}", hover_hint),
                        ],
                    )
                )
            out.append((None, [("bold #0D9488", f"╰{horiz}╯")]))
            return out

        if mode == "switch_model":
            models = self._modal_state.get("models") or self.llm.list_switchable_models()
            sel_idx = int(self._modal_state.get("selected_idx", 0))
            out.append(
                (
                    None,
                    [
                        ("bold #0D9488", "╭── "),
                        ("bold #F97316", "🤖 当前支持切换的大模型列表 (/model)"),
                        ("bold #0D9488", " ── "),
                        ("#0EA5E9", "↑↓ 选择 · Enter/鼠标点击切换 · Esc 关闭"),
                        ("bold #0D9488", " ──╮"),
                    ],
                )
            )
            for idx, entry in enumerate(models):
                is_sel = idx == sel_idx
                bg = ""
                marker = " ❯ " if is_sel else "   "
                cur_badge = "  ● 当前使用" if entry.get("is_current") else ""
                hover_hint = "  ◀ 点击切换" if is_sel else ""
                out.append(
                    (
                        idx,
                        [
                            ("#0D9488", "│"),
                            (f"bold #F97316{bg}" if is_sel else "#0D9488", f"{marker}{idx + 1}. "),
                            (f"bold #10B981{bg}" if is_sel else "#10B981", f"[{entry['provider_name']}] "),
                            (f"bold #0EA5E9{bg}" if is_sel else "bold", f"{entry['model']:<28}"),
                            (f"bold #F97316{bg}", cur_badge),
                            (f"bold #F97316{bg}", hover_hint),
                        ],
                    )
                )
            connect_idx = len(models)
            is_conn_sel = sel_idx == connect_idx
            conn_bg = ""
            conn_marker = " ❯ " if is_conn_sel else "   "
            conn_hint = "  ◀ 点击打开" if is_conn_sel else ""
            out.append(
                (
                    connect_idx,
                    [
                        ("#0D9488", "│"),
                        (f"bold #F97316{conn_bg}" if is_conn_sel else "#0D9488", f"{conn_marker}+ "),
                        (
                            f"bold #0EA5E9{conn_bg}" if is_conn_sel else "#0EA5E9",
                            "连接其他大模型厂商 (/connect: DeepSeek / GLM / MiMo / MiniMax / Qwen / OpenAI / Anthropic)",
                        ),
                        (f"bold #F97316{conn_bg}", conn_hint),
                    ],
                )
            )
            out.append((None, [("bold #0D9488", f"╰{horiz}╯")]))
            return out

        if mode == "resume_session":
            sessions = self._modal_state.get("sessions") or self.list_saved_sessions()
            sel_idx = int(self._modal_state.get("selected_idx", 0))
            dir_label = self.session_dir.name or str(self.session_dir)
            out.append(
                (
                    None,
                    [
                        ("bold #0D9488", "╭── "),
                        ("bold #F97316", f"💬 恢复历史对话 (/resume · {dir_label})"),
                        ("bold #0D9488", " ── "),
                        ("#0EA5E9", "↑↓/鼠标悬停选择 · Enter/点击恢复 · Esc 取消"),
                        ("bold #0D9488", " ──╮"),
                    ],
                )
            )
            preview_budget = max(22, min(56, border_w - 42))
            for idx, sess_item in enumerate(sessions[:15]):
                is_sel = idx == sel_idx
                marker = " ❯ " if is_sel else "   "
                t_disp = str(sess_item.get("updated_at_display", ""))
                n_turns = int(sess_item.get("turn_count", 1))
                latest_q = _truncate_middle_for_width(
                    str(sess_item.get("latest_prompt") or sess_item.get("title") or "(空对话)"),
                    preview_budget,
                )
                badge = (
                    "  [● 当前会话]"
                    if sess_item.get("is_current")
                    else ("  [最新]" if idx == 0 else "")
                )
                badge_style = (
                    "bold #10B981"
                    if (sess_item.get("is_current") or idx == 0)
                    else "#788596"
                )
                hover_hint = "  ◀ 点击恢复" if is_sel else ""
                out.append(
                    (
                        idx,
                        [
                            ("#0D9488", "│"),
                            ("bold #F97316" if is_sel else "#0D9488", f"{marker}{idx + 1}. "),
                            ("bold #0EA5E9" if is_sel else "#0EA5E9", f"[{t_disp}] "),
                            ("bold #10B981" if is_sel else "#10B981", f"({n_turns}轮) "),
                            ("bold #F97316" if is_sel else "", f"{latest_q}"),
                            (badge_style, badge),
                            ("bold #F97316", hover_hint),
                        ],
                    )
                )
            out.append((None, [("bold #0D9488", f"╰{horiz}╯")]))
            return out

        return out

