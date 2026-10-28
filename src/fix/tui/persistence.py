"""Per-project session state persistence, input history, and `/resume` & `/new` lifecycle mixin."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path
from datetime import datetime
from typing import Any, Callable

from rich.console import Console
from rich.markup import escape

from fix.core.context import RepoProfile
from fix.tui.composer import render_codex_user_prompt_to
from fix.tui.theme import (
    CleanMarkdown,
    PRIMARY_TEXT_STYLE,
    SECONDARY_STEP_TEXT_STYLE,
    _sync_live_terminal_width,
    normalize_line_range_notation,
)
from fix.protocol.schema import TaskSession
from fix.tui.tracker import CollapsibleProgressTracker, _render_session_result, format_explored_group_lines


class SessionPersistenceMixin:
    """Mixin managing per-directory `.fix/projects/` history and conversation session snapshots."""

    _last_session_save_ts: float = 0.0

    @staticmethod
    def _new_conversation_session_id() -> str:
        return f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _resolve_project_state_dir(session_dir: Path) -> Path:
        """Resolve the isolated per-directory state folder under `~/.fix/projects/<name>_<hash>/`."""
        resolved = session_dir.resolve()
        base_root = Path(
            os.environ.get("FIX_PROJECTS_STATE_DIR")
            or (Path.home() / ".fix" / "projects")
        )
        safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", resolved.name or "root").strip("_")[:32] or "root"
        dir_hash = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:16]
        state_dir = base_root / f"{safe_name}_{dir_hash}"
        state_dir.mkdir(parents=True, exist_ok=True)
        return state_dir

    def record_input_history(self, text: str) -> None:
        """Record a prompt into the current directory's isolated `FileHistory` (deduplicating consecutive identical entries)."""
        clean = text.strip()
        if not clean:
            return
        low = clean.lower()
        if low in ("/new", "/clear", "/exit", "/quit", "/q") or low.startswith("/resume"):
            return
        hist = self._file_history
        if not getattr(hist, "_loaded", False):
            try:
                hist._loaded_strings = list(hist.load_history_strings())
                hist._loaded = True
            except Exception:
                hist._loaded_strings = []
                hist._loaded = True
        existing = hist.get_strings()
        if not existing or existing[-1] != clean:
            hist.append_string(clean)
        if self._input_buffer_ref is not None and not getattr(self, "_browsing_history", False):
            try:
                all_items = hist.get_strings()
                tail_txt = self._input_buffer_ref.text
                self._input_buffer_ref._working_lines.clear()
                for item in all_items:
                    self._input_buffer_ref._working_lines.append(item)
                self._input_buffer_ref._working_lines.append(tail_txt)
                self._input_buffer_ref.working_index = len(self._input_buffer_ref._working_lines) - 1
            except Exception:
                pass

    @staticmethod
    def _format_relative_time(ts: float, now: float | None = None) -> str:
        curr = now if now is not None else time.time()
        diff = max(0.0, curr - ts)
        if diff < 60:
            return "刚刚"
        if diff < 3600:
            return f"{int(diff // 60)}分钟前"
        if diff < 86400:
            return f"{int(diff // 3600)}小时前"
        return f"{int(diff // 86400)}天前"

    def _persist_current_conversation_session(self) -> Path | None:
        """Persist the current session's multi-turn conversation to `self.sessions_dir / <session_id>.json`."""
        if not self._saved_turns_records:
            return None
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        now_ts = max(time.time(), SessionPersistenceMixin._last_session_save_ts + 1e-4)
        SessionPersistenceMixin._last_session_save_ts = now_ts
        first_prompt = str(self._saved_turns_records[0].get("user_prompt", "")).strip()
        latest_prompt = str(self._saved_turns_records[-1].get("user_prompt", "")).strip()
        latest_summary = str(self._saved_turns_records[-1].get("final_summary", "")).strip()
        payload = {
            "session_id": self.session_id,
            "directory": str(self.session_dir),
            "repo_root": str(self.toolkit.repo_root),
            "created_at": self.session_created_at,
            "updated_at": now_ts,
            "title": " ".join(first_prompt.split())[:120],
            "latest_prompt": " ".join(latest_prompt.split())[:120],
            "latest_summary_preview": " ".join(latest_summary.split())[:160],
            "turn_count": len(self._saved_turns_records),
            "turns": self._saved_turns_records,
        }
        target_path = self.sessions_dir / f"{self.session_id}.json"
        tmp_path = self.sessions_dir / f".{self.session_id}.json.tmp"
        try:
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp_path.replace(target_path)
        except Exception:
            try:
                target_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                return None
        return target_path

    def list_saved_sessions(self) -> list[dict[str, Any]]:
        """List saved historical conversation sessions for `self.session_dir`, sorted newest (`updated_at` desc) first."""
        if not self.sessions_dir.exists():
            return []
        now_ts = time.time()
        records: list[dict[str, Any]] = []
        for fpath in self.sessions_dir.glob("*.json"):
            if fpath.name.startswith("."):
                continue
            try:
                data = json.loads(fpath.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            turns = data.get("turns")
            if not isinstance(turns, list) or not turns:
                continue
            sid = str(data.get("session_id") or fpath.stem)
            created_at = float(data.get("created_at") or fpath.stat().st_mtime)
            updated_at = float(data.get("updated_at") or created_at)
            first_q = " ".join(str(turns[0].get("user_prompt", "")).split())
            latest_q = " ".join(str(turns[-1].get("user_prompt", "")).split())
            latest_ans = " ".join(str(turns[-1].get("final_summary", "")).split())
            dt_str = datetime.fromtimestamp(updated_at).strftime("%m-%d %H:%M")
            rel_str = self._format_relative_time(updated_at, now=now_ts)
            records.append(
                {
                    "session_id": sid,
                    "directory": str(data.get("directory") or self.session_dir),
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "updated_at_display": f"{dt_str} · {rel_str}",
                    "relative_time": rel_str,
                    "title": str(data.get("title") or first_q)[:120],
                    "latest_prompt": str(data.get("latest_prompt") or latest_q)[:120],
                    "latest_summary_preview": str(data.get("latest_summary_preview") or latest_ans)[:160],
                    "turn_count": len(turns),
                    "is_current": bool(sid == self.session_id and len(self._saved_turns_records) > 0),
                    "turns": turns,
                }
            )
        records.sort(key=lambda item: (float(item.get("updated_at", 0.0)), str(item.get("session_id", ""))), reverse=True)
        for idx, item in enumerate(records, start=1):
            item["index"] = idx
        return records

    def _make_turn_render_block(
        self,
        turn_id: int,
        user_prompt: str,
        completed_sess: TaskSession,
        saved_stream_blocks: list[dict[str, Any]],
        fold_state: dict[str, Any],
    ) -> Callable[[Console], None]:
        repo_name = Path(self.profile.repo_root).name

        def _render_turn_with_prompt(c: Console) -> None:
            if self._tui_mode:
                render_codex_user_prompt_to(c, user_prompt)
                if saved_stream_blocks:
                    is_collapsed = bool(fold_state.get("collapsed", True))
                    c.print(
                        self._format_execution_fold_header(
                            collapsed=is_collapsed,
                            turn_idx=turn_id,
                            stream_blocks=saved_stream_blocks,
                            session=completed_sess,
                        )
                    )
                    if not is_collapsed:
                        for blk in saved_stream_blocks:
                            CollapsibleProgressTracker._render_stream_block_to(c, blk)
            else:
                c.print(
                    f"[bold #0D9488]fix[/bold #0D9488] [bold #10B981]({escape(repo_name)})[/bold #10B981] "
                    f"[bold #F97316]❯[/bold #F97316] {escape(user_prompt)}"
                )
                for blk in saved_stream_blocks:
                    CollapsibleProgressTracker._render_stream_block_to(c, blk)
            _render_session_result(completed_sess, c, show_telemetry=not self._tui_mode)

        return _render_turn_with_prompt

    def open_resume_modal(self) -> bool:
        """Open the Codex-style interactive `/resume` historical conversation picker modal in Full-Screen TUI mode."""
        sessions = self.list_saved_sessions()
        if not sessions:
            self.show_toast("当前目录暂无历史对话记录", duration_sec=3.0)
            return False
        self._slash_popup_dismissed = True
        self._slash_stage2_cmd = ""
        self._shortcut_overlay_open = False
        if self._input_buffer_ref is not None:
            try:
                self._input_buffer_ref.cancel_completion()
                self._input_buffer_ref.reset()
            except Exception:
                pass
        self._modal_state = {
            "mode": "resume_session",
            "sessions": sessions,
            "selected_idx": 0,
            "filter_query": "",
            "error_msg": "",
            "status_msg": "",
        }
        self._scroll_offset_from_bottom = 0
        self._set_terminal_hover_mouse_tracking(True)
        self.show_toast("↑↓/鼠标悬停选择历史对话 · Enter/鼠标点击恢复 · Esc 取消", duration_sec=3.0)
        if self._tui_app is not None:
            try:
                self._tui_app.invalidate()
            except Exception:
                pass
        return True

    def start_new_conversation_session(self) -> str:
        """
        Start a brand-new conversation session in the current directory (Codex `/new` behavior):
        - Persists the previous conversation session if it had any turns so it remains available in `/resume`.
        - Allocates a fresh `session_id` and resets the screen to a clean new conversation view.
        - Preserves the current directory's `FileHistory` (Up/Down prompt history) and active model settings.
        """
        if self._saved_turns_records:
            self._persist_current_conversation_session()

        self._modal_state = None
        self._set_terminal_hover_mouse_tracking(False)
        self._modal_screen_row_to_item.clear()
        self._modal_doc_line_to_item.clear()
        self._sel_anchor = None
        self._sel_head = None
        self._sel_dragging = False
        if self._input_buffer_ref is not None:
            try:
                self._input_buffer_ref.cancel_completion()
                self._input_buffer_ref.reset()
            except Exception:
                pass

        self.conversation_context.clear()
        self._turn_fold_states.clear()
        self._turn_counter = 0
        self._saved_turns_records.clear()
        self.last_session = None
        self._task_elapsed_sec = 0.0
        self.session_id = self._new_conversation_session_id()
        self.session_created_at = time.time()
        self._scroll_offset_from_bottom = 0
        self._cached_completed_key = None
        self._cached_lines_key = None
        self._agent_state_label = "就绪"
        self._screen_blocks = [
            self._render_minimal_header_to if self._tui_mode else self._render_banner_to
        ]
        self.show_toast("✓ 已在当前目录新建对话", duration_sec=3.0)
        if not self._tui_mode:
            self.console.print("[bold #10B981]✓ 已在当前目录新建对话 (原对话已保存，可通过 `/resume` 随时恢复)。[/bold #10B981]")
        if self._tui_app is not None:
            try:
                self._tui_app.invalidate()
            except Exception:
                pass
        return self.session_id

    def resume_conversation_session(
        self,
        session_id_or_index: str | int | None = None,
    ) -> dict[str, Any] | None:
        """
        Restore a saved conversation session for the current directory (Codex `/resume` behavior):
        - Restores all historical turns into `self._screen_blocks`, `self.conversation_context`, and `self._turn_fold_states`.
        - Keeps `self.session_id` bound to the resumed conversation so subsequent prompts append seamlessly.
        - Pins `self._scroll_offset_from_bottom = 0` so the viewport stays positioned at the latest conversation turn!
        """
        sessions = self.list_saved_sessions()
        if not sessions:
            return None

        target: dict[str, Any] | None = None
        if session_id_or_index is None:
            target = sessions[0]
        elif isinstance(session_id_or_index, int):
            if 0 <= session_id_or_index < len(sessions):
                target = sessions[session_id_or_index]
        else:
            q = str(session_id_or_index).strip()
            if not q or q.lower() in ("latest", "last", "newest"):
                target = sessions[0]
            elif q.lstrip("#").isdigit():
                idx_1 = int(q.lstrip("#")) - 1
                if 0 <= idx_1 < len(sessions):
                    target = sessions[idx_1]
            if target is None:
                for cand in sessions:
                    sid = str(cand.get("session_id", ""))
                    if sid == q or sid.startswith(q):
                        target = cand
                        break
            if target is None:
                q_low = q.lower()
                for cand in sessions:
                    if (
                        q_low in str(cand.get("title", "")).lower()
                        or q_low in str(cand.get("latest_prompt", "")).lower()
                    ):
                        target = cand
                        break

        if target is None:
            return None

        # Close any active modal and reset modal mappings
        self._modal_state = None
        self._set_terminal_hover_mouse_tracking(False)
        self._modal_screen_row_to_item.clear()
        self._modal_doc_line_to_item.clear()
        if self._input_buffer_ref is not None:
            try:
                self._input_buffer_ref.reset()
            except Exception:
                pass

        self.session_id = str(target["session_id"])
        self.session_created_at = float(target.get("created_at") or time.time())
        self._saved_turns_records = list(target.get("turns") or [])
        self.conversation_context.clear()
        self._turn_fold_states.clear()
        self._turn_counter = 0
        self._screen_blocks = [
            self._render_minimal_header_to if self._tui_mode else self._render_banner_to
        ]

        for t_rec in self._saved_turns_records:
            self._turn_counter += 1
            t_id = self._turn_counter
            u_prompt = str(t_rec.get("user_prompt", ""))
            f_summary = str(t_rec.get("final_summary", ""))
            mod_files = [str(f) for f in (t_rec.get("modified_files") or [])]
            s_blocks = list(t_rec.get("stream_blocks") or [])
            self.record_input_history(u_prompt)

            restored_sess = TaskSession(
                repo_root=str(self.profile.repo_root),
                user_prompt=u_prompt,
                final_summary=f_summary,
                modified_files=set(mod_files),
                elapsed_sec=float(t_rec.get("elapsed_sec", 0.0)),
                tool_calls_count=int(t_rec.get("tool_calls_count", 0)),
                tool_calls_log=[str(x) for x in (t_rec.get("tool_calls_log") or [])],
                total_tokens=int(t_rec.get("total_tokens", 0)),
                prompt_tokens=int(t_rec.get("prompt_tokens", 0)),
                completion_tokens=int(t_rec.get("completion_tokens", 0)),
                llm_calls=int(t_rec.get("llm_calls", 0)),
            )
            fold_state: dict[str, Any] = {
                "collapsed": True,
                "turn_id": t_id,
                "blocks_count": len(s_blocks),
            }
            if s_blocks:
                self._turn_fold_states[t_id] = fold_state

            turn_block_fn = self._make_turn_render_block(
                turn_id=t_id,
                user_prompt=u_prompt,
                completed_sess=restored_sess,
                saved_stream_blocks=s_blocks,
                fold_state=fold_state,
            )
            self._screen_blocks.append(turn_block_fn)
            self.conversation_context.append(
                {
                    "user": u_prompt,
                    "assistant": f_summary,
                    "modified_files": ", ".join(mod_files),
                }
            )
            self.last_session = restored_sess
            self._task_elapsed_sec = restored_sess.elapsed_sec

        if len(self._screen_blocks) > 13:
            self._screen_blocks = [self._screen_blocks[0], *self._screen_blocks[-12:]]

        # Keep viewport positioned at the bottom on the latest restored conversation turn (Codex style!)
        self._scroll_offset_from_bottom = 0
        self._cached_completed_key = None
        self._cached_lines_key = None
        self._agent_state_label = "就绪"
        n_turns = len(self._saved_turns_records)
        self.show_toast(f"✓ 已恢复历史对话 (共 {n_turns} 轮 · 当前位于最新对话)", duration_sec=3.0)

        if not self._tui_mode:
            for blk_fn in self._screen_blocks[1:]:
                blk_fn(self.console)
            self.console.print(
                f"[bold #10B981]✓ 已恢复历史对话 `{escape(self.session_id)}` (共 {n_turns} 轮)，当前保持在最新对话位置。[/bold #10B981]"
            )

        if self._tui_app is not None:
            try:
                self._tui_app.invalidate()
            except Exception:
                pass
        return target

    def _auto_bootstrap_workspace(self) -> RepoProfile:
        """Silently initialize `.fix/`, invalidate stale landmarks, and scan co-changes on startup."""
        self.memory.invalidate_stale_facts()
        profile = self.context_mgr.detect_repo_profile(self.toolkit.repo_root)
        try:
            self.evolution.scan_git_cochanges(max_commits=80)
            self.evolution.install_git_post_commit_hook()
            self.looper.schedule_startup_idle_events()
        except Exception:
            pass
        return profile

