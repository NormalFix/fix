"""Deterministic scriptable mock LLM gateway (`ScriptedMockLLM`) for automated testing."""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel

from fix.llm.adapters import (
    LLMResponse,
    _estimate_tokens_from_messages,
    _estimate_tokens_from_text,
)
from fix.llm.gateway import LLMGateway, T


class ScriptedMockLLM(LLMGateway):
    """Deterministic scriptable LLM gateway for automated E2E tests and offline verification."""

    def __init__(
        self,
        tool_Queue: list[LLMResponse] | None = None,
        structured_handlers: dict[str, Callable[[list[dict[str, Any]]], BaseModel] | BaseModel] | None = None,
        step_callback: Callable[[list[dict[str, Any]], list[dict[str, Any]] | None], LLMResponse] | None = None,
    ) -> None:
        super().__init__(api_key="mock-key", base_url="http://mock.local")
        self.tool_queue: list[LLMResponse] = list(tool_Queue or [])
        self.structured_handlers = structured_handlers or {}
        self.step_callback = step_callback
        self.recorded_messages: list[list[dict[str, Any]]] = []
        self.structured_call_counts: dict[str, int] = {}

    async def _single_completion_call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        use_reasoning: bool = False,
        temperature: float = 0.1,
        max_tokens: int | None = 32768,
    ) -> LLMResponse:
        self.recorded_messages.append(list(messages))
        if self.step_callback is not None:
            resp = self.step_callback(messages, tools)
        elif self.tool_queue:
            resp = self.tool_queue.pop(0)
        else:
            resp = LLMResponse(content="Task completed.", tool_calls=[])

        p_tok = resp.prompt_tokens or _estimate_tokens_from_messages(messages)
        c_tok = resp.completion_tokens or (
            _estimate_tokens_from_text(resp.content) + len(resp.tool_calls) * 15
        )
        t_tok = resp.total_tokens or (p_tok + c_tok)
        return resp.model_copy(
            update={
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "total_tokens": t_tok,
            }
        )

    async def complete_structured(
        self,
        messages: list[dict[str, Any]],
        response_model: type[T],
        use_reasoning: bool = False,
        temperature: float = 0.0,
        max_tokens: int | None = 2048,
    ) -> T:
        model_name = response_model.__name__
        self.structured_call_counts[model_name] = self.structured_call_counts.get(model_name, 0) + 1
        handler = self.structured_handlers.get(model_name)
        if handler is not None:
            if callable(handler):
                result = handler(messages)
            elif isinstance(handler, list):
                result = handler.pop(0) if len(handler) > 1 else handler[0]
            else:
                result = handler
            p_tok = _estimate_tokens_from_messages(messages)
            c_tok = 45
            self.last_usage = (p_tok, c_tok, p_tok + c_tok)
            return result  # type: ignore[return-value]
        raise RuntimeError(f"No mock handler registered for structured model {model_name}")
