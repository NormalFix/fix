"""Protocol adapters for OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages APIs."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

from fix.llm.catalog import PROVIDER_SPECS, resolve_provider_id
from fix.protocol.schema import ToolCallRequest


class LLMResponse(BaseModel):
    """Normalized response from LLMGateway.complete_with_tools."""

    content: str = ""
    reasoning_content: str = ""
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    model: str = ""
    finish_reason: str = ""
    reasoning_tokens: int = 0
    continuation_rounds: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0



def _build_provider_headers(provider_id: str, api_key: str) -> dict[str, str]:
    """Construct HTTP headers tailored to the specific LLM vendor's authentication convention."""
    spec = PROVIDER_SPECS.get(provider_id, PROVIDER_SPECS["mimo"])
    auth_style = spec.get("auth_style", "bearer")
    if auth_style == "anthropic_x_api_key":
        return {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
    if auth_style == "mimo_hybrid":
        return {
            "Authorization": f"Bearer {api_key}",
            "api-key": api_key,
            "Content-Type": "application/json",
        }
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _extract_json_payload(raw_text: str) -> str:
    """Strip markdown code fences and isolate the outermost JSON object."""
    cleaned = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return cleaned[first_brace : last_brace + 1]
    return cleaned


def _estimate_tokens_from_messages(messages: list[dict[str, Any]]) -> int:
    total_chars = sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)
    return max(1, total_chars // 3)


def _estimate_tokens_from_text(text: str) -> int:
    return max(1, len(text) // 3) if text else 0



def _convert_openai_to_anthropic_payload(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    model: str,
    temperature: float,
    max_tokens: int | None,
) -> dict[str, Any]:
    """
    Convert OpenAI-style chat messages & tool schemas into Anthropic `/v1/messages` request format:
    - Extracts all `system` role messages into top-level `"system"`.
    - Converts `assistant` `tool_calls` into `{"type": "tool_use", "id", "name", "input"}` blocks.
    - Converts `tool` role outputs into `{"type": "tool_result", "tool_use_id", "content"}` blocks inside a `"user"` turn.
    - Merges consecutive messages of the same role so user/assistant turns strictly alternate.
    """
    system_parts: list[str] = []
    anth_messages: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content") or ""
        if role == "system":
            if content:
                system_parts.append(str(content))
            continue

        if role == "tool":
            tool_block = {
                "type": "tool_result",
                "tool_use_id": str(msg.get("tool_call_id") or "call_0"),
                "content": str(content),
            }
            if anth_messages and anth_messages[-1]["role"] == "user" and isinstance(anth_messages[-1]["content"], list):
                anth_messages[-1]["content"].append(tool_block)
            else:
                anth_messages.append({"role": "user", "content": [tool_block]})
            continue

        if role == "assistant":
            blocks: list[dict[str, Any]] = []
            if content:
                blocks.append({"type": "text", "text": str(content)})
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                raw_args = fn.get("arguments", "{}")
                try:
                    input_dict = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except Exception:
                    input_dict = {}
                if not isinstance(input_dict, dict):
                    input_dict = {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": str(tc.get("id") or "call_0"),
                        "name": str(fn.get("name") or ""),
                        "input": input_dict,
                    }
                )
            if not blocks:
                blocks.append({"type": "text", "text": ""})
            if anth_messages and anth_messages[-1]["role"] == "assistant":
                prev_c = anth_messages[-1]["content"]
                if isinstance(prev_c, str):
                    anth_messages[-1]["content"] = [{"type": "text", "text": prev_c}, *blocks]
                else:
                    prev_c.extend(blocks)
            else:
                anth_messages.append({"role": "assistant", "content": blocks})
            continue

        # Default: user message
        user_block = {"type": "text", "text": str(content)}
        if anth_messages and anth_messages[-1]["role"] == "user":
            prev_u = anth_messages[-1]["content"]
            if isinstance(prev_u, str):
                anth_messages[-1]["content"] = [{"type": "text", "text": prev_u}, user_block]
            else:
                prev_u.append(user_block)
        else:
            anth_messages.append({"role": "user", "content": [user_block]})

    if not anth_messages:
        anth_messages.append({"role": "user", "content": [{"type": "text", "text": "Hello"}]})

    eff_max_tokens = min(max_tokens or 16384, 16384)
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": eff_max_tokens,
        "messages": anth_messages,
    }
    # Claude 4.7, 4.8, and 5.x (Opus 5.5, Sonnet 5.5, Haiku 5.5, Fable 5.1) use adaptive thinking
    # and reject `temperature` with HTTP 400 on `/v1/messages`. Only include `temperature` for legacy models.
    m_low = model.lower()
    if not any(tag in m_low for tag in ("-5", "4-7", "4-8", "fable", "mythos")):
        payload["temperature"] = max(0.0, min(1.0, float(temperature)))
    if system_parts:
        payload["system"] = "\n\n".join(system_parts)

    if tools:
        anth_tools: list[dict[str, Any]] = []
        for t in tools:
            fn = t.get("function") if "function" in t else t
            if not isinstance(fn, dict):
                continue
            anth_tools.append(
                {
                    "name": str(fn.get("name") or ""),
                    "description": str(fn.get("description") or ""),
                    "input_schema": fn.get("parameters") or fn.get("input_schema") or {"type": "object", "properties": {}},
                }
            )
        if anth_tools:
            payload["tools"] = anth_tools

    return payload


def _parse_anthropic_response(
    data: dict[str, Any],
    model: str,
    messages: list[dict[str, Any]],
) -> LLMResponse:
    """Parse Anthropic `/v1/messages` response JSON into normalized `LLMResponse`."""
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    parsed_calls: list[ToolCallRequest] = []

    for blk in data.get("content") or []:
        if not isinstance(blk, dict):
            continue
        btype = blk.get("type")
        if btype == "text":
            txt = blk.get("text") or ""
            if txt:
                content_parts.append(str(txt))
        elif btype in ("thinking", "redacted_thinking"):
            th = blk.get("thinking") or blk.get("text") or ""
            if th:
                reasoning_parts.append(str(th))
        elif btype == "tool_use":
            raw_in = blk.get("input")
            args_dict = raw_in if isinstance(raw_in, dict) else {}
            parsed_calls.append(
                ToolCallRequest(
                    call_id=str(blk.get("id") or f"call_{len(parsed_calls)}"),
                    name=str(blk.get("name") or ""),
                    arguments=args_dict,
                )
            )

    stop_reason = str(data.get("stop_reason") or "")
    if stop_reason == "max_tokens":
        finish_reason = "length"
    elif stop_reason == "tool_use":
        finish_reason = "tool_calls"
    else:
        finish_reason = stop_reason or "stop"

    content_str = "\n".join(content_parts)
    reasoning_str = "\n".join(reasoning_parts)
    usage = data.get("usage") or {}
    p_tok = int(usage.get("input_tokens") or 0) or _estimate_tokens_from_messages(messages)
    c_tok = int(usage.get("output_tokens") or 0) or (
        _estimate_tokens_from_text(content_str + reasoning_str) + len(parsed_calls) * 18
    )
    r_tok = _estimate_tokens_from_text(reasoning_str) if reasoning_str else 0

    return LLMResponse(
        content=content_str,
        reasoning_content=reasoning_str,
        tool_calls=parsed_calls,
        model=str(data.get("model") or model),
        finish_reason=finish_reason,
        reasoning_tokens=r_tok,
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
        total_tokens=p_tok + c_tok,
    )

