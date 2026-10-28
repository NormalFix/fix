"""Multi-provider LLM gateway, vendor catalog, and protocol adapters."""

from __future__ import annotations

from fix.llm.adapters import (
    LLMResponse,
    _build_provider_headers,
    _convert_openai_to_anthropic_payload,
    _estimate_tokens_from_messages,
    _estimate_tokens_from_text,
    _extract_json_payload,
    _parse_anthropic_response,
)
from fix.llm.catalog import (
    PROVIDER_ORDER,
    PROVIDER_SPECS,
    _NON_CHAT_MODEL_KEYWORDS,
    _PROVIDER_ALIASES,
    _SHELL_RC_EXPORT_RE,
    _detect_default_provider,
    _filter_and_merge_provider_models,
    _load_fallback_config_from_shell_rc,
    infer_provider_from_url_or_model,
    resolve_provider_id,
)
from fix.llm.gateway import LLMGateway
from fix.llm.mock import ScriptedMockLLM

__all__ = [
    "LLMGateway",
    "LLMResponse",
    "PROVIDER_ORDER",
    "PROVIDER_SPECS",
    "ScriptedMockLLM",
    "_filter_and_merge_provider_models",
    "_load_fallback_config_from_shell_rc",
    "_detect_default_provider",
    "infer_provider_from_url_or_model",
    "resolve_provider_id",
]
