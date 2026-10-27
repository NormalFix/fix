"""Multi-provider async LLM Gateway supporting DeepSeek, GLM, MiMo, MiniMax, Qwen, OpenAI & Anthropic."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, TypeVar

import httpx
from pydantic import BaseModel, Field

from fix.schema import ToolCallRequest

T = TypeVar("T", bound=BaseModel)


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


# Official API specifications for the 7 supported LLM vendors (synced with latest 2026 official API docs)
PROVIDER_SPECS: dict[str, dict[str, Any]] = {
    "deepseek": {
        "provider_id": "deepseek",
        "display_name": "DeepSeek",
        "vendor_label": "DeepSeek (深度求索)",
        "base_url": "https://api.deepseek.com/v1",
        "alt_base_urls": ["https://api.deepseek.com/v1", "https://api.deepseek.com"],
        "protocol": "openai_chat",
        "auth_style": "bearer",
        "env_keys": ["DEEPSEEK_API_KEY"],
        "default_fast_model": "deepseek-flash",
        "default_reasoning_model": "deepseek-v4-pro",
        "curated_models": [
            "deepseek-flash",
            "deepseek-v4-pro",
        ],
        "retired_models": [
            "deepseek-chat",
            "deepseek-reasoner",
            "deepseek-coder",
            "deepseek-v4-flash",
            "deepseek-v4-flash-vision-exp",
            "deepseek-v3",
            "deepseek-r1",
        ],
        "api_notes": "OpenAI 兼容 (/v1/chat/completions, /v1/models) · 1M 上下文 · DeepSeek-V4.1-Flash / V4-Pro",
    },
    "glm": {
        "provider_id": "glm",
        "display_name": "GLM",
        "vendor_label": "GLM (智谱清言 BigModel)",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "alt_base_urls": ["https://open.bigmodel.cn/api/paas/v4"],
        "protocol": "openai_chat",
        "auth_style": "bearer",
        "env_keys": ["ZHIPUAI_API_KEY", "GLM_API_KEY", "ZHIPU_API_KEY"],
        "default_fast_model": "glm-5.3-flash",
        "default_reasoning_model": "glm-5.3",
        "curated_models": [
            "glm-5.3",
            "glm-5.3-flash",
            "glm-5.3-flashx",
            "glm-5.2",
            "glm-5.1",
            "glm-5",
            "glm-5-turbo",
            "glm-5v-turbo",
            "glm-4.7",
            "glm-4.7-flashx",
            "glm-4.7-flash",
            "glm-4.6",
        ],
        "retired_models": [
            "glm-4.5",
            "glm-4.5-air",
            "glm-4.5-x",
            "glm-4.5-airx",
            "glm-4.5-flash",
            "glm-4-plus",
            "glm-4-air",
            "glm-4-flash",
            "glm-z1-air",
            "glm-z1-flash",
            "glm-zero-preview",
        ],
        "api_notes": "智谱 BigModel v4 接口 (/api/paas/v4/chat/completions) · 1M 上下文 · GLM-5.3 旗舰系列",
    },
    "mimo": {
        "provider_id": "mimo",
        "display_name": "MiMo",
        "vendor_label": "MiMo (小米大模型)",
        "base_url": "https://token-plan-cn.xiaomimimo.com/v1",
        "alt_base_urls": [
            "https://token-plan-cn.xiaomimimo.com/v1",
            "https://api.xiaomimimo.com/v1",
        ],
        "protocol": "openai_chat",
        "auth_style": "mimo_hybrid",
        "env_keys": ["MIMO_API_KEY", "XIAOMI_API_KEY"],
        "default_fast_model": "mimo-v2.6-flash",
        "default_reasoning_model": "mimo-v2.6-pro",
        "curated_models": [
            "mimo-v2.6-pro",
            "mimo-v2.6-flash",
            "mimo-v2.6-pro-ultraspeed",
        ],
        "retired_models": [
            "mimo-v2.5-pro",
            "mimo-v2.5",
            "mimo-v2-pro",
            "mimo-v2-flash",
            "mimo-vl-7b",
        ],
        "api_notes": "小米 MiMo 双端点自适应 (Token Plan & api.xiaomimimo.com) · 1M 上下文 · MiMo-V2.6 系列",
    },
    "minimax": {
        "provider_id": "minimax",
        "display_name": "MiniMax",
        "vendor_label": "MiniMax (稀宇科技)",
        "base_url": "https://api.minimax.chat/v1",
        "alt_base_urls": [
            "https://api.minimax.chat/v1",
            "https://api.minimaxi.com/v1",
            "https://api.minimax.io/v1",
        ],
        "protocol": "openai_chat",
        "auth_style": "bearer",
        "env_keys": ["MINIMAX_API_KEY"],
        "default_fast_model": "MiniMax-M3.1-Flash",
        "default_reasoning_model": "MiniMax-M3",
        "curated_models": [
            "MiniMax-M3.1-Flash",
            "MiniMax-M3",
        ],
        "retired_models": [
            "MiniMax-M1",
            "MiniMax-Text-01",
            "abab6.5s-chat",
            "DeepSeek-R1",
            "MiniMax-M2",
            "MiniMax-M2.1",
            "MiniMax-M2.1-highspeed",
            "MiniMax-M2.5",
            "MiniMax-M2.5-highspeed",
            "MiniMax-M2.7",
            "MiniMax-M2.7-highspeed",
            "M2-her",
        ],
        "api_notes": "MiniMax OpenAI 兼容 (/v1/chat/completions) · 1M 上下文 · MiniMax-M3.1-Flash / M3",
    },
    "qwen": {
        "provider_id": "qwen",
        "display_name": "Qwen",
        "vendor_label": "Qwen (阿里通义千问 DashScope)",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "alt_base_urls": [
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        ],
        "protocol": "openai_chat",
        "auth_style": "bearer",
        "env_keys": ["DASHSCOPE_API_KEY", "QWEN_API_KEY"],
        "default_fast_model": "qwen3.7-plus",
        "default_reasoning_model": "qwen3.8-max",
        "curated_models": [
            "qwen3.8-max",
            "qwen3.7-plus",
            "qwen3.8-flash",
            "qwen3.7-flash",
            "qwen3.7-max",
            "qwen3-coder-plus",
            "qwen3-coder-flash",
            "qwen-plus",
            "qwen-flash",
            "qwen-max",
        ],
        "retired_models": [
            "qwen3-max",
            "qwen-turbo",
            "qwen-coder-plus",
            "qwq-plus",
            "qwq-32b",
            "qwen2.5-coder-32b-instruct",
        ],
        "api_notes": "阿里云百炼 DashScope 兼容接口 (/compatible-mode/v1/chat/completions) · 1M 上下文 · Qwen3.8 / 3.7 系列",
    },
    "openai": {
        "provider_id": "openai",
        "display_name": "OpenAI",
        "vendor_label": "OpenAI (GPT-6 / GPT-5)",
        "base_url": "https://api.openai.com/v1",
        "alt_base_urls": ["https://api.openai.com/v1"],
        "protocol": "openai_chat",
        "auth_style": "bearer",
        "env_keys": ["OPENAI_API_KEY"],
        "default_fast_model": "gpt-6.1-sol",
        "default_reasoning_model": "gpt-6-astra",
        "curated_models": [
            "gpt-6-astra",
            "gpt-6.1-sol",
            "gpt-6.1-terra",
            "gpt-6-luna",
            "gpt-6-sol",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-5.6-luna",
            "gpt-5.5",
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.3-codex",
        ],
        "retired_models": [
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
            "o1",
            "o1-mini",
            "o1-pro",
            "o3",
            "o3-mini",
            "o3-pro",
            "o4-mini",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-5-pro",
        ],
        "api_notes": "OpenAI 官方接口 (/v1/chat/completions & /v1/responses) · 1.05M 上下文 · GPT-6 Astra / 6.1 Sol",
    },
    "anthropic": {
        "provider_id": "anthropic",
        "display_name": "Anthropic",
        "vendor_label": "Anthropic (Claude)",
        "base_url": "https://api.anthropic.com/v1",
        "alt_base_urls": ["https://api.anthropic.com/v1"],
        "protocol": "anthropic_messages",
        "auth_style": "anthropic_x_api_key",
        "env_keys": ["ANTHROPIC_API_KEY"],
        "default_fast_model": "claude-sonnet-5-5",
        "default_reasoning_model": "claude-opus-5-5",
        "curated_models": [
            "claude-opus-5-5",
            "claude-sonnet-5-5",
            "claude-fable-5-1",
            "claude-haiku-5-5",
            "claude-opus-5",
            "claude-sonnet-5",
            "claude-opus-4-8",
            "claude-sonnet-4-6",
            "claude-opus-4-6",
            "claude-haiku-4-5",
        ],
        "retired_models": [
            "claude-sonnet-4-5-20250929",
            "claude-opus-4-1-20250805",
            "claude-3-7-sonnet-latest",
            "claude-3-5-sonnet-latest",
            "claude-3-5-haiku-latest",
            "claude-3-opus-20240229",
        ],
        "api_notes": "Anthropic 原生接口 (/v1/messages, /v1/models) · 1M 上下文 · Claude 5.5 / Fable 5.1 系列",
    },
}

_NON_CHAT_MODEL_KEYWORDS: tuple[str, ...] = (
    "embedding",
    "tts",
    "whisper",
    "transcribe",
    "moderation",
    "rerank",
    "dall-e",
    "gpt-image",
    "image-",
    "cogview",
    "cogvideo",
    "vidu",
    "speech-",
    "music-",
    "hailuo",
    "babbage",
    "davinci",
    "qwen-audio",
    "qwen-mt",
    "glm-asr",
    "glm-tts",
    "glm-ocr",
    "glm-image",
    "wan2",
    "wan3",
    "tripo",
    "happyhorse",
    "happyoyster",
)


def _filter_and_merge_provider_models(provider_id: str, fetched_ids: list[str]) -> list[str]:
    """
    Merge a provider's official `curated_models` with live `/models` IDs while filtering out
    retired/deprecated model names (e.g. `deepseek-chat`, `deepseek-reasoner`, `MiniMax-M1`)
    and non-chat models (embeddings, TTS, ASR, image, video, moderation, rerank).
    Places `curated_models` first so the vendor's latest recommended models always lead the list.
    """
    spec = PROVIDER_SPECS.get(provider_id, PROVIDER_SPECS["mimo"])
    curated = list(spec.get("curated_models") or [])
    retired_lower = {str(m).lower() for m in (spec.get("retired_models") or [])}
    valid_fetched: list[str] = []
    for raw_id in fetched_ids:
        m = str(raw_id or "").strip()
        if not m:
            continue
        m_low = m.lower()
        if m_low in retired_lower:
            continue
        if any(kw in m_low for kw in _NON_CHAT_MODEL_KEYWORDS):
            continue
        valid_fetched.append(m)
    return list(dict.fromkeys([*curated, *valid_fetched]))

PROVIDER_ORDER: list[str] = [
    "deepseek",
    "glm",
    "mimo",
    "minimax",
    "qwen",
    "openai",
    "anthropic",
]

_PROVIDER_ALIASES: dict[str, str] = {
    "deepseek": "deepseek",
    "深度求索": "deepseek",
    "ds": "deepseek",
    "glm": "glm",
    "zhipu": "glm",
    "bigmodel": "glm",
    "智谱": "glm",
    "智谱清言": "glm",
    "mimo": "mimo",
    "xiaomi": "mimo",
    "小米": "mimo",
    "minimax": "minimax",
    "minimaxi": "minimax",
    "稀宇": "minimax",
    "qwen": "qwen",
    "dashscope": "qwen",
    "aliyun": "qwen",
    "通义": "qwen",
    "通义千问": "qwen",
    "千问": "qwen",
    "openai": "openai",
    "gpt": "openai",
    "anthropic": "anthropic",
    "claude": "anthropic",
}


def resolve_provider_id(raw: str) -> str | None:
    """Resolve user input (provider ID, display name, Chinese alias, or 1-7 index) to canonical provider_id."""
    cleaned = raw.strip().lower()
    if not cleaned:
        return None
    if cleaned.isdigit():
        idx = int(cleaned) - 1
        if 0 <= idx < len(PROVIDER_ORDER):
            return PROVIDER_ORDER[idx]
    if cleaned in PROVIDER_SPECS:
        return cleaned
    if cleaned in _PROVIDER_ALIASES:
        return _PROVIDER_ALIASES[cleaned]
    for pid, spec in PROVIDER_SPECS.items():
        if cleaned in spec["display_name"].lower() or cleaned in spec["vendor_label"].lower():
            return pid
    return None


def infer_provider_from_url_or_model(base_url: str, model: str = "") -> str:
    """Infer canonical provider_id from base_url or model name."""
    u = (base_url or "").lower()
    m = (model or "").lower()
    if "deepseek.com" in u or m.startswith("deepseek"):
        return "deepseek"
    if "bigmodel.cn" in u or "zhipu" in u or m.startswith("glm"):
        return "glm"
    if "xiaomimimo.com" in u or m.startswith("mimo"):
        return "mimo"
    if "minimax" in u or m.startswith(("minimax", "abab")):
        return "minimax"
    if "dashscope" in u or "aliyuncs.com" in u or m.startswith(("qwen", "qwq")):
        return "qwen"
    if "anthropic.com" in u or m.startswith("claude"):
        return "anthropic"
    if "openai.com" in u or m.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    return "mimo"


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


_SHELL_RC_EXPORT_RE = re.compile(
    r"^\s*(?:export\s+)?(FIX_API_KEY|MIMO_API_KEY|XIAOMI_API_KEY|DEEPSEEK_API_KEY|ZHIPUAI_API_KEY|GLM_API_KEY|ZHIPU_API_KEY|MINIMAX_API_KEY|DASHSCOPE_API_KEY|QWEN_API_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY|FIX_BASE_URL|MIMO_BASE_URL|FIX_MODEL|FIX_REASONING_MODEL)\s*=\s*['\"]?([^'\"#\r\n]+)['\"]?"
)


def _load_fallback_config_from_shell_rc() -> dict[str, str]:
    """
    Read FIX_API_KEY / MIMO_API_KEY / FIX_BASE_URL / FIX_MODEL / FIX_REASONING_MODEL from
    `~/.zshrc`, `~/.zshenv`, or `~/.bash_profile` if the current terminal session was opened
    before those exports were added (and only when `PYTEST_CURRENT_TEST` is not set).
    """
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return {}
    found: dict[str, str] = {}
    home = Path.home()
    for rc_name in (".zshrc", ".zshenv", ".bash_profile", ".bashrc"):
        rc_path = home / rc_name
        try:
            if not rc_path.is_file():
                continue
            for line in rc_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                m = _SHELL_RC_EXPORT_RE.match(line)
                if m:
                    k, v = m.group(1), m.group(2).strip()
                    if v and not v.startswith("$"):
                        found[k] = v
        except Exception:
            continue
    return found


def _detect_default_provider(
    api_key: str,
    explicit_base_url: str | None,
    rc_cfg: dict[str, str] | None = None,
) -> tuple[str, str, str]:
    """
    Auto-detect base_url, fast_model, and reasoning_model based on API key format or environment.
    Defaults to Xiaomi MiMo Token Plan (`https://token-plan-cn.xiaomimimo.com/v1`,
    `mimo-v2.6-flash` / `mimo-v2.6-pro`), while also supporting all 7 official LLM vendors.
    """
    rc = rc_cfg or {}

    def _has_key(*names: str) -> bool:
        return any(bool(os.environ.get(n) or rc.get(n)) for n in names)

    has_fix_key = _has_key("FIX_API_KEY")
    is_mimo_token_plan = api_key.startswith(("tp-", "ttp-"))
    has_mimo_env = _has_key("MIMO_API_KEY", "XIAOMI_API_KEY")
    has_openai_only = _has_key("OPENAI_API_KEY") and not (
        has_fix_key or has_mimo_env or is_mimo_token_plan
    )

    if explicit_base_url:
        base_url = explicit_base_url.rstrip("/")
    elif is_mimo_token_plan:
        base_url = "https://token-plan-cn.xiaomimimo.com/v1"
    elif has_mimo_env:
        base_url = "https://api.xiaomimimo.com/v1"
    elif _has_key("DEEPSEEK_API_KEY") and not has_fix_key:
        base_url = PROVIDER_SPECS["deepseek"]["base_url"]
    elif _has_key("ZHIPUAI_API_KEY", "GLM_API_KEY", "ZHIPU_API_KEY") and not has_fix_key:
        base_url = PROVIDER_SPECS["glm"]["base_url"]
    elif _has_key("MINIMAX_API_KEY") and not has_fix_key:
        base_url = PROVIDER_SPECS["minimax"]["base_url"]
    elif _has_key("DASHSCOPE_API_KEY", "QWEN_API_KEY") and not has_fix_key:
        base_url = PROVIDER_SPECS["qwen"]["base_url"]
    elif (_has_key("ANTHROPIC_API_KEY") or api_key.startswith("sk-ant-")) and not has_fix_key:
        base_url = PROVIDER_SPECS["anthropic"]["base_url"]
    elif has_openai_only:
        base_url = PROVIDER_SPECS["openai"]["base_url"]
    else:
        base_url = "https://token-plan-cn.xiaomimimo.com/v1"

    pid = infer_provider_from_url_or_model(base_url)
    spec = PROVIDER_SPECS.get(pid, PROVIDER_SPECS["mimo"])
    return base_url, spec["default_fast_model"], spec["default_reasoning_model"]


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


class LLMGateway:
    """Zero-framework async LLM client supporting DeepSeek, GLM, MiMo, MiniMax, Qwen, OpenAI & Anthropic."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        fast_model: str | None = None,
        reasoning_model: str | None = None,
        timeout_sec: float = 150.0,
        providers_config_path: Path | None = None,
    ) -> None:
        self._providers_config_path: Path | None = providers_config_path or (
            None if os.environ.get("PYTEST_CURRENT_TEST") else Path.home() / ".fix" / "llm_providers.json"
        )
        self.connected_providers: dict[str, dict[str, Any]] = {}
        saved_active: dict[str, Any] = {}
        if self._providers_config_path is not None:
            saved_active = self._load_persisted_providers()

        rc_cfg = _load_fallback_config_from_shell_rc() if not api_key else {}
        self.api_key = (
            api_key
            or saved_active.get("api_key")
            or os.environ.get("FIX_API_KEY")
            or os.environ.get("MIMO_API_KEY")
            or os.environ.get("XIAOMI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("ZHIPUAI_API_KEY")
            or os.environ.get("GLM_API_KEY")
            or os.environ.get("ZHIPU_API_KEY")
            or os.environ.get("MINIMAX_API_KEY")
            or os.environ.get("DASHSCOPE_API_KEY")
            or os.environ.get("QWEN_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
            or rc_cfg.get("FIX_API_KEY")
            or rc_cfg.get("MIMO_API_KEY")
            or rc_cfg.get("XIAOMI_API_KEY")
            or rc_cfg.get("DEEPSEEK_API_KEY")
            or rc_cfg.get("ZHIPUAI_API_KEY")
            or rc_cfg.get("GLM_API_KEY")
            or rc_cfg.get("ZHIPU_API_KEY")
            or rc_cfg.get("MINIMAX_API_KEY")
            or rc_cfg.get("DASHSCOPE_API_KEY")
            or rc_cfg.get("QWEN_API_KEY")
            or rc_cfg.get("OPENAI_API_KEY")
            or rc_cfg.get("ANTHROPIC_API_KEY")
            or ""
        )
        env_base = (
            base_url
            or saved_active.get("base_url")
            or os.environ.get("FIX_BASE_URL")
            or os.environ.get("MIMO_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or rc_cfg.get("FIX_BASE_URL")
            or rc_cfg.get("MIMO_BASE_URL")
        )
        detected_base, default_fast, default_reasoning = _detect_default_provider(self.api_key, env_base, rc_cfg)
        self.base_url = detected_base
        raw_fast = (
            fast_model
            or saved_active.get("fast_model")
            or os.environ.get("FIX_MODEL")
            or rc_cfg.get("FIX_MODEL")
            or default_fast
        )
        raw_reasoning = (
            reasoning_model
            or saved_active.get("reasoning_model")
            or os.environ.get("FIX_REASONING_MODEL")
            or rc_cfg.get("FIX_REASONING_MODEL")
            or default_reasoning
        )
        self.provider_id: str = saved_active.get("provider_id") or infer_provider_from_url_or_model(
            self.base_url, raw_fast
        )
        if self.provider_id in PROVIDER_SPECS and fast_model is None:
            spec_init = PROVIDER_SPECS[self.provider_id]
            retired_init = {str(r).lower() for r in (spec_init.get("retired_models") or [])}
            if str(raw_fast).lower() in retired_init:
                raw_fast = spec_init["default_fast_model"]
            if reasoning_model is None and str(raw_reasoning).lower() in retired_init:
                raw_reasoning = spec_init["default_reasoning_model"]
        self.fast_model = raw_fast
        self.reasoning_model = raw_reasoning
        self.timeout_sec = timeout_sec
        self.last_usage: tuple[int, int, int] = (0, 0, 0)

        # Register the initial configured provider into `connected_providers` if API key is present
        if self.api_key and self.provider_id in PROVIDER_SPECS and self.provider_id not in self.connected_providers:
            init_models = _filter_and_merge_provider_models(
                self.provider_id, [self.fast_model, self.reasoning_model]
            )
            self.connected_providers[self.provider_id] = {
                "provider_id": self.provider_id,
                "api_key": self.api_key,
                "base_url": self.base_url,
                "models": init_models,
            }

    def _load_persisted_providers(self) -> dict[str, Any]:
        if self._providers_config_path is None or not self._providers_config_path.is_file():
            return {}
        try:
            raw = json.loads(self._providers_config_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                dirty = False
                conns = raw.get("connected_providers")
                if isinstance(conns, dict):
                    for pid, conn in list(conns.items()):
                        if isinstance(conn, dict) and pid in PROVIDER_SPECS:
                            old_list = list(conn.get("models") or [])
                            new_list = _filter_and_merge_provider_models(pid, old_list)
                            if new_list != old_list:
                                conn["models"] = new_list
                                dirty = True
                    self.connected_providers = conns
                active = raw.get("active")
                if isinstance(active, dict):
                    act_pid = str(
                        active.get("provider_id")
                        or infer_provider_from_url_or_model(
                            str(active.get("base_url") or ""),
                            str(active.get("fast_model") or ""),
                        )
                    )
                    if act_pid in PROVIDER_SPECS:
                        spec = PROVIDER_SPECS[act_pid]
                        retired_low = {str(r).lower() for r in (spec.get("retired_models") or [])}
                        if str(active.get("fast_model") or "").lower() in retired_low:
                            active["fast_model"] = spec["default_fast_model"]
                            dirty = True
                        if str(active.get("reasoning_model") or "").lower() in retired_low:
                            active["reasoning_model"] = spec["default_reasoning_model"]
                            dirty = True
                    if dirty:
                        try:
                            self._providers_config_path.write_text(
                                json.dumps(
                                    {"active": active, "connected_providers": self.connected_providers},
                                    ensure_ascii=False,
                                    indent=2,
                                ),
                                encoding="utf-8",
                            )
                        except Exception:
                            pass
                    return active
        except Exception:
            pass
        return {}

    def _save_persisted_providers(self) -> None:
        if self._providers_config_path is None:
            return
        try:
            self._providers_config_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "active": {
                    "provider_id": self.provider_id,
                    "api_key": self.api_key,
                    "base_url": self.base_url,
                    "fast_model": self.fast_model,
                    "reasoning_model": self.reasoning_model,
                },
                "connected_providers": self.connected_providers,
            }
            self._providers_config_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            try:
                os.chmod(self._providers_config_path, 0o600)
            except Exception:
                pass
        except Exception:
            pass

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def get_provider_spec(self, provider_id: str | None = None) -> dict[str, Any]:
        pid = provider_id or self.provider_id or infer_provider_from_url_or_model(self.base_url, self.fast_model)
        return PROVIDER_SPECS.get(pid, PROVIDER_SPECS["mimo"])

    def get_provider_models(self, provider_id: str | None = None) -> list[str]:
        """Return available models for `provider_id` (combining curated vendor models and filtered live API models)."""
        pid = provider_id or self.provider_id or infer_provider_from_url_or_model(self.base_url, self.fast_model)
        conn = self.connected_providers.get(pid) or {}
        live_models = list(conn.get("models") or [])
        extras: list[str] = []
        if pid == self.provider_id:
            if self.fast_model:
                extras.append(self.fast_model)
            if self.reasoning_model:
                extras.append(self.reasoning_model)
        merged = _filter_and_merge_provider_models(pid, [*live_models, *extras])
        return [m for m in merged if m]

    def list_supported_providers(self) -> list[dict[str, Any]]:
        """List all 7 supported LLM vendors with connection status, protocol notes, and models."""
        active_pid = self.provider_id or infer_provider_from_url_or_model(self.base_url, self.fast_model)
        items: list[dict[str, Any]] = []
        for idx, pid in enumerate(PROVIDER_ORDER, start=1):
            spec = PROVIDER_SPECS[pid]
            conn = self.connected_providers.get(pid)
            is_connected = bool(conn and conn.get("api_key")) or (pid == active_pid and bool(self.api_key))
            items.append(
                {
                    "index": idx,
                    "provider_id": pid,
                    "display_name": spec["display_name"],
                    "vendor_label": spec["vendor_label"],
                    "base_url": (conn.get("base_url") if conn else None) or spec["base_url"],
                    "protocol": spec["protocol"],
                    "api_notes": spec["api_notes"],
                    "connected": is_connected,
                    "active": pid == active_pid,
                    "models": self.get_provider_models(pid),
                    "default_fast_model": spec["default_fast_model"],
                    "default_reasoning_model": spec["default_reasoning_model"],
                }
            )
        return items

    def list_switchable_models(self) -> list[dict[str, Any]]:
        """
        Return a flat list of switchable models for `/model`:
        - First lists all models of the currently active provider.
        - Then lists models of any other connected providers.
        - If no other providers are connected yet, also includes the default models of all 7 supported providers
          (marked `connected=False`) so the user can see what Fix supports.
        """
        active_pid = self.provider_id or infer_provider_from_url_or_model(self.base_url, self.fast_model)
        result: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()

        # 1. Active provider models first
        active_spec = self.get_provider_spec(active_pid)
        for m_name in self.get_provider_models(active_pid):
            key = (active_pid, m_name)
            if key in seen:
                continue
            seen.add(key)
            is_current_fast = m_name == self.fast_model
            is_current_reas = m_name == self.reasoning_model
            result.append(
                {
                    "provider_id": active_pid,
                    "provider_name": active_spec["display_name"],
                    "vendor_label": active_spec["vendor_label"],
                    "model": m_name,
                    "base_url": self.base_url,
                    "connected": bool(self.api_key),
                    "is_active_provider": True,
                    "is_current_fast": is_current_fast,
                    "is_current_reasoning": is_current_reas,
                    "is_current": is_current_fast or is_current_reas,
                }
            )

        # 2. Other connected providers
        for pid in PROVIDER_ORDER:
            if pid == active_pid:
                continue
            conn = self.connected_providers.get(pid)
            if not (conn and conn.get("api_key")):
                continue
            spec = PROVIDER_SPECS[pid]
            for m_name in self.get_provider_models(pid):
                key = (pid, m_name)
                if key in seen:
                    continue
                seen.add(key)
                result.append(
                    {
                        "provider_id": pid,
                        "provider_name": spec["display_name"],
                        "vendor_label": spec["vendor_label"],
                        "model": m_name,
                        "base_url": conn.get("base_url") or spec["base_url"],
                        "connected": True,
                        "is_active_provider": False,
                        "is_current_fast": False,
                        "is_current_reasoning": False,
                        "is_current": False,
                    }
                )
        return result

    async def verify_and_fetch_models(
        self,
        provider_id: str,
        api_key: str,
        base_url: str | None = None,
    ) -> tuple[bool, list[str], str]:
        """
        Verify `api_key` against `provider_id`'s official API and discover supported models:
        1. Uses vendor-specific authentication (`Bearer`, `mimo_hybrid`, or `x-api-key` + `anthropic-version`).
        2. Queries `/models` (or `/v1/models` for Anthropic) first.
        3. If `/models` is unsupported (404/405) or returns empty, performs a lightweight 1-token probe against
           `/chat/completions` (or `/v1/messages` for Anthropic) to verify the key is genuinely valid.
        4. Stores the verified credentials and model list in `self.connected_providers[provider_id]`.
        """
        resolved_pid = resolve_provider_id(provider_id) or provider_id.strip().lower()
        if resolved_pid not in PROVIDER_SPECS:
            return False, [], f"不支持的大模型厂商: {provider_id}"
        clean_key = api_key.strip()
        if not clean_key:
            return False, [], "API Key 不能为空"

        spec = PROVIDER_SPECS[resolved_pid]
        curated = list(spec.get("curated_models") or [])
        headers = _build_provider_headers(resolved_pid, clean_key)

        if base_url:
            candidate_urls = [base_url.rstrip("/")]
        elif resolved_pid == "mimo":
            if clean_key.startswith(("tp-", "ttp-")):
                candidate_urls = [
                    "https://token-plan-cn.xiaomimimo.com/v1",
                    "https://api.xiaomimimo.com/v1",
                ]
            else:
                candidate_urls = [
                    "https://api.xiaomimimo.com/v1",
                    "https://token-plan-cn.xiaomimimo.com/v1",
                ]
        else:
            candidate_urls = list(spec.get("alt_base_urls") or [spec["base_url"]])

        last_error = "连接超时或 API Key 无效"

        for cand_base in candidate_urls:
            bypass_proxy = any(
                dom in cand_base.lower()
                for dom in (
                    "xiaomimimo.com",
                    "deepseek.com",
                    "bigmodel.cn",
                    "minimax",
                    "aliyuncs.com",
                    ".cn/",
                    ".cn:",
                    "localhost",
                    "127.0.0.1",
                )
            )
            try:
                async with httpx.AsyncClient(timeout=min(self.timeout_sec, 20.0), trust_env=not bypass_proxy) as client:
                    # Step 1: Query GET /models
                    models_resp = await client.get(f"{cand_base}/models", headers=headers)
                    if models_resp.status_code == 200:
                        body = models_resp.json()
                        if isinstance(body, dict) and body.get("base_resp", {}).get("status_code", 0) not in (0, None):
                            err_msg = body["base_resp"].get("status_msg") or "API Key 验证失败"
                            last_error = f"{spec['display_name']} 认证失败: {err_msg}"
                            continue
                        raw_items = body.get("data") or body.get("models") or [] if isinstance(body, dict) else []
                        fetched_ids: list[str] = []
                        if isinstance(raw_items, list):
                            for item in raw_items:
                                if isinstance(item, dict) and item.get("id"):
                                    fetched_ids.append(str(item["id"]))
                                elif isinstance(item, str):
                                    fetched_ids.append(item)
                        if fetched_ids:
                            merged_models = _filter_and_merge_provider_models(resolved_pid, fetched_ids)
                            self.connected_providers[resolved_pid] = {
                                "provider_id": resolved_pid,
                                "api_key": clean_key,
                                "base_url": cand_base,
                                "models": merged_models,
                            }
                            self._save_persisted_providers()
                            return (
                                True,
                                merged_models,
                                f"✓ 已成功连接 {spec['vendor_label']} (获取到 {len(merged_models)} 个可用模型)",
                            )
                    elif models_resp.status_code in (401, 403):
                        last_error = f"{spec['display_name']} API Key 认证失败 (HTTP {models_resp.status_code})"
                        continue

                    # Step 2: Lightweight probe to /chat/completions or /messages when /models is not exposed
                    probe_model = spec["default_fast_model"]
                    if spec["protocol"] == "anthropic_messages":
                        probe_url = f"{cand_base}/messages"
                        probe_payload: dict[str, Any] = {
                            "model": probe_model,
                            "max_tokens": 1,
                            "messages": [{"role": "user", "content": "hi"}],
                        }
                    else:
                        probe_url = f"{cand_base}/chat/completions"
                        if resolved_pid == "openai" and probe_model.lower().startswith(("gpt-5", "gpt-6", "o1", "o3", "o4")):
                            probe_payload = {
                                "model": probe_model,
                                "max_completion_tokens": 16,
                                "messages": [{"role": "user", "content": "hi"}],
                            }
                        else:
                            probe_payload = {
                                "model": probe_model,
                                "max_tokens": 1,
                                "messages": [{"role": "user", "content": "hi"}],
                            }
                        if resolved_pid == "minimax":
                            probe_payload["temperature"] = 0.01

                    probe_resp = await client.post(probe_url, headers=headers, json=probe_payload)
                    if probe_resp.status_code == 200:
                        p_body = probe_resp.json()
                        if isinstance(p_body, dict) and p_body.get("base_resp", {}).get("status_code", 0) not in (0, None):
                            err_msg = p_body["base_resp"].get("status_msg") or "API Key 验证失败"
                            last_error = f"{spec['display_name']} 认证失败: {err_msg}"
                            continue
                        merged_models = list(dict.fromkeys(curated))
                        self.connected_providers[resolved_pid] = {
                            "provider_id": resolved_pid,
                            "api_key": clean_key,
                            "base_url": cand_base,
                            "models": merged_models,
                        }
                        self._save_persisted_providers()
                        return (
                            True,
                            merged_models,
                            f"✓ 已成功连接 {spec['vendor_label']} (共 {len(merged_models)} 个可用模型)",
                        )
                    elif probe_resp.status_code in (401, 403):
                        last_error = f"{spec['display_name']} API Key 认证失败 (HTTP {probe_resp.status_code})"
                    else:
                        last_error = f"{spec['display_name']} 接口返回 HTTP {probe_resp.status_code}"
            except Exception as exc:
                last_error = f"连接 {spec['display_name']} 失败: {exc}"

        return False, [], last_error

    def activate_provider_model(
        self,
        provider_id: str,
        model_name: str,
        reasoning_model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Activate a connected provider and set the chosen model for immediate use."""
        resolved_pid = resolve_provider_id(provider_id) or provider_id.strip().lower()
        spec = PROVIDER_SPECS.get(resolved_pid, PROVIDER_SPECS["mimo"])
        conn = self.connected_providers.get(resolved_pid) or {}

        eff_key = api_key or conn.get("api_key") or self.api_key
        eff_base = (base_url or conn.get("base_url") or spec["base_url"]).rstrip("/")
        eff_fast = model_name.strip() or spec["default_fast_model"]
        eff_reasoning = (reasoning_model or eff_fast).strip()

        self.provider_id = resolved_pid
        self.api_key = eff_key
        self.base_url = eff_base
        self.fast_model = eff_fast
        self.reasoning_model = eff_reasoning

        existing_models = list(conn.get("models") or spec.get("curated_models") or [])
        if eff_fast not in existing_models:
            existing_models.insert(0, eff_fast)
        if eff_reasoning not in existing_models:
            existing_models.insert(0, eff_reasoning)

        self.connected_providers[resolved_pid] = {
            "provider_id": resolved_pid,
            "api_key": eff_key,
            "base_url": eff_base,
            "models": existing_models,
        }
        self._save_persisted_providers()

    def switch_model(self, fast_model: str, reasoning_model: str | None = None) -> tuple[str, str, str]:
        """
        Switch active model(s). Supports:
        - `<model_name>` on current provider
        - `<provider>:<model_name>` or `<provider>/<model_name>`
        - Automatically switches to another connected provider if `fast_model` belongs to it!
        Returns `(provider_display_name, fast_model, reasoning_model)`.
        """
        target_fast = fast_model.strip()
        target_reasoning = (reasoning_model or "").strip()

        explicit_pid: str | None = None
        for sep in (":", "/"):
            if sep in target_fast:
                prefix, rest = target_fast.split(sep, 1)
                cand_pid = resolve_provider_id(prefix)
                if cand_pid and rest.strip():
                    explicit_pid = cand_pid
                    target_fast = rest.strip()
                    break

        if explicit_pid is None:
            # Check if `target_fast` belongs to current provider first; if not, check other connected providers
            cur_models = self.get_provider_models(self.provider_id)
            if target_fast not in cur_models:
                for pid, conn in self.connected_providers.items():
                    if conn.get("api_key") and target_fast in self.get_provider_models(pid):
                        explicit_pid = pid
                        break

        target_pid = explicit_pid or self.provider_id or infer_provider_from_url_or_model(self.base_url, target_fast)
        eff_reas = target_reasoning or target_fast
        if explicit_pid and explicit_pid in self.connected_providers:
            self.activate_provider_model(explicit_pid, target_fast, eff_reas)
        else:
            self.fast_model = target_fast
            self.reasoning_model = eff_reas
            if target_pid in self.connected_providers:
                models_list = self.connected_providers[target_pid].setdefault("models", [])
                if target_fast not in models_list:
                    models_list.insert(0, target_fast)
            self._save_persisted_providers()

        spec = self.get_provider_spec(self.provider_id)
        return spec["display_name"], self.fast_model, self.reasoning_model

    @staticmethod
    def _prepare_openai_messages(
        messages: list[dict[str, Any]],
        provider_id: str,
        model: str,
        has_tools: bool = False,
    ) -> list[dict[str, Any]]:
        """Sanitize historical messages according to vendor-specific OpenAI-compatible protocol rules."""
        if provider_id == "mimo":
            return messages
        cleaned: list[dict[str, Any]] = []
        for m in messages:
            if m.get("role") == "assistant" and "reasoning_content" in m:
                # DeepSeek thinking mode requires reasoning_content when tools/tool_calls are active,
                # and ignores/strips it on non-tool turns without tools;
                # GLM, Qwen, MiniMax, and OpenAI strict validators reject unexpected reasoning_content on input.
                if provider_id == "deepseek" and (m.get("tool_calls") or has_tools):
                    cleaned.append(m)
                else:
                    m_copy = {k: v for k, v in m.items() if k != "reasoning_content"}
                    cleaned.append(m_copy)
            else:
                cleaned.append(m)
        return cleaned

    async def _single_anthropic_completion_call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str,
        temperature: float,
        max_tokens: int | None,
    ) -> LLMResponse:
        """Execute a native Anthropic `/v1/messages` completion call with full tool_use / tool_result conversion."""
        payload = _convert_openai_to_anthropic_payload(
            messages=messages,
            tools=tools,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        headers = _build_provider_headers("anthropic", self.api_key)
        bypass_proxy = any(dom in self.base_url.lower() for dom in (".cn/", ".cn:", "localhost", "127.0.0.1"))
        async with httpx.AsyncClient(timeout=self.timeout_sec, trust_env=not bypass_proxy) as client:
            resp = await client.post(
                f"{self.base_url}/messages",
                headers=headers,
                json=payload,
            )
            if resp.status_code == 400 and "temperature" in payload:
                payload.pop("temperature", None)
                resp = await client.post(
                    f"{self.base_url}/messages",
                    headers=headers,
                    json=payload,
                )
            if resp.status_code == 400 and payload.get("max_tokens", 0) > 4096:
                payload["max_tokens"] = 4096
                resp = await client.post(
                    f"{self.base_url}/messages",
                    headers=headers,
                    json=payload,
                )
            resp.raise_for_status()
            data = resp.json()

        return _parse_anthropic_response(data, model=model, messages=messages)

    async def _single_openai_responses_call(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str,
        max_tokens: int | None,
    ) -> LLMResponse:
        """Execute an OpenAI `/v1/responses` call (used by GPT-6 Astra / 6.1 Sol for tool calling)."""
        system_parts: list[str] = []
        input_items: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role", "user")
            content = str(m.get("content") or "")
            if role == "system":
                if content:
                    system_parts.append(content)
            elif role == "tool":
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": str(m.get("tool_call_id") or "call_0"),
                        "output": content,
                    }
                )
            elif role == "assistant":
                if content:
                    input_items.append({"role": "assistant", "content": content})
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    raw_args = fn.get("arguments", "{}")
                    args_str = raw_args if isinstance(raw_args, str) else json.dumps(raw_args, ensure_ascii=False)
                    input_items.append(
                        {
                            "type": "function_call",
                            "call_id": str(tc.get("id") or "call_0"),
                            "name": str(fn.get("name") or ""),
                            "arguments": args_str,
                        }
                    )
            else:
                input_items.append({"role": "user", "content": content})

        resp_tools: list[dict[str, Any]] = []
        for t in tools or []:
            fn = t.get("function") if "function" in t else t
            if isinstance(fn, dict) and fn.get("name"):
                resp_tools.append(
                    {
                        "type": "function",
                        "name": str(fn["name"]),
                        "description": str(fn.get("description") or ""),
                        "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
                    }
                )

        r_payload: dict[str, Any] = {
            "model": model,
            "input": input_items or [{"role": "user", "content": "Hello"}],
        }
        if system_parts:
            r_payload["instructions"] = "\n\n".join(system_parts)
        if resp_tools:
            r_payload["tools"] = resp_tools
        if max_tokens is not None:
            r_payload["max_output_tokens"] = max_tokens

        resp = await client.post(
            f"{self.base_url}/responses",
            headers=headers,
            json=r_payload,
        )
        resp.raise_for_status()
        data = resp.json()

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        parsed_calls: list[ToolCallRequest] = []
        for item in data.get("output") or []:
            if not isinstance(item, dict):
                continue
            itype = item.get("type")
            if itype == "message":
                for c in item.get("content") or []:
                    if isinstance(c, dict) and c.get("text"):
                        content_parts.append(str(c["text"]))
            elif itype == "reasoning":
                for s in item.get("summary") or []:
                    if isinstance(s, dict) and s.get("text"):
                        reasoning_parts.append(str(s["text"]))
            elif itype == "function_call":
                raw_args = item.get("arguments", "{}")
                try:
                    args_dict = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except Exception:
                    args_dict = {}
                parsed_calls.append(
                    ToolCallRequest(
                        call_id=str(item.get("call_id") or item.get("id") or f"call_{len(parsed_calls)}"),
                        name=str(item.get("name") or ""),
                        arguments=args_dict if isinstance(args_dict, dict) else {},
                    )
                )
        if not content_parts and isinstance(data.get("output_text"), str) and data["output_text"]:
            content_parts.append(data["output_text"])

        content_str = "\n".join(content_parts)
        reasoning_str = "\n".join(reasoning_parts)
        usage = data.get("usage") or {}
        p_tok = int(usage.get("input_tokens") or 0) or _estimate_tokens_from_messages(messages)
        c_tok = int(usage.get("output_tokens") or 0) or (
            _estimate_tokens_from_text(content_str + reasoning_str) + len(parsed_calls) * 18
        )
        out_details = usage.get("output_tokens_details") or {}
        r_tok = int(out_details.get("reasoning_tokens") or 0)
        finish_reason = "tool_calls" if parsed_calls else str(data.get("status") or "stop")
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

    async def _single_completion_call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        use_reasoning: bool = False,
        temperature: float = 0.1,
        max_tokens: int | None = 32768,
    ) -> LLMResponse:
        model = self.reasoning_model if use_reasoning else self.fast_model
        active_pid = self.provider_id or infer_provider_from_url_or_model(self.base_url, model)
        spec = PROVIDER_SPECS.get(active_pid, PROVIDER_SPECS["mimo"])

        # Route Anthropic native `/v1/messages` protocol
        if spec.get("protocol") == "anthropic_messages" or "anthropic.com" in self.base_url.lower():
            return await self._single_anthropic_completion_call(
                messages=messages,
                tools=tools,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        sanitized_messages = self._prepare_openai_messages(
            messages, active_pid, model, has_tools=bool(tools)
        )
        payload: dict[str, Any] = {
            "model": model,
            "messages": sanitized_messages,
        }

        m_low = model.lower()
        is_openai_o_series = active_pid == "openai" and m_low.startswith(("o1", "o3", "o4", "gpt-5", "gpt-6"))
        is_deepseek_reasoner = active_pid == "deepseek" and (
            use_reasoning or "reasoner" in m_low or "v4-pro" in m_low
        )

        if active_pid == "minimax":
            # MiniMax requires temperature in (0.0, 1.0]; temperature=0.0 is rejected with HTTP 400
            payload["temperature"] = max(0.01, min(1.0, float(temperature)))
        elif not is_openai_o_series and not is_deepseek_reasoner:
            payload["temperature"] = temperature

        if max_tokens is not None:
            eff_max = max_tokens
            if is_openai_o_series:
                payload["max_completion_tokens"] = eff_max
            else:
                payload["max_tokens"] = eff_max

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = _build_provider_headers(active_pid, self.api_key)

        bypass_proxy = any(
            dom in self.base_url.lower()
            for dom in (
                "xiaomimimo.com",
                "deepseek.com",
                "bigmodel.cn",
                "minimax",
                "aliyuncs.com",
                ".cn/",
                ".cn:",
                "localhost",
                "127.0.0.1",
            )
        )
        async with httpx.AsyncClient(timeout=self.timeout_sec, trust_env=not bypass_proxy) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            # If a provider rejects a high max_tokens (e.g. >16384 or >8192), fall back gracefully
            tok_key = "max_completion_tokens" if is_openai_o_series else "max_tokens"
            cur_max = payload.get(tok_key)
            if resp.status_code == 400 and isinstance(cur_max, int) and cur_max > 4096:
                for fallback_toks in (16384, 8192, 4096):
                    if fallback_toks >= cur_max:
                        continue
                    payload[tok_key] = fallback_toks
                    resp = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                    if resp.status_code < 400:
                        break
            # OpenAI GPT-6 Astra / 6.1 Sol require /v1/responses when called with tools
            if resp.status_code == 400 and active_pid == "openai" and tools:
                return await self._single_openai_responses_call(
                    client=client,
                    headers=headers,
                    messages=sanitized_messages,
                    tools=tools,
                    model=model,
                    max_tokens=max_tokens,
                )
            resp.raise_for_status()
            data = resp.json()

        if active_pid == "minimax" and isinstance(data, dict):
            base_resp = data.get("base_resp") or {}
            if base_resp.get("status_code", 0) not in (0, None):
                raise RuntimeError(
                    f"MiniMax API error ({base_resp.get('status_code')}): {base_resp.get('status_msg')}"
                )

        choice_entry = data["choices"][0]
        finish_reason = str(choice_entry.get("finish_reason") or "")
        choice = choice_entry["message"]
        content = choice.get("content") or ""
        reasoning_content = choice.get("reasoning_content") or ""
        raw_tool_calls = choice.get("tool_calls") or []
        parsed_calls: list[ToolCallRequest] = []

        for tc in raw_tool_calls:
            fn = tc.get("function", {})
            raw_args = fn.get("arguments", "{}")
            try:
                args_dict = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                args_dict = {}
            parsed_calls.append(
                ToolCallRequest(
                    call_id=tc.get("id") or f"call_{len(parsed_calls)}",
                    name=fn.get("name", ""),
                    arguments=args_dict if isinstance(args_dict, dict) else {},
                )
            )

        usage = data.get("usage") or {}
        comp_details = usage.get("completion_tokens_details") or {}
        r_tok = int(comp_details.get("reasoning_tokens") or 0)
        p_tok = int(usage.get("prompt_tokens") or 0) or _estimate_tokens_from_messages(messages)
        c_tok = int(usage.get("completion_tokens") or 0) or (
            _estimate_tokens_from_text(content + reasoning_content) + len(parsed_calls) * 18
        )
        t_tok = int(usage.get("total_tokens") or 0) or (p_tok + c_tok)

        return LLMResponse(
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=parsed_calls,
            model=model,
            finish_reason=finish_reason,
            reasoning_tokens=r_tok,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            total_tokens=t_tok,
        )

    async def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        use_reasoning: bool = False,
        temperature: float = 0.1,
        max_tokens: int | None = 32768,
        auto_continue: bool = True,
        max_continuations: int = 3,
    ) -> LLMResponse:
        """
        Execute a chat completion request with optional OpenAI-format tool definitions.
        - Uses a generous default `max_tokens=32768` so reasoning models (like `mimo-v2.6-flash` / `mimo-v2.6-pro`,
          which share `max_tokens` between `reasoning_tokens` and `content`) have ample budget for both deep
          chain-of-thought and long-form architectural reports.
        - When `finish_reason == "length"` and no tool calls were emitted, automatically performs seamless
          multi-turn continuation (`auto_continue=True`) from the exact cutoff point so long reports NEVER
          truncate mid-sentence or mid-word!
        """
        if not self.is_configured:
            raise RuntimeError(
                "LLM API key not configured. Use `/connect` to connect DeepSeek, GLM, MiMo, MiniMax, Qwen, OpenAI, or Anthropic."
            )

        base_resp = await self._single_completion_call(
            messages=messages,
            tools=tools,
            use_reasoning=use_reasoning,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Seamless continuation when output hit token limit (`finish_reason == "length"`) during a non-tool response
        if auto_continue and base_resp.finish_reason == "length" and not base_resp.tool_calls:
            acc_content = base_resp.content
            acc_reasoning = base_resp.reasoning_content
            total_p_tok = base_resp.prompt_tokens
            total_c_tok = base_resp.completion_tokens
            total_t_tok = base_resp.total_tokens
            total_r_tok = base_resp.reasoning_tokens
            cur_finish = base_resp.finish_reason
            rounds = 0

            for _ in range(max_continuations):
                if cur_finish != "length":
                    break
                rounds += 1
                cont_msgs = list(messages)
                if acc_content.strip():
                    tail_anchor = acc_content[-60:]
                    cont_msgs.append({"role": "assistant", "content": acc_content})
                    cont_msgs.append(
                        {
                            "role": "user",
                            "content": (
                                "[CONTINUE FROM EXACT CUTOFF]: Your previous response hit the output token limit "
                                f"and stopped right at `{tail_anchor}`. "
                                "Continue writing immediately from the exact next character without repeating any "
                                "previous text, headings, or preamble."
                            ),
                        }
                    )
                else:
                    # Reasoning tokens consumed the entire window before emitting visible `content`
                    cont_msgs.append(
                        {
                            "role": "user",
                            "content": (
                                "[OUTPUT FINAL RESPONSE NOW]: Internal reasoning is complete. "
                                "Do NOT call any tools; immediately output the complete, thorough Markdown response."
                            ),
                        }
                    )

                cont_resp = await self._single_completion_call(
                    messages=cont_msgs,
                    tools=None,
                    use_reasoning=False,  # Use fast/direct mode for seamless text continuation so tokens go to content!
                    temperature=temperature,
                    max_tokens=max_tokens or 32768,
                )
                total_p_tok += cont_resp.prompt_tokens
                total_c_tok += cont_resp.completion_tokens
                total_t_tok += cont_resp.total_tokens
                total_r_tok += cont_resp.reasoning_tokens
                cur_finish = cont_resp.finish_reason
                if cont_resp.reasoning_content:
                    acc_reasoning = (acc_reasoning + "\n" + cont_resp.reasoning_content).strip()
                if not cont_resp.content:
                    break
                acc_content += cont_resp.content

            base_resp = base_resp.model_copy(
                update={
                    "content": acc_content,
                    "reasoning_content": acc_reasoning,
                    "finish_reason": cur_finish,
                    "reasoning_tokens": total_r_tok,
                    "continuation_rounds": rounds,
                    "prompt_tokens": total_p_tok,
                    "completion_tokens": total_c_tok,
                    "total_tokens": total_t_tok,
                }
            )

        self.last_usage = (base_resp.prompt_tokens, base_resp.completion_tokens, base_resp.total_tokens)
        return base_resp

    async def complete_structured(
        self,
        messages: list[dict[str, Any]],
        response_model: type[T],
        use_reasoning: bool = False,
        temperature: float = 0.0,
        max_tokens: int | None = 2048,
    ) -> T:
        """Request a strictly validated Pydantic model output from the LLM."""
        schema_json = json.dumps(response_model.model_json_schema(), ensure_ascii=False, indent=2)
        schema_prompt = (
            "Respond ONLY with a single compact valid JSON object conforming strictly to the following "
            "JSON Schema (keep string fields concise and direct; no markdown or prose outside JSON):\n"
            f"```json\n{schema_json}\n```"
        )
        augmented_messages = list(messages)
        augmented_messages.append({"role": "system", "content": schema_prompt})

        resp = await self.complete_with_tools(
            messages=augmented_messages,
            tools=None,
            use_reasoning=use_reasoning,
            temperature=temperature,
            max_tokens=max_tokens,
            auto_continue=False,
        )
        json_str = _extract_json_payload(resp.content)
        return response_model.model_validate_json(json_str)


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
