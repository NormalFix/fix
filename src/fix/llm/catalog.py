"""Multi-vendor LLM registry, model lifecycle filtering, and zero-config shell RC API key discovery."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

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

