"""Helpers for optional turn-level routing.

Includes:
- cheap-vs-strong model routing for obviously simple turns
- dynamic reasoning-effort routing for deeper turns/tool-heavy turns
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

from utils import is_truthy_value

_COMPLEX_KEYWORDS = {
    "debug",
    "debugging",
    "implement",
    "implementation",
    "refactor",
    "patch",
    "traceback",
    "stacktrace",
    "exception",
    "error",
    "analyze",
    "analysis",
    "investigate",
    "architecture",
    "design",
    "compare",
    "benchmark",
    "optimize",
    "optimise",
    "review",
    "terminal",
    "shell",
    "tool",
    "tools",
    "pytest",
    "test",
    "tests",
    "plan",
    "planning",
    "delegate",
    "subagent",
    "cron",
    "docker",
    "kubernetes",
}

_REASONING_COMPLEX_KEYWORDS = _COMPLEX_KEYWORDS | {
    "strategy",
    "strategic",
    "workflow",
    "workflows",
    "hubspot",
    "diagnose",
    "diagnosis",
    "root",
    "cause",
    "troubleshoot",
    "troubleshooting",
    "issue",
    "issues",
    "tradeoff",
    "tradeoffs",
    "roadmap",
    "proposal",
    "proposals",
    "synthesis",
    "redesign",
}

_ACKNOWLEDGEMENT_PHRASES = {
    "yes",
    "yep",
    "yeah",
    "ok",
    "okay",
    "sure",
    "do it",
    "go ahead",
    "please do",
    "sounds good",
    "correct",
    "continue",
    "carry on",
}

_VALID_REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh")
_REASONING_ORDER = {name: idx for idx, name in enumerate(_VALID_REASONING_EFFORTS)}
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)


def _coerce_bool(value: Any, default: bool = False) -> bool:
    return is_truthy_value(value, default=default)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalized_words(text: str) -> set[str]:
    lowered = (text or "").lower()
    return {
        token.strip(".,:;!?()[]{}\"'`")
        for token in lowered.split()
        if token.strip(".,:;!?()[]{}\"'`")
    }


def _looks_like_acknowledgement(text: str) -> bool:
    normalized = " ".join((text or "").strip().lower().split())
    if not normalized:
        return False
    if normalized in _ACKNOWLEDGEMENT_PHRASES:
        return True
    words = normalized.split()
    if len(words) <= 3 and all(word in {"yes", "yep", "yeah", "ok", "okay", "sure", "do", "it", "go", "ahead", "please", "continue"} for word in words):
        return True
    return False


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
        return "\n".join(part for part in parts if part)
    if content is None:
        return ""
    return str(content)


def _latest_user_text(api_messages: list[dict] | None) -> str:
    for message in reversed(api_messages or []):
        if isinstance(message, dict) and message.get("role") == "user":
            return _content_to_text(message.get("content"))
    return ""


def _has_tool_results_since_last_user(api_messages: list[dict] | None) -> bool:
    last_user_index = None
    for idx in range(len(api_messages or []) - 1, -1, -1):
        message = api_messages[idx]
        if isinstance(message, dict) and message.get("role") == "user":
            last_user_index = idx
            break
    if last_user_index is None:
        return False
    for message in (api_messages or [])[last_user_index + 1 :]:
        if isinstance(message, dict) and message.get("role") == "tool":
            return True
    return False


def _normalize_reasoning_effort(effort: Any, default: str = "medium") -> str:
    normalized = str(effort or "").strip().lower()
    if normalized in _VALID_REASONING_EFFORTS:
        return normalized
    return default


def _effort_rank(effort: str) -> int:
    return _REASONING_ORDER.get(_normalize_reasoning_effort(effort), _REASONING_ORDER["medium"])


def _looks_reasoning_complex(text: str, routing_config: Optional[Dict[str, Any]]) -> bool:
    cfg = routing_config or {}
    text = (text or "").strip()
    if not text:
        return False

    words = _normalized_words(text)
    keyword_set = set(_REASONING_COMPLEX_KEYWORDS)
    custom_keywords = cfg.get("complex_keywords") or []
    if isinstance(custom_keywords, (list, tuple, set)):
        keyword_set.update(str(item).strip().lower() for item in custom_keywords if str(item).strip())

    if words & keyword_set:
        return True

    min_chars = _coerce_int(cfg.get("min_complex_chars"), 220)
    min_words = _coerce_int(cfg.get("min_complex_words"), 35)
    if len(text) >= min_chars or len(text.split()) >= min_words:
        return True
    if text.count("\n") > 1:
        return True
    if text.count("?") >= 2:
        return True
    if "```" in text or "`" in text:
        return True
    if _URL_RE.search(text):
        return True
    return False


def resolve_smart_reasoning_config(
    api_messages: list[dict] | None,
    base_reasoning_config: Optional[Dict[str, Any]],
    routing_config: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return the effective reasoning config for the current API call.

    The base reasoning config is the floor/default. When smart reasoning routing
    is enabled, obviously complex turns — or turns that have become tool-heavy —
    can escalate to a higher effort (for example high -> xhigh).
    """

    base = dict(base_reasoning_config) if isinstance(base_reasoning_config, dict) else base_reasoning_config
    cfg = routing_config or {}
    if not _coerce_bool(cfg.get("enabled"), False):
        return base

    if isinstance(base, dict) and base.get("enabled") is False:
        return base

    default_effort = _normalize_reasoning_effort(
        (base.get("effort") if isinstance(base, dict) else "")
        or cfg.get("default_effort")
        or "medium",
        default="medium",
    )
    complex_effort = _normalize_reasoning_effort(
        cfg.get("complex_effort") or cfg.get("max_effort") or default_effort,
        default=default_effort,
    )
    if _effort_rank(complex_effort) < _effort_rank(default_effort):
        complex_effort = default_effort

    effective = {"enabled": True, "effort": default_effort}
    latest_user_text = _latest_user_text(api_messages)
    if _looks_reasoning_complex(latest_user_text, cfg):
        effective["effort"] = complex_effort
        return effective

    if _coerce_bool(cfg.get("escalate_on_tool_results"), True) and _has_tool_results_since_last_user(api_messages):
        effective["effort"] = complex_effort
        return effective

    return effective


def choose_cheap_model_route(
    user_message: str,
    routing_config: Optional[Dict[str, Any]],
    *,
    has_prior_context: bool = False,
) -> Optional[Dict[str, Any]]:
    """Return the configured cheap-model route when a message looks simple.

    Conservative by design: if the message has signs of code/tool/debugging/
    long-form work, keep the primary model. Also avoids routing short replies
    like "yes"/"go ahead" to a cheap model when they depend on prior context.
    """
    cfg = routing_config or {}
    if not _coerce_bool(cfg.get("enabled"), False):
        return None

    cheap_model = cfg.get("cheap_model") or {}
    if not isinstance(cheap_model, dict):
        return None
    provider = str(cheap_model.get("provider") or "").strip().lower()
    model = str(cheap_model.get("model") or "").strip()
    if not provider or not model:
        return None

    text = (user_message or "").strip()
    if not text:
        return None
    if has_prior_context and _looks_like_acknowledgement(text):
        return None

    max_chars = _coerce_int(cfg.get("max_simple_chars"), 160)
    max_words = _coerce_int(cfg.get("max_simple_words"), 28)

    if len(text) > max_chars:
        return None
    if len(text.split()) > max_words:
        return None
    if text.count("\n") > 1:
        return None
    if "```" in text or "`" in text:
        return None
    if _URL_RE.search(text):
        return None

    words = _normalized_words(text)
    if words & _COMPLEX_KEYWORDS:
        return None

    route = dict(cheap_model)
    route["provider"] = provider
    route["model"] = model
    route["routing_reason"] = "simple_turn"
    return route


def resolve_turn_route(
    user_message: str,
    routing_config: Optional[Dict[str, Any]],
    primary: Dict[str, Any],
    *,
    has_prior_context: bool = False,
) -> Dict[str, Any]:
    """Resolve the effective model/runtime for one turn.

    Returns a dict with model/runtime/signature/label fields.
    """
    route = choose_cheap_model_route(
        user_message,
        routing_config,
        has_prior_context=has_prior_context,
    )
    if not route:
        return {
            "model": primary.get("model"),
            "runtime": {
                "api_key": primary.get("api_key"),
                "base_url": primary.get("base_url"),
                "provider": primary.get("provider"),
                "api_mode": primary.get("api_mode"),
                "command": primary.get("command"),
                "args": list(primary.get("args") or []),
                "credential_pool": primary.get("credential_pool"),
            },
            "label": None,
            "signature": (
                primary.get("model"),
                primary.get("provider"),
                primary.get("base_url"),
                primary.get("api_mode"),
                primary.get("command"),
                tuple(primary.get("args") or ()),
            ),
        }

    from hermes_cli.runtime_provider import resolve_runtime_provider

    explicit_api_key = None
    api_key_env = str(route.get("api_key_env") or "").strip()
    if api_key_env:
        explicit_api_key = os.getenv(api_key_env) or None

    try:
        runtime = resolve_runtime_provider(
            requested=route.get("provider"),
            explicit_api_key=explicit_api_key,
            explicit_base_url=route.get("base_url"),
        )
    except Exception:
        return {
            "model": primary.get("model"),
            "runtime": {
                "api_key": primary.get("api_key"),
                "base_url": primary.get("base_url"),
                "provider": primary.get("provider"),
                "api_mode": primary.get("api_mode"),
                "command": primary.get("command"),
                "args": list(primary.get("args") or []),
                "credential_pool": primary.get("credential_pool"),
            },
            "label": None,
            "signature": (
                primary.get("model"),
                primary.get("provider"),
                primary.get("base_url"),
                primary.get("api_mode"),
                primary.get("command"),
                tuple(primary.get("args") or ()),
            ),
        }

    return {
        "model": route.get("model"),
        "runtime": {
            "api_key": runtime.get("api_key"),
            "base_url": runtime.get("base_url"),
            "provider": runtime.get("provider"),
            "api_mode": runtime.get("api_mode"),
            "command": runtime.get("command"),
            "args": list(runtime.get("args") or []),
            "credential_pool": runtime.get("credential_pool"),
        },
        "label": f"smart route → {route.get('model')} ({runtime.get('provider')})",
        "signature": (
            route.get("model"),
            runtime.get("provider"),
            runtime.get("base_url"),
            runtime.get("api_mode"),
            runtime.get("command"),
            tuple(runtime.get("args") or ()),
        ),
    }
