from agent.smart_model_routing import (
    choose_cheap_model_route,
    resolve_smart_reasoning_config,
)


_BASE_CONFIG = {
    "enabled": True,
    "cheap_model": {
        "provider": "openrouter",
        "model": "google/gemini-2.5-flash",
    },
}


def test_returns_none_when_disabled():
    cfg = {**_BASE_CONFIG, "enabled": False}
    assert choose_cheap_model_route("what time is it in tokyo?", cfg) is None


def test_routes_short_simple_prompt():
    result = choose_cheap_model_route("what time is it in tokyo?", _BASE_CONFIG)
    assert result is not None
    assert result["provider"] == "openrouter"
    assert result["model"] == "google/gemini-2.5-flash"
    assert result["routing_reason"] == "simple_turn"


def test_skips_long_prompt():
    prompt = "please summarize this carefully " * 20
    assert choose_cheap_model_route(prompt, _BASE_CONFIG) is None


def test_skips_code_like_prompt():
    prompt = "debug this traceback: ```python\nraise ValueError('bad')\n```"
    assert choose_cheap_model_route(prompt, _BASE_CONFIG) is None


def test_skips_tool_heavy_prompt_keywords():
    prompt = "implement a patch for this docker error"
    assert choose_cheap_model_route(prompt, _BASE_CONFIG) is None


def test_skips_short_ack_with_prior_context():
    assert choose_cheap_model_route("yes", _BASE_CONFIG, has_prior_context=True) is None


def test_dynamic_reasoning_keeps_base_effort_for_simple_turn():
    result = resolve_smart_reasoning_config(
        [{"role": "user", "content": "please draft a short reply"}],
        {"enabled": True, "effort": "high"},
        {
            "enabled": True,
            "default_effort": "high",
            "complex_effort": "xhigh",
        },
    )

    assert result == {"enabled": True, "effort": "high"}


def test_dynamic_reasoning_escalates_complex_turn_to_xhigh():
    result = resolve_smart_reasoning_config(
        [{
            "role": "user",
            "content": "why is this HubSpot workflow failing to enrol contacts and what tradeoffs should we consider when redesigning it?",
        }],
        {"enabled": True, "effort": "high"},
        {
            "enabled": True,
            "default_effort": "high",
            "complex_effort": "xhigh",
        },
    )

    assert result == {"enabled": True, "effort": "xhigh"}


def test_dynamic_reasoning_escalates_after_tool_results_mid_turn():
    result = resolve_smart_reasoning_config(
        [
            {"role": "user", "content": "check this workflow issue"},
            {"role": "assistant", "content": "I'll inspect it."},
            {"role": "tool", "content": "Workflow has multiple branch conditions and conflicting enrollment criteria."},
        ],
        {"enabled": True, "effort": "high"},
        {
            "enabled": True,
            "default_effort": "high",
            "complex_effort": "xhigh",
            "escalate_on_tool_results": True,
        },
    )

    assert result == {"enabled": True, "effort": "xhigh"}


def test_dynamic_reasoning_keeps_explicit_base_xhigh():
    result = resolve_smart_reasoning_config(
        [{"role": "user", "content": "say hi"}],
        {"enabled": True, "effort": "xhigh"},
        {
            "enabled": True,
            "default_effort": "high",
            "complex_effort": "xhigh",
        },
    )

    assert result == {"enabled": True, "effort": "xhigh"}


def test_resolve_turn_route_preserves_primary_reason_when_not_routed():
    from agent.smart_model_routing import resolve_turn_route

    result = resolve_turn_route(
        "debug this traceback: ```python\nraise ValueError('bad')\n```",
        _BASE_CONFIG,
        {
            "model": "anthropic/claude-sonnet-4",
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
            "api_key": "***",
        },
    )

    assert result["routing_reason"] == "primary_model"


def test_resolve_turn_route_preserves_smart_route_reason(monkeypatch):
    from agent.smart_model_routing import resolve_turn_route

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
            "api_key": "***",
        },
    )
    result = resolve_turn_route(
        "what time is it in tokyo?",
        _BASE_CONFIG,
        {
            "model": "anthropic/claude-sonnet-4",
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
            "api_key": "***",
        },
    )

    assert result["model"] == "google/gemini-2.5-flash"
    assert result["routing_reason"] == "simple_turn"


def test_resolve_turn_route_falls_back_to_primary_when_route_runtime_cannot_be_resolved(monkeypatch):
    from agent.smart_model_routing import resolve_turn_route

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bad route")),
    )
    result = resolve_turn_route(
        "what time is it in tokyo?",
        _BASE_CONFIG,
        {
            "model": "anthropic/claude-sonnet-4",
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
            "api_key": "sk-primary",
        },
    )
    assert result["model"] == "anthropic/claude-sonnet-4"
    assert result["runtime"]["provider"] == "openrouter"
    assert result["label"] is None
    assert result["routing_reason"] == "route_runtime_error"
