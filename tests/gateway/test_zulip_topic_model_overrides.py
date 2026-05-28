"""Regression tests for Zulip topic-based model overrides."""

from __future__ import annotations

from gateway.config import Platform
from gateway.session import SessionSource, build_session_key


def _zulip_source(topic: str) -> SessionSource:
    return SessionSource(
        platform=Platform.ZULIP,
        chat_id="stream:595301",
        chat_name="arlo-strategy-marketing",
        chat_type="channel",
        user_id="damien@example.com",
        user_name="Damien",
        thread_id=topic,
        chat_topic=topic,
        parent_chat_id="595301",
    )


def _config() -> dict:
    return {
        "model": {"provider": "openai-codex", "default": "gpt-5.5"},
        "zulip": {
            "topic_model_overrides": [
                {
                    "ids": ["595299", "595300", "595301"],
                    "topic_prefix": "-",
                    "provider": "deepseek",
                    "model": "deepseek-v4-pro",
                    "base_url": "",
                    "api_key": "",
                }
            ]
        },
    }


def _runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {}
    runner.session_store = None
    return runner


def test_normal_zulip_parent_topic_keeps_codex_runtime(monkeypatch) -> None:
    from gateway import run as gateway_run

    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openai-codex",
            "api_key": "codex-key",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_mode": "codex_responses",
            "command": None,
            "args": [],
            "credential_pool": None,
        },
    )
    monkeypatch.setattr(
        gateway_run,
        "_resolve_provider_runtime_agent_kwargs",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("topic provider override should not resolve for normal topics")
        ),
    )

    model, runtime = _runner()._resolve_session_agent_runtime(
        source=_zulip_source("regular-topic"),
        user_config=_config(),
    )

    assert model == "gpt-5.5"
    assert runtime["provider"] == "openai-codex"
    assert runtime["api_mode"] == "codex_responses"


def test_dash_prefixed_zulip_topic_uses_deepseek_runtime(monkeypatch) -> None:
    from gateway import run as gateway_run

    calls = []

    def _resolve_provider(provider: str, **kwargs):
        calls.append((provider, kwargs))
        return None, {
            "provider": provider,
            "api_key": "deepseek-key",
            "base_url": "https://api.deepseek.com/v1",
            "api_mode": "chat_completions",
            "command": None,
            "args": [],
            "credential_pool": None,
        }

    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: (_ for _ in ()).throw(
            AssertionError("global Codex runtime should be skipped for provider topic overrides")
        ),
    )
    monkeypatch.setattr(gateway_run, "_resolve_provider_runtime_agent_kwargs", _resolve_provider)

    model, runtime = _runner()._resolve_session_agent_runtime(
        source=_zulip_source("- child-demo"),
        user_config=_config(),
    )

    assert model == "deepseek-v4-pro"
    assert runtime["provider"] == "deepseek"
    assert runtime["base_url"] == "https://api.deepseek.com/v1"
    assert calls == [
        (
            "deepseek",
            {
                "model": "deepseek-v4-pro",
                "explicit_base_url": None,
                "explicit_api_key": None,
                "api_mode": None,
            },
        )
    ]


def test_session_model_override_wins_over_dash_topic_model_override(monkeypatch) -> None:
    from gateway import run as gateway_run

    source = _zulip_source("- child-demo")
    session_key = build_session_key(source)
    runner = _runner()
    runner._session_model_overrides = {
        session_key: {
            "provider": "openai-codex",
            "model": "gpt-5.5",
            "api_key": "codex-key",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_mode": "codex_responses",
        }
    }
    monkeypatch.setattr(
        gateway_run,
        "_resolve_provider_runtime_agent_kwargs",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("topic provider override must not beat explicit /model override")
        ),
    )

    model, runtime = runner._resolve_session_agent_runtime(
        source=source,
        session_key=session_key,
        user_config=_config(),
    )

    assert model == "gpt-5.5"
    assert runtime["provider"] == "openai-codex"
    assert runtime["api_key"] == "codex-key"


def test_persisted_session_model_override_hydrates_and_beats_topic_override(monkeypatch) -> None:
    from gateway import run as gateway_run

    source = _zulip_source("- child-demo")
    session_key = build_session_key(source)

    class Store:
        seen_key = None

        def get_model_override(self, key: str):
            self.seen_key = key
            return {
                "provider": "openai-codex",
                "model": "gpt-5.5",
                "base_url": "https://chatgpt.com/backend-api/codex",
                "api_mode": "codex_responses",
            }

    store = Store()
    runner = _runner()
    runner.session_store = store

    calls = []

    def _resolve_provider(provider: str, **kwargs):
        calls.append((provider, kwargs))
        return None, {
            "provider": provider,
            "api_key": "resolved-codex-key",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_mode": "codex_responses",
            "command": None,
            "args": [],
            "credential_pool": None,
        }

    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: (_ for _ in ()).throw(
            AssertionError("default runtime must not be used for persisted session /model")
        ),
    )
    monkeypatch.setattr(gateway_run, "_resolve_provider_runtime_agent_kwargs", _resolve_provider)

    model, runtime = runner._resolve_session_agent_runtime(
        source=source,
        session_key=session_key,
        user_config=_config(),
    )

    assert store.seen_key == session_key
    assert model == "gpt-5.5"
    assert runtime["provider"] == "openai-codex"
    assert runtime["api_key"] == "resolved-codex-key"
    assert calls == [
        (
            "openai-codex",
            {
                "model": "gpt-5.5",
                "explicit_base_url": "https://chatgpt.com/backend-api/codex",
                "explicit_api_key": None,
                "api_mode": "codex_responses",
            },
        )
    ]


def test_model_command_persists_session_override(monkeypatch, tmp_path) -> None:
    import asyncio
    from types import SimpleNamespace

    from gateway import run as gateway_run
    from gateway.platforms.base import MessageEvent

    source = _zulip_source("regular-topic")
    session_key = build_session_key(source)
    runner = _runner()
    runner.adapters = {}
    runner._evict_cached_agent = lambda key: None

    class Store:
        persisted = []

        def get_model_override(self, key: str):
            return None

        def set_model_override(self, key: str, override):
            self.persisted.append((key, dict(override)))
            return True

    store = Store()
    runner.session_store = store

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"model": {"provider": "openai-codex", "default": "gpt-5.5"}},
    )

    import hermes_cli.model_switch as model_switch

    monkeypatch.setattr(
        model_switch,
        "switch_model",
        lambda **kwargs: SimpleNamespace(
            success=True,
            error_message=None,
            new_model="deepseek-v4-pro",
            target_provider="deepseek",
            api_key="deepseek-key",
            base_url="https://api.deepseek.com/v1",
            api_mode="chat_completions",
            provider_label="DeepSeek",
            warning_message=None,
            model_info=None,
        ),
    )
    monkeypatch.setattr(model_switch, "resolve_display_context_length", lambda *a, **k: None)

    response = asyncio.run(
        runner._handle_model_command(
            MessageEvent(text="/model deepseek-v4-pro --provider deepseek", source=source)
        )
    )

    assert "deepseek-v4-pro" in response
    assert runner._session_model_overrides[session_key]["model"] == "deepseek-v4-pro"
    assert store.persisted == [
        (
            session_key,
            {
                "model": "deepseek-v4-pro",
                "provider": "deepseek",
                "api_key": "deepseek-key",
                "base_url": "https://api.deepseek.com/v1",
                "api_mode": "chat_completions",
            },
        )
    ]
