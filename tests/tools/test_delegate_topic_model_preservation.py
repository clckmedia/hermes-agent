from __future__ import annotations

from gateway.config import Platform
from gateway.session import SessionSource


def test_delegate_children_still_read_deepseek_defaults(monkeypatch) -> None:
    from tools import delegate_tool

    monkeypatch.setattr("cli.CLI_CONFIG", {}, raising=False)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "delegation": {
                "provider": "deepseek",
                "model": "deepseek-v4-pro",
                "reasoning_effort": "high",
            }
        },
    )

    cfg = delegate_tool._load_config()

    assert cfg["provider"] == "deepseek"
    assert cfg["model"] == "deepseek-v4-pro"


def test_topic_model_override_match_is_separate_from_delegation_defaults() -> None:
    from gateway.run import _resolve_topic_model_override_for_source

    config = {
        "zulip": {
            "topic_model_overrides": [
                {
                    "ids": ["595299", "595300", "595301"],
                    "topic_prefix": "-",
                    "provider": "deepseek",
                    "model": "deepseek-v4-pro",
                }
            ]
        }
    }
    source = SessionSource(
        platform=Platform.ZULIP,
        chat_id="stream:595301",
        chat_name="arlo-strategy-marketing",
        chat_type="channel",
        user_id="damien@example.com",
        user_name="Damien",
        thread_id="- child-demo",
        chat_topic="- child-demo",
        parent_chat_id="595301",
    )

    override = _resolve_topic_model_override_for_source(config, source)

    assert override == {"provider": "deepseek", "model": "deepseek-v4-pro"}
