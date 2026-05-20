"""Tests for Slack channel_skill_bindings auto-skill resolution."""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest


def _install_clck_registry_fixture(monkeypatch, tmp_path):
    """Point Slack's CLCK registry lookup at deterministic temp JSON files."""
    from gateway.platforms import slack as slack_module

    routes_path = tmp_path / "clck_support_triage_client_routes.json"
    portal_path = tmp_path / "hubspot_portal_registry.json"
    routes_path.write_text(json.dumps({
        "client_routes": [
            {
                "client_key": "fhs_poly",
                "client_name": "FHS Poly",
                "contact_primary_name": "Ross Bennett",
                "contact_primary_email": "ross@fhs.com.au",
                "slack_channel_id": "C0A9GQUHKNK",
                "slack_channel_name": "fhs-poly",
                "hubspot_portal_id": "23619105",
                "hubspot_portal_key": "fhs_poly",
                "hubspot_portal_registry_name": "FHS Poly",
            },
            {
                "client_key": "neopharma_technologies",
                "client_name": "Neopharma Technologies",
                "contact_primary_name": "Joseph Davies",
                "contact_primary_email": "joe@neopharmatechnologies.com",
                "slack_channel_id": "C098K5JCHDW",
                "slack_channel_name": "neopharma-internal",
                "hubspot_portal_id": "442005585",
                "hubspot_portal_key": "neopharma",
                "hubspot_portal_registry_name": "Neopharma",
            },
        ]
    }), encoding="utf-8")
    portal_path.write_text(json.dumps({
        "clients": {
            "fhs_poly": {"client_name": "FHS Poly", "portal_id": 23619105},
            "neopharma": {"client_name": "Neopharma", "portal_id": 442005585},
        }
    }), encoding="utf-8")
    monkeypatch.setattr(slack_module, "_clck_registry_paths", lambda: (routes_path, portal_path))
    return slack_module, routes_path, portal_path


def _make_adapter(extra=None):
    """Create a minimal SlackAdapter stub with the given ``config.extra``."""
    from gateway.platforms.slack import SlackAdapter
    adapter = object.__new__(SlackAdapter)
    adapter.config = MagicMock()
    adapter.config.extra = extra or {}
    return adapter


def _resolve(adapter, channel_id, parent_id=None):
    from gateway.platforms.base import resolve_channel_skills
    return resolve_channel_skills(adapter.config.extra, channel_id, parent_id)


class TestSlackResolveChannelSkills:
    def test_no_bindings_returns_none(self):
        adapter = _make_adapter()
        assert _resolve(adapter, "D0ABC") is None

    def test_match_by_dm_channel_id(self):
        """The primary use case: binding a skill to a Slack DM channel."""
        adapter = _make_adapter({
            "channel_skill_bindings": [
                {"id": "D0ATH9TQ0G6", "skills": ["german-flashcards"]},
            ]
        })
        assert _resolve(adapter, "D0ATH9TQ0G6") == ["german-flashcards"]

    def test_match_by_parent_id_for_thread(self):
        """Slack threads inherit the parent channel's binding."""
        adapter = _make_adapter({
            "channel_skill_bindings": [
                {"id": "C0PARENT", "skills": ["parent-skill"]},
            ]
        })
        assert _resolve(adapter, "thread-ts-123", parent_id="C0PARENT") == ["parent-skill"]

    def test_no_match_returns_none(self):
        adapter = _make_adapter({
            "channel_skill_bindings": [
                {"id": "D0AAA", "skills": ["skill-a"]},
            ]
        })
        assert _resolve(adapter, "D0BBB") is None

    def test_single_skill_string(self):
        adapter = _make_adapter({
            "channel_skill_bindings": [
                {"id": "D0ATH9TQ0G6", "skill": "german-flashcards"},
            ]
        })
        assert _resolve(adapter, "D0ATH9TQ0G6") == ["german-flashcards"]

    def test_dedup_preserves_order(self):
        adapter = _make_adapter({
            "channel_skill_bindings": [
                {"id": "D0ATH9TQ0G6", "skills": ["a", "b", "a", "c", "b"]},
            ]
        })
        assert _resolve(adapter, "D0ATH9TQ0G6") == ["a", "b", "c"]

    def test_multiple_bindings_pick_correct(self):
        adapter = _make_adapter({
            "channel_skill_bindings": [
                {"id": "D0AAA", "skills": ["skill-a"]},
                {"id": "D0BBB", "skills": ["skill-b"]},
                {"id": "D0CCC", "skills": ["skill-c"]},
            ]
        })
        assert _resolve(adapter, "D0BBB") == ["skill-b"]

    def test_malformed_entry_skipped(self):
        """Non-dict entries should be ignored, not raise."""
        adapter = _make_adapter({
            "channel_skill_bindings": [
                "not-a-dict",
                {"id": "D0ABC", "skills": ["good"]},
            ]
        })
        assert _resolve(adapter, "D0ABC") == ["good"]

    def test_empty_skills_list_returns_none(self):
        adapter = _make_adapter({
            "channel_skill_bindings": [
                {"id": "D0ABC", "skills": []},
            ]
        })
        assert _resolve(adapter, "D0ABC") is None

    def test_empty_skill_string_returns_none(self):
        adapter = _make_adapter({
            "channel_skill_bindings": [
                {"id": "D0ABC", "skill": ""},
            ]
        })
        assert _resolve(adapter, "D0ABC") is None


class TestSlackMessageEventAutoSkill:
    """Integration-style test: verify auto_skill propagates to MessageEvent."""

    def test_message_event_carries_auto_skill(self):
        """Simulate the handler wiring: resolve + attach to MessageEvent."""
        from gateway.platforms.base import MessageEvent, MessageType, Platform, SessionSource, resolve_channel_skills

        config_extra = {
            "channel_skill_bindings": [
                {"id": "D0ATH9TQ0G6", "skills": ["german-flashcards"]},
            ]
        }
        auto_skill = resolve_channel_skills(config_extra, "D0ATH9TQ0G6", None)

        source = SessionSource(
            platform=Platform.SLACK,
            chat_id="D0ATH9TQ0G6",
            chat_name="Mats",
            chat_type="dm",
            user_id="U0ABC",
            user_name="Mats",
        )
        event = MessageEvent(
            text="work",
            message_type=MessageType.TEXT,
            source=source,
            raw_message={},
            message_id="123.456",
            auto_skill=auto_skill,
        )
        assert event.auto_skill == ["german-flashcards"]


class TestSlackCLCKClientRegistryContext:
    @pytest.mark.parametrize(
        ("channel_id", "channel_name", "client", "client_key", "portal_id", "contact"),
        [
            ("C0A9GQUHKNK", "fhs-poly", "FHS Poly", "fhs_poly", "23619105", "Ross Bennett <ross@fhs.com.au>"),
            ("C098K5JCHDW", "neopharma-internal", "Neopharma Technologies", "neopharma_technologies", "442005585", "Joseph Davies <joe@neopharmatechnologies.com>"),
        ],
    )
    def test_registry_context_for_known_clck_client_channels(
        self,
        monkeypatch,
        tmp_path,
        channel_id,
        channel_name,
        client,
        client_key,
        portal_id,
        contact,
    ):
        slack_module, routes_path, portal_path = _install_clck_registry_fixture(monkeypatch, tmp_path)

        ctx = slack_module._build_clck_slack_client_context(channel_id)

        assert ctx is not None
        assert ctx.channel_name == channel_name
        assert f"#{channel_name} ({channel_id})" in ctx.prompt
        assert f"Client: {client} (key: {client_key})" in ctx.prompt
        assert f"portal ID {portal_id}" in ctx.prompt
        assert contact in ctx.prompt
        assert str(routes_path) in ctx.prompt
        assert str(portal_path) in ctx.prompt
        assert "channel/client registry context beats recent/session guesses" in ctx.prompt
        assert "invoice/accounting defaults to Xero" in ctx.prompt
        assert "client-system writes without approval" in ctx.prompt
        assert "SERVICE_KEY" not in ctx.prompt
        assert "ACCESS_TOKEN" not in ctx.prompt
        assert "/secrets/" not in ctx.prompt

    def test_registry_context_fail_open_for_missing_or_unmatched_channel(self, monkeypatch, tmp_path):
        slack_module, _routes_path, _portal_path = _install_clck_registry_fixture(monkeypatch, tmp_path)

        assert slack_module._build_clck_slack_client_context("C0UNKNOWN") is None

        missing = tmp_path / "missing.json"
        monkeypatch.setattr(slack_module, "_clck_registry_paths", lambda: (missing, missing))
        assert slack_module._build_clck_slack_client_context("C0A9GQUHKNK") is None

    def test_registry_prompt_prepends_and_preserves_config_prompt(self, monkeypatch, tmp_path):
        slack_module, _routes_path, _portal_path = _install_clck_registry_fixture(monkeypatch, tmp_path)
        registry_prompt = slack_module._build_clck_slack_client_context("C0A9GQUHKNK").prompt

        combined = slack_module._combine_slack_channel_prompts(registry_prompt, "Configured channel prompt")

        assert combined.startswith("[CLCK client registry context")
        assert "Configured channel prompt" in combined
        assert combined.index("[CLCK client registry context") < combined.index("Configured channel prompt")

    @pytest.mark.asyncio
    async def test_slack_inbound_event_carries_registry_context_and_channel_name(self, monkeypatch, tmp_path):
        slack_module, _routes_path, _portal_path = _install_clck_registry_fixture(monkeypatch, tmp_path)
        from gateway.config import Platform

        adapter = object.__new__(slack_module.SlackAdapter)
        adapter.platform = Platform.SLACK
        adapter.config = MagicMock()
        adapter.config.extra = {
            "channel_prompts": {"C0A9GQUHKNK": "Configured FHS prompt"},
            "channel_skill_bindings": [
                {"id": "C0A9GQUHKNK", "skills": ["clck-client-operations"]},
            ],
        }
        adapter._dedup = MagicMock()
        adapter._dedup.is_duplicate.return_value = False
        adapter._lookup_assistant_thread_metadata = MagicMock(return_value={})
        adapter._team_bot_user_ids = {}
        adapter._bot_user_id = None
        adapter._channel_team = {}
        adapter._slack_free_response_channels = MagicMock(return_value=set())
        adapter._resolve_user_name = AsyncMock(return_value="Damien")
        adapter._reactions_enabled = MagicMock(return_value=False)
        adapter._reacting_message_ids = set()

        captured = {}

        async def capture_message(message_event):
            captured["event"] = message_event

        adapter.handle_message = capture_message

        await adapter._handle_slack_message({
            "channel": "C0A9GQUHKNK",
            "channel_type": "channel",
            "team": "T0CLCK",
            "ts": "123.456",
            "user": "U0DAMIEN",
            "text": "What is the portal?",
        })

        event = captured["event"]
        assert event.source.chat_name == "fhs-poly"
        assert event.channel_prompt.startswith("[CLCK client registry context")
        assert "FHS Poly (key: fhs_poly)" in event.channel_prompt
        assert "portal ID 23619105" in event.channel_prompt
        assert "Configured FHS prompt" in event.channel_prompt
        assert event.auto_skill == ["clck-client-operations"]
