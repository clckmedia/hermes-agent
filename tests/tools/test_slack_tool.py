"""Tests for read-only Slack retrieval tools."""

import json
import re
from unittest.mock import patch

import pytest

from tools import slack_tool
from tools.slack_tool import (
    _SLACK_READ_TOOL_NAMES,
    _sanitize_text,
    check_slack_read_requirements,
    slack_get_history,
    slack_get_thread,
    slack_list_conversations,
    slack_search_recent,
)


def _load(result_json: str) -> dict:
    return json.loads(result_json)


def test_check_requirements_uses_slack_bot_token(monkeypatch):
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    assert check_slack_read_requirements() is False

    monkeypatch.setenv("SLACK_BOT_TOKEN", "  xoxb-test-token  ")
    assert check_slack_read_requirements() is True


def test_list_conversations_clamps_limit_and_returns_bounded_fields(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test-token")
    api_response = {
        "ok": True,
        "channels": [
            {
                "id": "C1234567890",
                "name": "general",
                "is_channel": True,
                "is_private": False,
                "is_im": False,
                "is_archived": False,
                "num_members": 42,
                "topic": {"value": "A" * 500},
                "purpose": {"value": "Purpose"},
                "unneeded": "not returned",
            }
        ],
        "response_metadata": {"next_cursor": "next"},
    }

    with patch("tools.slack_tool._slack_api_get", return_value=api_response) as api:
        result = _load(slack_list_conversations({"types": "public_channel,private_channel", "limit": 999}))

    assert result["ok"] is True
    assert result["count"] == 1
    assert result["next_cursor"] == "next"
    assert result["conversations"][0] == {
        "id": "C1234567890",
        "name": "general",
        "is_private": False,
        "is_im": False,
        "is_mpim": False,
        "is_archived": False,
        "num_members": 42,
        "topic": "A" * (slack_tool._MAX_TOPIC_CHARS - 1) + "…",
        "purpose": "Purpose",
    }
    api.assert_called_once()
    assert api.call_args.args[1] == "conversations.list"
    assert api.call_args.args[2]["limit"] == "200"


def test_history_validates_channel_id_before_api_call(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test-token")

    with patch("tools.slack_tool._slack_api_get") as api:
        result = _load(slack_get_history({"channel_id": "not-a-channel", "limit": 999}))

    assert result["ok"] is False
    assert "channel_id" in result["error"]
    api.assert_not_called()


def test_history_redacts_tokens_private_urls_and_truncates_text(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test-token")
    secret = "xoxb-123456789012-secret-token-value"
    private_url = "https://files.slack.com/files-pri/T123-F456/download?access_token=abc123"
    long_text = f"hello {secret} {private_url} " + ("z" * 1500)
    api_response = {
        "ok": True,
        "messages": [
            {
                "type": "message",
                "user": "U123",
                "text": long_text,
                "ts": "1711111111.000100",
                "thread_ts": "1711111111.000100",
                "reply_count": 2,
                "files": [
                    {"id": "F1", "name": "secret.pdf", "mimetype": "application/pdf", "url_private": private_url}
                ],
            }
        ],
        "has_more": False,
    }

    with patch("tools.slack_tool._slack_api_get", return_value=api_response):
        result = _load(slack_get_history({"channel_id": "C1234567890", "limit": 500}))

    dumped = json.dumps(result)
    assert result["ok"] is True
    assert result["limit"] == 100
    assert secret not in dumped
    assert private_url not in dumped
    assert "files.slack.com" not in dumped
    message = result["messages"][0]
    assert len(message["text"]) <= slack_tool._MAX_MESSAGE_TEXT_CHARS + 1
    assert message["text_truncated"] is True
    assert message["files"] == [{"id": "F1", "name": "secret.pdf", "mimetype": "application/pdf"}]


def test_thread_uses_conversations_replies_and_shapes_permission_error(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test-token")
    api_response = {
        "ok": False,
        "error": "missing_scope",
        "needed": "channels:history",
        "provided": "chat:write,commands",
    }

    with patch("tools.slack_tool._slack_api_get", return_value=api_response) as api:
        result = _load(
            slack_get_thread(
                {
                    "channel_id": "C1234567890",
                    "thread_ts": "1711111111.000100",
                    "limit": 5,
                }
            )
        )

    assert result["ok"] is False
    assert result["reason"] == "missing_scope"
    assert result["error"] == "Slack API error: missing_scope"
    assert result["needed"] == "channels:history"
    assert result["provided"] == "chat:write,commands"
    assert api.call_args.args[1] == "conversations.replies"


@pytest.mark.parametrize("reason", ["not_in_channel", "not_allowed_token_type"])
def test_history_surfaces_slack_visibility_reasons_exactly(monkeypatch, reason):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test-token")

    with patch("tools.slack_tool._slack_api_get", return_value={"ok": False, "error": reason}):
        result = _load(slack_get_history({"channel_id": "C1234567890"}))

    assert result["ok"] is False
    assert result["reason"] == reason
    assert result["error"] == f"Slack API error: {reason}"


def test_search_recent_scans_bounded_history_without_slack_search_api(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test-token")

    def fake_api(_token, method, params, timeout=20):
        assert method != "search.messages"
        if method == "conversations.list":
            return {
                "ok": True,
                "channels": [
                    {"id": "C1111111111", "name": "alpha", "is_private": False},
                    {"id": "C2222222222", "name": "beta", "is_private": True},
                    {"id": "C3333333333", "name": "gamma", "is_private": False},
                ],
            }
        assert method == "conversations.history"
        if params["channel"] == "C1111111111":
            return {"ok": True, "messages": [{"ts": "1.000001", "user": "U1", "text": "Alpha launch update"}]}
        return {"ok": True, "messages": [{"ts": "2.000001", "user": "U2", "text": "nothing here"}]}

    with patch("tools.slack_tool._slack_api_get", side_effect=fake_api) as api:
        result = _load(slack_search_recent({"query": "alpha", "days": 999, "max_channels": 2, "limit": 50}))

    assert result["ok"] is True
    assert result["query"] == "alpha"
    assert result["days"] == 30
    assert result["max_channels"] == 2
    assert result["scanned_channels"] == 2
    assert result["matched_count"] == 1
    assert result["results"][0]["channel_id"] == "C1111111111"
    assert result["results"][0]["channel_name"] == "alpha"
    assert result["results"][0]["text"] == "Alpha launch update"
    methods = [call.args[1] for call in api.call_args_list]
    assert methods == ["conversations.list", "conversations.history", "conversations.history"]


def test_search_recent_accepts_channel_ids_and_returns_clean_no_results(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test-token")

    with patch("tools.slack_tool._slack_api_get", return_value={"ok": True, "messages": []}) as api:
        result = _load(slack_search_recent({"query": "needle", "channel_ids": ["C1111111111"], "limit": 1000}))

    assert result["ok"] is True
    assert result["matched_count"] == 0
    assert result["results"] == []
    assert result["limit"] == 25
    assert api.call_count == 1
    assert api.call_args.args[1] == "conversations.history"


def test_search_recent_counts_all_matches_but_bounds_returned_results(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test-token")

    with patch(
        "tools.slack_tool._slack_api_get",
        return_value={
            "ok": True,
            "messages": [
                {"ts": "1.000001", "user": "U1", "text": "needle first"},
                {"ts": "2.000001", "user": "U2", "text": "needle second"},
            ],
        },
    ):
        result = _load(slack_search_recent({"query": "needle", "channel_ids": ["C1111111111"], "limit": 1}))

    assert result["matched_count"] == 2
    assert len(result["results"]) == 1


def test_sanitize_text_redacts_private_urls_and_secret_query_params():
    text = (
        "url=https://example.com/path?access_token=supersecret "
        "private=https://files.slack.com/files-pri/T123-F456/download?pub_secret=abc "
        "token=xoxp-123456789012-secret"
    )

    cleaned = _sanitize_text(text)

    assert "supersecret" not in cleaned
    assert "files.slack.com" not in cleaned
    assert "xoxp-123456789012-secret" not in cleaned


def test_all_required_tools_are_listed_in_constant():
    assert set(_SLACK_READ_TOOL_NAMES) == {
        "slack_list_conversations",
        "slack_get_history",
        "slack_get_thread",
        "slack_search_recent",
        "slack_get_permalink",
    }
    assert all(re.match(r"^slack_", name) for name in _SLACK_READ_TOOL_NAMES)
