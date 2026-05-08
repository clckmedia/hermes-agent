"""Integration coverage for Slack mention-prefixed gateway commands."""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageType
from gateway.session import SessionSource, build_session_key


# Keep this file runnable through tests/gateway/test_commands*.py even when
# slack-bolt is not installed in the local test environment.
def _ensure_slack_mock():
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return

    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock

    slack_sdk = MagicMock()
    slack_sdk.web.async_client.AsyncWebClient = MagicMock

    for name, mod in [
        ("slack_bolt", slack_bolt),
        ("slack_bolt.async_app", slack_bolt.async_app),
        ("slack_bolt.adapter", slack_bolt.adapter),
        ("slack_bolt.adapter.socket_mode", slack_bolt.adapter.socket_mode),
        ("slack_bolt.adapter.socket_mode.async_handler", slack_bolt.adapter.socket_mode.async_handler),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)


_ensure_slack_mock()

import gateway.platforms.slack as _slack_mod  # noqa: E402

_slack_mod.SLACK_AVAILABLE = True

from gateway.platforms.slack import SlackAdapter  # noqa: E402


def _thread_source(thread_id: str = "171.000") -> SessionSource:
    return SessionSource(
        platform=Platform.SLACK,
        chat_id="C_THREAD",
        chat_type="group",
        user_id="U_USER",
        thread_id=thread_id,
    )


def _thread_session_key(thread_id: str = "171.000") -> str:
    return build_session_key(
        _thread_source(thread_id),
        group_sessions_per_user=True,
        thread_sessions_per_user=False,
    )


def _make_adapter() -> SlackAdapter:
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._app = MagicMock()
    adapter._app.client = AsyncMock()
    adapter._bot_user_id = "U_BOT"
    adapter._team_bot_user_ids["T_TEAM"] = "U_BOT"
    adapter._resolve_user_name = AsyncMock(return_value="Damien")
    adapter._fetch_thread_context = AsyncMock(return_value="[Thread context should not prefix commands]\n")
    adapter._fetch_thread_parent_text = AsyncMock(return_value="")
    adapter.handled_events = []
    adapter.sent_responses = []

    async def _handler(event):
        adapter.handled_events.append(event)
        return f"handled:{event.text}"

    async def _send_with_retry(chat_id, content, **kwargs):
        adapter.sent_responses.append((chat_id, content, kwargs))

    adapter._message_handler = _handler
    adapter._send_with_retry = _send_with_retry
    return adapter


@pytest.mark.asyncio
async def test_active_session_bypass_handles_slack_mention_prefixed_reset_for_current_thread_only():
    adapter = _make_adapter()
    current_key = _thread_session_key("171.000")
    other_key = _thread_session_key("999.000")
    adapter._active_sessions[current_key] = asyncio.Event()
    adapter._active_sessions[other_key] = asyncio.Event()

    await adapter._handle_slack_message({
        "text": "<@U_BOT> /reset",
        "user": "U_USER",
        "channel": "C_THREAD",
        "channel_type": "channel",
        "thread_ts": "171.000",
        "ts": "171.111",
        "team": "T_TEAM",
        "blocks": [
            {
                "type": "rich_text",
                "elements": [
                    {
                        "type": "rich_text_section",
                        "elements": [{"type": "text", "text": "ignored block payload"}],
                    }
                ],
            }
        ],
    })

    assert len(adapter.handled_events) == 1
    handled = adapter.handled_events[0]
    assert handled.text == "/reset"
    adapter._fetch_thread_context.assert_not_called()
    assert handled.message_type is MessageType.COMMAND
    assert handled.get_command() == "reset"
    assert handled.source.thread_id == "171.000"

    assert current_key not in adapter._pending_messages
    assert current_key not in adapter._active_sessions
    assert other_key in adapter._active_sessions
    assert adapter.sent_responses == [
        (
            "C_THREAD",
            "handled:/reset",
            {"reply_to": "171.111", "metadata": {"thread_id": "171.000"}},
        )
    ]
