"""Tests for base gateway message command parsing safety."""

import pytest

from gateway.config import Platform
from gateway.platforms.base import (
    MessageEvent,
    MessageType,
    coerce_plaintext_gateway_command,
)
from gateway.session import SessionSource


def _event(text: str, *, platform: Platform = Platform.TELEGRAM, chat_type: str = "dm") -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=platform,
            chat_id="C1",
            chat_type=chat_type,
            user_id="U1",
        ),
    )


def test_non_slack_mention_prefixed_slash_is_not_rewritten_as_command():
    event = _event("<@U_BOT> /reset", platform=Platform.TELEGRAM, chat_type="group")

    coerce_plaintext_gateway_command(event)

    assert event.text == "<@U_BOT> /reset"
    assert event.message_type is MessageType.TEXT
    assert event.is_command() is False
    assert event.get_command() is None


@pytest.mark.parametrize(
    "text",
    [
        "reset yourself",
        "reset this",
        "arlo reset",
        "please /reset this session",
    ],
)
def test_reset_prose_is_not_rewritten(text):
    event = _event(text)

    coerce_plaintext_gateway_command(event)

    assert event.text == text
    assert event.message_type is MessageType.TEXT
    assert event.get_command() is None
