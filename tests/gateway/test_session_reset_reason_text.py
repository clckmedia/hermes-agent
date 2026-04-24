"""Tests for user-facing wording around automatic session resets."""

from gateway.config import SessionResetPolicy
from gateway.session import describe_auto_reset_reason


def test_describe_auto_reset_reason_for_oversized_session():
    context_note, reason_text = describe_auto_reset_reason(
        "oversized",
        SessionResetPolicy(mode="idle", idle_minutes=4320, max_input_tokens=2_000_000),
    )

    assert "grew too large" in context_note
    assert "do not guess" in context_note
    assert "session_search" in context_note
    assert reason_text == "session grew too large"


def test_describe_auto_reset_reason_for_idle_session():
    context_note, reason_text = describe_auto_reset_reason(
        "idle",
        SessionResetPolicy(mode="idle", idle_minutes=150),
    )

    assert "inactivity" in context_note
    assert "do not guess" in context_note
    assert reason_text == "inactive for 2h 30m"
