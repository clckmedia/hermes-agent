"""Tests for user-facing wording around automatic session resets."""

from gateway.config import SessionResetPolicy
from gateway.run import should_gate_auto_reset_message
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


def test_suspended_restart_resets_gate_the_user_message():
    """Crash/restart-style resets should notify and stop before agent work.

    This prevents a context-dependent follow-up from spending minutes in a
    fresh session after the user has not yet seen that context was lost.
    """
    assert should_gate_auto_reset_message("suspended") is True


def test_idle_and_oversized_resets_do_not_gate_by_default():
    assert should_gate_auto_reset_message("idle") is False
    assert should_gate_auto_reset_message("daily") is False
    assert should_gate_auto_reset_message("oversized") is False
