"""Tests graceful in-loop compression for large retryable-session failures.

These cover the case where a session is already huge, but the provider returns a
retryable transport/server-style error instead of an explicit context-overflow
message. The agent should compact once before burning through normal retries.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import run_agent
from run_agent import AIAgent


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch):
    import time as _time

    monkeypatch.setattr(_time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(run_agent, "jittered_backoff", lambda *a, **k: 0.0)



def _mock_response(content="OK", finish_reason="stop", tool_calls=None, usage=None):
    msg = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning_content=None,
        reasoning=None,
    )
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    resp = SimpleNamespace(choices=[choice], model="test/model")
    resp.usage = SimpleNamespace(**usage) if usage else None
    return resp



def _make_status_error(status_code=502, message="Bad gateway"):
    err = Exception(message)
    err.status_code = status_code
    return err


@pytest.fixture()
def agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        a._cached_system_prompt = "You are helpful."
        a._use_prompt_caching = False
        a.tool_delay = 0
        a.compression_enabled = True
        a.save_trajectories = False
        return a


class TestLargeSessionRetryableCompression:
    def test_large_retryable_server_error_triggers_compression(self, agent):
        """A big session + 5xx should compact once before ordinary retries/fallback."""
        err_502 = _make_status_error(502, "Bad gateway")
        ok_resp = _mock_response(content="Recovered after compaction")
        agent.client.chat.completions.create.side_effect = [err_502, ok_resp]

        big_history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(160)
        ]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "hello"}],
                "compressed prompt",
            )
            result = agent.run_conversation("hello", conversation_history=big_history)

        mock_compress.assert_called_once()
        assert result["completed"] is True
        assert result["final_response"] == "Recovered after compaction"

    def test_small_retryable_server_error_does_not_trigger_compression(self, agent):
        """Normal-sized sessions should keep the existing retry path."""
        err_502 = _make_status_error(502, "Bad gateway")
        ok_resp = _mock_response(content="Recovered without compaction")
        agent.client.chat.completions.create.side_effect = [err_502, ok_resp]

        small_history = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=small_history)

        mock_compress.assert_not_called()
        assert result["completed"] is True
        assert result["final_response"] == "Recovered without compaction"

    def test_large_session_final_failure_includes_degradation_hint(self, agent):
        """If retries still fail after automatic compaction, the user gets a clear hint."""
        err_502 = _make_status_error(502, "Bad gateway")
        agent.client.chat.completions.create.side_effect = [err_502, err_502, err_502, err_502]

        big_history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(160)
        ]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "hello"}],
                "compressed prompt",
            )
            result = agent.run_conversation("hello", conversation_history=big_history)

        assert result["failed"] is True
        assert "context degradation is likely" in result["final_response"]
        assert "Hermes already tried one automatic compaction pass" in result["final_response"]
        assert "/compact" in result["final_response"]
        assert "/resume" in result["final_response"]

    def test_small_session_final_failure_stays_generic(self, agent):
        """Normal retry failures should not get the giant-session hint."""
        err_502 = _make_status_error(502, "Bad gateway")
        agent.client.chat.completions.create.side_effect = [err_502, err_502, err_502]

        small_history = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=small_history)

        mock_compress.assert_not_called()
        assert result["failed"] is True
        assert "context degradation is likely" not in result["final_response"]
        assert "/compact" not in result["final_response"]
