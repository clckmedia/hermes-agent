"""Tests for user-visible compression-summary warning emission."""

import os
from unittest.mock import MagicMock, patch


class TestCompressionSummaryWarningVisibility:
    def _make_agent(self):
        from run_agent import AIAgent

        with (
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}),
            patch("agent.context_compressor.get_model_context_length", return_value=100_000),
        ):
            return AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                quiet_mode=True,
                session_db=None,
                session_id="original-session",
                skip_context_files=True,
                skip_memory=True,
            )

    @staticmethod
    def _stub_compressor(
        *,
        fallback_used: bool,
        summary_error: str | None = (
            "peer closed connection without sending complete message body "
            "(incomplete chunked read)"
        ),
        aux_model: str | None = None,
        aux_error: str | None = None,
        local_digest_used: bool = False,
        local_digest_error: str | None = None,
    ):
        compressor = MagicMock()
        compressor.compress.return_value = [
            {"role": "assistant", "content": "summary or fallback"},
            {"role": "user", "content": "tail question"},
        ]
        compressor.compression_count = 1
        compressor.last_prompt_tokens = 0
        compressor.last_completion_tokens = 0
        compressor._last_summary_error = summary_error
        compressor._last_summary_fallback_used = fallback_used
        compressor._last_aux_model_failure_model = aux_model
        compressor._last_aux_model_failure_error = aux_error
        compressor._last_local_timeout_digest_used = local_digest_used
        compressor._last_local_timeout_digest_error = local_digest_error
        return compressor

    def test_summary_error_without_final_fallback_does_not_emit_public_warning(self):
        agent = self._make_agent()
        agent.context_compressor = self._stub_compressor(fallback_used=False)
        warnings = []
        agent._emit_warning = lambda message: warnings.append(message)

        agent._compress_context(
            [{"role": "user", "content": "m"}],
            "sys",
            approx_tokens=10_000,
        )

        assert warnings == []

    def test_summary_error_with_final_fallback_emits_public_warning(self):
        agent = self._make_agent()
        agent.context_compressor = self._stub_compressor(fallback_used=True)
        warnings = []
        agent._emit_warning = lambda message: warnings.append(message)

        agent._compress_context(
            [{"role": "user", "content": "m"}],
            "sys",
            approx_tokens=10_000,
        )

        assert len(warnings) == 1
        assert "Compression summary failed" in warnings[0]
        assert "Inserted a fallback context marker" in warnings[0]

    def test_codex_local_timeout_digest_emits_specific_note_not_aux_config_warning(self):
        agent = self._make_agent()
        agent.context_compressor = self._stub_compressor(
            fallback_used=False,
            summary_error=None,
            local_digest_used=True,
            local_digest_error="Codex auxiliary Responses stream exceeded 120.0s total timeout",
        )
        warnings = []
        agent._emit_warning = lambda message: warnings.append(message)

        agent._compress_context(
            [{"role": "user", "content": "m"}],
            "sys",
            approx_tokens=10_000,
        )

        assert len(warnings) == 1
        assert "Codex compression summariser timed out after 120s" in warnings[0]
        assert "local continuity digest" in warnings[0]
        assert "context is intact" in warnings[0]
        assert "No config change needed" in warnings[0]
        assert "Recovered using main model" not in warnings[0]
        assert "auxiliary.compression.model" not in warnings[0]

    def test_aux_model_failure_recovered_via_main_still_emits_config_warning(self):
        agent = self._make_agent()
        agent.context_compressor = self._stub_compressor(
            fallback_used=False,
            summary_error=None,
            aux_model="broken-aux-model",
            aux_error="404 model_not_found",
        )
        warnings = []
        agent._emit_warning = lambda message: warnings.append(message)

        agent._compress_context(
            [{"role": "user", "content": "m"}],
            "sys",
            approx_tokens=10_000,
        )

        assert len(warnings) == 1
        assert "Configured compression model 'broken-aux-model' failed" in warnings[0]
        assert "Recovered using main model" in warnings[0]
        assert "auxiliary.compression.model" in warnings[0]
