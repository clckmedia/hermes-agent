"""Tests for user-visible compression-summary warning emission."""

import os
from unittest.mock import MagicMock, patch


class TestCompressionSummaryWarningVisibility:
    def _make_agent(self):
        from run_agent import AIAgent

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
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
    def _stub_compressor(*, fallback_used: bool):
        compressor = MagicMock()
        compressor.compress.return_value = [
            {"role": "assistant", "content": "summary or fallback"},
            {"role": "user", "content": "tail question"},
        ]
        compressor.compression_count = 1
        compressor.last_prompt_tokens = 0
        compressor.last_completion_tokens = 0
        compressor._last_summary_error = (
            "peer closed connection without sending complete message body "
            "(incomplete chunked read)"
        )
        compressor._last_summary_fallback_used = fallback_used
        compressor._last_aux_model_failure_model = None
        compressor._last_aux_model_failure_error = None
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
