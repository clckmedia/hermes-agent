"""Tests for deterministic fallback compaction when LLM summarisation fails."""

from gateway.session import build_forced_compaction_fallback_history


class TestForcedCompactionFallbackHistory:
    def test_preserves_session_meta_and_recent_tail(self):
        history = [{"role": "session_meta", "tools": []}]
        for idx in range(10):
            history.append({"role": "user", "content": f"u{idx}"})
            history.append({"role": "assistant", "content": f"a{idx}"})

        compressed = build_forced_compaction_fallback_history(history, keep_last_messages=6)

        assert compressed[0]["role"] == "session_meta"
        assert compressed[1]["role"] == "assistant"
        assert "Automatic compression fallback" in compressed[1]["content"]
        assert compressed[2:] == history[-6:]

    def test_short_history_is_returned_unchanged(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]

        compressed = build_forced_compaction_fallback_history(history, keep_last_messages=6)

        assert compressed == history
