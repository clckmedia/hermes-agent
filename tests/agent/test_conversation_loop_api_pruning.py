from __future__ import annotations

from types import SimpleNamespace


def test_api_tool_result_pruning_only_touches_request_copy(monkeypatch) -> None:
    from agent import conversation_loop

    messages = [
        {"role": "tool", "content": "x" * 10_000, "tool_call_id": "call_1"},
        {"role": "user", "content": "recent request"},
    ]

    class Compressor:
        threshold_tokens = 100_000
        protect_last_n = 1
        tail_token_budget = 90_000
        seen_tail_tokens = None

        def _prune_old_tool_results(
            self,
            api_messages,
            protect_tail_count,
            protect_tail_tokens=None,
        ):
            self.seen_tail_tokens = protect_tail_tokens
            pruned = [msg.copy() for msg in api_messages]
            pruned[0]["content"] = "[tool output summarized]"
            return pruned, 1

    compressor = Compressor()
    agent = SimpleNamespace(
        compression_enabled=True,
        context_compressor=compressor,
        tools=[{"type": "function"}],
    )

    def _estimate(api_messages, *args, **kwargs):
        return 45_000 if api_messages[0]["content"].startswith("[tool") else 90_000

    monkeypatch.setattr(conversation_loop, "estimate_request_tokens_rough", _estimate)

    pruned, count, tokens = conversation_loop._maybe_prune_api_tool_results(
        agent,
        messages,
        approx_request_tokens=90_000,
    )

    assert count == 1
    assert tokens == 45_000
    assert pruned[0]["content"] == "[tool output summarized]"
    assert messages[0]["content"] == "x" * 10_000
    assert compressor.seen_tail_tokens == 30_000


def test_api_tool_result_pruning_skips_small_requests() -> None:
    from agent import conversation_loop

    class Compressor:
        threshold_tokens = 100_000
        protect_last_n = 1
        tail_token_budget = 90_000

        def _prune_old_tool_results(self, *args, **kwargs):
            raise AssertionError("small requests should not be pruned")

    agent = SimpleNamespace(
        compression_enabled=True,
        context_compressor=Compressor(),
        tools=[],
    )
    messages = [{"role": "tool", "content": "short", "tool_call_id": "call_1"}]

    pruned, count, tokens = conversation_loop._maybe_prune_api_tool_results(
        agent,
        messages,
        approx_request_tokens=40_000,
    )

    assert pruned is messages
    assert count == 0
    assert tokens == 40_000
