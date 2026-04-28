import threading
from unittest.mock import patch

import run_agent
from run_agent import AIAgent


def _make_tool_defs(*names: str):
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def test_spawn_background_review_inherits_parent_tool_scope(monkeypatch):
    captured_kwargs = []
    finished = threading.Event()

    class FakeReviewAgent:
        def __init__(self, **kwargs):
            captured_kwargs.append(kwargs)
            self._session_messages = []
            self._safe_print = lambda *a, **k: None

        def run_conversation(self, **kwargs):
            finished.set()
            return {"final_response": "done"}

        def close(self):
            pass

    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("memory", "skill_manage")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        parent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            platform="discord",
            provider="openrouter",
            enabled_toolsets=["memory", "skills"],
            disabled_toolsets=["messaging"],
        )

    parent._memory_store = None
    parent._memory_enabled = False
    parent._user_profile_enabled = False

    monkeypatch.setattr(run_agent, "AIAgent", FakeReviewAgent)

    parent._spawn_background_review(
        messages_snapshot=[{"role": "user", "content": "Hello"}],
        review_skills=True,
    )

    assert finished.wait(2.0), "Background review task should run"

    assert len(captured_kwargs) == 1
    review_kwargs = captured_kwargs[0]

    assert review_kwargs["model"] == parent.model
    assert review_kwargs["provider"] == parent.provider
    assert review_kwargs["platform"] == parent.platform
    assert review_kwargs["max_iterations"] == 8
    assert review_kwargs["quiet_mode"] is True
    assert review_kwargs["enabled_toolsets"] == parent.enabled_toolsets
    assert review_kwargs["disabled_toolsets"] == parent.disabled_toolsets
