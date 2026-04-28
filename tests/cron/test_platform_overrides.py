import sys
import types
from unittest.mock import MagicMock

import cron.scheduler as scheduler


class _CapturingAgent:
    last_init = None

    def __init__(self, *args, **kwargs):
        type(self).last_init = dict(kwargs)

    def run_conversation(self, prompt):
        return {
            "final_response": "ok",
            "messages": [],
            "api_calls": 1,
            "completed": True,
        }


def test_run_job_honors_cron_platform_overrides(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text(
        "model:\n"
        "  default: gpt-5.4\n"
        "agent:\n"
        "  reasoning_effort: low\n"
        "  service_tier: normal\n"
        "  platforms:\n"
        "    cron:\n"
        "      reasoning_effort: xhigh\n"
        "      service_tier: fast\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(scheduler, "_hermes_home", tmp_path)
    monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0")

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _CapturingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    fake_state = types.ModuleType("hermes_state")
    fake_db = MagicMock()
    fake_state.SessionDB = lambda: fake_db
    monkeypatch.setitem(sys.modules, "hermes_state", fake_state)

    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: None)

    import hermes_cli.runtime_provider as runtime_provider
    monkeypatch.setattr(
        runtime_provider,
        "resolve_runtime_provider",
        lambda **kwargs: {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "***",
            "command": None,
            "args": [],
        },
    )

    import agent.smart_model_routing as smart_model_routing
    monkeypatch.setattr(
        smart_model_routing,
        "resolve_turn_route",
        lambda prompt, smart_routing, primary: {
            "model": "gpt-5.4",
            "runtime": dict(primary),
            "label": None,
            "signature": ("gpt-5.4", "openrouter", "https://openrouter.ai/api/v1", "chat_completions", None, ()),
        },
    )

    import hermes_cli.models as models
    monkeypatch.setattr(models, "resolve_fast_mode_overrides", lambda model: {"service_tier": "priority"})

    _CapturingAgent.last_init = None
    ok, output, final_response, error = scheduler.run_job(
        {
            "id": "job123",
            "name": "Cron override test",
            "prompt": "Say hi",
            "schedule_display": "manual",
            "skills": [],
        }
    )

    assert ok is True
    assert error is None
    assert final_response == "ok"
    assert "Cron override test" in output
    assert _CapturingAgent.last_init is not None
    assert _CapturingAgent.last_init["reasoning_config"] == {"enabled": True, "effort": "xhigh"}
    assert _CapturingAgent.last_init["service_tier"] == "priority"
    assert _CapturingAgent.last_init["request_overrides"] == {"service_tier": "priority"}


def test_run_job_honors_job_reasoning_and_enabled_toolsets(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text(
        "model:\n"
        "  default: gpt-5.4\n"
        "agent:\n"
        "  reasoning_effort: high\n"
        "  platforms:\n"
        "    cron:\n"
        "      reasoning_effort: xhigh\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(scheduler, "_hermes_home", tmp_path)
    monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0")

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _CapturingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    fake_state = types.ModuleType("hermes_state")
    fake_db = MagicMock()
    fake_state.SessionDB = lambda: fake_db
    monkeypatch.setitem(sys.modules, "hermes_state", fake_state)

    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: None)

    import hermes_cli.runtime_provider as runtime_provider
    monkeypatch.setattr(
        runtime_provider,
        "resolve_runtime_provider",
        lambda **kwargs: {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "***",
            "command": None,
            "args": [],
        },
    )

    import agent.smart_model_routing as smart_model_routing
    monkeypatch.setattr(
        smart_model_routing,
        "resolve_turn_route",
        lambda prompt, smart_routing, primary: {
            "model": "gpt-5.4",
            "runtime": dict(primary),
            "label": None,
            "signature": ("gpt-5.4", "openrouter", "https://openrouter.ai/api/v1", "chat_completions", None, ()),
        },
    )

    _CapturingAgent.last_init = None
    ok, output, final_response, error = scheduler.run_job(
        {
            "id": "job456",
            "name": "Per-job override test",
            "prompt": "Say hi",
            "schedule_display": "manual",
            "skills": [],
            "reasoning_effort": "medium",
            "enabled_toolsets": ["terminal", "instantly-readonly"],
        }
    )

    assert ok is True
    assert error is None
    assert final_response == "ok"
    assert _CapturingAgent.last_init is not None
    assert _CapturingAgent.last_init["reasoning_config"] == {"enabled": True, "effort": "medium"}
    assert _CapturingAgent.last_init["enabled_toolsets"] == ["terminal", "instantly-readonly"]
    assert _CapturingAgent.last_init["disabled_toolsets"] == ["cronjob", "messaging", "clarify"]


def test_run_job_disables_mcp_server_toolsets_when_job_has_no_explicit_toolsets(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text(
        "model:\n"
        "  default: gpt-5.4\n"
        "mcp_servers:\n"
        "  activepieces:\n"
        "    url: https://cloud.activepieces.com/mcp\n"
        "  twilio:\n"
        "    command: npx\n"
        "  disabled-server:\n"
        "    url: https://example.invalid/mcp\n"
        "    enabled: false\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(scheduler, "_hermes_home", tmp_path)
    monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0")

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _CapturingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    fake_state = types.ModuleType("hermes_state")
    fake_db = MagicMock()
    fake_state.SessionDB = lambda: fake_db
    monkeypatch.setitem(sys.modules, "hermes_state", fake_state)

    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: None)

    import hermes_cli.runtime_provider as runtime_provider
    monkeypatch.setattr(
        runtime_provider,
        "resolve_runtime_provider",
        lambda **kwargs: {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "***",
            "command": None,
            "args": [],
        },
    )

    import agent.smart_model_routing as smart_model_routing
    monkeypatch.setattr(
        smart_model_routing,
        "resolve_turn_route",
        lambda prompt, smart_routing, primary: {
            "model": "gpt-5.4",
            "runtime": dict(primary),
            "label": None,
            "signature": ("gpt-5.4", "openrouter", "https://openrouter.ai/api/v1", "chat_completions", None, ()),
        },
    )

    _CapturingAgent.last_init = None
    ok, _output, final_response, error = scheduler.run_job(
        {
            "id": "job789",
            "name": "Default MCP hardening test",
            "prompt": "Say hi",
            "schedule_display": "manual",
            "skills": [],
        }
    )

    assert ok is True
    assert error is None
    assert final_response == "ok"
    assert _CapturingAgent.last_init is not None
    disabled = set(_CapturingAgent.last_init["disabled_toolsets"])
    assert {"cronjob", "messaging", "clarify"}.issubset(disabled)
    assert {"activepieces", "mcp-activepieces", "twilio", "mcp-twilio"}.issubset(disabled)
    assert "disabled-server" not in disabled
    assert "mcp-disabled-server" not in disabled
