"""Regression tests for config-driven always_include_skills prompt loading."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

from hermes_constants import get_skills_dir
from agent.prompt_builder import clear_skills_system_prompt_cache
from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list[dict]:
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


def _make_skill(skills_dir: Path, name: str, description: str, body: str) -> None:
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n\n{body}\n",
        encoding="utf-8",
    )


def _build_agent_prompt(config: dict, *, system_message: str | None = None) -> str:
    tool_defs = _make_tool_defs("skills_list", "skill_view", "skill_manage")
    toolset_map = {
        "skills_list": "skills",
        "skill_view": "skills",
        "skill_manage": "skills",
    }
    with (
        patch("run_agent.get_tool_definitions", return_value=tool_defs),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.get_toolset_for_tool", create=True, side_effect=toolset_map.get),
        patch("run_agent.OpenAI"),
        patch("hermes_cli.config.load_config", return_value=config),
    ):
        agent = AIAgent(
            model="anthropic/claude-sonnet-4",
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        return agent._build_system_prompt(system_message=system_message)


def test_always_include_skills_loads_full_body_and_keeps_skill_index():
    clear_skills_system_prompt_cache(clear_snapshot=True)
    skills_dir = get_skills_dir()
    _make_skill(
        skills_dir,
        "pinned-skill",
        "Pinned index description.",
        "PINNED_FULL_BODY_SENTINEL",
    )
    _make_skill(
        skills_dir,
        "index-only-skill",
        "Index-only description.",
        "INDEX_ONLY_BODY_SENTINEL",
    )

    config = {
        "agent": {"tool_use_enforcement": False},
        "skills": {"prompt_index": {"always_include_skills": ["pinned-skill"]}},
    }

    with patch("tools.skills_tool.SKILLS_DIR", skills_dir):
        prompt = _build_agent_prompt(config)

    assert "## Skills (mandatory)" in prompt
    assert "- pinned-skill: Pinned index description." in prompt
    assert "- index-only-skill: Index-only description." in prompt
    assert "PINNED_FULL_BODY_SENTINEL" in prompt
    assert "INDEX_ONLY_BODY_SENTINEL" not in prompt


def test_missing_always_include_skill_logs_warning_but_does_not_crash(caplog):
    clear_skills_system_prompt_cache(clear_snapshot=True)
    skills_dir = get_skills_dir()
    _make_skill(
        skills_dir,
        "present-skill",
        "Present index description.",
        "PRESENT_FULL_BODY_SENTINEL",
    )
    config = {
        "agent": {"tool_use_enforcement": False},
        "skills": {
            "prompt_index": {
                "always_include_skills": ["present-skill", "missing-skill"]
            }
        },
    }
    caplog.set_level(logging.WARNING, logger="agent.prompt_builder")

    with patch("tools.skills_tool.SKILLS_DIR", skills_dir):
        prompt = _build_agent_prompt(config)

    assert "PRESENT_FULL_BODY_SENTINEL" in prompt
    assert "Conversation started:" in prompt
    assert any("missing-skill" in record.message for record in caplog.records)


def test_always_include_skills_does_not_duplicate_cli_preloaded_skill_body():
    clear_skills_system_prompt_cache(clear_snapshot=True)
    skills_dir = get_skills_dir()
    _make_skill(
        skills_dir,
        "pinned-skill",
        "Pinned index description.",
        "PINNED_FULL_BODY_SENTINEL",
    )
    config = {
        "agent": {"tool_use_enforcement": False},
        "skills": {"prompt_index": {"always_include_skills": ["pinned-skill"]}},
    }
    preloaded_system_message = (
        '[SYSTEM: The user launched this CLI session with the "pinned-skill" skill '
        "preloaded. Treat its instructions as active guidance.]\n\n"
        "---\nname: pinned-skill\ndescription: Pinned index description.\n---\n\n"
        "# pinned-skill\n\nPINNED_FULL_BODY_SENTINEL"
    )

    with patch("tools.skills_tool.SKILLS_DIR", skills_dir):
        prompt = _build_agent_prompt(config, system_message=preloaded_system_message)

    assert prompt.count("PINNED_FULL_BODY_SENTINEL") == 1
