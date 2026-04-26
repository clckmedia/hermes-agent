"""Regression tests for config-driven always_include_skills prompt loading."""

import logging
from contextlib import ExitStack
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


def _make_skill(
    skills_dir: Path,
    name: str,
    description: str,
    body: str,
    *,
    category: str | None = None,
) -> None:
    skill_dir = skills_dir / category / name if category else skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n\n{body}\n",
        encoding="utf-8",
    )


def _build_agent_prompt(
    config: dict,
    *,
    system_message: str | None = None,
    skills_dir: Path | None = None,
) -> str:
    tool_defs = _make_tool_defs("skills_list", "skill_view", "skill_manage")
    toolset_map = {
        "skills_list": "skills",
        "skill_view": "skills",
        "skill_manage": "skills",
    }
    with ExitStack() as stack:
        stack.enter_context(patch("run_agent.get_tool_definitions", return_value=tool_defs))
        stack.enter_context(patch("run_agent.check_toolset_requirements", return_value={}))
        stack.enter_context(
            patch("run_agent.get_toolset_for_tool", create=True, side_effect=toolset_map.get)
        )
        stack.enter_context(patch("run_agent.OpenAI"))
        stack.enter_context(patch("hermes_cli.config.load_config", return_value=config))
        if skills_dir is not None:
            hermes_home = skills_dir.parent
            stack.enter_context(patch("agent.prompt_builder.get_skills_dir", return_value=skills_dir))
            stack.enter_context(patch("agent.prompt_builder.get_all_skills_dirs", return_value=[skills_dir]))
            stack.enter_context(patch("agent.prompt_builder.get_hermes_home", return_value=hermes_home))
            stack.enter_context(patch("tools.skills_tool.SKILLS_DIR", skills_dir))

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


def test_always_include_categories_loads_full_bodies_and_keeps_skill_index(tmp_path):
    clear_skills_system_prompt_cache(clear_snapshot=True)
    skills_dir = tmp_path / "skills"
    _make_skill(
        skills_dir,
        "outbound-alpha",
        "Outbound alpha description.",
        "OUTBOUND_ALPHA_FULL_BODY_SENTINEL",
        category="outbound",
    )
    _make_skill(
        skills_dir,
        "outbound-beta",
        "Outbound beta description.",
        "OUTBOUND_BETA_FULL_BODY_SENTINEL",
        category="outbound",
    )
    _make_skill(
        skills_dir,
        "docs-only",
        "Docs-only description.",
        "DOCS_ONLY_FULL_BODY_SENTINEL",
        category="docs",
    )
    config = {
        "agent": {"tool_use_enforcement": False},
        "skills": {
            "prompt_index": {
                "always_include_categories": ["outbound"],
                "max_categories": 5,
                "max_skills_per_category": 3,
                "max_total_skills": 20,
                "always_include_skills": [],
            }
        },
    }

    prompt = _build_agent_prompt(config, skills_dir=skills_dir)

    assert "## Skills (mandatory)" in prompt
    assert "  outbound:" in prompt
    assert "- outbound-alpha: Outbound alpha description." in prompt
    assert "- docs-only: Docs-only description." in prompt
    assert "OUTBOUND_ALPHA_FULL_BODY_SENTINEL" in prompt
    assert "OUTBOUND_BETA_FULL_BODY_SENTINEL" in prompt
    assert "DOCS_ONLY_FULL_BODY_SENTINEL" not in prompt


def test_named_pins_load_before_category_pins_and_category_duplicates_are_skipped(tmp_path):
    clear_skills_system_prompt_cache(clear_snapshot=True)
    skills_dir = tmp_path / "skills"
    _make_skill(
        skills_dir,
        "shared-skill",
        "Shared description.",
        "SHARED_FULL_BODY_SENTINEL",
        category="outbound",
    )
    _make_skill(
        skills_dir,
        "category-only",
        "Category-only description.",
        "CATEGORY_ONLY_FULL_BODY_SENTINEL",
        category="outbound",
    )
    config = {
        "agent": {"tool_use_enforcement": False},
        "skills": {
            "prompt_index": {
                "always_include_categories": ["outbound", "outbound"],
                "max_categories": 5,
                "max_skills_per_category": 3,
                "max_total_skills": 20,
                "always_include_skills": ["shared-skill"],
            }
        },
    }

    prompt = _build_agent_prompt(config, skills_dir=skills_dir)

    assert prompt.count("SHARED_FULL_BODY_SENTINEL") == 1
    assert prompt.count("CATEGORY_ONLY_FULL_BODY_SENTINEL") == 1
    assert prompt.index("SHARED_FULL_BODY_SENTINEL") < prompt.index(
        "CATEGORY_ONLY_FULL_BODY_SENTINEL"
    )


def test_category_pins_do_not_duplicate_cli_preloaded_skill_body(tmp_path):
    clear_skills_system_prompt_cache(clear_snapshot=True)
    skills_dir = tmp_path / "skills"
    _make_skill(
        skills_dir,
        "category-skill",
        "Category description.",
        "CATEGORY_PRELOADED_FULL_BODY_SENTINEL",
        category="outbound",
    )
    config = {
        "agent": {"tool_use_enforcement": False},
        "skills": {
            "prompt_index": {
                "always_include_categories": ["outbound"],
                "max_categories": 5,
                "max_skills_per_category": 3,
                "max_total_skills": 20,
            }
        },
    }
    preloaded_system_message = (
        '[SYSTEM: The user launched this CLI session with the "category-skill" skill '
        "preloaded. Treat its instructions as active guidance.]\n\n"
        "---\nname: category-skill\ndescription: Category description.\n---\n\n"
        "# category-skill\n\nCATEGORY_PRELOADED_FULL_BODY_SENTINEL"
    )

    prompt = _build_agent_prompt(
        config,
        system_message=preloaded_system_message,
        skills_dir=skills_dir,
    )

    assert prompt.count("CATEGORY_PRELOADED_FULL_BODY_SENTINEL") == 1


def test_missing_always_include_category_logs_warning_but_does_not_crash(tmp_path, caplog):
    clear_skills_system_prompt_cache(clear_snapshot=True)
    skills_dir = tmp_path / "skills"
    _make_skill(
        skills_dir,
        "present-skill",
        "Present description.",
        "PRESENT_CATEGORY_FULL_BODY_SENTINEL",
        category="outbound",
    )
    config = {
        "agent": {"tool_use_enforcement": False},
        "skills": {
            "prompt_index": {
                "always_include_categories": ["missing-category"],
                "max_categories": 5,
                "max_skills_per_category": 3,
                "max_total_skills": 20,
            }
        },
    }
    caplog.set_level(logging.WARNING, logger="agent.prompt_builder")

    prompt = _build_agent_prompt(config, skills_dir=skills_dir)

    assert "Conversation started:" in prompt
    assert "PRESENT_CATEGORY_FULL_BODY_SENTINEL" not in prompt
    assert any("missing-category" in record.message for record in caplog.records)


def test_always_include_category_caps_are_respected(tmp_path):
    clear_skills_system_prompt_cache(clear_snapshot=True)
    skills_dir = tmp_path / "skills"
    for category, names in {
        "outbound": ["a-alpha", "a-beta", "a-gamma"],
        "sales": ["b-alpha", "b-beta"],
        "hubspot": ["c-alpha"],
    }.items():
        for name in names:
            _make_skill(
                skills_dir,
                name,
                f"{name} description.",
                f"{name.upper().replace('-', '_')}_FULL_BODY_SENTINEL",
                category=category,
            )
    config = {
        "agent": {"tool_use_enforcement": False},
        "skills": {
            "prompt_index": {
                "always_include_categories": ["outbound", "sales", "hubspot"],
                "max_categories": 2,
                "max_skills_per_category": 2,
                "max_total_skills": 3,
            }
        },
    }

    prompt = _build_agent_prompt(config, skills_dir=skills_dir)

    assert "A_ALPHA_FULL_BODY_SENTINEL" in prompt
    assert "A_BETA_FULL_BODY_SENTINEL" in prompt
    assert "B_ALPHA_FULL_BODY_SENTINEL" in prompt
    assert "A_GAMMA_FULL_BODY_SENTINEL" not in prompt
    assert "B_BETA_FULL_BODY_SENTINEL" not in prompt
    assert "C_ALPHA_FULL_BODY_SENTINEL" not in prompt


def test_missing_named_pin_does_not_consume_category_total_cap(tmp_path):
    clear_skills_system_prompt_cache(clear_snapshot=True)
    skills_dir = tmp_path / "skills"
    _make_skill(
        skills_dir,
        "named-skill",
        "Named description.",
        "NAMED_FULL_BODY_SENTINEL",
    )
    _make_skill(
        skills_dir,
        "a-alpha",
        "Alpha description.",
        "A_ALPHA_FULL_BODY_SENTINEL",
        category="outbound",
    )
    _make_skill(
        skills_dir,
        "a-beta",
        "Beta description.",
        "A_BETA_FULL_BODY_SENTINEL",
        category="outbound",
    )
    config = {
        "agent": {"tool_use_enforcement": False},
        "skills": {
            "prompt_index": {
                "always_include_skills": ["named-skill", "missing-skill"],
                "always_include_categories": ["outbound"],
                "max_categories": 1,
                "max_skills_per_category": 3,
                "max_total_skills": 3,
            }
        },
    }

    prompt = _build_agent_prompt(config, skills_dir=skills_dir)

    assert "NAMED_FULL_BODY_SENTINEL" in prompt
    assert "A_ALPHA_FULL_BODY_SENTINEL" in prompt
    assert "A_BETA_FULL_BODY_SENTINEL" in prompt


def test_preloaded_category_skill_does_not_consume_category_total_cap(tmp_path):
    clear_skills_system_prompt_cache(clear_snapshot=True)
    skills_dir = tmp_path / "skills"
    _make_skill(
        skills_dir,
        "a-preloaded",
        "Preloaded description.",
        "A_PRELOADED_FULL_BODY_SENTINEL",
        category="outbound",
    )
    _make_skill(
        skills_dir,
        "b-beta",
        "Beta description.",
        "B_BETA_FULL_BODY_SENTINEL",
        category="outbound",
    )
    _make_skill(
        skills_dir,
        "c-gamma",
        "Gamma description.",
        "C_GAMMA_FULL_BODY_SENTINEL",
        category="outbound",
    )
    config = {
        "agent": {"tool_use_enforcement": False},
        "skills": {
            "prompt_index": {
                "always_include_categories": ["outbound"],
                "max_categories": 1,
                "max_skills_per_category": 3,
                "max_total_skills": 2,
            }
        },
    }
    preloaded_system_message = (
        '[SYSTEM: The user launched this CLI session with the "a-preloaded" skill '
        "preloaded. Treat its instructions as active guidance.]\n\n"
        "---\nname: a-preloaded\ndescription: Preloaded description.\n---\n\n"
        "# a-preloaded\n\nA_PRELOADED_FULL_BODY_SENTINEL"
    )

    prompt = _build_agent_prompt(
        config,
        system_message=preloaded_system_message,
        skills_dir=skills_dir,
    )

    assert prompt.count("A_PRELOADED_FULL_BODY_SENTINEL") == 1
    assert "B_BETA_FULL_BODY_SENTINEL" in prompt
    assert "C_GAMMA_FULL_BODY_SENTINEL" in prompt
