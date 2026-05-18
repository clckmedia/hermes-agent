"""Tests for the post-fallback artefact claim verifier.

Regression target: long tool-heavy sessions that hit provider fallback and then
produce a confident final answer claiming local files, Drive uploads, row counts,
or verified links without current-turn tool evidence.  The guard appends a
warning footer instead of letting unsupported artefact claims stand alone.
"""

from __future__ import annotations

import json

from run_agent import (
    AIAgent,
    _artifact_evidence_tokens_from_text,
    _collect_artifact_evidence_tokens_from_messages,
    _extract_post_fallback_artifact_claims,
)


def _bare_agent() -> AIAgent:
    agent = object.__new__(AIAgent)
    agent._turn_fallback_activated = True
    agent._turn_artifact_evidence_tokens = set()
    agent._post_fallback_artifact_guard_enabled = lambda: True
    return agent


def test_extracts_local_paths_drive_links_and_filenames():
    text = """
    Created /home/dmelsing/arlo-work/report.csv and summary.md.
    Drive: https://drive.google.com/file/d/1UdAfoNbIJ2NnEmy3WK103Zk7Odnj82k3/view?usp=drivesdk
    """

    claims = _extract_post_fallback_artifact_claims(text)
    displays = [c["display"] for c in claims]

    assert "/home/dmelsing/arlo-work/report.csv" in displays
    assert "summary.md" in displays
    assert any("drive.google.com/file/d/1UdAfoNbIJ2NnEmy3WK103Zk7Odnj82k3" in d for d in displays)
    assert any("1UdAfoNbIJ2NnEmy3WK103Zk7Odnj82k3" in c["tokens"] for c in claims)


def test_collects_evidence_only_from_tool_calls_and_results_not_user_or_final_text():
    messages = [
        {"role": "user", "content": "Please create /tmp/requested.csv"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "write_file",
                        "arguments": json.dumps({"path": "/tmp/created.csv", "content": "x"}),
                    }
                }
            ],
        },
        {"role": "tool", "name": "write_file", "content": '{"bytes_written": 1}'},
        {"role": "assistant", "content": "Created /tmp/final-only.csv"},
    ]

    tokens = _collect_artifact_evidence_tokens_from_messages(messages, current_turn_user_idx=0)

    assert "/tmp/created.csv" in tokens
    assert "created.csv" in tokens
    assert "/tmp/requested.csv" not in tokens
    assert "/tmp/final-only.csv" not in tokens


def test_footer_warns_when_post_fallback_claim_has_no_tool_evidence():
    agent = _bare_agent()
    final = """
    Outcome:
    - Created local artefact: /home/dmelsing/arlo-work/missing.csv
    - Uploaded to Drive: https://drive.google.com/file/d/1FakeDriveFileIdABCDEFGHijk/view?usp=drivesdk
    """
    messages = [
        {"role": "user", "content": "Make the report"},
        {"role": "tool", "name": "terminal", "content": "portalId 435014"},
        {"role": "assistant", "content": final},
    ]

    footer = agent._format_post_fallback_artifact_guard_footer(final, messages, current_turn_user_idx=0)

    assert "Post-fallback artefact verifier" in footer
    assert "/home/dmelsing/arlo-work/missing.csv" in footer
    assert "1FakeDriveFileIdABCDEFGHijk" in footer


def test_no_footer_when_not_post_fallback():
    agent = _bare_agent()
    agent._turn_fallback_activated = False
    final = "Created /tmp/report.csv"

    footer = agent._format_post_fallback_artifact_guard_footer(final, [], current_turn_user_idx=0)

    assert footer == ""


def test_no_footer_when_tool_evidence_contains_claimed_path_and_drive_id():
    agent = _bare_agent()
    final = """
    Created /tmp/report.csv.
    Uploaded: https://drive.google.com/file/d/1UdAfoNbIJ2NnEmy3WK103Zk7Odnj82k3/view?usp=drivesdk
    """
    agent._record_artifact_evidence(
        "execute_code",
        {"code": "write csv and upload"},
        "wrote /tmp/report.csv; id: 1UdAfoNbIJ2NnEmy3WK103Zk7Odnj82k3",
    )

    footer = agent._format_post_fallback_artifact_guard_footer(final, [], current_turn_user_idx=0)

    assert footer == ""


def test_disclaimed_unverified_claim_does_not_get_duplicate_footer():
    agent = _bare_agent()
    final = "I cannot verify /tmp/report.csv; treat it as unverified."

    footer = agent._format_post_fallback_artifact_guard_footer(final, [], current_turn_user_idx=0)

    assert footer == ""


def test_evidence_token_extractor_reads_drive_id_fields():
    tokens = _artifact_evidence_tokens_from_text('{"id": "1UdAfoNbIJ2NnEmy3WK103Zk7Odnj82k3"}')

    assert "1uda fonbij2nnemy3wk103zk7odnj82k3".replace(" ", "") in tokens
