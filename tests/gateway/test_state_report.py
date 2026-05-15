"""Tests for read-only gateway state reports."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key
from gateway.state_report import build_state_report, render_state_report
from hermes_state import SessionDB


class ReadOnlyStore:
    def __init__(self, sessions_dir):
        self.sessions_dir = sessions_dir
        self.config = GatewayConfig()
        self._entries = {}
        self._loaded = False

    def get_or_create_session(self, *_args, **_kwargs):  # pragma: no cover
        raise AssertionError("state report must not create or update sessions")

    def _save(self):  # pragma: no cover
        raise AssertionError("state report must not write sessions.json")


@pytest.fixture
def session_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        yield db
    finally:
        db.close()


def _source(topic="- Example Child Topic", thread_id="example-child-topic"):
    return SessionSource(
        platform=Platform.ZULIP,
        chat_id="stream:12345",
        chat_name="example-stream",
        chat_type="channel",
        user_id="example-user",
        thread_id=thread_id,
        chat_topic=topic,
        parent_chat_id="12345",
    )


def _entry(source, session_id, *, last_prompt_tokens=42_000, minutes_ago=0):
    updated_at = datetime(2026, 5, 15, 12, 0, 0) - timedelta(minutes=minutes_ago)
    key = build_session_key(source)
    entry = SessionEntry(
        session_key=key,
        session_id=session_id,
        created_at=updated_at,
        updated_at=updated_at,
        origin=source,
        display_name=source.chat_name,
        platform=source.platform,
        chat_type=source.chat_type,
        last_prompt_tokens=last_prompt_tokens,
    )
    return key, entry.to_dict()


def _write_sessions(sessions_dir, entries):
    sessions_dir.mkdir(parents=True, exist_ok=True)
    sessions_file = sessions_dir / "sessions.json"
    sessions_file.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return sessions_file


def _set_times(db, rows):
    with db._lock:
        for session_id, started_at, ended_at, end_reason in rows:
            db._conn.execute(
                "UPDATE sessions SET started_at=?, ended_at=?, end_reason=? WHERE id=?",
                (started_at, ended_at, end_reason, session_id),
            )


def test_resolves_current_zulip_session_without_creating_session(tmp_path, session_db):
    source = _source()
    key, entry = _entry(source, "sess-current", last_prompt_tokens=42_000)
    sessions_file = _write_sessions(tmp_path / "sessions", {key: entry})
    store = ReadOnlyStore(tmp_path / "sessions")
    session_db.create_session("sess-current", source="zulip")
    session_db.update_token_counts(
        "sess-current",
        input_tokens=1000,
        output_tokens=250,
        cache_read_tokens=500,
        model="openrouter/test-model",
        api_call_count=3,
        absolute=True,
    )

    before_json = sessions_file.read_text(encoding="utf-8")
    before_count = session_db.session_count()
    report = build_state_report(source, store, session_db)
    rendered = render_state_report(report)

    assert report["current_lane"]["platform"] == "zulip"
    assert report["current_lane"]["session_key"] == key
    assert report["current_lane"]["mapped_session_id"] == "sess-current"
    assert report["current_lane"]["compression_tip_session_id"] == "sess-current"
    assert report["context_health"]["last_prompt_tokens"] == 42_000
    assert report["context_health"]["data_source"] == "persisted"
    assert report["usage"]["total_tokens"] == 1750
    assert "example-stream / - Example Child Topic" in rendered
    assert sessions_file.read_text(encoding="utf-8") == before_json
    assert session_db.session_count() == before_count


def test_projects_stale_sessions_json_mapping_to_compression_tip(tmp_path, session_db):
    source = _source()
    key, entry = _entry(source, "sess-root", last_prompt_tokens=75_000)
    _write_sessions(tmp_path / "sessions", {key: entry})
    store = ReadOnlyStore(tmp_path / "sessions")

    session_db.create_session("sess-root", source="zulip")
    session_db.create_session("sess-tip", source="zulip", parent_session_id="sess-root")
    _set_times(
        session_db,
        [
            ("sess-root", 100.0, 200.0, "compression"),
            ("sess-tip", 201.0, None, None),
        ],
    )

    report = build_state_report(source, store, session_db)
    rendered = render_state_report(report)

    assert report["current_lane"]["mapped_session_id"] == "sess-root"
    assert report["current_lane"]["compression_tip_session_id"] == "sess-tip"
    assert report["current_lane"]["stale_mapping"] is True
    assert report["compression_handover"]["compression_chain_depth"] == 1
    assert "Warning: sessions.json maps to an ended compression parent" in rendered


def test_missing_live_and_zero_persisted_context_is_marked_unknown_stale(
    tmp_path, session_db
):
    source = _source()
    key, entry = _entry(source, "sess-empty", last_prompt_tokens=0)
    _write_sessions(tmp_path / "sessions", {key: entry})
    store = ReadOnlyStore(tmp_path / "sessions")
    session_db.create_session("sess-empty", source="zulip")

    report = build_state_report(source, store, session_db, live_agent=None)
    rendered = render_state_report(report)

    assert report["context_health"]["data_source"] == "unknown"
    assert report["context_health"]["last_prompt_tokens"] is None
    assert report["context_health"]["context_percentage"] is None
    assert report["compression_handover"]["risk"] == "unknown/stale"
    assert "Last prompt tokens: unknown (unknown)" in rendered


def test_live_agent_context_and_compression_count_are_labelled_live(
    tmp_path, session_db
):
    source = _source()
    key, entry = _entry(source, "sess-live", last_prompt_tokens=10_000)
    _write_sessions(tmp_path / "sessions", {key: entry})
    store = ReadOnlyStore(tmp_path / "sessions")
    session_db.create_session("sess-live", source="zulip")
    live_agent = SimpleNamespace(
        session_id="sess-live",
        context_compressor=SimpleNamespace(
            last_prompt_tokens=80_000,
            context_length=200_000,
            compression_count=2,
        ),
    )

    report = build_state_report(source, store, session_db, live_agent=live_agent)

    assert report["current_lane"]["live_session_id"] == "sess-live"
    assert report["context_health"]["data_source"] == "live"
    assert report["context_health"]["context_percentage"] == 40
    assert report["compression_handover"]["live_compression_count"] == 2
    assert report["compression_handover"]["live_compression_count_source"] == "live"
    assert report["compression_handover"]["risk"] == "prepare handover"


def test_related_zulip_lanes_are_metadata_only_and_topic_family_filtered(
    tmp_path, session_db
):
    current = _source("- Example Child Topic", "example-child-topic")
    parent = _source("- Example Parent Topic", "example-parent-topic")
    unrelated = _source("Unrelated Planning Topic", "unrelated-topic")
    other_stream = SessionSource(
        platform=Platform.ZULIP,
        chat_id="stream:67890",
        chat_name="other-stream",
        chat_type="channel",
        thread_id="other-parent-topic",
        chat_topic="- Example Parent Topic",
        parent_chat_id="67890",
    )
    entries = {}
    for src, sid, tokens, ago in [
        (current, "sess-current", 40_000, 0),
        (parent, "sess-parent", 65_000, 5),
        (unrelated, "sess-unrelated", 65_000, 1),
        (other_stream, "sess-other-stream", 65_000, 2),
    ]:
        key, entry = _entry(src, sid, last_prompt_tokens=tokens, minutes_ago=ago)
        entries[key] = entry
        session_db.create_session(sid, source="zulip")
    _write_sessions(tmp_path / "sessions", entries)
    store = ReadOnlyStore(tmp_path / "sessions")

    report = build_state_report(current, store, session_db)

    topics = [item["topic"] for item in report["related_zulip_lanes"]]
    assert topics == ["- Example Parent Topic"]
    related = report["related_zulip_lanes"][0]
    assert related["session_id"] == "sess-parent"
    assert related["compression_tip_session_id"] == "sess-parent"
    assert related["risk"] == "keep parent lean"
    assert "Unrelated Planning Topic" not in render_state_report(report)


def test_report_does_not_include_message_content_or_system_prompt(tmp_path, session_db):
    source = _source()
    key, entry = _entry(source, "sess-secret", last_prompt_tokens=42_000)
    _write_sessions(tmp_path / "sessions", {key: entry})
    store = ReadOnlyStore(tmp_path / "sessions")
    session_db.create_session("sess-secret", source="zulip")
    session_db.append_message("sess-secret", "user", "TOP_SECRET_MESSAGE_CONTENT")
    session_db.update_system_prompt("sess-secret", "TOP_SECRET_SYSTEM_PROMPT")

    report = build_state_report(source, store, session_db)
    rendered = render_state_report(report)
    serialised = json.dumps(report, sort_keys=True)

    assert report["safety"]["message_content_included"] is False
    assert "TOP_SECRET_MESSAGE_CONTENT" not in rendered
    assert "TOP_SECRET_SYSTEM_PROMPT" not in rendered
    assert "TOP_SECRET_MESSAGE_CONTENT" not in serialised
    assert "TOP_SECRET_SYSTEM_PROMPT" not in serialised


def test_state_command_is_registered_and_gateway_handler_is_read_only(
    tmp_path, session_db
):
    from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS, resolve_command

    command = resolve_command("state-report")
    assert command is not None
    assert command.name == "state"
    assert command.gateway_only is True
    assert "state" in GATEWAY_KNOWN_COMMANDS
    assert "state-report" in GATEWAY_KNOWN_COMMANDS


@pytest.mark.asyncio
async def test_gateway_state_handler_renders_report_without_session_creation(
    tmp_path, session_db
):
    from gateway.run import GatewayRunner

    source = _source()
    key, entry = _entry(source, "sess-handler", last_prompt_tokens=42_000)
    sessions_file = _write_sessions(tmp_path / "sessions", {key: entry})
    store = ReadOnlyStore(tmp_path / "sessions")
    session_db.create_session("sess-handler", source="zulip")

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner.session_store = store
    runner._session_db = session_db
    runner._session_key_for_source = lambda _source: key
    event = MessageEvent(text="/state", source=source, message_id="m1")

    before_json = sessions_file.read_text(encoding="utf-8")
    before_count = session_db.session_count()
    rendered = await runner._handle_state_command(event)

    assert "State report" in rendered
    assert "No message content was read or included." in rendered
    assert sessions_file.read_text(encoding="utf-8") == before_json
    assert session_db.session_count() == before_count
