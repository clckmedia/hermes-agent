"""Tests for oversized-session auto-rotation in SessionStore.

These guard the gateway-level safety valve that rotates a session before it
turns into a multi-million-token monster. The decision uses persisted session
stats from SQLite so it still works even when the in-memory SessionEntry has
stale counters.
"""

from datetime import datetime, timedelta
from unittest.mock import patch

from gateway.config import GatewayConfig, Platform, SessionResetPolicy
from gateway.session import SessionEntry, SessionSource, SessionStore, build_session_key


class FakeDB:
    def __init__(self, stats_by_session_id=None):
        self.stats_by_session_id = stats_by_session_id or {}
        self.ended = []
        self.created = []

    def get_session(self, session_id: str):
        return self.stats_by_session_id.get(session_id)

    def end_session(self, session_id: str, reason: str):
        self.ended.append((session_id, reason))

    def create_session(self, **kwargs):
        self.created.append(kwargs)


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="channel-123",
        chat_type="channel",
        user_id="user-456",
        user_name="Damien",
    )


def _make_store(tmp_path, policy: SessionResetPolicy, db: FakeDB) -> SessionStore:
    config = GatewayConfig(default_reset_policy=policy)
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=tmp_path, config=config)
    store._loaded = True
    store._db = db
    return store


def _make_entry(source: SessionSource, session_id: str = "sess-old") -> SessionEntry:
    now = datetime.now()
    return SessionEntry(
        session_key=build_session_key(source),
        session_id=session_id,
        created_at=now,
        updated_at=now,
        origin=source,
        platform=source.platform,
        chat_type=source.chat_type,
    )


class TestOversizedSessionReset:
    def test_get_or_create_session_resets_when_cumulative_input_tokens_exceed_limit(self, tmp_path):
        source = _make_source()
        old_entry = _make_entry(source, session_id="sess-over-budget")
        db = FakeDB(
            {
                "sess-over-budget": {
                    "id": "sess-over-budget",
                    "input_tokens": 2_500,
                    "message_count": 12,
                }
            }
        )
        store = _make_store(
            tmp_path,
            SessionResetPolicy(mode="idle", idle_minutes=4320, max_input_tokens=2_000),
            db,
        )
        store._entries[old_entry.session_key] = old_entry

        new_entry = store.get_or_create_session(source)

        assert new_entry.session_id != "sess-over-budget"
        assert new_entry.was_auto_reset is True
        assert new_entry.auto_reset_reason == "oversized"
        assert new_entry.reset_had_activity is True
        assert db.ended == [("sess-over-budget", "session_reset")]
        assert db.created[0]["session_id"] == new_entry.session_id

    def test_get_or_create_session_resets_when_message_count_exceeds_limit(self, tmp_path):
        source = _make_source()
        old_entry = _make_entry(source, session_id="sess-too-many-messages")
        db = FakeDB(
            {
                "sess-too-many-messages": {
                    "id": "sess-too-many-messages",
                    "input_tokens": 50,
                    "message_count": 301,
                }
            }
        )
        store = _make_store(
            tmp_path,
            SessionResetPolicy(mode="idle", idle_minutes=4320, max_message_count=300),
            db,
        )
        store._entries[old_entry.session_key] = old_entry

        new_entry = store.get_or_create_session(source)

        assert new_entry.session_id != "sess-too-many-messages"
        assert new_entry.auto_reset_reason == "oversized"

    def test_get_or_create_session_notifies_idle_reset_when_db_has_activity(self, tmp_path):
        source = _make_source()
        old_entry = _make_entry(source, session_id="sess-idle-active")
        old_entry.updated_at = datetime.now() - timedelta(minutes=181)
        old_entry.total_tokens = 0  # modern token stats live in SQLite, not this legacy counter
        db = FakeDB(
            {
                "sess-idle-active": {
                    "id": "sess-idle-active",
                    "input_tokens": 50_000,
                    "message_count": 12,
                }
            }
        )
        store = _make_store(
            tmp_path,
            SessionResetPolicy(mode="idle", idle_minutes=180, max_input_tokens=2_000_000),
            db,
        )
        store._entries[old_entry.session_key] = old_entry

        new_entry = store.get_or_create_session(source)

        assert new_entry.session_id != "sess-idle-active"
        assert new_entry.was_auto_reset is True
        assert new_entry.auto_reset_reason == "idle"
        assert new_entry.reset_had_activity is True
        assert db.ended == [("sess-idle-active", "session_reset")]

    def test_get_or_create_session_keeps_session_when_below_oversized_limits(self, tmp_path):
        source = _make_source()
        old_entry = _make_entry(source, session_id="sess-healthy")
        db = FakeDB(
            {
                "sess-healthy": {
                    "id": "sess-healthy",
                    "input_tokens": 1_999,
                    "message_count": 300,
                }
            }
        )
        store = _make_store(
            tmp_path,
            SessionResetPolicy(
                mode="idle",
                idle_minutes=4320,
                max_input_tokens=2_000,
                max_message_count=301,
            ),
            db,
        )
        store._entries[old_entry.session_key] = old_entry

        current_entry = store.get_or_create_session(source)

        assert current_entry.session_id == "sess-healthy"
        assert current_entry.was_auto_reset is False
        assert db.ended == []
        assert db.created == []
