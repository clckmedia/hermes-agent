"""Tests for approaching-limit session warnings."""

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, SessionResetPolicy
from gateway.session import SessionEntry, SessionSource, SessionStore, build_session_key
from gateway.run import GatewayRunner


class FakeDB:
    def __init__(self, stats_by_session_id=None):
        self.stats_by_session_id = stats_by_session_id or {}

    def get_session(self, session_id: str):
        return self.stats_by_session_id.get(session_id)


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="channel-123",
        chat_type="channel",
        user_id="user-456",
        user_name="Damien",
    )


def _make_entry(source: SessionSource, session_id: str = "sess-live") -> SessionEntry:
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


def _make_store(tmp_path, policy: SessionResetPolicy, db: FakeDB) -> SessionStore:
    config = GatewayConfig(default_reset_policy=policy)
    store = SessionStore(sessions_dir=tmp_path, config=config)
    store._loaded = True
    store._db = db
    return store


class TestSessionStoreSizeWarnings:
    def test_warning_emitted_once_when_session_approaches_input_limit(self, tmp_path):
        source = _make_source()
        entry = _make_entry(source)
        db = FakeDB({"sess-live": {"input_tokens": 1_700_000, "message_count": 120}})
        store = _make_store(
            tmp_path,
            SessionResetPolicy(mode="idle", max_input_tokens=2_000_000, warning_threshold_fraction=0.8),
            db,
        )

        first = store.consume_session_size_warning(entry)
        second = store.consume_session_size_warning(entry)

        assert "approaching the auto-rotation limit" in first
        assert "1,700,000 / 2,000,000" in first
        assert second is None
        assert entry.size_warning_sent is True

    def test_warning_emitted_once_when_session_exceeds_input_limit(self, tmp_path):
        source = _make_source()
        entry = _make_entry(source)
        db = FakeDB({"sess-live": {"input_tokens": 2_100_000, "message_count": 120}})
        store = _make_store(
            tmp_path,
            SessionResetPolicy(mode="idle", max_input_tokens=2_000_000, warning_threshold_fraction=0.8),
            db,
        )

        first = store.consume_session_size_warning(entry)
        second = store.consume_session_size_warning(entry)

        assert "has exceeded the auto-rotation limit" in first
        assert "2,100,000 / 2,000,000" in first
        assert "next message" in first
        assert second is None
        assert entry.size_warning_sent is True

    def test_warning_flag_clears_when_session_drops_below_threshold(self, tmp_path):
        source = _make_source()
        entry = _make_entry(source)
        db = FakeDB({"sess-live": {"input_tokens": 1_700_000, "message_count": 120}})
        store = _make_store(
            tmp_path,
            SessionResetPolicy(mode="idle", max_input_tokens=2_000_000, warning_threshold_fraction=0.8),
            db,
        )

        assert store.consume_session_size_warning(entry) is not None
        db.stats_by_session_id["sess-live"] = {"input_tokens": 500_000, "message_count": 120}

        assert store.consume_session_size_warning(entry) is None
        assert entry.size_warning_sent is False

        db.stats_by_session_id["sess-live"] = {"input_tokens": 1_650_000, "message_count": 120}
        assert store.consume_session_size_warning(entry) is not None
        assert entry.size_warning_sent is True


@pytest.mark.asyncio
async def test_gateway_runner_sends_session_size_warning_once(tmp_path):
    source = _make_source()
    entry = _make_entry(source)
    db = FakeDB({"sess-live": {"input_tokens": 1_700_000, "message_count": 120}})
    store = _make_store(
        tmp_path,
        SessionResetPolicy(mode="idle", max_input_tokens=2_000_000, warning_threshold_fraction=0.8),
        db,
    )
    adapter = AsyncMock()

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.session_store = store
    runner.adapters = {Platform.DISCORD: adapter}

    await runner._maybe_send_session_size_warning(source, entry, metadata={"thread_id": "abc"})
    await runner._maybe_send_session_size_warning(source, entry, metadata={"thread_id": "abc"})

    assert adapter.send.await_count == 1
    sent_text = adapter.send.await_args.args[1]
    assert "approaching the auto-rotation limit" in sent_text
    assert "/resume" in sent_text
