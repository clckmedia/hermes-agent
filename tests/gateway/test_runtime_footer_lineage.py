"""Tests for gateway runtime footer session-lineage metadata."""

from __future__ import annotations

from gateway.run import _compression_chain_depth_from_session_db


class FakeSessionDB:
    def __init__(self, rows):
        self.rows = rows
        self.requested_ids = []

    def get_session(self, session_id):
        self.requested_ids.append(session_id)
        return self.rows.get(session_id)


def test_compression_chain_depth_counts_contiguous_compression_ancestors_only():
    db = FakeSessionDB(
        {
            "tip": {"id": "tip", "parent_session_id": "parent-2", "end_reason": None},
            "parent-2": {
                "id": "parent-2",
                "parent_session_id": "parent-1",
                "end_reason": "compression",
            },
            "parent-1": {
                "id": "parent-1",
                "parent_session_id": None,
                "end_reason": "compression",
            },
        }
    )

    assert _compression_chain_depth_from_session_db(db, "tip") == 2


def test_compression_chain_depth_omits_fresh_or_non_compression_parent():
    fresh = FakeSessionDB({"tip": {"id": "tip", "parent_session_id": None}})
    branched = FakeSessionDB(
        {
            "tip": {"id": "tip", "parent_session_id": "branch-parent"},
            "branch-parent": {
                "id": "branch-parent",
                "parent_session_id": None,
                "end_reason": "branch",
            },
        }
    )

    assert _compression_chain_depth_from_session_db(fresh, "tip") == 0
    assert _compression_chain_depth_from_session_db(branched, "tip") == 0


def test_compression_chain_depth_handles_missing_db_or_cycles():
    assert _compression_chain_depth_from_session_db(None, "tip") == 0
    assert _compression_chain_depth_from_session_db(FakeSessionDB({}), "tip") == 0

    loopy = FakeSessionDB(
        {
            "tip": {"id": "tip", "parent_session_id": "parent", "end_reason": None},
            "parent": {
                "id": "parent",
                "parent_session_id": "tip",
                "end_reason": "compression",
            },
        }
    )

    assert _compression_chain_depth_from_session_db(loopy, "tip") == 1
