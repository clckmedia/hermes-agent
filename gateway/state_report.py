"""Read-only gateway state report helpers.

The reporter deliberately reads only lightweight session metadata from
``sessions.json``/``SessionDB`` and optional live agent counters.  It must not
create sessions, append transcript messages, mutate config, or query message
content.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .config import Platform
from .session import SessionEntry, SessionSource, build_session_key

RISK_HEALTHY = "healthy"
RISK_KEEP_PARENT_LEAN = "keep parent lean"
RISK_PREPARE_HANDOVER = "prepare handover"
RISK_HANDOVER_BEFORE_MAJOR = "handover before new major cycle"
RISK_UNKNOWN_STALE = "unknown/stale"

STATIC_SKILL_SUGGESTIONS = [
    # Stable Hermes runtime skill; avoid environment-specific workflow names.
    "hermes-agent",
]


class _ReadOnlySessionDBView:
    """Minimal read-only SessionDB view for standalone/local report builds."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        uri = f"file:{self.db_path}?mode=ro"
        self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self._conn.close()

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_compression_tip(self, session_id: str) -> str | None:
        current = session_id
        for _ in range(100):
            row = self._conn.execute(
                "SELECT id FROM sessions "
                "WHERE parent_session_id = ? "
                "  AND started_at >= ("
                "      SELECT ended_at FROM sessions "
                "      WHERE id = ? AND end_reason = 'compression'"
                "  ) "
                "ORDER BY started_at DESC LIMIT 1",
                (current, current),
            ).fetchone()
            if row is None:
                return current
            current = row["id"]
        return current


def _platform_value(platform: Platform | str | None) -> str:
    if isinstance(platform, Platform):
        return platform.value
    return str(platform or "unknown")


def _coerce_source(source: SessionSource | Mapping[str, Any]) -> SessionSource:
    if isinstance(source, SessionSource):
        return source
    if isinstance(source, Mapping):
        return SessionSource.from_dict(dict(source))
    raise TypeError("source must be a SessionSource or mapping")


def _session_key_for_source(source: SessionSource, session_store: Any) -> str:
    config = getattr(session_store, "config", None)
    return build_session_key(
        source,
        group_sessions_per_user=getattr(config, "group_sessions_per_user", True),
        thread_sessions_per_user=getattr(config, "thread_sessions_per_user", False),
    )


def _entry_to_dict(entry: Any) -> dict[str, Any]:
    if isinstance(entry, SessionEntry):
        return entry.to_dict()
    if hasattr(entry, "to_dict"):
        data = entry.to_dict()
        return data if isinstance(data, dict) else {}
    if isinstance(entry, Mapping):
        return dict(entry)
    return {}


def _read_session_entries(
    session_store: Any = None,
    *,
    hermes_home: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Read session-key mappings without invoking SessionStore write paths."""

    if isinstance(session_store, Mapping):
        return {str(key): _entry_to_dict(value) for key, value in session_store.items()}

    loaded = bool(getattr(session_store, "_loaded", False))
    entries = getattr(session_store, "_entries", None)
    if loaded and isinstance(entries, Mapping):
        return {str(key): _entry_to_dict(value) for key, value in entries.items()}

    sessions_file: Path | None = None
    sessions_dir = getattr(session_store, "sessions_dir", None)
    if sessions_dir is not None:
        sessions_file = Path(sessions_dir) / "sessions.json"
    elif hermes_home is not None:
        sessions_file = Path(hermes_home) / "sessions" / "sessions.json"

    if sessions_file is None or not sessions_file.exists():
        return {}

    try:
        data = json.loads(sessions_file.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        str(key): dict(value)
        for key, value in data.items()
        if isinstance(value, Mapping)
    }


def _origin_from_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    origin = entry.get("origin")
    if isinstance(origin, Mapping):
        return dict(origin)
    return {}


def _zulip_stream_id(origin_or_source: Mapping[str, Any] | SessionSource) -> str | None:
    if isinstance(origin_or_source, SessionSource):
        parent = origin_or_source.parent_chat_id
        chat = origin_or_source.chat_id
    else:
        parent = origin_or_source.get("parent_chat_id")
        chat = origin_or_source.get("chat_id")
    raw = parent or chat
    if raw is None:
        return None
    text = str(raw)
    if text.startswith("stream:"):
        return text.split(":", 1)[1]
    return text


def _zulip_topic(origin_or_source: Mapping[str, Any] | SessionSource) -> str | None:
    if isinstance(origin_or_source, SessionSource):
        return origin_or_source.chat_topic or origin_or_source.thread_id
    topic = origin_or_source.get("chat_topic") or origin_or_source.get("thread_id")
    return str(topic) if topic is not None else None


def _normalise_topic_family(topic: str | None) -> str:
    if not topic:
        return ""
    text = topic.lower()
    text = re.sub(r"\bsession\s*\d+\b", " ", text)
    text = re.sub(r"\b(child|parent|controller|ctrl|p/c)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _topics_match(current_topic: str | None, candidate_topic: str | None) -> bool:
    current = _normalise_topic_family(current_topic)
    candidate = _normalise_topic_family(candidate_topic)
    if not current:
        return True
    if not candidate:
        return False
    return current == candidate or current in candidate or candidate in current


def _parse_updated_at(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).timestamp()
        except ValueError:
            return 0.0
    return 0.0


def _format_timestamp(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value)).isoformat(timespec="seconds")
        except Exception:
            return str(value)
    return str(value)


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _get_session_row(session_db: Any, session_id: str | None) -> dict[str, Any] | None:
    if not session_db or not session_id or not hasattr(session_db, "get_session"):
        return None
    try:
        row = session_db.get_session(session_id)
    except Exception:
        return None
    return dict(row) if isinstance(row, Mapping) else None


def _compression_tip(session_db: Any, session_id: str | None) -> str | None:
    if not session_db or not session_id or not hasattr(session_db, "get_compression_tip"):
        return None
    try:
        tip = session_db.get_compression_tip(session_id)
    except Exception:
        return None
    return str(tip) if tip else None


def _compression_chain_depth(session_db: Any, session_id: str | None) -> int | None:
    """Approximate durable compression count by walking DB parent lineage."""

    if not session_db or not session_id:
        return None
    depth = 0
    current = _get_session_row(session_db, session_id)
    for _ in range(100):
        if not current:
            return depth
        parent_id = current.get("parent_session_id")
        if not parent_id:
            return depth
        parent = _get_session_row(session_db, str(parent_id))
        if not parent:
            return depth
        parent_ended = parent.get("ended_at")
        current_started = current.get("started_at")
        if parent.get("end_reason") != "compression":
            return depth
        if parent_ended is None or current_started is None:
            return depth
        try:
            if float(current_started) < float(parent_ended):
                return depth
        except (TypeError, ValueError):
            return depth
        depth += 1
        current = parent
    return depth


def _usage_snapshot(session_row: Mapping[str, Any] | None) -> dict[str, Any]:
    if not session_row:
        return {"data_source": "unknown"}
    input_tokens = _safe_int(session_row.get("input_tokens")) or 0
    output_tokens = _safe_int(session_row.get("output_tokens")) or 0
    cache_read = _safe_int(session_row.get("cache_read_tokens")) or 0
    cache_write = _safe_int(session_row.get("cache_write_tokens")) or 0
    reasoning = _safe_int(session_row.get("reasoning_tokens")) or 0
    return {
        "data_source": "persisted",
        "session_id": session_row.get("id"),
        "model": session_row.get("model"),
        "started_at": _format_timestamp(session_row.get("started_at")),
        "ended_at": _format_timestamp(session_row.get("ended_at")),
        "end_reason": session_row.get("end_reason"),
        "message_count": _safe_int(session_row.get("message_count")) or 0,
        "tool_call_count": _safe_int(session_row.get("tool_call_count")) or 0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "reasoning_tokens": reasoning,
        "total_tokens": input_tokens + output_tokens + cache_read + cache_write,
        "api_call_count": _safe_int(session_row.get("api_call_count")) or 0,
        "cost_status": session_row.get("cost_status"),
    }


def _live_context(live_agent: Any) -> dict[str, Any]:
    if live_agent is None:
        return {}
    ctx = getattr(live_agent, "context_compressor", None)
    if ctx is None:
        return {}
    last_prompt = _safe_int(getattr(ctx, "last_prompt_tokens", None))
    context_length = _safe_int(getattr(ctx, "context_length", None))
    compression_count = _safe_int(getattr(ctx, "compression_count", None))
    result: dict[str, Any] = {}
    if last_prompt is not None:
        result["last_prompt_tokens"] = last_prompt
    if context_length is not None:
        result["context_length"] = context_length
    if compression_count is not None:
        result["compression_count"] = compression_count
    return result


def _context_health(
    entry: Mapping[str, Any] | None,
    live_agent: Any,
) -> dict[str, Any]:
    live = _live_context(live_agent)
    live_last = live.get("last_prompt_tokens")
    live_context_length = live.get("context_length")
    persisted_last = _safe_int(entry.get("last_prompt_tokens")) if entry else None

    if live_last is not None or live_context_length is not None:
        last_prompt = live_last if live_last is not None else persisted_last
        data_source = "live"
    else:
        last_prompt = persisted_last
        data_source = "persisted" if last_prompt else "unknown"

    context_length = live_context_length
    percentage = None
    percentage_source = "unknown"
    if last_prompt and context_length:
        percentage = min(100.0, (last_prompt / context_length) * 100)
        percentage_source = data_source

    return {
        "last_prompt_tokens": last_prompt if last_prompt else None,
        "last_prompt_tokens_source": data_source,
        "context_length": context_length,
        "context_length_source": "live" if context_length else "unknown",
        "context_percentage": percentage,
        "context_percentage_source": percentage_source,
        "data_source": data_source,
    }


def _risk_from_state(
    *,
    last_prompt_tokens: int | None,
    context_percentage: float | None,
    compression_count: int | None,
    chain_depth: int | None,
    mapped_session_id: str | None,
) -> tuple[str, str]:
    if not mapped_session_id:
        return RISK_UNKNOWN_STALE, "No current session mapping was found."

    risk = RISK_HEALTHY
    reason = "Context pressure looks low from available metadata."

    if context_percentage is not None:
        if context_percentage >= 85:
            risk = RISK_HANDOVER_BEFORE_MAJOR
            reason = "Context pressure is very high."
        elif context_percentage >= 70:
            risk = RISK_PREPARE_HANDOVER
            reason = "Context pressure is high."
        elif context_percentage >= 50:
            risk = RISK_KEEP_PARENT_LEAN
            reason = "Context pressure is moderate."
    elif last_prompt_tokens:
        if last_prompt_tokens >= 120_000:
            risk = RISK_HANDOVER_BEFORE_MAJOR
            reason = "Persisted prompt token count is very high."
        elif last_prompt_tokens >= 90_000:
            risk = RISK_PREPARE_HANDOVER
            reason = "Persisted prompt token count is high."
        elif last_prompt_tokens >= 50_000:
            risk = RISK_KEEP_PARENT_LEAN
            reason = "Persisted prompt token count is moderate."
    else:
        return RISK_UNKNOWN_STALE, "No live or persisted prompt token count is available."

    count = compression_count if compression_count is not None else chain_depth
    if count is not None:
        if count >= 3:
            return (
                RISK_HANDOVER_BEFORE_MAJOR,
                "Multiple compression hops are already present.",
            )
        if count >= 2 and risk in (RISK_HEALTHY, RISK_KEEP_PARENT_LEAN):
            return RISK_PREPARE_HANDOVER, "Compression depth is building."
        if count >= 1 and risk == RISK_HEALTHY:
            return RISK_KEEP_PARENT_LEAN, "A compression hop already exists."

    return risk, reason


def _recommendation_for_risk(risk: str) -> str:
    if risk == RISK_HEALTHY:
        return "Continue normally."
    if risk == RISK_KEEP_PARENT_LEAN:
        return "Keep the parent lane lean; use child lanes for heavy work."
    if risk == RISK_PREPARE_HANDOVER:
        return "Prepare a handover before adding broad new scope."
    if risk == RISK_HANDOVER_BEFORE_MAJOR:
        return "Handover before starting a new major cycle."
    return "Treat context as unknown or stale until refreshed."


def _related_zulip_lanes(
    *,
    entries: Mapping[str, Mapping[str, Any]],
    current_key: str,
    current_source: SessionSource,
    session_db: Any,
    limit: int = 5,
) -> list[dict[str, Any]]:
    if current_source.platform != Platform.ZULIP:
        return []

    current_stream = _zulip_stream_id(current_source)
    current_topic = _zulip_topic(current_source)
    related: list[dict[str, Any]] = []

    for key, entry in entries.items():
        if key == current_key:
            continue
        origin = _origin_from_entry(entry)
        if origin.get("platform") != Platform.ZULIP.value:
            continue
        if _zulip_stream_id(origin) != current_stream:
            continue
        topic = _zulip_topic(origin)
        if not _topics_match(current_topic, topic):
            continue

        mapped = entry.get("session_id")
        tip = _compression_tip(session_db, str(mapped)) if mapped else None
        tip_id = tip or (str(mapped) if mapped else None)
        chain_depth = _compression_chain_depth(session_db, tip_id)
        health = _context_health(entry, live_agent=None)
        risk, _reason = _risk_from_state(
            last_prompt_tokens=health.get("last_prompt_tokens"),
            context_percentage=health.get("context_percentage"),
            compression_count=None,
            chain_depth=chain_depth,
            mapped_session_id=str(mapped) if mapped else None,
        )
        related.append(
            {
                "topic": topic,
                "session_key": key,
                "session_id": str(mapped) if mapped else None,
                "compression_tip_session_id": tip_id,
                "updated_at": entry.get("updated_at"),
                "risk": risk,
                "risk_source": "rough estimate",
            }
        )

    related.sort(key=lambda item: _parse_updated_at(item.get("updated_at")), reverse=True)
    return related[:limit]


def _maybe_readonly_db(
    session_db: Any,
    hermes_home: str | Path | None,
) -> tuple[Any, bool]:
    if session_db is not None or hermes_home is None:
        return session_db, False
    db_path = Path(hermes_home) / "state.db"
    if not db_path.exists():
        return None, False
    try:
        return _ReadOnlySessionDBView(db_path), True
    except Exception:
        return None, False


def build_state_report(
    source: SessionSource | Mapping[str, Any],
    session_store: Any,
    session_db: Any,
    live_agent: Any = None,
    hermes_home: str | Path | None = None,
) -> dict[str, Any]:
    """Build a read-only report for the current gateway source.

    This function never calls ``get_or_create_session()``, never loads message
    transcripts, and never writes to ``sessions.json`` or ``state.db``.
    """

    source_obj = _coerce_source(source)
    entries = _read_session_entries(session_store, hermes_home=hermes_home)
    session_key = _session_key_for_source(source_obj, session_store)
    entry = entries.get(session_key)
    mapped_session_id = str(entry.get("session_id")) if entry and entry.get("session_id") else None

    session_db_view, owns_db = _maybe_readonly_db(session_db, hermes_home)
    try:
        tip_session_id = _compression_tip(session_db_view, mapped_session_id)
        if mapped_session_id and not tip_session_id:
            tip_session_id = mapped_session_id
        stale_mapping = bool(
            mapped_session_id and tip_session_id and mapped_session_id != tip_session_id
        )
        tip_row = _get_session_row(session_db_view, tip_session_id)
        mapped_row = _get_session_row(session_db_view, mapped_session_id)
        session_row = tip_row or mapped_row
        context = _context_health(entry, live_agent)
        live_context = _live_context(live_agent)
        live_compression_count = live_context.get("compression_count")
        chain_depth = _compression_chain_depth(session_db_view, tip_session_id)
        risk, reason = _risk_from_state(
            last_prompt_tokens=context.get("last_prompt_tokens"),
            context_percentage=context.get("context_percentage"),
            compression_count=live_compression_count,
            chain_depth=chain_depth,
            mapped_session_id=mapped_session_id,
        )

        return {
            "schema_version": 1,
            "mode": "read-only",
            "current_lane": {
                "platform": _platform_value(source_obj.platform),
                "zulip_stream_id": _zulip_stream_id(source_obj)
                if source_obj.platform == Platform.ZULIP
                else None,
                "zulip_stream_name": source_obj.chat_name
                if source_obj.platform == Platform.ZULIP
                else None,
                "zulip_topic": _zulip_topic(source_obj)
                if source_obj.platform == Platform.ZULIP
                else None,
                "session_key": session_key,
                "mapped_session_id": mapped_session_id,
                "live_session_id": getattr(live_agent, "session_id", None)
                if live_agent is not None
                else None,
                "compression_tip_session_id": tip_session_id,
                "stale_mapping": stale_mapping,
                "stale_mapping_warning": (
                    "sessions.json maps to an ended compression parent; using the projected tip"
                    if stale_mapping
                    else None
                ),
                "data_source": "persisted" if entry else "unknown",
            },
            "context_health": context,
            "compression_handover": {
                "live_compression_count": live_compression_count,
                "live_compression_count_source": "live"
                if live_compression_count is not None
                else "unknown",
                "compression_chain_depth": chain_depth,
                "compression_chain_depth_source": "persisted"
                if chain_depth is not None
                else "unknown",
                "risk": risk,
                "risk_source": "rough estimate",
                "reason": reason,
                "recommendation": _recommendation_for_risk(risk),
            },
            "usage": _usage_snapshot(session_row),
            "related_zulip_lanes": _related_zulip_lanes(
                entries=entries,
                current_key=session_key,
                current_source=source_obj,
                session_db=session_db_view,
            ),
            "skills": {
                "data_source": "static suggestion",
                "items": list(STATIC_SKILL_SUGGESTIONS),
            },
            "safety": {
                "message_content_included": False,
                "write_paths_used": False,
            },
        }
    finally:
        if owns_db and session_db_view is not None:
            session_db_view.close()


def _fmt_int(value: Any) -> str:
    number = _safe_int(value)
    return f"{number:,}" if number is not None else "unknown"


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "unknown"
    try:
        return f"{float(value):.0f}%"
    except (TypeError, ValueError):
        return "unknown"


def _fmt_id(value: Any) -> str:
    return f"`{value}`" if value else "unknown"


def render_state_report(report: Mapping[str, Any]) -> str:
    """Render a Zulip-friendly report without message content."""

    lane = report.get("current_lane") or {}
    context = report.get("context_health") or {}
    handover = report.get("compression_handover") or {}
    usage = report.get("usage") or {}
    related = report.get("related_zulip_lanes") or []
    skills = report.get("skills") or {}

    stream_topic = "unknown"
    if lane.get("platform") == "zulip":
        stream = lane.get("zulip_stream_name") or lane.get("zulip_stream_id") or "unknown"
        topic = lane.get("zulip_topic") or "unknown"
        stream_topic = f"{stream} / {topic}"

    lines = [
        "📍 **State report** _(read-only metadata)_",
        "",
        "**Current lane**",
        f"- Platform: {lane.get('platform', 'unknown')}",
        f"- Zulip stream/topic: {stream_topic}",
        f"- Session key: {_fmt_id(lane.get('session_key'))}",
        f"- Mapped session: {_fmt_id(lane.get('mapped_session_id'))}",
        f"- Compression tip: {_fmt_id(lane.get('compression_tip_session_id'))}",
    ]
    if lane.get("stale_mapping_warning"):
        lines.append(f"- Warning: {lane['stale_mapping_warning']}")

    lines.extend(
        [
            "",
            "**Context health**",
            "- Last prompt tokens: "
            f"{_fmt_int(context.get('last_prompt_tokens'))} "
            f"({context.get('last_prompt_tokens_source', 'unknown')})",
            "- Context length: "
            f"{_fmt_int(context.get('context_length'))} "
            f"({context.get('context_length_source', 'unknown')})",
            "- Context pressure: "
            f"{_fmt_pct(context.get('context_percentage'))} "
            f"({context.get('context_percentage_source', 'unknown')})",
            "",
            "**Compression / handover**",
            "- Live compression count: "
            f"{_fmt_int(handover.get('live_compression_count'))} "
            f"({handover.get('live_compression_count_source', 'unknown')})",
            "- Compression chain depth: "
            f"{_fmt_int(handover.get('compression_chain_depth'))} "
            f"({handover.get('compression_chain_depth_source', 'unknown')})",
            f"- Risk: {handover.get('risk', RISK_UNKNOWN_STALE)} "
            f"({handover.get('risk_source', 'unknown')})",
            f"- Recommendation: {handover.get('recommendation', _recommendation_for_risk(RISK_UNKNOWN_STALE))}",
            "",
            "**Usage**",
            f"- Data source: {usage.get('data_source', 'unknown')}",
            f"- Model: {usage.get('model') or 'unknown'}",
            f"- Tokens: {_fmt_int(usage.get('total_tokens'))} total "
            f"({_fmt_int(usage.get('input_tokens'))} in, "
            f"{_fmt_int(usage.get('output_tokens'))} out, "
            f"{_fmt_int(usage.get('cache_read_tokens'))} cache read)",
            f"- API calls: {_fmt_int(usage.get('api_call_count'))}",
            f"- Messages/tools: {_fmt_int(usage.get('message_count'))} / {_fmt_int(usage.get('tool_call_count'))}",
        ]
    )

    if related:
        lines.extend(["", "**Related Zulip lanes**"])
        for item in related:
            tip = item.get("compression_tip_session_id") or item.get("session_id")
            updated = item.get("updated_at") or "unknown"
            lines.append(
                "- "
                f"{item.get('topic') or 'unknown topic'} — "
                f"session {_fmt_id(item.get('session_id'))}; "
                f"tip {_fmt_id(tip)}; updated {updated}; "
                f"risk {item.get('risk', RISK_UNKNOWN_STALE)} "
                f"({item.get('risk_source', 'unknown')})"
            )
    else:
        lines.extend(["", "**Related Zulip lanes**", "- None found from session metadata."])

    static_items = skills.get("items") or []
    if static_items:
        lines.extend(
            [
                "",
                "**Skills**",
                "- Static suggestions only: " + ", ".join(f"`{item}`" for item in static_items),
            ]
        )

    lines.extend(
        [
            "",
            "No message content was read or included.",
        ]
    )
    return "\n".join(lines)
