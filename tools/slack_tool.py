"""Read-only Slack retrieval tools.

These tools use the Slack Web API with the configured bot token to inspect
visible conversations, history, and threads. They deliberately implement only
read endpoints: no posting, reactions, edits, joins, invites, deletes, or admin
methods.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterable, Optional

from agent.redact import redact_sensitive_text
from tools.registry import registry

logger = logging.getLogger(__name__)

SLACK_API_BASE = "https://slack.com/api"

_SLACK_READ_TOOL_NAMES = [
    "slack_list_conversations",
    "slack_get_history",
    "slack_get_thread",
    "slack_search_recent",
    "slack_get_permalink",
]

_DEFAULT_CONVERSATION_TYPES = "public_channel,private_channel,mpim,im"
_ALLOWED_CONVERSATION_TYPES = {
    "public_channel",
    "private_channel",
    "mpim",
    "im",
}

_MAX_LIST_LIMIT = 200
_MAX_HISTORY_LIMIT = 100
_MAX_THREAD_LIMIT = 100
_MAX_SEARCH_RESULT_LIMIT = 25
_MAX_SEARCH_CHANNELS = 50
_MAX_SEARCH_DAYS = 30
_MAX_MESSAGE_TEXT_CHARS = 1000
_MAX_TOPIC_CHARS = 200
_MAX_FILE_SUMMARIES = 5

_SLACK_CHANNEL_RE = re.compile(r"^[CGD][A-Z0-9]{8,}$")
_SLACK_TS_RE = re.compile(r"^\d{9,}(?:\.\d{1,6})?$")
_SLACK_TOKEN_RE = re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{8,}\b")
_URL_SECRET_QUERY_RE = re.compile(
    r"([?&](?:access_token|api[_-]?key|auth[_-]?token|token|signature|sig|pub_secret)=)([^&#\s<>\"']+)",
    re.IGNORECASE,
)
_GENERIC_SECRET_ASSIGN_RE = re.compile(
    r"\b(access_token|api[_-]?key|auth[_-]?token|signature|sig|pub_secret)\s*=\s*([^\s,;<>\"']+)",
    re.IGNORECASE,
)
_SLACK_PRIVATE_URL_RE = re.compile(
    r"https?://(?:files\.slack\.com|[A-Za-z0-9.-]+\.slack\.com/files-pri|[A-Za-z0-9.-]*slack-edge\.com)/[^\s<>\"')]+",
    re.IGNORECASE,
)


def _json(payload: dict[str, Any]) -> str:
    """Serialize a payload after recursively redacting known secrets."""
    return json.dumps(_sanitize_payload(payload), ensure_ascii=False)


def _sanitize_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _sanitize_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_payload(v) for v in value]
    if isinstance(value, str):
        return _sanitize_text(value)
    return value


def _sanitize_text(text: Any) -> str:
    """Redact tokens, secret query params, and Slack private file URLs."""
    if text is None:
        return ""
    cleaned = str(text)
    cleaned = redact_sensitive_text(cleaned)
    cleaned = _SLACK_TOKEN_RE.sub("[REDACTED_SLACK_TOKEN]", cleaned)
    cleaned = _SLACK_PRIVATE_URL_RE.sub("[REDACTED_SLACK_PRIVATE_URL]", cleaned)
    cleaned = _URL_SECRET_QUERY_RE.sub(lambda m: f"{m.group(1)}***", cleaned)
    cleaned = _GENERIC_SECRET_ASSIGN_RE.sub(lambda m: f"{m.group(1)}=***", cleaned)
    return cleaned


def _truncate(text: Any, max_chars: int) -> tuple[str, bool]:
    cleaned = _sanitize_text(text)
    if len(cleaned) <= max_chars:
        return cleaned, False
    if max_chars <= 1:
        return "…", True
    return cleaned[: max_chars - 1].rstrip() + "…", True


def _clamp_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def _get_bot_token() -> Optional[str]:
    """Resolve Slack bot token from the process environment."""
    return os.getenv("SLACK_BOT_TOKEN", "").strip() or None


def check_slack_read_requirements() -> bool:
    """Slack read tools are available only when a bot token is configured."""
    return bool(_get_bot_token())


def _validate_channel_id(channel_id: Any) -> Optional[str]:
    channel = str(channel_id or "").strip()
    if not channel:
        return None
    if not _SLACK_CHANNEL_RE.fullmatch(channel):
        return None
    return channel


def _validate_ts(value: Any) -> Optional[str]:
    ts = str(value or "").strip()
    if not ts:
        return None
    if not _SLACK_TS_RE.fullmatch(ts):
        return None
    return ts


def _normalise_ts_param(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    ts = str(value).strip()
    try:
        float(ts)
    except (TypeError, ValueError):
        return None
    return ts


def _normalise_conversation_types(value: Any) -> str:
    if value in (None, ""):
        return _DEFAULT_CONVERSATION_TYPES
    if isinstance(value, str):
        raw_types = [part.strip() for part in value.split(",")]
    elif isinstance(value, Iterable):
        raw_types = [str(part).strip() for part in value]
    else:
        return _DEFAULT_CONVERSATION_TYPES
    types = [part for part in raw_types if part in _ALLOWED_CONVERSATION_TYPES]
    return ",".join(types) or _DEFAULT_CONVERSATION_TYPES


def _normalise_channel_ids(value: Any, *, max_channels: int) -> tuple[list[str], Optional[str]]:
    if value in (None, ""):
        return [], None
    if isinstance(value, str):
        raw_ids = [part.strip() for part in value.split(",")]
    elif isinstance(value, Iterable):
        raw_ids = [str(part).strip() for part in value]
    else:
        return [], "channel_ids must be a list of Slack conversation IDs or a comma-separated string"

    channel_ids: list[str] = []
    for raw_id in raw_ids:
        if not raw_id:
            continue
        channel_id = _validate_channel_id(raw_id)
        if not channel_id:
            return [], f"Invalid Slack channel_id in channel_ids: {raw_id}"
        if channel_id not in channel_ids:
            channel_ids.append(channel_id)
        if len(channel_ids) >= max_channels:
            break
    return channel_ids, None


def _slack_api_get(
    token: str,
    method: str,
    params: Optional[dict[str, Any]] = None,
    timeout: int = 20,
) -> dict[str, Any]:
    """Call a Slack Web API GET endpoint.

    The token is sent only in the Authorization header and is never included in
    URLs, logs, or returned error payloads.
    """
    safe_params = {
        str(key): str(value)
        for key, value in (params or {}).items()
        if value not in (None, "")
    }
    query = urllib.parse.urlencode(safe_params)
    url = f"{SLACK_API_BASE}/{method}"
    if query:
        url = f"{url}?{query}"

    request = urllib.request.Request(
        url,
        method="GET",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "Hermes-Agent Slack read tools",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        retry_after = exc.headers.get("Retry-After") if exc.headers else None
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            data = {}
        if data.get("error"):
            reason = str(data["error"])
        elif exc.code == 429:
            reason = "rate_limited"
        else:
            reason = f"http_{exc.code}"
        result: dict[str, Any] = {"ok": False, "error": reason, "http_status": exc.code}
        if retry_after:
            result["retry_after"] = retry_after
        return result
    except urllib.error.URLError as exc:
        return {"ok": False, "error": "network_error", "detail": _sanitize_text(exc)}
    except Exception as exc:  # pragma: no cover - defensive transport guard
        return {"ok": False, "error": "request_failed", "detail": _sanitize_text(exc)}

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return {"ok": False, "error": "invalid_json_response"}
    if isinstance(data, dict):
        return data
    return {"ok": False, "error": "unexpected_response_type"}


def _slack_error_payload(method: str, response: dict[str, Any]) -> dict[str, Any]:
    reason = _sanitize_text(response.get("error") or "unknown")
    payload: dict[str, Any] = {
        "ok": False,
        "error": f"Slack API error: {reason}",
        "reason": reason,
        "method": method,
    }
    for key in ("needed", "provided", "retry_after", "http_status", "detail"):
        if response.get(key) not in (None, ""):
            payload[key] = response.get(key)
    return payload


def _file_summary(file_obj: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in ("id", "name", "title", "mimetype", "filetype", "created"):
        if file_obj.get(key) not in (None, ""):
            summary[key] = file_obj.get(key)
    return summary


def _message_summary(message: dict[str, Any], *, include_channel: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    text, truncated = _truncate(message.get("text", ""), _MAX_MESSAGE_TEXT_CHARS)
    files = message.get("files") or []
    reactions = message.get("reactions") or []
    result: dict[str, Any] = {
        "type": message.get("type", "message"),
        "ts": str(message.get("ts", "")),
        "user": message.get("user") or message.get("bot_id") or message.get("username"),
        "text": text,
        "text_truncated": truncated,
    }
    for key in ("subtype", "thread_ts", "reply_count", "reply_users_count"):
        if message.get(key) not in (None, ""):
            result[key] = message.get(key)
    if reactions:
        result["reaction_count"] = sum(int(reaction.get("count", 0) or 0) for reaction in reactions)
    if files:
        result["file_count"] = len(files)
        result["files"] = [_file_summary(f) for f in files[:_MAX_FILE_SUMMARIES] if isinstance(f, dict)]
    if include_channel:
        result["channel_id"] = include_channel.get("id")
        if include_channel.get("name"):
            result["channel_name"] = include_channel.get("name")
    return result


def _conversation_summary(channel: dict[str, Any]) -> dict[str, Any]:
    topic = channel.get("topic") if isinstance(channel.get("topic"), dict) else {}
    purpose = channel.get("purpose") if isinstance(channel.get("purpose"), dict) else {}
    topic_text, _ = _truncate(topic.get("value", ""), _MAX_TOPIC_CHARS)
    purpose_text, _ = _truncate(purpose.get("value", ""), _MAX_TOPIC_CHARS)
    return {
        "id": channel.get("id"),
        "name": channel.get("name") or channel.get("user") or channel.get("id"),
        "is_private": bool(channel.get("is_private", False)),
        "is_im": bool(channel.get("is_im", False)),
        "is_mpim": bool(channel.get("is_mpim", False)),
        "is_archived": bool(channel.get("is_archived", False)),
        "num_members": channel.get("num_members"),
        "topic": topic_text,
        "purpose": purpose_text,
    }


def _token_or_error() -> tuple[Optional[str], Optional[str]]:
    token = _get_bot_token()
    if not token:
        return None, "SLACK_BOT_TOKEN not configured."
    return token, None


def slack_list_conversations(args: dict[str, Any], **_kwargs: Any) -> str:
    """List Slack conversations visible to the bot."""
    token, error = _token_or_error()
    if error:
        return _json({"ok": False, "error": error})

    limit = _clamp_int(args.get("limit"), default=100, minimum=1, maximum=_MAX_LIST_LIMIT)
    params = {
        "types": _normalise_conversation_types(args.get("types")),
        "limit": str(limit),
        "exclude_archived": "false",
        "cursor": str(args.get("cursor") or "").strip(),
    }
    method = "conversations.list"
    response = _slack_api_get(token, method, params)
    if not response.get("ok"):
        return _json(_slack_error_payload(method, response))

    channels = [c for c in response.get("channels", []) if isinstance(c, dict)]
    conversations = [_conversation_summary(c) for c in channels[:limit]]
    metadata = response.get("response_metadata") if isinstance(response.get("response_metadata"), dict) else {}
    return _json(
        {
            "ok": True,
            "types": params["types"],
            "limit": limit,
            "count": len(conversations),
            "conversations": conversations,
            "next_cursor": metadata.get("next_cursor", ""),
        }
    )


def slack_get_history(args: dict[str, Any], **_kwargs: Any) -> str:
    """Fetch bounded Slack channel history visible to the bot."""
    token, error = _token_or_error()
    if error:
        return _json({"ok": False, "error": error})

    channel_id = _validate_channel_id(args.get("channel_id"))
    if not channel_id:
        return _json({"ok": False, "error": "Valid Slack channel_id is required."})

    limit = _clamp_int(args.get("limit"), default=20, minimum=1, maximum=_MAX_HISTORY_LIMIT)
    params = {
        "channel": channel_id,
        "limit": str(limit),
        "oldest": _normalise_ts_param(args.get("oldest")),
        "latest": _normalise_ts_param(args.get("latest")),
        "cursor": str(args.get("cursor") or "").strip(),
    }
    method = "conversations.history"
    response = _slack_api_get(token, method, params)
    if not response.get("ok"):
        return _json(_slack_error_payload(method, response))

    raw_messages = [m for m in response.get("messages", []) if isinstance(m, dict)]
    messages = [_message_summary(m) for m in raw_messages[:limit]]
    metadata = response.get("response_metadata") if isinstance(response.get("response_metadata"), dict) else {}
    return _json(
        {
            "ok": True,
            "channel_id": channel_id,
            "limit": limit,
            "count": len(messages),
            "has_more": bool(response.get("has_more", False)),
            "next_cursor": metadata.get("next_cursor", ""),
            "messages": messages,
        }
    )


def slack_get_thread(args: dict[str, Any], **_kwargs: Any) -> str:
    """Fetch bounded Slack thread replies visible to the bot."""
    token, error = _token_or_error()
    if error:
        return _json({"ok": False, "error": error})

    channel_id = _validate_channel_id(args.get("channel_id"))
    thread_ts = _validate_ts(args.get("thread_ts"))
    if not channel_id:
        return _json({"ok": False, "error": "Valid Slack channel_id is required."})
    if not thread_ts:
        return _json({"ok": False, "error": "Valid Slack thread_ts is required."})

    limit = _clamp_int(args.get("limit"), default=20, minimum=1, maximum=_MAX_THREAD_LIMIT)
    params = {
        "channel": channel_id,
        "ts": thread_ts,
        "limit": str(limit),
        "cursor": str(args.get("cursor") or "").strip(),
    }
    method = "conversations.replies"
    response = _slack_api_get(token, method, params)
    if not response.get("ok"):
        return _json(_slack_error_payload(method, response))

    raw_messages = [m for m in response.get("messages", []) if isinstance(m, dict)]
    messages = [_message_summary(m) for m in raw_messages[:limit]]
    metadata = response.get("response_metadata") if isinstance(response.get("response_metadata"), dict) else {}
    return _json(
        {
            "ok": True,
            "channel_id": channel_id,
            "thread_ts": thread_ts,
            "limit": limit,
            "count": len(messages),
            "has_more": bool(response.get("has_more", False)),
            "next_cursor": metadata.get("next_cursor", ""),
            "messages": messages,
        }
    )


def _candidate_channels(
    token: str,
    channel_ids: list[str],
    *,
    max_channels: int,
) -> tuple[list[dict[str, Any]], Optional[dict[str, Any]]]:
    if channel_ids:
        return ([{"id": channel_id, "name": ""} for channel_id in channel_ids[:max_channels]], None)

    method = "conversations.list"
    response = _slack_api_get(
        token,
        method,
        {
            "types": _DEFAULT_CONVERSATION_TYPES,
            "limit": str(max_channels),
            "exclude_archived": "true",
        },
    )
    if not response.get("ok"):
        return [], _slack_error_payload(method, response)
    channels = [c for c in response.get("channels", []) if isinstance(c, dict)]
    return channels[:max_channels], None


def slack_search_recent(args: dict[str, Any], **_kwargs: Any) -> str:
    """Search recent Slack history via bounded conversations.history scans."""
    token, error = _token_or_error()
    if error:
        return _json({"ok": False, "error": error})

    query = str(args.get("query") or "").strip()
    if not query:
        return _json({"ok": False, "error": "query is required."})

    days = _clamp_int(args.get("days"), default=7, minimum=1, maximum=_MAX_SEARCH_DAYS)
    max_channels = _clamp_int(args.get("max_channels"), default=20, minimum=1, maximum=_MAX_SEARCH_CHANNELS)
    result_limit = _clamp_int(args.get("limit"), default=10, minimum=1, maximum=_MAX_SEARCH_RESULT_LIMIT)
    channel_ids, channel_error = _normalise_channel_ids(args.get("channel_ids"), max_channels=max_channels)
    if channel_error:
        return _json({"ok": False, "error": channel_error})

    candidates, candidate_error = _candidate_channels(token, channel_ids, max_channels=max_channels)
    if candidate_error:
        return _json(candidate_error)

    oldest = str(time.time() - (days * 86400))
    query_lc = query.lower()
    results: list[dict[str, Any]] = []
    channel_errors: list[dict[str, Any]] = []
    scanned_messages = 0
    matched_messages = 0
    scanned_channels = 0

    for channel in candidates[:max_channels]:
        channel_id = str(channel.get("id") or "")
        if not _validate_channel_id(channel_id):
            continue
        scanned_channels += 1
        method = "conversations.history"
        response = _slack_api_get(
            token,
            method,
            {
                "channel": channel_id,
                "limit": str(_MAX_HISTORY_LIMIT),
                "oldest": oldest,
            },
        )
        if not response.get("ok"):
            shaped = _slack_error_payload(method, response)
            channel_errors.append(
                {
                    "channel_id": channel_id,
                    "channel_name": channel.get("name") or "",
                    "reason": shaped.get("reason"),
                    "error": shaped.get("error"),
                }
            )
            logger.info(
                "Slack recent search skipped channel %s due to API reason=%s",
                channel_id,
                shaped.get("reason"),
            )
            continue

        for message in [m for m in response.get("messages", []) if isinstance(m, dict)]:
            scanned_messages += 1
            text = str(message.get("text") or "")
            if query_lc not in text.lower():
                continue
            matched_messages += 1
            if len(results) < result_limit:
                results.append(_message_summary(message, include_channel=channel))

    return _json(
        {
            "ok": True,
            "query": query,
            "days": days,
            "oldest": oldest,
            "max_channels": max_channels,
            "limit": result_limit,
            "candidate_channels": len(candidates),
            "scanned_channels": scanned_channels,
            "scanned_messages": scanned_messages,
            "matched_count": matched_messages,
            "results": results,
            "channel_errors": channel_errors,
        }
    )


def slack_get_permalink(args: dict[str, Any], **_kwargs: Any) -> str:
    """Fetch a Slack permalink for a visible message."""
    token, error = _token_or_error()
    if error:
        return _json({"ok": False, "error": error})

    channel_id = _validate_channel_id(args.get("channel_id"))
    message_ts = _validate_ts(args.get("message_ts"))
    if not channel_id:
        return _json({"ok": False, "error": "Valid Slack channel_id is required."})
    if not message_ts:
        return _json({"ok": False, "error": "Valid Slack message_ts is required."})

    method = "chat.getPermalink"
    response = _slack_api_get(token, method, {"channel": channel_id, "message_ts": message_ts})
    if not response.get("ok"):
        return _json(_slack_error_payload(method, response))
    return _json(
        {
            "ok": True,
            "channel_id": channel_id,
            "message_ts": message_ts,
            "permalink": response.get("permalink", ""),
        }
    )


SLACK_LIST_CONVERSATIONS_SCHEMA = {
    "name": "slack_list_conversations",
    "description": (
        "Read-only: list Slack conversations visible to the configured bot token. "
        "Returns bounded metadata only; it does not join or modify channels."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "types": {
                "type": "string",
                "description": "Comma-separated Slack conversation types. Default: public_channel,private_channel,mpim,im.",
            },
            "limit": {"type": "integer", "description": "Max conversations to return (1-200)."},
            "cursor": {"type": "string", "description": "Slack pagination cursor from next_cursor."},
        },
        "required": [],
    },
}

SLACK_GET_HISTORY_SCHEMA = {
    "name": "slack_get_history",
    "description": (
        "Read-only: fetch bounded Slack conversation history for a channel the bot can see. "
        "Surfaces Slack visibility errors such as not_in_channel and missing_scope exactly."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "channel_id": {"type": "string", "description": "Slack conversation ID (C..., G..., or D...)."},
            "limit": {"type": "integer", "description": "Max messages to return (1-100)."},
            "oldest": {"type": "string", "description": "Optional oldest Slack timestamp/epoch boundary."},
            "latest": {"type": "string", "description": "Optional latest Slack timestamp/epoch boundary."},
            "cursor": {"type": "string", "description": "Slack pagination cursor from next_cursor."},
        },
        "required": ["channel_id"],
    },
}

SLACK_GET_THREAD_SCHEMA = {
    "name": "slack_get_thread",
    "description": "Read-only: fetch bounded Slack thread replies for a visible channel/thread.",
    "parameters": {
        "type": "object",
        "properties": {
            "channel_id": {"type": "string", "description": "Slack conversation ID (C..., G..., or D...)."},
            "thread_ts": {"type": "string", "description": "Slack thread parent timestamp."},
            "limit": {"type": "integer", "description": "Max messages to return (1-100)."},
            "cursor": {"type": "string", "description": "Slack pagination cursor from next_cursor."},
        },
        "required": ["channel_id", "thread_ts"],
    },
}

SLACK_SEARCH_RECENT_SCHEMA = {
    "name": "slack_search_recent",
    "description": (
        "Read-only: search recent Slack messages by scanning bounded conversations.history results. "
        "Does not call Slack search.messages. Use channel_ids to scope when possible."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Case-insensitive text to search for."},
            "channel_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of Slack conversation IDs. A comma-separated string also works.",
            },
            "days": {"type": "integer", "description": "Recent window in days (1-30). Default: 7."},
            "max_channels": {"type": "integer", "description": "Max visible channels to scan when channel_ids is omitted (1-50)."},
            "limit": {"type": "integer", "description": "Max matching messages to return (1-25)."},
        },
        "required": ["query"],
    },
}

SLACK_GET_PERMALINK_SCHEMA = {
    "name": "slack_get_permalink",
    "description": "Read-only: fetch Slack's permalink for a visible message.",
    "parameters": {
        "type": "object",
        "properties": {
            "channel_id": {"type": "string", "description": "Slack conversation ID (C..., G..., or D...)."},
            "message_ts": {"type": "string", "description": "Slack message timestamp."},
        },
        "required": ["channel_id", "message_ts"],
    },
}


registry.register(
    name="slack_list_conversations",
    toolset="slack",
    schema=SLACK_LIST_CONVERSATIONS_SCHEMA,
    handler=slack_list_conversations,
    check_fn=check_slack_read_requirements,
    requires_env=["SLACK_BOT_TOKEN"],
    emoji="🔎",
)

registry.register(
    name="slack_get_history",
    toolset="slack",
    schema=SLACK_GET_HISTORY_SCHEMA,
    handler=slack_get_history,
    check_fn=check_slack_read_requirements,
    requires_env=["SLACK_BOT_TOKEN"],
    emoji="🕘",
)

registry.register(
    name="slack_get_thread",
    toolset="slack",
    schema=SLACK_GET_THREAD_SCHEMA,
    handler=slack_get_thread,
    check_fn=check_slack_read_requirements,
    requires_env=["SLACK_BOT_TOKEN"],
    emoji="🧵",
)

registry.register(
    name="slack_search_recent",
    toolset="slack",
    schema=SLACK_SEARCH_RECENT_SCHEMA,
    handler=slack_search_recent,
    check_fn=check_slack_read_requirements,
    requires_env=["SLACK_BOT_TOKEN"],
    emoji="🔍",
)

registry.register(
    name="slack_get_permalink",
    toolset="slack",
    schema=SLACK_GET_PERMALINK_SCHEMA,
    handler=slack_get_permalink,
    check_fn=check_slack_read_requirements,
    requires_env=["SLACK_BOT_TOKEN"],
    emoji="🔗",
)
