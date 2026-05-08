"""Zulip platform adapter using the Zulip Events API long-polling path.

This adapter deliberately does not expose a public webhook receiver. It logs in
with the bot identity, registers an Events API queue, and long-polls /events.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import unquote, urlsplit

try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by requirement check only
    aiohttp = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    resolve_channel_prompt,
)

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 10_000
MAX_INLINE_UPLOAD_BYTES = 120_000
UPLOAD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]*?/user_uploads/[^)\s]+)\)|(?<!\()((?:https?://[^\s)>]+)?/user_uploads/[^\s)>]+)")
TEXT_UPLOAD_SUFFIXES = {
    ".txt",
    ".md",
    ".markdown",
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".xml",
    ".html",
    ".htm",
    ".log",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".css",
    ".scss",
    ".sql",
}


def check_zulip_requirements() -> bool:
    """Check if Zulip adapter dependencies are available."""
    return AIOHTTP_AVAILABLE


class ZulipAdapter(BasePlatformAdapter):
    """Zulip bot adapter backed by /register + /events long polling."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.ZULIP)
        extra = config.extra or {}
        self.site: str = str(extra.get("site") or "").rstrip("/")
        self.bot_email: str = str(extra.get("bot_email") or "")
        self.api_key: str = str(config.api_key or extra.get("bot_api_key") or "")
        self.dm_only: bool = self._coerce_bool(extra.get("dm_only"), True)
        self.allowed_streams = {str(v) for v in (extra.get("allowed_streams") or []) if str(v)}
        self.all_public_streams: bool = self._coerce_bool(extra.get("all_public_streams"), False)
        self._session: Optional[aiohttp.ClientSession] = None if aiohttp else None
        self._poll_task: Optional[asyncio.Task] = None
        self._queue_id: Optional[str] = None
        self._last_event_id: int = -1
        self._stop_event = asyncio.Event()
        # Zulip's typing API requires user IDs for direct messages, while
        # Hermes chat IDs use stable email-based dm:<email> values for routing.
        # Cache IDs observed on inbound messages so the working indicator can
        # be sent during that user's turn without changing chat/session IDs.
        self._dm_user_ids: Dict[str, int] = {}
        self._active_typing_payloads: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _coerce_bool(value: Any, default: bool = True) -> bool:
        if value is None:
            return default
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "1", "yes", "on"):
                return True
            if lowered in ("false", "0", "no", "off"):
                return False
            return default
        return bool(value)

    def _auth(self) -> aiohttp.BasicAuth:
        return aiohttp.BasicAuth(self.bot_email, self.api_key)

    def _register_payload(self) -> Dict[str, str]:
        """Build Zulip Events API registration payload."""
        payload = {
            "event_types": json.dumps(["message"]),
            "apply_markdown": "false",
            "client_gravatar": "false",
        }
        if self.all_public_streams and not self.dm_only:
            payload["all_public_streams"] = "true"
        return payload

    async def connect(self) -> bool:
        if not self.site or not self.bot_email or not self.api_key:
            logger.warning("[Zulip] Missing ZULIP_SITE, ZULIP_BOT_EMAIL, or ZULIP_BOT_API_KEY")
            return False
        if not AIOHTTP_AVAILABLE:
            logger.warning("[Zulip] aiohttp is not installed")
            return False

        timeout = aiohttp.ClientTimeout(total=None, connect=20, sock_read=120)
        self._session = aiohttp.ClientSession(timeout=timeout, auth=self._auth())
        try:
            registered = await self._api_post(
                "/register",
                self._register_payload(),
            )
            self._queue_id = str(registered["queue_id"])
            self._last_event_id = int(registered.get("last_event_id", -1))
        except Exception as e:
            logger.warning("[Zulip] Failed to register event queue: %s", e)
            await self.disconnect()
            return False

        self._stop_event.clear()
        self._poll_task = asyncio.create_task(self._poll_events())
        self._mark_connected()
        logger.info("[Zulip] Connected to %s as %s", self.site, self.bot_email)
        return True

    async def disconnect(self) -> None:
        self._stop_event.set()
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        if self._session:
            await self._session.close()
            self._session = None
        self._mark_disconnected()

    async def _poll_events(self) -> None:
        backoff = 1.0
        while not self._stop_event.is_set():
            try:
                events = await self._api_get(
                    "/events",
                    {
                        "queue_id": self._queue_id or "",
                        "last_event_id": str(self._last_event_id),
                    },
                )
                backoff = 1.0
                for event in events.get("events", []):
                    event_id = event.get("id")
                    if isinstance(event_id, int):
                        self._last_event_id = max(self._last_event_id, event_id)
                    if event.get("type") != "message":
                        continue
                    message_event = self._message_event_from_zulip(event.get("message") or {})
                    if message_event is not None:
                        await self._materialize_uploads(message_event)
                        await self.handle_message(message_event)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                delay = min(backoff, 60.0) + random.uniform(0, 0.5)
                logger.warning("[Zulip] Event poll failed; retrying in %.1fs: %s", delay, e)
                await asyncio.sleep(delay)
                backoff = min(backoff * 2, 60.0)

    def _message_event_from_zulip(self, message: Dict[str, Any]) -> Optional[MessageEvent]:
        """Convert a Zulip message object into Hermes' normalized MessageEvent."""
        sender_email = str(message.get("sender_email") or "")
        if sender_email and sender_email.lower() == self.bot_email.lower():
            return None

        message_type = str(message.get("type") or "")
        if message_type == "private":
            chat_id = self._dm_chat_id(message)
            chat_type = "dm"
            chat_name = None
            thread_id = None
            chat_topic = None
            parent_chat_id = None
            self._cache_dm_user_id(chat_id, message)
        elif message_type == "stream":
            if self.dm_only:
                return None
            stream_id = message.get("stream_id")
            stream_name = str(message.get("display_recipient") or stream_id or "")
            if self.allowed_streams and stream_name not in self.allowed_streams and str(stream_id) not in self.allowed_streams:
                return None
            chat_id = f"stream:{stream_id}" if stream_id is not None else f"stream:{stream_name}"
            chat_type = "channel"
            chat_name = stream_name
            thread_id = str(message.get("subject") or message.get("topic") or "") or None
            chat_topic = thread_id
            parent_chat_id = str(stream_id) if stream_id is not None else None
        else:
            return None

        text = str(message.get("content") or "")
        channel_prompt = resolve_channel_prompt(
            self.config.extra or {},
            chat_id,
            parent_chat_id,
        )
        return MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=self.build_source(
                chat_id=chat_id,
                chat_name=chat_name,
                chat_type=chat_type,
                user_id=sender_email or None,
                user_name=message.get("sender_full_name") or sender_email or None,
                thread_id=thread_id,
                chat_topic=chat_topic,
                parent_chat_id=parent_chat_id,
                message_id=str(message.get("id")) if message.get("id") is not None else None,
            ),
            raw_message=message,
            message_id=str(message.get("id")) if message.get("id") is not None else None,
            channel_prompt=channel_prompt,
        )

    async def _materialize_uploads(self, event: MessageEvent) -> None:
        """Download Zulip /user_uploads links and inject safe local context.

        Zulip can replace a long pasted prompt with a markdown link such as
        ``[PastedText.txt](/user_uploads/...)``. The model cannot read that
        virtual Zulip path directly; if we leave it untouched it may search for
        a same-named temp file and pick up stale content. Downloading the exact
        upload URL here makes the turn deterministic and keeps the attachment
        tied to the inbound message ID.
        """
        uploads = self._extract_upload_links(event.text)
        if not uploads:
            return

        injected_blocks = []
        for upload in uploads:
            filename = upload["filename"]
            upload_path = upload["path"]
            try:
                data, content_type = await self._download_upload_bytes(upload_path)
                local_path = self._write_upload_file(event, upload_path, filename, data)
                event.media_urls.append(str(local_path))
                event.media_types.append(content_type or "application/octet-stream")
                injected_blocks.append(self._upload_context_block(filename, local_path, data, content_type))
            except Exception as exc:
                logger.warning("[Zulip] Failed to download upload %s: %s", upload_path, exc)
                injected_blocks.append(
                    "[Zulip attachment unavailable: "
                    f"{filename}]\n"
                    "I could not download this Zulip upload from the exact message link. "
                    "Do not search /tmp or reuse a same-named local file; ask the user to resend the attachment."
                )

        if injected_blocks:
            event.text = f"{event.text}\n\n" + "\n\n".join(injected_blocks)

    def _extract_upload_links(self, text: str) -> list[Dict[str, str]]:
        uploads: list[Dict[str, str]] = []
        seen: set[str] = set()
        site_host = urlsplit(self.site).netloc.lower()
        for match in UPLOAD_LINK_RE.finditer(text or ""):
            label = (match.group(1) or "").strip()
            href = (match.group(2) or match.group(3) or "").strip()
            parsed = urlsplit(href)
            if parsed.scheme:
                if site_host and parsed.netloc.lower() != site_host:
                    continue
                upload_path = parsed.path
                if parsed.query:
                    upload_path = f"{upload_path}?{parsed.query}"
            else:
                upload_path = href
            path_only = urlsplit(upload_path).path
            if not path_only.startswith("/user_uploads/") or upload_path in seen:
                continue
            seen.add(upload_path)
            filename = label or Path(unquote(path_only)).name or "zulip-upload"
            uploads.append({"filename": self._safe_upload_filename(filename), "path": upload_path})
        return uploads

    async def _download_upload_bytes(self, upload_path: str) -> tuple[bytes, str]:
        if not self.site:
            raise RuntimeError("Zulip site is not configured")
        owns_session = self._session is None
        session = self._session
        if session is None:
            timeout = aiohttp.ClientTimeout(total=60)
            session = aiohttp.ClientSession(timeout=timeout, auth=self._auth())
        try:
            url = f"{self.site}{upload_path}"
            async with session.get(url) as resp:
                data = await resp.read()
                if resp.status >= 400:
                    raise RuntimeError(f"HTTP {resp.status}")
                return data, resp.headers.get("Content-Type", "")
        finally:
            if owns_session:
                await session.close()

    def _write_upload_file(self, event: MessageEvent, upload_path: str, filename: str, data: bytes) -> Path:
        message_id = event.message_id or str((event.raw_message or {}).get("id") or "unknown")
        digest = hashlib.sha256(f"{message_id}:{upload_path}".encode("utf-8")).hexdigest()[:12]
        directory = Path(tempfile.gettempdir()) / "hermes-zulip-uploads" / message_id / digest
        directory.mkdir(parents=True, exist_ok=True)
        local_path = directory / self._safe_upload_filename(filename)
        local_path.write_bytes(data)
        return local_path

    @staticmethod
    def _safe_upload_filename(filename: str) -> str:
        candidate = Path(unquote(filename or "zulip-upload")).name.strip() or "zulip-upload"
        safe = re.sub(r"[^A-Za-z0-9._ -]", "_", candidate).strip(". ")
        return safe or "zulip-upload"

    def _upload_context_block(self, filename: str, local_path: Path, data: bytes, content_type: str) -> str:
        header = f"[Zulip attachment downloaded: {filename}]\nLocal path: {local_path}"
        if not self._looks_like_text_upload(filename, content_type, data):
            return f"{header}\nAttachment is binary or too large to inline; inspect the local path if needed."
        text = data.decode("utf-8", errors="replace")
        return (
            f"{header}\n"
            f"--- Begin attached file: {filename} ---\n"
            f"{text}\n"
            f"--- End attached file: {filename} ---"
        )

    @staticmethod
    def _looks_like_text_upload(filename: str, content_type: str, data: bytes) -> bool:
        if len(data) > MAX_INLINE_UPLOAD_BYTES or b"\x00" in data[:4096]:
            return False
        lowered_type = (content_type or "").lower()
        if lowered_type.startswith("text/"):
            return True
        suffix = Path(filename).suffix.lower()
        return suffix in TEXT_UPLOAD_SUFFIXES

    def _dm_chat_id(self, message: Dict[str, Any]) -> str:
        sender_email = str(message.get("sender_email") or "")
        recipients = message.get("display_recipient") or []
        if isinstance(recipients, list):
            for recipient in recipients:
                email = str((recipient or {}).get("email") or "")
                if email and email.lower() != self.bot_email.lower():
                    return f"dm:{email}"
        return f"dm:{sender_email}" if sender_email else "dm:unknown"

    def _cache_dm_user_id(self, chat_id: str, message: Dict[str, Any]) -> None:
        """Remember the Zulip user ID for an inbound DM chat.

        Zulip's /typing endpoint requires numeric user IDs for direct-message
        recipients. Inbound message events include sender_id, and often IDs in
        display_recipient, so cache the non-bot participant against Hermes'
        email-based dm:<email> chat ID.
        """
        user_id = message.get("sender_id")
        recipients = message.get("display_recipient") or []
        if isinstance(recipients, list):
            for recipient in recipients:
                if not isinstance(recipient, dict):
                    continue
                email = str(recipient.get("email") or "")
                if email and email.lower() == self.bot_email.lower():
                    continue
                user_id = recipient.get("id", user_id)
                break
        try:
            if user_id is not None:
                self._dm_user_ids[chat_id] = int(user_id)
        except (TypeError, ValueError):
            logger.debug("[Zulip] Could not cache DM user ID for %s: %r", chat_id, user_id)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        metadata = metadata or {}
        try:
            data = self._send_payload(chat_id, content, metadata)
            response = await self._api_post("/messages", data)
            message_id = response.get("id") or response.get("message_id")
            return SendResult(success=True, message_id=str(message_id) if message_id is not None else None, raw_response=response)
        except Exception as e:
            return SendResult(success=False, error=str(e))

    def _send_payload(self, chat_id: str, content: str, metadata: Dict[str, Any]) -> Dict[str, str]:
        if chat_id.startswith("dm:"):
            recipients = [v.strip() for v in chat_id[3:].split(",") if v.strip()]
            return {"type": "direct", "to": json.dumps(recipients), "content": content}
        if chat_id.startswith("stream:"):
            stream_ref = chat_id[len("stream:") :]
            topic = metadata.get("thread_id") or metadata.get("topic")
            if "/" in stream_ref and not topic:
                stream_ref, topic = stream_ref.split("/", 1)
            if not topic:
                topic = "Hermes"
            return {"type": "stream", "to": stream_ref, "topic": str(topic), "content": content}
        if "@" in chat_id:
            return {"type": "private", "to": json.dumps([chat_id]), "content": content}
        raise ValueError("Zulip chat_id must be dm:<email> or stream:<stream_id-or-name>[/topic]")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Send Zulip native typing notifications for the active turn."""
        payload = self._typing_payload(chat_id, "start", metadata or {})
        if not payload:
            return
        try:
            await self._api_post("/typing", payload)
            self._active_typing_payloads[chat_id] = payload
        except Exception as e:
            logger.debug("[Zulip] typing start failed for %s: %s", chat_id, e)

    async def stop_typing(self, chat_id: str) -> None:
        """Clear a Zulip native typing notification if one was started."""
        payload = self._active_typing_payloads.pop(chat_id, None)
        if not payload:
            return
        payload = dict(payload)
        payload["op"] = "stop"
        try:
            await self._api_post("/typing", payload)
        except Exception as e:
            logger.debug("[Zulip] typing stop failed for %s: %s", chat_id, e)

    def _typing_payload(self, chat_id: str, op: str, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if chat_id.startswith("dm:"):
            user_id = self._dm_user_ids.get(chat_id)
            if user_id is None:
                logger.debug("[Zulip] Cannot send DM typing for %s without cached user ID", chat_id)
                return None
            return {"type": "direct", "op": op, "to": json.dumps([user_id])}

        if chat_id.startswith("stream:"):
            stream_ref = chat_id[len("stream:") :]
            topic = metadata.get("thread_id") or metadata.get("topic")
            if "/" in stream_ref and not topic:
                stream_ref, topic = stream_ref.split("/", 1)
            if not stream_ref or not topic:
                logger.debug("[Zulip] Cannot send channel typing without stream ID and topic")
                return None
            return {"type": "stream", "op": op, "stream_id": str(stream_ref), "topic": str(topic)}

        return None

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        if chat_id.startswith("dm:"):
            return {"name": chat_id[3:], "type": "dm", "chat_id": chat_id}
        if chat_id.startswith("stream:"):
            stream = chat_id[len("stream:") :].split("/", 1)[0]
            return {"name": stream, "type": "channel", "chat_id": chat_id}
        return {"name": chat_id, "type": "unknown", "chat_id": chat_id}

    async def _api_get(self, path: str, params: Dict[str, str]) -> Dict[str, Any]:
        if not self._session:
            raise RuntimeError("Zulip HTTP session is not connected")
        async with self._session.get(f"{self.site}/api/v1{path}", params=params) as resp:
            payload = await resp.json(content_type=None)
            if resp.status >= 400 or payload.get("result") == "error":
                raise RuntimeError(payload.get("msg") or f"HTTP {resp.status}")
            return payload

    async def _api_post(self, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        owns_session = self._session is None
        session = self._session
        if session is None:
            timeout = aiohttp.ClientTimeout(total=60)
            session = aiohttp.ClientSession(timeout=timeout, auth=self._auth())
        try:
            async with session.post(f"{self.site}/api/v1{path}", data=data) as resp:
                payload = await resp.json(content_type=None)
                if resp.status >= 400 or payload.get("result") == "error":
                    raise RuntimeError(payload.get("msg") or f"HTTP {resp.status}")
                return payload
        finally:
            if owns_session:
                await session.close()
