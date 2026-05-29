"""Generic webhook platform adapter.

Runs an aiohttp HTTP server that receives webhook POSTs from external
services (GitHub, GitLab, JIRA, Stripe, etc.), validates HMAC signatures,
transforms payloads into agent prompts, and routes responses back to the
source or to another configured platform.

Configuration lives in config.yaml under platforms.webhook.extra.routes.
Each route defines:
  - events: which event types to accept (header-based filtering)
  - secret: HMAC secret for signature validation (REQUIRED)
  - prompt: template string formatted with the webhook payload
  - skills: optional list of skills to load for the agent
  - deliver: where to send the response (github_comment, telegram, etc.)
  - deliver_extra: additional delivery config (repo, pr_number, chat_id)
  - deliver_only: if true, skip the agent — the rendered prompt IS the
    message that gets delivered.  Use for external push notifications
    (Supabase, monitoring alerts, inter-agent pings) where zero LLM cost
    and sub-second delivery matter more than agent reasoning.

Security:
  - HMAC secret is required per route (validated at startup)
  - Rate limiting per route (fixed-window, configurable)
  - Idempotency cache prevents duplicate agent runs on webhook retries
  - Body size limits checked before reading payload
  - Set secret to "INSECURE_NO_AUTH" to skip validation (testing only)
"""

import asyncio
import base64
import binascii
import hashlib
import hmac
import json
import logging
import re
import subprocess
import time
from typing import Any, Dict, List, Optional

try:
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

_BUILTIN_DELIVER_PLATFORMS = {
    "telegram", "discord", "slack", "signal", "sms", "whatsapp",
    "matrix", "mattermost", "homeassistant", "email", "dingtalk",
    "feishu", "wecom", "wecom_callback", "weixin", "bluebubbles",
    "qqbot", "yuanbao", "zulip",
}

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8644
_INSECURE_NO_AUTH = "INSECURE_NO_AUTH"
_DYNAMIC_ROUTES_FILENAME = "webhook_subscriptions.json"
_SUPPORT_REASONING_TIMEOUT_SECONDS = 5.0

# Hostnames/IP literals that only serve connections originating on the same
# machine. Anything else is treated as a public bind for safety-rail purposes.
_LOOPBACK_HOSTS = frozenset({
    "127.0.0.1",
    "localhost",
    "::1",
    "ip6-localhost",
    "ip6-loopback",
})


def _is_loopback_host(host: str) -> bool:
    """True when `host` binds only to the local machine.

    Covers IPv4 loopback, the standard `localhost` alias, IPv6 loopback in
    both bracketed and bare form, and the common Debian-style aliases. Any
    falsy value (empty string, None) is conservatively treated as non-loopback
    because an unset host usually means the platform-default public bind.
    """
    if not host:
        return False
    return host.strip().lower() in _LOOPBACK_HOSTS


def check_webhook_requirements() -> bool:
    """Check if webhook adapter dependencies are available."""
    return AIOHTTP_AVAILABLE


class WebhookAdapter(BasePlatformAdapter):
    """Generic webhook receiver that triggers agent runs from HTTP POSTs."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.WEBHOOK)
        self._host: str = config.extra.get("host", DEFAULT_HOST)
        self._port: int = int(config.extra.get("port", DEFAULT_PORT))
        self._global_secret: str = config.extra.get("secret", "")
        self._static_routes: Dict[str, dict] = config.extra.get("routes", {})
        self._dynamic_routes: Dict[str, dict] = {}
        self._dynamic_routes_mtime: float = 0.0
        self._routes: Dict[str, dict] = dict(self._static_routes)
        self._runner = None

        # Delivery info keyed by session chat_id.
        #
        # Read by every send() invocation for the chat_id (status messages
        # AND the final response).  Cleaned up via TTL on each POST so the
        # dict stays bounded — see _prune_delivery_info().  Do NOT pop on
        # send(), or interim status messages (e.g. fallback notifications,
        # context-pressure warnings) will consume the entry before the
        # final response arrives, causing the response to silently fall
        # back to the "log" deliver type.
        self._delivery_info: Dict[str, dict] = {}
        self._delivery_info_created: Dict[str, float] = {}

        # Reference to gateway runner for cross-platform delivery (set externally)
        self.gateway_runner = None

        # Idempotency: TTL cache of recently processed delivery IDs.
        # Prevents duplicate agent runs when webhook providers retry.
        self._seen_deliveries: Dict[str, float] = {}
        self._idempotency_ttl: int = 3600  # 1 hour

        # Rate limiting: per-route timestamps in a fixed window.
        self._rate_counts: Dict[str, List[float]] = {}
        self._rate_limit: int = int(config.extra.get("rate_limit", 30))  # per minute

        # Body size limit (auth-before-body pattern)
        self._max_body_bytes: int = int(
            config.extra.get("max_body_bytes", 1_048_576)
        )  # 1MB

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        # Load agent-created subscriptions before validating
        self._reload_dynamic_routes()

        # Validate routes at startup — secret is required per route
        for name, route in self._routes.items():
            secret = route.get("secret", self._global_secret)
            if not secret:
                raise ValueError(
                    f"[webhook] Route '{name}' has no HMAC secret. "
                    f"Set 'secret' on the route or globally. "
                    f"For testing without auth, set secret to '{_INSECURE_NO_AUTH}'."
                )

            # Safety rail: refuse to start if INSECURE_NO_AUTH is combined with a
            # non-loopback bind. The escape hatch is for local testing only;
            # serving an unauthenticated route on a public interface is a
            # deployment-grade footgun we'd rather crash early than ship.
            if secret == _INSECURE_NO_AUTH and not _is_loopback_host(self._host):
                raise ValueError(
                    f"[webhook] Route '{name}' uses INSECURE_NO_AUTH secret "
                    f"but is bound to non-loopback host '{self._host}'. "
                    f"INSECURE_NO_AUTH is for local testing only. "
                    f"Refusing to start to prevent accidental exposure."
                )
            # deliver_only routes bypass the agent — the POST body becomes a
            # direct push notification via the configured delivery target.
            # Validate up-front so misconfiguration surfaces at startup rather
            # than on the first webhook POST.
            if route.get("deliver_only"):
                deliver = route.get("deliver", "log")
                if not deliver or deliver == "log":
                    raise ValueError(
                        f"[webhook] Route '{name}' has deliver_only=true but "
                        f"deliver is '{deliver}'. Direct delivery requires a "
                        f"real target (telegram, discord, slack, github_comment, etc.)."
                    )

        app = web.Application()
        app.router.add_get("/health", self._handle_health)
        app.router.add_post("/webhooks/{route_name}", self._handle_webhook)

        # Port conflict detection — fail fast if port is already in use
        import socket as _socket
        try:
            with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as _s:
                _s.settimeout(1)
                _s.connect(('127.0.0.1', self._port))
            logger.error('[webhook] Port %d already in use. Set a different port in config.yaml: platforms.webhook.port', self._port)
            return False
        except (ConnectionRefusedError, OSError):
            pass  # port is free

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()
        self._mark_connected()

        route_names = ", ".join(self._routes.keys()) or "(none configured)"
        logger.info(
            "[webhook] Listening on %s:%d — routes: %s",
            self._host,
            self._port,
            route_names,
        )
        return True

    async def disconnect(self) -> None:
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self._mark_disconnected()
        logger.info("[webhook] Disconnected")

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Deliver the agent's response to the configured destination.

        chat_id is ``webhook:{route}:{delivery_id}``.  The delivery info
        stored during webhook receipt is read with ``.get()`` (not popped)
        so that interim status messages emitted before the final response
        — fallback-model notifications, context-pressure warnings, etc. —
        do not consume the entry and silently downgrade the final response
        to the ``log`` deliver type.  TTL cleanup happens on POST.
        """
        delivery = self._delivery_info.get(chat_id, {})
        deliver_type = delivery.get("deliver", "log")

        if deliver_type == "log":
            logger.info("[webhook] Response for %s: %s", chat_id, content[:200])
            return SendResult(success=True)

        if deliver_type == "github_comment":
            return await self._deliver_github_comment(content, delivery)

        # Cross-platform delivery — any platform with a gateway adapter.
        # Check both built-in names and plugin-registered platforms.
        _is_known_platform = deliver_type in _BUILTIN_DELIVER_PLATFORMS
        if not _is_known_platform:
            try:
                from gateway.platform_registry import platform_registry
                _is_known_platform = platform_registry.is_registered(deliver_type)
            except Exception:
                pass
        if self.gateway_runner and _is_known_platform:
            return await self._deliver_cross_platform(
                deliver_type, content, delivery
            )

        logger.warning("[webhook] Unknown deliver type: %s", deliver_type)
        return SendResult(
            success=False, error=f"Unknown deliver type: {deliver_type}"
        )

    def _prune_delivery_info(self, now: float) -> None:
        """Drop delivery_info entries older than the idempotency TTL.

        Mirrors the cleanup pattern used for ``_seen_deliveries``.  Called
        on each POST so the dict size is bounded by ``rate_limit * TTL``
        even if many webhooks fire and never receive a final response.
        """
        cutoff = now - self._idempotency_ttl
        stale = [
            k
            for k, t in self._delivery_info_created.items()
            if t < cutoff
        ]
        for k in stale:
            self._delivery_info.pop(k, None)
            self._delivery_info_created.pop(k, None)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "webhook"}

    # ------------------------------------------------------------------
    # HTTP handlers
    # ------------------------------------------------------------------

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        """GET /health — simple health check."""
        return web.json_response({"status": "ok", "platform": "webhook"})

    def _reload_dynamic_routes(self) -> None:
        """Reload agent-created subscriptions from disk if the file changed."""
        from hermes_constants import get_hermes_home
        hermes_home = get_hermes_home()
        subs_path = hermes_home / _DYNAMIC_ROUTES_FILENAME
        if not subs_path.exists():
            if self._dynamic_routes:
                self._dynamic_routes = {}
                self._routes = dict(self._static_routes)
                logger.debug("[webhook] Dynamic subscriptions file removed, cleared dynamic routes")
            return
        try:
            mtime = subs_path.stat().st_mtime
            if mtime <= self._dynamic_routes_mtime:
                return  # No change
            data = json.loads(subs_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return
            # Merge: static routes take precedence over dynamic ones.
            # Reject any dynamic route whose effective secret is empty —
            # an empty secret would cause _handle_webhook to skip HMAC
            # validation entirely, letting unauthenticated callers in.
            new_dynamic: Dict[str, dict] = {}
            for k, v in data.items():
                if k in self._static_routes:
                    continue
                effective_secret = v.get("secret", self._global_secret)
                if not effective_secret:
                    logger.warning(
                        "[webhook] Dynamic route '%s' skipped: 'secret' is "
                        "missing or empty. Set a valid HMAC secret, or use "
                        "'%s' to explicitly disable auth (testing only).",
                        k,
                        _INSECURE_NO_AUTH,
                    )
                    continue
                if (
                    effective_secret == _INSECURE_NO_AUTH
                    and not _is_loopback_host(self._host)
                ):
                    logger.warning(
                        "[webhook] Dynamic route '%s' skipped: INSECURE_NO_AUTH "
                        "is only allowed on loopback hosts. Current host: '%s'.",
                        k,
                        self._host,
                    )
                    continue
                new_dynamic[k] = v
            self._dynamic_routes = new_dynamic
            self._routes = {**self._dynamic_routes, **self._static_routes}
            self._dynamic_routes_mtime = mtime
            logger.info(
                "[webhook] Reloaded %d dynamic route(s): %s",
                len(self._dynamic_routes),
                ", ".join(self._dynamic_routes.keys()) or "(none)",
            )
        except Exception as e:
            logger.error("[webhook] Failed to reload dynamic routes: %s", e)

    async def _handle_webhook(self, request: "web.Request") -> "web.Response":
        """POST /webhooks/{route_name} — receive and process a webhook event."""
        # Hot-reload dynamic subscriptions on each request (mtime-gated, cheap)
        self._reload_dynamic_routes()

        route_name = request.match_info.get("route_name", "")
        route_config = self._routes.get(route_name)

        if not route_config:
            return web.json_response(
                {"error": f"Unknown route: {route_name}"}, status=404
            )

        # ── Auth-before-body ─────────────────────────────────────
        # Check Content-Length before reading the full payload.
        content_length = request.content_length or 0
        if content_length > self._max_body_bytes:
            return web.json_response(
                {"error": "Payload too large"}, status=413
            )

        # Read body (must be done before any validation)
        try:
            raw_body = await request.read()
        except Exception as e:
            logger.error("[webhook] Failed to read body: %s", e)
            return web.json_response({"error": "Bad request"}, status=400)

        # Validate HMAC signature FIRST (skip only for the explicit local-test
        # INSECURE_NO_AUTH mode). Missing/empty secrets must fail closed here,
        # not only during connect(), so direct handler reuse cannot turn a
        # network webhook route into an unauthenticated agent-dispatch surface.
        secret = route_config.get("secret", self._global_secret)
        if not secret:
            logger.error(
                "[webhook] Route %s has no HMAC secret; refusing request",
                route_name,
            )
            return web.json_response(
                {"error": "Webhook route is missing an HMAC secret"},
                status=403,
            )
        if secret != _INSECURE_NO_AUTH:
            if not self._validate_signature(request, raw_body, secret):
                logger.warning(
                    "[webhook] Invalid signature for route %s", route_name
                )
                return web.json_response(
                    {"error": "Invalid signature"}, status=401
                )

        # ── Rate limiting (after auth) ───────────────────────────
        now = time.time()
        window = self._rate_counts.setdefault(route_name, [])
        window[:] = [t for t in window if now - t < 60]
        if len(window) >= self._rate_limit:
            return web.json_response(
                {"error": "Rate limit exceeded"}, status=429
            )
        window.append(now)

        # Parse payload
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            # Try form-encoded as fallback
            try:
                import urllib.parse

                payload = dict(
                    urllib.parse.parse_qsl(raw_body.decode("utf-8"))
                )
            except Exception:
                return web.json_response(
                    {"error": "Cannot parse body"}, status=400
                )

        # Check event type filter
        event_type = (
            request.headers.get("X-GitHub-Event", "")
            or request.headers.get("X-GitLab-Event", "")
            or payload.get("event_type", "")
            or payload.get("type", "")
            or "unknown"
        )
        allowed_events = route_config.get("events", [])
        if allowed_events and event_type not in allowed_events:
            logger.debug(
                "[webhook] Ignoring event %s for route %s (allowed: %s)",
                event_type,
                route_name,
                allowed_events,
            )
            return web.json_response(
                {"status": "ignored", "event": event_type}
            )

        # Format prompt from template
        prompt_template = route_config.get("prompt", "")
        prompt = self._render_prompt(
            prompt_template, payload, event_type, route_name
        )

        # Inject skill content if configured.
        # We call build_skill_invocation_message() directly rather than
        # using /skill-name slash commands — the gateway's command parser
        # would intercept those and break the flow.
        skills = route_config.get("skills", [])
        if skills:
            try:
                from agent.skill_commands import (
                    build_skill_invocation_message,
                    get_skill_commands,
                )

                skill_cmds = get_skill_commands()
                for skill_name in skills:
                    cmd_key = f"/{skill_name}"
                    if cmd_key in skill_cmds:
                        skill_content = build_skill_invocation_message(
                            cmd_key, user_instruction=prompt
                        )
                        if skill_content:
                            prompt = skill_content
                            break  # Load the first matching skill
                    else:
                        logger.warning(
                            "[webhook] Skill '%s' not found", skill_name
                        )
            except Exception as e:
                logger.warning("[webhook] Skill loading failed: %s", e)

        # Build a unique delivery ID
        delivery_id = request.headers.get(
            "X-GitHub-Delivery",
            request.headers.get(
                "svix-id",
                request.headers.get("X-Request-ID", str(int(time.time() * 1000))),
            ),
        )

        # ── Idempotency ─────────────────────────────────────────
        # Skip duplicate deliveries (webhook retries).
        now = time.time()
        # Prune expired entries
        self._seen_deliveries = {
            k: v
            for k, v in self._seen_deliveries.items()
            if now - v < self._idempotency_ttl
        }
        if delivery_id in self._seen_deliveries:
            logger.info(
                "[webhook] Skipping duplicate delivery %s", delivery_id
            )
            return web.json_response(
                {"status": "duplicate", "delivery_id": delivery_id},
                status=200,
            )
        self._seen_deliveries[delivery_id] = now

        # ── CLCK HubSpot support triage ─────────────────────────
        # Deterministic internal card path for ActivePieces support-intake
        # events.  This intentionally bypasses the agent so the MVP cannot
        # send email, post to client Slack, or write to HubSpot while still
        # replying in the supplied internal Slack thread.
        if event_type == "hubspot_support_triage":
            payload = await self._enrich_hubspot_support_triage_payload(payload, route_config)
            if self._hubspot_support_triage_should_suppress(payload):
                logger.info(
                    "[webhook] support-triage suppressed as deterministic noise route=%s delivery_id=%s",
                    route_name,
                    delivery_id,
                )
                return web.json_response(
                    {
                        "status": "suppressed",
                        "route": route_name,
                        "event": event_type,
                        "delivery_id": delivery_id,
                        "handler": "hubspot_support_triage",
                    },
                    status=202,
                )
            delivery = {
                "deliver": route_config.get("deliver", "slack"),
                "deliver_extra": self._render_delivery_extra(
                    route_config.get("deliver_extra", {}), payload
                ),
                "payload": payload,
            }
            content = self._format_hubspot_support_triage_card(payload)
            logger.info(
                "[webhook] support-triage event route=%s target=%s msg_len=%d delivery=%s",
                route_name,
                delivery["deliver"],
                len(content),
                delivery_id,
            )
            try:
                result = await self._direct_deliver(content, delivery)
            except Exception:
                logger.exception(
                    "[webhook] support-triage delivery failed route=%s delivery=%s",
                    route_name,
                    delivery_id,
                )
                return web.json_response(
                    {"status": "error", "error": "Delivery failed", "delivery_id": delivery_id},
                    status=502,
                )

            if result.success:
                # Store Slack top-level ts → HubSpot ticket mapping for reply tracking.
                # If this intake was routed into an existing Slack thread, keep the
                # original parent ts and do not overwrite it with the threaded reply ts.
                slack_ts = getattr(result, "message_id", None) or ""
                reasoning = payload.get("support_reasoning", {}) if isinstance(payload.get("support_reasoning"), dict) else {}
                ticket_id = reasoning.get("ticket_id", "")
                slack_payload = payload.get("slack") if isinstance(payload.get("slack"), dict) else {}
                existing_thread_ts = reasoning.get("slack_thread_ts") or slack_payload.get("thread_ts") or ""
                if slack_ts and ticket_id and not existing_thread_ts:
                    try:
                        await self._store_slack_thread_ts(ticket_id, slack_ts)
                    except Exception:
                        logger.warning("[webhook] Failed to store slack_thread_ts (non-fatal)")

                return web.json_response(
                    {
                        "status": "accepted",
                        "route": route_name,
                        "event": event_type,
                        "delivery_id": delivery_id,
                        "handler": "hubspot_support_triage",
                    },
                    status=202,
                )
            logger.warning(
                "[webhook] support-triage target rejected route=%s target=%s error=%s",
                route_name,
                delivery["deliver"],
                result.error,
            )
            return web.json_response(
                {"status": "error", "error": "Delivery failed", "delivery_id": delivery_id},
                status=502,
            )

        # ── Client reply tracking ───────────────────────────────────
        # Detects when a client replies to a support@ thread and posts
        # a threaded update to the original Slack triage card.
        if event_type == "support_client_reply":
            gmail = payload.get("gmail") if isinstance(payload.get("gmail"), dict) else {}
            reply_subject = gmail.get("subject") or payload.get("subject") or ""
            reply_body = (
                gmail.get("snippet") or gmail.get("body_preview")
                or payload.get("summary") or ""
            )
            reply_sender = gmail.get("from") or ""

            # Find the HubSpot ticket by subject to get slack_thread_ts
            ticket = await self._find_ticket_by_subject(reply_subject)
            if not ticket:
                logger.info(
                    "[webhook] client-reply no ticket found for subject=%s",
                    reply_subject[:80],
                )
                return web.json_response(
                    {"status": "no_ticket_found", "delivery_id": delivery_id},
                    status=200,
                )

            ticket_id = ticket["id"]
            props = ticket.get("properties", {})
            slack_ts = props.get("slack_thread_ts", "")
            ticket_url = f"https://app.hubspot.com/contacts/435014/ticket/{ticket_id}"

            if not slack_ts:
                logger.info(
                    "[webhook] client-reply no slack_thread_ts for ticket=%s", ticket_id
                )
                return web.json_response(
                    {"status": "no_slack_ts", "delivery_id": delivery_id},
                    status=200,
                )

            # Build and post threaded reply card to the original Slack thread
            reply_card = (
                f"📬 *Client reply on existing support ticket*\n"
                f"- From: {reply_sender}\n"
                f"- Update: {reply_body[:500]}\n"
                f"- Next: review this update in the existing working thread and reply with `@Arlo` plus the next bounded action/approval.\n\n"
                f"_<{ticket_url}|View ticket in HubSpot>_"
            )

            try:
                reply_result = await self._send_slack_threaded_reply(
                    "C0B3PQE0CHG", slack_ts, reply_card
                )
            except Exception:
                logger.exception(
                    "[webhook] client-reply Slack delivery failed ticket=%s", ticket_id
                )
                return web.json_response(
                    {"status": "error", "error": "Slack reply delivery failed", "delivery_id": delivery_id},
                    status=502,
                )

            if not reply_result.success:
                logger.warning(
                    "[webhook] client-reply Slack rejected ticket=%s error=%s",
                    ticket_id, reply_result.error,
                )
                return web.json_response(
                    {"status": "error", "error": "Slack reply rejected", "delivery_id": delivery_id},
                    status=502,
                )

            # Move ticket to "Waiting on us" (stage 3) — client replied
            try:
                await self._update_ticket_stage(ticket_id, "3")
            except Exception:
                logger.warning("[webhook] client-reply stage update failed (non-fatal)")

            logger.info(
                "[webhook] client-reply posted ticket=%s slack_thread=%s",
                ticket_id, slack_ts,
            )
            return web.json_response(
                {
                    "status": "reply_posted",
                    "ticket_id": ticket_id,
                    "delivery_id": delivery_id,
                },
                status=202,
            )

        # ── Direct delivery mode (deliver_only) ─────────────────
        # Skip the agent entirely — the rendered prompt IS the message we
        # deliver.  Use case: external services (Supabase, monitoring,
        # cron jobs, other agents) that need to push a plain notification
        # to a user's chat with zero LLM cost.  Reuses the same HMAC auth,
        # rate limiting, idempotency, and template rendering as agent mode.
        if route_config.get("deliver_only"):
            delivery = {
                "deliver": route_config.get("deliver", "log"),
                "deliver_extra": self._render_delivery_extra(
                    route_config.get("deliver_extra", {}), payload
                ),
                "payload": payload,
            }
            logger.info(
                "[webhook] direct-deliver event=%s route=%s target=%s msg_len=%d delivery=%s",
                event_type,
                route_name,
                delivery["deliver"],
                len(prompt),
                delivery_id,
            )
            try:
                result = await self._direct_deliver(prompt, delivery)
            except Exception:
                logger.exception(
                    "[webhook] direct-deliver failed route=%s delivery=%s",
                    route_name,
                    delivery_id,
                )
                return web.json_response(
                    {"status": "error", "error": "Delivery failed", "delivery_id": delivery_id},
                    status=502,
                )

            if result.success:
                return web.json_response(
                    {
                        "status": "delivered",
                        "route": route_name,
                        "target": delivery["deliver"],
                        "delivery_id": delivery_id,
                    },
                    status=200,
                )
            # Delivery attempted but target rejected it — surface as 502
            # with a generic error (don't leak adapter-level detail).
            logger.warning(
                "[webhook] direct-deliver target rejected route=%s target=%s error=%s",
                route_name,
                delivery["deliver"],
                result.error,
            )
            return web.json_response(
                {"status": "error", "error": "Delivery failed", "delivery_id": delivery_id},
                status=502,
            )

        # Use delivery_id in session key so concurrent webhooks on the
        # same route get independent agent runs (not queued/interrupted).
        session_chat_id = f"webhook:{route_name}:{delivery_id}"

        # Store delivery info for send().  Read by every send() invocation
        # for this chat_id (interim status messages and the final response),
        # so we do NOT pop on send.  TTL-based cleanup keeps the dict bounded.
        deliver_config = {
            "deliver": route_config.get("deliver", "log"),
            "deliver_extra": self._render_delivery_extra(
                route_config.get("deliver_extra", {}), payload
            ),
            "payload": payload,
        }
        self._delivery_info[session_chat_id] = deliver_config
        self._delivery_info_created[session_chat_id] = now
        self._prune_delivery_info(now)

        # Build source and event
        source = self.build_source(
            chat_id=session_chat_id,
            chat_name=f"webhook/{route_name}",
            chat_type="webhook",
            user_id=f"webhook:{route_name}",
            user_name=route_name,
        )
        event = MessageEvent(
            text=prompt,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=payload,
            message_id=delivery_id,
        )

        logger.info(
            "[webhook] %s event=%s route=%s prompt_len=%d delivery=%s",
            request.method,
            event_type,
            route_name,
            len(prompt),
            delivery_id,
        )

        # Non-blocking — return 202 Accepted immediately
        task = asyncio.create_task(self.handle_message(event))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

        return web.json_response(
            {
                "status": "accepted",
                "route": route_name,
                "event": event_type,
                "delivery_id": delivery_id,
            },
            status=202,
        )

    # ------------------------------------------------------------------
    # Signature validation
    # ------------------------------------------------------------------

    def _validate_signature(
        self, request: "web.Request", body: bytes, secret: str
    ) -> bool:
        """Validate webhook signature (GitHub, GitLab, Svix, generic HMAC-SHA256)."""
        def _header(name: str) -> str:
            return (
                request.headers.get(name, "")
                or request.headers.get(name.lower(), "")
                or request.headers.get(name.upper(), "")
            )

        # Svix / AgentMail:
        #   svix-id: msg_...
        #   svix-timestamp: unix seconds
        #   svix-signature: v1,<base64-hmac> [v1,<base64-hmac> ...]
        # Signed content is: "{id}.{timestamp}.{raw_body}".  Svix secrets
        # usually start with "whsec_" and the remainder is base64-encoded.
        svix_id = _header("svix-id")
        svix_timestamp = _header("svix-timestamp")
        svix_signature = _header("svix-signature")
        if svix_id or svix_timestamp or svix_signature:
            return self._validate_svix_signature(
                body=body,
                secret=secret,
                msg_id=svix_id,
                timestamp=svix_timestamp,
                signature_header=svix_signature,
            )

        # GitHub: X-Hub-Signature-256 = sha256=<hex>
        gh_sig = request.headers.get("X-Hub-Signature-256", "")
        if gh_sig:
            expected = "sha256=" + hmac.new(
                secret.encode(), body, hashlib.sha256
            ).hexdigest()
            return hmac.compare_digest(gh_sig, expected)

        # GitLab: X-Gitlab-Token = <plain secret>
        gl_token = request.headers.get("X-Gitlab-Token", "")
        if gl_token:
            return hmac.compare_digest(gl_token, secret)

        # Generic: X-Webhook-Signature = <hex HMAC-SHA256>
        generic_sig = request.headers.get("X-Webhook-Signature", "")
        if generic_sig:
            expected = hmac.new(
                secret.encode(), body, hashlib.sha256
            ).hexdigest()
            return hmac.compare_digest(generic_sig, expected)

        # No recognised signature header but secret is configured → reject
        logger.debug(
            "[webhook] Secret configured but no signature header found"
        )
        return False

    def _validate_svix_signature(
        self,
        body: bytes,
        secret: str,
        msg_id: str,
        timestamp: str,
        signature_header: str,
        tolerance_seconds: int = 300,
    ) -> bool:
        """Validate Svix-compatible signatures used by AgentMail webhooks."""
        if not (msg_id and timestamp and signature_header and secret):
            return False

        try:
            ts = int(timestamp)
        except (TypeError, ValueError):
            return False
        if abs(int(time.time()) - ts) > tolerance_seconds:
            logger.warning("[webhook] Svix signature timestamp outside replay window")
            return False

        if secret.startswith("whsec_"):
            encoded_secret = secret.removeprefix("whsec_")
            try:
                key = base64.b64decode(encoded_secret, validate=True)
            except (binascii.Error, ValueError):
                logger.debug("[webhook] Invalid whsec_ Svix signing secret")
                return False
        else:
            # Be permissive for providers that document Svix-style headers but
            # hand out raw shared secrets rather than whsec_ base64 secrets.
            logger.debug("[webhook] Validating Svix-style signature with raw secret")
            key = secret.encode()

        signed_content = msg_id.encode() + b"." + timestamp.encode() + b"." + body
        expected = base64.b64encode(
            hmac.new(key, signed_content, hashlib.sha256).digest()
        ).decode()

        # Svix can send multiple signatures separated by spaces during secret
        # rotation. Each entry is formatted as "vN,<base64>".
        for part in signature_header.split():
            try:
                version, signature = part.split(",", 1)
            except ValueError:
                continue
            if version == "v1" and hmac.compare_digest(signature, expected):
                return True
        return False

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------

    def _render_prompt(
        self,
        template: str,
        payload: dict,
        event_type: str,
        route_name: str,
    ) -> str:
        """Render a prompt template with the webhook payload.

        Supports dot-notation access into nested dicts:
        ``{pull_request.title}`` → ``payload["pull_request"]["title"]``

        Special token ``{__raw__}`` dumps the entire payload as indented
        JSON (truncated to 4000 chars).  Useful for monitoring alerts or
        any webhook where the agent needs to see the full payload.
        """
        if not template:
            truncated = json.dumps(payload, indent=2)[:4000]
            return (
                f"Webhook event '{event_type}' on route "
                f"'{route_name}':\n\n```json\n{truncated}\n```"
            )

        def _resolve(match: re.Match) -> str:
            key = match.group(1)
            # Special token: dump the entire payload as JSON
            if key == "__raw__":
                return json.dumps(payload, indent=2)[:4000]
            value: Any = payload
            for part in key.split("."):
                if isinstance(value, dict):
                    value = value.get(part, f"{{{key}}}")
                else:
                    return f"{{{key}}}"
            if isinstance(value, (dict, list)):
                return json.dumps(value, indent=2)[:2000]
            return str(value)

        return re.sub(r"\{([a-zA-Z0-9_.]+)\}", _resolve, template)

    def _render_delivery_extra(
        self, extra: dict, payload: dict
    ) -> dict:
        """Render delivery_extra template values with payload data."""
        rendered: Dict[str, Any] = {}
        for key, value in extra.items():
            if isinstance(value, str):
                rendered[key] = self._render_prompt(value, payload, "", "")
            else:
                rendered[key] = value
        return rendered

    def _get_env_or_dotenv(self, name: str) -> str:
        """Read a secret from process env, falling back to ~/.hermes/.env without logging it."""
        import os

        value = os.environ.get(name, "")
        if value:
            return value
        env_path = os.path.expanduser("~/.hermes/.env")
        if not os.path.exists(env_path):
            return ""
        try:
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, raw_value = line.split("=", 1)
                    if key.strip() == name:
                        return raw_value.strip().strip('"').strip("'")
        except Exception:
            return ""
        return ""

    def _normalise_support_subject(self, subject: str) -> str:
        """Normalise email reply/forward subjects for ticket continuation matching."""
        text = str(subject or "").strip().lower()
        # Gmail/Outlook can stack prefixes: Re: Fwd: RE: Original subject
        previous = None
        while text and text != previous:
            previous = text
            text = re.sub(r"^\s*(?:re|fw|fwd)\s*:\s*", "", text, flags=re.I)
            text = re.sub(r"^\s*\[[^\]]*(?:external|secure|bulk)[^\]]*\]\s*", "", text, flags=re.I)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:160]

    async def _enrich_hubspot_support_triage_payload(
        self, payload: dict, route_config: Optional[dict] = None
    ) -> dict:
        """Classify support@ intake with DeepSeek and attach structured reasoning.

        Fail-open: if DeepSeek is unavailable, times out, or errors, the
        deterministic base card still renders with a safe fallback marker.
        """
        if payload.get("event_type") != "hubspot_support_triage":
            return payload

        enriched = dict(payload)

        route_config = route_config or {}
        try:
            timeout = float(
                route_config.get(
                    "support_reasoning_timeout_seconds",
                    _SUPPORT_REASONING_TIMEOUT_SECONDS,
                )
            )
        except (TypeError, ValueError):
            timeout = _SUPPORT_REASONING_TIMEOUT_SECONDS
        timeout = max(0.5, min(timeout, 10.0))

        try:
            reasoning = await self._run_deepseek_classifier(payload, timeout=timeout)
            if not isinstance(reasoning, dict):
                raise ValueError("DeepSeek classifier returned non-object JSON")
            enriched["support_reasoning"] = reasoning

            # After classification, lookup HubSpot ticket and assign owner
            try:
                hubspot_enrichment = await self._enrich_hubspot_ticket(payload, reasoning, timeout=timeout)
                enriched["support_reasoning"].update(hubspot_enrichment)
                if hubspot_enrichment.get("slack_thread_ts"):
                    slack_payload = enriched.get("slack") if isinstance(enriched.get("slack"), dict) else {}
                    slack_payload = dict(slack_payload)
                    slack_payload.setdefault("channel_id", "C0B3PQE0CHG")
                    slack_payload["thread_ts"] = hubspot_enrichment["slack_thread_ts"]
                    enriched["slack"] = slack_payload
            except Exception:
                logger.warning("[webhook] HubSpot ticket enrichment failed (non-fatal)")
        except asyncio.TimeoutError:
            logger.warning(
                "[webhook] support-triage DeepSeek unavailable: timeout after %.1fs",
                timeout,
            )
            enriched["support_reasoning"] = self._support_reasoning_unavailable("timeout")
        except Exception as exc:
            logger.warning(
                "[webhook] support-triage DeepSeek unavailable: %s",
                type(exc).__name__,
            )
            enriched["support_reasoning"] = self._support_reasoning_unavailable("error")
        return enriched

    async def _maybe_revalidate_support_triage_matcher(
        self, payload: dict, route_config: Optional[dict] = None
    ) -> dict:
        """Deprecated: support@ intake hard-routes to C0B3PQE0CHG. No matcher revalidation."""
        return payload

    async def _run_support_matcher_worker(self, payload: dict, *, timeout: float) -> dict:
        """Deprecated: support@ intake hard-routes to C0B3PQE0CHG."""
        raise NotImplementedError("support matcher worker is deprecated")

    async def _run_deepseek_classifier(self, payload: dict, *, timeout: float) -> dict:
        """Classify a support@ intake email using DeepSeek and return structured JSON.

        Returns a dict with: mode, issue_type, assignee, summary, actions (list),
        draft_reply (str|null), internal_note (str|null).
        """
        api_key = self._get_env_or_dotenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not set")

        gmail = payload.get("gmail") if isinstance(payload.get("gmail"), dict) else {}
        support = payload.get("support") if isinstance(payload.get("support"), dict) else {}
        subject = gmail.get("subject") or payload.get("subject") or ""
        body = (
            support.get("summary")
            or support.get("requested_action")
            or gmail.get("snippet")
            or gmail.get("body_preview")
            or payload.get("summary")
            or ""
        )
        sender = gmail.get("from") or gmail.get("sender_email") or gmail.get("original_sender_email") or ""

        system_prompt = (
            "You are classifying a support intake email for CLCK, a B2B lead generation and HubSpot consultancy. "
            "Return ONLY valid JSON with no markdown, no code fences, no extra text.\n\n"
            "CLASSIFICATION RULES:\n"
            "- mode: \"action_plan\" if Arlo can take concrete actions (HubSpot changes, reports, automations, debugging, content/KB work, integrations, ticket routing, pixel tracking, code fixes). "
            "\"draft_reply\" if this is primarily a question from a client needing a written answer. "
            "\"summary_only\" if this is an FYI or status update with no action needed.\n"
            "- IMPORTANT: when the sender is from @clck.com.au (@damien, @marinda, @benson, @team) and the email is a forward or BCC, "
            "the account management/comms are already handled by the sender. Classify the UNDERLYING TECHNICAL REQUEST as action_plan, "
            "not draft_reply. The intake is about what work needs to be done, not what email to send back.\n"
            "- assignee: \"benson\" for technical tickets (HubSpot configuration, integrations, automations, "
            "reporting/dashboards, KB/content migration, debugging, tracking/pixels, ticket routing, "
            "ActivePieces/automation failures, API/workflow/form/property changes). "
            "\"marinda\" for administrative/coordination tickets (service agreements, billing/invoicing, "
            "scheduling/calendar, client coordination, scope/pricing questions, proposals, "
            "outbound/campaign strategy, general inquiries, anything not clearly technical).\n"
            "- summary: one concise sentence describing the request.\n"
            "- issue_type: short category label (e.g. \"HubSpot reporting dashboard update\", "
            "\"service agreement revision\", \"HubSpot workflow debugging\", "
            "\"outbound campaign strategy question\", \"general inquiry\").\n"
            "- actions: for mode=action_plan, a list of 1-5 concrete steps Arlo can take. "
            "Start each with a verb. Be specific about what system/area to inspect.\n"
            "- draft_reply: for mode=draft_reply, the email reply text (Australian English, direct, "
            "use contractions). Null otherwise.\n"
            "- internal_note: for mode=summary_only, a one-line note about what this is. Null otherwise.\n\n"
            "SCHEMA:\n"
            '{"mode": "action_plan|draft_reply|summary_only", "issue_type": "...", '
            '"assignee": "benson|marinda|internal_review", "summary": "...", '
            '"actions": ["..."]|[], "draft_reply": "..."|null, "internal_note": "..."|null}'
        )

        user_prompt = (
            f"Subject: {subject}\n"
            f"Sender: {sender}\n"
            f"Body: {body}"
        )

        timeout_sec = max(1.0, min(timeout, 10.0))
        try:
            import aiohttp
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_sec)) as session:
                async with session.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "deepseek-chat",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 0.1,
                        "max_tokens": 800,
                        "response_format": {"type": "json_object"},
                    },
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"DeepSeek API returned {resp.status}: {text[:200]}")
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"]
                    return json.loads(content)
        except json.JSONDecodeError:
            raise ValueError("DeepSeek returned invalid JSON")
        except (KeyError, IndexError):
            raise ValueError("DeepSeek response missing expected fields")
        except ImportError:
            raise RuntimeError("aiohttp not available for DeepSeek API call")

    async def _enrich_hubspot_ticket(
        self, payload: dict, reasoning: dict, *, timeout: float
    ) -> dict:
        """Look up the HubSpot ticket and mark existing Slack-thread continuations.

        Returns ticket_url/ticket_id plus slack_thread_ts/existing_ticket_update when
        the ticket already has a Slack parent thread stored. Fail-open on API issues.
        """
        token = self._get_env_or_dotenv("HUBSPOT_ACCESS_TOKEN")
        if not token:
            return {}

        gmail = payload.get("gmail") if isinstance(payload.get("gmail"), dict) else {}
        subject = gmail.get("subject") or payload.get("subject") or ""
        if not subject:
            return {}

        result: Dict[str, Any] = {}
        ticket = await self._find_ticket_by_subject(subject)
        if not ticket:
            return result

        ticket_id = ticket["id"]
        props = ticket.get("properties", {})
        result["ticket_id"] = ticket_id
        result["ticket_url"] = f"https://app.hubspot.com/contacts/435014/ticket/{ticket_id}"

        slack_thread_ts = props.get("slack_thread_ts") or ""
        if slack_thread_ts:
            result["slack_thread_ts"] = slack_thread_ts
            result["existing_ticket_update"] = True

        # Assign owner based on DeepSeek classification. Non-fatal and preserves
        # the existing behaviour for new and continuing tickets.
        try:
            import aiohttp

            timeout_sec = max(1.0, min(timeout * 0.5, 4.0))
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_sec)) as session:
                headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
                assignee = reasoning.get("assignee", "")
                if assignee == "benson":
                    await self._assign_hubspot_owner(session, ticket_id, "team@clck.com.au", headers)
                elif assignee == "marinda":
                    await self._assign_hubspot_owner(session, ticket_id, "marinda@clck.com.au", headers)
        except Exception:
            pass

        return result

    async def _assign_hubspot_owner(
        self, session, ticket_id: str, owner_email: str, headers: dict
    ) -> None:
        """Look up owner ID by email, then assign the ticket. Non-fatal."""
        try:
            # Find owner ID
            async with session.get(
                f"https://api.hubapi.com/crm/v3/owners/?email={owner_email}",
                headers=headers,
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    owners = data.get("results", [])
                    if owners:
                        owner_id = owners[0]["id"]
                        # Assign ticket
                        await session.patch(
                            f"https://api.hubapi.com/crm/v3/objects/tickets/{ticket_id}",
                            headers=headers,
                            json={"properties": {"hubspot_owner_id": owner_id}},
                        )
        except Exception:
            pass

    async def _store_slack_thread_ts(self, ticket_id: str, slack_ts: str) -> None:
        """Store the Slack message timestamp on the HubSpot ticket. Non-fatal."""
        token = self._get_env_or_dotenv("HUBSPOT_ACCESS_TOKEN")
        if not token:
            return
        try:
            import aiohttp
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=4)) as session:
                headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
                async with session.patch(
                    f"https://api.hubapi.com/crm/v3/objects/tickets/{ticket_id}",
                    headers=headers,
                    json={"properties": {"slack_thread_ts": slack_ts}},
                ) as resp:
                    if resp.status != 200:
                        logger.warning(
                            "[webhook] slack_thread_ts store failed: HTTP %s", resp.status
                        )
        except Exception:
            pass

    async def _find_ticket_by_subject(self, subject: str) -> Optional[dict]:
        """Find a HubSpot ticket by subject. Uses GET list endpoint (search has scope issues)."""
        token = self._get_env_or_dotenv("HUBSPOT_ACCESS_TOKEN")
        if not token or not subject:
            return None
        try:
            import aiohttp
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=4)) as session:
                headers = {"Authorization": f"Bearer {token}"}
                url = (
                    "https://api.hubapi.com/crm/v3/objects/tickets"
                    "?limit=50&properties=subject,slack_thread_ts,hs_pipeline_stage,hubspot_owner_id"
                    "&archived=false"
                )
                async with session.get(url, headers=headers) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.json()
                    results = data.get("results", [])
                    subject_key = self._normalise_support_subject(subject)
                    for ticket in results:
                        ticket_subject = self._normalise_support_subject(
                            ticket.get("properties", {}).get("subject", "")
                        )
                        if subject_key and ticket_subject and (
                            subject_key == ticket_subject
                            or subject_key in ticket_subject
                            or ticket_subject in subject_key
                        ):
                            return ticket

                after = data.get("paging", {}).get("next", {}).get("after")
                while after:
                    async with session.get(f"{url}&after={after}", headers=headers) as resp:
                        if resp.status != 200:
                            break
                        data = await resp.json()
                        for ticket in data.get("results", []):
                            ticket_subject = self._normalise_support_subject(
                                ticket.get("properties", {}).get("subject", "")
                            )
                            if subject_key and ticket_subject and (
                                subject_key == ticket_subject
                                or subject_key in ticket_subject
                                or ticket_subject in subject_key
                            ):
                                return ticket
                        after = data.get("paging", {}).get("next", {}).get("after")
        except Exception:
            pass
        return None

    async def _send_slack_threaded_reply(
        self, channel: str, thread_ts: str, text: str
    ) -> "SendResult":
        """Post a threaded reply to a Slack channel. Uses the gateway Slack adapter."""
        if not self.gateway_runner:
            return SendResult(success=False, error="No gateway runner")
        try:
            adapter = self.gateway_runner.adapters.get(Platform("slack"))
            if not adapter:
                return SendResult(success=False, error="Slack adapter not connected")
            metadata = {"thread_id": thread_ts}
            return await adapter.send(channel, text, metadata=metadata)
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def _update_ticket_stage(self, ticket_id: str, stage_id: str) -> None:
        """Move a HubSpot ticket to a new pipeline stage. Non-fatal."""
        token = self._get_env_or_dotenv("HUBSPOT_ACCESS_TOKEN")
        if not token:
            return
        try:
            import aiohttp
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=4)) as session:
                headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
                async with session.patch(
                    f"https://api.hubapi.com/crm/v3/objects/tickets/{ticket_id}",
                    headers=headers,
                    json={"properties": {"hs_pipeline_stage": stage_id}},
                ) as resp:
                    if resp.status != 200:
                        logger.warning(
                            "[webhook] ticket stage update failed: HTTP %s ticket=%s",
                            resp.status, ticket_id,
                        )
        except Exception:
            pass

    def _support_reasoning_unavailable(self, reason: str) -> dict:
        return {
            "status": "enrichment_unavailable",
            "evidence_status": reason,
            "evidence_supported": False,
            "read_only_findings": [
                "Support reasoning enrichment unavailable; deterministic base triage card rendered fail-open."
            ],
            "recommended_internal_action": (
                "Use the base triage card for now; rerun local read-only support reasoning before relying on evidence-backed findings."
            ),
            "clarification_question": "Local support reasoning enrichment did not complete for this intake.",
        }

    def _hubspot_support_triage_should_suppress(self, payload: dict) -> bool:
        """Return True when support reasoning marked the intake as no-Slack noise."""
        reasoning = payload.get("support_reasoning") if isinstance(payload.get("support_reasoning"), dict) else None
        if reasoning is None:
            support = payload.get("support") if isinstance(payload.get("support"), dict) else {}
            reasoning = support.get("reasoning") if isinstance(support.get("reasoning"), dict) else None
        if reasoning is None:
            reasoning = payload.get("reasoning") if isinstance(payload.get("reasoning"), dict) else None
        if not reasoning:
            return False
        status = str(reasoning.get("status") or reasoning.get("evidence_status") or "").strip().lower()
        action = str(
            reasoning.get("recommended_internal_action")
            or reasoning.get("recommended_next_action")
            or ""
        ).strip().lower()
        return status == "support_noise" or ("suppress" in action and "no slack" in action)

    def _format_hubspot_support_triage_card(self, payload: dict) -> str:
        """Build the deterministic CLCK internal support-triage card.

        This formatter is deliberately side-effect free.  It only uses the
        webhook payload supplied by ActivePieces/matcher and always states the
        MVP safety boundary: no email sent, no HubSpot write, no client Slack
        post.
        """
        def _nested(data: Any, *keys: str) -> Any:
            value = data
            for key in keys:
                if not isinstance(value, dict):
                    return None
                value = value.get(key)
            return value

        def _text(value: Any, default: str = "unknown") -> str:
            if value is None:
                return default
            if isinstance(value, (dict, list)):
                rendered = json.dumps(value, ensure_ascii=False)
            else:
                rendered = str(value)
            rendered = " ".join(rendered.split())
            return rendered if rendered else default

        def _truthy(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            if value is None:
                return False
            return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

        def _collapse_whitespace(value: str) -> str:
            return " ".join(str(value or "").split()).strip()

        def _trim_forwarded_request(value: str) -> str:
            text = str(value or "")
            text = text.replace("\r", "\n")
            # Remove common forwarded-email transport headers so cards start with the ask.
            marker = re.search(
                r"\b(?:Hello|Hi|Hey)\s+(?:[A-Za-z][A-Za-z'-]+|team|there|all)\b",
                text,
                flags=re.I,
            )
            if marker:
                text = text[marker.start():]
            else:
                text = re.sub(r"^-+\s*Forwarded message\s*-+\s*", "", text, flags=re.I)
            # Keep the newest client reply, not the whole historical thread.
            text = re.split(
                r"\s[-—]{5,}\s*From\s*:|\s+On\s+(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun|\d{1,2})\b.+?\bwrote\s*:",
                text,
                maxsplit=1,
                flags=re.I,
            )[0]
            return _collapse_whitespace(text)

        def _strip_contact_noise(value: str) -> str:
            text = str(value or "")
            # Strip obvious signature/link noise so Slack does not create detached link unfurls.
            text = re.sub(r"\bhttps?://\S+", "", text, flags=re.I)
            text = re.sub(r"\bwww\.\S+", "", text, flags=re.I)
            text = re.sub(
                r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
                "",
                text,
                flags=re.I,
            )
            text = re.sub(r"\b(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}\b", "", text)
            return _collapse_whitespace(text)

        def _clip(value: str, limit: int) -> str:
            text = _collapse_whitespace(value)
            if len(text) <= limit:
                return text
            return text[: max(0, limit - 1)].rstrip() + "…"

        def _triage_ref(value: Any) -> str:
            """Return a short stable reference for Slack thread follow-up."""
            raw = _collapse_whitespace(value if value is not None else "")
            if not raw:
                return "ST-unknown"
            digest = hashlib.sha1(raw.encode("utf-8", "ignore")).hexdigest()[:10].upper()
            return f"ST-{digest}"

        def _compact_support_summary(value: str, limit: int = 480) -> str:
            text = _strip_contact_noise(_trim_forwarded_request(value))
            return _clip(text, limit) if text else _clip(str(value or ""), limit)

        def _summarise_hubspot_change_request(value: str) -> str:
            return _compact_support_summary(value, limit=700)

        def _summarise_support_request(value: str) -> str:
            raw = str(value or "")
            text = _compact_support_summary(raw, limit=520)
            lower = text.lower()
            if "meta pixel" in lower and "hubspot" in lower:
                pixel_match = re.search(r"\bpixel\s*[-:–—]?\s*(\d{8,})\b", raw, flags=re.I)
                pixel_suffix = f" Pixel ID: {pixel_match.group(1)}." if pixel_match else ""
                return _clip(
                    "Client asked CLCK to help get the correct Meta pixel sorted in HubSpot; "
                    "the current Meta account appears connected but the pixel is wrong or cannot be added."
                    f"{pixel_suffix}",
                    520,
                )
            return _clip(text if text else raw, 520)

        gmail = payload.get("gmail") if isinstance(payload.get("gmail"), dict) else {}
        support = payload.get("support") if isinstance(payload.get("support"), dict) else {}
        matcher = {}
        sender = _text(
            gmail.get("from")
            or gmail.get("sender_email")
            or gmail.get("original_sender_email")
            or payload.get("sender_email")
        )
        original_sender = _text(
            gmail.get("original_sender_email")
            or payload.get("original_sender_email"),
            "",
        )
        # When Damien forwards, show the original client sender
        original_sender_line = ""
        if original_sender and original_sender != sender:
            original_sender_line = f" / forwarded from: {original_sender}"
        source_mailbox = _text(
            gmail.get("source_mailbox") or gmail.get("to") or payload.get("source_mailbox")
        )
        support_intake = "support@clck.com.au" in " ".join(
            [source_mailbox.lower(), _text(gmail.get("to") or payload.get("to"), "").lower()]
        )
        subject = _text(gmail.get("subject") or payload.get("subject"))
        summary = _text(
            matcher.get("request_summary_cleaned")
            or support.get("request_summary_cleaned")
            or payload.get("request_summary_cleaned")
            or support.get("summary")
            or payload.get("summary")
            or gmail.get("snippet")
            or gmail.get("body_preview"),
            "No summary supplied.",
        )

        triage_id = _triage_ref(
            payload.get("event_key")
            or payload.get("dedupe_key")
            or gmail.get("message_id")
            or gmail.get("id")
            or gmail.get("rfc822_message_id")
            or f"{sender}|{source_mailbox}|{subject}|{summary}"
        )

        match_line = "support@ intake"
        processing_hint_line = ""
        hubspot_access_needed = False  # simplified: no per-client portal matching
        hubspot_status = "HubSpot Help Desk ticket created automatically; Arlo is triaging from Slack."
        raw_request_text = " ".join(
            [
                _text(support.get("requested_action"), ""),
                _text(support.get("action"), ""),
                summary,
                subject,
            ]
        )
        requested_action = raw_request_text.lower()
        requested_emails = re.findall(
            r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", raw_request_text, flags=re.I
        )
        requested_email = requested_emails[0].lower() if requested_emails else "the supplied email address"
        activepieces_source = any(
            term in requested_action
            for term in (
                "activepieces",
                "noreply@activepieces.com",
                "activepieces.com",
            )
        )
        flow_issue_alert = bool(
            re.search(
                r'\bflow\s+(?:has\s+an\s+issue|["“][^"”]+["”]\s+has\s+an\s+issue)\b',
                raw_request_text,
                flags=re.I,
            )
        )
        activepieces_resolution_prompt = (
            "please review the issue, fix it, and mark it as resolved" in requested_action
        )
        automation_alert = activepieces_source or (flow_issue_alert and activepieces_resolution_prompt)
        support_action_request = support_intake and any(
            term in requested_action
            for term in (
                "please review",
                "please fix",
                "fix it",
                "get this fixed",
                "can you help",
                "needs attention",
                "has an issue",
                "issue detected",
                "error",
                "failed",
                "failure",
                "alert",
                "warning",
                "not working",
                "broken",
            )
        )
        system_area_patterns = (
            ("ActivePieces / automation flow", ("activepieces", "flow has an issue", "automation", "webhook", "zapier", "make.com")),
            ("Cloudflare / DNS / domain or website hosting", ("cloudflare", "dns", "ssl", "certificate", "domain", "hosting", "website down", "site down")),
            ("Google / SEO / Search Console", ("google search console", "gsc", "indexing", "crawl", "sitemap", "search console")),
            ("email / deliverability / mailbox", ("gmail", "email", "mailbox", "deliverability", "bounce", "spf", "dkim", "dmarc")),
            ("ads / tracking / analytics", ("meta pixel", "facebook pixel", "google ads", "analytics", "ga4", "tag manager", "gtm")),
            ("HubSpot", ("hubspot", "crm", "deal", "ticket", "pipeline", "workflow", "form")),
        )
        inferred_system_area = next(
            (area for area, terms in system_area_patterns if any(term in requested_action for term in terms)),
            "best-fit source system from the forwarded request",
        )
        flow_name_match = re.search(r'flow\s+(?:has an issue\s+)?["“]([^"”]+)["”]', raw_request_text, flags=re.I)
        automation_flow_name = flow_name_match.group(1).strip() if flow_name_match else "the named ActivePieces flow"
        if automation_alert:
            hubspot_status = "not applicable for ActivePieces automation repair; no HubSpot write in MVP."
        requires_hubspot = _truthy(support.get("requires_hubspot_access")) or _truthy(
            payload.get("requires_hubspot_access")
        )
        hubspot_change_terms = (
            "hubspot_change",
            "change hubspot",
            "update hubspot",
            "hubspot update",
            "edit hubspot",
            "create deal",
            "delete",
            "pipeline stage",
            "property",
            "workflow",
        )
        scope_terms = ("scope", "commercial", "pricing", "quote", "proposal", "contract")
        ticket_routing_terms = (
            "service board",
            "2nd service board",
            "second service board",
            "new service board",
            "create a ticket",
            "create ticket",
            "email-to-ticket",
            "email to ticket",
            "team email",
            "connected inbox",
            "help desk",
            "conversations inbox",
        )
        hubspot_change = any(term in requested_action for term in hubspot_change_terms)
        scope_decision = any(term in requested_action for term in scope_terms)
        ticket_routing_question = any(term in requested_action for term in ticket_routing_terms)

        issue_type = "general support triage"
        client_ask = summary
        likely_system_area = "Client support context / owner review"
        if automation_alert:
            issue_type = "ActivePieces automation failure"
            client_ask = f"Fix the ActivePieces flow issue for {automation_flow_name}."
            likely_system_area = "ActivePieces automation / CLCK operations flow"
        elif ticket_routing_question:
            issue_type = "ticket intake routing / email-to-ticket"
            client_ask = (
                f"Confirm whether emails sent to {requested_email} can create tickets "
                "in the requested service board."
            )
            likely_system_area = (
                "HubSpot Help Desk or Conversations Inbox team email channel, plus ticket "
                "pipeline/stage defaults and ticket-source automation."
            )
        elif hubspot_change:
            issue_type = "HubSpot configuration change request"
            client_ask = _summarise_hubspot_change_request(summary)
            summary = client_ask
            likely_system_area = "HubSpot CRM configuration"
        elif scope_decision:
            issue_type = "scope/commercial question"
            client_ask = summary
            likely_system_area = "CLCK commercial/scope decision"
        elif requires_hubspot:
            issue_type = "HubSpot read-only support check"
            client_ask = _summarise_support_request(summary)
            summary = client_ask
            likely_system_area = "HubSpot portal inspection"
        elif support_action_request:
            issue_type = "inferred internal support task"
            client_ask = "Review and fix the reported issue from the forwarded support request."
            likely_system_area = inferred_system_area

        def _compact_findings(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, list):
                rendered_items = [_text(item, "") for item in value]
                rendered = "; ".join(item for item in rendered_items if item)
            elif isinstance(value, dict):
                rendered_items = [f"{_text(key, '')}: {_text(val, '')}" for key, val in value.items()]
                rendered = "; ".join(item for item in rendered_items if item.strip(": "))
            else:
                rendered = _text(value, "")
            return rendered[:1200]

        reasoning = payload.get("support_reasoning") if isinstance(payload.get("support_reasoning"), dict) else None
        if reasoning is None and isinstance(support.get("reasoning"), dict):
            reasoning = support.get("reasoning")
        if reasoning is None and isinstance(payload.get("reasoning"), dict):
            reasoning = payload.get("reasoning")
        reasoning = reasoning or {}
        read_only_findings = _compact_findings(reasoning.get("read_only_findings") or reasoning.get("findings"))
        work_mode = _text(reasoning.get("work_mode"), "") if reasoning else ""
        explicit_client_reply_required = reasoning.get("client_reply_required") if reasoning else None
        client_reply_required = _truthy(explicit_client_reply_required) if explicit_client_reply_required is not None else None
        internal_plan = _compact_findings(reasoning.get("internal_plan")) if reasoning else ""
        approval_needed = _text(reasoning.get("approval_needed"), "") if reasoning else ""
        clarification_question = _text(
            reasoning.get("clarification_question") or reasoning.get("question_for_client"), ""
        )

        if reasoning:
            issue_type = _text(reasoning.get("issue_type"), issue_type)
            reasoning_client_ask = _text(reasoning.get("client_ask"), "")
            if reasoning_client_ask:
                client_ask = reasoning_client_ask
            likely_system_area = _text(reasoning.get("likely_system_area"), likely_system_area)
            reasoning_issue_type = issue_type.lower()
            if "hubspot change" in reasoning_issue_type or "hubspot configuration" in reasoning_issue_type:
                hubspot_change = True
                scope_decision = False
            if "activepieces" in reasoning_issue_type or "automation failure" in reasoning_issue_type:
                automation_alert = True
                hubspot_change = False
                scope_decision = False
                hubspot_status = "not applicable for ActivePieces automation repair; no HubSpot write in MVP."
            if "lead-gen/outbound" in reasoning_issue_type or "campaign messaging" in reasoning_issue_type:
                hubspot_change = False
                scope_decision = False
                support_action_request = False
                hubspot_status = "not applicable for lead-gen/outbound strategy work; no HubSpot write in MVP."
            if "inferred internal support task" in reasoning_issue_type:
                support_action_request = True
                hubspot_change = False
                scope_decision = False
                if not requires_hubspot:
                    hubspot_status = "not applicable unless first inspection finds HubSpot is the affected system; no writes in MVP."

        raw_owner = matcher.get("assignee_hint") or matcher.get("owner_primary") or payload.get("owner")
        if hubspot_change and not support_intake:
            owner_hint = _text(raw_owner, "unknown/manual review")
            risk_level = "HubSpot change requested; scope check required before implementation"
            if hubspot_access_needed:
                next_action = (
                    "Request/grant HubSpot portal access, then scope the requested HubSpot changes against the client service record before assigning implementation tasks."
                )
            else:
                next_action = (
                    f"{owner_hint} to action the in-scope HubSpot support tasks and flag only specific scope/input risks from the scoped findings."
                )
            draft_reply = (
                "Thanks for sending this through. I’ll check it against the HubSpot support scope, turn the in-scope items into tasks, "
                "and flag any specific inputs or scope boundaries rather than waiting on a generic approval step."
            )
        elif automation_alert:
            owner_hint = "internal_review"
            risk_level = "internal automation failure; repair task required"
            next_action = (
                f"Inspect the latest ActivePieces run(s) for {automation_flow_name}, identify the failed step/error, "
                "apply a bounded fix if safe, validate the flow, and avoid triggering downstream emails/Slack/client actions unless explicitly approved."
            )
            draft_reply = "No client reply needed; internal automation repair task."
        elif support_action_request:
            owner_hint = "internal_review"
            risk_level = "internal support task; inspect first, then act within safety rails"
            next_action = (
                f"Inspect the likely source system ({inferred_system_area}), identify the root cause from the supplied alert/request details, "
                "fix bounded low-risk internal configuration/code where safe, and ask only for specific missing access or approval before external writes, sends, client Slack posts, or risky changes."
            )
            draft_reply = "No client reply drafted yet; first inspect and report the concrete finding/action."
        elif scope_decision:
            owner_hint = "Damien"
            risk_level = "scope/commercial decision required"
            next_action = "Damien to review the commercial/scope question before a client reply is sent."
            draft_reply = (
                "Thanks for sending this through. I’ll check this properly on our side "
                "and come back with the next step."
            )
        else:
            owner_hint = "internal_review"
            risk_level = "support@ intake — Arlo is first point of triage"
            next_action = (
                "Scope this as a support@ work request. Use the sender/contact/domain and request details to identify the relevant portal/account if needed; ask only for specific missing access, account-selection, or implementation approval details."
            )
            draft_reply = ""

        def _is_unsafe_hubspot_change_boilerplate(value: str) -> bool:
            text = str(value or "")
            return bool(
                re.search(
                    r"damien\s+approv\w*|has damien approved|exact hubspot change|propose the exact change",
                    text,
                    re.I,
                )
            )

        if reasoning:
            proposed_next_action = _text(
                reasoning.get("recommended_internal_action") or reasoning.get("recommended_next_action"),
                "",
            )
            if proposed_next_action and not (
                (
                    hubspot_change
                    and _is_unsafe_hubspot_change_boilerplate(proposed_next_action)
                )
            ):
                next_action = proposed_next_action
            reasoning_status = _text(reasoning.get("status") or reasoning.get("evidence_status"), "").lower()
            evidence_supported = _truthy(reasoning.get("evidence_supported")) or reasoning_status in {
                "supported",
                "scoped",
                "evidence_supported",
                "complete",
                "ok",
            }
            reasoning_risk = _text(reasoning.get("risk_action_level") or reasoning.get("risk_level"), "")
            if reasoning_risk:
                risk_level = reasoning_risk
            if work_mode == "internal_action_plan":
                risk_level = reasoning_risk or "internal action plan; approval required before writes/sends"
            elif work_mode == "client_answer_draft":
                risk_level = reasoning_risk or "client answer draft; approval required before send"
            elif work_mode == "needs_specific_info":
                risk_level = reasoning_risk or "specific missing information/access needed"
            if evidence_supported and read_only_findings:
                if hubspot_status.lower().startswith("not applicable"):
                    pass
                elif hubspot_change and reasoning_status == "scoped":
                    hubspot_status = "portal/token found; support scope check completed; no writes in MVP."
                else:
                    hubspot_status = "portal/token found; read-only inspection completed; no writes in MVP."
                if requires_hubspot and not hubspot_change and not ticket_routing_question:
                    risk_level = "read-only findings available; approval required before HubSpot write"
            reasoning_draft = _text(
                reasoning.get("draft_client_reply") or reasoning.get("draft_reply"), ""
            )
            if client_reply_required is False or work_mode == "internal_action_plan":
                draft_reply = ""
            elif evidence_supported and reasoning_draft and (client_reply_required is not False):
                draft_reply = reasoning_draft
            elif clarification_question:
                draft_reply = f"Draft intentionally withheld: {clarification_question}"

        if automation_alert:
            clarification_question = ""
            if draft_reply.lower().startswith("draft intentionally withheld"):
                draft_reply = "No client reply needed; internal automation repair task."
            if not read_only_findings:
                read_only_findings = (
                    f"Forwarded ActivePieces alert detected for {automation_flow_name}; "
                    "treat it as an internal automation repair request, not a HubSpot/client-match question."
                )
        elif support_action_request:
            clarification_question = ""
            if not requires_hubspot:
                hubspot_status = "not applicable unless first inspection finds HubSpot is the affected system; no writes in MVP."
            if not read_only_findings:
                read_only_findings = (
                    "Forwarded support@ item looks like an instruction to investigate and fix; "
                    f"likely first system to inspect: {inferred_system_area}."
                )

        # --- DeepSeek mode-aware reasoning ---
        deepseek_mode = reasoning.get("mode", "") if reasoning else ""
        deepseek_assignee = reasoning.get("assignee", "") if reasoning else ""
        deepseek_actions = reasoning.get("actions") if reasoning and isinstance(reasoning.get("actions"), list) else []
        deepseek_draft = reasoning.get("draft_reply") if reasoning else None
        deepseek_note = reasoning.get("internal_note") if reasoning else None
        deepseek_summary = reasoning.get("summary", "") if reasoning else ""

        if deepseek_summary:
            summary = deepseek_summary

        existing_ticket_update = False
        if reasoning:
            existing_ticket_update = _truthy(reasoning.get("existing_ticket_update")) or bool(
                _text(reasoning.get("slack_thread_ts"), "")
            )
        card_title = "**CLCK support ticket update**" if existing_ticket_update else "**CLCK support triage**"
        working_session_line = (
            f"`{triage_id}` · Update on an existing HubSpot ticket; posted into the existing Slack working thread. Reply with `@Arlo` plus the next bounded action/approval."
            if existing_ticket_update
            else f"`{triage_id}` · Reply in this thread with `@Arlo` plus new facts/approval; this thread becomes the working session for this card."
        )
        continuation_line = "- Continuation: existing HubSpot ticket update; not a new support request." if existing_ticket_update else ""

        if reasoning:
            issue_type = _text(reasoning.get("issue_type"), issue_type)
            reasoning_client_ask = _text(reasoning.get("client_ask"), "")
            if reasoning_client_ask:
                client_ask = reasoning_client_ask

        # --- Assignee hint ---
        if deepseek_assignee == "benson":
            owner_hint = "Benson / team@clck.com.au"
        elif deepseek_assignee == "marinda":
            owner_hint = "Marinda"
        else:
            owner_hint = "internal_review"

        # --- Legacy fallback: old reasoning format without mode field ---
        # When reasoning is present but has old field names (from tests or old worker),
        # render the legacy card format for backward compatibility.
        has_legacy_fields = reasoning and (
            reasoning.get("read_only_findings")
            or reasoning.get("internal_plan")
            or reasoning.get("recommended_internal_action")
            or reasoning.get("work_mode")
        ) and not deepseek_mode

        if has_legacy_fields:
            # Legacy card rendering — preserve old format for backward compat
            triage_title = "**CLCK HubSpot support ticket update**" if existing_ticket_update and "hubspot" in issue_type.lower() else card_title
            lines = [
                triage_title,
                working_session_line,
                "",
                "**1) Request**",
                f"- Request summary: {_clip(summary, 700)}",
                f"- Client ask: {_clip(client_ask, 420)}",
                f"- Issue type: {_clip(issue_type, 160)}",
                f"- Likely system area: {_clip(likely_system_area, 240)}",
                "",
                "**2) Routing / context**",
                f"- Source: support@clck.com.au intake",
                f"- HubSpot status: {_clip(hubspot_status, 260)}",
            ]
            if continuation_line:
                lines.append(continuation_line)
            lines.extend([
                f"- Owner/assignee hint: {_clip(owner_hint, 120)}",
                f"- Risk/action level: {_clip(risk_level, 260)}",
                f"- Sender/source/subject: {_clip(sender + original_sender_line, 200)} / {_clip(source_mailbox, 120)} / {_clip(subject, 180)}",
            ])
            section_number = 3
            if read_only_findings:
                lines.extend([
                    "",
                    f"**{section_number}) Findings**",
                    f"- Read-only findings: {_clip(read_only_findings, 760)}",
                ])
                section_number += 1
            if clarification_question:
                lines.extend([
                    "",
                    f"**{section_number}) Clarification needed**",
                    f"- Clarification question: {_clip(clarification_question, 360)}",
                ])
                section_number += 1
            lines.extend([
                "",
                f"**{section_number}) Recommended action**",
                f"- Recommended internal next action: {_clip(next_action, 620)}",
            ])
            if internal_plan:
                lines.append(f"- Internal plan: {_clip(internal_plan, 760)}")
            if approval_needed:
                lines.append(f"- Approval needed: {_clip(approval_needed, 420)}")
            show_draft_reply = bool(draft_reply) and (client_reply_required is not False) and work_mode != "internal_action_plan"
            if show_draft_reply:
                lines.append(f"- Draft client reply: {_clip(draft_reply, 500)}")
            lines.append("- Safety: no email sent; no HubSpot write; no client Slack post.")
            return "\n".join(lines)

        # --- Build card ---
        if deepseek_mode:
            # Mode-aware card (DeepSeek classified)
            lines = [
                card_title,
                working_session_line,
                "",
                "**Request**",
                f"- Summary: {_clip(summary, 700)}",
                f"- Issue type: {_clip(issue_type, 160)}",
                "",
                "**Context**",
                f"- Source: support@clck.com.au intake",
                f"- Client: {_clip(original_sender or sender, 160)}",
                f"- HubSpot ticket: {_clip(reasoning.get('ticket_url', 'auto-created by Help Desk'), 120)}",
            ]
            if continuation_line:
                lines.append(continuation_line)
            lines.extend([
                f"- Suggested assignee: {_clip(owner_hint, 120)}",
                f"- Sender/subject: {_clip(sender + original_sender_line, 200)} / {_clip(subject, 180)}",
            ])

            if deepseek_mode == "action_plan" and deepseek_actions:
                lines.append("")
                lines.append("**Arlo can:**")
                for i, action in enumerate(deepseek_actions, 1):
                    lines.append(f"{i}. {_clip(action, 380)}")
                lines.append("")
                lines.append("_Reply `approved` and Arlo will execute these actions. Reply with changes to redirect._")
            elif deepseek_mode == "draft_reply" and deepseek_draft:
                lines.append("")
                lines.append("**Draft reply:**")
                lines.append(f"{_clip(deepseek_draft, 900)}")
                lines.append("")
                lines.append("_Reply `approved` and Arlo will send this reply. Reply with edits to revise._")
            elif deepseek_mode == "summary_only" and deepseek_note:
                lines.append("")
                lines.append("**Note:**")
                lines.append(f"{_clip(deepseek_note, 500)}")
                lines.append("")
                lines.append("_No action needed — FYI only._")
            else:
                lines.append("")
                lines.append("**Recommended action**")
                lines.append(f"- {_clip(next_action, 620)}")
        else:
            # Deterministic fallback (no DeepSeek reasoning or legacy mode)
            lines = [
                card_title,
                working_session_line,
                "",
                "**1) Request**",
                f"- Request summary: {_clip(summary, 700)}",
                f"- Client ask: {_clip(client_ask, 420)}",
                f"- Issue type: {_clip(issue_type, 160)}",
                f"- Likely system area: {_clip(likely_system_area, 240)}",
                "",
                "**2) Routing / context**",
                f"- Source: support@clck.com.au intake",
                f"- HubSpot status: {_clip(hubspot_status, 260)}",
            ]
            if continuation_line:
                lines.append(continuation_line)
            lines.extend([
                f"- Owner/assignee hint: {_clip(owner_hint, 120)}",
                f"- Risk/action level: {_clip(risk_level, 260)}",
                f"- Sender/source/subject: {_clip(sender + original_sender_line, 200)} / {_clip(source_mailbox, 120)} / {_clip(subject, 180)}",
                "",
                "**3) Recommended action**",
                f"- Recommended internal next action: {_clip(next_action, 620)}",
            ])
            if draft_reply:
                lines.append(f"- Draft client reply: {_clip(draft_reply, 500)}")

        lines.append("")
        lines.append("- Safety: no email sent; no HubSpot write; no client Slack post.")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Response delivery
    # ------------------------------------------------------------------

    async def _direct_deliver(
        self, content: str, delivery: dict
    ) -> SendResult:
        """Deliver *content* directly without invoking the agent.

        Used by ``deliver_only`` routes: the rendered template becomes the
        literal message body, and we dispatch to the same delivery helpers
        that the agent-mode ``send()`` flow uses.  All target types that
        work in agent mode work here — Telegram, Discord, Slack, GitHub
        PR comments, etc.
        """
        deliver_type = delivery.get("deliver", "log")

        if deliver_type == "log":
            # Shouldn't reach here — startup validation rejects deliver_only
            # with deliver=log — but guard defensively.
            logger.info("[webhook] direct-deliver log-only: %s", content[:200])
            return SendResult(success=True)

        if deliver_type == "github_comment":
            return await self._deliver_github_comment(content, delivery)

        # Fall through to the cross-platform dispatcher, which validates the
        # target name and routes via the gateway runner.
        return await self._deliver_cross_platform(
            deliver_type, content, delivery
        )

    async def _deliver_github_comment(
        self, content: str, delivery: dict
    ) -> SendResult:
        """Post agent response as a GitHub PR/issue comment via ``gh`` CLI."""
        extra = delivery.get("deliver_extra", {})
        repo = extra.get("repo", "")
        pr_number = extra.get("pr_number", "")

        if not repo or not pr_number:
            logger.error(
                "[webhook] github_comment delivery missing repo or pr_number"
            )
            return SendResult(
                success=False, error="Missing repo or pr_number"
            )

        try:
            result = subprocess.run(
                [
                    "gh",
                    "pr",
                    "comment",
                    str(pr_number),
                    "--repo",
                    repo,
                    "--body",
                    content,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                logger.info(
                    "[webhook] Posted comment on %s#%s", repo, pr_number
                )
                return SendResult(success=True)
            else:
                logger.error(
                    "[webhook] gh pr comment failed: %s", result.stderr
                )
                return SendResult(success=False, error=result.stderr)
        except FileNotFoundError:
            logger.error(
                "[webhook] 'gh' CLI not found — install GitHub CLI for "
                "github_comment delivery"
            )
            return SendResult(
                success=False, error="gh CLI not installed"
            )
        except Exception as e:
            logger.error("[webhook] github_comment delivery error: %s", e)
            return SendResult(success=False, error=str(e))

    async def _deliver_cross_platform(
        self, platform_name: str, content: str, delivery: dict
    ) -> SendResult:
        """Route response to another platform (telegram, discord, etc.)."""
        if not self.gateway_runner:
            return SendResult(
                success=False,
                error="No gateway runner for cross-platform delivery",
            )

        try:
            target_platform = Platform(platform_name)
        except ValueError:
            return SendResult(
                success=False, error=f"Unknown platform: {platform_name}"
            )

        adapter = self.gateway_runner.adapters.get(target_platform)
        if not adapter:
            return SendResult(
                success=False,
                error=f"Platform {platform_name} not connected",
            )

        # Use home channel if no specific chat_id in deliver_extra
        extra = delivery.get("deliver_extra", {})
        chat_id = extra.get("chat_id", "")
        if not chat_id:
            home = self.gateway_runner.config.get_home_channel(target_platform)
            if home:
                chat_id = home.chat_id
            else:
                return SendResult(
                    success=False,
                    error=f"No chat_id or home channel for {platform_name}",
                )

        # Pass thread_id from deliver_extra so Telegram forum topics work
        metadata = None
        thread_id = (extra.get("message_thread_id") or extra.get("thread_id") or "").strip()
        # Skip unresolved placeholders and garbage values that would break Slack.
        if thread_id and not thread_id.startswith("{"):
            metadata = {"thread_id": thread_id}

        return await adapter.send(chat_id, content, metadata=metadata)
