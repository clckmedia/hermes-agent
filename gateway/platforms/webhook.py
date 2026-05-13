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
import contextlib
import hashlib
import hmac
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path
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

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8644
_INSECURE_NO_AUTH = "INSECURE_NO_AUTH"
_DYNAMIC_ROUTES_FILENAME = "webhook_subscriptions.json"
_SUPPORT_REASONING_SCRIPT = Path.home() / ".hermes" / "scripts" / "clck_support_triage_reasoning.py"
_SUPPORT_REASONING_TIMEOUT_SECONDS = 5.0


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

        # Cross-platform delivery — any platform with a gateway adapter
        if self.gateway_runner and deliver_type in (
            "telegram",
            "discord",
            "slack",
            "signal",
            "sms",
            "whatsapp",
            "matrix",
            "mattermost",
            "homeassistant",
            "email",
            "dingtalk",
            "feishu",
            "wecom",
            "wecom_callback",
            "weixin",
            "bluebubbles",
            "qqbot",
            "zulip",
        ):
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
            # Merge: static routes take precedence over dynamic ones
            self._dynamic_routes = {
                k: v for k, v in data.items()
                if k not in self._static_routes
            }
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

        # Validate HMAC signature FIRST (skip for INSECURE_NO_AUTH testing mode)
        secret = route_config.get("secret", self._global_secret)
        if secret and secret != _INSECURE_NO_AUTH:
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
            request.headers.get("X-Request-ID", str(int(time.time() * 1000))),
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
        """Validate webhook signature (GitHub, GitLab, generic HMAC-SHA256)."""
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

    async def _enrich_hubspot_support_triage_payload(
        self, payload: dict, route_config: Optional[dict] = None
    ) -> dict:
        """Attach local read-only support reasoning to CLCK support triage payloads.

        This is fail-open by design: if the local worker is missing, times out,
        has no credentials, or errors, the deterministic base card still renders
        with a safe unavailable marker.  The worker is a local helper and must
        not deliver messages or write to HubSpot.
        """
        if payload.get("event_type") != "hubspot_support_triage":
            return payload
        if "support_reasoning" in payload:
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
            reasoning = await self._run_support_reasoning_worker(payload, timeout=timeout)
            if not isinstance(reasoning, dict):
                raise ValueError("support reasoning worker returned non-object JSON")
            enriched["support_reasoning"] = reasoning
        except asyncio.TimeoutError:
            logger.warning(
                "[webhook] support-triage reasoning unavailable: timeout after %.1fs",
                timeout,
            )
            enriched["support_reasoning"] = self._support_reasoning_unavailable("timeout")
        except Exception as exc:
            logger.warning(
                "[webhook] support-triage reasoning unavailable: %s",
                type(exc).__name__,
            )
            enriched["support_reasoning"] = self._support_reasoning_unavailable("error")
        return enriched

    async def _run_support_reasoning_worker(self, payload: dict, *, timeout: float) -> dict:
        """Run the local support reasoning helper and return its JSON object."""
        if not _SUPPORT_REASONING_SCRIPT.exists():
            raise FileNotFoundError("support reasoning worker not found")

        raw_payload = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(_SUPPORT_REASONING_SCRIPT),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(raw_payload),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            process.kill()
            with contextlib.suppress(Exception):
                await process.communicate()
            raise

        if process.returncode != 0:
            # Do not log stderr content: helper/library errors can include local paths
            # or credential context.  The card only needs a safe unavailable status.
            raise RuntimeError(f"support reasoning worker exited {process.returncode}")
        if stderr:
            logger.debug("[webhook] support-triage reasoning worker wrote stderr")
        decoded = stdout.decode("utf-8").strip()
        return json.loads(decoded)

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
        matcher = payload.get("matcher") if isinstance(payload.get("matcher"), dict) else None
        if matcher is None:
            matcher = payload.get("match") if isinstance(payload.get("match"), dict) else None
        teable_match = _nested(payload, "teable", "client_match")
        if matcher is None and isinstance(teable_match, dict):
            matcher = teable_match
        matcher = matcher or {}

        sender = _text(
            gmail.get("from")
            or gmail.get("sender_email")
            or gmail.get("original_sender_email")
            or payload.get("sender_email")
        )
        source_mailbox = _text(
            gmail.get("source_mailbox") or gmail.get("to") or payload.get("source_mailbox")
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

        decision = _text(matcher.get("decision") or matcher.get("match_result"), "fallback")
        reason = _text(matcher.get("reason") or matcher.get("match_reason"), "no_safe_match")
        client_name = matcher.get("client_name") or _nested(matcher, "client", "name")
        matched = decision == "route_client" and bool(client_name)
        if matched:
            match_line = f"matched client: {_text(client_name)} ({reason})"
        else:
            match_line = f"fallback/{reason}"
        processing_hint_line = ""
        explicit_hint_used = _truthy(matcher.get("explicit_client_hint_trusted")) and matcher.get("explicit_client_hint_matched_client")
        if explicit_hint_used:
            hint_client_name = (
                _nested(matcher, "explicit_client_hint_matched_client", "client_name")
                or matcher.get("client_name")
                or client_name
            )
            processing_hint_line = f"Processing hint: CLCK-forwarded as {_text(hint_client_name)}"

        portal_id = (
            matcher.get("portal_id")
            or matcher.get("hubspot_portal_id")
            or _nested(payload, "hubspot", "portal_id")
        )
        hubspot_access_status = _text(
            matcher.get("hubspot_access_status") or _nested(payload, "hubspot", "access_status"),
            "",
        ).lower()
        hubspot_access_needed = _truthy(matcher.get("hubspot_access_needed")) or _truthy(
            _nested(payload, "hubspot", "access_needed")
        )
        token_found = bool(
            matcher.get("hubspot_token_reference")
            or matcher.get("token_reference")
            or _truthy(matcher.get("hubspot_token_reference_present"))
            or _truthy(_nested(payload, "hubspot", "token_found"))
        )
        if hubspot_access_status == "connected" or (portal_id and token_found):
            hubspot_status = "portal/token found; read-only inspection skipped; no writes in MVP."
        elif hubspot_access_status == "access_needed" or hubspot_access_needed:
            hubspot_status = "support-active route; HubSpot access needed before inspection or implementation; no writes in MVP."
        elif hubspot_access_status == "not_applicable":
            hubspot_status = "not applicable for HubSpot support action; read-only inspection skipped; no writes in MVP."
        elif portal_id:
            hubspot_status = "portal found; token not found; read-only inspection skipped; no writes in MVP."
        else:
            hubspot_status = "portal/token not found; read-only inspection skipped; no writes in MVP."

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
        if ticket_routing_question:
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

        def _hubspot_change_has_actionable_detail(value: str) -> bool:
            text = _summarise_hubspot_change_request(value).lower()
            words = re.findall(r"[a-z0-9]+", text)
            if len(words) < 12:
                return False
            action_terms = (
                "add ",
                "remove",
                "hide",
                "restrict",
                "field",
                "form",
                "notification",
                "association",
                "label",
                "permission",
                "commerce",
                "pipeline",
                "property",
                "workflow",
                "deal",
                "ticket",
                "pixel",
                "event landing",
                "humanitix",
                "humantix",
            )
            return any(term in text for term in action_terms)

        generic_change_clarification = bool(
            clarification_question
            and matched
            and hubspot_change
            and _hubspot_change_has_actionable_detail(summary or client_ask)
            and re.search(r"what exact hubspot change|has damien approved|exact hubspot change", clarification_question, re.I)
        )
        if generic_change_clarification:
            clarification_question = ""

        raw_owner = matcher.get("assignee_hint") or matcher.get("owner_primary") or payload.get("owner")
        if hubspot_change:
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
        elif scope_decision:
            owner_hint = "Damien"
            risk_level = "scope/commercial decision required"
            next_action = "Damien to review the commercial/scope question before a client reply is sent."
            draft_reply = (
                "Thanks for sending this through. I’ll check this properly on our side "
                "and come back with the next step."
            )
        elif not matched:
            owner_hint = "unknown/manual review"
            risk_level = "approval required"
            next_action = "Manually confirm the client/route before replying."
            draft_reply = (
                "Thanks for this. I’m checking where this should sit on our side and "
                "will come back to you shortly."
            )
        elif ticket_routing_question:
            owner_hint = _text(raw_owner, "unknown/manual review")
            risk_level = "HubSpot routing/configuration question; read-only check before reply"
            if hubspot_access_needed:
                next_action = (
                    "Request/grant HubSpot portal access, then check whether "
                    f"{requested_email} is connected/forwarded as a HubSpot team email or help desk channel "
                    "and which ticket pipeline/stage it creates into."
                )
                draft_reply = (
                    "Draft intentionally withheld: HubSpot access is needed before confirming whether this "
                    "email address can feed the requested service board."
                )
            else:
                next_action = (
                    "Inspect HubSpot read-only: Help Desk/Conversations channel accounts for "
                    f"{requested_email}, then Tickets > Pipelines/Automate for the target service-board "
                    "pipeline and default stage. If the channel is absent, propose connecting/forwarding the "
                    "address before Damien approves any setup change."
                )
                draft_reply = (
                    "Draft intentionally withheld: first confirm whether the email address is already "
                    "connected and which ticket pipeline/stage it feeds, so we don’t promise a route that may need setup."
                )
        elif requires_hubspot:
            owner_hint = _text(raw_owner, "unknown/manual review")
            risk_level = "HubSpot read-only inspection needed"
            if hubspot_access_needed:
                next_action = "Request/grant HubSpot portal access before inspection; keep triage and reply drafting internal."
            else:
                next_action = "Inspect HubSpot read-only, then post findings and a draft for approval."
            draft_reply = (
                "Thanks for this. I’ll take a look in HubSpot and come back with the "
                "next step shortly."
            )
        else:
            owner_hint = _text(raw_owner, "unknown/manual review")
            risk_level = "safe question/draft only"
            next_action = "Owner to review the draft and approve the reply before anything is sent."
            draft_reply = "Thanks for this. I’ll take a look and come back with the next step shortly."

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
                hubspot_change
                and matched
                and _is_unsafe_hubspot_change_boilerplate(proposed_next_action)
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
            if evidence_supported and read_only_findings:
                if hubspot_change and reasoning_status == "scoped":
                    hubspot_status = "portal/token found; support scope check completed; no writes in MVP."
                else:
                    hubspot_status = "portal/token found; read-only inspection completed; no writes in MVP."
                if requires_hubspot and not hubspot_change and not ticket_routing_question:
                    risk_level = "read-only findings available; approval required before HubSpot write"
            reasoning_draft = _text(
                reasoning.get("draft_client_reply") or reasoning.get("draft_reply"), ""
            )
            if evidence_supported and reasoning_draft:
                draft_reply = reasoning_draft
            elif clarification_question:
                draft_reply = f"Draft intentionally withheld: {clarification_question}"

        lines = [
            "**CLCK HubSpot support triage**",
            f"`{triage_id}` · Reply in this thread with `@Arlo` plus new facts/approval; this thread becomes the working session for this card.",
            "",
            "**1) Request**",
            f"- Request summary: {_clip(summary, 700)}",
            f"- Client ask: {_clip(client_ask, 420)}",
            f"- Issue type: {_clip(issue_type, 160)}",
            f"- Likely system area: {_clip(likely_system_area, 240)}",
            "",
            "**2) Routing / context**",
            f"- Client match: {_clip(match_line, 180)}",
            f"- HubSpot status: {_clip(hubspot_status, 260)}",
            f"- Owner/assignee hint: {_clip(owner_hint, 120)}",
            f"- Risk/action level: {_clip(risk_level, 260)}",
            f"- Sender/source/subject: {_clip(sender, 120)} / {_clip(source_mailbox, 120)} / {_clip(subject, 180)}",
        ]
        if processing_hint_line:
            lines.append(f"- {_clip(processing_hint_line, 220)}")
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
            f"- Draft client reply: {_clip(draft_reply, 500)}",
            "- Safety: no email sent; no HubSpot write; no client Slack post.",
        ])
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
        thread_id = extra.get("message_thread_id") or extra.get("thread_id")
        if thread_id:
            metadata = {"thread_id": thread_id}

        return await adapter.send(chat_id, content, metadata=metadata)
