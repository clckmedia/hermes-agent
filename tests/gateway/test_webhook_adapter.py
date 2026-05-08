"""Unit tests for the generic webhook platform adapter.

Covers:
- HMAC signature validation (GitHub, GitLab, generic)
- Prompt rendering with dot-notation template variables
- Event type filtering
- HTTP handler behaviour (404, 202, health)
- Idempotency cache (duplicate delivery IDs)
- Rate limiting (fixed-window, per route)
- Body size limits
- INSECURE_NO_AUTH bypass
- Session isolation for concurrent webhooks
- Delivery info cleanup after send()
- connect / disconnect lifecycle
"""

import asyncio
import hashlib
import hmac
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SendResult
from gateway.platforms.webhook import (
    WebhookAdapter,
    _INSECURE_NO_AUTH,
    check_webhook_requirements,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    routes=None,
    secret="",
    rate_limit=30,
    max_body_bytes=1_048_576,
    host="0.0.0.0",
    port=0,  # let OS pick a free port in tests
):
    """Build a PlatformConfig suitable for WebhookAdapter."""
    extra = {
        "host": host,
        "port": port,
        "routes": routes or {},
        "rate_limit": rate_limit,
        "max_body_bytes": max_body_bytes,
    }
    if secret:
        extra["secret"] = secret
    return PlatformConfig(enabled=True, extra=extra)


def _make_adapter(routes=None, **kwargs):
    """Create a WebhookAdapter with sensible defaults for testing."""
    config = _make_config(routes=routes, **kwargs)
    return WebhookAdapter(config)


def _create_app(adapter: WebhookAdapter) -> web.Application:
    """Build the aiohttp Application from the adapter (without starting a full server)."""
    app = web.Application()
    app.router.add_get("/health", adapter._handle_health)
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    return app


def _mock_request(headers=None, body=b"", content_length=None, match_info=None):
    """Build a lightweight mock aiohttp request for non-HTTP tests."""
    req = MagicMock()
    req.headers = headers or {}
    req.content_length = content_length if content_length is not None else len(body)
    req.match_info = match_info or {}
    req.method = "POST"

    async def _read():
        return body

    req.read = _read
    return req


def _github_signature(body: bytes, secret: str) -> str:
    """Compute X-Hub-Signature-256 for *body* using *secret*."""
    return "sha256=" + hmac.new(
        secret.encode(), body, hashlib.sha256
    ).hexdigest()


def _generic_signature(body: bytes, secret: str) -> str:
    """Compute X-Webhook-Signature (plain HMAC-SHA256 hex) for *body*."""
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


# ===================================================================
# Signature validation
# ===================================================================


class TestValidateSignature:
    """Tests for WebhookAdapter._validate_signature."""

    def test_validate_github_signature_valid(self):
        """Valid X-Hub-Signature-256 is accepted."""
        adapter = _make_adapter()
        body = b'{"action": "opened"}'
        secret = "webhook-secret-42"
        sig = _github_signature(body, secret)
        req = _mock_request(headers={"X-Hub-Signature-256": sig})
        assert adapter._validate_signature(req, body, secret) is True

    def test_validate_github_signature_invalid(self):
        """Wrong X-Hub-Signature-256 is rejected."""
        adapter = _make_adapter()
        body = b'{"action": "opened"}'
        secret = "webhook-secret-42"
        req = _mock_request(headers={"X-Hub-Signature-256": "sha256=deadbeef"})
        assert adapter._validate_signature(req, body, secret) is False

    def test_validate_gitlab_token(self):
        """GitLab plain-token match via X-Gitlab-Token."""
        adapter = _make_adapter()
        secret = "gl-token-value"
        req = _mock_request(headers={"X-Gitlab-Token": secret})
        assert adapter._validate_signature(req, b"{}", secret) is True

    def test_validate_gitlab_token_wrong(self):
        """Wrong X-Gitlab-Token is rejected."""
        adapter = _make_adapter()
        req = _mock_request(headers={"X-Gitlab-Token": "wrong"})
        assert adapter._validate_signature(req, b"{}", "correct") is False

    def test_validate_no_signature_with_secret_rejects(self):
        """Secret configured but no recognised signature header → reject."""
        adapter = _make_adapter()
        req = _mock_request(headers={})  # no sig headers at all
        assert adapter._validate_signature(req, b"{}", "my-secret") is False

    def test_validate_no_secret_allows_all(self):
        """When the secret is empty/falsy, the validator is never even called
        by the handler (secret check is 'if secret and secret != _INSECURE...').
        Verify that an empty secret isn't accidentally passed to the validator."""
        # This tests the semantics: empty secret means skip validation entirely.
        # The handler code does: if secret and secret != _INSECURE_NO_AUTH: validate
        # So with an empty secret, _validate_signature is never reached.
        # We just verify the code path is correct by constructing an adapter
        # with no secret and confirming the route config resolves to "".
        adapter = _make_adapter(
            routes={"test": {"prompt": "hello"}},
            secret="",
        )
        # The route has no secret, global secret is empty
        route_secret = adapter._routes["test"].get("secret", adapter._global_secret)
        assert not route_secret  # empty → validation is skipped in handler

    def test_validate_generic_signature_valid(self):
        """Valid X-Webhook-Signature (generic HMAC-SHA256 hex) is accepted."""
        adapter = _make_adapter()
        body = b'{"event": "push"}'
        secret = "generic-secret"
        sig = _generic_signature(body, secret)
        req = _mock_request(headers={"X-Webhook-Signature": sig})
        assert adapter._validate_signature(req, body, secret) is True


# ===================================================================
# Prompt rendering
# ===================================================================


class TestRenderPrompt:
    """Tests for WebhookAdapter._render_prompt."""

    def test_render_prompt_dot_notation(self):
        """Dot-notation {pull_request.title} resolves nested keys."""
        adapter = _make_adapter()
        payload = {"pull_request": {"title": "Fix bug", "number": 42}}
        result = adapter._render_prompt(
            "PR #{pull_request.number}: {pull_request.title}",
            payload,
            "pull_request",
            "github",
        )
        assert result == "PR #42: Fix bug"

    def test_render_prompt_missing_key_preserved(self):
        """{nonexistent} is left as-is when key doesn't exist in payload."""
        adapter = _make_adapter()
        result = adapter._render_prompt(
            "Hello {nonexistent}!",
            {"action": "opened"},
            "push",
            "test",
        )
        assert "{nonexistent}" in result

    def test_render_prompt_no_template_dumps_json(self):
        """Empty template → JSON dump fallback with event/route context."""
        adapter = _make_adapter()
        payload = {"key": "value"}
        result = adapter._render_prompt("", payload, "push", "my-route")
        assert "push" in result
        assert "my-route" in result
        assert "key" in result


# ===================================================================
# Delivery extra rendering
# ===================================================================


class TestRenderDeliveryExtra:
    def test_render_delivery_extra_templates(self):
        """String values in deliver_extra are rendered with payload data."""
        adapter = _make_adapter()
        extra = {"repo": "{repository.full_name}", "pr_number": "{number}", "static": 42}
        payload = {"repository": {"full_name": "org/repo"}, "number": 7}
        result = adapter._render_delivery_extra(extra, payload)
        assert result["repo"] == "org/repo"
        assert result["pr_number"] == "7"
        assert result["static"] == 42  # non-string left as-is


# ===================================================================
# Event filtering
# ===================================================================


class TestEventFilter:
    """Tests for event type filtering in _handle_webhook."""

    @pytest.mark.asyncio
    async def test_event_filter_accepts_matching(self):
        """Matching event type passes through."""
        routes = {
            "gh": {
                "secret": _INSECURE_NO_AUTH,
                "events": ["pull_request"],
                "prompt": "PR: {action}",
            }
        }
        adapter = _make_adapter(routes=routes)
        # Stub handle_message to avoid running the agent
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/gh",
                json={"action": "opened"},
                headers={"X-GitHub-Event": "pull_request"},
            )
            assert resp.status == 202

    @pytest.mark.asyncio
    async def test_event_filter_rejects_non_matching(self):
        """Non-matching event type returns 200 with status=ignored."""
        routes = {
            "gh": {
                "secret": _INSECURE_NO_AUTH,
                "events": ["pull_request"],
                "prompt": "test",
            }
        }
        adapter = _make_adapter(routes=routes)

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/gh",
                json={"action": "opened"},
                headers={"X-GitHub-Event": "push"},
            )
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "ignored"

    @pytest.mark.asyncio
    async def test_event_filter_empty_allows_all(self):
        """No events list → accept any event type."""
        routes = {
            "all": {
                "secret": _INSECURE_NO_AUTH,
                "prompt": "got it",
            }
        }
        adapter = _make_adapter(routes=routes)
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/all",
                json={"action": "any"},
                headers={"X-GitHub-Event": "whatever"},
            )
            assert resp.status == 202


# ===================================================================
# HTTP handling
# ===================================================================


class TestHTTPHandling:

    @pytest.mark.asyncio
    async def test_unknown_route_returns_404(self):
        """POST to an unknown route returns 404."""
        adapter = _make_adapter(routes={"real": {"secret": _INSECURE_NO_AUTH, "prompt": "x"}})
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/webhooks/nonexistent", json={"a": 1})
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_webhook_handler_returns_202(self):
        """Valid request returns 202 Accepted."""
        routes = {"test": {"secret": _INSECURE_NO_AUTH, "prompt": "hi"}}
        adapter = _make_adapter(routes=routes)
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/webhooks/test", json={"data": "value"})
            assert resp.status == 202
            data = await resp.json()
            assert data["status"] == "accepted"
            assert data["route"] == "test"

    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """GET /health returns 200 with status=ok."""
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/health")
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "ok"
            assert data["platform"] == "webhook"

    @pytest.mark.asyncio
    async def test_connect_starts_server(self):
        """connect() starts the HTTP listener and marks adapter as connected."""
        routes = {"r1": {"secret": _INSECURE_NO_AUTH, "prompt": "x"}}
        adapter = _make_adapter(routes=routes, port=0)
        # Use port 0 — the OS picks a free port, but aiohttp requires a real bind.
        # We just test that the method completes and marks connected.
        # Need to mock TCPSite to avoid actual binding.
        with patch("gateway.platforms.webhook.web.AppRunner") as MockRunner, \
             patch("gateway.platforms.webhook.web.TCPSite") as MockSite:
            mock_runner_inst = AsyncMock()
            MockRunner.return_value = mock_runner_inst
            mock_site_inst = AsyncMock()
            MockSite.return_value = mock_site_inst

            result = await adapter.connect()
            assert result is True
            assert adapter.is_connected
            mock_runner_inst.setup.assert_awaited_once()
            mock_site_inst.start.assert_awaited_once()

        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect_cleans_up(self):
        """disconnect() stops the server and marks adapter disconnected."""
        adapter = _make_adapter()
        # Simulate a runner that was previously set up
        mock_runner = AsyncMock()
        adapter._runner = mock_runner
        adapter._running = True

        await adapter.disconnect()
        mock_runner.cleanup.assert_awaited_once()
        assert adapter._runner is None
        assert not adapter.is_connected


# ===================================================================
# Idempotency
# ===================================================================


class TestIdempotency:

    @pytest.mark.asyncio
    async def test_duplicate_delivery_id_returns_200(self):
        """Second request with same delivery ID returns 200 duplicate."""
        routes = {"idem": {"secret": _INSECURE_NO_AUTH, "prompt": "test"}}
        adapter = _make_adapter(routes=routes)
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            headers = {"X-GitHub-Delivery": "delivery-123"}
            resp1 = await cli.post("/webhooks/idem", json={"a": 1}, headers=headers)
            assert resp1.status == 202

            resp2 = await cli.post("/webhooks/idem", json={"a": 1}, headers=headers)
            assert resp2.status == 200
            data = await resp2.json()
            assert data["status"] == "duplicate"

    @pytest.mark.asyncio
    async def test_expired_delivery_id_allows_reprocess(self):
        """After TTL expires, the same delivery ID is accepted again."""
        routes = {"idem": {"secret": _INSECURE_NO_AUTH, "prompt": "test"}}
        adapter = _make_adapter(routes=routes)
        adapter._idempotency_ttl = 1  # 1 second TTL for test speed
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            headers = {"X-GitHub-Delivery": "delivery-456"}

            resp1 = await cli.post("/webhooks/idem", json={"x": 1}, headers=headers)
            assert resp1.status == 202

            # Backdate the cache entry so it appears expired
            adapter._seen_deliveries["delivery-456"] = time.time() - 3700

            resp2 = await cli.post("/webhooks/idem", json={"x": 1}, headers=headers)
            assert resp2.status == 202  # re-accepted


# ===================================================================
# Rate limiting
# ===================================================================


class TestRateLimiting:

    @pytest.mark.asyncio
    async def test_rate_limit_rejects_excess(self):
        """Exceeding the rate limit returns 429."""
        routes = {"limited": {"secret": _INSECURE_NO_AUTH, "prompt": "test"}}
        adapter = _make_adapter(routes=routes, rate_limit=2)
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            # Two requests within limit
            for i in range(2):
                resp = await cli.post(
                    "/webhooks/limited",
                    json={"n": i},
                    headers={"X-GitHub-Delivery": f"d-{i}"},
                )
                assert resp.status == 202, f"Request {i} should be accepted"

            # Third request should be rate-limited
            resp = await cli.post(
                "/webhooks/limited",
                json={"n": 99},
                headers={"X-GitHub-Delivery": "d-99"},
            )
            assert resp.status == 429

    @pytest.mark.asyncio
    async def test_rate_limit_window_resets(self):
        """After the 60-second window passes, requests are allowed again."""
        routes = {"limited": {"secret": _INSECURE_NO_AUTH, "prompt": "test"}}
        adapter = _make_adapter(routes=routes, rate_limit=1)
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/limited",
                json={"n": 1},
                headers={"X-GitHub-Delivery": "d-a"},
            )
            assert resp.status == 202

            # Backdate all rate-limit timestamps to > 60 seconds ago
            adapter._rate_counts["limited"] = [time.time() - 120]

            resp = await cli.post(
                "/webhooks/limited",
                json={"n": 2},
                headers={"X-GitHub-Delivery": "d-b"},
            )
            assert resp.status == 202  # allowed again


# ===================================================================
# Body size limit
# ===================================================================


class TestBodySize:

    @pytest.mark.asyncio
    async def test_oversized_payload_rejected(self):
        """Content-Length > max_body_bytes returns 413."""
        routes = {"big": {"secret": _INSECURE_NO_AUTH, "prompt": "test"}}
        adapter = _make_adapter(routes=routes, max_body_bytes=100)

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            large_payload = {"data": "x" * 200}
            resp = await cli.post(
                "/webhooks/big",
                json=large_payload,
                headers={"Content-Length": "999999"},
            )
            assert resp.status == 413


# ===================================================================
# INSECURE_NO_AUTH
# ===================================================================


class TestInsecureNoAuth:

    @pytest.mark.asyncio
    async def test_insecure_no_auth_skips_validation(self):
        """Setting secret to _INSECURE_NO_AUTH bypasses signature check."""
        routes = {"open": {"secret": _INSECURE_NO_AUTH, "prompt": "hello"}}
        adapter = _make_adapter(routes=routes)
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            # No signature header at all — should still be accepted
            resp = await cli.post("/webhooks/open", json={"test": True})
            assert resp.status == 202


# ===================================================================
# Session isolation
# ===================================================================


class TestSessionIsolation:

    @pytest.mark.asyncio
    async def test_concurrent_webhooks_get_independent_sessions(self):
        """Two events on the same route produce different session keys."""
        routes = {"ci": {"secret": _INSECURE_NO_AUTH, "prompt": "build"}}
        adapter = _make_adapter(routes=routes)

        captured_events = []

        async def _capture(event):
            captured_events.append(event)

        adapter.handle_message = _capture

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp1 = await cli.post(
                "/webhooks/ci",
                json={"ref": "main"},
                headers={"X-GitHub-Delivery": "aaa-111"},
            )
            assert resp1.status == 202

            resp2 = await cli.post(
                "/webhooks/ci",
                json={"ref": "dev"},
                headers={"X-GitHub-Delivery": "bbb-222"},
            )
            assert resp2.status == 202

        # Wait for the async tasks to be created
        await asyncio.sleep(0.05)

        assert len(captured_events) == 2
        ids = {ev.source.chat_id for ev in captured_events}
        assert len(ids) == 2, "Each delivery must have a unique session chat_id"


# ===================================================================
# Delivery info cleanup
# ===================================================================


class TestDeliveryCleanup:

    @pytest.mark.asyncio
    async def test_delivery_info_survives_multiple_sends(self):
        """send() must NOT pop delivery_info.

        Interim status messages (fallback notifications, context-pressure
        warnings, etc.) flow through the same send() path as the final
        response.  If the entry were popped on the first send, the final
        response would silently downgrade to the ``log`` deliver type.
        Regression test for that bug.
        """
        adapter = _make_adapter()
        chat_id = "webhook:test:d-xyz"
        adapter._delivery_info[chat_id] = {
            "deliver": "log",
            "deliver_extra": {},
            "payload": {"x": 1},
        }
        adapter._delivery_info_created[chat_id] = time.time()

        # First send (e.g. an interim status message)
        result1 = await adapter.send(chat_id, "Status: switching to fallback")
        assert result1.success is True
        # Entry must still be present so the final send can read it
        assert chat_id in adapter._delivery_info

        # Second send (the final agent response)
        result2 = await adapter.send(chat_id, "Final agent response")
        assert result2.success is True
        assert chat_id in adapter._delivery_info

    @pytest.mark.asyncio
    async def test_delivery_info_pruned_via_ttl(self):
        """Stale delivery_info entries are dropped on the next POST."""
        adapter = _make_adapter()
        adapter._idempotency_ttl = 60  # short TTL for the test
        now = time.time()

        # Stale entry — older than TTL
        adapter._delivery_info["webhook:test:old"] = {"deliver": "log"}
        adapter._delivery_info_created["webhook:test:old"] = now - 120

        # Fresh entry — should survive
        adapter._delivery_info["webhook:test:new"] = {"deliver": "log"}
        adapter._delivery_info_created["webhook:test:new"] = now - 5

        adapter._prune_delivery_info(now)

        assert "webhook:test:old" not in adapter._delivery_info
        assert "webhook:test:old" not in adapter._delivery_info_created
        assert "webhook:test:new" in adapter._delivery_info
        assert "webhook:test:new" in adapter._delivery_info_created


# ===================================================================
# check_webhook_requirements
# ===================================================================


class TestCheckRequirements:
    def test_returns_true_when_aiohttp_available(self):
        assert check_webhook_requirements() is True

    @patch("gateway.platforms.webhook.AIOHTTP_AVAILABLE", False)
    def test_returns_false_without_aiohttp(self):
        assert check_webhook_requirements() is False


# ===================================================================
# __raw__ template token
# ===================================================================


class TestRawTemplateToken:
    """Tests for the {__raw__} special token in _render_prompt."""

    def test_raw_resolves_to_full_json_payload(self):
        """{__raw__} in a template dumps the entire payload as JSON."""
        adapter = _make_adapter()
        payload = {"action": "opened", "number": 42}
        result = adapter._render_prompt(
            "Payload: {__raw__}", payload, "push", "test"
        )
        expected_json = json.dumps(payload, indent=2)
        assert result == f"Payload: {expected_json}"

    def test_raw_truncated_at_4000_chars(self):
        """{__raw__} output is truncated at 4000 characters for large payloads."""
        adapter = _make_adapter()
        # Build a payload whose JSON repr exceeds 4000 chars
        payload = {"data": "x" * 5000}
        result = adapter._render_prompt("{__raw__}", payload, "push", "test")
        assert len(result) <= 4000

    def test_raw_mixed_with_other_variables(self):
        """{__raw__} can be mixed with regular template variables."""
        adapter = _make_adapter()
        payload = {"action": "closed", "number": 7}
        result = adapter._render_prompt(
            "Action={action} Raw={__raw__}", payload, "push", "test"
        )
        assert result.startswith("Action=closed Raw=")
        assert '"action": "closed"' in result
        assert '"number": 7' in result


# ===================================================================
# CLCK HubSpot support triage
# ===================================================================


class TestHubSpotSupportTriage:
    """Tests for the CLCK support triage webhook card path."""

    def _support_triage_routes(self):
        return {
            "hubspot-support-triage": {
                "secret": _INSECURE_NO_AUTH,
                "events": ["hubspot_support_triage"],
                "prompt": "generic fallback prompt should not be used",
                "deliver": "slack",
                "deliver_extra": {
                    "chat_id": "{slack.channel_id}",
                    "thread_id": "{slack.thread_ts}",
                },
            }
        }

    def _attach_slack_runner(self, adapter):
        slack_adapter = AsyncMock()
        slack_adapter.send = AsyncMock(return_value=SendResult(success=True))
        runner = MagicMock()
        runner.adapters = {Platform.SLACK: slack_adapter}
        runner.config.get_home_channel.return_value = None
        adapter.gateway_runner = runner
        return slack_adapter

    def test_unknown_sender_formats_fallback_manual_review_card(self):
        adapter = _make_adapter()
        card = adapter._format_hubspot_support_triage_card(
            {
                "event_type": "hubspot_support_triage",
                "gmail": {
                    "from": "Unknown Person <unknown@example.invalid>",
                    "to": "support@clck.com.au",
                    "subject": "Can you help?",
                },
                "support": {"summary": "Unknown sender needs HubSpot help."},
                "matcher": {
                    "decision": "fallback",
                    "reason": "no_safe_match",
                    "client_name": None,
                    "assignee_hint": "internal_review",
                },
            }
        )

        assert "Request summary: Unknown sender needs HubSpot help." in card
        assert "Sender/source/subject: Unknown Person <unknown@example.invalid> / support@clck.com.au / Can you help?" in card
        assert "Client match: fallback/no_safe_match" in card
        assert "Owner/assignee hint: unknown/manual review" in card
        assert "Recommended internal next action: Manually confirm the client/route before replying." in card
        assert "Safety: no email sent; no HubSpot write; no client Slack post." in card

    def test_explicit_client_hint_line_renders_and_directive_is_not_summary(self):
        adapter = _make_adapter()
        card = adapter._format_hubspot_support_triage_card(
            {
                "event_type": "hubspot_support_triage",
                "gmail": {
                    "from": "Damien <damien@clck.com.au>",
                    "source_mailbox": "support@clck.com.au",
                    "subject": "Fwd: 2ND SERVICE BOARD",
                },
                "support": {
                    "summary": "Process as: Off Track RV\n\nCan anything sent to pdmelb@offtrackrv.com create a ticket?",
                },
                "matcher": {
                    "decision": "route_client",
                    "reason": "safe_match",
                    "client_name": "Off Track RV",
                    "owner_primary": "Damien",
                    "assignee_hint": "Damien",
                    "hubspot_access_status": "connected",
                    "hubspot_token_reference_present": True,
                    "explicit_client_hint": "Off Track RV",
                    "explicit_client_hint_type": "client_name",
                    "explicit_client_hint_source": "damien@clck.com.au",
                    "explicit_client_hint_trusted": True,
                    "explicit_client_hint_matched_client": {"client_name": "Off Track RV"},
                    "request_summary_cleaned": "Can anything sent to pdmelb@offtrackrv.com create a ticket?",
                },
            }
        )

        assert "Request summary: Can anything sent to pdmelb@offtrackrv.com create a ticket?" in card
        assert "Processing hint: CLCK-forwarded as Off Track RV" in card
        assert "Process as:" not in card

    def test_matched_safe_question_formats_draft_only_card(self):
        adapter = _make_adapter()
        card = adapter._format_hubspot_support_triage_card(
            {
                "event_type": "hubspot_support_triage",
                "gmail": {
                    "from": "Jane <jane@vacationer.example>",
                    "source_mailbox": "support@clck.com.au",
                    "subject": "Where do I find the import?",
                },
                "support": {
                    "summary": "Client asks where to find the HubSpot import view.",
                    "priority": "normal",
                },
                "matcher": {
                    "decision": "route_client",
                    "reason": "safe_match",
                    "client_name": "Vacationer Caravans",
                    "owner_primary": "Damien",
                    "assignee_hint": "Damien",
                    "portal_id": "123456",
                    "hubspot_token_reference": {"type": "env", "token_env": "HUBSPOT_X"},
                },
            }
        )

        assert "Client match: matched client: Vacationer Caravans" in card
        assert "Owner/assignee hint: Damien" in card
        assert "Risk/action level: safe question/draft only" in card
        assert "HubSpot status: portal/token found; read-only inspection skipped; no writes in MVP." in card
        assert "Draft client reply:" in card
        assert "Thanks for this. I’ll take a look and come back with the next step shortly." in card

    def test_off_track_service_board_question_gives_specific_internal_check(self):
        adapter = _make_adapter()
        card = adapter._format_hubspot_support_triage_card(
            {
                "event_type": "hubspot_support_triage",
                "gmail": {
                    "from": "Justin Borg <justin@offtrackrv.com>",
                    "source_mailbox": "support@clck.com.au",
                    "subject": "2ND SERVICE BOARD",
                    "snippet": (
                        "Hi mate, With the second service board, can I get anything sent to "
                        "this email address pdmelb@offtrackrv.com to create a ticket in the new service board?"
                    ),
                },
                "support": {
                    "summary": (
                        "Justin asks whether anything sent to pdmelb@offtrackrv.com can "
                        "create a ticket in the new service board."
                    ),
                },
                "matcher": {
                    "decision": "route_client",
                    "reason": "safe_match",
                    "client_name": "Off Track RV",
                    "owner_primary": "Damien",
                    "assignee_hint": "Damien",
                    "portal_id": "441989220",
                    "hubspot_access_status": "connected",
                    "hubspot_token_reference_present": True,
                },
            }
        )

        assert "Client match: matched client: Off Track RV (safe_match)" in card
        assert "Issue type: ticket intake routing / email-to-ticket" in card
        assert "Client ask: Confirm whether emails sent to pdmelb@offtrackrv.com can create tickets in the requested service board." in card
        assert "Likely system area: HubSpot Help Desk or Conversations Inbox team email channel" in card
        assert "Recommended internal next action: Inspect HubSpot read-only: Help Desk/Conversations channel accounts for pdmelb@offtrackrv.com" in card
        assert "Tickets > Pipelines/Automate" in card
        assert "Draft client reply: Draft intentionally withheld:" in card
        assert "I’ll take a look and come back with the next step shortly" not in card
        assert "Safety: no email sent; no HubSpot write; no client Slack post." in card

    def test_off_track_enriched_reasoning_renders_evidence_backed_answer(self):
        adapter = _make_adapter()
        card = adapter._format_hubspot_support_triage_card(
            {
                "event_type": "hubspot_support_triage",
                "gmail": {
                    "from": "Justin Borg <justin@offtrackrv.com>",
                    "source_mailbox": "support@clck.com.au",
                    "subject": "2ND SERVICE BOARD",
                    "snippet": (
                        "Can anything sent to pdmelb@offtrackrv.com create a ticket "
                        "in the new service board?"
                    ),
                },
                "support": {
                    "summary": (
                        "Justin asks whether anything sent to pdmelb@offtrackrv.com can "
                        "create a ticket in the new service board."
                    ),
                },
                "matcher": {
                    "decision": "route_client",
                    "reason": "safe_match",
                    "client_name": "Off Track RV",
                    "owner_primary": "Damien",
                    "assignee_hint": "Damien",
                    "portal_id": "441989220",
                    "hubspot_access_status": "connected",
                    "hubspot_token_reference_present": True,
                },
                "support_reasoning": {
                    "status": "evidence_supported",
                    "issue_type": "ticket intake routing / email-to-ticket",
                    "client_ask": (
                        "Confirm whether pdmelb@offtrackrv.com can create tickets in "
                        "the new service board."
                    ),
                    "likely_system_area": "HubSpot Help Desk email channel and ticket pipeline defaults.",
                    "read_only_findings": [
                        "pdmelb@offtrackrv.com is not currently connected as a HubSpot Help Desk/Conversations email channel",
                        "Pre-Delivery Pipeline exists and has a New stage",
                        "support@offtrackrv.com currently lands in Warranty Pipeline",
                    ],
                    "recommended_internal_action": (
                        "Confirm mailbox vs alias/group, then connect/configure pdmelb@offtrackrv.com "
                        "as a Help Desk team email channel with default ticket settings "
                        "Pre-Delivery Pipeline > New."
                    ),
                    "draft_client_reply": (
                        "At the moment pdmelb@offtrackrv.com isn’t connected to HubSpot for ticket creation. "
                        "Can you confirm whether that address is a real mailbox we can connect, or an alias/group "
                        "that needs forwarding? Once confirmed, we can set it to create tickets in "
                        "Pre-Delivery Pipeline > New."
                    ),
                },
            }
        )

        assert "Read-only findings: pdmelb@offtrackrv.com is not currently connected" in card
        assert "Pre-Delivery Pipeline exists and has a New stage" in card
        assert "support@offtrackrv.com currently lands in Warranty Pipeline" in card
        assert "Recommended internal next action: Confirm mailbox vs alias/group" in card
        assert "Draft client reply: At the moment pdmelb@offtrackrv.com isn’t connected" in card
        assert "I’ll take a look and come back with the next step shortly" not in card
        assert "Safety: no email sent; no HubSpot write; no client Slack post." in card

    def test_access_needed_client_routes_but_blocks_hubspot_inspection(self):
        adapter = _make_adapter()
        card = adapter._format_hubspot_support_triage_card(
            {
                "event_type": "hubspot_support_triage",
                "gmail": {
                    "from": "Matt <matt@playconnectgroup.com.au>",
                    "source_mailbox": "support@clck.com.au",
                    "subject": "Can you check HubSpot?",
                },
                "support": {
                    "summary": "Client asks for a read-only HubSpot check.",
                    "requires_hubspot_access": True,
                },
                "matcher": {
                    "decision": "route_client",
                    "reason": "safe_match",
                    "client_name": "Playconnect Group",
                    "owner_primary": "Damien",
                    "assignee_hint": "Damien",
                    "portal_id": "442443150",
                    "hubspot_access_status": "access_needed",
                    "hubspot_access_needed": True,
                    "hubspot_token_reference_present": False,
                },
            }
        )

        assert "Client match: matched client: Playconnect Group" in card
        assert "HubSpot status: support-active route; HubSpot access needed before inspection or implementation; no writes in MVP." in card
        assert "Recommended internal next action: Request/grant HubSpot portal access before inspection; keep triage and reply drafting internal." in card
        assert "Safety: no email sent; no HubSpot write; no client Slack post." in card

    def test_requested_hubspot_change_formats_approval_benson_no_write_card(self):
        adapter = _make_adapter()
        card = adapter._format_hubspot_support_triage_card(
            {
                "event_type": "hubspot_support_triage",
                "gmail": {
                    "from": "Ops <ops@client.example>",
                    "source_mailbox": "support@clck.com.au",
                    "subject": "Please update our pipeline stage",
                },
                "support": {
                    "summary": "Client asks CLCK to change a HubSpot pipeline stage.",
                    "requested_action": "hubspot_change",
                    "requires_hubspot_access": True,
                },
                "matcher": {
                    "decision": "route_client",
                    "reason": "safe_match",
                    "client_name": "Example Client",
                    "owner_primary": "Damien",
                    "assignee_hint": "Damien",
                    "portal_id": "456789",
                    "hubspot_token_reference": {"type": "env", "token_env": "HUBSPOT_Y"},
                },
            }
        )

        assert "Owner/assignee hint: Benson" in card
        assert "Risk/action level: HubSpot change requested; approval required" in card
        assert "Recommended internal next action: Benson to inspect read-only and propose the exact change; Damien approves before any HubSpot write." in card
        assert "Safety: no email sent; no HubSpot write; no client Slack post." in card

    @pytest.mark.asyncio
    async def test_existing_support_reasoning_is_not_enriched_again(self):
        adapter = _make_adapter()

        worker = AsyncMock(return_value={"status": "should_not_run"})
        adapter._run_support_reasoning_worker = worker
        payload = {
            "event_type": "hubspot_support_triage",
            "support_reasoning": {"status": "provided"},
        }

        enriched = await adapter._enrich_hubspot_support_triage_payload(payload)

        assert enriched is payload
        worker.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_support_triage_worker_failure_adds_unavailable_reasoning(self):
        adapter = _make_adapter()
        adapter._run_support_reasoning_worker = AsyncMock(side_effect=RuntimeError("boom"))
        payload = {"event_type": "hubspot_support_triage"}

        enriched = await adapter._enrich_hubspot_support_triage_payload(payload)

        assert enriched is not payload
        assert enriched["support_reasoning"]["status"] == "enrichment_unavailable"
        assert enriched["support_reasoning"]["evidence_supported"] is False
        assert "support_reasoning" not in payload

    @pytest.mark.asyncio
    async def test_support_triage_worker_success_renders_enriched_fields(self):
        adapter = _make_adapter(routes=self._support_triage_routes())
        adapter.handle_message = AsyncMock()
        slack_adapter = self._attach_slack_runner(adapter)
        adapter._run_support_reasoning_worker = AsyncMock(
            return_value={
                "status": "evidence_supported",
                "evidence_supported": True,
                "issue_type": "ticket intake routing / email-to-ticket",
                "client_ask": "Confirm whether pdmelb@offtrackrv.com can create tickets.",
                "read_only_findings": ["pdmelb@offtrackrv.com is not connected as a HubSpot email channel"],
                "recommended_internal_action": "Confirm mailbox ownership, then connect the Help Desk channel after approval.",
                "draft_client_reply": "pdmelb@offtrackrv.com is not connected yet; we need to connect it before ticket creation.",
            }
        )

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/hubspot-support-triage",
                json={
                    "event_type": "hubspot_support_triage",
                    "gmail": {
                        "from": "Justin <justin@offtrackrv.com>",
                        "source_mailbox": "support@clck.com.au",
                        "subject": "2ND SERVICE BOARD",
                        "snippet": "Can pdmelb@offtrackrv.com create a ticket?",
                    },
                    "support": {"summary": "Client asks about email-to-ticket routing."},
                    "matcher": {"decision": "route_client", "reason": "safe_match", "client_name": "Off Track RV"},
                    "slack": {"channel_id": "C0ASKKH52RK", "thread_ts": ""},
                },
                headers={"X-Request-ID": "support-triage-reasoning-success"},
            )
            assert resp.status == 202

        adapter.handle_message.assert_not_called()
        slack_adapter.send.assert_awaited_once()
        content = slack_adapter.send.await_args.args[1]
        assert "Read-only findings: pdmelb@offtrackrv.com is not connected" in content
        assert "Recommended internal next action: Confirm mailbox ownership" in content
        assert "Draft client reply: pdmelb@offtrackrv.com is not connected yet" in content
        assert "Safety: no email sent; no HubSpot write; no client Slack post." in content

    @pytest.mark.asyncio
    async def test_support_triage_noise_reasoning_suppresses_slack_delivery(self):
        adapter = _make_adapter(routes=self._support_triage_routes())
        adapter.handle_message = AsyncMock()
        slack_adapter = self._attach_slack_runner(adapter)
        adapter._run_support_reasoning_worker = AsyncMock(
            return_value={
                "status": "support_noise",
                "evidence_status": "suppressed_noise",
                "issue_type": "support_noise",
                "recommended_internal_action": "Suppress/no Slack card; do not draft, route, or inspect HubSpot.",
            }
        )

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/hubspot-support-triage",
                json={
                    "event_type": "hubspot_support_triage",
                    "gmail": {
                        "from": "Caitlin <caitlin@pandadoc.com>",
                        "source_mailbox": "support@clck.com.au",
                        "subject": "CLCK, ready to get started?",
                    },
                    "matcher": {"decision": "fallback", "reason": "no_safe_match"},
                    "slack": {"channel_id": "C0ASKKH52RK", "thread_ts": ""},
                },
                headers={"X-Request-ID": "support-triage-noise-suppressed"},
            )
            assert resp.status == 202
            data = await resp.json()

        assert data["status"] == "suppressed"
        adapter.handle_message.assert_not_called()
        slack_adapter.send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_support_triage_worker_timeout_fails_open_to_base_card(self):
        adapter = _make_adapter(routes=self._support_triage_routes())
        adapter.handle_message = AsyncMock()
        slack_adapter = self._attach_slack_runner(adapter)
        adapter._run_support_reasoning_worker = AsyncMock(side_effect=asyncio.TimeoutError())

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/hubspot-support-triage",
                json={
                    "event_type": "hubspot_support_triage",
                    "gmail": {
                        "from": "Unknown <unknown@example.invalid>",
                        "source_mailbox": "support@clck.com.au",
                        "subject": "TEST support triage",
                    },
                    "support": {"summary": "Fake internal test payload."},
                    "matcher": {"decision": "fallback", "reason": "no_safe_match"},
                    "slack": {"channel_id": "C0ASKKH52RK", "thread_ts": ""},
                },
                headers={"X-Request-ID": "support-triage-reasoning-timeout"},
            )
            assert resp.status == 202

        adapter.handle_message.assert_not_called()
        slack_adapter.send.assert_awaited_once()
        content = slack_adapter.send.await_args.args[1]
        assert "Request summary: Fake internal test payload." in content
        assert "Client match: fallback/no_safe_match" in content
        assert "Support reasoning enrichment unavailable" in content
        assert "Safety: no email sent; no HubSpot write; no client Slack post." in content

    @pytest.mark.asyncio
    async def test_support_triage_event_delivers_card_to_slack_thread_without_agent(self):
        routes = {
            "hubspot-support-triage": {
                "secret": _INSECURE_NO_AUTH,
                "events": ["hubspot_support_triage"],
                "prompt": "generic fallback prompt should not be used",
                "deliver": "slack",
                "deliver_extra": {
                    "chat_id": "{slack.channel_id}",
                    "thread_id": "{slack.thread_ts}",
                },
            }
        }
        adapter = _make_adapter(routes=routes)
        adapter.handle_message = AsyncMock()
        adapter._run_support_reasoning_worker = AsyncMock(side_effect=RuntimeError("worker disabled in test"))
        slack_adapter = AsyncMock()
        slack_adapter.send = AsyncMock(return_value=SendResult(success=True))
        runner = MagicMock()
        runner.adapters = {Platform.SLACK: slack_adapter}
        runner.config.get_home_channel.return_value = None
        adapter.gateway_runner = runner

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/hubspot-support-triage",
                json={
                    "event_type": "hubspot_support_triage",
                    "gmail": {
                        "from": "Unknown <unknown@example.invalid>",
                        "source_mailbox": "support@clck.com.au",
                        "subject": "TEST support triage",
                    },
                    "support": {"summary": "Fake internal test payload."},
                    "matcher": {"decision": "fallback", "reason": "no_safe_match"},
                    "slack": {"channel_id": "C0ASKKH52RK", "thread_ts": "1777784852.911829"},
                },
                headers={"X-Request-ID": "support-triage-test-1"},
            )
            assert resp.status == 202
            data = await resp.json()

        assert data["status"] == "accepted"
        adapter.handle_message.assert_not_called()
        slack_adapter.send.assert_awaited_once()
        chat_id, content = slack_adapter.send.await_args.args[:2]
        assert chat_id == "C0ASKKH52RK"
        assert "Request summary: Fake internal test payload." in content
        assert "Client match: fallback/no_safe_match" in content
        assert slack_adapter.send.await_args.kwargs["metadata"] == {"thread_id": "1777784852.911829"}


# ===================================================================
# Cross-platform delivery thread_id passthrough
# ===================================================================


class TestDeliverCrossPlatformThreadId:
    """Tests for thread_id passthrough in _deliver_cross_platform."""

    def _setup_adapter_with_mock_target(self):
        """Set up a webhook adapter with a mocked gateway_runner and target adapter."""
        adapter = _make_adapter()
        mock_target = AsyncMock()
        mock_target.send = AsyncMock(return_value=SendResult(success=True))

        mock_runner = MagicMock()
        mock_runner.adapters = {Platform("telegram"): mock_target}
        mock_runner.config.get_home_channel.return_value = None

        adapter.gateway_runner = mock_runner
        return adapter, mock_target

    @pytest.mark.asyncio
    async def test_thread_id_passed_as_metadata(self):
        """thread_id from deliver_extra is passed as metadata to adapter.send()."""
        adapter, mock_target = self._setup_adapter_with_mock_target()
        delivery = {
            "deliver_extra": {
                "chat_id": "12345",
                "thread_id": "999",
            }
        }
        await adapter._deliver_cross_platform("telegram", "hello", delivery)
        mock_target.send.assert_awaited_once_with(
            "12345", "hello", metadata={"thread_id": "999"}
        )

    @pytest.mark.asyncio
    async def test_message_thread_id_passed_as_thread_id(self):
        """message_thread_id from deliver_extra is mapped to thread_id in metadata."""
        adapter, mock_target = self._setup_adapter_with_mock_target()
        delivery = {
            "deliver_extra": {
                "chat_id": "12345",
                "message_thread_id": "888",
            }
        }
        await adapter._deliver_cross_platform("telegram", "hello", delivery)
        mock_target.send.assert_awaited_once_with(
            "12345", "hello", metadata={"thread_id": "888"}
        )

    @pytest.mark.asyncio
    async def test_no_thread_id_sends_no_metadata(self):
        """When no thread_id is present, metadata is None."""
        adapter, mock_target = self._setup_adapter_with_mock_target()
        delivery = {
            "deliver_extra": {
                "chat_id": "12345",
            }
        }
        await adapter._deliver_cross_platform("telegram", "hello", delivery)
        mock_target.send.assert_awaited_once_with(
            "12345", "hello", metadata=None
        )
