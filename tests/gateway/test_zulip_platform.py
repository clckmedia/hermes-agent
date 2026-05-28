import asyncio
import json

import pytest
from unittest.mock import MagicMock, patch

from gateway.config import GatewayConfig, Platform, PlatformConfig, _apply_env_overrides
from gateway.platforms.base import MessageType


def test_zulip_env_loading_from_process_env(monkeypatch):
    monkeypatch.setenv("ZULIP_SITE", "https://zulip.example.com")
    monkeypatch.setenv("ZULIP_BOT_EMAIL", "arlo-bot@example.com")
    monkeypatch.setenv("ZULIP_BOT_API_KEY", "bot-secret")
    monkeypatch.setenv("ZULIP_HOME_CHANNEL", "dm:user@example.com")
    monkeypatch.setenv("ZULIP_ALLOWED_USERS", "user@example.com")
    monkeypatch.setenv("ZULIP_ALL_PUBLIC_STREAMS", "true")

    config = GatewayConfig()
    _apply_env_overrides(config)

    pconfig = config.platforms[Platform.ZULIP]
    assert pconfig.enabled is True
    assert pconfig.api_key == "bot-secret"
    assert pconfig.extra["site"] == "https://zulip.example.com"
    assert pconfig.extra["bot_email"] == "arlo-bot@example.com"
    assert pconfig.extra["allowed_users"] == "user@example.com"
    assert pconfig.extra["all_public_streams"] is True
    assert pconfig.home_channel.chat_id == "dm:user@example.com"
    assert Platform.ZULIP in config.get_connected_platforms()


def test_zulip_env_loading_from_secrets_file(tmp_path, monkeypatch):
    import gateway.config as config_mod

    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir()
    (secrets_dir / "zulip.env").write_text(
        "ZULIP_SITE=https://zulip.example.com\n"
        "ZULIP_BOT_EMAIL=arlo-bot@example.com\n"
        "ZULIP_BOT_API_KEY=bot-secret\n"
        "ZULIP_ALLOWED_USERS=user@example.com\n"
        "ZULIP_ALL_PUBLIC_STREAMS=true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config_mod, "get_hermes_home", lambda: tmp_path)
    monkeypatch.delenv("ZULIP_SITE", raising=False)
    monkeypatch.delenv("ZULIP_BOT_EMAIL", raising=False)
    monkeypatch.delenv("ZULIP_BOT_API_KEY", raising=False)

    config = GatewayConfig()
    config_mod._apply_env_overrides(config)

    pconfig = config.platforms[Platform.ZULIP]
    assert pconfig.enabled is True
    assert pconfig.extra["site"] == "https://zulip.example.com"
    assert pconfig.extra["bot_email"] == "arlo-bot@example.com"
    assert pconfig.extra["allowed_users"] == "user@example.com"
    assert pconfig.extra["all_public_streams"] is True
    assert pconfig.api_key == "bot-secret"


def test_zulip_register_payload_can_request_all_public_streams():
    from gateway.config import PlatformConfig
    from gateway.platforms.zulip import ZulipAdapter

    adapter = ZulipAdapter(
        PlatformConfig(
            enabled=True,
            api_key="bot-secret",
            extra={
                "site": "https://zulip.example.com",
                "bot_email": "arlo-bot@example.com",
                "dm_only": False,
                "all_public_streams": True,
            },
        )
    )

    assert adapter._register_payload()["all_public_streams"] == "true"


def test_zulip_register_payload_does_not_request_public_streams_in_dm_only_mode():
    from gateway.config import PlatformConfig
    from gateway.platforms.zulip import ZulipAdapter

    adapter = ZulipAdapter(
        PlatformConfig(
            enabled=True,
            api_key="bot-secret",
            extra={
                "site": "https://zulip.example.com",
                "bot_email": "arlo-bot@example.com",
                "dm_only": True,
                "all_public_streams": True,
            },
        )
    )

    assert "all_public_streams" not in adapter._register_payload()


def test_zulip_parse_dm_message():
    from gateway.config import PlatformConfig
    from gateway.platforms.zulip import ZulipAdapter

    adapter = ZulipAdapter(
        PlatformConfig(
            enabled=True,
            api_key="bot-secret",
            extra={"site": "https://zulip.example.com", "bot_email": "arlo-bot@example.com"},
        )
    )
    event = adapter._message_event_from_zulip(
        {
            "id": 123,
            "type": "private",
            "sender_email": "damien@example.com",
            "sender_full_name": "Damien",
            "content": "hello **arlo**",
            "display_recipient": [
                {"email": "damien@example.com", "full_name": "Damien"},
                {"email": "arlo-bot@example.com", "full_name": "Arlo"},
            ],
        }
    )

    assert event is not None
    assert event.text == "hello **arlo**"
    assert event.message_type is MessageType.TEXT
    assert event.message_id == "123"
    assert event.source.platform is Platform.ZULIP
    assert event.source.chat_type == "dm"
    assert event.source.chat_id == "dm:damien@example.com"
    assert event.source.user_id == "damien@example.com"
    assert event.source.user_name == "Damien"


def test_zulip_parse_channel_topic_message():
    from gateway.config import PlatformConfig
    from gateway.platforms.zulip import ZulipAdapter

    adapter = ZulipAdapter(
        PlatformConfig(
            enabled=True,
            api_key="bot-secret",
            extra={"site": "https://zulip.example.com", "bot_email": "arlo-bot@example.com", "dm_only": False},
        )
    )
    event = adapter._message_event_from_zulip(
        {
            "id": 456,
            "type": "stream",
            "sender_email": "user@example.com",
            "sender_full_name": "User",
            "content": "channel note",
            "display_recipient": "general",
            "stream_id": 99,
            "subject": "ops",
        }
    )

    assert event is not None
    assert event.source.chat_type == "channel"
    assert event.source.chat_id == "stream:99"
    assert event.source.chat_name == "general"
    assert event.source.thread_id == "ops"
    assert event.source.chat_topic == "ops"


def test_zulip_applies_channel_prompt_from_parent_stream_id():
    from gateway.config import PlatformConfig
    from gateway.platforms.zulip import ZulipAdapter

    adapter = ZulipAdapter(
        PlatformConfig(
            enabled=True,
            api_key="bot-secret",
            extra={
                "site": "https://zulip.example.com",
                "bot_email": "arlo-bot@example.com",
                "dm_only": False,
                "channel_prompts": {"99": "Tech lane: check integration skills first."},
            },
        )
    )

    event = adapter._message_event_from_zulip(
        {
            "id": 456,
            "type": "stream",
            "sender_email": "user@example.com",
            "sender_full_name": "User",
            "content": "channel note",
            "display_recipient": "tech",
            "stream_id": 99,
            "subject": "api-mcp",
        }
    )

    assert event is not None
    assert event.channel_prompt == "Tech lane: check integration skills first."


def test_zulip_applies_exact_chat_prompt_before_parent_stream_id():
    from gateway.config import PlatformConfig
    from gateway.platforms.zulip import ZulipAdapter

    adapter = ZulipAdapter(
        PlatformConfig(
            enabled=True,
            api_key="bot-secret",
            extra={
                "site": "https://zulip.example.com",
                "bot_email": "arlo-bot@example.com",
                "dm_only": False,
                "channel_prompts": {
                    "stream:99": "Exact stream prompt.",
                    "99": "Parent stream prompt.",
                },
            },
        )
    )

    event = adapter._message_event_from_zulip(
        {
            "id": 457,
            "type": "stream",
            "sender_email": "user@example.com",
            "content": "channel note",
            "display_recipient": "tech",
            "stream_id": 99,
            "subject": "api-mcp",
        }
    )

    assert event is not None
    assert event.channel_prompt == "Exact stream prompt."


def test_zulip_filters_self_messages():
    from gateway.config import PlatformConfig
    from gateway.platforms.zulip import ZulipAdapter

    adapter = ZulipAdapter(
        PlatformConfig(
            enabled=True,
            api_key="bot-secret",
            extra={"site": "https://zulip.example.com", "bot_email": "arlo-bot@example.com"},
        )
    )

    assert adapter._message_event_from_zulip(
        {"id": 1, "type": "private", "sender_email": "arlo-bot@example.com", "content": "loop"}
    ) is None


def test_zulip_authorizes_allowed_user_from_platform_config(monkeypatch):
    from gateway.run import GatewayRunner

    gw = GatewayRunner.__new__(GatewayRunner)
    gw.config = GatewayConfig(
        platforms={
            Platform.ZULIP: PlatformConfig(
                enabled=True,
                extra={"allowed_users": "damien@clck.com.au"},
            )
        }
    )
    gw.pairing_store = MagicMock()
    gw.pairing_store.is_approved.return_value = False

    source = MagicMock()
    source.platform = Platform.ZULIP
    source.user_id = "damien@clck.com.au"
    source.chat_id = "dm:damien@clck.com.au"
    source.chat_type = "dm"

    with patch.dict("os.environ", {}, clear=True):
        assert gw._is_user_authorized(source) is True
        assert gw._get_unauthorized_dm_behavior(Platform.ZULIP) == "ignore"

    source.user_id = "other@example.com"
    with patch.dict("os.environ", {}, clear=True):
        assert gw._is_user_authorized(source) is False


@pytest.mark.asyncio
async def test_zulip_send_dm_path(monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platforms.zulip import ZulipAdapter

    calls = []

    async def fake_post(self, path, data):
        calls.append((path, data))
        return {"id": 789}

    monkeypatch.setattr(ZulipAdapter, "_api_post", fake_post)
    adapter = ZulipAdapter(
        PlatformConfig(
            enabled=True,
            api_key="bot-secret",
            extra={"site": "https://zulip.example.com", "bot_email": "arlo-bot@example.com"},
        )
    )

    result = await adapter.send("dm:damien@example.com", "hi")

    assert result.success is True
    assert result.message_id == "789"
    assert calls == [("/messages", {"type": "direct", "to": json.dumps(["damien@example.com"]), "content": "hi"})]


@pytest.mark.asyncio
async def test_zulip_materializes_user_upload_text_file(monkeypatch, tmp_path):
    import gateway.platforms.zulip as zulip_mod
    from gateway.config import PlatformConfig
    from gateway.platforms.zulip import ZulipAdapter

    async def fake_download(self, upload_path):
        assert upload_path == "/user_uploads/90056/abc/PastedText.txt"
        return b"Real prompt from this Zulip upload", "text/plain"

    monkeypatch.setattr(zulip_mod.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(ZulipAdapter, "_download_upload_bytes", fake_download)
    adapter = ZulipAdapter(
        PlatformConfig(
            enabled=True,
            api_key="bot-secret",
            extra={"site": "https://zulip.example.com", "bot_email": "arlo-bot@example.com", "dm_only": False},
        )
    )
    event = adapter._message_event_from_zulip(
        {
            "id": 592712608,
            "type": "stream",
            "sender_email": "damien@example.com",
            "content": "[PastedText.txt](/user_uploads/90056/abc/PastedText.txt)",
            "display_recipient": "arlo-child-tasks",
            "stream_id": 595303,
            "subject": "20260504_113258_527f855f",
        }
    )

    await adapter._materialize_uploads(event)

    assert "Real prompt from this Zulip upload" in event.text
    assert "--- Begin attached file: PastedText.txt ---" in event.text
    assert len(event.media_urls) == 1
    assert "/592712608/" in event.media_urls[0]
    assert event.media_urls[0].endswith("/PastedText.txt")
    assert (tmp_path / "hermes-zulip-uploads" / "592712608").exists()


@pytest.mark.asyncio
async def test_zulip_upload_download_failure_warns_without_tmp_search(monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platforms.zulip import ZulipAdapter

    async def fake_download(self, upload_path):
        raise RuntimeError("HTTP 404")

    monkeypatch.setattr(ZulipAdapter, "_download_upload_bytes", fake_download)
    adapter = ZulipAdapter(
        PlatformConfig(
            enabled=True,
            api_key="bot-secret",
            extra={"site": "https://zulip.example.com", "bot_email": "arlo-bot@example.com", "dm_only": False},
        )
    )
    event = adapter._message_event_from_zulip(
        {
            "id": 592712609,
            "type": "stream",
            "sender_email": "damien@example.com",
            "content": "[PastedText.txt](/user_uploads/90056/missing/PastedText.txt)",
            "display_recipient": "arlo-child-tasks",
            "stream_id": 595303,
            "subject": "child-1",
        }
    )

    await adapter._materialize_uploads(event)

    assert "Zulip attachment unavailable: PastedText.txt" in event.text
    assert "Do not search /tmp or reuse a same-named local file" in event.text
    assert event.media_urls == []


@pytest.mark.asyncio
async def test_zulip_send_typing_uses_cached_dm_user_id(monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platforms.zulip import ZulipAdapter

    calls = []

    async def fake_post(self, path, data):
        calls.append((path, data.copy()))
        return {"result": "success"}

    monkeypatch.setattr(ZulipAdapter, "_api_post", fake_post)
    adapter = ZulipAdapter(
        PlatformConfig(
            enabled=True,
            api_key="bot-secret",
            extra={"site": "https://zulip.example.com", "bot_email": "arlo-bot@example.com"},
        )
    )
    event = adapter._message_event_from_zulip(
        {
            "id": 123,
            "type": "private",
            "sender_id": 9,
            "sender_email": "damien@example.com",
            "display_recipient": [
                {"id": 9, "email": "damien@example.com", "full_name": "Damien"},
                {"id": 10, "email": "arlo-bot@example.com", "full_name": "Arlo"},
            ],
            "content": "hello",
        }
    )

    await adapter.send_typing(event.source.chat_id)
    await adapter.stop_typing(event.source.chat_id)

    assert calls == [
        ("/typing", {"type": "direct", "op": "start", "to": json.dumps([9])}),
        ("/typing", {"type": "direct", "op": "stop", "to": json.dumps([9])}),
    ]


@pytest.mark.asyncio
async def test_zulip_send_typing_uses_channel_topic_metadata(monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platforms.zulip import ZulipAdapter

    calls = []

    async def fake_post(self, path, data):
        calls.append((path, data.copy()))
        return {"result": "success"}

    monkeypatch.setattr(ZulipAdapter, "_api_post", fake_post)
    adapter = ZulipAdapter(
        PlatformConfig(
            enabled=True,
            api_key="bot-secret",
            extra={"site": "https://zulip.example.com", "bot_email": "arlo-bot@example.com", "dm_only": False},
        )
    )

    await adapter.send_typing("stream:99", metadata={"thread_id": "Zulip questions"})
    await adapter.stop_typing("stream:99")

    assert calls == [
        (
            "/typing",
            {"type": "stream", "op": "start", "stream_id": "99", "topic": "Zulip questions"},
        ),
        (
            "/typing",
            {"type": "stream", "op": "stop", "stream_id": "99", "topic": "Zulip questions"},
        ),
    ]


@pytest.mark.asyncio
async def test_send_message_tool_routes_zulip(monkeypatch):
    from gateway.config import Platform, PlatformConfig
    import tools.send_message_tool as send_tool

    calls = []

    async def fake_send_zulip(pconfig, chat_id, message, thread_id=None):
        calls.append((pconfig, chat_id, message, thread_id))
        return {"success": True, "platform": "zulip", "chat_id": chat_id, "message_id": "42"}

    monkeypatch.setattr(send_tool, "_send_zulip", fake_send_zulip)
    pconfig = PlatformConfig(
        enabled=True,
        api_key="bot-secret",
        extra={"site": "https://zulip.example.com", "bot_email": "arlo-bot@example.com"},
    )

    result = await send_tool._send_to_platform(Platform.ZULIP, pconfig, "dm:damien@example.com", "hello")

    assert result["success"] is True
    assert result["message_id"] == "42"
    assert calls == [(pconfig, "dm:damien@example.com", "hello", None)]
