import json
from pathlib import Path


def test_openai_tts_failure_falls_back_to_edge(monkeypatch, tmp_path):
    import tools.tts_tool as tts

    out = tmp_path / "fallback.mp3"

    monkeypatch.setattr(tts, "_load_tts_config", lambda: {"provider": "openai", "use_gateway": True})
    monkeypatch.setattr(tts, "_import_openai_client", lambda: object())
    monkeypatch.setattr(tts, "_generate_openai_tts", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("402 insufficient credits")))
    monkeypatch.setattr(tts, "_import_edge_tts", lambda: object())

    async def _fake_edge(text, output_path, tts_config):
        Path(output_path).write_bytes(b"edge-fallback")
        return output_path

    monkeypatch.setattr(tts, "_generate_edge_tts", _fake_edge)
    monkeypatch.setattr(tts, "_check_neutts_available", lambda: False)

    result = json.loads(tts.text_to_speech_tool("hello world", output_path=str(out)))

    assert result["success"] is True
    assert result["provider"] == "edge"
    assert Path(result["file_path"]).exists()
    assert Path(result["file_path"]).read_bytes() == b"edge-fallback"
    assert result.get("fallback_from") == "openai"
    assert "402 insufficient credits" in (result.get("fallback_reason") or "")
