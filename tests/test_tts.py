"""Tests for the ElevenLabs TTS helper and the /api/fleet/tts route.

Owner: Dashboard
"""

import asyncio

from fastapi.testclient import TestClient

import backend.tts as tts
from backend.main import app

client = TestClient(app)


class FakeResponse:
    content = b"fake-mp3-bytes"

    def raise_for_status(self):
        pass


class FakeClient:
    calls: list = []

    def __init__(self, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def post(self, url, headers, json):
        FakeClient.calls.append((url, headers["xi-api-key"], json))
        return FakeResponse()


def test_missing_key_returns_503(monkeypatch, tmp_path):
    monkeypatch.setattr(tts, "CACHE_DIR", tmp_path)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    response = client.post("/api/fleet/tts", json={"text": "Warning. Well 2 is under observation."})
    assert response.status_code == 503


def test_empty_and_long_text_rejected():
    assert client.post("/api/fleet/tts", json={"text": ""}).status_code == 422
    assert client.post("/api/fleet/tts", json={"text": "a" * 501}).status_code == 422


def test_key_and_voice_come_from_environment_and_audio_is_cached(monkeypatch, tmp_path):
    FakeClient.calls = []
    monkeypatch.setattr(tts, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(tts.httpx, "AsyncClient", FakeClient)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    monkeypatch.delenv("ELEVENLABS_VOICE_ID", raising=False)

    first = asyncio.run(tts.synthesize("Critical alert. Well 19 requires attention."))
    second = asyncio.run(tts.synthesize("Critical alert. Well 19 requires attention."))

    assert first == second == b"fake-mp3-bytes"
    # The second call is served from the disk cache.
    assert len(FakeClient.calls) == 1
    url, key, body = FakeClient.calls[0]
    assert url.endswith(tts.DEFAULT_VOICE_ID)
    assert key == "test-key"
    assert body["voice_settings"] == tts.VOICE_SETTINGS

    # A different voice is a different cache entry and a different request.
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "other-voice")
    asyncio.run(tts.synthesize("Critical alert. Well 19 requires attention."))
    assert FakeClient.calls[-1][0].endswith("other-voice")


def test_cached_audio_served_without_key(monkeypatch, tmp_path):
    monkeypatch.setattr(tts, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(tts.httpx, "AsyncClient", FakeClient)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    asyncio.run(tts.synthesize("Cached line."))

    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    response = client.post("/api/fleet/tts", json={"text": "Cached line."})
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
