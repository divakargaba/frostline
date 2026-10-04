"""ElevenLabs TTS helper with file-based caching.

The API key and voice are read from the environment (see .env.example) and
never reach the browser.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import httpx

# NORA, the operations assistant voice: "Sarah - Mature, Reassuring, Confident"
# (calm, professional, North American). Override with ELEVENLABS_VOICE_ID.
DEFAULT_VOICE_ID = "EXAVITQu4vr4xnSDxMaL"
MODEL_ID = "eleven_flash_v2_5"
# High stability and no style keep delivery even; slightly slow for clarity.
VOICE_SETTINGS = {"stability": 0.8, "similarity_boost": 0.75, "style": 0.0, "speed": 0.95}
# Under data/processed, which is git-ignored, so audio is never committed.
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "tts_cache"


def _voice_id() -> str:
    return os.getenv("ELEVENLABS_VOICE_ID", "").strip() or DEFAULT_VOICE_ID


def _cache_path(text: str) -> Path:
    settings = json.dumps(VOICE_SETTINGS, sort_keys=True)
    key = hashlib.sha256(f"{_voice_id()}|{MODEL_ID}|{settings}|{text}".encode()).hexdigest()[:16]
    return CACHE_DIR / f"{key}.mp3"


async def synthesize(text: str) -> bytes:
    """Return mp3 bytes for *text*, using a file cache to avoid repeat API calls."""
    cached = _cache_path(text)
    if cached.exists():
        return cached.read_bytes()

    api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is not set")

    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{_voice_id()}",
            headers={"xi-api-key": api_key, "Content-Type": "application/json"},
            json={"text": text, "model_id": MODEL_ID, "voice_settings": VOICE_SETTINGS},
        )
        response.raise_for_status()

    audio = response.content
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(audio)
    return audio
