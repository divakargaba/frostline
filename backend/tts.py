"""ElevenLabs TTS helper with file-based caching."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import httpx

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")  # George
MODEL_ID = "eleven_flash_v2_5"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "tts_cache"


def _cache_path(text: str) -> Path:
    key = hashlib.sha256(text.encode()).hexdigest()[:16]
    return CACHE_DIR / f"{key}.mp3"


async def synthesize(text: str) -> bytes:
    """Return mp3 bytes for *text*, using a file cache to avoid repeat API calls."""
    cached = _cache_path(text)
    if cached.exists():
        return cached.read_bytes()

    if not ELEVENLABS_API_KEY:
        raise RuntimeError("ELEVENLABS_API_KEY is not set")

    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}",
            headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"},
            json={"text": text, "model_id": MODEL_ID, "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}},
        )
        response.raise_for_status()

    audio = response.content
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(audio)
    return audio
