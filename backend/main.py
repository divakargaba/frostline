"""FastAPI backend for Frostline.

Owner: Div

Provides the HTTP API and SSE streaming endpoint for the dashboard.

Routes:
  - GET  /wells          — List available wells
  - GET  /stream/{well}  — SSE stream of events for a well
  - GET  /results        — Retrieve evaluation results
  - POST /tts            — Text-to-speech via ElevenLabs

TODO:
  - Implement well listing from loaded data
  - Implement SSE streaming with watcher + agent integration
  - Implement results endpoint (read from results/ directory)
  - Implement TTS proxy to ElevenLabs API
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Frostline", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/wells")
async def list_wells() -> list[dict]:
    """List available wells.

    Returns:
        List of dicts with well_id and metadata.
    """
    raise NotImplementedError


@app.get("/stream/{well}")
async def stream_well(well: str):
    """SSE stream of events for a well.

    Event types: tick, watch_trigger, tool_call, tool_result, decision, phase_marker.

    Args:
        well: Well identifier.
    """
    raise NotImplementedError


@app.get("/results")
async def get_results() -> dict:
    """Retrieve evaluation results.

    Returns:
        Dict with evaluation metrics and comparison data.
    """
    raise NotImplementedError


@app.post("/tts")
async def text_to_speech(text: str) -> dict:
    """Convert text to speech via ElevenLabs.

    Args:
        text: Text to synthesize.

    Returns:
        Dict with audio_url or base64 audio data.
    """
    raise NotImplementedError
