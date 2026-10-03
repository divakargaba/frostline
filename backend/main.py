"""FastAPI backend for Frostline.

Owner: Div

Routes:
  - GET  /health         — System health + capabilities
  - GET  /wells          — List available wells/instances
  - GET  /stream/{id}    — SSE stream of events for a well
  - GET  /results        — Evaluation results table
  - POST /tts            — Text-to-speech stub
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

import backend.config as config
from backend.config import (
    CORS_ORIGINS, DATA_PROCESSED, DATA_RAW_3W, DATA_SEED,
    FRONTEND_MOCKS, LLM_MODE, OPENROUTER_API_KEY, OPENROUTER_MODEL, RESULTS,
)
from backend.replay import list_available_instances, load_demo_recording, load_instance
from backend.schemas import (
    EndEvent, ErrorEvent, HealthResponse, PhaseMarkerEvent, ResultsResponse,
    TickEvent, TTSRequest, WellInfo,
)

log = logging.getLogger("frostline.api")

app = FastAPI(title="Frostline", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _has_physics() -> bool:
    try:
        from src.physics import hydrate_margin
        hydrate_margin(200.0, 50.0)
        return True
    except (ImportError, NotImplementedError):
        return False


def _has_model() -> bool:
    try:
        from src.model import predict  # noqa: F401
        return True
    except (ImportError, NotImplementedError, AttributeError):
        return False


def _llm_mode() -> str:
    if LLM_MODE == "mock":
        return "mock"
    if OPENROUTER_API_KEY and OPENROUTER_MODEL:
        return "openrouter"
    return "mock"


@app.get("/health")
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        has_processed=DATA_PROCESSED.exists() and (DATA_PROCESSED / "index.csv").exists(),
        has_raw=DATA_RAW_3W.exists(),
        has_seed=(DATA_SEED / "well_hydrate_seed.csv").exists(),
        has_physics=_has_physics(),
        has_model=_has_model(),
        llm_mode=_llm_mode(),
        llm_model=OPENROUTER_MODEL if _llm_mode() == "openrouter" else "mock",
    )


@app.get("/wells")
async def list_wells() -> list[WellInfo]:
    instances = list_available_instances()
    return [WellInfo(**i) for i in instances]


@app.get("/stream/{instance_id}")
async def stream_well(
    request: Request,
    instance_id: str,
    speed: int = Query(default=60, ge=1, le=6000),
    from_minute: int = Query(default=0, ge=0),
    to_minute: int | None = Query(default=None),
    cached: bool = Query(default=False),
    record: bool = Query(default=False),
):
    """SSE stream of tick/phase_marker/end events for a well instance."""

    if cached:
        events = load_demo_recording(instance_id)
        if events is None:
            raise HTTPException(404, f"No cached recording for '{instance_id}'")
        return EventSourceResponse(_stream_cached(request, events, speed, instance_id))

    inst = load_instance(instance_id)
    if inst is None:
        raise HTTPException(
            404,
            f"Instance '{instance_id}' not found. "
            f"Available sources: processed={DATA_PROCESSED.exists()}, "
            f"raw={DATA_RAW_3W.exists()}, seed={(DATA_SEED / 'well_hydrate_seed.csv').exists()}"
        )

    end = to_minute if to_minute is not None else inst.n_minutes
    end = min(end, inst.n_minutes)

    recording_path = None
    if record:
        config.DATA_DEMO.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        recording_path = config.DATA_DEMO / f"{instance_id}_{ts}.jsonl"

    return EventSourceResponse(
        _stream_live(request, inst, speed, from_minute, end, recording_path)
    )


async def _stream_live(request, inst, speed, from_min, to_min, recording_path):
    """Generate SSE events from replay instance."""
    event_id = 0
    delay = 1.0 / speed
    rec_file = None
    if recording_path:
        rec_file = open(recording_path, "w")

    prev_phase = None

    try:
        for idx in range(from_min, to_min):
            if await request.is_disconnected():
                break

            tick_data = inst.tick(idx)
            phase = tick_data.get("phase", "normal")

            # Phase marker on change (including initial)
            if phase != prev_phase:
                pm = {"t": tick_data["t"], "phase": phase}
                try:
                    PhaseMarkerEvent(**pm)
                    event_id += 1
                    evt = {"type": "phase_marker", "data": pm}
                    if rec_file:
                        rec_file.write(json.dumps(evt) + "\n")
                    yield {"event": "phase_marker", "id": str(event_id), "data": json.dumps(pm)}
                except Exception as e:
                    log.error("Phase marker validation: %s", e)
                    event_id += 1
                    err = {"t": tick_data["t"], "message": str(e)}
                    yield {"event": "error", "id": str(event_id), "data": json.dumps(err)}
                prev_phase = phase

            # Tick
            tick_payload = {
                "t": tick_data["t"],
                "sensors": tick_data["sensors"],
                "margin_C": tick_data.get("margin_C"),
                "p_hydrate": tick_data.get("p_hydrate"),
                "p_lookalike": tick_data.get("p_lookalike"),
                "p_normal": tick_data.get("p_normal"),
            }
            try:
                TickEvent(**tick_payload)
                event_id += 1
                evt = {"type": "tick", "data": tick_payload}
                if rec_file:
                    rec_file.write(json.dumps(evt) + "\n")
                yield {"event": "tick", "id": str(event_id), "data": json.dumps(tick_payload)}
            except Exception as e:
                log.error("Tick validation: %s", e)
                event_id += 1
                err = {"t": tick_data["t"], "message": str(e)}
                yield {"event": "error", "id": str(event_id), "data": json.dumps(err)}

            await asyncio.sleep(delay)

        # End event
        event_id += 1
        end_data = {
            "t": datetime.now().isoformat(),
            "total_minutes": to_min - from_min,
            "instance_id": inst.metadata.get("instance_id", ""),
        }
        evt = {"type": "end", "data": end_data}
        if rec_file:
            rec_file.write(json.dumps(evt) + "\n")
        yield {"event": "end", "id": str(event_id), "data": json.dumps(end_data)}

    finally:
        if rec_file:
            rec_file.close()


async def _stream_cached(request, events, speed, instance_id):
    """Replay pre-recorded .jsonl events with timing scaled by speed."""
    event_id = 0
    delay = 1.0 / speed

    for evt in events:
        if await request.is_disconnected():
            break

        event_type = evt.get("type", "tick")
        data = evt.get("data", evt)
        event_id += 1
        yield {"event": event_type, "id": str(event_id), "data": json.dumps(data)}
        await asyncio.sleep(delay)


@app.get("/results")
async def get_results() -> ResultsResponse:
    # Try real results first
    summary_csv = RESULTS / "summary.csv"
    if summary_csv.exists():
        import pandas as pd
        df = pd.read_csv(summary_csv)
        systems = df.to_dict("records")
        return ResultsResponse(systems=systems, mock=False)

    # Fall back to mock
    mock_path = FRONTEND_MOCKS / "results.json"
    if mock_path.exists():
        data = json.loads(mock_path.read_text())
        data["mock"] = True
        return ResultsResponse(**data)

    raise HTTPException(404, "No results available")


@app.post("/tts")
async def text_to_speech(req: TTSRequest):
    """TTS stub — the dashboard owner implements this."""
    raise HTTPException(501, "TTS not implemented — dashboard owner's task")
