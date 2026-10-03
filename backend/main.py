"""FastAPI backend for Frostline.

Owner: Div

Routes (Div's pipeline):
  - GET  /health         — System health + capabilities
  - GET  /wells          — List available wells/instances
  - GET  /stream/{id}    — SSE stream of events for a well
  - GET  /results        — Evaluation results table
  - POST /tts            — Text-to-speech stub
  - POST /fleet/sessions — Create fleet replay session
  - GET  /fleet/sessions/{id}/stream — Fleet SSE stream

Routes (Mico's research + fleet):
  - /api/fleet/*         — Four-well guided demo (backend/fleet.py router)
  - /api/live/*          — Live agent sessions (backend/live.py router)
  - /api/health          — Research health check
  - /api/research        — Research results
  - /api/scenarios       — Replay scenario list
"""
from __future__ import annotations

import asyncio
from functools import lru_cache
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
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

# Mico's routers
from backend.live import router as live_router
from backend.fleet import router as fleet_router, capabilities as fleet_capabilities

log = logging.getLogger("frostline.api")

app = FastAPI(title="Frostline", version="0.2.0")

# Include Mico's routers (at /api/fleet and /api/live — no overlap with Div's /fleet/*)
app.include_router(live_router)
app.include_router(fleet_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS + ["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for running LLM calls without blocking the event loop
_executor = ThreadPoolExecutor(max_workers=2)


def _has_physics() -> bool:
    try:
        from src.physics import hydrate_margin
        hydrate_margin(200.0, 50.0)
        return True
    except (ImportError, NotImplementedError):
        return False


def _has_model() -> bool:
    try:
        from src.tools import get_model_adapter
        adapter = get_model_adapter()
        if adapter.ready:
            return True
    except Exception:
        pass
    try:
        from src.model import predict  # noqa: F401
        return True
    except (ImportError, NotImplementedError, AttributeError):
        return False


def _model_status() -> str:
    try:
        from src.tools import get_model_adapter
        adapter = get_model_adapter()
        if adapter.ready:
            return f"loaded via {adapter.source} ({len(adapter.classes)} classes)"
        return adapter.error or "not loaded"
    except Exception:
        return "adapter error"


def _llm_mode() -> str:
    if LLM_MODE == "mock":
        return "mock"
    if OPENROUTER_API_KEY and OPENROUTER_MODEL:
        return "openrouter"
    return "mock"


# ---------------------------------------------------------------------------
# Div's routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    from src.llm import get_usage_tracker, _get_or_build_pool
    usage = get_usage_tracker().summary()
    pool = _get_or_build_pool()
    providers = pool.provider_status() if pool else []
    return {
        "status": "ok",
        "has_processed": DATA_PROCESSED.exists() and (DATA_PROCESSED / "index.csv").exists(),
        "has_raw": DATA_RAW_3W.exists(),
        "has_seed": (DATA_SEED / "well_hydrate_seed.csv").exists(),
        "has_physics": _has_physics(),
        "has_model": _has_model(),
        "model_status": _model_status(),
        "llm_mode": _llm_mode(),
        "llm_model": OPENROUTER_MODEL if _llm_mode() == "openrouter" else "mock",
        "llm_providers": providers,
        "llm_usage": usage,
    }


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
    agent: bool = Query(default=True),
    pause_on_agent: bool = Query(default=True),
    cache_only: bool = Query(default=False),
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
        _stream_live(request, inst, speed, from_minute, end, recording_path,
                     agent_enabled=agent, pause_on_agent=pause_on_agent,
                     cache_only=cache_only)
    )


async def _stream_live(request, inst, speed, from_min, to_min, recording_path,
                       agent_enabled=True, pause_on_agent=True, cache_only=False):
    """Generate SSE events from replay instance, with watcher + agent."""
    from src.watcher import Watcher
    from src.tools import AgentContext

    event_id = 0
    delay = 1.0 / speed
    rec_file = None
    if recording_path:
        rec_file = open(recording_path, "w")

    prev_phase = None
    watcher = Watcher()
    well_id = inst.metadata.get("well_id", inst.metadata.get("instance_id", ""))
    instance_id = inst.metadata.get("instance_id", "")
    agent_decisions: list[dict] = []

    def _emit(event_type: str, data: dict):
        nonlocal event_id
        event_id += 1
        if rec_file:
            rec_file.write(json.dumps({"type": event_type, "data": data}) + "\n")
        return {"event": event_type, "id": str(event_id), "data": json.dumps(data)}

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
                    yield _emit("phase_marker", pm)
                except Exception as e:
                    log.error("Phase marker validation: %s", e)
                    yield _emit("error", {"t": tick_data["t"], "message": str(e)})
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
                yield _emit("tick", tick_payload)
            except Exception as e:
                log.error("Tick validation: %s", e)
                yield _emit("error", {"t": tick_data["t"], "message": str(e)})

            # Watcher check
            if agent_enabled:
                trigger = watcher.check_tick(tick_data, well_id, idx)
                if trigger:
                    yield _emit("watch_trigger", {
                        "t": trigger["t"],
                        "reason": trigger["reason"],
                        "score": trigger["score"],
                    })

                    # Run agent in thread pool
                    ctx = AgentContext(
                        instance_id=instance_id,
                        well_id=well_id,
                        minute_index=idx,
                        df=inst.df,
                        decisions=agent_decisions,
                        thresholds=watcher.thresholds,
                    )

                    loop = asyncio.get_event_loop()
                    try:
                        from src.agent import run_agent
                        agent_events = await loop.run_in_executor(
                            _executor,
                            lambda: list(run_agent(ctx, trigger, cache_only=cache_only))
                        )

                        for ae in agent_events:
                            if await request.is_disconnected():
                                break
                            yield _emit(ae["type"], ae["data"])

                            if ae["type"] == "decision":
                                decision_data = ae["data"]
                                agent_decisions.append(decision_data)
                                watcher.record_decision(
                                    well_id,
                                    decision_data.get("decision", "DISMISS"),
                                    idx,
                                    score=trigger["score"],
                                    recheck_min=decision_data.get("recheck_min"),
                                )
                    except Exception as e:
                        log.error("Agent error: %s", e)
                        yield _emit("error", {"t": tick_data["t"], "message": f"Agent error: {e}"})

            await asyncio.sleep(delay)

        # End event
        end_data = {
            "t": datetime.now().isoformat(),
            "total_minutes": to_min - from_min,
            "instance_id": inst.metadata.get("instance_id", ""),
        }
        yield _emit("end", end_data)

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


# ---------------------------------------------------------------------------
# Fleet endpoints — Mico's router at /api/fleet handles the full fleet demo.
# Div's old /fleet/* endpoints are retired (replaced by Mico's architecture).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Mico's research routes
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def seed_report():
    from src.research import experiment
    return experiment()


@lru_cache(maxsize=1)
def seed_replays():
    from src.research import seed_scenarios
    return seed_scenarios(seed_report())


def real_report():
    path = ROOT / "data/processed/research/real_report.json"
    return json.loads(path.read_text()) if path.exists() else {
        "status": "not_run",
        "message": "Run python scripts/fetch_research_data.py then python -m src.real_pilot to reproduce the 3W pilot.",
    }


@lru_cache(maxsize=2)
def read_real_replays(mtime):
    return json.loads((ROOT / "data/processed/research/real_replays.json").read_text())


def scenarios():
    path = ROOT / "data/processed/research/real_replays.json"
    return {**seed_replays(), **(read_real_replays(path.stat().st_mtime_ns) if path.exists() else {})}


@app.get("/api/health")
def api_health():
    return {
        "status": "ok",
        "mode": "local research replay",
        "real_pilot": real_report()["status"],
        "fleet": fleet_capabilities(),
    }


@app.get("/api/research")
def api_research():
    return {**seed_report(), "real": real_report()}


@app.get("/api/research/export")
def export_results():
    return Response(
        json.dumps(api_research(), indent=2, allow_nan=False),
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="frostline-research-results.json"'},
    )


@app.post("/api/experiments/run")
def run_experiment():
    seed_report.cache_clear()
    seed_replays.cache_clear()
    return api_research()


@app.get("/api/scenarios")
def list_scenarios():
    return [
        {k: v for k, v in item.items() if k not in ["frames", "policy"]}
        | {"frame_count": len(item["frames"])}
        for item in scenarios().values()
    ]


@app.get("/api/scenarios/{scenario_id}")
def get_scenario(scenario_id: str):
    found = scenarios().get(scenario_id)
    if found is None:
        raise HTTPException(404, "Unknown replay scenario")
    return found


# ---------------------------------------------------------------------------
# Static frontend (Mico's built dashboard)
# ---------------------------------------------------------------------------

DIST = ROOT / "frontend/dist"
if (DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")


@app.get("/dashboard", include_in_schema=False)
def dashboard():
    if not (DIST / "index.html").exists():
        raise HTTPException(503, "Build the dashboard: cd frontend; npm ci; npm run build")
    return FileResponse(DIST / "index.html")
