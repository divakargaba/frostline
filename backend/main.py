"""Local Frostline research API, replay stream and built dashboard."""
import asyncio
from functools import lru_cache
import json
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from src.research import ROOT, experiment, seed_scenarios
from backend.live import router as live_router

app = FastAPI(title="Frostline research prototype", version="0.2.0")
app.include_router(live_router)
app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"], allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


@lru_cache(maxsize=1)
def seed_report():
    return experiment()


@lru_cache(maxsize=1)
def seed_replays():
    return seed_scenarios(seed_report())


def real_report():
    path = ROOT / "data/processed/research/real_report.json"
    return json.loads(path.read_text()) if path.exists() else {"status": "not_run", "message": "Run python scripts/fetch_research_data.py then python -m src.real_pilot to reproduce the 3W pilot."}


@lru_cache(maxsize=2)
def read_real_replays(mtime):
    return json.loads((ROOT / "data/processed/research/real_replays.json").read_text())


def scenarios():
    path = ROOT / "data/processed/research/real_replays.json"
    return {**seed_replays(), **(read_real_replays(path.stat().st_mtime_ns) if path.exists() else {})}


@app.get("/api/health")
def health():
    return {"status": "ok", "mode": "local research replay", "real_pilot": real_report()["status"]}


@app.get("/api/research")
@app.get("/results")
def results():
    return {**seed_report(), "real": real_report()}


@app.get("/api/research/export")
def export_results():
    return Response(json.dumps(results(), indent=2, allow_nan=False), media_type="application/json",
                    headers={"Content-Disposition": 'attachment; filename="frostline-research-results.json"'})


@app.post("/api/experiments/run")
def run_experiment():
    seed_report.cache_clear()
    seed_replays.cache_clear()
    return results()


@app.get("/api/scenarios")
def list_scenarios():
    return [{k: v for k, v in item.items() if k not in ["frames", "policy"]} | {"frame_count": len(item["frames"])} for item in scenarios().values()]


@app.get("/api/scenarios/{scenario_id}")
def scenario(scenario_id: str):
    found = scenarios().get(scenario_id)
    if found is None:
        raise HTTPException(404, "Unknown replay scenario")
    return found


@app.get("/wells")
def wells():
    return [{"well_id": item["id"], "instance_id": item["id"], "source": item["provenance"], "sensors_available": list(item["sensor_labels"]), "has_hydrate_event": any("hydrate" in f["ground_truth"].lower() or f["ground_truth"] == "Event" for f in item["frames"])} for item in scenarios().values()]


@app.get("/stream/{well}")
async def stream(well: str, delay_ms: int = Query(0, ge=0, le=2000), start: int = Query(0, ge=0)):
    replay = scenario(well)
    if start >= len(replay["frames"]):
        raise HTTPException(422, "Start exceeds replay length")
    async def events():
        for index, frame in enumerate(replay["frames"][start:], start):
            tick = {"type": "tick", "t": frame["t"], "sensors": frame["sensors"], "ground_truth": frame["ground_truth"]}
            yield f"id: {index}\nevent: tick\ndata: {json.dumps(tick)}\n\n"
            yield f"event: decision\ndata: {json.dumps(frame)}\n\n"
            if delay_ms:
                await asyncio.sleep(delay_ms / 1000)
        yield 'event: complete\ndata: {"complete":true}\n\n'
    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.post("/tts")
def tts():
    raise HTTPException(503, "Voice is not configured. The dashboard provides the complete operator brief as text.")


DIST = ROOT / "frontend/dist"
if (DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")


@app.get("/", include_in_schema=False)
def dashboard():
    if not (DIST / "index.html").exists():
        raise HTTPException(503, "Build the dashboard: cd frontend; npm ci; npm run build")
    return FileResponse(DIST / "index.html")
