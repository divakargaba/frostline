"""Frostline API: the four-well fleet pipeline (/api/fleet), health, scores, and the built dashboard."""
from __future__ import annotations

import csv
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.config import CORS_ORIGINS, FRONTEND_DIST, RESULTS
from backend.fleet import capabilities, router as fleet_router

app = FastAPI(title="Frostline", version="0.3.0")
app.include_router(fleet_router)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

SCORE_COLUMNS = ["name", "description", "caught", "total_events", "missed", "false_alarms_per_day", "mean_lead_time_min", "misdiagnosis_rate", "notes"]
NUMERIC = {"caught", "total_events", "missed", "false_alarms_per_day", "mean_lead_time_min", "misdiagnosis_rate"}


def number(text):
    text = (text or "").strip()
    if not text:
        return None
    value = float(text)
    return int(value) if value.is_integer() and "." not in text else value


@lru_cache(maxsize=2)
def read_scores(mtime_ns):
    with (RESULTS / "summary.csv").open(newline="", encoding="utf-8") as stream:
        return [{c: number(row.get(c)) if c in NUMERIC else (row.get(c) or "") for c in SCORE_COLUMNS} for row in csv.DictReader(stream)]


@app.get("/api/health")
def health():
    from src.llm import get_usage_tracker
    return {"status": "ok", "fleet": capabilities(), "llm_usage": get_usage_tracker().summary()}


@app.get("/api/scores")
def scores():
    """Offline baseline ladder (B0, B1, B1-revised, M1, M3) from results/summary.csv, in file order."""
    path = RESULTS / "summary.csv"
    if not path.exists():
        raise HTTPException(404, "No scores yet. Run: python eval.py")
    return {"rows": read_scores(path.stat().st_mtime_ns)}


if (FRONTEND_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")


@app.get("/dashboard", include_in_schema=False)
def dashboard():
    if not (FRONTEND_DIST / "index.html").exists():
        raise HTTPException(503, "Build the dashboard: cd frontend; npm ci; npm run build")
    return FileResponse(FRONTEND_DIST / "index.html")
