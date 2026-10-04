"""App configuration: paths, env, CORS."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
# Provider keys live in the project's ignored .env; real environment variables win.
load_dotenv(ROOT / ".env")

DATA_RAW_3W = ROOT / "data" / "raw" / "3W" / "dataset"
DATA_DEMO = ROOT / "data" / "demo"
RESULTS = ROOT / "results"
FRONTEND_DIST = ROOT / "frontend" / "dist"
FLEET_LLM_MAX = int(os.getenv("FLEET_LLM_MAX", "18"))
CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
