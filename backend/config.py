"""App configuration — paths, env vars, CORS.

Owner: Div
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW_3W = ROOT / "data" / "raw" / "3W" / "dataset"
DATA_SEED = ROOT / "data" / "seed"
DATA_DEMO = ROOT / "data" / "demo"
RESULTS = ROOT / "results"
FRONTEND_MOCKS = ROOT / "frontend" / "mocks"

# --- LLM ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "")
LLM_MODE = os.getenv("LLM_MODE", "")  # "mock" forces mock
LLM_MAX_REQUESTS_PER_RUN = int(os.getenv("LLM_MAX_REQUESTS_PER_RUN", "3"))

# --- CORS ---
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
