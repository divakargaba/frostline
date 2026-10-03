# Frostline — Hydrate Early-Warning Agent

Frostline is an agentic early-warning system for gas hydrate formation in offshore oil wells. It ingests real-time sensor telemetry (pressure, temperature, flow rates), applies physics-based hydrate equilibrium models and ML event classifiers trained on the Petrobras 3W dataset, and uses an LLM tool-calling agent to triage anomalies into actionable decisions (ALERT / WATCH / DISMISS) with recommended inhibitor dosing. The goal is to maximise lead time before hydrate blockages while minimising false alarms.

Built for the IEEE YP Industry Hackathon 2026 — Energy & Infrastructure stream, Case 9 (Option B).

## Team Roles

| Role | Owner | Scope |
|------|-------|-------|
| Data + ML | TBD | Data loading, feature engineering, LightGBM classifier, onset detection |
| Physics + Eval | TBD | Hydrate equilibrium models, forecasting, evaluation baselines |
| Agent + Backend | Div | Watcher, LLM agent loop, RAG, tools, FastAPI backend |
| Dashboard + Presentation | TBD | Vite + React frontend, visualisation, demo |

## Setup

```bash
# 1. Create and activate a Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Fetch the 3W dataset (sparse clone — classes 4,6,7,8,9 only)
bash scripts/fetch_3w.sh

# 4. Copy and fill in environment variables
cp .env.example .env

# 5. Start the backend
uvicorn backend.main:app --reload
```

## Data

This project uses the **3W dataset** by Petrobras, licensed under **CC BY 4.0**.

- Repository: <https://github.com/petrobras/3W>
- Citation: Vargas, R. E. V. et al. *A Realistic and Public Dataset with Rare Undesirable Real Events in Oil Wells.* Journal of Petroleum Science and Engineering, 2019.

Event classes used: **4** (flow instability), **6** (quick restriction in PCK), **7** (scaling in PCK), **8** (hydrate in production line), **9** (hydrate in service line). See `CLAUDE.md` for detailed data facts.

## API Contract

### `GET /wells`

```json
[{"well_id": "WELL-00019", "instance_id": "WELL-00019_20240107", "source": "real", "sensors_available": ["P-PDG", "T-PDG", "P-TPT", "T-TPT", "P-MON-CKP", "QGL"], "has_hydrate_event": true}]
```

### `GET /stream/{instance_id}?speed=60&cached=false`

SSE stream. Event types:

| Event | Payload keys |
|-------|-------------|
| `tick` | `t`, `sensors`, `margin_C`, `p_hydrate`, `p_lookalike`, `p_normal` |
| `phase_marker` | `t`, `phase` ("forming" or "established") |
| `watch_trigger` | `t`, `reason`, `score` |
| `tool_call` | `t`, `call_id`, `tool`, `args` |
| `tool_result` | `t`, `call_id`, `tool`, `result` |
| `decision` | `t`, `decision`, `recheck_min`, `confidence`, `diagnosis`, `onset_eta`, `dose_wt_pct`, `dose_in_range`, `evidence`, `playbook_refs`, `brief` |

See `CLAUDE.md` for full JSON examples and `backend/schemas.py` for Pydantic models.

### `GET /results`

```json
{"systems": [{"name": "B0", "description": "Always normal", "caught": 0, "missed": 14, "false_alarms_per_day": 0.0, "mean_lead_time_min": null, "misdiagnosis_rate": 0.0}]}
```

### `POST /tts`

Request: `{"text": "Hydrate alert on well 19."}` → Response: `audio/mpeg`
