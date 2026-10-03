# Frostline — Hydrate Early-Warning Agent

Frostline is an agentic early-warning system for gas hydrate formation in offshore oil wells. It ingests real-time sensor telemetry (pressure, temperature, flow rates), applies physics-based hydrate equilibrium models and ML event classifiers trained on the Petrobras 3W dataset, and uses an LLM tool-calling agent to triage anomalies into actionable decisions (ALERT / WATCH / DISMISS) with recommended inhibitor dosing. The goal is to maximise lead time before hydrate blockages while minimising false alarms.

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

We use event classes relevant to hydrate formation and flow-assurance anomalies:
- **4** — Severe slugging
- **6** — Flow instability
- **7** — Rapid productivity loss
- **8** — Quick restriction increase
- **9** — Hydrate in production line

## API Contract

The SSE stream (`GET /stream/{well}`) emits the following event types:

| Event Type | Description |
|---|---|
| `tick` | Raw sensor reading forwarded to the frontend |
| `watch_trigger` | Watcher detected an anomaly; agent invoked |
| `tool_call` | Agent is calling a tool (name + args) |
| `tool_result` | Tool returned a result |
| `decision` | Agent final decision: ALERT / WATCH / DISMISS with rationale |
| `phase_marker` | Phase transition detected (normal → forming → established) |
