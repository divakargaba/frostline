# Fleet API Contract

One runtime pipeline: four held-out 3W wells replayed together on a shared clock, scored every
source minute by the fleet LightGBM model (`models/fleet_model.pkl`), escalated by the alarm
policy (`src/fleet_policy.py`), and reviewed by a bounded investigation queue. Reviews use
rule-based evidence checks by default; LLM investigations are an optional per-run toggle.

Implementation: `backend/fleet.py` (router), `backend/main.py` (app, health, scores, static).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | `{status, fleet: capabilities, llm_usage}` |
| GET | `/api/scores` | Offline baseline ladder from `results/summary.csv` |
| GET | `/api/fleet/catalog` | Wells, speeds, model and LLM readiness |
| GET | `/api/fleet/results` | Offline fleet model report (`{"status":"not_run"}` if absent) |
| GET | `/api/fleet/workflow-results` | Offline workflow evaluation report |
| POST | `/api/fleet/sessions` | Start a run → snapshot (201) |
| GET | `/api/fleet/sessions/{id}` | Current snapshot |
| POST | `/api/fleet/sessions/{id}/control` | Replay control → snapshot |
| POST | `/api/fleet/sessions/{id}/incidents/{well}/actions` | Operator action → snapshot |
| GET | `/api/fleet/sessions/{id}/events` | SSE stream of snapshots |
| GET | `/api/fleet/sessions/{id}/export` | JSON journal download (run, audit, events) |
| GET | `/dashboard`, `/assets/*` | Built frontend (`frontend/dist`), when present |

### `GET /api/fleet/catalog`

```json
{
  "wells": [{"id": "WELL-00001", "name": "Well 1", "source_file": ""}],
  "default_speed": 12,
  "speeds": [12, 30, 60, 120],
  "model_ready": true,
  "llm_configured": true,
  "llm_providers": ["gemini", "groq"],
  "readiness_message": "Model ready. ..."
}
```

- `llm_configured` is true when any provider key is set (`GEMINI_API_KEY`, `GROQ_API_KEY`,
  `OPENROUTER_API_KEY`); `llm_providers` lists provider names in pool order, never keys.
- `LLM_MODE=mock` reports no providers.

### `POST /api/fleet/sessions`

```json
{"speed": 12, "use_llm": false}
```

- `speed`: one of `12 | 30 | 60 | 120` source minutes per real minute (default 12; 422 otherwise).
- `use_llm` (default false): route investigations through the LLM investigator
  (`src/fleet_llm.investigate`) over the provider pool (Gemini → Groq → OpenRouter) with the BM25
  playbook. If no provider is configured the run still starts on rules, the snapshot reports
  `use_llm: false`, and `readiness_message` explains why.
- 503 when the fleet model is not trained; 409 when four runs are already active.
- The run loads the recordings (`status: "preparing"`) and then plays continuously (`"running"`).

### Snapshot (`FleetRun`)

Returned by every session endpoint and carried as the SSE `payload`.

```json
{
  "id": "…", "status": "preparing|running|paused|completed|cancelled|failed",
  "speed": 12, "elapsed_seconds": 2400, "index": 40, "total": 180,
  "agent_mode": "rules|live",
  "use_llm": false, "llm_configured": false, "llm_providers": [],
  "model_ready": true, "readiness_message": "…",
  "wells": [Well, Well, Well, Well],
  "priority": ["WELL-00019", "WELL-00002", "WELL-00006", "WELL-00001"],
  "last_event_id": 57, "requests_used": 0, "request_budget": 18, "error": null
}
```

- `agent_mode` becomes `"live"` only after an LLM investigation has passed validation.
- `requests_used` / `request_budget`: provider requests spent by this run (`FLEET_LLM_MAX`, default 18).
  Requests are also capped at 10/minute across runs, two concurrent investigations across runs, and a
  provider circuit breaker pauses LLM reviews after an account rate limit (5 min) or a nonzero cost.

### Well

```json
{
  "id": "WELL-00019", "name": "Well 19", "status": "normal|watch|attention|unavailable",
  "source_file": "WELL-00019_20120601165020.parquet", "source_timestamp": "2012-06-02T22:34:00",
  "quality": {"status": "good|degraded|unavailable", "summary": "…", "missing": [], "invalid": [], "unchanged": []},
  "prediction": {"scores": {"normal": 0.1, "hydrate": 0.85, "lookalike": 0.05}, "model_id": "fleet-…",
                 "threshold": 0.7, "recovery_threshold": 0.6, "persistence_minutes": 10, "recovery_minutes": 5,
                 "alarm_streak": 4, "alarm_active": false},
  "frames": [{"t": "…", "elapsed_seconds": 2340, "sensors": {"P-PDG": 251.2, "…": null}, "risk_score": 0.85,
              "limits": {"hydrate_threshold": 0.7, "alarm_active": false, "activation_streak": 4, "recovery_streak": 0,
                         "pressure_trigger_bar": 2.0, "pressure_change_bar": -0.4, "divergence_change_bar": 0.1}}],
  "incident": {"id": "…", "acknowledged": false, "completed": false, "opened_at": "…", "note": "",
               "checks": {}, "condition_cleared": false} ,
  "assessment": {"summary": "…", "evidence": ["…"], "next_step": "…", "source": "rules|live", "uncertainty": "…",
                 "category": "…", "checks": [{"id": "…", "label": "…", "reason": "…", "status": "pending", "definition": "…"}],
                 "tools": [{"name": "sensor_quality", "status": "done", "summary": "…"}], "recheck_minutes": 15},
  "investigation": "idle|queued|running|complete|unavailable", "activity": "…",
  "last_assessed": "…", "next_check": "…",
  "timeline": [{"id": "…", "kind": "detected|seen|checked|observation|review_completed|recheck|recovered", "at": "…", "recorded_at": "…", "summary": "…"}],
  "followup": {"last_checked_at": "…", "next_check_at": "…", "trigger": "…", "change_summary": "…", "before": {}, "after": {}}
}
```

- `frames`: the last 240 observed minutes (30 warm-up minutes precede `elapsed_seconds` 0).
- `prediction` is `null` until the first scored minute. Scores are uncalibrated model outputs.
- `insights` (per well, refreshed every minute; `null` before the first) comes from
  `src/sensor_insights.py`: `channels[]` gives each sensor's value, 10-minute change and trend
  (`rising | falling | steady | unavailable | inactive`), and `inferences[]` gives combined
  pressure/temperature observations (`{id, level: watch|info, sensors, text, why}`), e.g. a growing
  tubing or line pressure difference, cooling at rising pressure (toward hydrate-forming
  conditions), pressure and temperature falling together (flow loss), or pressure moving with a
  fixed choke. They are observations, not diagnoses, and do not change scores or the alarm policy.
  The catalog's `sensors[]` lists `{code, name, location, unit, why}` for all ten channels. The
  LLM investigator gets the same inferences through its `sensor_changes` tool.
- `frames[].limits` records the limits applied at that minute (`null` for warm-up frames):
  `hydrate_threshold` is the activation threshold while no alarm is active and the recovery
  threshold while one is (hysteresis); `pressure_trigger_bar` is `max(2 bar, 2% of P-TPT at the
  start of the 10-minute window)`, compared against `|pressure_change_bar|` and
  `|divergence_change_bar|`.
- Live LLM assessments add `playbook_refs` (BM25 playbook document ids).

### `POST /api/fleet/sessions/{id}/control`

```json
{"action": "pause|resume|step|cancel|speed|llm", "speed": 30, "use_llm": true}
```

- `step` (only while paused): advance every well one source minute and finish its queued reviews;
  the run stays paused.
- `speed` requires `speed`; `llm` requires `use_llm` and toggles LLM investigations mid-run (still
  falls back to rules without a provider).
- Unknown actions (including the removed `inject`, `next_moment`, `mode`) → 422.
  Invalid state (e.g. stepping while running, controlling an ended run) → 409.

### `POST /api/fleet/sessions/{id}/incidents/{well}/actions`

```json
{"action": "acknowledge|observation|complete|check|recheck", "note": "…", "check_id": "…",
 "result": "confirmed|not_confirmed|unavailable", "incident_id": "…", "check_definition": "…"}
```

Operator reports are recorded as human reports, never ground truth; they queue a re-review but
cannot clear a sensor-driven alarm. Validation failures (stale incident, changed check, no incident) → 422.

### `GET /api/fleet/sessions/{id}/events?after=N`

Server-sent events. Each message:

```
id: 58
event: fleet
data: {"id": 58, "type": "snapshot", "payload": <snapshot>}
```

Resume with `?after=<last id>` or the `Last-Event-ID` header; a cursor ahead of the run → 422.
A `: heartbeat` comment is sent every 15 s; the stream ends after a terminal status.

### `GET /api/scores`

```json
{"rows": [{"name": "B0", "description": "Always normal — no detection", "caught": 0, "total_events": 12,
           "missed": 12, "false_alarms_per_day": 0.0, "mean_lead_time_min": null,
           "misdiagnosis_rate": 0.0, "notes": "…"}]}
```

Rows are in file order (B0, B1, B1-revised, M1, M3); empty cells are `null`. 404 if `results/summary.csv`
has not been generated (`python eval.py`).
