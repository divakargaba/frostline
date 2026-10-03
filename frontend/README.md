# Frostline dashboard

React, TypeScript, Vite, Recharts and Lucide. Measurements and decisions come from the Python API; the legacy `mocks/` files are unused. Fonts are bundled for offline demos.

## Operator view

`FleetDashboard.tsx` is the default **Well map**. It shows four independent Petrobras 3W well recordings, a selectable map, pressure/temperature/model-score trends, the priority queue, and one concise assessment with an operator next step. The wide layout keeps trends under the map and decisions at the right; narrow screens show the map, priorities and assessment, then trends.

The map is illustrative: source recordings have independent timestamps and no shared geography. Original timestamps remain in Evidence; charts use elapsed replay minutes, with negative minutes showing prior history. Displayed dates preserve source time without claiming a source timezone. Pressure is supplied in bar and temperature in °C. Missing measurements stay visible as gaps. The model score is not a calibrated probability.

- **Start field replay** creates a server session and prepares its recordings/model. **Pause/Resume** controls the existing run; **Step** advances one replay step while paused. **Restart** cancels the previous run and starts fresh. Speed changes replay time, not source measurements.
- Selecting a map well or priority row updates the same charts and assessment. Evidence includes data quality, source file/timestamp, available tool history, and the operator record.
- **Acknowledge** records that the incident was seen. It does not clear its condition or stop monitoring. Observations are human reports, not evaluation labels. Completing a review also leaves monitoring active.
- **Try a telemetry fault** changes incoming pressure availability for the selected well. Injected inputs are excluded from published accuracy claims.

## Agent modes and results

Each assessment identifies its source: **live LLM**, **Model + rules**, or **recorded**. A configured provider is not proof that a live call succeeded; the assessment source indicates what produced that result. Without an OpenRouter key, numerical monitoring and rule-based assessments work, and the UI displays the missing-key notice. Credentials belong in the server's ignored `.env`, never in `VITE_*` variables or browser code. The prototype provides operator advice; it does not operate plant controls.

**Results** begins with the real-well fleet evaluation: training/evaluation recording counts, grouped validation, candidate decisions, frozen policy, catches, false alerts, missed minutes, detection delay, and limitations. It reports regressions alongside improvements. The older synthetic seed experiment is separate below it.

**More → Seed sandbox** retains `LiveMission.tsx` / `WellOverview.tsx` and the one-well synthetic `/api/live` workflow. Recorded replays, the earlier real-well pilot, and methodology are also available under More. These views are not the source of the four-well operating feed.

## API and session behavior

`fleetTypes.ts` defines the browser contract. The server owns replay progression, inference, investigation scheduling, incident state, and the journal.

| Endpoint                                                    | Purpose                                       |
| ----------------------------------------------------------- | --------------------------------------------- |
| `GET /api/fleet/catalog`                                    | Wells, speeds, model/provider readiness       |
| `POST /api/fleet/sessions`                                  | Start a run with `{ "speed": 60 }`            |
| `GET /api/fleet/sessions/{id}`                              | Current complete snapshot                     |
| `POST /api/fleet/sessions/{id}/control`                     | Pause, resume, step, cancel, speed, or inject |
| `GET /api/fleet/sessions/{id}/events?after=N`               | SSE journal after event cursor N              |
| `POST /api/fleet/sessions/{id}/incidents/{well_id}/actions` | Acknowledge, observation, or complete         |
| `GET /api/fleet/sessions/{id}/export`                       | Download snapshot and journal                 |
| `GET /api/fleet/results`                                    | Reproducible fleet evaluation                 |

The SSE event name is `fleet`; each envelope contains `id`, `type: "snapshot"`, and a full run in `payload`. The browser ignores older snapshots, reconnects through EventSource, and retains the session ID in localStorage for reload recovery. Sessions live in server memory; after a server restart the UI clears the missing session and offers a new replay.

## Run locally

First follow the root README to create the Python environment, fetch the verified dataset subset, and install backend dependencies. Build the fleet model from the repository root if it is missing:

```powershell
.venv/Scripts/python.exe scripts/train_fleet_model.py
```

Build the frontend:

```powershell
cd frontend
npm ci
npm run build
cd ..
./scripts/start_dashboard.ps1
```

Open [Frostline](http://127.0.0.1:8000/). `npm run build` checks TypeScript and writes the ignored `frontend/dist/` directory served by FastAPI.

For frontend development, keep the backend running and use `npm run dev` from `frontend/`. Vite serves [port 5173](http://127.0.0.1:5173/) and proxies `/api` and `/stream` to port 8000.

See the root README for the dataset protocol, backend setup, and validation limits.
