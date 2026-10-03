# Frostline dashboard

React, TypeScript, Vite, Recharts and Lucide. Measurements and decisions come from the Python API; the legacy `mocks/` files are unused. Fonts are bundled for offline demos.

## Operator view

`FleetDashboard.tsx` is the default **Well map**. It shows four independent Petrobras 3W well recordings, a selectable map, pressure/temperature/model-score trends, the priority queue, and one concise assessment with an operator next step. The wide layout keeps trends under the map and decisions at the right; narrow screens show the map, priorities and assessment, then trends.

The map is illustrative: source recordings have independent timestamps and no shared geography. Original timestamps remain in Evidence; charts use elapsed replay minutes, with negative minutes showing prior history. Displayed dates preserve source time without claiming a source timezone. Pressure is supplied in bar and temperature in °C. Missing measurements stay visible as gaps. The model score is not a calibrated probability.

- **Guided demo** is the default. **Start walkthrough** prepares a paused overview with prior-history charts; current readings have not been assessed yet. **Next important moment** processes every intervening minute until a detected change has been assessed, then pauses. **Continue demo** advances again. The selected well stays fixed; **View Well …** opens the affected well when requested.
- The guided banner distinguishes **Historical replay paused** from actual scanning or evidence checking. Checkpoints use observed changes, not future labels. **Stop scanning** pauses advancement. Reading evidence, acknowledging an incident, or saving a finding never resumes the replay.
- **Continuous replay** offers a comfortable pace of one data-minute reading every five seconds, plus faster two-second, one-second, and half-second intervals. Its speed selector controls playback timing, not measurements. Switching modes pauses first. **Read 1 minute** advances one source minute and settles its evidence checks before pausing. **Restart** creates a fresh run.
- Selecting a map well or priority row updates the same charts and assessment. Evidence includes data quality, source file/timestamp, available tool history, and the operator record.
- **Acknowledge** records that the incident was seen. The button shows loading only while that save is pending, followed by persistent confirmation. It does not clear the condition or trigger a sensor check. Observations are human reports, not evaluation labels. Completing a review preserves the replay controls; monitoring continues when the replay runs.
- **Operator checks** expands a small assessment-specific checklist. Choose Confirmed, Not confirmed, or Unable to check, add an optional note, and save each finding. Saved findings survive new snapshots and reloads while the server session exists. They can trigger another evidence review, but cannot clear measured conditions.
- **Recheck now** requests another evidence review. While paused, **Queue recheck** saves the request. In Guided demo, **Continue demo** first checks queued findings against the same readings and pauses at that follow-up; another Continue advances. In Continuous replay, **Resume automatic checks** runs queued work and resumes readings. An in-flight review can finish after a manual pause. Ended runs accept operator records but cannot schedule new checks; start a new replay to continue.
- The incident milestones show recorded Detected, Seen, Checked, Recheck, and Recovered events independently. Times are actual source-clock positions; missing milestones stay blank. Evidence contains the full incident history and the latest assessment's before/after evidence. Checklist save times use the browser's local clock for the actual save, clearly labelled separately.
- **Try a telemetry fault** changes incoming pressure availability for the selected well. Injected inputs are excluded from published accuracy claims.

## Agent modes and results

Each assessment identifies its source: **live LLM**, **Model + rules**, or **recorded**. The current fleet workflow deliberately defers external LLM investigations and exposes that state through `llm_deferred`; automatic numerical and rule-based evidence checks remain active. The fleet UI shows the deferred state without prompting the operator to configure a key. A configured provider is not proof that a live call succeeded. Credentials belong in the server's ignored `.env`, never in `VITE_*` variables or browser code. The prototype provides operator advice; it does not operate plant controls.

**Results** begins with the real-well fleet evaluation: training/evaluation recording counts, grouped validation, candidate decisions, frozen policy, catches, false alerts, missed minutes, detection delay, and limitations. It reports regressions alongside improvements. The older synthetic seed experiment is separate below it.

The collapsed **Classifier comparison** distinguishes classifier selection from alarm-policy tuning and identifies retained incumbents honestly. The separate **Operator workflow** panel reads the saved workflow evaluation. Real-recording attention/watch burden and event coverage are shown separately from scripted lifecycle pass/fail checks. A watch or attention overlapping an event is not proof of a correct diagnosis. The panel makes no claim of LLM performance or field validation.

**More → Seed sandbox** retains `LiveMission.tsx` / `WellOverview.tsx` and the one-well synthetic `/api/live` workflow. Recorded replays, the earlier real-well pilot, and methodology are also available under More. These views are not the source of the four-well operating feed.

## API and session behavior

`fleetTypes.ts` defines the browser contract. The server owns replay progression, inference, investigation scheduling, incident state, and the journal.

| Endpoint                                                    | Purpose                                                          |
| ----------------------------------------------------------- | ---------------------------------------------------------------- |
| `GET /api/fleet/catalog`                                    | Wells, default mode, speeds, model/provider readiness            |
| `POST /api/fleet/sessions`                                  | Start with `{ "mode": "guided", "speed": 12 }`                   |
| `GET /api/fleet/sessions/{id}`                              | Current complete snapshot                                        |
| `POST /api/fleet/sessions/{id}/control`                     | Next moment, mode, pause, resume, step, cancel, speed, or inject |
| `GET /api/fleet/sessions/{id}/events?after=N`               | SSE journal after event cursor N                                 |
| `POST /api/fleet/sessions/{id}/incidents/{well_id}/actions` | Acknowledge, observation, complete, check, or recheck            |
| `GET /api/fleet/sessions/{id}/export`                       | Download snapshot and journal                                    |
| `GET /api/fleet/results`                                    | Reproducible fleet evaluation                                    |
| `GET /api/fleet/workflow-results`                           | Saved workflow evaluation and scripted checks                    |

Action bodies include the currently viewed `incident_id` to reject a save against a superseded incident. A checklist submission uses `{ "action": "check", "incident_id": "…", "check_id": "…", "check_definition": "…", "result": "confirmed", "note": "…" }`; a recheck uses `{ "action": "recheck", "incident_id": "…" }`. Check results are `confirmed`, `not_confirmed`, or `unavailable`. The definition hash prevents a stale answer being applied when the required sensor check changes.

Guided progression uses `{ "action": "next_moment" }`; a mode switch uses `{ "action": "mode", "mode": "continuous" }`. Each snapshot includes `mode` and `guide.phase` (`overview`, `seeking`, `assessing`, `checkpoint`, `manual_pause`, or `ended`), plus an optional checkpoint with its actual reason, affected well, and source timestamp. The browser renders these server states directly.

The SSE event name is `fleet`; each envelope contains `id`, `type: "snapshot"`, and a full run in `payload`. The browser ignores older snapshots, reconnects through EventSource, and retains the session ID in localStorage for reload recovery. Sessions live in server memory; after a server restart the UI clears the missing session and offers a new replay.

## Run locally

First follow the root README to create the Python environment, fetch the verified dataset subset, and install backend dependencies. Build the fleet model from the repository root if it is missing:

```powershell
.venv/Scripts/python.exe scripts/train_fleet_model.py
.venv/Scripts/python.exe scripts/evaluate_fleet_workflow.py
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
