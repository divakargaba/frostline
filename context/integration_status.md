# Integration Status — 2026-10-03

## Branches

| Branch | Author | Last Commit | Ahead/Behind main |
|--------|--------|-------------|-------------------|
| agent-backend | divakargaba | Add recorded demo runs and updated candidates | 0 / 0 |
| data-ml | Zaine2k | Updated claude.md | 1 / 11 |
| mico-branch | micobenissa | Build live agent dashboard and reproducible research workflow | 1 / 10 |

## What came in from each branch

### data-ml (Zaine2k)
- `CLAUDE.md` — added "Scaling to Multiple Wells and Pipes" section with fleet Mermaid diagram
- `context/claude.md` — replaced minimal context with full Project Bible (copy of CLAUDE.md content)
- `logs/load.log` — 3113-line loader output log
- `logs/load_run.out` — 1548-line loader run output

**Merge status:** Clean merge into agent-backend. 113 tests pass.

### mico-branch (micobenissa)
- `.gitignore`, `CLAUDE.md`, `README.md` — shared files (modified)
- `backend/main.py` — **complete rewrite** with research-oriented API (`/api/research`, `/api/scenarios`, static file serving)
- `backend/live.py` — new live agent router
- `eval.py` — modified
- `frontend/` — full Vite+React+TS dashboard (`LiveMission.tsx`, `types.ts`, `vite.config.ts`, etc.)
- `src/live_agent.py`, `src/real_pilot.py`, `src/research.py` — new research/pilot modules
- `tests/test_live_agent.py`, `tests/test_real_pilot.py`, `tests/test_research.py`, `tests/test_research_api.py` — new tests
- `requirements-research.txt`, `scripts/fetch_research_data.py`, `scripts/start_dashboard.ps1`

**Merge status:** CONFLICT in `backend/main.py` (3 conflict regions). mico-branch completely replaced the backend with a different API structure. Merge aborted. Needs manual resolution — their research routes and our replay/agent routes must coexist.

**Conflicting files and ownership:**
- `backend/main.py` — Div's section. Their version replaces all replay/agent/watcher/SSE logic with research replay endpoints. Must keep our version and selectively add their research routes.

## Physics: src/physics.py

All three functions raise `NotImplementedError`:

| Function | Signature | Status |
|----------|-----------|--------|
| `hydrate_temp_towler_mokhatab` | `(p_psia: float, sg: float) -> float` | NOT IMPLEMENTED |
| `hydrate_margin` | `(p_bar: float, t_c: float, sg: float = 0.65) -> float` | NOT IMPLEMENTED |
| `hammerschmidt_dose` | `(delta_t_c: float, inhibitor: str = "methanol") -> float` | NOT IMPLEMENTED |

## Forecast: src/forecast.py

All three functions raise `NotImplementedError`:

| Function | Signature | Status |
|----------|-----------|--------|
| `train_quantile_model` | `(feature_df: pd.DataFrame) -> object` | NOT IMPLEMENTED |
| `physics_extrapolation` | `(window_df: pd.DataFrame, sg: float = 0.65) -> dict` | NOT IMPLEMENTED |
| `forecast_onset` | `(window_df: pd.DataFrame, model: object = None) -> dict` | NOT IMPLEMENTED |

## Model: src/model.py

All four functions raise `NotImplementedError`:

| Function | Signature | Status |
|----------|-----------|--------|
| `train` | `(feature_df: pd.DataFrame, target_col: str = "phase") -> object` | NOT IMPLEMENTED |
| `predict` | `(window_df: pd.DataFrame, model: object = None) -> pd.DataFrame` | NOT IMPLEMENTED |
| `save_model` | `(model: object, path: str) -> None` | NOT IMPLEMENTED |
| `load_model` | `(path: str) -> object` | NOT IMPLEMENTED |

**No saved model file** exists in `models/` (no `.pkl`, `.joblib`, or similar).

## Test results

```
113 passed, 20 deselected (pending = physics/eval/model tests)
```

The 20 deselected are marked `@pytest.mark.pending` — all in `tests/test_physics.py`, `tests/test_forecast.py`, `tests/test_eval.py`, `tests/test_model.py`.

## /health availability flags

```json
{
  "has_processed": false,
  "has_raw": true,
  "has_seed": true,
  "has_physics": false,
  "has_model": true,
  "llm_mode": "openrouter"
}
```

- `has_physics: false` — physics.py functions all raise NotImplementedError
- `has_model: true` — this flag checks if `src/model.py` exists, not whether the model is trained. The model is NOT trained and predict() raises NotImplementedError
- `has_processed: false` — no processed parquet files in data/processed/

## Safe to merge into main?

- **data-ml:** Yes, safe. Only adds docs (CLAUDE.md updates) and log files. No code changes. Already merged into agent-backend and tests pass.
- **mico-branch:** No, not yet. `backend/main.py` conflict must be resolved manually. Their research API (`/api/research`, `/api/scenarios`) and our replay/agent API (`/stream`, `/wells`, `/health`) need to coexist in the same file. Their frontend additions (new files) would merge cleanly; only `backend/main.py` conflicts.

## Impact on agent pipeline

The agent currently works end-to-end using heuristic fallbacks for all three missing modules:
- `classify_event` → heuristic pressure/temperature trend classifier
- `hydrate_margin` → returns `available: false`
- `forecast_onset` → returns `available: false`
- `methanol_dose` → returns `available: false`

Once physics.py and model.py are implemented, the agent tools will auto-detect and use them with no code changes needed. The tool dispatch in `src/tools.py` already has try/except blocks that fall back gracefully.
