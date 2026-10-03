# Merge Plan — agent-backend + mico-branch

## Branch Summary

| Branch | Owner | Focus |
|--------|-------|-------|
| `agent-backend` (ours) | Div | LLM agent pipeline, fleet replay, watcher, tools, multi-provider LLM |
| `mico-branch` (Mico) | Dashboard | Research API, live agent dashboard, seed/3W scoring, frontend |

## Endpoint Overlap

| Endpoint | Ours | Theirs | Conflict? |
|----------|------|--------|-----------|
| `GET /health` | ✓ (capabilities, providers) | `GET /api/health` (mode, real_pilot) | No — different paths |
| `GET /wells` | ✓ (list instances) | — | No conflict |
| `GET /stream/{id}` | ✓ (SSE single-well) | — | No conflict |
| `GET /results` | ✓ (eval table) | ✓ (research report) | **Conflict** — both define `/results` |
| `POST /tts` | ✓ (stub) | — | No conflict |
| `POST /fleet/sessions` | ✓ (fleet replay) | — | No conflict |
| `GET /fleet/sessions/{id}/stream` | ✓ (fleet SSE) | — | No conflict |
| `POST /api/live/sessions` | — | ✓ (their session system) | No conflict |
| `POST /api/live/sessions/{id}/control` | — | ✓ (pause/resume/inject) | No conflict |
| `GET /api/live/sessions/{id}/events` | — | ✓ (their SSE) | No conflict |
| `GET /api/research` | — | ✓ (seed + real report) | No conflict |

## File Conflicts

| File | Both changed? | Resolution |
|------|--------------|------------|
| `backend/main.py` | **Yes** — they rewrote it entirely | Use ours as base, add their `/api/live` router import |
| `CLAUDE.md` | Yes (minor) | Manual merge, keep both additions |
| `README.md` | Yes | Manual merge |
| `eval.py` | They modified | No conflict (not ours) |

## Session Model Comparison

| Feature | Our FleetSession | Their Session (live_agent.py) |
|---------|-----------------|------------------------------|
| Multi-well | ✓ (4 wells, shared clock) | ✗ (single scenario) |
| LLM agent | ✓ (tool-calling, Gemini/Groq) | ✗ (deterministic controller) |
| Watcher | ✓ (per-well, cooldown, recheck) | ✓ (policy-based threshold) |
| Pause/resume | ✓ | ✓ |
| SSE events | ✓ (fleet_tick, assessment, etc.) | ✓ (different event schema) |
| Recording | ✓ (JSONL) | ✗ |
| Fault injection | ✗ | ✓ (pressure_dip, frozen_feed) |
| Priority queue | ✓ | ✗ |

## Recommended Plan

1. **Base: our `backend/main.py`** — it has all working routes plus fleet endpoints.
2. **Port their router**: Add `from backend.live import router as live_router` and `app.include_router(live_router)` to our main.py. Their `backend/live.py` and `src/live_agent.py` can coexist alongside our code.
3. **Fix `/results` conflict**: Keep ours at `/results`, move theirs to `/api/research` (they already have that route).
4. **Frontend**: Their frontend (`frontend/src/LiveMission.tsx`) currently calls `/api/live/*` endpoints — those will work as-is via the router include. For fleet view, they'd build a new component using `docs/api_fleet.md`.
5. **No need to merge session models** — they serve different purposes (their deterministic demo vs our LLM fleet).

## Steps

1. Merge `main` into both branches first
2. Merge `agent-backend` into `main`
3. Cherry-pick `mico-branch` commit, resolve `backend/main.py` conflict (keep ours, add router import)
4. Test: `pytest -m "not pending"` + their tests

## Risks

- Their `backend/main.py` is a full rewrite — git will show a conflict. Manual resolution needed but straightforward (add their router to our file).
- Their `src/live_agent.py` imports `src.research` which has its own dependencies (`requirements-research.txt`). Need to install those.
- Their frontend is self-contained and won't conflict with our backend changes.
