# Merge Plan — integration + mico-branch

## Mico's Fleet Architecture (a9e61c2)

His `backend/fleet.py` at `/api/fleet` is a guided-demo replay engine:
1. **Trigger**: `src/fleet_policy.py` runs causal pressure/model checks each tick (no LLM).
2. **Evidence**: 4 local tools (`sensor_quality`, `causal_model`, `pressure_trends`, `incident_followup`) — all pure numerical, no RAG or playbook.
3. **LLM path** (optional, `use_llm=True`): `src/fleet_llm.py` calls OpenRouter (nvidia/nemotron free, qwen fallback). 3 attempts, 30 s deadline, 8 max tool calls. Returns `{status, diagnosis, actions, evidence, recheck_minutes}`.
4. **Budget**: per-session `attempt_budget` (default ~18), circuit-breaker on provider failure.
5. **Output**: SSE events via `FleetSession.publish()` — well snapshots with `status/diagnosis/tools/assessment/activity`.
6. **Model**: `src/fleet_model.py` trains its own LGBMClassifier on `data/raw/research/` (separate data copy). Outputs `models/fleet_model.pkl`.
7. **Playbook**: 4 inline advisory entries in `fleet_llm.py` (sensor_review, hydrate_review, restriction_review, followup_review). Not file-based.

## Touchpoints to Plug In Our Pieces

### (a) LLM provider pool (src/llm.py → src/fleet_llm.py)

Mico's `fleet_llm.py` uses raw `httpx` to OpenRouter. Ours uses `src/llm.py` with Gemini first (has quota), Groq, OpenRouter fallback, rate-limiting, and usage tracking.

**Plan**: Replace `fleet_llm._request()` with a wrapper that calls `src.llm.get_llm_client()`. Keep his tool schema and prompt. This gives us Gemini as primary (OpenRouter free is exhausted) and automatic rotation.

### (b) RAG playbook (src/rag.py → fleet_llm tools)

Mico has 4 inline playbook entries. Ours has 14 verified docs with BM25 in `src/rag.py`.

**Plan**: Add `search_playbook` as a tool in his `TOOL_SCHEMAS`. When the LLM calls it, route to `src.rag.search()`. Add returned `playbook_refs` IDs to his assessment output. The inline entries stay as fallback when RAG is unavailable.

### (c) One model for the fleet

| | Zaine's (src/model.py) | Mico's (src/fleet_model.py) |
|---|---|---|
| Training data | `data/processed/` (full 3W, 1506 files) | `data/raw/research/` (subset, ~40 files) |
| Classes | 5 fine → 3 grouped | 3 direct |
| CV | LOWO on 9+ real wells | GroupKFold on subset |
| Held out | WELL-00006, -00019, -00042 | Same demo wells |
| Smoothing | EWM span=10 | None |
| Warmup | 60 min | ~10 min |

**Recommendation**: Use **Zaine's model** (`models/lgbm_hydrate.pkl`). It trains on 10× more data, has proper LOWO validation, and holds out the demo wells. Mico's `fleet_model.py` can stay as a lightweight alternative but shouldn't be the default.

**Implementation**: In `fleet_policy.py` and `fleet_llm._model()`, load the scorer from `src.model.predict` instead of `src.fleet_model.load_bundle`. Both return `p_hydrate/p_lookalike/p_normal` so the interface is compatible.

## Merge Steps

1. `git merge --no-ff origin/mico-branch` on integration.
2. **backend/main.py**: Keep ours as base. Add `from backend.fleet import router as fleet_router; app.include_router(fleet_router)` (his fleet is at `/api/fleet`, ours at `/fleet` — no overlap).
3. **backend/fleet.py**: His replaces ours (we retire our fleet.py — rename ours to `backend/fleet_legacy.py` temporarily for reference).
4. **CLAUDE.md, README.md, eval.py**: Manual merge, keep both additions.
5. **New files** (no conflict): `src/fleet_llm.py`, `src/fleet_model.py`, `src/fleet_policy.py`, `src/live_agent.py`, `src/real_pilot.py`, `src/research.py`, `frontend/*`, `scripts/*`, `tests/test_fleet_*.py`.
6. Install `requirements-research.txt`.
7. Run full test suite. Fix import issues in my files only.
8. Wire LLM provider + RAG + Zaine's model (steps a/b/c above).

## Risks

- His `backend/fleet.py` is a full rewrite (904 lines) with different SSE event schema — dashboard expects his format.
- His fleet uses `data/raw/research/` which may not exist on this laptop (need to check / create symlinks).
- His tests import `src.fleet_policy`, `src.real_pilot` which have their own deps.
