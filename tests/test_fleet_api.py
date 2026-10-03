"""Fleet lifecycle, isolation, causal boundaries and asynchronous scheduling."""
import asyncio
from copy import deepcopy
import json
import math
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.fleet import ATTEMPTS, PROVIDER_CIRCUIT, SESSIONS, FleetSession, WELLS, source_frames


@pytest.fixture(autouse=True)
def isolated_sessions(monkeypatch):
    # Test key choices must not be overwritten by the developer's local .env.
    monkeypatch.setattr("backend.fleet.load_local_env", lambda: None)
    SESSIONS.clear()
    ATTEMPTS.clear()
    PROVIDER_CIRCUIT.update(until=0.0, reason="")
    yield
    SESSIONS.clear()
    ATTEMPTS.clear()
    PROVIDER_CIRCUIT.update(until=0.0, reason="")


def ready():
    session = FleetSession()
    SESSIONS[session.id] = session
    asyncio.run(session.prepare())
    return session


def test_four_real_feeds_are_distinct_and_truth_is_outside_runtime():
    session = ready()
    session.advance()
    assert len(session.wells) == 4
    assert len({w["source_file"] for w in session.wells.values()}) == 4
    for well in WELLS:
        snap = session.agent_snapshot(well)
        text = json.dumps(snap)
        assert all(k not in text for k in ["ground_truth", "t_form", "source_class", '"class"', '"state"'])
        assert all(row["t"] <= snap["as_of"] for row in snap["readings"])
        assert snap["model"]["version"] == session.bundle["model_id"]
    # Mutating a session's observed history cannot rewrite another run's source.
    original = source_frames()[WELLS[0]]["warmup"].iloc[-1]["P-TPT"]
    session.history[WELLS[0]].loc[session.history[WELLS[0]].index[-1], "P-TPT"] = -999
    assert source_frames()[WELLS[0]]["warmup"].iloc[-1]["P-TPT"] == original


def test_real_model_alarm_persists_and_acknowledgment_does_not_clear_it():
    session = ready()
    for _ in range(40):
        session.advance()
    well = "WELL-00019"
    assert session.wells[well]["status"] == "attention"
    assert session.state[well]["alarm"]["active"]
    incident = session.wells[well]["incident"]["id"]
    session.operator_action(well, "acknowledge", None)
    assert session.wells[well]["incident"]["acknowledged"]
    session.advance()
    assert session.wells[well]["status"] == "attention"
    assert session.wells[well]["incident"]["id"] == incident
    session.operator_action(well, "complete", None)
    assert session.wells[well]["status"] == "attention"


def test_injection_targets_future_input_of_only_selected_well():
    session = ready()
    session.advance()
    other_before = session.wells["WELL-00019"]["frames"][-1]["sensors"].copy()
    source_before = source_frames()[WELLS[0]]["future"].copy()
    session.control("inject", well_id=WELLS[0], fault="pressure_offline")
    session.advance()
    assert session.wells[WELLS[0]]["status"] == "unavailable"
    assert session.wells[WELLS[0]]["frames"][-1]["risk_score"] is None
    assert session.wells["WELL-00019"]["frames"][-1]["sensors"]["P-TPT"] is not None
    assert session.wells["WELL-00019"]["frames"][-2]["sensors"] == other_before
    assert source_frames()[WELLS[0]]["future"].equals(source_before)
    assert any(e.get("excluded_from_benchmark") for e in session.audit)


def test_quotas_are_shared_across_sessions_and_session_budget_cannot_overrun():
    async def scenario():
        a, b = FleetSession(), FleetSession()
        SESSIONS.update({a.id: a, b.id: b})
        for _ in range(10):
            assert await a.reserve_attempt()
        assert not await b.reserve_attempt()
        ATTEMPTS.clear()
        for _ in range(8):
            assert await a.reserve_attempt()
        assert not await a.reserve_attempt()
        assert a.requests_used == 18
        a.status = "cancelled"
        assert not await a.reserve_attempt()
    asyncio.run(scenario())


def test_slow_investigation_does_not_stop_feeds_and_superseded_result_is_ignored(monkeypatch):
    import src.fleet_llm
    async def scenario():
        gate = asyncio.Event()
        async def slow(snapshot, emit, **kwargs):
            await gate.wait()
            return {"brief": "An obsolete result", "source": "live_llm", "next_action": "Continue", "evidence": [], "recheck_minutes": 15}
        monkeypatch.setattr(src.fleet_llm, "investigate", slow)
        s = FleetSession(); SESSIONS[s.id] = s
        await s.prepare(); s.advance()
        for well in WELLS:
            s.queue(well, "Test simultaneous signals")
        s.dispatch()
        await asyncio.sleep(0)
        assert len(s.jobs) == 2
        working = next(iter(s.jobs))
        before = s.index
        s.advance()
        assert s.index == before + 1
        s.state[working]["revision"] += 1
        s.status = "paused"
        gate.set()
        await asyncio.gather(*list(s.jobs.values()))
        assert s.wells[working]["assessment"]["summary"] != "An obsolete result"
        assert working in s.pending
        assert s.status == "paused"
    asyncio.run(scenario())


def test_running_review_has_no_duplicate_and_rechecks_at_exact_five_minutes(monkeypatch):
    import src.fleet_llm

    async def scenario():
        gate = asyncio.Event()

        async def delayed(snapshot, emit, **kwargs):
            await gate.wait()
            return {"brief": "Monitor the unchanged invalid sensor channel.",
                    "source": "live_llm", "status": "watch", "next_action": "Verify telemetry.",
                    "evidence": [], "recheck_minutes": 5}

        monkeypatch.setattr(src.fleet_llm, "investigate", delayed)
        session = FleetSession()
        SESSIONS[session.id] = session
        await session.prepare()
        # Hold evidence stable so a new job can only be due to scheduling.
        session.sources = deepcopy(session.sources)
        well = "WELL-00006"
        future = session.sources[well]["future"]
        future.iloc[:10] = future.iloc[0].to_numpy()
        session.advance()
        session.pending.clear()
        revision = session.state[well]["revision"]
        job = asyncio.create_task(session.investigate_well(well))
        session.jobs[well] = job
        await asyncio.sleep(0)
        session.advance()
        assert session.state[well]["revision"] == revision
        assert well not in session.pending
        gate.set()
        await job
        assert well not in session.pending
        assert well not in session.jobs

        assessed_at = pd.Timestamp(session.wells[well]["source_timestamp"])
        due_at = pd.Timestamp(session.wells[well]["next_check"])
        assert due_at - assessed_at == pd.Timedelta(minutes=5)
        for _ in range(4):
            session.advance()
            assert well not in session.pending
            assert pd.Timestamp(session.wells[well]["source_timestamp"]) < due_at
        session.advance()
        assert pd.Timestamp(session.wells[well]["source_timestamp"]) == due_at
        assert well in session.pending

    asyncio.run(scenario())


@pytest.mark.parametrize("metadata,permanent", [
    ({"account_limited": True, "failure_reason": "Account rate limit"}, False),
    ({"failure_reason": "Provider reported a nonzero cost"}, True),
])
def test_provider_circuit_blocks_other_sessions_without_spending_attempts(monkeypatch, metadata, permanent):
    import src.fleet_llm

    async def scenario():
        async def limited(snapshot, emit, *, reserve_attempt, **kwargs):
            assert await reserve_attempt()
            return {"source": "rules_fallback", "status": "watch", "brief": "Numerical monitoring continues.",
                    "next_action": "Review telemetry.", "evidence": [], "recheck_minutes": 15,
                    "metadata": metadata}

        monkeypatch.setattr(src.fleet_llm, "investigate", limited)
        first, other = FleetSession(), FleetSession()
        SESSIONS.update({first.id: first, other.id: other})
        await first.prepare()
        first.advance()
        await first.investigate_well("WELL-00006")
        assert first.requests_used == 1
        if permanent:
            assert math.isinf(PROVIDER_CIRCUIT["until"])
        else:
            assert PROVIDER_CIRCUIT["until"] > time.monotonic() + 290
        assert PROVIDER_CIRCUIT["reason"]
        assert not await other.reserve_attempt()
        newly_opened = FleetSession()
        SESSIONS[newly_opened.id] = newly_opened
        assert not await newly_opened.reserve_attempt()
        assert other.requests_used == newly_opened.requests_used == 0
        assert len(ATTEMPTS) == 1
        assert first.snapshot()["agent_mode"] == "rules"
        if not permanent:
            PROVIDER_CIRCUIT["until"] = time.monotonic() - 1
            assert await other.reserve_attempt()
            assert other.requests_used == 1

    asyncio.run(scenario())


def test_configured_key_does_not_claim_live_mode_until_success(monkeypatch):
    import src.fleet_llm

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-no-network")

    async def scenario():
        source = "rules_fallback"

        async def controlled(snapshot, emit, **kwargs):
            return {"source": source, "status": "watch", "brief": "Evidence reviewed.",
                    "next_action": "Verify telemetry.", "evidence": [], "recheck_minutes": 5}

        monkeypatch.setattr(src.fleet_llm, "investigate", controlled)
        session = FleetSession()
        SESSIONS[session.id] = session
        assert session.snapshot()["llm_configured"]
        assert session.snapshot()["agent_mode"] == "rules"
        await session.prepare()
        session.advance()
        await session.investigate_well("WELL-00006")
        assert session.snapshot()["agent_mode"] == "rules"
        source = "live_llm"
        await session.investigate_well("WELL-00006")
        assert session.snapshot()["agent_mode"] == "live"
        assert session.wells["WELL-00006"]["assessment"]["source"] == "live"

    asyncio.run(scenario())


def test_missing_provider_is_explicit_and_keeps_numerical_risk(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    async def scenario():
        s = FleetSession(); SESSIONS[s.id] = s
        await s.prepare()
        for _ in range(40): s.advance()
        await s.investigate_well("WELL-00019")
        well = s.wells["WELL-00019"]
        assert well["status"] == "attention"
        assert well["assessment"]["source"] == "rules"
        assert well["investigation"] == "unavailable"
        assert s.requests_used == 0
        assert any(e["kind"] == "tool_call" for e in s.audit)
    asyncio.run(scenario())


def test_api_validation_export_and_reconnect_cursor():
    s = ready(); s.advance(); s.control("pause")
    with TestClient(app) as client:
        catalog = client.get("/api/fleet/catalog").json()
        assert catalog["model_ready"] and len(catalog["wells"]) == 4
        assert client.post("/api/fleet/sessions", json={"speed": 999}).status_code == 422
        original = client.get(f"/api/fleet/sessions/{s.id}").json()
        resumed = client.post(f"/api/fleet/sessions/{s.id}/control", json={"action": "resume"}).json()
        assert resumed["id"] == original["id"]
        assert resumed["index"] == original["index"]
        assert resumed["status"] == "running"
        s.control("cancel")
        journal = client.get(f"/api/fleet/sessions/{s.id}/export").json()
        assert journal["run"]["status"] == "cancelled"
        assert journal["audit"]
        last = len(s.events)
        stream = client.get(f"/api/fleet/sessions/{s.id}/events?after={last-1}").text
        assert stream.count("event: fleet") == 1
        assert client.get(f"/api/fleet/sessions/{s.id}/events?after={last+1}").status_code == 422
        assert client.post(f"/api/fleet/sessions/{s.id}/incidents/not-a-well/actions", json={"action":"acknowledge"}).status_code == 422
