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
    session.operator_action(well, "complete", "Review completed; waiting for engineering assessment.")
    assert session.wells[well]["status"] == "attention"
    assert "waiting for engineering" in session.wells[well]["incident"]["note"]


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
        s = FleetSession(use_llm=True); SESSIONS[s.id] = s
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
        session = FleetSession(use_llm=True)
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
        first, other = FleetSession(use_llm=True), FleetSession(use_llm=True)
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
        newly_opened = FleetSession(use_llm=True)
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
        session = FleetSession(use_llm=True)
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
        s = FleetSession(use_llm=True); SESSIONS[s.id] = s
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


def test_local_checks_complete_without_provider_and_paused_operator_flow_is_explicit(monkeypatch):
    import src.fleet_llm

    async def forbidden(*args, **kwargs):
        raise AssertionError("Deferred LLM must never be called")

    monkeypatch.setattr(src.fleet_llm, "investigate", forbidden)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-no-network")

    async def scenario():
        session = FleetSession()
        SESSIONS[session.id] = session
        await session.prepare()
        session.advance()
        well = "WELL-00006"
        # Starting a new incident must not display an earlier recovered follow-up.
        session.wells[well]["followup"] = {"last_checked_at": "old", "after": {"summary": "Old recovered incident"}}
        session.open_incident(well)
        assert session.wells[well]["followup"]["last_checked_at"] is None
        assert "after" not in session.wells[well]["followup"]
        await session.investigate_well(well)
        item = session.wells[well]
        assert item["investigation"] == "complete"
        assert session.requests_used == 0 and session.snapshot()["llm_deferred"]
        assert all(t["status"] == "done" for t in item["assessment"]["tools"])
        incident_id = item["incident"]["id"]
        completed_trigger = item["followup"]["trigger"]
        before = deepcopy(session.wells[WELLS[0]])
        session.control("pause")
        session.operator_action(well, "acknowledge", incident_id=incident_id)
        check = item["assessment"]["checks"][0]
        session.operator_action(well, "check", check_id=check["id"], check_definition=check["definition"], result="unavailable", note="Independent reading unavailable", incident_id=incident_id)
        session.operator_action(well, "recheck", incident_id=incident_id)
        assert item["followup"]["trigger"] == completed_trigger
        assert well in session.pending and item["investigation"] == "queued"
        index = session.index
        session.dispatch()
        assert not session.jobs and session.index == index
        session.control("resume")
        while session.pending:
            session.dispatch()
            await asyncio.gather(*list(session.jobs.values()))
        assert item["followup"]["before"] and item["followup"]["after"]
        assert "no new measurements" in item["followup"]["change_summary"]
        assert item["followup"]["trigger"] == "Operator requested an evidence recheck"
        assert item["assessment"]["checks"][0]["status"] == "unavailable"
        assert item["status"] == "watch" and not item["incident"]["condition_cleared"]
        kinds = [event["kind"] for event in item["timeline"]]
        assert all(kind in kinds for kind in ["detected", "seen", "checked", "recheck"])
        assert all(event["at"] == item["source_timestamp"] for event in item["timeline"])
        assert session.wells[WELLS[0]]["incident"] == before["incident"]
        assert not any(e["kind"] in {"seen", "checked"} for e in session.wells[WELLS[0]]["timeline"])
        session.control("cancel")
        with pytest.raises(ValueError, match="ended"):
            session.operator_action(well, "recheck", incident_id=incident_id)
        session.operator_action(well, "observation", "End-of-run note", incident_id=incident_id)
        assert not session.pending

    asyncio.run(scenario())


def test_action_api_rejects_stale_incidents_and_invalid_checks_without_mutation():
    session = ready()
    session.advance()
    item = session.wells["WELL-00006"]
    original = deepcopy(item)
    path = f"/api/fleet/sessions/{session.id}/incidents/WELL-00006/actions"
    with TestClient(app) as client:
        assert client.post(path, json={"action": "acknowledge", "incident_id": "old"}).status_code == 422
        assert client.post(path, json={"action": "check", "check_id": "unknown", "result": "confirmed"}).status_code == 422
        assert client.post(path, json={"action": "check", "check_id": "unknown", "result": "unsafe"}).status_code == 422
        assert item == original
        check = item["assessment"]["checks"][0]
        reply = client.post(path, json={"action": "check", "incident_id": item["incident"]["id"], "check_id": check["id"], "check_definition": check["definition"], "result": "not_confirmed"})
        assert reply.status_code == 200
        assert item["incident"]["checks"][check["id"]]["status"] == "not_confirmed"
        journal = client.get(f"/api/fleet/sessions/{session.id}/export").json()
        assert journal["audit"][-2]["human_report_not_ground_truth"]


def test_changed_check_targets_require_new_finding_and_reject_stale_ui():
    session = ready()
    session.sources = deepcopy(session.sources)
    session.advance()
    well = "WELL-00006"
    item = session.wells[well]
    old = deepcopy(item["assessment"]["checks"][0])
    session.operator_action(well, "check", check_id=old["id"], result="confirmed", check_definition=old["definition"])
    assert item["assessment"]["checks"][0]["status"] == "confirmed"
    future = session.sources[well]["future"]
    future.loc[future.index[1], "T-TPT"] = float("nan")
    future.loc[future.index[1], "invalid_T-TPT"] = 1.
    session.advance()
    new = item["assessment"]["checks"][0]
    assert new["id"] == old["id"] and new["definition"] != old["definition"]
    assert new["status"] == "pending" and "T-TPT" in new["label"]
    with pytest.raises(ValueError, match="check changed"):
        session.operator_action(well, "check", check_id=old["id"], result="confirmed", check_definition=old["definition"])
    assert any(old["label"] in e["summary"] for e in item["timeline"] if e["kind"] == "checked")


def test_local_review_uses_one_timestamp_even_with_a_tick_waiting():
    async def scenario():
        session = FleetSession()
        await session.prepare()
        session.advance()
        well = "WELL-00006"

        async def tick():
            session.advance()

        task = asyncio.create_task(tick())
        await session.investigate_well(well)
        item = session.wells[well]
        assert item["last_assessed"] == item["followup"]["after"]["at"] == item["source_timestamp"]
        tool_events = [e for e in session.audit if e["kind"] in {"tool_call", "tool_result"} and e.get("well_id") == well]
        assert len(tool_events) == 8
        assert all(e["as_of"] == item["last_assessed"] for e in tool_events)
        await task
    asyncio.run(scenario())


def test_outage_review_trace_reports_abstention_instead_of_invented_scores():
    async def scenario():
        session = FleetSession()
        await session.prepare()
        session.control("inject", well_id=WELLS[0], fault="pressure_offline")
        session.advance()
        await session.investigate_well(WELLS[0])
        item = session.wells[WELLS[0]]
        assert item["status"] == "unavailable"
        trace = {t["name"]: t["summary"] for t in item["assessment"]["tools"]}
        assert "abstained" in trace["causal_model"]
        assert "unavailable" in trace["pressure_trends"]
        assert item["investigation"] == "complete"
    asyncio.run(scenario())


def test_recurring_checks_do_not_erase_incident_milestones():
    session = FleetSession()
    well = WELLS[0]
    session.wells[well]["source_timestamp"] = "2024-01-01T00:00:00"
    for kind in ["detected", "seen", "checked", "review_completed", "recovered"]:
        session.timeline(well, kind, f"Recorded {kind}")
    for minute in range(100):
        session.timeline(well, "recheck", f"Routine check {minute}")
    entries = session.wells[well]["timeline"]
    assert len(entries) == 80
    assert entries[0]["kind"] == "detected"
    assert entries[-1]["summary"] == "Routine check 99"
    assert {e["kind"] for e in entries} == {"detected", "seen", "checked", "review_completed", "recovered", "recheck"}
