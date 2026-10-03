import asyncio
from dataclasses import asdict
import json
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from src.live_agent import LiveSession, SensorAgent, SESSIONS, clean_tick, fingerprint
from src.research import Policy, SENSORS, experiment, load_seed, split_seed


def tick(hour, pressure=280, temp=85, flow=12):
    return {"t": (pd.Timestamp("2024-01-01") + pd.Timedelta(hours=hour)).isoformat(), "sensors": dict(zip(SENSORS, [pressure, temp, flow]))}


def consume(agent, reading):
    events = list(agent.observe(reading))
    return events, next(e["payload"] for e in events if e["type"] == "decision")


def test_tool_execution_happens_after_call_event_not_before():
    agent = SensorAgent(Policy(20, 10, 1, False, 277, 83, 11))
    stream = agent.observe(tick(0))
    assert not agent.history
    call = next(stream)
    assert call["type"] == "tool_call"
    assert agent.calls == 0
    assert next(stream)["type"] == "tool_result"
    assert agent.calls == 1


def test_tools_are_conditional_and_faults_change_decisions():
    policy = Policy(20, 10, 1, False, 277, 83, 11)
    healthy, decision = consume(SensorAgent(policy), tick(0))
    assert decision["decision"] == "DISMISS"
    assert [e["payload"]["tool"] for e in healthy if e["type"] == "tool_call"] == ["check_sensor_quality"]
    anomaly, decision = consume(SensorAgent(policy), tick(0, 265, 81, 9))
    assert decision["decision"] == "ALERT"
    assert decision["notify"]
    assert len([e for e in anomaly if e["type"] == "tool_call"]) == 4
    offline, decision = consume(SensorAgent(policy), tick(0, None))
    assert decision["decision"] == "WATCH"
    assert [e["payload"]["tool"] for e in offline if e["type"] == "tool_call"] == ["check_sensor_quality", "inspect_sensor_history"]
    _, dip = consume(SensorAgent(policy), tick(0, 265))
    assert dip["decision"] == "WATCH"


def test_agent_rejects_labels_and_out_of_order_readings():
    agent = SensorAgent(Policy(10, None, 1, False, 277))
    with pytest.raises(ValueError, match="sensors only"):
        list(agent.observe({**tick(0), "label": 1}))
    consume(agent, tick(0))
    with pytest.raises(ValueError, match="advance in time"):
        consume(agent, tick(0))


def test_rechecks_run_at_scheduled_sensor_time_and_recover():
    agent = SensorAgent(Policy(20, 10, 1, False, 277, 83, 11))
    _, first = consume(agent, tick(0))
    assert first["next_recheck"] == "2024-01-01T04:00:00"
    for h in range(1, 4):
        events, _ = consume(agent, tick(h))
        assert len([e for e in events if e["type"] == "tool_call"]) == 1
    # Quiet monitoring must NOT move the prior recheck deadline on every tick.
    events, _ = consume(agent, tick(4))
    assert any(e["type"] == "trigger" and e["payload"]["reason"] == "recheck" for e in events)
    _, alert = consume(agent, tick(5, 265, 81, 9))
    assert alert["decision"] == "ALERT"
    events, recovery = consume(agent, tick(6))
    assert recovery["diagnosis"] == "Conditions recovered"
    assert any(e["type"] == "trigger" for e in events)


def test_freeze_quality_and_notification_cooldown():
    agent = SensorAgent(Policy(20, 10, 1, False, 277, 83, 11))
    notifications = []
    for i in range(8):
        _, decision = consume(agent, tick(i, 265 + i / 10, 81 + i / 10, 9 + i / 100))
        if decision["notify"]:
            notifications.append(i)
    assert notifications == [0, 6]


def test_frozen_feed_changes_to_quality_investigation():
    agent = SensorAgent(Policy(20, 10, 1, False, 277, 83, 11))
    for i in range(6):
        events, decision = consume(agent, tick(i))
    assert decision["decision"] == "WATCH"
    assert events[1]["payload"]["result"]["frozen"] == SENSORS


def test_full_incremental_mission_matches_frozen_research():
    async def run():
        session = LiveSession(paced=False)
        session.control("resume")
        await session.run()
        return session
    session = asyncio.run(run())
    report = experiment()
    assert session.status == "completed", session.error
    assert len(session.trials) == 24
    assert session.tool_calls == sum(e["type"] in ["tool_result", "candidate_result"] for e in session.events)
    assert session.selection["trial"] == report["selection"]["selected_trial"]
    assert session.selection["policy_id"] == report["policy_id"]
    measured = report["systems"][-1]["test"]
    assert {k: session.score[k] for k in measured} == measured
    # Every evaluator outcome follows a decision, never a forecast of future rows.
    for i, e in enumerate(session.events):
        if e["type"] == "outcome":
            decision = next(x for x in reversed(session.events[:i]) if x["type"] == "decision")
            assert decision["payload"]["t"] == e["payload"]["t"]
    json.dumps(session.snapshot(), allow_nan=False)


def test_final_test_labels_cannot_change_promotion_or_decisions():
    async def run(data):
        session = LiveSession(data=data, paced=False)
        session.control("resume")
        await session.run()
        return session
    original = load_seed()
    changed = original.copy()
    changed.loc[480:, "label"] = 1 - changed.loc[480:, "label"]
    a, b = asyncio.run(run(original)), asyncio.run(run(changed))
    assert a.selection == b.selection
    assert [e["payload"] for e in a.events if e["type"] == "decision"] == [e["payload"] for e in b.events if e["type"] == "decision"]
    assert a.score != b.score


def test_pause_step_cancel_and_future_input_injection():
    async def run():
        session = LiveSession("sandbox", paced=False)
        task = asyncio.create_task(session.run())
        await asyncio.sleep(0)
        assert session.tick_index == 0
        session.control("inject", fault="pressure_offline")
        session.control("step")
        assert session.snapshot()["stepping"]
        with pytest.raises(ValueError, match="already executing"):
            session.control("step")
        await asyncio.sleep(0)
        assert session.tick_index == 1
        assert session.status == "paused"
        assert not session.snapshot()["stepping"]
        assert session.last_decision["decision"] == "WATCH"
        assert session.frames[0]["sensors"]["pressure_bar"] is None
        assert session.score is None
        assert session.snapshot()["injected_excluded"] == 1
        assert session.snapshot()["tool_calls"] == 2
        # Later faults and readings must never rewrite earlier journal entries.
        intervention = next(e for e in session.events if e["type"] == "intervention")
        assert intervention["payload"]["fault"]["remaining"] == 4
        first_tick = next(e for e in session.events if e["type"] == "tick")
        assert first_tick["payload"]["decision"] is None
        assert first_tick["payload"]["fault"]["remaining"] == 3
        frozen_events = len(session.events)
        await asyncio.sleep(0)
        assert len(session.events) == frozen_events
        session.control("inject", fault="restore")
        session.control("step")
        await asyncio.sleep(0)
        assert session.tick_index == 2
        assert session.frames[1]["injection"] is None
        assert session.frames[1]["ground_truth"] is None  # tainted prior history
        assert session.snapshot()["injected_excluded"] == 2
        assert first_tick["payload"]["fault"]["remaining"] == 3
        session.control("cancel")
        await task
        assert session.status == "cancelled"
        assert session.tick_index == 2
    asyncio.run(run())


def test_live_api_controls_and_sse_reconnect_cursor():
    with TestClient(app) as client:
        created = client.post("/api/live/sessions", json={"scenario": "sandbox", "speed": 8})
        assert created.status_code == 201
        id = created.json()["id"]
        base = f"/api/live/sessions/{id}"
        assert client.get(base).json()["tick_index"] == 0
        assert client.post(base + "/control", json={"action": "speed", "speed": 3}).status_code == 422
        assert client.post(base + "/control", json={"action": "cancel"}).status_code == 200
        snapshot = client.get(base).json()
        response = client.get(base + "/events?after=1")
        parsed = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        assert parsed and min(e["id"] for e in parsed) > 1
        assert len({e["id"] for e in parsed}) == len(parsed)
        resumed = client.get(base + "/events?after=1", headers={"Last-Event-ID": str(parsed[-1]["id"])})
        newer = [json.loads(line[6:]) for line in resumed.text.splitlines() if line.startswith("data: ")]
        assert all(e["id"] > parsed[-1]["id"] for e in newer)
        assert client.get(base + "/events?after=999999").status_code == 422
        assert client.get(base + "/events", headers={"Last-Event-ID": "invalid"}).status_code == 422
        assert client.post(base + "/control", json={"action": "resume"}).status_code == 409
        export = client.get(base + "/export")
        assert export.status_code == 200
        assert 'attachment;' in export.headers["content-disposition"]
        assert export.json()["events"][-1]["id"] >= snapshot["last_event_id"]
        assert client.get("/api/live/sessions/missing").status_code == 404
        assert client.post("/api/live/sessions", json={"scenario": "not-real"}).status_code == 422
