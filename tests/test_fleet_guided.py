"""Guided replay lifecycle, using deterministic sensors and real monitor state.

These are synthetic control-flow tests, not classifier performance claims.
Every source minute still passes through the four independent monitor states.
"""
import asyncio
from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import pytest

import backend.fleet as fleet
import src.fleet_llm as fleet_llm
import src.fleet_model as fleet_model


ANNOTATIONS = {"class", "state", "label", "phase", "ground_truth"}


@pytest.fixture
def guided_rig(monkeypatch):
    fleet.SESSIONS.clear()
    fleet.ATTEMPTS.clear()
    fleet.PROVIDER_CIRCUIT.update(until=0., reason="")
    fleet.source_frames.cache_clear()
    monkeypatch.setattr(fleet, "TOTAL", 30)
    monkeypatch.setattr(fleet, "load_local_env", lambda: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-never-use-network")
    rig = {"calls": [], "event_enabled": True, "total": 30}
    bundle = {
        "excluded_wells": list(fleet.WELLS), "training_wells": ["WELL-TEST-TRAIN"],
        "model_id": "guided-synthetic-model", "policy": {
            "activation_threshold": .7, "persistence_minutes": 3,
            "recovery_threshold": .6, "recovery_minutes": 5,
        },
    }
    monkeypatch.setattr(fleet_model, "load_bundle", lambda: deepcopy(bundle))

    def prepare_recording(path):
        well = next(w for w in fleet.WELLS if w in Path(path).name)
        well_index = fleet.WELLS.index(well)
        start = pd.Timestamp(fleet.OFFSETS[well]) - pd.Timedelta(minutes=60)
        frame = pd.DataFrame(index=pd.date_range(start, periods=60 + rig["total"], freq="min"))
        for name, value in {
            "P-PDG": 100. + well_index, "T-PDG": 80., "P-TPT": 200.,
            "T-TPT": 75., "P-MON-CKP": 180., "P-JUS-CKP": 175.,
            "T-JUS-CKP": 60., "ABER-CKP": 50., "QGL": 0.,
        }.items():
            frame[name] = value
        # The operating session must strip these, even from its warm-up history.
        for name in ANNOTATIONS:
            frame[name] = 108
        return frame

    def infer_window(frame, frozen_bundle):
        assert not ANNOTATIONS.intersection(frame.columns)
        well = fleet.WELLS[int(round(frame["P-PDG"].iloc[-1] - 100))]
        minute = int((frame.index[-1] - pd.Timestamp(fleet.OFFSETS[well])).total_seconds() / 60)
        assert minute >= 0, "The overview must not secretly consume a replay minute"
        assert frame.index.is_monotonic_increasing and not frame.index.has_duplicates
        rig["calls"].append((well, minute, tuple(frame.index)))
        elevated = rig["event_enabled"] and well == "WELL-00019" and 3 <= minute <= 14
        hydrate, other = (.85 if elevated else .1), .05
        return {
            "scores": {"normal": 1 - hydrate - other, "hydrate": hydrate, "lookalike": other},
            "quality": {"blocked": False, "status": "good", "missing": [], "invalid": [],
                        "unchanged": [], "available_pressure": list(fleet.PRESSURES),
                        "line_pressure_available": True, "summary": "Three usable pressure channels."},
            "evidence": {"timestamp": frame.index[-1].isoformat(), "model_id": frozen_bundle["model_id"]},
        }

    async def forbidden_provider(*args, **kwargs):
        raise AssertionError("A public guided replay must never invoke the deferred LLM")

    monkeypatch.setattr(fleet_model, "prepare_recording", prepare_recording)
    monkeypatch.setattr(fleet_model, "infer_window", infer_window)
    monkeypatch.setattr(fleet_llm, "investigate", forbidden_provider)
    yield rig
    fleet.source_frames.cache_clear()
    fleet.SESSIONS.clear()
    fleet.ATTEMPTS.clear()
    fleet.PROVIDER_CIRCUIT.update(until=0., reason="")


async def until(predicate, timeout=3.):
    async def wait():
        while not predicate():
            await asyncio.sleep(.001)
    await asyncio.wait_for(wait(), timeout)


async def start_guided():
    session = fleet.FleetSession(speed=12, mode="guided")
    fleet.SESSIONS[session.id] = session
    session.task = asyncio.create_task(session.run())
    await until(lambda: session.status == "paused" and session.snapshot()["guide"]["phase"] == "overview")
    return session


async def close(session):
    if session.status not in fleet.TERMINAL:
        session.control("cancel")
    tasks = [t for t in [session.task, *session.jobs.values()] if t is not None]
    if tasks:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 2.)


async def next_checkpoint(session):
    previous = (session.snapshot()["guide"].get("checkpoint") or {}).get("id")
    session.control("next_moment")
    await until(lambda: session.status in fleet.TERMINAL or (
        session.status == "paused"
        and session.snapshot()["guide"]["phase"] == "checkpoint"
        and (session.snapshot()["guide"].get("checkpoint") or {}).get("id") != previous))
    return session.snapshot()["guide"].get("checkpoint")


def assert_causal_progress(session, calls):
    assert len(calls) == len(fleet.WELLS) * session.index
    for well in fleet.WELLS:
        seen = [(minute, prefix) for name, minute, prefix in calls if name == well]
        assert [minute for minute, _ in seen] == list(range(session.index))
        for minute, prefix in seen:
            expected = pd.Timestamp(fleet.OFFSETS[well]) + pd.Timedelta(minutes=minute)
            assert prefix[-1] == expected
            assert all(stamp <= expected for stamp in prefix)
        assert not ANNOTATIONS.intersection(session.history[well].columns)
    assert session.requests_used == 0 and session.snapshot()["llm_deferred"]


def test_api_defaults_and_overview_freeze_before_first_reading(guided_rig):
    async def scenario():
        config = fleet.Start()
        assert config.mode == "guided" and config.speed == 12
        catalog = fleet.catalog()
        assert catalog["default_mode"] == "guided" and catalog["default_speed"] == 12
        assert 12 in catalog["speeds"]
        created = await fleet.start(config)
        session = fleet.SESSIONS[created["id"]]
        try:
            await until(lambda: session.status == "paused")
            before = session.snapshot()
            await asyncio.sleep(.03)
            assert session.snapshot() == before
            assert before["mode"] == "guided" and before["guide"]["phase"] == "overview"
            assert session.index == 0 and not guided_rig["calls"]
            assert not session.jobs and not session.pending
            assert all(well["last_assessed"] is None for well in before["wells"])
        finally:
            await close(session)
    asyncio.run(scenario())


def test_next_moment_processes_each_minute_then_freezes_completed_evidence(guided_rig):
    async def scenario():
        session = await start_guided()
        try:
            checkpoint = await next_checkpoint(session)
            assert checkpoint and checkpoint["well_id"] == "WELL-00019"
            assert checkpoint["kind"] in {"concern", "escalation"}
            assert checkpoint["index"] == session.index
            assert checkpoint["source_timestamp"] == session.wells[checkpoint["well_id"]]["source_timestamp"]
            assert session.index >= 4
            assert not session.pending and not session.jobs
            assert session.wells[checkpoint["well_id"]]["investigation"] == "complete"
            assert_causal_progress(session, guided_rig["calls"])
            frozen = session.snapshot()
            await asyncio.sleep(.03)
            assert session.snapshot() == frozen
            # A reconnect resumes at the same checkpoint, without moving the cursor.
            assert (await fleet.snapshot(session.id))["guide"] == frozen["guide"]

            class ConnectedRequest:
                async def is_disconnected(self):
                    return False

            stream = await fleet.events(session.id, ConnectedRequest(), after=len(session.events) - 1, last_event_id=None)
            message = await anext(stream.body_iterator)
            payload = json.loads(message.split("data: ", 1)[1].strip())["payload"]
            assert payload["guide"]["checkpoint"]["id"] == checkpoint["id"]
            await stream.body_iterator.aclose()
            prior_index, prior_id = session.index, checkpoint["id"]
            session.control("resume")
            await until(lambda: session.status in fleet.TERMINAL or (
                session.status == "paused" and session.snapshot()["guide"]["phase"] == "checkpoint"))
            assert session.index > prior_index
            assert session.snapshot()["guide"]["checkpoint"]["id"] != prior_id
            assert_causal_progress(session, guided_rig["calls"])
        finally:
            await close(session)
    asyncio.run(scenario())


def test_checkpoint_waits_for_all_queued_and_running_reviews(guided_rig):
    async def scenario():
        session = await start_guided()
        gate, started = asyncio.Event(), asyncio.Event()
        original = session.investigate_well

        async def slow(well):
            started.set()
            try:
                await gate.wait()
                await original(well)
            finally:
                session.jobs.pop(well, None)
                session.wake.set()

        session.investigate_well = slow
        try:
            session.control("next_moment")
            await asyncio.wait_for(started.wait(), 2.)
            await until(lambda: session.snapshot()["guide"]["phase"] == "assessing")
            paused_at = session.index
            assert session.jobs
            assert session.snapshot()["guide"]["phase"] != "checkpoint"
            await asyncio.sleep(.03)
            assert session.index == paused_at
            session.control("pause")
            assert session.snapshot()["guide"]["phase"] == "manual_pause"
            session.control("resume")
            gate.set()
            await until(lambda: session.status == "paused" and session.snapshot()["guide"]["phase"] == "checkpoint")
            assert session.index == paused_at
            assert not session.jobs and not session.pending
            assert all(item["investigation"] not in {"queued", "running"} for item in session.wells.values())
        finally:
            gate.set()
            await close(session)
    asyncio.run(scenario())


def test_operator_findings_stay_paused_then_get_a_distinct_same_reading_followup(guided_rig):
    async def scenario():
        session = await start_guided()
        try:
            checkpoint = await next_checkpoint(session)
            well = checkpoint["well_id"]
            item = session.wells[well]
            incident_id = item["incident"]["id"]
            index = session.index
            session.operator_action(well, "acknowledge", incident_id=incident_id)
            check = item["assessment"]["checks"][0]
            session.operator_action(well, "check", check_id=check["id"], result="unavailable",
                                    note="Independent measurement not available", incident_id=incident_id,
                                    check_definition=check["definition"])
            await asyncio.sleep(.03)
            assert session.status == "paused" and session.index == index
            assert well in session.pending and not session.jobs
            followup = await next_checkpoint(session)
            assert followup["kind"] == "operator_followup"
            assert followup["id"] != checkpoint["id"] and session.index == index
            assert "no new measurements" in item["followup"]["change_summary"]
            assert item["assessment"]["checks"][0]["status"] == "unavailable"
            assert not session.pending and not session.jobs
            await next_checkpoint(session)
            assert session.index > index
            assert_causal_progress(session, guided_rig["calls"])
        finally:
            await close(session)
    asyncio.run(scenario())


def test_single_step_settles_assessments_and_keeps_readings_paused(guided_rig):
    async def scenario():
        session = await start_guided()
        try:
            session.control("step")
            await until(lambda: session.index == 1 and session.status == "paused"
                        and not session.jobs and not session.pending)
            assert session.snapshot()["guide"]["checkpoint"]["kind"] == "step"
            assert all(w["last_assessed"] == w["source_timestamp"] for w in session.wells.values())
            frozen = session.snapshot()
            await asyncio.sleep(.03)
            assert session.snapshot() == frozen
            assert_causal_progress(session, guided_rig["calls"])
        finally:
            await close(session)
    asyncio.run(scenario())


def test_manual_pause_cancel_and_continuous_twelve_are_explicit(guided_rig):
    async def scenario():
        session = await start_guided()
        try:
            session.control("mode", mode="continuous")
            assert session.status == "paused" and session.snapshot()["mode"] == "continuous"
            session.control("speed", speed=12)
            session.control("resume")
            await until(lambda: session.index >= 1)
            index = session.index
            await asyncio.sleep(.03)
            assert session.index == index, "Continuous 12x must not fast-forward like guided seeking"
            session.control("pause")
            assert session.snapshot()["guide"]["phase"] == "manual_pause"
            await asyncio.sleep(.03)
            assert session.index == index
            session.control("mode", mode="guided")
            assert session.status == "paused"
            session.control("next_moment")
            await until(lambda: session.index > index)
            session.control("cancel")
            cancelled_index = session.index
            await close(session)
            assert session.status == "cancelled"
            assert session.snapshot()["guide"]["phase"] == "ended"
            assert session.index == cancelled_index and not session.pending
            with pytest.raises(ValueError, match="ended"):
                session.control("next_moment")
        finally:
            await close(session)
    asyncio.run(scenario())


def test_guided_reaches_recovery_and_end_without_skipping_source_minutes(guided_rig):
    async def scenario():
        session = await start_guided()
        kinds = []
        try:
            for _ in range(12):
                checkpoint = await next_checkpoint(session)
                if checkpoint:
                    kinds.append(checkpoint["kind"])
                if session.status in fleet.TERMINAL:
                    break
                assert not session.jobs and not session.pending
            assert "recovery" in kinds
            assert session.status == "completed" and session.index == fleet.TOTAL
            assert session.snapshot()["guide"]["phase"] == "ended"
            assert not session.jobs and not session.pending
            assert_causal_progress(session, guided_rig["calls"])
            incident = session.wells["WELL-00019"]["incident"]
            assert incident["condition_cleared"]
            assert any(event["kind"] == "recovered" for event in session.wells["WELL-00019"]["timeline"])
        finally:
            await close(session)
    asyncio.run(scenario())


def test_guided_seek_yields_to_manual_pause_before_reaching_end(guided_rig):
    guided_rig["event_enabled"] = False

    async def scenario():
        session = await start_guided()
        try:
            session.control("next_moment")
            await until(lambda: session.index >= 1)
            assert session.index < fleet.TOTAL
            session.control("pause")
            index = session.index
            await asyncio.sleep(.03)
            assert session.index == index and session.status == "paused"
            assert session.snapshot()["guide"]["phase"] == "manual_pause"
            assert_causal_progress(session, guided_rig["calls"])
        finally:
            await close(session)
    asyncio.run(scenario())


def test_cancel_interrupts_assessments_without_reading_more_data(guided_rig):
    async def scenario():
        session = await start_guided()
        gate, started = asyncio.Event(), asyncio.Event()

        async def blocked_review(well):
            started.set()
            try:
                await gate.wait()
            finally:
                session.jobs.pop(well, None)
                session.wake.set()

        session.investigate_well = blocked_review
        try:
            session.control("next_moment")
            await asyncio.wait_for(started.wait(), 2.)
            await until(lambda: session.snapshot()["guide"]["phase"] == "assessing")
            index = session.index
            session.control("cancel")
            await close(session)
            assert session.index == index
            assert session.status == "cancelled" and not session.jobs and not session.pending
            assert session.snapshot()["guide"]["phase"] == "ended"
            assert session.requests_used == 0
        finally:
            gate.set()
            await close(session)
    asyncio.run(scenario())


@pytest.mark.parametrize("interrupted_review", ["new_concern", "operator_followup"])
def test_continuous_advancement_cannot_resurrect_an_old_guided_checkpoint(guided_rig, interrupted_review):
    async def scenario():
        session = await start_guided()
        gate, started = asyncio.Event(), asyncio.Event()
        original = session.investigate_well

        async def delayed(well):
            started.set()
            try:
                await gate.wait()
                await original(well)
            finally:
                session.jobs.pop(well, None)
                session.wake.set()

        try:
            if interrupted_review == "operator_followup":
                checkpoint = await next_checkpoint(session)
                well = checkpoint["well_id"]
                item = session.wells[well]
                check = item["assessment"]["checks"][0]
                session.operator_action(well, "check", check_id=check["id"], result="unavailable",
                                        incident_id=item["incident"]["id"], check_definition=check["definition"])
            session.investigate_well = delayed
            session.control("next_moment")
            await asyncio.wait_for(started.wait(), 2.)
            await until(lambda: session.snapshot()["guide"]["phase"] == "assessing")
            interrupted_index = session.index
            session.control("pause")
            session.control("mode", mode="continuous")
            session.control("resume")
            await until(lambda: session.index > interrupted_index)
            gate.set()
            await until(lambda: not session.jobs and not session.pending)
            session.control("mode", mode="guided")
            resumed_index = session.index
            checkpoint = await next_checkpoint(session)
            assert session.index > resumed_index, "Already-consumed guided work must not create a stale same-reading stop"
            assert checkpoint["kind"] != "operator_followup"
            assert checkpoint["index"] == session.index
            if checkpoint["well_id"]:
                assert checkpoint["source_timestamp"] == session.wells[checkpoint["well_id"]]["source_timestamp"]
            assert_causal_progress(session, guided_rig["calls"])
        finally:
            gate.set()
            await close(session)
    asyncio.run(scenario())
