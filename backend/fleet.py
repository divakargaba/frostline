"""Fleet replay engine — shared-clock multi-well replay with priority queue.

Owner: Div

Manages fleet sessions: 4 wells advance on one clock, watcher per well,
agent investigations queued by priority, SSE stream for all events.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.config import DATA_DEMO, FLEET_LLM_MAX, FLEET_MANIFEST
from backend.replay import ReplayInstance, load_instance
from backend.schemas import (
    AckResponse,
    FleetAssessmentEvent,
    FleetCounters,
    FleetEndEvent,
    FleetStatusEvent,
    FleetTickEvent,
    PriorityEntry,
)
from src.watcher import Watcher

log = logging.getLogger("frostline.fleet")

_executor = ThreadPoolExecutor(max_workers=2)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Incident:
    incident_id: str
    well_id: str
    display_id: str
    created_minute: int
    last_updated_minute: int
    score: float
    category: str  # review_now, investigate, watch, normal
    acknowledged: bool = False
    assessments: list[dict] = field(default_factory=list)
    trigger_reason: str = ""


@dataclass
class FleetWell:
    display_id: str
    instance_id: str
    well_id: str
    inst: ReplayInstance
    watcher: Watcher
    from_minute: int
    to_minute: int
    decisions: list[dict] = field(default_factory=list)
    incidents: dict[str, Incident] = field(default_factory=dict)
    finished: bool = False
    lat: float = 0.0
    lon: float = 0.0
    prev_phase: str | None = None
    event_class: int = 0
    event_label: str = ""


@dataclass
class FleetSession:
    session_id: str
    wells: list[FleetWell]
    speed: float
    mode: str  # "live" | "cached" | "cache_only"
    agent_enabled: bool

    current_minute: int = 0
    max_minutes: int = 180
    paused: bool = False
    finished: bool = False

    incidents: dict[str, Incident] = field(default_factory=dict)
    investigation_queue: list[str] = field(default_factory=list)
    investigating: str | None = None
    held: bool = False

    llm_attempts_used: int = 0
    llm_attempts_max: int = FLEET_LLM_MAX

    priority_list: list[PriorityEntry] = field(default_factory=list)

    recording_path: Path | None = None
    _rec_file: Any = field(default=None, repr=False)

    subscribers: list[asyncio.Queue] = field(default_factory=list)

    def emit(self, event_type: str, data: dict):
        """Send an event to all subscribers and optionally record it."""
        evt = {"type": event_type, "data": data}
        if self._rec_file:
            self._rec_file.write(json.dumps(evt) + "\n")
            self._rec_file.flush()
        for q in self.subscribers:
            try:
                q.put_nowait(evt)
            except asyncio.QueueFull:
                pass  # drop if subscriber is slow


# ---------------------------------------------------------------------------
# Priority logic (deterministic, no LLM)
# ---------------------------------------------------------------------------

def _categorize_incident(incident: Incident) -> str:
    if incident.score >= 0.7:
        return "review_now"
    # Check latest assessment decision
    if incident.assessments:
        last = incident.assessments[-1]
        if last.get("decision") == "ALERT":
            return "review_now"
        if last.get("decision") == "WATCH":
            return "investigate"
    if incident.score >= 0.45:
        return "investigate"
    if incident.score >= 0.2:
        return "watch"
    return "normal"


def _build_priority_list(session: FleetSession) -> list[PriorityEntry]:
    """Build sorted priority list from all incidents."""
    entries: list[dict] = []

    for well in session.wells:
        # Find the highest-priority incident for this well
        best_incident: Incident | None = None
        for inc in well.incidents.values():
            if best_incident is None or inc.score > best_incident.score:
                best_incident = inc

        if best_incident is None:
            # No incidents — normal
            entries.append({
                "well_id": well.well_id,
                "display_id": well.display_id,
                "category": "normal",
                "score": 0.0,
                "headline": "Normal operation" if not well.finished else "Replay complete",
                "unacknowledged": False,
                "next_check_minute": None,
                "sort_key": (3, 0, 0, well.well_id),  # category order: normal=3
            })
            continue

        cat = _categorize_incident(best_incident)
        best_incident.category = cat

        # Headline from latest assessment
        headline = best_incident.trigger_reason
        if best_incident.assessments:
            last = best_incident.assessments[-1]
            headline = last.get("brief", headline)[:120]

        # Next check
        next_check = None
        if best_incident.assessments:
            last = best_incident.assessments[-1]
            next_check = last.get("next_check_minute")

        cat_order = {"review_now": 0, "investigate": 1, "watch": 2, "normal": 3}
        time_unack = session.current_minute - best_incident.created_minute if not best_incident.acknowledged else 0

        entries.append({
            "well_id": well.well_id,
            "display_id": well.display_id,
            "category": cat,
            "score": best_incident.score,
            "headline": headline,
            "unacknowledged": not best_incident.acknowledged,
            "next_check_minute": next_check,
            "sort_key": (cat_order.get(cat, 3), -best_incident.score, -time_unack, well.well_id),
        })

    entries.sort(key=lambda e: e["sort_key"])

    result = []
    for rank, e in enumerate(entries):
        e.pop("sort_key")
        result.append(PriorityEntry(priority_rank=rank + 1, **e))
    return result


def _build_counters(session: FleetSession) -> FleetCounters:
    all_incidents = list(session.incidents.values())
    needs_review = sum(1 for i in all_incidents if _categorize_incident(i) == "review_now")
    unacknowledged = sum(1 for i in all_incidents if not i.acknowledged)
    reliable = sum(1 for w in session.wells if not w.finished)
    checks_due = sum(1 for w in session.wells
                     for i in w.incidents.values()
                     if i.assessments and i.assessments[-1].get("next_check_minute") is not None
                     and i.assessments[-1]["next_check_minute"] <= session.current_minute)
    return FleetCounters(
        needs_review=needs_review,
        unacknowledged=unacknowledged,
        reliable_feeds=reliable,
        checks_due=checks_due,
    )


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

_sessions: dict[str, FleetSession] = {}


def _load_fleet_manifest() -> list[dict]:
    if FLEET_MANIFEST.exists():
        return json.loads(FLEET_MANIFEST.read_text()).get("wells", [])
    return []


def create_session(speed: int = 60, agent: bool = True,
                   mode: str = "live", record: bool = False) -> FleetSession:
    """Create a new fleet session with all wells from the manifest."""
    manifest = _load_fleet_manifest()
    if not manifest:
        raise ValueError("No fleet manifest found at data/demo/fleet.json")

    session_id = str(uuid.uuid4())[:8]
    wells: list[FleetWell] = []
    max_window = 0

    for w in manifest:
        inst = load_instance(w["instance_id"])
        if inst is None:
            log.warning("Could not load %s, skipping", w["instance_id"])
            continue
        window_len = w["to_minute"] - w["from_minute"]
        max_window = max(max_window, window_len)
        wells.append(FleetWell(
            display_id=w["display_id"],
            instance_id=w["instance_id"],
            well_id=w["well_id"],
            inst=inst,
            watcher=Watcher(),
            from_minute=w["from_minute"],
            to_minute=min(w["to_minute"], inst.n_minutes),
            lat=w.get("lat", 0),
            lon=w.get("lon", 0),
            event_class=w.get("event_class", 0),
            event_label=w.get("event_label", ""),
        ))

    recording_path = None
    rec_file = None
    if record:
        DATA_DEMO.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        recording_path = DATA_DEMO / f"fleet_{ts}.jsonl"
        rec_file = open(recording_path, "w")

    session = FleetSession(
        session_id=session_id,
        wells=wells,
        speed=speed,
        mode=mode,
        agent_enabled=agent,
        max_minutes=max_window,
        recording_path=recording_path,
        _rec_file=rec_file,
    )

    if mode == "cache_only":
        session.agent_enabled = True  # agent runs but cache_only

    _sessions[session_id] = session
    return session


def get_session(session_id: str) -> FleetSession | None:
    return _sessions.get(session_id)


def acknowledge_incident(session: FleetSession, incident_id: str) -> AckResponse | None:
    incident = session.incidents.get(incident_id)
    if incident is None:
        return None
    incident.acknowledged = True
    # Re-emit fleet status
    session.priority_list = _build_priority_list(session)
    session.emit("fleet_status", _make_status_event(session).model_dump())
    return AckResponse(incident_id=incident_id, acknowledged=True)


# ---------------------------------------------------------------------------
# SSE event helpers
# ---------------------------------------------------------------------------

def _make_status_event(session: FleetSession) -> FleetStatusEvent:
    session.priority_list = _build_priority_list(session)
    return FleetStatusEvent(
        session_id=session.session_id,
        held=session.held,
        reason="investigating" if session.held else "",
        current_minute=session.current_minute,
        priority_list=session.priority_list,
        counters=_build_counters(session),
    )


# ---------------------------------------------------------------------------
# Main replay loop
# ---------------------------------------------------------------------------

async def run_fleet_stream(session: FleetSession, queue: asyncio.Queue):
    """Main fleet replay coroutine. Advances shared clock, emits events."""
    delay = 1.0 / session.speed

    # Emit initial status
    status = _make_status_event(session)
    session.emit("fleet_status", status.model_dump())

    while session.current_minute < session.max_minutes and not session.finished:
        if session.paused:
            await asyncio.sleep(0.1)
            continue

        # Process each well at this minute
        pending_investigations: list[tuple[FleetWell, dict]] = []

        for well in session.wells:
            if well.finished:
                continue

            abs_minute = well.from_minute + session.current_minute
            if abs_minute >= well.to_minute or abs_minute >= well.inst.n_minutes:
                well.finished = True
                continue

            # Get tick data
            tick_data = well.inst.tick(abs_minute)
            phase = tick_data.get("phase", "normal")

            # Phase marker on change
            if phase != well.prev_phase:
                session.emit("fleet_phase_marker", {
                    "session_id": session.session_id,
                    "well_id": well.well_id,
                    "display_id": well.display_id,
                    "t": tick_data["t"],
                    "phase": phase,
                })
                well.prev_phase = phase

            # Tick event
            tick_evt = {
                "session_id": session.session_id,
                "well_id": well.well_id,
                "display_id": well.display_id,
                "replay_minute": session.current_minute,
                "original_t": tick_data["t"],
                "sensors": tick_data["sensors"],
                "margin_C": tick_data.get("margin_C"),
                "p_hydrate": tick_data.get("p_hydrate"),
                "p_lookalike": tick_data.get("p_lookalike"),
                "p_normal": tick_data.get("p_normal"),
            }
            quality = tick_data.get("quality")
            if quality:
                tick_evt["quality"] = quality
            session.emit("fleet_tick", tick_evt)

            # Watcher check
            if session.agent_enabled:
                trigger = well.watcher.check_tick(tick_data, well.well_id, abs_minute)
                if trigger:
                    # Create or update incident
                    incident_id = f"{well.well_id}_{session.current_minute}"
                    # Check if existing incident for this well should be updated
                    existing = None
                    for inc in well.incidents.values():
                        if not inc.acknowledged or trigger["score"] - inc.score >= 0.15:
                            existing = inc
                            break

                    if existing and trigger["score"] - existing.score >= 0.15:
                        # Deterioration — update existing incident
                        existing.score = trigger["score"]
                        existing.last_updated_minute = session.current_minute
                        existing.trigger_reason = trigger["reason"]
                        incident_id = existing.incident_id
                    elif existing is None or trigger["score"] > 0:
                        # New incident
                        incident = Incident(
                            incident_id=incident_id,
                            well_id=well.well_id,
                            display_id=well.display_id,
                            created_minute=session.current_minute,
                            last_updated_minute=session.current_minute,
                            score=trigger["score"],
                            category="watch",
                            trigger_reason=trigger["reason"],
                        )
                        well.incidents[incident_id] = incident
                        session.incidents[incident_id] = incident

                    session.emit("fleet_watch_trigger", {
                        "session_id": session.session_id,
                        "well_id": well.well_id,
                        "display_id": well.display_id,
                        "t": trigger["t"],
                        "reason": trigger["reason"],
                        "score": trigger["score"],
                        "incident_id": incident_id,
                    })

                    pending_investigations.append((well, trigger))

        # Process investigations (sequentially, one at a time)
        for well, trigger in pending_investigations:
            if session.llm_attempts_used >= session.llm_attempts_max:
                # Budget exhausted — emit fallback
                _emit_fallback_assessment(session, well, trigger)
                continue

            session.held = True
            session.investigating = well.well_id
            session.emit("fleet_status", _make_status_event(session).model_dump())

            try:
                await _run_investigation(session, well, trigger)
            except Exception as e:
                log.error("Investigation error for %s: %s", well.well_id, e)
                _emit_fallback_assessment(session, well, trigger)
            finally:
                session.held = False
                session.investigating = None

        # Update priority list
        session.priority_list = _build_priority_list(session)
        session.emit("fleet_status", _make_status_event(session).model_dump())

        session.current_minute += 1
        await asyncio.sleep(delay)

    # End event
    end_evt = FleetEndEvent(
        session_id=session.session_id,
        total_minutes=session.current_minute,
        llm_attempts_used=session.llm_attempts_used,
    )
    session.emit("fleet_end", end_evt.model_dump())
    session.finished = True

    # Close recording file
    if session._rec_file:
        session._rec_file.close()
        session._rec_file = None


async def _run_investigation(session: FleetSession, well: FleetWell,
                             trigger: dict):
    """Run agent investigation for a well trigger."""
    from src.agent import run_agent
    from src.tools import AgentContext

    abs_minute = well.from_minute + session.current_minute
    cache_only = session.mode == "cache_only"

    ctx = AgentContext(
        instance_id=well.instance_id,
        well_id=well.well_id,
        minute_index=abs_minute,
        df=well.inst.df,
        decisions=well.decisions,
        thresholds=well.watcher.thresholds,
    )

    loop = asyncio.get_event_loop()
    agent_events = await loop.run_in_executor(
        _executor,
        lambda: list(run_agent(ctx, trigger, cache_only=cache_only))
    )

    if not cache_only:
        session.llm_attempts_used += 1

    for ae in agent_events:
        evt_type = ae["type"]
        data = ae["data"]

        if evt_type == "tool_call":
            session.emit("fleet_tool_call", {
                "session_id": session.session_id,
                "well_id": well.well_id,
                "display_id": well.display_id,
                "t": data.get("t", ""),
                "call_id": data.get("call_id", ""),
                "tool": data.get("tool", ""),
                "args": data.get("args", {}),
                "requested_by": data.get("requested_by", "system"),
            })
        elif evt_type == "tool_result":
            session.emit("fleet_tool_result", {
                "session_id": session.session_id,
                "well_id": well.well_id,
                "display_id": well.display_id,
                "t": data.get("t", ""),
                "call_id": data.get("call_id", ""),
                "tool": data.get("tool", ""),
                "result": data.get("result", {}),
                "requested_by": data.get("requested_by", "system"),
            })
        elif evt_type == "decision":
            decision = data
            well.decisions.append(decision)

            # Update incident
            incident_id = f"{well.well_id}_{session.current_minute}"
            incident = session.incidents.get(incident_id)
            if incident is None:
                # Find latest incident for this well
                for inc in reversed(list(well.incidents.values())):
                    incident = inc
                    break

            cat = "normal"
            if incident:
                incident.assessments.append(decision)
                incident.last_updated_minute = session.current_minute
                if decision.get("decision") == "ALERT":
                    incident.score = max(incident.score, decision.get("confidence", 0.8))
                cat = _categorize_incident(incident)
                incident.category = cat

            # Record decision in watcher
            well.watcher.record_decision(
                well.well_id,
                decision.get("decision", "DISMISS"),
                abs_minute,
                score=trigger["score"],
                recheck_min=decision.get("recheck_min"),
            )

            next_check = None
            if decision.get("recheck_min"):
                next_check = session.current_minute + decision["recheck_min"]

            assessment = FleetAssessmentEvent(
                session_id=session.session_id,
                well_id=well.well_id,
                display_id=well.display_id,
                incident_id=incident.incident_id if incident else incident_id,
                decision=decision.get("decision", "DISMISS"),
                category=cat,
                confidence=decision.get("confidence", 0.5),
                diagnosis=decision.get("diagnosis", "unknown"),
                brief=decision.get("brief", ""),
                next_check_minute=next_check,
                evidence_refs=decision.get("evidence", []),
                playbook_refs=decision.get("playbook_refs", []),
                source=decision.get("source", "agent"),
            )
            session.emit("fleet_assessment", assessment.model_dump())


def _emit_fallback_assessment(session: FleetSession, well: FleetWell,
                              trigger: dict):
    """Emit a fallback assessment when LLM budget is exhausted or fails."""
    incident_id = f"{well.well_id}_{session.current_minute}"
    incident = session.incidents.get(incident_id)
    if incident is None:
        for inc in reversed(list(well.incidents.values())):
            incident = inc
            break

    assessment = FleetAssessmentEvent(
        session_id=session.session_id,
        well_id=well.well_id,
        display_id=well.display_id,
        incident_id=incident.incident_id if incident else incident_id,
        decision="WATCH",
        category="investigate",
        confidence=0.3,
        diagnosis="unknown",
        brief=f"LLM budget exhausted ({session.llm_attempts_used}/{session.llm_attempts_max}). "
              f"Monitoring continues. Trigger: {trigger.get('reason', 'unknown')}",
        source="fallback",
    )
    session.emit("fleet_assessment", assessment.model_dump())

    if incident:
        incident.assessments.append(assessment.model_dump())


# ---------------------------------------------------------------------------
# Cached fleet replay
# ---------------------------------------------------------------------------

def load_fleet_recording() -> list[dict] | None:
    """Load the latest fleet recording from data/demo/."""
    recordings = sorted(DATA_DEMO.glob("fleet_*.jsonl"), reverse=True)
    if not recordings:
        return None
    events = []
    for line in recordings[0].read_text().splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events
