"""Four independent historical feeds with a bounded, event-driven investigation queue."""
from __future__ import annotations

import asyncio
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import time
import uuid

import pandas as pd
from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
SENSORS = ["P-PDG", "T-PDG", "P-TPT", "T-TPT", "P-MON-CKP", "P-JUS-CKP", "T-JUS-CKP", "ABER-CKP", "QGL"]
PRESSURES = ["P-PDG", "P-TPT", "P-MON-CKP"]
WELLS = ["WELL-00001", "WELL-00002", "WELL-00006", "WELL-00019"]
# These are fixed presentation excerpts, not classifier inputs or benchmark selection.
OFFSETS = {"WELL-00001": "2017-02-01T02:03:00", "WELL-00002": "2014-02-12T20:15:00", "WELL-00006": "2018-06-18T11:31:00", "WELL-00019": "2012-06-02T21:55:00"}
RANK = {"attention": 0, "unavailable": 1, "watch": 2, "normal": 3}
TERMINAL = {"completed", "cancelled", "failed"}
TOTAL = 180
SESSIONS: dict[str, "FleetSession"] = {}
ATTEMPTS: deque[float] = deque()
PROVIDER_CIRCUIT = {"until": 0.0, "reason": ""}
router = APIRouter(prefix="/api/fleet", tags=["Four-well operations"])


def load_local_env():
    """Read only this project's ignored provider configuration; never return its values."""
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[7:]
            key, sep, value = line.partition("=")
            if sep and key.strip() in {"OPENROUTER_API_KEY", "OPENROUTER_MODEL"}:
                value = value.strip().strip("\"'")
                if value:
                    os.environ.setdefault(key.strip(), value)


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        return safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def capabilities():
    load_local_env()
    configured = bool(os.getenv("OPENROUTER_API_KEY", "").strip())
    try:
        from src.fleet_model import load_bundle
        bundle = load_bundle()
        ready = bool(bundle)
        reason = "Model ready. OpenRouter is configured; live tool verification is pending." if configured else "Model ready. Add an OpenRouter key to enable live LLM investigations."
    except Exception as exc:
        ready = False
        reason = "Train the fleet model before starting: python scripts/train_fleet_model.py"
        if not isinstance(exc, (ImportError, FileNotFoundError)):
            reason = "The saved fleet model could not be loaded. Rebuild it with scripts/train_fleet_model.py."
    return {"model_ready": ready, "llm_configured": configured, "readiness_message": reason}


@lru_cache(maxsize=1)
def source_frames():
    from src.fleet_model import DEMO_RECORDINGS, prepare_recording
    result = {}
    for entry in DEMO_RECORDINGS:
        well = entry["well_id"]
        path = Path(entry["path"])
        if not path.is_absolute():
            path = ROOT / path
        frame = prepare_recording(path)
        start = frame.index.searchsorted(pd.Timestamp(OFFSETS[well]))
        if start < 60 or len(frame) - start < TOTAL:
            raise ValueError(f"Insufficient replay history for {well}")
        # Annotation columns never enter the operating session, tools, or event stream.
        sensors = frame.drop(columns=[c for c in ["class", "state", "label", "phase", "ground_truth"] if c in frame])
        result[well] = {"source_file": path.name, "warmup": sensors.iloc[start - 60:start].copy(), "future": sensors.iloc[start:start + TOTAL].copy()}
    if set(result) != set(WELLS):
        raise ValueError("The fleet requires four distinct configured wells")
    return result


def empty_well(well):
    return {"id": well, "name": f"Well {well.removeprefix('WELL-').lstrip('0') or '0'}", "status": "unavailable", "source_file": "", "source_timestamp": None,
            "quality": {"status": "unavailable", "summary": "Loading historical measurements", "missing": [], "invalid": [], "unchanged": []},
            "last_assessed": None, "next_check": None, "incident": None, "assessment": None, "frames": [], "investigation": "idle", "activity": "Preparing recording"}


class FleetSession:
    def __init__(self, speed=60):
        self.id = uuid.uuid4().hex
        self.speed = speed
        self.status = "preparing"
        self.index = 0
        self.wells = {w: empty_well(w) for w in WELLS}
        self.events = []
        self.audit = []
        self.changed = asyncio.Event()
        self.wake = asyncio.Event()
        self.pending: set[str] = set()
        self.jobs: dict[str, asyncio.Task] = {}
        self.history = {}
        self.sources = {}
        self.state = {w: {"alarm": {}, "other_run": 0, "other_active": False, "other_recovery": 0, "normal_run": 0, "agent_status": "normal", "base_status": "unavailable", "revision": 0, "due": 0, "last_score": None, "revision_score": None, "revision_pressure": None, "last_quality": None, "last_investigation_index": -100, "priorities": []} for w in WELLS}
        self.bundle = None
        self.requests_used = 0
        self.live_verified = False
        self.request_budget = 18
        self.fault = None
        self.error = None
        self.caps = capabilities()
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.task = None
        self.step_requested = False

    def priority(self):
        return sorted(WELLS, key=lambda w: (RANK[self.wells[w]["status"]], bool((self.wells[w].get("incident") or {}).get("acknowledged")), self.state[w]["due"], WELLS.index(w)))

    def snapshot(self):
        return safe({"id": self.id, "status": self.status, "speed": self.speed, "elapsed_seconds": self.index * 60, "index": self.index, "total": TOTAL,
                     "agent_mode": "live" if self.live_verified else "rules", **self.caps, "wells": [deepcopy(self.wells[w]) for w in WELLS],
                     "priority": self.priority(), "last_event_id": len(self.events), "requests_used": self.requests_used, "request_budget": self.request_budget,
                     "fault": self.fault, "error": self.error})

    def publish(self, reason="update"):
        event_id = len(self.events) + 1
        payload = self.snapshot()
        payload["last_event_id"] = event_id
        self.events.append({"id": event_id, "type": "snapshot", "payload": payload})
        self.audit.append({"kind": reason, "event_id": event_id, "elapsed_seconds": self.index * 60})
        self.changed.set()

    async def prepare(self):
        from src.fleet_model import load_bundle
        self.sources = await asyncio.to_thread(source_frames)
        self.bundle = await asyncio.to_thread(load_bundle)
        excluded = set(self.bundle.get("excluded_wells", self.bundle.get("excluded_well_ids", WELLS)))
        training = set(self.bundle.get("training_wells", self.bundle.get("train_wells", self.bundle.get("training_well_ids", []))))
        if training.intersection(WELLS) or not set(WELLS).issubset(excluded):
            raise ValueError("The demo model must hold out every recording from all four demo wells")
        for well, source in self.sources.items():
            self.history[well] = source["warmup"].copy()
            self.wells[well]["source_file"] = source["source_file"]
            # Warm-up provides preceding measurements, not assumed healthy labels.
            for t, row in source["warmup"].iloc[-30:].iterrows():
                self.wells[well]["frames"].append({"t": t.isoformat(), "elapsed_seconds": int((t - source["future"].index[0]).total_seconds()), "sensors": safe(row.reindex(SENSORS).to_dict()), "risk_score": None})
            self.wells[well]["activity"] = "Ready to monitor"
        self.status = "running"
        self.publish("prepared")

    def default_assessment(self, well, result, status):
        scores = result.get("scores", {})
        quality = result.get("quality", {})
        evidence = []
        if scores.get("hydrate") is not None:
            evidence.append(f"Hydrate model score {scores['hydrate']:.2f}; alarm threshold {self.bundle['policy']['activation_threshold']:.2f}.")
        if scores.get("lookalike") is not None:
            evidence.append(f"Other restriction/scaling model score {scores['lookalike']:.2f}.")
        if quality.get("summary"):
            evidence.append(quality["summary"])
        text = {
            "normal": ("No persistent process concern in the available signals.", "Continue monitoring."),
            "watch": ("A changing sensor pattern needs more evidence.", "Review the recent trends and await the scheduled check."),
            "attention": ("A persistent abnormal pattern needs operator review.", "Review the pressure trends and request an engineering check."),
            "unavailable": ("Pressure evidence is unavailable; the condition cannot be assessed.", "Verify pressure telemetry before relying on a process assessment."),
        }[status]
        if status == "watch" and quality.get("invalid"):
            text = ("Some sensor values are invalid; verify telemetry before diagnosing the well.", "Check the flagged channels and compare the remaining pressure trends.")
            evidence = [quality.get("summary", "Invalid sensor values excluded."), *evidence]
        elif status == "attention" and self.state[well]["alarm"].get("active"):
            text = ("A persistent hydrate-like pattern needs operator review.", "Review pressure trends and ask the responsible engineer to assess the well.")
        return {"summary": text[0], "evidence": evidence[:2], "next_step": text[1], "source": "rules", "uncertainty": "Numerical monitoring; LLM review pending." if self.caps["llm_configured"] and status != "normal" else "Model scores are not a confirmed diagnosis.", "tools": []}

    def queue(self, well, reason):
        self.pending.add(well)
        if well not in self.jobs:
            self.wells[well]["investigation"] = "queued"
            self.wells[well]["activity"] = reason

    def advance(self):
        from src.fleet_model import infer_window, advance_alarm
        if self.index >= TOTAL:
            return
        for well in WELLS:
            item = self.wells[well]
            state = self.state[well]
            row = self.sources[well]["future"].iloc[self.index:self.index + 1].copy()
            if self.fault and self.fault["well_id"] == well:
                row.loc[:, PRESSURES] = float("nan")
                self.fault["remaining"] -= 1
                if self.fault["remaining"] <= 0:
                    self.fault = None
            self.history[well] = pd.concat([self.history[well], row]).iloc[-180:]
            result = infer_window(self.history[well], self.bundle)
            scores = result.get("scores", {})
            quality = result.get("quality", {})
            blocked = bool(quality.get("blocked", quality.get("status") == "unavailable"))
            t = row.index[-1]
            alarm = advance_alarm(state["alarm"], scores.get("hydrate"), not blocked, t, self.bundle["policy"])
            state["alarm"] = alarm.get("state", alarm)
            active = bool(alarm.get("active", alarm.get("alarm", False)))
            p_other = scores.get("lookalike")
            state["other_run"] = state["other_run"] + 1 if not blocked and p_other is not None and p_other >= .65 else 0
            if state["other_run"] >= 3:
                state["other_active"] = True
            state["other_recovery"] = state["other_recovery"] + 1 if not blocked and p_other is not None and p_other < .55 else 0
            if state["other_recovery"] >= 5:
                state["other_active"] = False
            recent_pressure = self.history[well]["P-TPT"].iloc[-11:].dropna()
            changing_pressure = len(recent_pressure) >= 6 and abs(recent_pressure.iloc[-1] - recent_pressure.iloc[0]) >= max(2., abs(recent_pressure.iloc[0]) * .02)
            previous = item["status"]
            previous_base = state["base_status"]
            if blocked:
                status = "attention" if previous == "attention" else "unavailable"
            elif active or state["other_active"]:
                status = "attention"
            elif (scores.get("hydrate") or 0) >= .35 or (p_other or 0) >= .5 or changing_pressure or quality.get("invalid"):
                status = "watch"
            else:
                status = "normal"
            base_status = status
            state["base_status"] = base_status
            state["normal_run"] = state["normal_run"] + 1 if base_status == "normal" else 0
            if state["normal_run"] >= 5:
                state["agent_status"] = "normal"
            if RANK[state["agent_status"]] < RANK[status]:
                status = state["agent_status"]
            score = scores.get("hydrate")
            pressure = safe(row.iloc[-1].get("P-TPT"))
            quality_signature = (blocked, tuple(sorted(quality.get("missing", []))), tuple(sorted(quality.get("invalid", []))))
            score_shift = score is not None and state["revision_score"] is not None and abs(score - state["revision_score"]) >= .2
            pressure_shift = pressure is not None and state["revision_pressure"] is not None and abs(pressure - state["revision_pressure"]) >= max(5., abs(state["revision_pressure"]) * .03)
            material = status != previous or base_status != previous_base or quality_signature != state["last_quality"] or score_shift or pressure_shift
            if material:
                state["revision"] += 1
                state["revision_score"] = score
                state["revision_pressure"] = pressure
                state["last_quality"] = quality_signature
            item.update(status=status, quality={"status": "unavailable" if blocked else quality.get("status", "good"), "summary": quality.get("summary", ""), "missing": quality.get("missing", []), "invalid": quality.get("invalid", []), "unchanged": quality.get("unchanged", [])}, source_timestamp=t.isoformat())
            item["frames"].append({"t": t.isoformat(), "elapsed_seconds": self.index * 60, "sensors": safe(row.iloc[-1].reindex(SENSORS).to_dict()), "risk_score": scores.get("hydrate")})
            item["frames"] = item["frames"][-240:]
            if material or item["assessment"] is None:
                item["assessment"] = self.default_assessment(well, result, status)
            if status != "normal":
                if not item["incident"] or item["incident"].get("condition_cleared"):
                    item["incident"] = {"id": uuid.uuid4().hex, "acknowledged": False, "completed": False, "opened_at": t.isoformat(), "note": ""}
                elif RANK[status] < RANK[previous]:
                    item["incident"]["acknowledged"] = False
                    item["incident"]["completed"] = False
            elif item["incident"]:
                item["incident"]["condition_cleared"] = True
            due = self.index >= state["due"]
            if status != "normal" and (material or (due and well not in self.jobs)):
                self.queue(well, "Investigating changed evidence" if material else "Scheduled reassessment")
            elif due:
                state["due"] = self.index + (15 if status != "normal" else 30)
            state["result"] = result
            state["last_score"] = score
            item["next_check"] = (t + pd.Timedelta(minutes=max(1, state["due"] - self.index))).isoformat()
        self.index += 1
        self.publish("tick")

    def agent_snapshot(self, well):
        item = self.wells[well]
        state = self.state[well]
        readings = [{"t": t.isoformat(), "sensors": safe(row.reindex(SENSORS).to_dict())} for t, row in self.history[well].iloc[-61:].iterrows()]
        return {"well_id": well, "incident_revision": state["revision"], "as_of": item["source_timestamp"], "severity": "telemetry" if item["status"] == "unavailable" else item["status"],
                "readings": readings, "quality": safe(state["result"]["quality"]), "model": {"available": True, "scores": safe(state["result"]["scores"]), "version": self.bundle.get("model_id", "fleet-v1")},
                "history": deepcopy(state["priorities"][-4:]), "operator_observation": (item.get("incident") or {}).get("note", ""),
                "fleet": [{"well_id": w, "severity": self.wells[w]["status"], "acknowledged": (self.wells[w]["incident"] or {}).get("acknowledged", False)} for w in self.priority()],
                "policy_version": self.bundle.get("model_id", "fleet-v1")}

    async def reserve_attempt(self):
        now = time.monotonic()
        while ATTEMPTS and now - ATTEMPTS[0] >= 60:
            ATTEMPTS.popleft()
        if self.status in TERMINAL or self.requests_used >= self.request_budget or len(ATTEMPTS) >= 10 or now < PROVIDER_CIRCUIT["until"]:
            return False
        self.requests_used += 1
        ATTEMPTS.append(now)
        return True

    async def investigate_well(self, well):
        from src.fleet_llm import investigate
        self.pending.discard(well)
        item = self.wells[well]
        state = self.state[well]
        snapshot = self.agent_snapshot(well)
        revision = state["revision"]
        item["investigation"] = "running"
        item["activity"] = "Checking the current evidence"
        trace = []
        self.publish("investigation_started")

        async def emit(event):
            if self.status in TERMINAL:
                return
            payload = safe(event.get("payload", {}))
            kind = event.get("type", "agent_activity")
            self.audit.append({"kind": kind, "well_id": well, "as_of": snapshot["as_of"], "payload": payload})
            name = payload.get("tool", payload.get("name", "Evidence check"))
            if kind == "tool_call":
                trace.append({"name": name, "status": "running", "summary": payload.get("purpose", payload.get("summary", ""))})
                item["activity"] = f"Checking {str(name).replace('_', ' ')}"
            elif kind == "tool_result":
                existing = next((v for v in reversed(trace) if v["name"] == name and v["status"] == "running"), None)
                summary = payload.get("summary") or payload.get("result", {}).get("summary", "Evidence received") if isinstance(payload.get("result", {}), dict) else "Evidence received"
                if existing:
                    existing.update(status="done", summary=str(summary)[:300])
            elif kind == "agent_activity":
                item["activity"] = payload.get("message", payload.get("summary", "Assessing evidence"))
            if item["assessment"]:
                item["assessment"]["tools"] = deepcopy(trace[-12:])
            self.publish(kind)

        try:
            result = await investigate(snapshot, emit, reserve_attempt=self.reserve_attempt)
            metadata = result.get("metadata", {})
            if metadata.get("account_limited"):
                PROVIDER_CIRCUIT.update(until=time.monotonic() + 300, reason="Provider account rate limit; automatic reviews are temporarily paused.")
            if "nonzero" in str(metadata.get("failure_reason", "")).lower() or "non-zero" in str(metadata.get("failure_reason", "")).lower():
                PROVIDER_CIRCUIT.update(until=float("inf"), reason="Provider reported a nonzero cost; restart after verifying free-model configuration.")
            self.audit.append({"kind": "assessment", "well_id": well, "snapshot_revision": revision, "result": safe(result)})
            if self.status in TERMINAL:
                return
            if state["revision"] != revision:
                item["activity"] = "New evidence arrived; refreshing assessment"
                self.queue(well, item["activity"])
                return
            live = result.get("source") == "live_llm"
            self.live_verified = self.live_verified or live
            proposed = "unavailable" if result.get("status") == "telemetry" else result.get("status", item["status"])
            if live and proposed in RANK and RANK[proposed] < RANK[item["status"]]:
                state["agent_status"] = proposed
                state["normal_run"] = 0
                item["status"] = proposed
                if item["incident"]:
                    item["incident"]["acknowledged"] = False
                    item["incident"]["completed"] = False
                else:
                    item["incident"] = {"id": uuid.uuid4().hex, "acknowledged": False, "completed": False, "opened_at": snapshot["as_of"], "note": ""}
            evidence = [e.get("summary", str(e)) if isinstance(e, dict) else str(e) for e in result.get("evidence", [])]
            item["assessment"] = {"summary": result.get("brief", "Review the available sensor evidence."), "evidence": evidence[:3], "next_step": result.get("next_action", "Verify the sensor trends."),
                                  "source": "live" if live else "rules", "uncertainty": " ".join(filter(None, [result.get("alternative", ""), str(result.get("missing_evidence", "")) if result.get("missing_evidence") else ""])), "tools": trace[-12:]}
            if not live:
                # Provider failure must not replace measured evidence with generic prose.
                item["assessment"] = {**self.default_assessment(well, state["result"], item["status"]), "tools": trace[-12:],
                                      "uncertainty": "Live LLM review unavailable. " + str(metadata.get("failure_reason") or "Numerical monitoring continues.")}
            item["last_assessed"] = snapshot["as_of"]
            item["investigation"] = "complete" if live else "unavailable"
            item["activity"] = "Assessment complete" if live else result.get("metadata", {}).get("failure_reason") or "Numerical assessment · LLM unavailable"
            if not live and time.monotonic() < PROVIDER_CIRCUIT["until"]:
                item["activity"] = PROVIDER_CIRCUIT["reason"]
            recheck = max(1, min(60, int(result.get("recheck_minutes", 15))))
            state["due"] = max(0, self.index - 1) + recheck
            state["last_investigation_index"] = self.index
            item["next_check"] = (pd.Timestamp(item["source_timestamp"]) + pd.Timedelta(minutes=recheck)).isoformat()
            state["priorities"].append({"as_of": snapshot["as_of"], "brief": result.get("brief"), "status": item["status"], "source": result.get("source")})
            self.publish("assessment")
        except asyncio.CancelledError:
            raise
        except Exception:
            item["investigation"] = "unavailable"
            item["activity"] = "Investigation unavailable; numerical monitoring continues"
            state["due"] = max(0, self.index - 1) + 15
            self.publish("investigation_failed")
        finally:
            self.jobs.pop(well, None)
            if well in self.pending and self.status not in TERMINAL:
                item["investigation"] = "queued"
            self.wake.set()

    def dispatch(self):
        if self.status != "running":
            return
        # Concurrency is shared across sessions, not multiplied by opening another tab.
        total_jobs = sum(len(s.jobs) for s in SESSIONS.values())
        for well in self.priority():
            if total_jobs >= 2:
                break
            if well in self.pending and well not in self.jobs:
                self.pending.remove(well)
                self.jobs[well] = asyncio.create_task(self.investigate_well(well))
                total_jobs += 1

    async def run(self):
        try:
            await self.prepare()
            next_tick = time.monotonic()
            while self.status not in TERMINAL:
                if self.index >= TOTAL:
                    if not self.jobs and not self.pending:
                        self.status = "completed"
                        self.publish("completed")
                        break
                elif self.step_requested or (self.status == "running" and time.monotonic() >= next_tick):
                    stepping = self.step_requested
                    self.step_requested = False
                    self.advance()
                    next_tick = time.monotonic() + 60 / self.speed
                    if stepping:
                        self.status = "paused"
                        self.publish("stepped")
                self.dispatch()
                self.wake.clear()
                try:
                    await asyncio.wait_for(self.wake.wait(), timeout=.1)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self.status = "failed"
            self.error = f"Fleet preparation or monitoring failed: {type(exc).__name__}: {str(exc)[:180]}"
            self.publish("failed")
        finally:
            for job in list(self.jobs.values()):
                job.cancel()

    def control(self, action, speed=None, well_id=None, fault=None):
        if self.status == "preparing":
            if action != "cancel":
                raise ValueError("Wait for the recordings to load")
        elif self.status in TERMINAL:
            raise ValueError("This run has ended; start a new run")
        if action == "pause":
            self.status = "paused"
        elif action == "resume":
            self.status = "running"
        elif action == "step":
            self.status = "paused"
            self.step_requested = True
        elif action == "cancel":
            self.status = "cancelled"
            self.pending.clear()
            for job in list(self.jobs.values()):
                job.cancel()
            if self.task:
                self.task.cancel()
        elif action == "speed":
            if speed not in {30, 60, 120}:
                raise ValueError("Choose speed 30, 60, or 120")
            self.speed = speed
        elif action == "inject":
            if well_id not in WELLS or fault not in {"pressure_offline", "restore"}:
                raise ValueError("Choose a well and a supported fault")
            self.fault = None if fault == "restore" else {"well_id": well_id, "kind": fault, "remaining": 5}
            self.state[well_id]["revision"] += 1
            self.audit.append({"kind": "injection", "well_id": well_id, "fault": fault, "excluded_from_benchmark": True, "at_index": self.index})
        self.publish(action)
        self.wake.set()

    def operator_action(self, well, action, note):
        if well not in self.wells:
            raise ValueError("Unknown well")
        item = self.wells[well]
        if not item["incident"]:
            raise ValueError("This well has no incident")
        if action == "acknowledge":
            item["incident"]["acknowledged"] = True
        elif action == "complete":
            item["incident"]["completed"] = True
        elif action == "observation":
            if not note or not note.strip():
                raise ValueError("Enter an observation")
            item["incident"]["note"] = note.strip()
            self.state[well]["revision"] += 1
            if self.status not in TERMINAL:
                self.queue(well, "Reviewing the operator observation")
        self.audit.append({"kind": "operator_action", "well_id": well, "action": action, "note": note, "human_report_not_ground_truth": True, "at_index": self.index})
        self.publish("operator_action")
        self.wake.set()


class Start(BaseModel):
    speed: Literal[30, 60, 120] = 60


class Control(BaseModel):
    action: Literal["pause", "resume", "step", "cancel", "speed", "inject"]
    speed: Literal[30, 60, 120] | None = None
    well_id: str | None = None
    fault: Literal["pressure_offline", "restore"] | None = None


class OperatorAction(BaseModel):
    action: Literal["acknowledge", "observation", "complete"]
    note: str | None = Field(default=None, max_length=500)


def get_session(id):
    if id not in SESSIONS:
        raise HTTPException(404, "Run not found or server restarted. Start a new fleet replay.")
    return SESSIONS[id]


@router.get("/catalog")
def catalog():
    return {**capabilities(), "wells": [{"id": w, "name": empty_well(w)["name"], "source_file": ""} for w in WELLS], "default_speed": 60, "speeds": [30, 60, 120], "description": "Four historical recordings replayed together; map positions are illustrative."}


@router.get("/results")
def fleet_results():
    path = ROOT / "data/processed/research/fleet_report.json"
    return json.loads(path.read_text()) if path.exists() else {"status": "not_run"}


@router.post("/sessions", status_code=201)
async def start(config: Start):
    if not capabilities()["model_ready"]:
        raise HTTPException(503, capabilities()["readiness_message"])
    if sum(s.status not in TERMINAL for s in SESSIONS.values()) >= 4:
        raise HTTPException(409, "Four replays are already active. Stop an existing run first.")
    # Retain a small, bounded in-memory history. Exports preserve complete journals.
    for key in list(SESSIONS):
        if len(SESSIONS) < 8:
            break
        if SESSIONS[key].status in TERMINAL:
            del SESSIONS[key]
    session = FleetSession(config.speed)
    SESSIONS[session.id] = session
    session.publish("created")
    session.task = asyncio.create_task(session.run())
    return session.snapshot()


@router.get("/sessions/{id}")
async def snapshot(id: str):
    return get_session(id).snapshot()


@router.post("/sessions/{id}/control")
async def control(id: str, command: Control):
    session = get_session(id)
    try:
        session.control(**command.model_dump())
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    return session.snapshot()


@router.post("/sessions/{id}/incidents/{well}/actions")
async def operator_action(id: str, well: str, command: OperatorAction):
    session = get_session(id)
    try:
        session.operator_action(well, command.action, command.note)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return session.snapshot()


@router.get("/sessions/{id}/events")
async def events(id: str, request: Request, after: int = Query(0, ge=0), last_event_id: str | None = Header(None)):
    session = get_session(id)
    try:
        cursor = max(after, int(last_event_id or 0))
    except ValueError:
        raise HTTPException(422, "Invalid event cursor")
    if cursor > len(session.events):
        raise HTTPException(422, "Event cursor is ahead of this run")
    async def stream():
        nonlocal cursor
        while not await request.is_disconnected():
            for event in session.events[cursor:]:
                cursor = event["id"]
                yield f"id: {cursor}\nevent: fleet\ndata: {json.dumps(event, allow_nan=False)}\n\n"
            if session.status in TERMINAL and cursor >= len(session.events):
                break
            session.changed.clear()
            if cursor < len(session.events):
                continue
            try:
                await asyncio.wait_for(session.changed.wait(), timeout=15)
            except asyncio.TimeoutError:
                yield ": heartbeat\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/sessions/{id}/export")
async def export(id: str):
    session = get_session(id)
    payload = {"run": session.snapshot(), "audit": session.audit, "events": session.events, "method": "Four independent historical excerpts. Labels are excluded from runtime. Injected data is excluded from published benchmark scores."}
    return Response(json.dumps(safe(payload), indent=2, allow_nan=False), media_type="application/json", headers={"Content-Disposition": f'attachment; filename="frostline-fleet-{id[:8]}.json"'})
