"""Four independent historical feeds with a bounded, event-driven investigation queue."""
from __future__ import annotations

import asyncio
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
import json
import hashlib
import math
import os
from pathlib import Path
import time
from typing import Literal
import uuid

import pandas as pd

from src.sensor_insights import SENSOR_INFO, sensor_insights
from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from backend.config import FLEET_LLM_MAX, ROOT

SENSORS = ["P-PDG", "T-PDG", "P-TPT", "T-TPT", "P-MON-CKP", "P-JUS-CKP", "T-JUS-CKP", "ABER-CKP", "QGL"]
WELLS = ["WELL-00001", "WELL-00002", "WELL-00006", "WELL-00019"]
# These are fixed presentation excerpts, not classifier inputs or benchmark selection.
OFFSETS = {"WELL-00001": "2017-02-01T02:03:00", "WELL-00002": "2014-02-12T20:15:00", "WELL-00006": "2018-06-18T11:31:00", "WELL-00019": "2012-06-02T21:55:00"}
RANK = {"attention": 0, "unavailable": 1, "watch": 2, "normal": 3}
TERMINAL = {"completed", "cancelled", "failed"}
SPEEDS = [12, 30, 60, 120]
TOTAL = 180
MIN_INVESTIGATION_GAP = 90  # replay minutes between automatic investigations per well
# Provider-pool calls include pacing and slower hosted models, so they get a wider window than direct OpenRouter.
LLM_ATTEMPT_SECONDS, LLM_DEADLINE_SECONDS = 20.0, 45.0
SESSIONS: dict[str, "FleetSession"] = {}
ATTEMPTS: deque[float] = deque()
PROVIDER_CIRCUIT = {"until": 0.0, "reason": ""}
router = APIRouter(prefix="/api/fleet", tags=["Four-well operations"])


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


@lru_cache(maxsize=2)
def _bundle(mtime_ns):
    from src.fleet_model import load_bundle
    return load_bundle()


def cached_bundle():
    """The trusted bundle, unpickled once per file version and shared read-only by sessions."""
    from src.fleet_model import BUNDLE_PATH
    return _bundle(Path(BUNDLE_PATH).stat().st_mtime_ns)


def model_status():
    try:
        cached_bundle()
        return True, ""
    except (ImportError, FileNotFoundError):
        return False, "Train the fleet model before starting: python scripts/train_fleet_model.py"
    except Exception:
        return False, "The saved fleet model could not be loaded. Rebuild it with scripts/train_fleet_model.py."


def capabilities():
    from src.llm import configured_providers
    ready, reason = model_status()
    providers = configured_providers()
    if ready:
        reason = ("Model ready. Rule-based evidence checks run by default; LLM investigations are available via " + ", ".join(providers) + "."
                  if providers else "Model ready. Rule-based evidence checks are active; set GEMINI_API_KEY, GROQ_API_KEY or OPENROUTER_API_KEY to enable LLM investigations.")
    return {"model_ready": ready, "llm_configured": bool(providers), "llm_providers": providers, "readiness_message": reason}


def llm_transport():
    """Provider-pool transport for live investigations, or None when no provider is usable."""
    from src.llm import make_fleet_transport
    return make_fleet_transport()


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
            "last_assessed": None, "next_check": None, "incident": None, "assessment": None, "frames": [], "investigation": "idle", "activity": "Preparing recording",
            "prediction": None, "insights": None, "timeline": [], "followup": {"last_checked_at": None, "next_check_at": None, "trigger": "", "change_summary": "Awaiting the first assessment."}}


def frame(t, elapsed, sensors, risk, limits=None):
    # Frames are JSON-safe and never mutated after creation, so snapshots share them by reference.
    return {"t": t.isoformat(), "elapsed_seconds": int(elapsed), "sensors": safe(sensors.reindex(SENSORS).to_dict()), "risk_score": safe(risk), "limits": safe(limits)}


def limits(alarm, policy, signals):
    """The decision limits applied at this minute: hydrate hysteresis and the window-relative pressure trigger."""
    active = bool(alarm.get("active"))
    return {"hydrate_threshold": policy["recovery_threshold"] if active else policy["activation_threshold"],
            "alarm_active": active, "activation_streak": int(alarm.get("activation_streak", 0) or 0),
            "recovery_streak": int(alarm.get("recovery_streak", 0) or 0),
            "pressure_trigger_bar": signals.get("pressure_trigger_bar"), "pressure_change_bar": signals.get("pressure_change_bar"),
            "divergence_change_bar": signals.get("line_difference_change_bar")}


class FleetSession:
    def __init__(self, speed=12, *, use_llm=False):
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
        self.state = {w: {"alarm": {}, "other_run": 0, "other_active": False, "other_recovery": 0, "normal_run": 0, "agent_status": "normal", "base_status": "unavailable", "revision": 0, "due": 0, "last_score": None, "revision_score": None, "revision_pressure": None, "last_quality": None, "last_investigation_index": -100, "last_investigation_category": None, "last_investigation_status": None, "priorities": []} for w in WELLS}
        self.llm_investigations_used = 0
        self.bundle = None
        self.requests_used = 0
        self.request_budget = FLEET_LLM_MAX
        self.live_verified = False
        self.error = None
        self.caps = capabilities()
        self.set_llm(use_llm)
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.task = None
        self.step_requested = False
        self.settling = False

    def set_llm(self, requested):
        """LLM review is opt-in per run; without a configured provider the run stays on rules."""
        self.llm_requested = bool(requested)
        self.use_llm = self.llm_requested and self.caps["llm_configured"]

    def priority(self):
        return sorted(WELLS, key=lambda w: (RANK[self.wells[w]["status"]], bool((self.wells[w].get("incident") or {}).get("acknowledged")), self.state[w]["due"], WELLS.index(w)))

    def snapshot(self):
        caps = dict(self.caps)
        if self.llm_requested and not self.use_llm:
            caps["readiness_message"] = "LLM investigations were requested but no provider key is configured; using rule-based evidence checks."
        wells = [{**safe({k: v for k, v in self.wells[w].items() if k != "frames"}), "frames": list(self.wells[w]["frames"])} for w in WELLS]
        return {"id": self.id, "status": self.status, "speed": self.speed, "elapsed_seconds": self.index * 60, "index": self.index, "total": TOTAL,
                "agent_mode": "live" if self.live_verified else "rules", "use_llm": self.use_llm, **caps, "wells": wells,
                "priority": self.priority(), "last_event_id": len(self.events), "requests_used": self.requests_used,
                "request_budget": self.request_budget, "llm_investigations_used": self.llm_investigations_used, "llm_investigation_cap": FLEET_LLM_MAX, "error": self.error}

    def publish(self, reason="update"):
        event_id = len(self.events) + 1
        payload = self.snapshot()
        payload["last_event_id"] = event_id
        self.events.append({"id": event_id, "type": "snapshot", "payload": payload})
        self.audit.append({"kind": reason, "event_id": event_id, "elapsed_seconds": self.index * 60})
        self.changed.set()

    async def prepare(self):
        self.sources = await asyncio.to_thread(source_frames)
        self.bundle = await asyncio.to_thread(cached_bundle)
        excluded = set(self.bundle.get("excluded_wells", self.bundle.get("excluded_well_ids", WELLS)))
        training = set(self.bundle.get("training_wells", self.bundle.get("train_wells", self.bundle.get("training_well_ids", []))))
        if training.intersection(WELLS) or not set(WELLS).issubset(excluded):
            raise ValueError("The demo model must hold out every recording from all four demo wells")
        for well, source in self.sources.items():
            self.history[well] = source["warmup"].copy()
            self.wells[well]["source_file"] = source["source_file"]
            start = source["future"].index[0]
            # Warm-up provides preceding measurements, not assumed healthy labels.
            self.wells[well]["frames"] = [frame(t, (t - start).total_seconds(), row, None) for t, row in source["warmup"].iloc[-30:].iterrows()]
            self.wells[well]["activity"] = "Ready to monitor"
        self.status = "running"
        self.publish("prepared")

    def default_assessment(self, well, result, status):
        from src.fleet_policy import operator_assessment
        assessment = operator_assessment(result, self.history[well], status, self.state[well], self.bundle["policy"])
        return self.merge_checks(well, assessment)

    def should_investigate(self, well, status, previous, material, assessment, operator_request=False):
        """Gate investigations to prevent flooding."""
        state = self.state[well]
        gap = self.index - state["last_investigation_index"]
        category = assessment.get("category", assessment.get("scenario", "monitoring"))
        if operator_request:
            return True
        if status == "attention" and RANK.get(state.get("last_investigation_status", previous), 3) > RANK["attention"]:
            return True
        last_cat = state.get("last_investigation_category")
        if last_cat is not None and category != last_cat and category not in ("normal", "monitoring"):
            return True
        if state["last_investigation_index"] <= 0:
            return True
        if gap < MIN_INVESTIGATION_GAP:
            return False
        if material and status != state.get("last_investigation_status"):
            return True
        if self.index >= state["due"] and status != "normal":
            return True
        return False

    def queue(self, well, reason):
        self.pending.add(well)
        self.state[well]["queued_reason"] = reason
        if well not in self.jobs:
            self.wells[well]["investigation"] = "queued"
            self.wells[well]["activity"] = reason

    def timeline(self, well, kind, summary):
        item = self.wells[well]
        item["timeline"].append({"id": uuid.uuid4().hex, "kind": kind,
                                 "at": item["source_timestamp"], "recorded_at": datetime.now(timezone.utc).isoformat(), "summary": summary})
        if len(item["timeline"]) > 80:
            entries = item["timeline"]
            # Frequent scheduled checks must not evict the incident milestones
            # used by the operator's Detected / Seen / Checked / Recovered view.
            keep = {next((i for i, entry in enumerate(entries) if entry["kind"] == "detected"), 0)}
            for milestone in ["seen", "checked", "review_completed", "recovered"]:
                found = next((i for i in range(len(entries) - 1, -1, -1) if entries[i]["kind"] == milestone), None)
                if found is not None:
                    keep.add(found)
            for i in range(len(entries) - 1, -1, -1):
                if len(keep) >= 80:
                    break
                keep.add(i)
            item["timeline"] = [entries[i] for i in sorted(keep)]

    def open_incident(self, well):
        item = self.wells[well]
        item["incident"] = {"id": uuid.uuid4().hex, "acknowledged": False, "completed": False,
                            "opened_at": item["source_timestamp"], "note": "", "checks": {}, "condition_cleared": False}
        item["timeline"] = []
        item["last_assessed"] = None
        item["followup"] = {"last_checked_at": None, "next_check_at": item["next_check"],
                            "trigger": "New incident", "change_summary": "New evidence is awaiting its first review."}
        self.timeline(well, "detected", item["assessment"]["summary"] if item["assessment"] else "New evidence needs review.")

    def merge_checks(self, well, assessment):
        checks = (self.wells[well].get("incident") or {}).get("checks", {})
        result = deepcopy(assessment)
        result["category"] = result.get("category", result.get("scenario", "monitoring"))
        merged = []
        for check in result.get("checks", []):
            definition = hashlib.sha256(json.dumps([check["id"], check["label"], check["reason"]]).encode()).hexdigest()[:16]
            saved = checks.get(check["id"], {})
            saved = saved if saved.get("definition") == definition else {}
            merged.append({**check, "status": "pending", "note": "", "updated_at": None, **saved, "definition": definition})
        result["checks"] = merged
        return result

    def finish_followup(self, well, before, trigger):
        item = self.wells[well]
        after = {"at": item["source_timestamp"], "status": item["status"],
                 "summary": item["assessment"]["summary"], "evidence": item["assessment"]["evidence"]}
        if not before:
            change = "First evidence check completed."
        elif before["at"] == after["at"]:
            change = "Same source reading reviewed; no new measurements have arrived."
        elif before["status"] != after["status"]:
            change = f"Status changed from {before['status']} to {after['status']} with new measurements."
        elif before["evidence"] != after["evidence"]:
            change = "New measurements reviewed; the status is unchanged."
        else:
            change = "New readings checked; the evidence and status remain stable."
        item["followup"] = {"last_checked_at": after["at"], "next_check_at": item["next_check"],
                            "trigger": trigger, "change_summary": change, "before": before, "after": after}
        if item["incident"]:
            self.timeline(well, "recheck", change)

    def advance(self):
        from src.fleet_model import infer_window
        from src.fleet_policy import advance_monitor
        if self.index >= TOTAL:
            return
        policy = self.bundle["policy"]
        for well in WELLS:
            item = self.wells[well]
            state = self.state[well]
            row = self.sources[well]["future"].iloc[self.index:self.index + 1]
            self.history[well] = pd.concat([self.history[well], row]).iloc[-180:]
            result = infer_window(self.history[well], self.bundle)
            scores = result.get("scores", {})
            quality = result.get("quality", {})
            blocked = bool(quality.get("blocked", quality.get("status") == "unavailable"))
            t = row.index[-1]
            previous = item["status"]
            monitored = advance_monitor(state, result, self.history[well], policy)
            state.update(monitored["state"])
            status, material = monitored["status"], monitored["material"]
            item.update(status=status, quality={"status": "unavailable" if blocked else quality.get("status", "good"), "summary": quality.get("summary", ""), "missing": quality.get("missing", []), "invalid": quality.get("invalid", []), "unchanged": quality.get("unchanged", [])}, source_timestamp=t.isoformat())
            alarm = state.get("alarm", {})
            item["frames"] = item["frames"][-239:] + [frame(t, self.index * 60, row.iloc[-1], scores.get("hydrate"), limits(alarm, policy, monitored["signals"]))]
            item["prediction"] = {"scores": safe(scores), "model_id": self.bundle.get("model_id", "fleet-v1"),
                                  "threshold": policy.get("activation_threshold"), "recovery_threshold": policy.get("recovery_threshold"),
                                  "persistence_minutes": policy.get("persistence_minutes"), "recovery_minutes": policy.get("recovery_minutes"),
                                  "alarm_streak": int(alarm.get("activation_streak", 0) or 0), "alarm_active": bool(alarm.get("active"))}
            item["insights"] = sensor_insights(self.history[well])
            if material or item["assessment"] is None:
                item["assessment"] = self.default_assessment(well, result, status)
            if status != "normal":
                if not item["incident"] or item["incident"].get("condition_cleared"):
                    self.open_incident(well)
                    item["assessment"] = self.merge_checks(well, monitored["assessment"])
                elif RANK[status] < RANK[previous]:
                    item["incident"]["acknowledged"] = False
                    item["incident"]["completed"] = False
                    self.timeline(well, "detected", "The concern escalated; review the updated evidence.")
            elif item["incident"] and not item["incident"].get("condition_cleared"):
                item["incident"]["condition_cleared"] = True
                self.timeline(well, "recovered", "Available measurements now meet the monitoring recovery rules.")
                if state.get("last_investigation_status") == "attention" and self.index - state["last_investigation_index"] >= MIN_INVESTIGATION_GAP:
                    self.queue(well, "Checking recovery against the earlier concern")
            due = self.index >= state["due"]
            first_assessment = state["last_investigation_index"] <= -100
            if first_assessment and well not in self.jobs:
                self.queue(well, "Initial assessment")
            elif status != "normal" and well not in self.jobs:
                if self.should_investigate(well, status, previous, material, monitored["assessment"]):
                    self.queue(well, "Investigating changed evidence" if material else "Scheduled reassessment")
                elif due:
                    state["due"] = self.index + monitored["recheck_minutes"]
            elif due:
                state["due"] = self.index + monitored["recheck_minutes"]
            state["result"] = result
            state["last_score"] = scores.get("hydrate")
            item["next_check"] = item["followup"]["next_check_at"] = (t + pd.Timedelta(minutes=max(1, state["due"] - self.index))).isoformat()
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
        if self.status in TERMINAL or self.requests_used >= self.request_budget or len(ATTEMPTS) >= 10 or now < PROVIDER_CIRCUIT["until"] or self.llm_investigations_used >= FLEET_LLM_MAX:
            return False
        self.requests_used += 1
        ATTEMPTS.append(now)
        return True

    def schedule_recheck(self, well, recheck, as_of):
        state, item = self.state[well], self.wells[well]
        state["due"] = max(0, self.index - 1) + recheck
        state["last_investigation_index"] = self.index
        state["last_investigation_category"] = (item.get("assessment") or {}).get("category", (item.get("assessment") or {}).get("scenario"))
        state["last_investigation_status"] = item["status"]
        item["next_check"] = (pd.Timestamp(item["source_timestamp"]) + pd.Timedelta(minutes=recheck)).isoformat()
        item["last_assessed"] = as_of

    async def investigate_well(self, well):
        from src.fleet_llm import investigate
        self.pending.discard(well)
        item = self.wells[well]
        state = self.state[well]
        snapshot = self.agent_snapshot(well)
        revision = state["revision"]
        before = deepcopy(item["followup"].get("after"))
        trigger = state.pop("queued_reason", "Scheduled reassessment")
        transport = llm_transport() if self.use_llm else None
        live_run = transport is not None
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
            # Local checks are atomic with no suspension points; only network
            # tool calls need intermediate snapshots while the operator waits.
            if live_run:
                self.publish(kind)

        try:
            if not live_run:
                # Rule checks reuse the current numerical result without consuming another
                # reading, so all evidence and timestamps describe the same observed prefix.
                from src.fleet_policy import observation_context
                context = observation_context(self.history[well])
                has_score = any(v is not None for v in state["result"].get("scores", {}).values())
                for name, summary in [
                    ("sensor_quality", state["result"]["quality"].get("summary", "Sensor availability checked.")),
                    ("causal_model", "Reviewed the existing frozen-model score from observed history; labels excluded." if has_score else "Model abstained: no usable pressure evidence."),
                    ("pressure_trends", "Reviewed the measured production-line pressure difference." if context["line_difference_bar"] is not None else "Production-line pressure comparison unavailable; a required channel is missing or invalid."),
                    ("incident_followup", "Applied the scenario checklist and retained operator reports separately."),
                ]:
                    await emit({"type": "tool_call", "payload": {"tool": name, "purpose": summary}})
                    await emit({"type": "tool_result", "payload": {"tool": name, "summary": summary}})
                if self.status in TERMINAL:
                    return
                if state["revision"] != revision:
                    self.queue(well, "New evidence arrived; refreshing assessment")
                    return
                item["assessment"] = self.merge_checks(well, {**self.default_assessment(well, state["result"], item["status"]), "tools": trace})
                self.schedule_recheck(well, int(item["assessment"].get("recheck_minutes", 15)), snapshot["as_of"])
                item["investigation"] = "complete"
                item["activity"] = "Evidence checked; monitoring continues"
                self.finish_followup(well, before, trigger)
                state["priorities"].append({"as_of": snapshot["as_of"], "brief": item["assessment"]["summary"], "status": item["status"], "source": "rules"})
                self.audit.append({"kind": "assessment", "well_id": well, "snapshot_revision": revision, "result": safe(item["assessment"]), "source": "rules"})
                self.publish("assessment")
                return
            result = await investigate(snapshot, emit, transport=transport, reserve_attempt=self.reserve_attempt, allowed_models=None,
                                       attempt_seconds=LLM_ATTEMPT_SECONDS, deadline_seconds=LLM_DEADLINE_SECONDS)
            metadata = result.get("metadata", {})
            failure = str(metadata.get("failure_reason", "")).lower()
            if metadata.get("account_limited"):
                PROVIDER_CIRCUIT.update(until=time.monotonic() + 300, reason="Provider account rate limit; automatic reviews are temporarily paused.")
            if "nonzero" in failure or "non-zero" in failure:
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
                    self.open_incident(well)
            if live:
                evidence = [e.get("summary", str(e)) if isinstance(e, dict) else str(e) for e in result.get("evidence", [])]
                item["assessment"] = {**self.merge_checks(well, self.default_assessment(well, state["result"], item["status"])),
                                      "summary": result.get("brief", "Review the available sensor evidence."), "evidence": evidence[:3],
                                      "next_step": result.get("next_action", "Verify the sensor trends."), "source": "live",
                                      "uncertainty": " ".join(filter(None, [result.get("alternative", ""), str(result.get("missing_evidence") or "")])),
                                      "playbook_refs": result.get("playbook_refs", []), "tools": trace[-12:]}
                item["investigation"] = "complete"
                item["activity"] = "LLM assessment complete"
            else:
                # Provider failure must not replace measured evidence with generic prose.
                item["assessment"] = {**self.default_assessment(well, state["result"], item["status"]), "tools": trace[-12:],
                                      "uncertainty": "Live LLM review unavailable. " + str(metadata.get("failure_reason") or "Numerical monitoring continues.")}
                item["investigation"] = "unavailable"
                item["activity"] = PROVIDER_CIRCUIT["reason"] if time.monotonic() < PROVIDER_CIRCUIT["until"] else metadata.get("failure_reason") or "Numerical assessment · LLM unavailable"
            self.schedule_recheck(well, max(1, min(60, int(result.get("recheck_minutes", 15)))), snapshot["as_of"])
            if live:
                self.llm_investigations_used += 1
            self.finish_followup(well, before, trigger)
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
        if self.status != "running" and not self.settling:
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
                if self.step_requested:
                    # One source minute for all wells; its reviews finish before the step is reported.
                    self.step_requested = False
                    self.advance()
                    self.settling = True
                    next_tick = time.monotonic() + 60 / self.speed
                elif self.status == "running" and self.index >= TOTAL:
                    if not self.jobs and not self.pending:
                        self.status = "completed"
                        self.publish("completed")
                        break
                elif self.status == "running" and time.monotonic() >= next_tick:
                    self.advance()
                    next_tick = time.monotonic() + 60 / self.speed
                self.dispatch()
                if self.settling and not self.jobs and not self.pending:
                    self.settling = False
                    self.publish("stepped")
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

    def control(self, action, speed=None, use_llm=None):
        if self.status == "preparing":
            if action != "cancel":
                raise ValueError("Wait for the recordings to load")
        elif self.status in TERMINAL:
            raise ValueError("This run has ended; start a new run")
        if action == "pause":
            self.status = "paused"
            self.step_requested = self.settling = False
        elif action == "resume":
            self.status = "running"
            self.settling = False
        elif action == "step":
            if self.status == "running":
                raise ValueError("Pause before stepping one source minute")
            self.step_requested = True
        elif action == "cancel":
            self.status = "cancelled"
            self.step_requested = self.settling = False
            self.pending.clear()
            for job in list(self.jobs.values()):
                job.cancel()
            if self.task:
                self.task.cancel()
        elif action == "speed":
            if speed not in SPEEDS:
                raise ValueError("Choose speed 12, 30, 60, or 120")
            self.speed = speed
        elif action == "llm":
            if use_llm is None:
                raise ValueError("Choose whether LLM investigations are on")
            self.caps = capabilities()
            self.set_llm(use_llm)
        else:
            raise ValueError("Unknown replay action")
        self.publish(action)
        self.wake.set()

    def operator_action(self, well, action, note=None, check_id=None, result=None, incident_id=None, check_definition=None):
        if well not in self.wells:
            raise ValueError("Unknown well")
        item = self.wells[well]
        if not item["incident"]:
            raise ValueError("This well has no incident")
        if incident_id is not None and incident_id != item["incident"]["id"]:
            raise ValueError("The incident changed. Review the current incident before saving.")
        stamp = datetime.now(timezone.utc).isoformat()
        note = (note or "").strip()
        if action == "acknowledge":
            item["incident"]["acknowledged"] = True
            item["incident"]["acknowledged_at"] = stamp
            self.timeline(well, "seen", "Operator acknowledged the alert; monitoring continues.")
        elif action == "complete":
            item["incident"]["completed"] = True
            item["incident"]["completed_at"] = stamp
            if note:
                item["incident"]["note"] = note
                self.timeline(well, "observation", "Operator observation saved: " + note)
                self.state[well]["revision"] += 1
                if self.status not in TERMINAL:
                    self.queue(well, "Reviewing the completed operator review and observation")
            self.timeline(well, "review_completed", "Operator review recorded; recovery still depends on new sensor evidence.")
        elif action == "observation":
            if not note:
                raise ValueError("Enter an observation")
            item["incident"]["note"] = note
            self.timeline(well, "observation", "Operator observation saved: " + note)
            self.state[well]["revision"] += 1
            if self.status not in TERMINAL:
                self.queue(well, "Reviewing the operator observation")
        elif action == "check":
            check = next((c for c in (item.get("assessment") or {}).get("checks", []) if c["id"] == check_id), None)
            if check is None or result not in {"confirmed", "not_confirmed", "unavailable"}:
                raise ValueError("Choose a current check and a valid result")
            if check_definition != check["definition"]:
                raise ValueError("The requested check changed. Review its current measurements before saving.")
            item["incident"].setdefault("checks", {})[check_id] = {"status": result, "note": note, "updated_at": stamp, "definition": check["definition"]}
            item["assessment"] = self.merge_checks(well, item["assessment"])
            self.timeline(well, "checked", f"{check['label']}: {result.replace('_', ' ')}. Operator report; sensor state unchanged.")
            self.state[well]["revision"] += 1
            if self.status not in TERMINAL:
                self.queue(well, "Reviewing the saved operator check")
        elif action == "recheck":
            if self.status in TERMINAL:
                raise ValueError("This run has ended. Start a new replay to check new measurements.")
            self.queue(well, "Operator requested an evidence recheck")
        else:
            raise ValueError("Unknown operator action")
        self.audit.append({"kind": "operator_action", "well_id": well, "incident_id": item["incident"]["id"], "action": action, "note": note or None,
                           "check_id": check_id, "result": result, "human_report_not_ground_truth": True, "at_index": self.index})
        self.publish("operator_action")
        self.wake.set()


class Start(BaseModel):
    speed: Literal[12, 30, 60, 120] = 12
    use_llm: bool = False


class Control(BaseModel):
    action: Literal["pause", "resume", "step", "cancel", "speed", "llm"]
    speed: Literal[12, 30, 60, 120] | None = None
    use_llm: bool | None = None


class OperatorAction(BaseModel):
    action: Literal["acknowledge", "observation", "complete", "check", "recheck"]
    note: str | None = Field(default=None, max_length=500)
    check_id: str | None = Field(default=None, max_length=100)
    result: Literal["confirmed", "not_confirmed", "unavailable"] | None = None
    incident_id: str | None = Field(default=None, max_length=100)
    check_definition: str | None = Field(default=None, max_length=64)


def get_session(id):
    if id not in SESSIONS:
        raise HTTPException(404, "Run not found or server restarted. Start a new fleet replay.")
    return SESSIONS[id]


def read_report(name):
    path = ROOT / "data/processed/research" / name
    return json.loads(path.read_text()) if path.exists() else {"status": "not_run"}


@router.get("/catalog")
def catalog():
    return {"wells": [{"id": w, "name": empty_well(w)["name"], "source_file": ""} for w in WELLS],
            "default_speed": 12, "speeds": SPEEDS, "sensors": SENSOR_INFO, **capabilities()}


@router.get("/results")
def fleet_results():
    return read_report("fleet_report.json")


@router.get("/workflow-results")
def workflow_results():
    return read_report("fleet_workflow_report.json")


@router.post("/sessions", status_code=201)
async def start(config: Start):
    caps = capabilities()
    if not caps["model_ready"]:
        raise HTTPException(503, caps["readiness_message"])
    if sum(s.status not in TERMINAL for s in SESSIONS.values()) >= 4:
        raise HTTPException(409, "Four replays are already active. Stop an existing run first.")
    # Retain a small, bounded in-memory history. Exports preserve complete journals.
    for key in list(SESSIONS):
        if len(SESSIONS) < 8:
            break
        if SESSIONS[key].status in TERMINAL:
            del SESSIONS[key]
    session = FleetSession(config.speed, use_llm=config.use_llm)
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
        session.operator_action(well, **command.model_dump())
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
    # Readings appear once, in the final run snapshot (it retains the whole
    # replay); per-event snapshots keep every other field without the frames.
    events = [{**event, "payload": {**event["payload"], "wells": [{k: v for k, v in w.items() if k != "frames"} for w in event["payload"]["wells"]]}}
              for event in session.events]
    payload = {"run": session.snapshot(), "audit": session.audit, "events": events, "method": "Four independent historical excerpts. Labels are excluded from runtime."}
    return Response(json.dumps(safe(payload), separators=(",", ":"), allow_nan=False), media_type="application/json", headers={"Content-Disposition": f'attachment; filename="frostline-fleet-{id[:8]}.json"'})


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)


@router.post("/tts")
async def tts(body: TTSRequest):
    from backend.tts import synthesize
    try:
        audio = await synthesize(body.text)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    except Exception:
        raise HTTPException(502, "TTS synthesis failed")
    return Response(audio, media_type="audio/mpeg")
