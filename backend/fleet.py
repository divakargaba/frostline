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
        reason = "Model ready. Automatic evidence checks are enabled; LLM investigations are deferred."
    except Exception as exc:
        ready = False
        reason = "Train the fleet model before starting: python scripts/train_fleet_model.py"
        if not isinstance(exc, (ImportError, FileNotFoundError)):
            reason = "The saved fleet model could not be loaded. Rebuild it with scripts/train_fleet_model.py."
    return {"model_ready": ready, "llm_configured": configured, "llm_deferred": True, "readiness_message": reason}


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
            "timeline": [], "followup": {"last_checked_at": None, "next_check_at": None, "trigger": "", "change_summary": "Awaiting the first assessment."}}


class FleetSession:
    def __init__(self, speed=60, *, use_llm=False, mode="continuous"):
        if mode not in {"guided", "continuous"}:
            raise ValueError("Choose guided or continuous replay")
        self.id = uuid.uuid4().hex
        self.speed = speed
        self.mode = mode
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
        # Provider integration is retained for explicit tests/future work. Public runs
        # stay local even if a developer happens to have a provider key configured.
        self.use_llm = use_llm
        self.request_budget = 18
        self.fault = None
        self.error = None
        self.caps = capabilities()
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.task = None
        self.step_requested = False
        self.guide = {"phase": "overview", "checkpoint": None}
        self._guide_seen_watch = set()
        self._guide_seen_watch_recovery = set()
        self._guide_scenarios = {}
        self._guide_candidates = []
        self._guide_checkpoint_pending = None
        self._guide_settle = None
        self._guide_operator_wells = set()

    def priority(self):
        return sorted(WELLS, key=lambda w: (RANK[self.wells[w]["status"]], bool((self.wells[w].get("incident") or {}).get("acknowledged")), self.state[w]["due"], WELLS.index(w)))

    def snapshot(self):
        return safe({"id": self.id, "status": self.status, "speed": self.speed, "mode": self.mode, "guide": deepcopy(self.guide), "elapsed_seconds": self.index * 60, "index": self.index, "total": TOTAL,
                     "agent_mode": "live" if self.live_verified else "rules", **self.caps, "wells": [deepcopy(self.wells[w]) for w in WELLS],
                     "priority": self.priority(), "last_event_id": len(self.events), "requests_used": self.requests_used, "request_budget": self.request_budget,
                     "fault": self.fault, "error": self.error})

    def publish(self, reason="update"):
        if (not self.use_llm and self.mode == "guided" and self.guide["phase"] in {"seeking", "assessing"}
                and reason in {"tick", "investigation_started", "assessment"}):
            # Guided seeking has no useful intermediate reading to linger on.
            # Keep every audit event, but publish the next settled snapshot as
            # one coherent view instead of thousands of full-history copies.
            self.audit.append({"kind": reason, "elapsed_seconds": self.index * 60, "snapshot_coalesced": True})
            return
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
        # Warm-up charts are observed context, not a secretly advanced alarm
        # state. Both modes process their first monitor tick at replay index 0.
        self.status = "paused" if self.mode == "guided" else "running"
        self.guide["phase"] = "overview" if self.mode == "guided" else "seeking"
        self.publish("prepared")

    def guide_checkpoint(self, kind, well=None, *, title=None, summary=None):
        item = self.wells[well] if well else None
        titles = {
            "concern": "A new concern needs review", "escalation": "A concern needs priority review",
            "recovery": "The monitored concern has eased", "telemetry_loss": "Pressure telemetry needs verification",
            "telemetry_recovery": "Pressure telemetry has improved", "operator_followup": "Your requested review is complete",
            "step": "One source minute reviewed", "ended": "The historical replay has ended",
        }
        return {"id": uuid.uuid4().hex, "kind": kind, "title": title or titles[kind],
                "summary": summary or ((item.get("assessment") or {}).get("summary") if item else None) or "Review the current measurements and completed evidence checks.",
                "well_id": well, "source_timestamp": item["source_timestamp"] if item else None, "index": self.index}

    @staticmethod
    def critical_telemetry(quality):
        return tuple(sorted(set(quality.get("missing", []) + quality.get("invalid", [])) & set(PRESSURES)))

    def record_guided_change(self, well, previous, previous_quality, had_reading, assessment):
        """Presentation stopping points use observed state only, never labels.

        Repeated watch/recheck events still update the full UI and audit. Only
        their presentation stops are deduplicated. Escalations and critical
        telemetry transitions always remain eligible for a checkpoint.
        """
        item = self.wells[well]
        status = item["status"]
        scenario = assessment.get("scenario", assessment.get("category", "monitoring"))
        earlier_scenario = self._guide_scenarios.get(well)
        self._guide_scenarios[well] = scenario
        old_quality = self.critical_telemetry(previous_quality) if had_reading else ()
        new_quality = self.critical_telemetry(item["quality"])
        kind = None
        if status == "attention" and previous != "attention":
            kind = "escalation"
        elif new_quality != old_quality:
            kind = "telemetry_recovery" if old_quality and set(new_quality).issubset(old_quality) else "telemetry_loss"
        elif had_reading and previous == "attention" and status != "attention":
            kind = "recovery"
        elif status == "watch" and (well, scenario) not in self._guide_seen_watch:
            kind = "concern"
        elif had_reading and previous == "watch" and status == "normal" and (well, earlier_scenario) not in self._guide_seen_watch_recovery:
            kind = "recovery"
            self._guide_seen_watch_recovery.add((well, earlier_scenario))
        if status == "watch":
            self._guide_seen_watch.add((well, scenario))
        if kind and self.mode == "guided":
            candidate = self.guide_checkpoint(kind, well, summary=assessment["summary"])
            # advance() has not yet incremented the shared presentation clock.
            candidate["index"] = self.index + 1
            self._guide_candidates.append(candidate)

    def begin_settling(self, phase, checkpoint=None):
        self._guide_settle = phase
        self._guide_checkpoint_pending = checkpoint
        self.guide["phase"] = "assessing"
        self.status = "running"
        self.publish("guide_assessing")

    def settle_checkpoint(self):
        """Called only when all current and queued investigations have finished."""
        checkpoint = self._guide_checkpoint_pending
        phase = self._guide_settle
        self._guide_settle = None
        self._guide_checkpoint_pending = None
        if self.index >= TOTAL:
            self.status = "completed"
            self.guide = {"phase": "ended", "checkpoint": self.guide_checkpoint("ended", summary="All available source minutes and queued evidence reviews are complete. The final well conditions remain visible.")}
            self.publish("completed")
            return
        if phase == "operator_followup":
            well = checkpoint.get("well_id") if checkpoint else None
            if well:
                item = self.wells[well]
                checkpoint["summary"] = item["followup"]["change_summary"] + " " + item["assessment"]["next_step"]
            self._guide_operator_wells.clear()
        if checkpoint is None:
            checkpoint = self.guide_checkpoint("step" if phase == "step" else "concern")
        # A successful stop is published after assessments, never while jobs are
        # still mutating the card the operator is about to read.
        checkpoint["index"] = self.index
        self.guide = {"phase": "checkpoint", "checkpoint": checkpoint}
        self.status = "paused"
        self.publish("guide_checkpoint" if self.mode == "guided" else "stepped")

    def continue_guided(self):
        if self.status == "running":
            return
        if self._guide_operator_wells:
            well = next((w for w in self.priority() if w in self._guide_operator_wells), None)
            checkpoint = self.guide_checkpoint("operator_followup", well)
            self.begin_settling("operator_followup", checkpoint)
        elif self._guide_checkpoint_pending:
            self.begin_settling("checkpoint", self._guide_checkpoint_pending)
        elif self.jobs or self.pending:
            # A manually interrupted assessment is finished at its current
            # reading before any further historical measurements are consumed.
            self.begin_settling("checkpoint", self.guide_checkpoint("operator_followup", summary="The interrupted evidence review has finished at the same source reading."))
        else:
            self.status = "running"
            self.guide["phase"] = "seeking"
            self._guide_settle = None
            self._guide_candidates.clear()

    def clear_dormant_guidance(self):
        """Continuous playback must not later revive an earlier guided stop."""
        self._guide_checkpoint_pending = None
        self._guide_candidates.clear()
        self._guide_operator_wells.clear()
        self.guide["checkpoint"] = None

    def default_assessment(self, well, result, status):
        from src.fleet_policy import operator_assessment
        assessment = operator_assessment(result, self.history[well], status, self.state[well], self.bundle["policy"])
        return self.merge_checks(well, assessment)

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
        if self.mode == "continuous":
            self.clear_dormant_guidance()
        self._guide_candidates = []
        for well in WELLS:
            item = self.wells[well]
            state = self.state[well]
            previous_quality = item["quality"]
            had_reading = item["source_timestamp"] is not None
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
            previous = item["status"]
            monitored = advance_monitor(state, result, self.history[well], self.bundle["policy"])
            state.update(monitored["state"])
            status, material = monitored["status"], monitored["material"]
            score = scores.get("hydrate")
            item.update(status=status, quality={"status": "unavailable" if blocked else quality.get("status", "good"), "summary": quality.get("summary", ""), "missing": quality.get("missing", []), "invalid": quality.get("invalid", []), "unchanged": quality.get("unchanged", [])}, source_timestamp=t.isoformat())
            item["frames"].append({"t": t.isoformat(), "elapsed_seconds": self.index * 60, "sensors": safe(row.iloc[-1].reindex(SENSORS).to_dict()), "risk_score": scores.get("hydrate")})
            item["frames"] = item["frames"][-240:]
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
                self.queue(well, "Checking recovery against the earlier concern")
            due = self.index >= state["due"]
            if status != "normal" and (material or (due and well not in self.jobs)):
                self.queue(well, "Investigating changed evidence" if material else "Scheduled reassessment")
            elif due and not self.use_llm and well not in self.jobs:
                self.queue(well, "Scheduled review of normal measurements")
            elif due:
                state["due"] = self.index + monitored["recheck_minutes"]
            state["result"] = result
            state["last_score"] = score
            item["next_check"] = (t + pd.Timedelta(minutes=max(1, state["due"] - self.index))).isoformat()
            item["followup"]["next_check_at"] = item["next_check"]
            self.record_guided_change(well, previous, previous_quality, had_reading, monitored["assessment"])
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
        before = deepcopy(item["followup"].get("after"))
        trigger = state.pop("queued_reason", "Scheduled reassessment")
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
            # Local checks are atomic and have no suspension points. Their full
            # evidence remains in the audit/trace; only live network tool calls
            # need intermediate snapshots while the operator is waiting.
            if self.use_llm:
                self.publish(kind)

        try:
            if not self.use_llm:
                # These checks use the current numerical result without consuming
                # another reading or advancing persistence a second time. Keep
                # this local work atomic: emit has no suspension points, so all
                # evidence and timestamps describe the same observed prefix.
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
                recheck = int(item["assessment"].get("recheck_minutes", 15))
                state["due"] = max(0, self.index - 1) + recheck
                state["last_investigation_index"] = self.index
                item["next_check"] = (pd.Timestamp(item["source_timestamp"]) + pd.Timedelta(minutes=recheck)).isoformat()
                item["last_assessed"] = snapshot["as_of"]
                item["investigation"] = "complete"
                item["activity"] = "Evidence checked; monitoring continues"
                self.finish_followup(well, before, trigger)
                state["priorities"].append({"as_of": snapshot["as_of"], "brief": item["assessment"]["summary"], "status": item["status"], "source": "rules"})
                self.audit.append({"kind": "assessment", "well_id": well, "snapshot_revision": revision, "result": safe(item["assessment"]), "llm_deferred": True})
                self.publish("assessment")
                return
            # Use Div's provider pool (Gemini -> Groq -> OpenRouter) if available
            from src.llm import make_fleet_transport
            transport = make_fleet_transport()
            result = await investigate(snapshot, emit, transport=transport, reserve_attempt=self.reserve_attempt)
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
            elif self.mode == "continuous":
                self._guide_operator_wells.discard(well)
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
                if self.step_requested:
                    self.step_requested = False
                    if self.index < TOTAL:
                        self.advance()
                    next_tick = time.monotonic() + 60 / self.speed
                    self.begin_settling("step", self.guide_checkpoint("step", summary="One source minute was processed for all four wells. Evidence reviews are complete; inspect the updated measurements before continuing."))
                elif self.status == "running" and self._guide_settle:
                    self.dispatch()
                    if not self.jobs and not self.pending:
                        self.settle_checkpoint()
                elif self.status == "running" and self.index >= TOTAL:
                    self.begin_settling("ended")
                elif self.status == "running" and self.mode == "guided":
                    # Each batch is one full four-well minute. Yielding below
                    # keeps pause/cancel responsive without dropping any input.
                    self.advance()
                    if self._guide_candidates:
                        order = {"escalation": 0, "telemetry_loss": 1, "recovery": 2, "telemetry_recovery": 3, "concern": 4}
                        candidates = sorted(self._guide_candidates, key=lambda c: (order[c["kind"]], self.priority().index(c["well_id"])))
                        checkpoint = candidates[0]
                        others = len({c["well_id"] for c in candidates}) - 1
                        if others:
                            checkpoint["summary"] += f" Changes at {others} other {'well are' if others == 1 else 'wells are'} visible in the fleet view."
                        self.begin_settling("checkpoint", checkpoint)
                    elif self.index >= TOTAL:
                        self.begin_settling("ended")
                elif self.status == "running" and time.monotonic() >= next_tick:
                    self.advance()
                    next_tick = time.monotonic() + 60 / self.speed
                self.dispatch()
                if self.status == "running" and self.mode == "guided" and not self._guide_settle:
                    await asyncio.sleep(0)
                    continue
                self.wake.clear()
                try:
                    await asyncio.wait_for(self.wake.wait(), timeout=.1)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self.status = "failed"
            self.guide["phase"] = "ended"
            self.error = f"Fleet preparation or monitoring failed: {type(exc).__name__}: {str(exc)[:180]}"
            self.publish("failed")
        finally:
            for job in list(self.jobs.values()):
                job.cancel()

    def control(self, action, speed=None, well_id=None, fault=None, mode=None):
        if self.status == "preparing":
            if action != "cancel":
                raise ValueError("Wait for the recordings to load")
        elif self.status in TERMINAL:
            raise ValueError("This run has ended; start a new run")
        if action == "pause":
            self.status = "paused"
            self.step_requested = False
            self._guide_settle = None
            self.guide["phase"] = "manual_pause"
        elif action == "resume":
            if self.mode == "guided":
                self.continue_guided()
            else:
                self.status = "running"
                self._guide_settle = None
                self.clear_dormant_guidance()
                self.guide["phase"] = "seeking"
        elif action == "next_moment":
            if self.mode != "guided":
                raise ValueError("Switch to guided mode before choosing the next important moment")
            self.continue_guided()
        elif action == "step":
            if self.status == "running":
                raise ValueError("Pause before stepping one source minute")
            self.status = "paused"
            self.step_requested = True
            self._guide_settle = None
            self._guide_operator_wells.clear()
        elif action == "cancel":
            self.status = "cancelled"
            self.guide["phase"] = "ended"
            self.step_requested = False
            self._guide_settle = None
            self.pending.clear()
            for job in list(self.jobs.values()):
                job.cancel()
            if self.task:
                self.task.cancel()
        elif action == "speed":
            if speed not in {12, 30, 60, 120}:
                raise ValueError("Choose speed 12, 30, 60, or 120")
            self.speed = speed
        elif action == "mode":
            if mode not in {"guided", "continuous"}:
                raise ValueError("Choose guided or continuous replay")
            self.mode = mode
            self.status = "paused"
            self.step_requested = False
            self._guide_settle = None
            self.guide["phase"] = "manual_pause"
        elif action == "inject":
            if well_id not in WELLS or fault not in {"pressure_offline", "restore"}:
                raise ValueError("Choose a well and a supported fault")
            self.fault = None if fault == "restore" else {"well_id": well_id, "kind": fault, "remaining": 5}
            self.state[well_id]["revision"] += 1
            self.audit.append({"kind": "injection", "well_id": well_id, "fault": fault, "excluded_from_benchmark": True, "at_index": self.index})
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
        if action == "acknowledge":
            item["incident"]["acknowledged"] = True
            item["incident"]["acknowledged_at"] = stamp
            self.timeline(well, "seen", "Operator acknowledged the alert; monitoring continues.")
        elif action == "complete":
            item["incident"]["completed"] = True
            item["incident"]["completed_at"] = stamp
            if note and note.strip():
                item["incident"]["note"] = note.strip()
                self.timeline(well, "observation", "Operator observation saved: " + note.strip())
                self.state[well]["revision"] += 1
                if self.status not in TERMINAL:
                    self.queue(well, "Reviewing the completed operator review and observation")
            self.timeline(well, "review_completed", "Operator review recorded; recovery still depends on new sensor evidence.")
        elif action == "observation":
            if not note or not note.strip():
                raise ValueError("Enter an observation")
            item["incident"]["note"] = note.strip()
            self.timeline(well, "observation", "Operator observation saved: " + note.strip())
            self.state[well]["revision"] += 1
            if self.status not in TERMINAL:
                self.queue(well, "Reviewing the operator observation")
        elif action == "check":
            check = next((c for c in (item.get("assessment") or {}).get("checks", []) if c["id"] == check_id), None)
            if check is None or result not in {"confirmed", "not_confirmed", "unavailable"}:
                raise ValueError("Choose a current check and a valid result")
            if check_definition != check["definition"]:
                raise ValueError("The requested check changed. Review its current measurements before saving.")
            item["incident"].setdefault("checks", {})[check_id] = {"status": result, "note": (note or "").strip(), "updated_at": stamp, "definition": check["definition"]}
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
        if action != "acknowledge" and well in self.pending:
            self._guide_operator_wells.add(well)
            if self.mode == "guided" and self.status == "running" and not self._guide_settle:
                self.begin_settling("operator_followup", self.guide_checkpoint("operator_followup", well))
        self.audit.append({"kind": "operator_action", "well_id": well, "incident_id": item["incident"]["id"], "action": action, "note": note,
                           "check_id": check_id, "result": result, "human_report_not_ground_truth": True, "at_index": self.index})
        self.publish("operator_action")
        self.wake.set()


class Start(BaseModel):
    speed: Literal[12, 30, 60, 120] = 12
    mode: Literal["guided", "continuous"] = "guided"


class Control(BaseModel):
    action: Literal["pause", "resume", "step", "cancel", "speed", "inject", "next_moment", "mode"]
    speed: Literal[12, 30, 60, 120] | None = None
    mode: Literal["guided", "continuous"] | None = None
    well_id: str | None = None
    fault: Literal["pressure_offline", "restore"] | None = None


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


@router.get("/catalog")
def catalog():
    return {**capabilities(), "wells": [{"id": w, "name": empty_well(w)["name"], "source_file": ""} for w in WELLS],
            "default_mode": "guided", "modes": ["guided", "continuous"], "default_speed": 12, "speeds": [12, 30, 60, 120],
            "description": "Four historical recordings replayed together; map positions are illustrative."}


@router.get("/results")
def fleet_results():
    path = ROOT / "data/processed/research/fleet_report.json"
    return json.loads(path.read_text()) if path.exists() else {"status": "not_run"}


@router.get("/workflow-results")
def workflow_results():
    path = ROOT / "data/processed/research/fleet_workflow_report.json"
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
    session = FleetSession(config.speed, mode=config.mode)
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
    payload = {"run": session.snapshot(), "audit": session.audit, "events": session.events, "method": "Four independent historical excerpts. Labels are excluded from runtime. Injected data is excluded from published benchmark scores."}
    return Response(json.dumps(safe(payload), indent=2, allow_nan=False), media_type="application/json", headers={"Content-Disposition": f'attachment; filename="frostline-fleet-{id[:8]}.json"'})
