"""Stateful sensor controller and resumable live experiment sessions.

The controller receives one timestamp + sensor vector at a time. Labels belong
to the evaluator, which is called only after the controller decides. Tool calls
execute as the generator advances; none of the decisions are loaded from the
research replay artifacts. This is a bounded deterministic agent, not an LLM.
"""
from dataclasses import asdict
from copy import deepcopy
from datetime import datetime, timezone
import asyncio
import hashlib
import json
import math
import time
import uuid
import pandas as pd

from src.research import Policy, SENSORS, fit_policy, load_seed, metrics, predict, split_seed


def fingerprint(policy):
    return hashlib.sha256(json.dumps(asdict(policy), sort_keys=True).encode()).hexdigest()[:12]


def event(kind, title, message, **payload):
    return {"type": kind, "title": title, "message": message, "payload": payload}


def clean_tick(row):
    return {"t": pd.Timestamp(row["timestamp"]).isoformat(), "sensors": {s: None if pd.isna(row[s]) else float(row[s]) for s in SENSORS}}


class SensorAgent:
    """Conditional tool planner, short-term memory, follow-up and cooldown."""
    def __init__(self, policy):
        self.policy = policy
        self.history = []
        self.last_decision = None
        self.next_recheck = None
        self.last_notification = None
        self.calls = 0

    def tool(self, name, reason, args, function):
        yield event("tool_call", name, reason, tool=name, args=args)
        # Run ONLY after the call event has been consumed by the scheduler.
        result = function()
        self.calls += 1
        yield event("tool_result", name, result["summary"], tool=name, result=result)
        return result

    def quality(self):
        current = self.history[-1]
        missing = [s for s in SENSORS if current["sensors"][s] is None]
        history = self.history[-6:]
        adjacent = all(pd.Timestamp(b["t"]) - pd.Timestamp(a["t"]) == pd.Timedelta(hours=1) for a, b in zip(history, history[1:]))
        frozen = [s for s in SENSORS if len(history) == 6 and adjacent and all(t["sensors"][s] is not None for t in history) and len({t["sensors"][s] for t in history}) == 1]
        block = "pressure_bar" in missing or bool(frozen) or (self.policy.confirmation_percentile is not None and {"temp_C", "flow_Lps"}.issubset(missing))
        return {"missing": missing, "frozen": frozen, "blocked": block, "summary": f"Missing: {', '.join(missing) or 'none'}. Frozen across six hourly readings: {', '.join(frozen) or 'none'}."}

    def window(self):
        readings = self.history[-6:]
        changes = {}
        for s in SENSORS:
            valid = [r for r in readings if r["sensors"][s] is not None]
            changes[s] = round(valid[-1]["sensors"][s] - valid[0]["sensors"][s], 3) if len(valid) >= 2 else None
        return {"readings": len(readings), "from": readings[0]["t"], "to": readings[-1]["t"], "changes": changes,
                "summary": f"Read {len(readings)} available historical samples. Pressure change: {changes['pressure_bar']} bar; temperature: {changes['temp_C']} °C; flow: {changes['flow_Lps']} L/s."}

    def corroboration(self):
        values = self.history[-1]["sensors"]
        t, f = values["temp_C"], values["flow_Lps"]
        low_t = t is not None and self.policy.temperature_cutoff is not None and t < self.policy.temperature_cutoff
        low_f = f is not None and self.policy.flow_cutoff is not None and f < self.policy.flow_cutoff
        return {"temperature_low": low_t, "flow_low": low_f, "temperature_cutoff": self.policy.temperature_cutoff,
                "flow_cutoff": self.policy.flow_cutoff, "required": self.policy.confirmation_percentile is not None,
                "summary": f"Temperature confirms: {'yes' if low_t else 'no'}. Flow confirms: {'yes' if low_f else 'no'}. " + ("One confirmation is required." if self.policy.confirmation_percentile else "The current pressure-only policy does not require confirmation.")}

    def policy_check(self):
        frame = pd.DataFrame([{ "timestamp": pd.Timestamp(r["t"]), **r["sensors"]} for r in self.history])
        satisfied = bool(predict(frame, self.policy).iloc[-1])
        return {"satisfied": satisfied, "consecutive_hours": self.policy.consecutive,
                "summary": f"Frozen rule {'satisfied' if satisfied else 'not satisfied'} across {self.policy.consecutive} required hourly sample(s). Missing readings and gaps break persistence."}

    def observe(self, tick):
        """Yield actual tool executions and one final decision; no labels accepted."""
        if set(tick) != {"t", "sensors"} or set(tick["sensors"]) != set(SENSORS):
            raise ValueError("Agent input must contain timestamp and sensors only")
        now = pd.Timestamp(tick["t"])
        if self.history and now <= pd.Timestamp(self.history[-1]["t"]):
            raise ValueError("Agent inputs must advance in time")
        self.history.append({"t": tick["t"], "sensors": dict(tick["sensors"])})
        pressure = tick["sensors"]["pressure_bar"]
        crossing = pressure is not None and pressure < self.policy.pressure_cutoff
        followup = self.next_recheck is not None and now >= self.next_recheck
        previous = self.last_decision["decision"] if self.last_decision else None
        quality = yield from self.tool("check_sensor_quality", "Check whether current evidence can support a decision.", {"lookback_samples": 6}, self.quality)
        if quality["blocked"]:
            yield event("trigger", "Investigate telemetry", "Sensor quality blocks event diagnosis; inspect only past sensor history.", reason="quality")
            yield from self.tool("inspect_sensor_history", "Locate missing or unchanged values before interpreting a pressure anomaly.", {"lookback_samples": 6}, self.window)
            decision, diagnosis, rationale = "WATCH", "Telemetry needs attention", "Defer event diagnosis. Restore trustworthy pressure and corroborating sensor evidence."
        elif crossing or followup or previous in ["WATCH", "ALERT"]:
            why = "Scheduled recheck is due." if followup else "Pressure crossed the active threshold."
            yield event("trigger", "Open an investigation", why, reason="recheck" if followup else "pressure", pressure_bar=pressure, cutoff_bar=self.policy.pressure_cutoff)
            yield from self.tool("read_recent_window", "Inspect only readings that have already arrived.", {"lookback_samples": 6}, self.window)
            corroboration = yield from self.tool("compare_secondary_sensors", "Check temperature and flow before applying the active rule.", {}, self.corroboration)
            check = yield from self.tool("test_active_policy", "Verify thresholds and persistence on this session's actual sensor history.", {"policy_id": fingerprint(self.policy)}, self.policy_check)
            if check["satisfied"]:
                decision, diagnosis = "ALERT", "Deterioration rule confirmed"
                rationale = f"Pressure {pressure:.2f} < {self.policy.pressure_cutoff:.2f} bar. " + ("A second sensor confirms the pattern." if corroboration["required"] else "This pressure-only policy escalates without requiring corroboration.")
            elif crossing:
                decision, diagnosis, rationale = "WATCH", "Pressure anomaly unconfirmed", "The pressure trigger is present, but corroboration or persistence is insufficient. Recheck the next input."
            else:
                decision, diagnosis, rationale = "DISMISS", "Conditions recovered", "The scheduled investigation found no active pressure trigger. Return to monitoring."
        else:
            decision, diagnosis, rationale = "DISMISS", "Continue monitoring", f"Pressure {pressure:.2f} bar is above the {self.policy.pressure_cutoff:.2f} bar threshold; no deeper investigation is needed."

        elapsed = (now - self.last_notification).total_seconds() / 3600 if self.last_notification is not None else math.inf
        notify = decision == "ALERT" and elapsed >= 6
        if notify:
            self.last_notification = now
        recheck_hours = 1 if decision in ["WATCH", "ALERT"] else 4
        if self.next_recheck is None or followup or decision in ["WATCH", "ALERT"] or previous in ["WATCH", "ALERT"]:
            self.next_recheck = now + pd.Timedelta(hours=recheck_hours)
        recheck_hours = int((self.next_recheck - now).total_seconds() / 3600)
        record = {"t": tick["t"], "decision": decision, "diagnosis": diagnosis, "rationale": rationale,
                  "notify": notify, "quality": "degraded" if quality["missing"] or quality["frozen"] else "good",
                  "policy_id": fingerprint(self.policy), "next_recheck": self.next_recheck.isoformat(),
                  "recheck_hours": recheck_hours, "cooldown_remaining_hours": max(0, 6 - elapsed) if elapsed != math.inf else 0}
        self.last_decision = record
        yield event("decision", diagnosis, rationale, **record)
        if notify:
            yield event("notification", "Operator alert raised", "A local demo notification was raised. Investigation remains advisory; no equipment command is sent.", t=tick["t"])
        elif decision == "ALERT":
            yield event("cooldown", "Repeat notification suppressed", f"Another operator notification is due in {max(0, 6 - elapsed):.0f} simulated hour(s).", t=tick["t"])
        yield event("scheduled", "Follow-up scheduled", f"Investigate again at {self.next_recheck.isoformat()}; continue checking each incoming sensor sample.", t=tick["t"], next_recheck=self.next_recheck.isoformat())


CATALOG = [
    {"id": "mission", "title": "Learn, then prove", "description": "Observe a 52-hour validation window, test 24 policies, freeze the winner, then process the full 10-day test.", "samples": 292},
    {"id": "sandbox", "title": "Fault-injection sandbox", "description": "48 historical normal hours. Inject faults into future inputs and watch the agent choose different investigations.", "samples": 48},
    {"id": "test", "title": "Final test with the selected policy", "description": "Process all 240 held-out hourly readings. Investigations and decisions are computed as each input arrives.", "samples": 240},
]
FAULTS = {"pressure_dip": 1, "pressure_offline": 4, "hydrate_pattern": 4, "frozen_feed": 8}


class LiveSession:
    def __init__(self, scenario="mission", speed=2, data=None, paced=True):
        self.id = uuid.uuid4().hex
        self.scenario, self.speed, self.paced = scenario, speed, paced
        self.calibration, self.validation, self.test = split_seed(load_seed() if data is None else data)
        self.status, self.phase = "paused", "ready"
        self.created = self.touched = time.monotonic()
        self.events, self.frames, self.trials = [], [], []
        self.policy = fit_policy(self.calibration, 10, None)
        self.agent = SensorAgent(self.policy)
        self.tick_index, self.total, self.step_budget = 0, 0, 0
        self.last_decision, self.score, self.selection = None, None, None
        self.tool_calls, self.injected_excluded = 0, 0
        self.fault = None
        self.changed, self.wake = asyncio.Event(), asyncio.Event()
        self.task = None
        self.error = None
        self.publish(event("created", "Mission ready", "Start the feed to execute the agent. Controller: deterministic, stateful, conditional tool execution.", scenario=scenario, policy=asdict(self.policy), policy_id=fingerprint(self.policy)))

    def publish(self, item):
        item = {**deepcopy(item), "id": len(self.events) + 1, "phase": self.phase, "server_time": datetime.now(timezone.utc).isoformat()}
        if item["type"] in ["tool_result", "candidate_result"]:
            self.tool_calls += 1
        self.events.append(item)
        self.changed.set()
        self.touched = time.monotonic()
        return item

    def snapshot(self):
        return {"id": self.id, "scenario": self.scenario, "status": self.status, "phase": self.phase,
                "speed": self.speed, "tick_index": self.tick_index, "total": self.total,
                "policy": asdict(self.policy), "policy_id": fingerprint(self.policy), "fault": self.fault,
                "frames": self.frames, "decision": self.last_decision, "score": self.score, "selection": self.selection,
                "trials": self.trials, "events": self.events[-120:], "last_event_id": len(self.events), "error": self.error,
                "tool_calls": self.tool_calls, "injected_excluded": self.injected_excluded, "stepping": bool(self.step_budget)}

    def control(self, action, speed=None, fault=None):
        if self.status in ["completed", "cancelled", "failed"]:
            raise ValueError("This run has ended. Start a new mission.")
        if action == "pause":
            self.status, self.step_budget = "paused", 0
        elif action == "resume":
            self.status, self.step_budget = "running", 0
        elif action == "step":
            if self.status != "paused":
                raise ValueError("Pause before stepping")
            if self.step_budget:
                raise ValueError("A step is already executing")
            self.step_budget = 1
        elif action == "cancel":
            self.status, self.step_budget = "cancelled", 0
        elif action == "speed":
            if speed not in [1, 2, 4, 8]:
                raise ValueError("Speed must be 1, 2, 4 or 8")
            self.speed = speed
        elif action == "inject":
            if self.phase not in ["observe", "prove", "ready"]:
                raise ValueError("Fault injection is available while processing sensor inputs")
            if fault not in FAULTS and fault != "restore":
                raise ValueError("Unknown fault")
            self.fault = None if fault == "restore" else {"kind": fault, "remaining": FAULTS[fault], "anchor": None}
            self.publish(event("intervention", "Input change queued" if self.fault else "Input restored", "Applies to the next sensor input. Injected rows are excluded from benchmark scoring.", fault=self.fault))
        else:
            raise ValueError("Unknown control action")
        self.publish(event("status", "Executing one step" if self.step_budget else self.status.capitalize(), f"Feed {'stepping' if self.step_budget else self.status}; playback speed {self.speed}×.", status=self.status, speed=self.speed, stepping=bool(self.step_budget)))
        self.wake.set()

    async def gate(self):
        while self.status == "paused" and self.step_budget == 0:
            self.wake.clear()
            await self.wake.wait()
        if self.status == "cancelled":
            raise asyncio.CancelledError

    async def emit(self, item):
        await self.gate()
        self.publish(item)
        if self.paced:
            await asyncio.sleep((.32 if item["type"] == "tick" else .12) / self.speed)

    async def finish_unit(self):
        if self.step_budget:
            self.step_budget = 0
            self.publish(event("status", "Step complete", "Paused at the next input boundary.", status="paused", speed=self.speed, stepping=False))

    def modify(self, tick):
        tick = {"t": tick["t"], "sensors": dict(tick["sensors"])}
        fault = self.fault
        if not fault:
            return tick, None
        kind = fault["kind"]
        p = self.policy
        if kind == "pressure_dip":
            tick["sensors"]["pressure_bar"] = p.pressure_cutoff - 15
            tick["sensors"]["temp_C"] = max(tick["sensors"]["temp_C"] or 85, (p.temperature_cutoff or 85) + 3)
            tick["sensors"]["flow_Lps"] = max(tick["sensors"]["flow_Lps"] or 12, (p.flow_cutoff or 12) + 2)
        elif kind == "pressure_offline":
            tick["sensors"]["pressure_bar"] = None
        elif kind == "hydrate_pattern":
            tick["sensors"] = {"pressure_bar": p.pressure_cutoff - 15, "temp_C": (p.temperature_cutoff or 82) - 2, "flow_Lps": (p.flow_cutoff or 10.5) - 1}
        elif kind == "frozen_feed":
            if fault["anchor"] is None:
                fault["anchor"] = dict(self.agent.history[-1]["sensors"] if self.agent.history else tick["sensors"])
            tick["sensors"] = dict(fault["anchor"])
        fault["remaining"] -= 1
        if fault["remaining"] == 0:
            self.fault = None
        return tick, kind

    async def feed(self, frame, phase, policy):
        self.phase, self.policy = phase, policy
        self.agent = SensorAgent(policy)
        self.frames, self.last_decision, self.score = [], None, None
        self.injected_excluded = 0
        self.tick_index, self.total = 0, len(frame)
        self.fault = None if phase == "prove" else self.fault
        scored_rows, scored_predictions, excluded = [], [], 0
        tainted_until = -1
        await self.emit(event("phase", "Prove the frozen policy" if phase == "prove" else "Observe and investigate", "The backend will consume one reading at a time; labels are available only to the outcome evaluator.", phase=phase, total=len(frame), policy=asdict(policy), policy_id=fingerprint(policy)))
        for _, row in frame.iterrows():
            await self.gate()
            tick, injection = self.modify(clean_tick(row))
            self.tick_index += 1
            if injection:
                tainted_until = self.tick_index + 5
            contaminated = self.tick_index <= tainted_until
            point = {**tick, "index": self.tick_index, "injection": injection, "decision": None}
            self.frames.append(point)
            await self.emit(event("tick", "Sensor input arrived", f"Reading {self.tick_index} of {self.total}." + (f" Injected: {injection}." if injection else ""), **point, fault=self.fault, total=self.total))
            generator = self.agent.observe(tick)
            while True:
                await self.gate()
                try:
                    item = next(generator)
                except StopIteration:
                    break
                if item["type"] == "decision":
                    point["decision"] = item["payload"]["decision"]
                    self.last_decision = item["payload"]
                await self.emit(item)
            # Ground truth is read only AFTER the decision; never passed to tools.
            if contaminated:
                excluded += 1
            else:
                scored_rows.append({"timestamp": pd.Timestamp(tick["t"]), **tick["sensors"], "label": int(row.label)})
                scored_predictions.append(self.last_decision["decision"] == "ALERT")
            self.score = metrics(pd.DataFrame(scored_rows), scored_predictions) if scored_rows else None
            if self.score:
                self.score = {**self.score, "injected_excluded": excluded, "processed": self.tick_index}
            point["ground_truth"] = None if contaminated else "Event" if int(row.label) == 1 else "Normal"
            self.injected_excluded = excluded
            await self.emit(event("outcome", "Outcome evaluated", "Injected reading or its five-sample history excluded from benchmark scoring." if contaminated else "The evaluator revealed this historical label after the decision.", t=tick["t"], label=point["ground_truth"], score=self.score, injected_excluded=excluded, tool_calls=self.tool_calls))
            await self.finish_unit()

    async def learn(self):
        self.phase, self.trials, self.fault = "learn", [], None
        await self.emit(event("phase", "Start the improvement cycle", "Fit on days 1–10; score on original days 11–20. Operator injections are excluded. Days 21–30 remain untouched.", phase="learn"))
        incumbent = fit_policy(self.calibration, 10, None)
        incumbent_score = metrics(self.validation, predict(self.validation[["timestamp", *SENSORS]], incumbent))
        champion = None
        best_cost = incumbent_score["cost"]
        await self.emit(event("baseline", "Measure the incumbent", f"Validation cost is {best_cost}: 5 × {incumbent_score['false_negative']} missed bad hours + {incumbent_score['false_positive']} false hours.", metrics=incumbent_score))
        for pressure in [5, 10, 15, 20]:
            for confirmation in [10, 20, 30]:
                for consecutive in [1, 2]:
                    await self.gate()
                    trial_id = len(self.trials) + 1
                    name = f"P{pressure} pressure + P{confirmation} confirmation · {consecutive}h"
                    await self.emit(event("proposal", f"Propose candidate {trial_id}/24", name, trial=trial_id, pressure_percentile=pressure, confirmation_percentile=confirmation, consecutive=consecutive))
                    await self.emit(event("tool_call", "evaluate_candidate", "Fit on calibration sensors and score on the separate historical validation period.", tool="evaluate_candidate", args={"trial": trial_id, "fit_days": "1–10", "score_days": "11–20"}))
                    await self.gate()
                    candidate = fit_policy(self.calibration, pressure, confirmation, consecutive)
                    score = metrics(self.validation, predict(self.validation[["timestamp", *SENSORS]], candidate))
                    key = (score["cost"], score["false_positive"], score["delay_hours"] if score["delay_hours"] is not None else math.inf, trial_id)
                    accepted = score["cost"] < best_cost
                    reason = f"Cost {score['cost']} {'beats' if accepted else 'does not beat'} current best {best_cost}; {score['true_positive']}/{score['bad_hours']} bad hours caught, {score['false_positive']} false hours."
                    trial = {"id": trial_id, "name": name, "policy": asdict(candidate), "metrics": score, "accepted": accepted, "reason": reason}
                    self.trials.append(trial)
                    if champion is None or key < champion[0]:
                        champion = (key, trial)
                    best_cost = min(best_cost, score["cost"])
                    await self.emit(event("candidate_result", "Keep candidate" if accepted else "Reject candidate", reason, trial=trial, best_cost=best_cost))
                    await self.finish_unit()
        await self.gate()
        winner = champion[1]
        promoted = winner["metrics"]["cost"] < incumbent_score["cost"]
        config = winner["policy"] if promoted else asdict(incumbent)
        training = pd.concat([self.calibration, self.validation], ignore_index=True)
        self.policy = fit_policy(training, config["pressure_percentile"], config["confirmation_percentile"], config["consecutive"])
        self.selection = {"promoted": promoted, "trial": winner["id"] if promoted else None, "incumbent_cost": incumbent_score["cost"], "selected_cost": winner["metrics"]["cost"] if promoted else incumbent_score["cost"], "policy_id": fingerprint(self.policy)}
        await self.emit(event("promotion", "Promote and freeze" if promoted else "Retain the incumbent", "Refit chosen percentiles on days 1–20. The next stage will evaluate on all 240 untouched final-test hours.", selection=self.selection, policy=asdict(self.policy), policy_id=fingerprint(self.policy)))
        return self.policy

    async def run(self):
        try:
            await self.gate()
            if self.scenario == "mission":
                await self.feed(self.validation.iloc[152:204].copy(), "observe", self.policy)
                selected = await self.learn()
                await self.feed(self.test, "prove", selected)
            else:
                # These modes use the same bounded historical selection, computed
                # on creation of their live run, never the cached report payload.
                from src.research import select_policy
                chosen = select_policy(self.calibration, self.validation)["selected_config"]
                policy = fit_policy(pd.concat([self.calibration, self.validation]), chosen["pressure_percentile"], chosen["confirmation_percentile"], chosen["consecutive"])
                await self.feed(self.test if self.scenario == "test" else self.validation.iloc[:48], "prove" if self.scenario == "test" else "observe", policy)
            self.phase, self.status = "complete", "completed"
            self.publish(event("complete", "Mission complete", "All requested inputs have been processed. Review decisions, failures and the immutable event journal.", score=self.score, selection=self.selection, status=self.status))
        except asyncio.CancelledError:
            self.status = "cancelled"
            self.publish(event("cancelled", "Run stopped", "No further inputs or candidate tests will be processed.", status="cancelled"))
        except Exception as exc:
            self.status, self.error = "failed", str(exc)
            self.publish(event("failed", "Run failed", self.error, status="failed"))


SESSIONS: dict[str, LiveSession] = {}


def create_session(scenario, speed):
    if scenario not in [c["id"] for c in CATALOG]:
        raise ValueError("Unknown live scenario")
    now = time.monotonic()
    for key, session in list(SESSIONS.items()):
        if session.status in ["completed", "failed", "cancelled"] and now - session.touched > 3600:
            del SESSIONS[key]
    if sum(s.status in ["running", "paused"] for s in SESSIONS.values()) >= 8:
        raise ValueError("Eight sessions are active. Stop an existing run first.")
    session = LiveSession(scenario, speed)
    SESSIONS[session.id] = session
    session.task = asyncio.create_task(session.run())
    return session
