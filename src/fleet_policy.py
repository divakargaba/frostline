"""Causal numerical monitoring and concrete operator checks shared by replay/evaluation.

This module receives observed sensors and model outputs only. It does not read
event labels, source classes, operator notes, future rows, or equipment controls.
The fixed heuristics below are engineering review triggers, not diagnoses; their
held-out evaluation must never be used to tune these same thresholds.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.fleet_model import PRESSURES, advance_alarm

POLICY_VERSION = "fleet-monitor-v1"
RANK = {"attention": 0, "unavailable": 1, "watch": 2, "normal": 3}
OTHER_ACTIVATION = .65
OTHER_RECOVERY = .55
PRESSURE_ABSOLUTE_BAR = 2.
PRESSURE_RELATIVE_CHANGE = .02


def _number(value):
    try:
        return float(value) if value is not None and math.isfinite(float(value)) else None
    except (TypeError, ValueError):
        return None


def observation_context(observed_frame):
    """Small evidence view of the supplied prefix, with no labels or future data.

    A precomputed context with this exact schema is also accepted for efficient
    evaluation. ``iter_observation_contexts`` constructs it using this function.
    """
    if isinstance(observed_frame, Mapping):
        return observed_frame
    if observed_frame.empty or not isinstance(observed_frame.index, pd.DatetimeIndex):
        raise ValueError("Observed sensor history needs timestamps and at least one row")
    if not observed_frame.index.is_monotonic_increasing or observed_frame.index.has_duplicates:
        raise ValueError("Observed timestamps must increase")
    names = [*PRESSURES, "ABER-CKP"]
    frame = observed_frame.reindex(columns=names).iloc[-11:]
    values = frame.to_numpy(dtype=float, copy=True)
    # Invalid pressure sentinels cannot create a pressure-loss trigger.
    values[~np.isfinite(values)] = np.nan
    for column, name in enumerate(names):
        if name.startswith("P-"):
            values[np.abs(values[:, column]) >= 1e25, column] = np.nan
        else:
            values[(values[:, column] < 0) | (values[:, column] > 100), column] = np.nan
        flag = f"invalid_{name}"
        if flag in observed_frame:
            values[observed_frame[flag].iloc[-len(frame):].fillna(0).to_numpy() > 0, column] = np.nan
    return _context_from_values(frame.index[-1].isoformat(), values)


def _context_from_values(timestamp, values):
    context = {"timestamp": timestamp, "pressures": {}, "line_difference_bar": None,
               "line_difference_change_bar": None, "line_pair_count": 0,
               "choke_change_percentage_points": None}
    for column, name in enumerate(PRESSURES):
        valid = values[:, column][np.isfinite(values[:, column])]
        current = _number(values[-1, column])
        context["pressures"][name] = {
            "current_bar": current, "start_bar": _number(valid[0]) if len(valid) else None,
            "change_bar": _number(valid[-1] - valid[0]) if len(valid) >= 2 and current is not None else None,
            "samples": int(len(valid)),
        }
    paired = np.isfinite(values[:, 1]) & np.isfinite(values[:, 2])
    context["line_pair_count"] = int(paired.sum())
    if paired[-1]:
        differences = values[paired, 1] - values[paired, 2]
        context["line_difference_bar"] = float(differences[-1])
        if len(differences) >= 2:
            context["line_difference_change_bar"] = float(differences[-1] - differences[0])
    valid_choke = values[:, 3][np.isfinite(values[:, 3])]
    if len(valid_choke) >= 2 and np.isfinite(values[-1, 3]):
        context["choke_change_percentage_points"] = float(valid_choke[-1] - valid_choke[0])
    return context


def iter_observation_contexts(sensor_frame):
    """Linear-time contexts identical to calling observation_context per prefix."""
    names = [*PRESSURES, "ABER-CKP"]
    values = sensor_frame.reindex(columns=names).to_numpy(dtype=float, copy=True)
    values[~np.isfinite(values)] = np.nan
    for column, name in enumerate(names):
        bad = np.abs(values[:, column]) >= 1e25 if name.startswith("P-") else (values[:, column] < 0) | (values[:, column] > 100)
        if f"invalid_{name}" in sensor_frame:
            bad |= sensor_frame[f"invalid_{name}"].fillna(0).to_numpy() > 0
        values[bad, column] = np.nan
    for i, stamp in enumerate(sensor_frame.index):
        yield _context_from_values(stamp.isoformat(), values[max(0, i - 10):i + 1])


def _signals(result, context):
    quality = result.get("quality", {})
    pressures = context["pressures"]
    line = pressures["P-TPT"]
    reference = line["start_bar"]
    threshold = max(PRESSURE_ABSOLUTE_BAR, abs(reference or 0) * PRESSURE_RELATIVE_CHANGE)
    change = line["change_bar"]
    changing = line["samples"] >= 6 and change is not None and abs(change) >= threshold
    dp_change = context["line_difference_change_bar"]
    divergence = context["line_pair_count"] >= 6 and dp_change is not None and abs(dp_change) >= threshold
    # A missing choke-line pair limits interpretation, even if downhole pressure
    # still permits a model score. Optional missing channels alone do not alarm.
    missing_line = [name for name in ["P-TPT", "P-MON-CKP"] if name in quality.get("missing", [])]
    return {"pressure_changing": bool(changing), "pressure_loss": bool(changing and change < 0),
            "pressure_divergence": bool(divergence), "pressure_change_bar": change,
            "pressure_trigger_bar": threshold, "line_difference_change_bar": dp_change,
            "missing_line_channels": missing_line,
            "telemetry_limited": bool(quality.get("blocked") or quality.get("invalid") or missing_line)}


def _check(identifier, label, reason):
    return {"id": identifier, "label": label, "reason": reason, "status": "pending"}


def _assessment(result, context, status, state, policy, signals):
    quality = result.get("quality", {})
    scores = result.get("scores", {})
    invalid = quality.get("invalid", [])
    missing = quality.get("missing", [])
    checks, evidence = [], []
    blocked = bool(quality.get("blocked", quality.get("status") == "unavailable"))
    hydrate = _number(scores.get("hydrate"))
    other = _number(scores.get("lookalike"))
    trigger = policy["activation_threshold"]
    common_alternative = "Pressure changes can reflect operations, another restriction, or telemetry. The cause is not established."
    if blocked:
        scenario = "telemetry_missing"
        summary = "Pressure measurements are unavailable; verify telemetry before assessing the process."
        if status == "attention":
            summary = "Pressure telemetry is unavailable; the earlier process concern remains open."
        flagged = [name for name in PRESSURES if name in missing or name in invalid] or list(PRESSURES)
        evidence = ["No usable pressure channel is available."]
        checks = [_check("verify_pressure_telemetry", "Verify " + ", ".join(flagged), "Compare with an independent available measurement or the responsible instrumentation operator."),
                  _check("restore_verified_readings", "Confirm fresh, usable pressure readings have returned", "An acknowledgment or note cannot establish physical recovery.")]
        alternative, recheck = "A data outage cannot establish that the well is normal or identify a physical fault.", 1
    elif invalid or signals["missing_line_channels"]:
        scenario = "telemetry_invalid" if invalid else "telemetry_partial"
        flagged = invalid or signals["missing_line_channels"]
        summary = "Verify the flagged channels before interpreting the well condition."
        evidence = [("Invalid: " if invalid else "Missing line-pressure channels: ") + ", ".join(flagged) + "."]
        checks = [_check("verify_flagged_channels", "Verify " + ", ".join(flagged), "Check channel validity against an independent measurement; excluded values are not process evidence."),
                  _check("compare_valid_pressure", "Compare the remaining valid pressure trends", "A usable channel can support a limited assessment while the flagged measurement is checked.")]
        alternative, recheck = "Sensor quality limits the assessment; a simultaneous process concern is still possible.", 1
        if status == "attention":
            checks.append(_check("engineering_review", "Share the persistent concern with the responsible engineer", "Telemetry verification does not clear the existing process alarm."))
    elif state.get("alarm", {}).get("active") or (hydrate is not None and hydrate >= .35):
        scenario = "hydrate_pattern"
        persisted = bool(state.get("alarm", {}).get("active"))
        summary = "A persistent hydrate-like model pattern needs review." if persisted else "The hydrate model signal is elevated; verify it against measured trends."
        evidence = [f"Hydrate score {hydrate:.2f}; activation threshold {trigger:.2f}." if hydrate is not None else "An earlier hydrate model alarm remains open.",
                    f"Activation requires {policy['persistence_minutes']} consecutive usable minutes; scores are not probabilities."]
        checks = [_check("compare_pressure_temperature", "Compare pressure and temperature trends across available channels", "Look for agreement between measurements rather than accepting a model score as a diagnosis."),
                  _check("confirm_recent_operations", "Confirm whether a recent choke or operating change explains the pattern", "Ask the operator or consult an existing log; the dataset does not supply an operations log."),
                  _check("engineering_review", "Send the evidence and alternatives for engineering review", "Use the approved site procedure; this prototype cannot prescribe treatment.")]
        alternative, recheck = common_alternative, 5 if status == "attention" else 1
    elif state.get("other_active") or (other is not None and other >= .5):
        scenario = "restriction_pattern"
        summary = "A restriction or scaling model pattern needs independent verification."
        evidence = [f"Restriction/scaling score {other:.2f}; persistent trigger {OTHER_ACTIVATION:.2f}." if other is not None else "An earlier restriction/scaling model alarm remains open."]
        checks = [_check("compare_choke_pressures", "Compare the subsea-tree and upstream-choke pressure trends", "A changing production-line pressure difference is evidence to investigate, not confirmation of a blockage."),
                  _check("confirm_recent_operations", "Confirm recent choke changes with the operator", "Operational changes can explain pressure changes and are not recorded as sensor ground truth."),
                  _check("engineering_review", "Request engineering review if the pattern persists", "Keep hydrate and instrumentation issues as alternatives.")]
        alternative, recheck = common_alternative, 5 if status == "attention" else 1
    elif signals["pressure_divergence"]:
        scenario = "pressure_divergence"
        summary = "The production-line pressure difference is changing."
        evidence = [f"P-TPT minus P-MON-CKP changed by {context['line_difference_change_bar']:.2f} bar across the recent observed window.",
                    f"Current pressure difference: {context['line_difference_bar']:.2f} bar."]
        checks = [_check("compare_choke_pressures", "Verify both line-pressure channels against their recent trends", "P-TPT is at the subsea tree; P-MON-CKP is upstream of the production choke."),
                  _check("confirm_recent_operations", "Confirm whether the choke setting changed recently", "A pressure difference alone cannot distinguish an operating change from a restriction."),
                  _check("engineering_review", "Escalate an unexplained persistent divergence for review", "Share the measured change and the operator observation.")]
        alternative, recheck = common_alternative, 1
    elif signals["pressure_changing"]:
        scenario = "pressure_loss" if signals["pressure_loss"] else "pressure_rise"
        summary = "Tubing pressure is falling; compare independent measurements." if signals["pressure_loss"] else "Tubing pressure is rising; compare independent measurements."
        evidence = [f"P-TPT changed by {signals['pressure_change_bar']:.2f} bar across the recent observed window.",
                    f"The fixed review trigger for this window is {signals['pressure_trigger_bar']:.2f} bar."]
        checks = [_check("compare_valid_pressure", "Compare P-TPT with the other available pressure channels", "Determine whether the change is local to one measurement or supported independently."),
                  _check("confirm_recent_operations", "Confirm recent choke or operating changes with the operator", "A pressure change is not a hydrate diagnosis."),
                  _check("engineering_review", "Request review if the pressure change remains unexplained", "Provide the observed magnitude and time window.")]
        alternative, recheck = common_alternative, 1
    elif status != "normal":
        scenario = "open_concern"
        summary = "An earlier concern remains open while recovery is checked."
        evidence = ["Recovery needs consecutive normal observations; operator acknowledgment does not clear the condition."]
        checks = [_check("review_open_evidence", "Review the latest evidence against the open concern", "Use the incident history and record what was verified."),
                  _check("engineering_review", "Share unresolved findings with the responsible engineer", "The current available evidence does not establish a specific cause.")]
        alternative, recheck = "The earlier concern may be resolving; continue checking fresh observations.", 5
    else:
        scenario = "normal"
        summary = "No persistent concern is present in the available measurements."
        evidence = [f"Hydrate score {hydrate:.2f}, below the {trigger:.2f} activation threshold." if hydrate is not None else "No model score is available.",
                    "No qualifying recent pressure change or pressure divergence."]
        alternative, recheck = "This is a limited sensor assessment, not proof that every process condition is normal.", 30
    if context.get("choke_change_percentage_points") is not None and abs(context["choke_change_percentage_points"]) >= .5 and scenario != "normal":
        evidence.append(f"Measured choke opening changed by {context['choke_change_percentage_points']:.2f} percentage points; intent is unknown.")
    return {"summary": summary, "evidence": evidence, "next_step": checks[0]["label"] if checks else "Continue monitoring.",
            "source": "rules", "uncertainty": alternative, "alternative": alternative, "alternatives": [alternative],
            "tools": [], "scenario": scenario, "category": scenario, "checks": checks, "recheck_minutes": recheck,
            "policy_version": POLICY_VERSION}


def operator_assessment(result, observed_frame, status, state, policy):
    """Refresh the same assessment without advancing alarm or recovery state."""
    context = observation_context(observed_frame)
    return _assessment(result, context, status, state, policy, _signals(result, context))


def advance_monitor(state, result, observed_frame, policy):
    """Advance one well by one observed minute; inputs are never mutated.

    The returned state preserves caller-owned fields. Incident acknowledgments,
    completed checks and notes are deliberately absent from the physical policy.
    LLM escalation may be supplied through ``agent_status``; five fresh normal
    minutes release that extra priority without modifying the numerical alarm.
    """
    state = dict(state or {})
    context = observation_context(observed_frame)
    now = pd.Timestamp(context["timestamp"])
    quality, scores = result.get("quality", {}), result.get("scores", {})
    blocked = bool(quality.get("blocked", quality.get("status") == "unavailable"))
    previous = state.get("status", state.get("base_status", "unavailable"))
    previous_base = state.get("base_status", "unavailable")
    previous_time = state.get("monitor_last_t")
    if previous_time is not None and now <= pd.Timestamp(previous_time):
        raise ValueError("Monitor timestamps must increase")
    if previous_time is not None and now - pd.Timestamp(previous_time) != pd.Timedelta(minutes=1):
        state.update(other_run=0, other_recovery=0, normal_run=0)
    state["monitor_last_t"] = now.isoformat()
    hydrate, other = _number(scores.get("hydrate")), _number(scores.get("lookalike"))
    state["alarm"] = advance_alarm(state.get("alarm", {}), hydrate, not blocked, now, policy)
    state["other_run"] = state.get("other_run", 0) + 1 if not blocked and other is not None and other >= OTHER_ACTIVATION else 0
    state["other_recovery"] = state.get("other_recovery", 0) + 1 if not blocked and other is not None and other < OTHER_RECOVERY else 0
    state.setdefault("other_active", False)
    if state["other_run"] >= 3:
        state["other_active"] = True
    if state["other_recovery"] >= 5:
        state["other_active"] = False
    signals = _signals(result, context)
    if blocked:
        status = "attention" if previous == "attention" else "unavailable"
    elif state["alarm"]["active"] or state["other_active"]:
        status = "attention"
    elif (hydrate or 0) >= .35 or (other or 0) >= .5 or signals["pressure_changing"] or signals["pressure_divergence"] or signals["telemetry_limited"]:
        status = "watch"
    else:
        status = "normal"
    base_status = status
    state["base_status"] = base_status
    state["normal_run"] = state.get("normal_run", 0) + 1 if base_status == "normal" else 0
    state.setdefault("agent_status", "normal")
    if state["normal_run"] >= 5:
        state["agent_status"] = "normal"
    if RANK[state["agent_status"]] < RANK[status]:
        status = state["agent_status"]
    pressure = context["pressures"]["P-TPT"]["current_bar"]
    quality_signature = (blocked, tuple(sorted(quality.get("missing", []))), tuple(sorted(quality.get("invalid", []))))
    score_shift = hydrate is not None and state.get("revision_score") is not None and abs(hydrate - state["revision_score"]) >= .2
    pressure_shift = pressure is not None and state.get("revision_pressure") is not None and abs(pressure - state["revision_pressure"]) >= max(5., abs(state["revision_pressure"]) * .03)
    material = status != previous or base_status != previous_base or quality_signature != state.get("last_quality") or score_shift or pressure_shift
    if material:
        state.update(revision=state.get("revision", 0) + 1, revision_score=hydrate, revision_pressure=pressure, last_quality=quality_signature)
    state.update(status=status, last_score=hydrate)
    assessment = _assessment(result, context, status, state, policy, signals)
    return {"state": state, "status": status, "base_status": base_status, "assessment": assessment,
            "signals": signals, "recheck_minutes": assessment["recheck_minutes"], "material": bool(material)}
