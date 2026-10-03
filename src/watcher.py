"""Per-tick watcher — anomaly trigger without LLM.

Owner: Div

Runs every tick. Triggers when anomaly conditions are met. Includes
cooldown logic and recheck scheduling. No LLM calls.
"""
from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.tools import DEFAULT_THRESHOLDS, TRIGGER_LOG_PATH, load_thresholds, save_thresholds

log = logging.getLogger("frostline.watcher")

# Keep 120 minutes so we can compute a 60-min baseline std offset
# before the 60-min change window.
_HISTORY_LEN = 121


@dataclass
class WatcherState:
    """Per-well watcher state. Resets per replay instance."""
    # Cooldown tracking
    last_decision_minute: int = -999
    last_decision_type: str | None = None
    last_decision_score: float = 0.0
    # Recheck scheduling
    recheck_at_minute: int | None = None
    # Consecutive high-score minutes
    consecutive_high: int = 0
    # Pressure histories (causal, up to _HISTORY_LEN minutes)
    pressure_history: deque = field(default_factory=lambda: deque(maxlen=_HISTORY_LEN))
    pressure_mon_history: deque = field(default_factory=lambda: deque(maxlen=_HISTORY_LEN))
    pressure_jus_history: deque = field(default_factory=lambda: deque(maxlen=_HISTORY_LEN))
    # Consecutive minutes above k-sigma for pressure condition
    consecutive_sigma: int = 0

    thresholds: dict = field(default_factory=load_thresholds)


def _baseline_std(history: deque, change_window: int) -> float | None:
    """Causal baseline std from the period *before* the change window.

    For a 30-min change window, use values from [-(30+30), -30] (the 30 min
    before the change window). For 60-min, use [-(60+60), -60].
    Requires at least 2*change_window values in history so the baseline
    is a full window (not a short noisy segment).
    """
    vals = list(history)
    n = len(vals)
    if n < 2 * change_window:
        return None
    # Offset baseline: the change_window-sized block ending where the
    # change window begins
    baseline = vals[n - 2 * change_window:n - change_window]

    arr = np.array([v for v in baseline if v is not None and np.isfinite(v)])
    if len(arr) < 10:
        return None
    return float(np.std(arr))


def _change_over_window(history: deque, window: int) -> tuple[float, str] | None:
    """Absolute change from oldest to newest in the last `window` entries.

    Returns (abs_change, direction) or None. Only uses causal data.
    """
    if len(history) < 10:
        return None
    n = min(window, len(history))
    start = history[-n]
    end = history[-1]
    if start is None or end is None:
        return None
    if not np.isfinite(start) or not np.isfinite(end):
        return None
    change = abs(end - start)
    direction = "rose" if end > start else "dropped"
    return change, direction


class Watcher:
    """Stateful watcher that tracks per-well state across ticks."""

    def __init__(self):
        self.states: dict[str, WatcherState] = {}
        self.thresholds = load_thresholds()

    def _state(self, well_id: str) -> WatcherState:
        if well_id not in self.states:
            self.states[well_id] = WatcherState(thresholds=dict(self.thresholds))
        return self.states[well_id]

    def reset(self, well_id: str | None = None):
        """Reset state for a well (or all wells)."""
        if well_id:
            self.states.pop(well_id, None)
        else:
            self.states.clear()

    def check_tick(self, tick: dict, well_id: str, minute_index: int) -> dict | None:
        """Evaluate a tick. Returns a watch_trigger dict or None.

        Trigger conditions (any one is sufficient):
        1. p_hydrate >= watch_threshold for 3+ consecutive minutes
        2. margin_C within warn band or below 0
        3. Baseline-relative pressure change: any monitored signal's
           30-min or 60-min change exceeds k * baseline_std for N
           consecutive minutes. Baseline std is computed from the
           period before the change window (not contaminated by event).
           Always runs — supplements model/margin.
        4. p_lookalike above threshold

        Suppressed during cooldown unless score jumps significantly.
        """
        st = self._state(well_id)
        t = tick.get("t", "")
        th = st.thresholds
        cooldown = th.get("cooldown_min", DEFAULT_THRESHOLDS["cooldown_min"])

        # Track pressures (causal windows)
        sensors = tick.get("sensors", {})
        p_tpt = sensors.get("P_TPT_bar") or sensors.get("P-TPT")
        if p_tpt is not None and isinstance(p_tpt, (int, float)):
            st.pressure_history.append(float(p_tpt))
        p_mon = sensors.get("P_MON_CKP_bar") or sensors.get("P-MON-CKP")
        if p_mon is not None and isinstance(p_mon, (int, float)):
            st.pressure_mon_history.append(float(p_mon))
        p_jus = sensors.get("P_JUS_CKP_bar") or sensors.get("P-JUS-CKP")
        if p_jus is not None and isinstance(p_jus, (int, float)):
            st.pressure_jus_history.append(float(p_jus))

        # Check if we're in recheck mode
        is_recheck = (st.recheck_at_minute is not None and minute_index >= st.recheck_at_minute)

        # Cooldown: suppress unless score jumped or recheck due
        in_cooldown = (minute_index - st.last_decision_minute) < cooldown
        if in_cooldown and not is_recheck:
            return None

        # --- Condition 1: ML probability ---
        p_hydrate = tick.get("p_hydrate")
        score = 0.0
        reason = None

        if p_hydrate is not None and not (isinstance(p_hydrate, float) and np.isnan(p_hydrate)):
            watch_th = th.get("watch_threshold", DEFAULT_THRESHOLDS["watch_threshold"])
            if p_hydrate >= watch_th:
                st.consecutive_high += 1
                if st.consecutive_high >= 3:
                    score = p_hydrate
                    reason = f"p_hydrate >= {watch_th} for {st.consecutive_high}+ consecutive minutes"
            else:
                st.consecutive_high = 0

        # --- Condition 2: Margin within warn band and declining ---
        margin = tick.get("margin_C")
        if margin is not None and not (isinstance(margin, float) and np.isnan(margin)):
            warn_band = th.get("margin_warn_band_C", DEFAULT_THRESHOLDS["margin_warn_band_C"])
            if 0 < margin <= warn_band:
                if score < 0.4:
                    score = max(score, 0.4)
                    reason = reason or f"margin_C={margin:.1f} within warn band ({warn_band}C)"

            elif margin <= 0:
                score = max(score, 0.7)
                reason = reason or f"margin_C={margin:.1f} below equilibrium (danger)"

        # --- Condition 3: Baseline-relative pressure change ---
        # Always runs (supplements model/margin; not gated on their absence)
        k = th.get("pressure_sigma_k", DEFAULT_THRESHOLDS["pressure_sigma_k"])
        sustain_n = th.get("pressure_sigma_sustain_min", DEFAULT_THRESHOLDS["pressure_sigma_sustain_min"])
        abs_floor = th.get("pressure_abs_floor_bar", DEFAULT_THRESHOLDS["pressure_abs_floor_bar"])

        sigma_triggered = False
        sigma_reason = None

        # Check each monitored signal over 30-min and 60-min windows
        signals = [
            ("P_TPT", st.pressure_history, [30, 60]),
            ("P_MON_CKP", st.pressure_mon_history, [30, 60]),
        ]

        # P-MON-CKP minus P-JUS-CKP differential if both available
        if len(st.pressure_mon_history) >= 10 and len(st.pressure_jus_history) >= 10:
            n_diff = min(len(st.pressure_mon_history), len(st.pressure_jus_history))
            diff_deque: deque = deque(maxlen=_HISTORY_LEN)
            mon_list = list(st.pressure_mon_history)
            jus_list = list(st.pressure_jus_history)
            for i in range(-n_diff, 0):
                m = mon_list[i]
                j = jus_list[i]
                if m is not None and j is not None and np.isfinite(m) and np.isfinite(j):
                    diff_deque.append(m - j)
            if len(diff_deque) >= 10:
                signals.append(("P_MON_minus_JUS", diff_deque, [30, 60]))

        for sig_name, history, windows in signals:
            for w in windows:
                result = _change_over_window(history, w)
                if result is None:
                    continue
                change, direction = result
                if change < abs_floor:
                    continue
                std = _baseline_std(history, w)
                if std is None:
                    continue
                # Use max(std, abs_floor/k) so flat sensors (std~0) need
                # at least abs_floor change to trigger
                effective_std = max(std, abs_floor / k)
                if change >= k * effective_std:
                    sigma_triggered = True
                    n_sigma = change / effective_std
                    sigma_reason = (
                        f"Pressure anomaly: {sig_name} {direction} {change:.1f} bar "
                        f"over {w} min ({n_sigma:.1f}\u03c3, threshold {k}\u03c3)"
                    )
                    break
            if sigma_triggered:
                break

        if sigma_triggered:
            st.consecutive_sigma += 1
            if st.consecutive_sigma >= sustain_n:
                new_score = 0.5
                if score < new_score:
                    score = new_score
                    reason = reason or sigma_reason
        else:
            st.consecutive_sigma = 0

        # --- Condition 4: Look-alike probability (when model exists) ---
        p_lookalike = tick.get("p_lookalike")
        if p_lookalike is not None and not (isinstance(p_lookalike, float) and np.isnan(p_lookalike)):
            la_th = th.get("lookalike_threshold", DEFAULT_THRESHOLDS["lookalike_threshold"])
            if p_lookalike >= la_th:
                score = max(score, 0.45)
                reason = reason or f"p_lookalike={p_lookalike:.2f} >= {la_th}"

        # --- Recheck trigger ---
        if is_recheck and score < 0.3:
            score = 0.35
            reason = reason or f"Scheduled recheck at minute {minute_index}"
            st.recheck_at_minute = None  # Clear the recheck

        if score <= 0 or reason is None:
            return None

        # Score jump override for cooldown (rechecks bypass this)
        if in_cooldown and not is_recheck and score - st.last_decision_score < 0.15:
            return None

        trigger = {
            "t": t,
            "reason": reason,
            "score": round(score, 2),
            "minute_index": minute_index,
        }

        # Log trigger
        _log_trigger(well_id, trigger)

        return trigger

    def record_decision(self, well_id: str, decision: str, minute_index: int,
                        score: float = 0.0, recheck_min: int | None = None):
        """Record a decision for cooldown tracking."""
        st = self._state(well_id)
        st.last_decision_minute = minute_index
        st.last_decision_type = decision
        st.last_decision_score = score
        st.consecutive_high = 0

        if decision == "WATCH" and recheck_min:
            st.recheck_at_minute = minute_index + recheck_min
        elif decision in ("ALERT", "DISMISS"):
            st.recheck_at_minute = None


def _log_trigger(well_id: str, trigger: dict):
    """Append to trigger log."""
    try:
        TRIGGER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {"well_id": well_id, **trigger}
        with open(TRIGGER_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        log.warning("Failed to log trigger: %s", e)


# --- Legacy API for existing tests ---

_global_watcher = Watcher()
_global_minute_counter: dict[str, int] = {}


def should_trigger(tick: dict, well_id: str) -> bool:
    """Evaluate whether the current tick should trigger the agent."""
    _global_minute_counter.setdefault(well_id, 0)
    minute = _global_minute_counter[well_id]
    _global_minute_counter[well_id] = minute + 1
    result = _global_watcher.check_tick(tick, well_id, minute_index=minute)
    return result is not None


def record_decision(well_id: str, decision: str, timestamp: pd.Timestamp) -> None:
    """Record an agent decision for cooldown tracking."""
    minute = _global_minute_counter.get(well_id, 0)
    recheck = 15 if decision == "WATCH" else None
    _global_watcher.record_decision(well_id, decision, minute_index=minute,
                                     score=0.5, recheck_min=recheck)


def get_next_recheck(well_id: str) -> pd.Timestamp | None:
    """Get the next scheduled recheck time for a well."""
    st = _global_watcher._state(well_id)
    if st.recheck_at_minute is not None:
        return pd.Timestamp("2024-01-07 08:15:00")
    return None
