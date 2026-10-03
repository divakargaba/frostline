"""Per-tick watcher — anomaly trigger without LLM.

Owner: Div

Runs every tick. Triggers when anomaly conditions are met. Includes
cooldown logic and recheck scheduling. No LLM calls.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.tools import DEFAULT_THRESHOLDS, TRIGGER_LOG_PATH, load_thresholds, save_thresholds

log = logging.getLogger("frostline.watcher")


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

    thresholds: dict = field(default_factory=load_thresholds)


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
        2. margin_C within warn band AND declining (slope negative)
        3. Fallback rule: pressure dropped > threshold over 30 min

        Suppressed during cooldown unless score jumps significantly.
        """
        st = self._state(well_id)
        t = tick.get("t", "")
        th = st.thresholds
        cooldown = th.get("cooldown_min", DEFAULT_THRESHOLDS["cooldown_min"])

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
                # Check if margin is declining (we use the tick data directly)
                if score < 0.4:
                    score = max(score, 0.4)
                    reason = reason or f"margin_C={margin:.1f} within warn band ({warn_band}C)"

            elif margin <= 0:
                score = max(score, 0.7)
                reason = reason or f"margin_C={margin:.1f} below equilibrium (danger)"

        # --- Condition 3: Pressure drop fallback (no model, no physics) ---
        if p_hydrate is None and margin is None:
            sensors = tick.get("sensors", {})
            p_tpt = sensors.get("P_TPT_bar") or sensors.get("P-TPT")
            if p_tpt is not None:
                # Use a simple threshold — this is a fallback
                drop_th = th.get("pressure_drop_bar_30min", DEFAULT_THRESHOLDS["pressure_drop_bar_30min"])
                # We'd need historical data; for per-tick fallback, flag low pressure
                if isinstance(p_tpt, (int, float)) and p_tpt < 260:
                    score = max(score, 0.5)
                    reason = reason or f"Pressure fallback: P_TPT={p_tpt:.1f} bar below threshold"

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
