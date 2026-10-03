"""Per-tick watcher — anomaly trigger without LLM.

Owner: Div

Lightweight rule-based trigger that runs on every sensor tick.
If the watcher fires, it invokes the LLM agent. Includes cooldown
logic (don't re-trigger within N minutes of a recent decision) and
recheck scheduling (schedule a follow-up check after WATCH decisions).

TODO:
  - Implement tick evaluation logic (threshold + ML probability)
  - Implement cooldown tracking per well
  - Implement recheck scheduler
  - Wire up to SSE stream in backend
"""

import pandas as pd


def should_trigger(tick: dict, well_id: str) -> bool:
    """Evaluate whether the current tick should trigger the agent.

    Args:
        tick: Dict of current sensor readings.
        well_id: Well identifier.

    Returns:
        True if the agent should be invoked.
    """
    raise NotImplementedError


def record_decision(well_id: str, decision: str, timestamp: pd.Timestamp) -> None:
    """Record an agent decision for cooldown tracking.

    Args:
        well_id: Well identifier.
        decision: Agent decision (ALERT/WATCH/DISMISS).
        timestamp: Timestamp of the decision.
    """
    raise NotImplementedError


def get_next_recheck(well_id: str) -> pd.Timestamp | None:
    """Get the next scheduled recheck time for a well.

    Args:
        well_id: Well identifier.

    Returns:
        Timestamp of next recheck, or None if no recheck scheduled.
    """
    raise NotImplementedError
