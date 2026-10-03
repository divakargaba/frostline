"""Agent tool definitions.

Owner: Div

Defines the tools available to the LLM agent, each with a JSON schema
for structured invocation. Tools wrap the underlying modules (load, model,
physics, forecast, rag) into a clean interface the agent can call.

Tools:
  - get_window: Fetch the latest N minutes of sensor data for a well
  - hydrate_margin: Calculate subcooling margin at current conditions
  - classify_event: Run ML classifier on a data window
  - forecast_onset: Predict time to established hydrate phase
  - methanol_dose: Calculate required methanol dose for a given subcooling
  - well_history: Retrieve past decisions for a well
  - search_playbook: BM25 search over operations playbook

TODO:
  - Implement each tool function
  - Define JSON schemas for each tool
  - Register tools in a TOOL_REGISTRY dict
"""

import pandas as pd

# JSON schemas for each tool, used by the LLM agent.
TOOL_SCHEMAS: list[dict] = []  # TODO: populate


def get_window(well_id: str, minutes: int = 60) -> pd.DataFrame:
    """Fetch the latest N minutes of sensor data for a well.

    Args:
        well_id: Well identifier.
        minutes: Number of minutes to look back.

    Returns:
        DataFrame with sensor readings for the requested window.
    """
    raise NotImplementedError


def hydrate_margin(p_bar: float, t_c: float, sg: float = 0.65) -> dict:
    """Calculate subcooling margin at the given conditions.

    Args:
        p_bar: Current pressure in bar.
        t_c: Current temperature in Celsius.
        sg: Gas specific gravity.

    Returns:
        Dict with margin_c, hydrate_temp_c, status (safe/warning/danger).
    """
    raise NotImplementedError


def classify_event(well_id: str, minutes: int = 60) -> dict:
    """Run the ML event classifier on the latest data window.

    Args:
        well_id: Well identifier.
        minutes: Window size in minutes.

    Returns:
        Dict with predicted_class, probability, phase.
    """
    raise NotImplementedError


def forecast_onset(well_id: str, minutes: int = 60) -> dict:
    """Forecast time to established hydrate phase.

    Args:
        well_id: Well identifier.
        minutes: Window size for feature computation.

    Returns:
        Dict with q10, q50, q90, physics_estimate (all in minutes).
    """
    raise NotImplementedError


def methanol_dose(delta_t_c: float) -> dict:
    """Calculate required methanol dose for a given subcooling.

    Args:
        delta_t_c: Required temperature depression in Celsius.

    Returns:
        Dict with dose_weight_pct, inhibitor.
    """
    raise NotImplementedError


def well_history(well_id: str, last_n: int = 10) -> list[dict]:
    """Retrieve past agent decisions for a well.

    Args:
        well_id: Well identifier.
        last_n: Number of most recent decisions to return.

    Returns:
        List of decision dicts with timestamp, decision, rationale.
    """
    raise NotImplementedError


def search_playbook(query: str, top_k: int = 3) -> list[dict]:
    """BM25 search over the operations playbook.

    Args:
        query: Free-text search query.
        top_k: Number of results to return.

    Returns:
        List of dicts with doc_id, title, snippet, score.
    """
    raise NotImplementedError
