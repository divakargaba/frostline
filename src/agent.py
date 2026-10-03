"""LLM tool-calling agent loop.

Owner: Div

Orchestrates the hydrate-detection reasoning loop:
  1. Receives a trigger from the watcher
  2. Calls tools (max 8 calls) to gather evidence
  3. Produces a structured decision: ALERT / WATCH / DISMISS
  4. Caches decisions by (well_id, minute) to avoid redundant calls

TODO:
  - Implement tool-calling loop with Anthropic / OpenAI client
  - Implement structured decision output parsing
  - Implement decision cache (well_id, minute) -> decision
  - Implement max-call guard (8 tool calls)
  - Wire up tool registry from src/tools.py
"""


def run_agent(well_id: str, trigger_context: dict) -> dict:
    """Run the agent loop for a triggered well.

    Args:
        well_id: Well identifier.
        trigger_context: Dict with trigger reason, current tick data, etc.

    Returns:
        Dict with decision (ALERT/WATCH/DISMISS), rationale, tool_calls log.
    """
    raise NotImplementedError


def get_cached_decision(well_id: str, minute: str) -> dict | None:
    """Look up a cached decision for a (well, minute) pair.

    Args:
        well_id: Well identifier.
        minute: Minute-resolution timestamp string.

    Returns:
        Cached decision dict, or None if not found.
    """
    raise NotImplementedError


def cache_decision(well_id: str, minute: str, decision: dict) -> None:
    """Cache an agent decision.

    Args:
        well_id: Well identifier.
        minute: Minute-resolution timestamp string.
        decision: Decision dict to cache.
    """
    raise NotImplementedError
