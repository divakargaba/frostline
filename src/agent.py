"""LLM tool-calling agent loop.

Owner: Div

Orchestrates the hydrate-detection reasoning loop:
  1. Receives a trigger from the watcher
  2. Calls tools (max 8 calls) to gather evidence
  3. Produces a structured decision: ALERT / WATCH / DISMISS
  4. Caches decisions by (instance_id, minute_index, model) to avoid redundant calls
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Generator

from backend.schemas import DecisionEvent, EvidenceItem, OnsetEta
from src.llm import LLMResponse, MockLLMClient, ToolCall, get_llm_client
from src.tools import AgentContext, TOOL_SCHEMAS, dispatch_tool

log = logging.getLogger("frostline.agent")

CACHE_DIR = Path("data/cache/agent")
MAX_TOOL_CALLS = 8
TOTAL_TIMEOUT_S = 60

SYSTEM_PROMPT = """You are an offshore production advisor watching for hydrate formation in a subsea oil well. You have access to tools that provide real sensor data, physics calculations, ML predictions, and operational playbooks.

Your job: analyze the evidence and decide one of:
- ALERT: hydrate formation is likely, recommend immediate action
- WATCH(recheck_min=N): suspicious but inconclusive, recheck in N minutes
- DISMISS: not a hydrate event (may be a look-alike: scaling, restriction, flow instability)

RULES:
1. Every number in your brief MUST come from a tool result. Never invent numbers.
2. Cite which tool provided each piece of evidence.
3. You are advisory only — never claim you are taking control actions.
4. If a sensor is missing, say so and lean on other available evidence.
5. Distinguish hydrate from look-alikes: hydrate shows BOTH pressure drop AND temperature drop toward equilibrium. Scaling shows pressure drop but temperature stays normal.
6. Keep the brief to 2-3 sentences, readable aloud by an operator.

After gathering evidence, respond with ONLY a JSON object (no markdown, no explanation outside the JSON):
{
  "decision": "ALERT" | "WATCH" | "DISMISS",
  "recheck_min": null or integer,
  "confidence": 0.0-1.0,
  "diagnosis": "hydrate_production_line" | "hydrate_service_line" | "scaling" | "restriction" | "flow_instability" | "normal",
  "onset_eta": {"p10": minutes_or_null, "p50": minutes_or_null, "p90": minutes_or_null},
  "dose_wt_pct": number_or_null,
  "dose_in_range": boolean_or_null,
  "evidence": [{"tool": "tool_name", "summary": "one line"}],
  "playbook_refs": ["filename.md"],
  "brief": "2-3 sentence operator brief"
}"""


def _cache_key(instance_id: str, minute_index: int, model: str) -> str:
    raw = f"{instance_id}:{minute_index}:{model}"
    return hashlib.md5(raw.encode()).hexdigest()


def _load_cache(instance_id: str, minute_index: int, model: str) -> list[dict] | None:
    key = _cache_key(instance_id, minute_index, model)
    path = CACHE_DIR / f"{key}.jsonl"
    if not path.exists():
        return None
    events = []
    for line in path.read_text().splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def _save_cache(instance_id: str, minute_index: int, model: str, events: list[dict]):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(instance_id, minute_index, model)
    path = CACHE_DIR / f"{key}.jsonl"
    with open(path, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


# --- Legacy API for existing tests ---

def get_cached_decision(well_id: str, minute: str) -> dict | None:
    path = CACHE_DIR / "legacy"
    if not path.exists():
        return None
    key_path = path / f"{well_id}_{minute}.json"
    if key_path.exists():
        return json.loads(key_path.read_text())
    return None


def cache_decision(well_id: str, minute: str, decision: dict) -> None:
    path = CACHE_DIR / "legacy"
    path.mkdir(parents=True, exist_ok=True)
    key_path = path / f"{well_id}_{minute}.json"
    key_path.write_text(json.dumps(decision))


def _build_rule_fallback(ctx: AgentContext, tool_results: dict[str, dict]) -> dict:
    """Rule-based fallback decision built only from tool results."""
    classify = tool_results.get("classify_event", {})
    margin = tool_results.get("hydrate_margin", {})
    forecast = tool_results.get("forecast_onset", {})
    dose = tool_results.get("methanol_dose", {})

    p_h = classify.get("p_hydrate", 0)
    p_l = classify.get("p_lookalike", 0)
    margin_c = margin.get("margin_C")

    evidence = []
    brief_parts = []

    if classify.get("available"):
        evidence.append({"tool": "classify_event", "summary": f"p_hydrate={p_h}"})
        brief_parts.append(f"Classification: {p_h*100:.0f}% hydrate probability.")

    if margin.get("available"):
        evidence.append({"tool": "hydrate_margin", "summary": f"margin={margin_c}C ({margin.get('status', 'unknown')})"})
        brief_parts.append(f"Subcooling margin: {margin_c}C.")

    # Decision logic
    if p_h >= 0.6 or (margin_c is not None and margin_c < 0):
        decision = "ALERT"
        confidence = max(p_h, 0.7)
        diagnosis = "hydrate_production_line"
    elif p_h >= 0.3 or (margin_c is not None and margin_c < 3):
        decision = "WATCH"
        confidence = 0.5
        diagnosis = "hydrate_production_line"
    elif p_l >= 0.4:
        decision = "DISMISS"
        confidence = 0.6
        diagnosis = "scaling" if p_l > p_h else "normal"
    else:
        decision = "DISMISS"
        confidence = 0.7
        diagnosis = "normal"

    onset = forecast if forecast.get("available") else {}
    dose_result = dose if dose.get("available") else {}

    if forecast.get("available"):
        evidence.append({"tool": "forecast_onset", "summary": f"p50={onset.get('p50')} min"})

    return {
        "decision": decision,
        "recheck_min": 15 if decision == "WATCH" else None,
        "confidence": round(confidence, 2),
        "diagnosis": diagnosis,
        "onset_eta": {"p10": onset.get("p10"), "p50": onset.get("p50"), "p90": onset.get("p90")},
        "dose_wt_pct": dose_result.get("dose_wt_pct"),
        "dose_in_range": dose_result.get("in_range"),
        "evidence": evidence,
        "playbook_refs": [],
        "brief": " ".join(brief_parts) if brief_parts else "Insufficient data for assessment.",
        "source": "rule_fallback",
    }


def _extract_numbers(text: str) -> set[float]:
    """Extract all numbers from a string."""
    nums = set()
    for m in re.finditer(r'-?\d+\.?\d*', text):
        try:
            nums.add(float(m.group()))
        except ValueError:
            pass
    return nums


def _check_traceability(decision: dict, tool_results: dict[str, dict]) -> bool:
    """Check that every number in brief/evidence comes from a tool result."""
    # Collect all numbers from tool results
    tool_nums = set()
    for result in tool_results.values():
        for v in result.values():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                tool_nums.add(float(v))
                # Also add common roundings
                tool_nums.add(round(float(v), 1))
                tool_nums.add(round(float(v), 0))
            elif isinstance(v, dict):
                for vv in v.values():
                    if isinstance(vv, (int, float)) and not isinstance(vv, bool):
                        tool_nums.add(float(vv))
                        tool_nums.add(round(float(vv), 1))

    # Check numbers in brief
    brief_nums = _extract_numbers(decision.get("brief", ""))
    # Allow common constants (0, 1, 2, 3, etc.) and percentages
    brief_nums -= {0.0, 1.0, 2.0, 3.0, 100.0}

    untraceable = brief_nums - tool_nums
    if untraceable:
        log.warning("Untraceable numbers in brief: %s", untraceable)
        return False
    return True


def run_agent(ctx: AgentContext, trigger: dict) -> Generator[dict, None, None]:
    """Run the agent loop. Yields tool_call, tool_result, and decision events.

    Args:
        ctx: AgentContext with instance data up to current minute
        trigger: The watch_trigger that started this investigation
    """
    client = get_llm_client()
    model_name = getattr(client, "model", "mock")

    # Check cache
    cached = _load_cache(ctx.instance_id, ctx.minute_index, model_name)
    if cached:
        log.info("Cache hit for %s minute %d", ctx.instance_id, ctx.minute_index)
        yield from cached
        return

    t = trigger.get("t", "")
    events: list[dict] = []
    tool_results: dict[str, dict] = {}
    total_tokens = 0
    total_cost = 0.0

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"A watcher trigger fired for well {ctx.well_id} at {t}. "
            f"Reason: {trigger.get('reason', 'unknown')}. "
            f"Score: {trigger.get('score', 0)}. "
            f"Investigate using the available tools and provide your decision."
        )},
    ]

    start = time.time()
    n_calls = 0

    for attempt in range(MAX_TOOL_CALLS + 2):  # Extra iterations for final answer
        if time.time() - start > TOTAL_TIMEOUT_S:
            log.warning("Agent timeout after %.1fs", time.time() - start)
            break

        try:
            resp = client.chat(messages, tools=TOOL_SCHEMAS, temperature=0, max_tokens=2048)
        except Exception as e:
            log.error("LLM call failed: %s", e)
            # Try fallback models
            resp = _try_fallback_models(messages, TOOL_SCHEMAS)
            if resp is None:
                break

        total_tokens += resp.usage.get("total_tokens", 0)
        total_cost += resp.cost_estimate

        if not resp.tool_calls:
            # Final answer — try to parse as decision JSON
            decision = _parse_decision(resp.content, ctx, tool_results, messages, client)
            if decision:
                decision["_tokens"] = total_tokens
                decision["_cost"] = round(total_cost, 6)
                evt = {"type": "decision", "data": decision}
                events.append(evt)
                yield evt
                _save_cache(ctx.instance_id, ctx.minute_index, model_name, events)
                return
            break

        # Process tool calls
        for tc in resp.tool_calls:
            if n_calls >= MAX_TOOL_CALLS:
                log.warning("Max tool calls reached (%d)", MAX_TOOL_CALLS)
                break
            n_calls += 1

            # Emit tool_call event
            tc_evt = {
                "type": "tool_call",
                "data": {"t": t, "call_id": tc.id, "tool": tc.name, "args": tc.arguments},
            }
            events.append(tc_evt)
            yield tc_evt

            # Execute tool
            result = dispatch_tool(tc.name, ctx, tc.arguments)
            tool_results[tc.name] = result

            # Emit tool_result event
            tr_evt = {
                "type": "tool_result",
                "data": {"t": t, "call_id": tc.id, "tool": tc.name, "result": result},
            }
            events.append(tr_evt)
            yield tr_evt

            # Add to messages for next LLM call
            messages.append({
                "role": "assistant",
                "content": resp.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                ],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })

        if n_calls >= MAX_TOOL_CALLS:
            # Force a final answer
            messages.append({
                "role": "user",
                "content": "You have used all available tool calls. Please provide your final decision now as a JSON object.",
            })

    # If we got here without a decision, use rule fallback
    log.warning("Agent loop ended without LLM decision, using rule fallback")
    fallback = _build_rule_fallback(ctx, tool_results)
    fallback["_tokens"] = total_tokens
    fallback["_cost"] = round(total_cost, 6)
    evt = {"type": "decision", "data": fallback}
    events.append(evt)
    yield evt
    _save_cache(ctx.instance_id, ctx.minute_index, model_name, events)


def _parse_decision(content: str | None, ctx: AgentContext,
                    tool_results: dict, messages: list, client) -> dict | None:
    """Parse LLM response as decision JSON. Retry once on failure."""
    if not content:
        return None

    for attempt in range(2):
        try:
            # Strip markdown code fences if present
            text = content.strip()
            if text.startswith("```"):
                text = re.sub(r'^```\w*\n?', '', text)
                text = re.sub(r'\n?```$', '', text)
            d = json.loads(text)

            # Validate with Pydantic
            dec = DecisionEvent(
                t=messages[1]["content"].split(" at ")[1].split(".")[0] if " at " in messages[1]["content"] else "",
                decision=d["decision"],
                recheck_min=d.get("recheck_min"),
                confidence=d.get("confidence", 0.5),
                diagnosis=d.get("diagnosis", "unknown"),
                onset_eta=OnsetEta(**d.get("onset_eta", {})),
                dose_wt_pct=d.get("dose_wt_pct"),
                dose_in_range=d.get("dose_in_range"),
                evidence=[EvidenceItem(**e) for e in d.get("evidence", [])],
                playbook_refs=d.get("playbook_refs", []),
                brief=d.get("brief", ""),
            )

            result = d.copy()

            # Traceability check
            if not _check_traceability(result, tool_results) and attempt == 0:
                messages.append({"role": "user", "content": (
                    "Your brief contains numbers that don't match any tool result. "
                    "Please revise: every number in the brief must come directly from a tool result."
                )})
                try:
                    resp = client.chat(messages, tools=TOOL_SCHEMAS, temperature=0, max_tokens=2048)
                    content = resp.content
                    continue
                except Exception:
                    pass

            return result

        except (json.JSONDecodeError, KeyError, Exception) as e:
            if attempt == 0:
                log.warning("Decision parse failed (attempt %d): %s", attempt + 1, e)
                messages.append({"role": "user", "content": (
                    f"Your response was not valid JSON or was missing required fields. Error: {e}. "
                    "Please respond with ONLY the JSON decision object, no other text."
                )})
                try:
                    resp = client.chat(messages, tools=TOOL_SCHEMAS, temperature=0, max_tokens=2048)
                    content = resp.content
                except Exception:
                    return None
            else:
                log.error("Decision parse failed after retry: %s", e)
                return None

    return None


def _try_fallback_models(messages: list, tools: list) -> LLMResponse | None:
    """Try fallback models from OPENROUTER_FALLBACK_MODELS env var."""
    import os
    from src.llm import OpenRouterClient
    from backend.config import OPENROUTER_API_KEY

    fallbacks = os.getenv("OPENROUTER_FALLBACK_MODELS", "")
    if not fallbacks or not OPENROUTER_API_KEY:
        return None

    for model in fallbacks.split(","):
        model = model.strip()
        if not model:
            continue
        try:
            log.info("Trying fallback model: %s", model)
            client = OpenRouterClient(OPENROUTER_API_KEY, model)
            return client.chat(messages, tools=tools, temperature=0, max_tokens=2048)
        except Exception as e:
            log.warning("Fallback model %s failed: %s", model, e)

    return None


# ---------------------------------------------------------------------------
# Mock LLM scenarios for development
# ---------------------------------------------------------------------------

def _mock_hydrate_scenario() -> list[LLMResponse]:
    """Scripted responses for hydrate ALERT path."""
    return [
        # Call 1: model calls get_window
        LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="c1", name="get_window", arguments={"minutes": 60})],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            model="mock",
        ),
        # Call 2: model calls classify_event
        LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="c2", name="classify_event", arguments={})],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            model="mock",
        ),
        # Call 3: model calls hydrate_margin
        LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="c3", name="hydrate_margin", arguments={})],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            model="mock",
        ),
        # Call 4: model calls forecast_onset
        LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="c4", name="forecast_onset", arguments={})],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            model="mock",
        ),
        # Call 5: model calls methanol_dose
        LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="c5", name="methanol_dose", arguments={"target_shift_C": 5.0})],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            model="mock",
        ),
    ]


def _mock_scaling_scenario() -> list[LLMResponse]:
    """Scripted responses for scaling DISMISS path."""
    return [
        LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="c1", name="get_window", arguments={"minutes": 60})],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            model="mock",
        ),
        LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="c2", name="classify_event", arguments={})],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            model="mock",
        ),
        LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="c3", name="hydrate_margin", arguments={})],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            model="mock",
        ),
    ]


def get_mock_client_for_scenario(scenario: str = "hydrate") -> MockLLMClient:
    """Get a MockLLMClient pre-loaded with a scenario's tool calls.

    The final answer (decision JSON) is appended as the last planned response.
    The decision is built from whatever the tools actually return, so it adapts
    to the data.
    """
    if scenario == "hydrate":
        planned = _mock_hydrate_scenario()
    else:
        planned = _mock_scaling_scenario()
    return MockLLMClient(planned=planned)
