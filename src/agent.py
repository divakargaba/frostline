"""LLM tool-calling agent loop — adaptive tool choice design.

Owner: Div

Adaptive agent (LLM chooses which tools to use):
  Round 0: Code pre-runs get_window + classify_event (always needed).
  Round 1: LLM sees pre-run results + trigger info, requests additional
           tools it needs in ONE parallel batch (requested_by: "agent").
  Round 2: LLM receives tool results, returns structured decision JSON.
  Optional Round 3: only if LLM explicitly flags a concern, within budget.

Budget: LLM_MAX_REQUESTS_PER_RUN (default 3). Retries count.
Cache: by (instance_id, minute_index, model, PROMPT_VERSION).
Fallback: rule_fallback when pool exhausted or budget exceeded.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Generator

from backend.schemas import DecisionEvent, EvidenceItem, OnsetEta
from src.llm import LLMResponse, MockLLMClient, ToolCall, get_llm_client
from src.tools import AgentContext, TOOL_SCHEMAS, dispatch_tool

log = logging.getLogger("frostline.agent")

CACHE_DIR = Path("data/cache/agent")
PROMPT_VERSION = "v2"  # Bump when system prompt changes materially
MAX_TOOL_CALLS = 8
LLM_MAX_REQUESTS_PER_RUN = int(os.getenv("LLM_MAX_REQUESTS_PER_RUN", "3"))

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
7. Before making ALERT or DISMISS, request search_playbook to cite a relevant procedure.
8. Request ALL additional tools you need in a SINGLE response using parallel tool calls.

You have already been given results from get_window and classify_event (pre-run by the system). Based on this initial evidence, decide which additional tools you need. Available tools:
- hydrate_margin: Calculate subcooling margin (use when pressure/temperature suggest hydrate risk)
- forecast_onset: Predict time to established hydrate (use when hydrate is likely)
- methanol_dose: Calculate inhibitor dose (use when preparing ALERT recommendation)
- well_history: Get past triggers and thresholds (use for context on repeat events)
- search_playbook: Search operations playbook (use to cite procedures for ALERT or DISMISS)

Request the tools you need NOW in one batch. After receiving results, provide your decision.

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
  "playbook_refs": ["doc_id"],
  "brief": "2-3 sentence operator brief"
}"""


def _cache_key(instance_id: str, minute_index: int, model: str) -> str:
    raw = f"{instance_id}:{minute_index}:{model}:{PROMPT_VERSION}"
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
    playbook = tool_results.get("search_playbook", {})

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

    # Playbook refs from search results
    playbook_refs = []
    if playbook.get("available"):
        for r in playbook.get("results", []):
            if r.get("id"):
                playbook_refs.append(r["id"])

    # Check for scaling pattern: pressure rise in get_window with stable temperature
    window = tool_results.get("get_window", {})
    is_pressure_rise = False
    if window.get("available"):
        slopes = window.get("slopes", {})
        # P-MON-CKP rising while T-TPT stable = scaling signature
        p_mon_slope = slopes.get("P-MON-CKP_slope_60min") or slopes.get("P-MON-CKP_slope_30min")
        t_tpt_slope = slopes.get("T-TPT_slope_60min") or slopes.get("T-TPT_slope_30min")
        if p_mon_slope is not None and p_mon_slope > 0.005:
            if t_tpt_slope is None or abs(t_tpt_slope) < 0.01:
                is_pressure_rise = True

    # Decision logic
    if p_h >= 0.6 or (margin_c is not None and margin_c < 0):
        decision = "ALERT"
        confidence = max(p_h, 0.7)
        diagnosis = "hydrate_production_line"
    elif p_h >= 0.3 or (margin_c is not None and margin_c < 3):
        decision = "WATCH"
        confidence = 0.5
        diagnosis = "hydrate_production_line"
    elif is_pressure_rise or p_l >= 0.4:
        decision = "DISMISS"
        confidence = 0.65
        diagnosis = "scaling"
        if is_pressure_rise:
            evidence.append({"tool": "get_window", "summary": "P-MON-CKP rising, T-TPT stable — scaling pattern"})
        brief_parts.append("Upstream pressure rising with stable temperature — consistent with scaling, not hydrate.")
        if "scaling_vs_hydrate" not in playbook_refs:
            playbook_refs.append("scaling_vs_hydrate")
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
        "playbook_refs": playbook_refs,
        "brief": " ".join(brief_parts) if brief_parts else "Insufficient data for assessment.",
        "source": "rule_fallback",
    }


def _extract_numbers(text: str) -> set[float]:
    nums = set()
    for m in re.finditer(r'-?\d+\.?\d*', text):
        try:
            nums.add(float(m.group()))
        except ValueError:
            pass
    return nums


def _check_traceability(decision: dict, tool_results: dict[str, dict]) -> bool:
    tool_nums = set()
    for result in tool_results.values():
        for v in result.values():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                tool_nums.add(float(v))
                tool_nums.add(round(float(v), 1))
                tool_nums.add(round(float(v), 0))
            elif isinstance(v, dict):
                for vv in v.values():
                    if isinstance(vv, (int, float)) and not isinstance(vv, bool):
                        tool_nums.add(float(vv))
                        tool_nums.add(round(float(vv), 1))
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        for vv in item.values():
                            if isinstance(vv, (int, float)) and not isinstance(vv, bool):
                                tool_nums.add(float(vv))

    brief_nums = _extract_numbers(decision.get("brief", ""))
    brief_nums -= {0.0, 1.0, 2.0, 3.0, 100.0}

    untraceable = brief_nums - tool_nums
    if untraceable:
        log.warning("Untraceable numbers in brief: %s", untraceable)
        return False
    return True


def _playbook_query(trigger: dict) -> str:
    """Build a relevant playbook search query from the trigger reason."""
    reason = trigger.get("reason", "")
    if "P_MON_CKP" in reason and "rose" in reason:
        return "scaling vs hydrate pressure rise choke restriction"
    elif "p_lookalike" in reason:
        return "scaling flow instability look-alike vs hydrate"
    elif "dropped" in reason or "p_hydrate" in reason:
        return "hydrate alert response inhibitor injection"
    return "hydrate early warning signs detection"


def run_agent(ctx: AgentContext, trigger: dict,
              cache_only: bool = False) -> Generator[dict, None, None]:
    """Run the two-round agent loop. Yields tool_call, tool_result, decision events.

    If cache_only=True, only use cache or rule_fallback — never call the LLM.
    """
    client = get_llm_client()
    model_name = getattr(client, "model", "mock")

    # Check cache first
    cached = _load_cache(ctx.instance_id, ctx.minute_index, model_name)
    if cached:
        log.info("Cache hit for %s minute %d", ctx.instance_id, ctx.minute_index)
        yield from cached
        return

    if cache_only:
        # No cache, no LLM — use rule fallback with pre-run tools
        tool_results = {}
        t = trigger.get("t", "")
        events: list[dict] = []

        for tool_name in ("get_window", "classify_event"):
            result = dispatch_tool(tool_name, ctx, {})
            tool_results[tool_name] = result
            tc_evt = {"type": "tool_call", "data": {"t": t, "call_id": f"sys_{tool_name}", "tool": tool_name, "args": {}, "requested_by": "system"}}
            tr_evt = {"type": "tool_result", "data": {"t": t, "call_id": f"sys_{tool_name}", "tool": tool_name, "result": result, "requested_by": "system"}}
            events.extend([tc_evt, tr_evt])
            yield tc_evt
            yield tr_evt

        fallback = _build_rule_fallback(ctx, tool_results)
        fallback["_tokens"] = 0
        fallback["_cost"] = 0.0
        fallback["_requests"] = 0
        evt = {"type": "decision", "data": fallback}
        events.append(evt)
        yield evt
        _save_cache(ctx.instance_id, ctx.minute_index, model_name, events)
        return

    t = trigger.get("t", "")
    events: list[dict] = []
    tool_results: dict[str, dict] = {}
    total_tokens = 0
    total_cost = 0.0
    n_requests = 0
    max_requests = LLM_MAX_REQUESTS_PER_RUN
    n_tool_calls = 0

    # --- Round 0: Pre-run only get_window + classify_event ---
    # These two are always needed. LLM chooses additional tools adaptively.
    pre_run_specs = [
        ("get_window", {"minutes": 60}),
        ("classify_event", {}),
    ]
    pre_run_summary = []

    for tool_name, args in pre_run_specs:
        result = dispatch_tool(tool_name, ctx, args)
        tool_results[tool_name] = result
        n_tool_calls += 1

        tc_evt = {
            "type": "tool_call",
            "data": {"t": t, "call_id": f"sys_{tool_name}", "tool": tool_name,
                     "args": args, "requested_by": "system"},
        }
        tr_evt = {
            "type": "tool_result",
            "data": {"t": t, "call_id": f"sys_{tool_name}", "tool": tool_name,
                     "result": result, "requested_by": "system"},
        }
        events.extend([tc_evt, tr_evt])
        yield tc_evt
        yield tr_evt

        pre_run_summary.append(f"[{tool_name}] {json.dumps(result)}")

    # --- Round 1: LLM sees pre-run results + chooses additional tools ---
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"A watcher trigger fired for well {ctx.well_id} at {t}. "
            f"Reason: {trigger.get('reason', 'unknown')}. "
            f"Score: {trigger.get('score', 0)}.\n\n"
            f"The following tools were pre-run by the system:\n"
            + "\n".join(pre_run_summary) + "\n\n"
            f"Based on this initial evidence, request any additional tools you need "
            f"in one batch, then provide your JSON decision."
        )},
    ]

    start = time.time()

    for round_num in range(max_requests):
        if time.time() - start > 300:
            log.warning("Agent timeout after %.1fs", time.time() - start)
            break

        if n_requests >= max_requests:
            log.warning("Budget exhausted (%d requests)", n_requests)
            break

        try:
            resp = client.chat(messages, tools=TOOL_SCHEMAS, temperature=0, max_tokens=2048)
            n_requests += 1
        except Exception as e:
            log.error("LLM call failed: %s", e)
            n_requests += 1
            break

        # Check for pool exhaustion
        if resp.model == "exhausted":
            log.warning("Model pool exhausted")
            break

        total_tokens += resp.usage.get("total_tokens", 0)
        total_cost += resp.cost_estimate

        if not resp.tool_calls:
            # Final answer — try to parse as decision JSON
            decision = _parse_decision(resp.content, ctx, tool_results, messages, client,
                                       n_requests, max_requests)
            if decision:
                if decision and n_requests < max_requests:
                    n_requests += decision.pop("_extra_requests", 0)
                decision["_tokens"] = total_tokens
                decision["_cost"] = round(total_cost, 6)
                decision["_requests"] = n_requests
                decision["_model"] = resp.model
                evt = {"type": "decision", "data": decision}
                events.append(evt)
                yield evt
                _save_cache(ctx.instance_id, ctx.minute_index, model_name, events)
                return
            break

        # Process ALL tool calls from this response (parallel batch)
        assistant_tool_calls = []
        for tc in resp.tool_calls:
            if n_tool_calls >= MAX_TOOL_CALLS:
                log.warning("Max tool calls reached (%d)", MAX_TOOL_CALLS)
                break
            n_tool_calls += 1

            tc_evt = {
                "type": "tool_call",
                "data": {"t": t, "call_id": tc.id, "tool": tc.name,
                         "args": tc.arguments, "requested_by": "agent"},
            }
            events.append(tc_evt)
            yield tc_evt

            result = dispatch_tool(tc.name, ctx, tc.arguments)
            tool_results[tc.name] = result

            tr_evt = {
                "type": "tool_result",
                "data": {"t": t, "call_id": tc.id, "tool": tc.name,
                         "result": result, "requested_by": "agent"},
            }
            events.append(tr_evt)
            yield tr_evt

            assistant_tool_calls.append({
                "id": tc.id, "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}
            })

        # Add all tool calls + results to messages
        messages.append({
            "role": "assistant",
            "content": resp.content,
            "tool_calls": assistant_tool_calls,
        })
        for tc in resp.tool_calls:
            if tc.name in tool_results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(tool_results[tc.name]),
                })

    # If we got here without a decision, use rule fallback
    log.warning("Agent loop ended without LLM decision, using rule fallback")
    fallback = _build_rule_fallback(ctx, tool_results)
    fallback["_tokens"] = total_tokens
    fallback["_cost"] = round(total_cost, 6)
    fallback["_requests"] = n_requests
    fallback["_model"] = getattr(client, "model", "mock")
    evt = {"type": "decision", "data": fallback}
    events.append(evt)
    yield evt
    _save_cache(ctx.instance_id, ctx.minute_index, model_name, events)


def _parse_decision(content: str | None, ctx: AgentContext,
                    tool_results: dict, messages: list, client,
                    n_requests: int, max_requests: int) -> dict | None:
    """Parse LLM response as decision JSON. Retry once on failure."""
    if not content:
        return None

    extra_requests = 0
    for attempt in range(2):
        try:
            text = content.strip()
            if text.startswith("```"):
                text = re.sub(r'^```\w*\n?', '', text)
                text = re.sub(r'\n?```$', '', text)

            # Try to find JSON in the response
            if not text.startswith("{"):
                match = re.search(r'\{[\s\S]*\}', text)
                if match:
                    text = match.group()

            d = json.loads(text)

            # Validate with Pydantic
            dec = DecisionEvent(
                t=messages[1]["content"].split(" at ")[1].split(".")[0] if " at " in messages[1]["content"] else "",
                decision=d["decision"],
                recheck_min=d.get("recheck_min"),
                confidence=d.get("confidence", 0.5),
                diagnosis=d.get("diagnosis", "unknown"),
                onset_eta=OnsetEta(**(d.get("onset_eta") or {})),
                dose_wt_pct=d.get("dose_wt_pct"),
                dose_in_range=d.get("dose_in_range"),
                evidence=[EvidenceItem(**e) for e in d.get("evidence", [])],
                playbook_refs=d.get("playbook_refs", []),
                brief=d.get("brief", ""),
            )

            result = d.copy()

            # Traceability check — retry once if budget allows
            if not _check_traceability(result, tool_results) and attempt == 0 and (n_requests + extra_requests) < max_requests:
                messages.append({"role": "user", "content": (
                    "Your brief contains numbers that don't match any tool result. "
                    "Please revise: every number in the brief must come directly from a tool result."
                )})
                try:
                    resp = client.chat(messages, tools=TOOL_SCHEMAS, temperature=0, max_tokens=2048)
                    extra_requests += 1
                    content = resp.content
                    continue
                except Exception:
                    pass

            result["_extra_requests"] = extra_requests
            return result

        except (json.JSONDecodeError, KeyError, Exception) as e:
            if attempt == 0 and (n_requests + extra_requests) < max_requests:
                log.warning("Decision parse failed (attempt %d): %s", attempt + 1, e)
                messages.append({"role": "user", "content": (
                    f"Your response was not valid JSON or was missing required fields. Error: {e}. "
                    "Please respond with ONLY the JSON decision object, no other text."
                )})
                try:
                    resp = client.chat(messages, tools=TOOL_SCHEMAS, temperature=0, max_tokens=2048)
                    extra_requests += 1
                    content = resp.content
                except Exception:
                    return None
            else:
                log.error("Decision parse failed after retry: %s", e)
                return None

    return None


# ---------------------------------------------------------------------------
# Mock LLM scenarios for development
# ---------------------------------------------------------------------------

def _mock_hydrate_scenario() -> list[LLMResponse]:
    """Scripted responses for hydrate ALERT path.

    Round 1: LLM requests additional tools (parallel batch).
    Round 2: LLM returns decision JSON.
    """
    return [
        # Round 1: LLM requests additional tools in parallel
        LLMResponse(
            content=None,
            tool_calls=[
                ToolCall(id="c1", name="hydrate_margin", arguments={}),
                ToolCall(id="c2", name="forecast_onset", arguments={}),
                ToolCall(id="c3", name="search_playbook", arguments={"query": "hydrate alert response"}),
            ],
            usage={"prompt_tokens": 500, "completion_tokens": 50, "total_tokens": 550},
            model="mock",
        ),
        # Round 2: LLM returns ALERT decision
        LLMResponse(
            content=json.dumps({
                "decision": "ALERT",
                "recheck_min": None,
                "confidence": 0.82,
                "diagnosis": "hydrate_production_line",
                "onset_eta": {"p10": None, "p50": None, "p90": None},
                "dose_wt_pct": None,
                "dose_in_range": None,
                "evidence": [
                    {"tool": "classify_event", "summary": "Heuristic: 65% hydrate probability"},
                    {"tool": "get_window", "summary": "Pressure and temperature declining"},
                    {"tool": "search_playbook", "summary": "Hydrate response checklist recommends inhibitor injection"},
                ],
                "playbook_refs": ["hydrate_response"],
                "brief": "Pressure and temperature both declining. Heuristic classifier gives 65% hydrate probability. Recommend immediate inhibitor injection per hydrate response checklist.",
            }),
            tool_calls=[],
            usage={"prompt_tokens": 800, "completion_tokens": 200, "total_tokens": 1000},
            model="mock",
        ),
    ]


def _mock_scaling_scenario() -> list[LLMResponse]:
    """Scripted responses for scaling DISMISS path."""
    return [
        # Round 1: LLM requests search_playbook
        LLMResponse(
            content=None,
            tool_calls=[
                ToolCall(id="c1", name="search_playbook", arguments={"query": "scaling vs hydrate"}),
            ],
            usage={"prompt_tokens": 500, "completion_tokens": 30, "total_tokens": 530},
            model="mock",
        ),
        # Round 2: LLM returns DISMISS decision
        LLMResponse(
            content=json.dumps({
                "decision": "DISMISS",
                "recheck_min": None,
                "confidence": 0.78,
                "diagnosis": "scaling",
                "onset_eta": {"p10": None, "p50": None, "p90": None},
                "dose_wt_pct": None,
                "dose_in_range": None,
                "evidence": [
                    {"tool": "classify_event", "summary": "Heuristic: 5% hydrate, 10% look-alike"},
                    {"tool": "get_window", "summary": "No significant pressure or temperature drop"},
                    {"tool": "search_playbook", "summary": "Scaling shows pressure drop without temperature shift"},
                ],
                "playbook_refs": ["scaling_vs_hydrate"],
                "brief": "No significant pressure or temperature drop detected. Heuristic classifier gives only 5% hydrate probability. This appears to be normal operation.",
            }),
            tool_calls=[],
            usage={"prompt_tokens": 800, "completion_tokens": 200, "total_tokens": 1000},
            model="mock",
        ),
    ]


def get_mock_client_for_scenario(scenario: str = "hydrate") -> MockLLMClient:
    """Get a MockLLMClient pre-loaded with a scenario's responses."""
    if scenario == "hydrate":
        planned = _mock_hydrate_scenario()
    else:
        planned = _mock_scaling_scenario()
    return MockLLMClient(planned=planned)
