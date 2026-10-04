#!/usr/bin/env python3
"""Smoke test: one real fleet LLM investigation through the provider pool (Gemini -> Groq -> OpenRouter).

Replays 40 source minutes of the four demo wells, then asks the LLM investigator to review
WELL-00019 exactly as a `use_llm: true` run would. Uses at most 3 provider requests.

Usage: python scripts/llm_smoke_test.py   (needs a provider key in .env and the trained fleet model)
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def run(well="WELL-00019"):
    from backend.fleet import SESSIONS, FleetSession
    session = FleetSession(use_llm=True)
    SESSIONS[session.id] = session
    print("Providers:", ", ".join(session.caps["llm_providers"]))
    await session.prepare()
    for _ in range(40):
        session.advance()
    await session.investigate_well(well)
    result = next(e["result"] for e in reversed(session.audit) if e["kind"] == "assessment" and "result" in e)
    meta = result.get("metadata", {})
    item = session.wells[well]
    print(f"Source: {result.get('source')}  provider: {meta.get('provider')}  model: {meta.get('model')}  attempts: {meta.get('attempts')}  tools: {meta.get('tool_calls')}  latency: {meta.get('latency_ms')} ms")
    print(f"Status: {item['status']}  investigation: {item['investigation']}")
    print(f"Brief: {item['assessment']['summary']}")
    if meta.get("failure_reason"):
        print(f"Fallback reason: {meta['failure_reason']}")
    return result.get("source") == "live_llm"


def main():
    import backend.config  # noqa: F401  (loads .env)
    from src.llm import configured_providers
    if not configured_providers():
        print("SKIP - no provider configured. Set GEMINI_API_KEY, GROQ_API_KEY or OPENROUTER_API_KEY in .env (and unset LLM_MODE=mock).")
        return 0
    ok = asyncio.run(run())
    print("OK" if ok else "DEGRADED - the investigation fell back to rules")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
