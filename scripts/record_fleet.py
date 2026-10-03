#!/usr/bin/env python3
"""Record a fleet demo directly (no running server needed).

Usage:
    python scripts/record_fleet.py              # rules-only
    LLM_MODE=live python scripts/record_fleet.py  # with LLM (needs GEMINI_API_KEY)

Output: data/demo/fleet-recording-<session_id>.json
"""
import asyncio
import json
import os
import sys
import time

os.environ.setdefault("LLM_MODE", "mock")


async def main():
    from backend.fleet import FleetSession, safe

    use_llm = os.environ.get("LLM_MODE") == "live"
    session = FleetSession(speed=6000, use_llm=use_llm, mode="continuous")
    print(f"Session: {session.id}")
    print(f"LLM: {'live' if use_llm else 'rules-only'}")
    print(f"Wells: {list(session.wells.keys())}")

    start = time.time()
    await session.run()
    elapsed = time.time() - start

    snap = session.snapshot()
    print(f"\nCompleted in {elapsed:.1f}s, {snap['index']}/{snap['total']} minutes")

    for w in snap["wells"]:
        print(f"  {w['well_id']}: status={w['status']}, investigation={w['investigation']}")

    # Export
    os.makedirs("data/demo", exist_ok=True)
    outfile = f"data/demo/fleet-recording-{session.id[:8]}.json"
    payload = {"run": snap, "audit": session.audit, "events": session.events,
               "method": "Four independent historical excerpts. Labels are excluded from runtime."}
    with open(outfile, "w") as f:
        json.dump(safe(payload), f, indent=2, allow_nan=False)
    print(f"\nSaved: {outfile} ({os.path.getsize(outfile) / 1024:.0f} KB)")


if __name__ == "__main__":
    asyncio.run(main())
