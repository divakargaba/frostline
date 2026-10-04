#!/usr/bin/env python3
"""Record a fleet demo directly (no running server needed).

Usage:
    python scripts/record_fleet.py              # rules-only
    USE_LLM=1 python scripts/record_fleet.py    # LLM investigations (needs a provider key in .env)

Output: data/demo/fleet-recording-<session_id>.json
"""
import asyncio
import json
import os
import sys
import time



async def main():
    from backend.fleet import FleetSession, safe

    session = FleetSession(speed=6000, use_llm=os.environ.get("USE_LLM") == "1")
    print(f"Session: {session.id}")
    print(f"LLM: {'on (' + ', '.join(session.caps['llm_providers']) + ')' if session.use_llm else 'rules-only'}")
    print(f"Wells: {list(session.wells.keys())}")

    start = time.time()
    await session.run()
    elapsed = time.time() - start

    snap = session.snapshot()
    print(f"\nCompleted in {elapsed:.1f}s, {snap['index']}/{snap['total']} minutes")

    for w in snap["wells"]:
        print(f"  {w['id']}: status={w['status']}, investigation={w['investigation']}")

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
