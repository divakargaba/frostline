#!/usr/bin/env python3
"""Record a fleet demo with real LLM decisions.

Usage:
    LLM_MIN_INTERVAL_S=15 python scripts/record_fleet.py

Requires GEMINI_API_KEY in .env. Wait for Gemini quota to reset if needed
(free tier: 20 requests/day, resets every few hours).
"""
import asyncio
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")

from backend.fleet import create_session, run_fleet_stream


async def main():
    session = create_session(speed=6000, agent=True, mode="live", record=True)
    print(f"Session: {session.session_id}")
    print(f"Recording to: {session.recording_path}")
    print(f"Wells: {len(session.wells)}, max_minutes: {session.max_minutes}")

    queue = asyncio.Queue(maxsize=50000)
    session.subscribers.append(queue)
    await run_fleet_stream(session, queue)

    events = []
    while not queue.empty():
        events.append(queue.get_nowait())

    assessments = [e for e in events if e["type"] == "fleet_assessment"]
    tool_calls = [e for e in events if e["type"] == "fleet_tool_call"]

    print(f"\n=== RESULTS ===")
    print(f"Assessments: {len(assessments)}")
    print(f"LLM attempts: {session.llm_attempts_used}")

    all_from_llm = True
    for i, a in enumerate(assessments):
        d = a["data"]
        src = d["source"]
        if src != "agent":
            all_from_llm = False
        agent_tools = [tc["data"]["tool"] for tc in tool_calls
                       if tc["data"]["well_id"] == d["well_id"]
                       and tc["data"].get("requested_by") == "agent"]
        sp = any(tc["data"]["tool"] == "search_playbook" for tc in tool_calls
                 if tc["data"]["well_id"] == d["well_id"])
        print(f"\n  #{i+1} {d['display_id']}: {d['decision']} (source={src})")
        print(f"     Agent tools: {agent_tools}")
        print(f"     search_playbook: {sp}")
        print(f"     Brief: {d['brief'][:120]}")

    if all_from_llm:
        # Update fleet.json to point at this recording
        manifest_path = "data/demo/fleet.json"
        manifest = json.loads(open(manifest_path).read())
        manifest["demo_recording"] = session.recording_path.name
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")
        print(f"\n✓ All decisions from LLM. Updated fleet.json -> {session.recording_path.name}")
    else:
        print(f"\n✗ Some decisions from fallback. Recording NOT linked in fleet.json.")
        print("  Wait for Gemini quota to reset and re-run.")
        sys.exit(1)

    print(f"\nRecording: {session.recording_path}")


if __name__ == "__main__":
    asyncio.run(main())
