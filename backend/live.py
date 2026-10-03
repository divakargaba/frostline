"""Session control and reconnectable server-sent events for the live agent."""
import asyncio
import json
from typing import Literal
from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from src.live_agent import CATALOG, SESSIONS, create_session

router = APIRouter(prefix="/api/live", tags=["Live agent"])


class Start(BaseModel):
    scenario: Literal["mission", "sandbox", "test"] = "mission"
    speed: Literal[1, 2, 4, 8] = 2


class Control(BaseModel):
    action: Literal["pause", "resume", "step", "cancel", "speed", "inject"]
    speed: Literal[1, 2, 4, 8] | None = None
    fault: Literal["pressure_dip", "pressure_offline", "hydrate_pattern", "frozen_feed", "restore"] | None = None


def get_session(id):
    session = SESSIONS.get(id)
    if session is None:
        raise HTTPException(404, "Run not found or server restarted. Start a new mission.")
    return session


@router.get("/catalog")
def catalog():
    return {"scenarios": CATALOG, "controller": "Stateful deterministic agent", "source": "Historical sensor feed, evaluated incrementally", "llm_enabled": False}


@router.post("/sessions", status_code=201)
async def start(config: Start):
    try:
        session = create_session(config.scenario, config.speed)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return session.snapshot()


@router.get("/sessions/{id}")
async def snapshot(id: str):
    return get_session(id).snapshot()


@router.post("/sessions/{id}/control")
async def control(id: str, command: Control):
    session = get_session(id)
    try:
        session.control(command.action, command.speed, command.fault)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"status": session.status, "speed": session.speed, "fault": session.fault}


@router.get("/sessions/{id}/events")
async def stream(id: str, request: Request, after: int = Query(0, ge=0), last_event_id: str | None = Header(None)):
    session = get_session(id)
    try:
        cursor = max(after, int(last_event_id or 0))
    except ValueError:
        raise HTTPException(422, "Last-Event-ID must be an integer")
    if cursor > len(session.events):
        raise HTTPException(422, "Event cursor is ahead of this session")
    async def events():
        nonlocal cursor
        while not await request.is_disconnected():
            for item in session.events[cursor:]:
                cursor = item["id"]
                yield f"id: {cursor}\nevent: agent\ndata: {json.dumps(item, allow_nan=False)}\n\n"
            if session.status in ["completed", "cancelled", "failed"] and cursor >= len(session.events):
                break
            session.changed.clear()
            if len(session.events) > cursor:
                continue
            try:
                await asyncio.wait_for(session.changed.wait(), timeout=15)
            except asyncio.TimeoutError:
                yield ": heartbeat\n\n"
    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/sessions/{id}/export")
async def export(id: str):
    session = get_session(id)
    payload = {**session.snapshot(), "events": session.events, "method": "Sensor input → conditional tool calls → decision → evaluator label. Separate historical validation selects the policy."}
    return Response(json.dumps(payload, indent=2, allow_nan=False), media_type="application/json", headers={"Content-Disposition": f'attachment; filename="frostline-run-{id[:8]}.json"'})
