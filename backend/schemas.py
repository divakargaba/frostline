"""Pydantic models for API request/response contracts.

Owner: Div

These are the shared data contracts between backend and frontend.
Every SSE event and HTTP response must validate against these schemas.
"""

from __future__ import annotations

from pydantic import BaseModel


# --- /wells response ---


class WellInfo(BaseModel):
    well_id: str
    instance_id: str
    source: str  # "real" | "simulated" | "drawn"
    sensors_available: list[str]
    has_hydrate_event: bool


# --- SSE event payloads ---


class TickEvent(BaseModel):
    t: str
    sensors: dict[str, float | None]
    margin_C: float | None = None
    p_hydrate: float | None = None
    p_lookalike: float | None = None
    p_normal: float | None = None


class PhaseMarkerEvent(BaseModel):
    t: str
    phase: str  # "forming" | "established"


class WatchTriggerEvent(BaseModel):
    t: str
    reason: str
    score: float


class ToolCallEvent(BaseModel):
    t: str
    call_id: str
    tool: str
    args: dict


class ToolResultEvent(BaseModel):
    t: str
    call_id: str
    tool: str
    result: dict


class EvidenceItem(BaseModel):
    tool: str
    summary: str


class OnsetEta(BaseModel):
    p10: float | None = None
    p50: float | None = None
    p90: float | None = None


class DecisionEvent(BaseModel):
    t: str
    decision: str  # "ALERT" | "WATCH" | "DISMISS"
    recheck_min: int | None = None
    confidence: float
    diagnosis: str
    onset_eta: OnsetEta
    dose_wt_pct: float | None = None
    dose_in_range: bool | None = None
    evidence: list[EvidenceItem]
    playbook_refs: list[str]
    brief: str


# --- /results response ---


class SystemResult(BaseModel):
    name: str
    description: str
    caught: int
    missed: int
    false_alarms_per_day: float
    mean_lead_time_min: float | None = None
    misdiagnosis_rate: float


class ResultsResponse(BaseModel):
    systems: list[SystemResult]


# --- /tts request ---


class TTSRequest(BaseModel):
    text: str
