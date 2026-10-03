"""Pydantic models for API request/response contracts.

Owner: Div

These are the shared data contracts between backend and frontend.
Every SSE event and HTTP response must validate against these schemas.
"""

from __future__ import annotations

from pydantic import BaseModel


# --- /health response ---


class HealthResponse(BaseModel):
    status: str
    has_processed: bool
    has_raw: bool
    has_seed: bool
    has_physics: bool
    has_model: bool
    llm_mode: str  # "openrouter" | "mock"
    llm_model: str


# --- /wells response ---


class WellInfo(BaseModel):
    well_id: str
    instance_id: str
    source: str  # "real" | "simulated" | "drawn" | "seed"
    sensors_available: list[str]
    has_hydrate_event: bool
    has_forming_phase: bool = False
    n_minutes: int = 0


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
    phase: str  # "normal" | "forming" | "established"


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


class ErrorEvent(BaseModel):
    t: str
    message: str


class EndEvent(BaseModel):
    t: str
    total_minutes: int
    instance_id: str


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
    mock: bool = False


# --- /tts request ---


class TTSRequest(BaseModel):
    text: str


# --- Fleet models ---


class FleetSessionCreate(BaseModel):
    speed: int = 60
    agent: bool = True
    mode: str = "live"  # "live" | "cached" | "cache_only"
    record: bool = False


class FleetSessionResponse(BaseModel):
    session_id: str
    n_wells: int
    max_minutes: int
    mode: str


class FleetTickEvent(BaseModel):
    session_id: str
    well_id: str
    display_id: str
    replay_minute: int
    original_t: str
    sensors: dict[str, float | None]
    margin_C: float | None = None
    p_hydrate: float | None = None
    p_lookalike: float | None = None
    p_normal: float | None = None
    quality: dict[str, str] | None = None


class FleetPhaseMarkerEvent(BaseModel):
    session_id: str
    well_id: str
    display_id: str
    t: str
    phase: str


class FleetWatchTriggerEvent(BaseModel):
    session_id: str
    well_id: str
    display_id: str
    t: str
    reason: str
    score: float
    incident_id: str


class FleetToolCallEvent(BaseModel):
    session_id: str
    well_id: str
    display_id: str
    t: str
    call_id: str
    tool: str
    args: dict
    requested_by: str = "system"


class FleetToolResultEvent(BaseModel):
    session_id: str
    well_id: str
    display_id: str
    t: str
    call_id: str
    tool: str
    result: dict
    requested_by: str = "system"


class PriorityEntry(BaseModel):
    well_id: str
    display_id: str
    category: str  # "review_now" | "investigate" | "watch" | "normal"
    priority_rank: int
    headline: str
    score: float = 0.0
    unacknowledged: bool = True
    next_check_minute: int | None = None


class FleetCounters(BaseModel):
    needs_review: int = 0
    unacknowledged: int = 0
    reliable_feeds: int = 0
    checks_due: int = 0


class FleetAssessmentEvent(BaseModel):
    session_id: str
    well_id: str
    display_id: str
    incident_id: str
    decision: str  # "ALERT" | "WATCH" | "DISMISS"
    category: str  # "review_now" | "investigate" | "watch" | "normal"
    confidence: float
    diagnosis: str
    brief: str
    next_action: str = ""
    next_check_minute: int | None = None
    uncertainty: str = ""
    evidence_refs: list[dict] = []
    playbook_refs: list[str] = []
    source: str = "agent"  # "agent" | "fallback"


class FleetStatusEvent(BaseModel):
    session_id: str
    held: bool
    reason: str = ""
    current_minute: int
    priority_list: list[PriorityEntry]
    counters: FleetCounters


class FleetEndEvent(BaseModel):
    session_id: str
    total_minutes: int
    llm_attempts_used: int


class AckRequest(BaseModel):
    pass


class AckResponse(BaseModel):
    incident_id: str
    acknowledged: bool
