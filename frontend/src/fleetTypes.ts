export type FleetStatus =
  "preparing" | "running" | "paused" | "completed" | "cancelled" | "failed";
export type WellStatus = "normal" | "watch" | "attention" | "unavailable";
export type AgentSource = "live" | "rules" | "recorded";
export type FleetMode = "guided" | "continuous";
export type FleetGuide = {
  phase:
    | "overview"
    | "seeking"
    | "assessing"
    | "checkpoint"
    | "manual_pause"
    | "ended";
  checkpoint: {
    id: string;
    kind:
      | "concern"
      | "escalation"
      | "recovery"
      | "telemetry_loss"
      | "telemetry_recovery"
      | "operator_followup"
      | "step"
      | "ended";
    title: string;
    summary: string;
    well_id: string | null;
    source_timestamp: string | null;
    index: number;
  } | null;
};
export type ToolTrace = {
  name: string;
  status: "running" | "done" | "failed";
  summary?: string;
  sources?: string[];
};
export type FleetFrame = {
  t: string;
  elapsed_seconds: number;
  sensors: Record<string, number | null>;
  risk_score: number | null;
};
export type FleetCheckResult = "confirmed" | "not_confirmed" | "unavailable";
export type FleetCheck = {
  id: string;
  definition: string;
  label: string;
  reason: string;
  status: "pending" | FleetCheckResult;
  note?: string;
  updated_at?: string | null;
};
export type FleetTimelineEvent = {
  id: string;
  kind:
    | "detected"
    | "seen"
    | "checked"
    | "recheck"
    | "recovered"
    | "observation"
    | "review_completed";
  at: string;
  recorded_at?: string;
  summary: string;
};
export type FleetEvidenceSnapshot = {
  at: string;
  status: WellStatus;
  summary: string;
  evidence: string[];
};
export type FleetWell = {
  id: string;
  name: string;
  status: WellStatus;
  source_file: string;
  source_timestamp: string | null;
  quality: {
    status: "good" | "degraded" | "unavailable";
    summary: string;
    missing: string[];
    invalid: string[];
    unchanged: string[];
  };
  last_assessed: string | null;
  next_check: string | null;
  incident: {
    id: string;
    acknowledged: boolean;
    completed: boolean;
    condition_cleared?: boolean;
    note?: string;
  } | null;
  assessment: {
    summary: string;
    evidence: string[];
    next_step: string;
    source: AgentSource;
    uncertainty?: string;
    tools: ToolTrace[];
    category?: string;
    alternatives?: string[];
    checks?: FleetCheck[];
    playbook_refs?: string[];
  } | null;
  timeline?: FleetTimelineEvent[];
  followup?: {
    last_checked_at: string | null;
    next_check_at: string | null;
    trigger: string;
    change_summary: string;
    before?: FleetEvidenceSnapshot | null;
    after?: FleetEvidenceSnapshot | null;
  };
  frames: FleetFrame[];
  investigation: "idle" | "queued" | "running" | "complete" | "unavailable";
  activity: string;
};
export type FleetRun = {
  id: string;
  status: FleetStatus;
  mode: FleetMode;
  guide: FleetGuide;
  speed: number;
  elapsed_seconds: number;
  index: number;
  total: number;
  agent_mode: AgentSource;
  llm_configured: boolean;
  llm_deferred?: boolean;
  model_ready: boolean;
  readiness_message: string;
  wells: FleetWell[];
  priority: string[];
  last_event_id: number;
  requests_used: number;
  request_budget: number;
  error?: string;
  fault: { well_id: string; remaining: number; kind: string } | null;
};
export type FleetCatalog = {
  wells: { id: string; name: string; source_file: string }[];
  default_speed: number;
  default_mode: FleetMode;
  speeds: number[];
  model_ready: boolean;
  llm_configured: boolean;
  llm_deferred?: boolean;
  readiness_message: string;
};
export type FleetEvent = { id: number; type: string; payload: FleetRun };
export type FleetWorkflowReport = {
  status: "complete" | "not_run";
  generated_at?: string;
  model_id?: string;
  policy_version?: string;
  summary?: { passed: number; total: number };
  metrics?: {
    recordings: number;
    observed_minutes: number;
    labelled_minutes: number;
    normal_minutes: number;
    workflow_attention_minutes: number;
    workflow_watch_minutes: number;
    workflow_unavailable_minutes: number;
    workflow_false_attention_minutes: number;
    workflow_false_watch_minutes: number;
    workflow_review_episodes: number;
    model_only_false_attention_minutes: number;
    hydrate_events_detected: number;
    hydrate_events_total: number;
    restriction_events_detected: number;
    restriction_events_total: number;
    scaling_events_detected: number;
    scaling_events_total: number;
  };
  scenarios?: {
    id: string;
    name: string;
    status: "pass" | "fail";
    evidence: string;
    metrics?: Record<string, number>;
  }[];
  limits?: string[];
};
