export type FleetStatus =
  "preparing" | "running" | "paused" | "completed" | "cancelled" | "failed";
export type WellStatus = "normal" | "watch" | "attention" | "unavailable";
export type AgentSource = "live" | "rules" | "recorded";
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
  } | null;
  frames: FleetFrame[];
  investigation: "idle" | "queued" | "running" | "complete" | "unavailable";
  activity: string;
};
export type FleetRun = {
  id: string;
  status: FleetStatus;
  speed: number;
  elapsed_seconds: number;
  index: number;
  total: number;
  agent_mode: AgentSource;
  llm_configured: boolean;
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
  speeds: number[];
  model_ready: boolean;
  llm_configured: boolean;
  readiness_message: string;
};
export type FleetEvent = { id: number; type: string; payload: FleetRun };
