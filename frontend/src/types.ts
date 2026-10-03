export interface Metrics {
  true_positive: number;
  false_positive: number;
  false_negative: number;
  bad_hours: number;
  evaluated_hours: number;
  excluded_hours: number;
  precision: number | null;
  recall: number | null;
  false_alarm_episodes: number;
  false_alarms_per_normal_day: number | null;
  events_detected: number;
  event_count: number;
  delay_hours: number | null;
  cost: number;
}
export interface Policy {
  pressure_percentile: number;
  confirmation_percentile: number | null;
  consecutive: number;
  smoothing: boolean;
  pressure_cutoff: number;
  temperature_cutoff: number | null;
  flow_cutoff: number | null;
}
export interface System {
  id: string;
  name: string;
  description: string;
  validation: Metrics;
  test: Metrics;
  policy: Policy | null;
}
export interface Trial {
  id: number;
  policy: Policy;
  metrics: Metrics;
  name: string;
  action: string;
  selected: boolean;
}
export interface RealReport {
  status: string;
  message?: string;
  protocol: string;
  revision: string;
  license: string;
  selection: string;
  wells: number;
  minutes: number;
  total_minutes: number;
  macro_f1: number;
  hydrate_recall: number;
  confusion_matrix: number[][];
  class_names: string[];
  events_detected: number;
  hydrate_recordings: number;
  early_events: number;
  established_recordings: number;
  lookalikes_flagged: number;
  lookalike_recordings: number;
  false_alarms_per_normal_day: number;
  recordings: {
    file: string;
    well: string;
    class: number;
    fold: number;
    detected: boolean;
    early: boolean;
    lead_minutes: number | null;
    misdiagnosed: boolean;
    false_alarm_episodes: number;
  }[];
  folds: {
    fold: number;
    train_wells: string[];
    test_wells: string[];
    overlap: number;
    train_minutes: number;
    test_minutes: number;
  }[];
  limits: string[];
}
export interface Report {
  dataset: string;
  rows: number;
  missing_pressure: number;
  policy_id: string;
  dataset_sha256: string;
  split: {
    name: string;
    days: number;
    rows: number;
    usable: number;
    start: string;
    end: string;
  }[];
  systems: System[];
  selection: {
    trials: Trial[];
    selected_config: Policy;
    selected_trial: number | null;
    promoted: boolean;
    incumbent_cost: number;
    validation_cost: number;
    objective: string;
    weight_note: string;
    reason: string;
  };
  limits: string[];
  sources: { title: string; url: string }[];
  real: RealReport;
}
export interface Frame {
  t: string;
  decision: "ALERT" | "WATCH" | "DISMISS";
  diagnosis: string;
  brief: string;
  notify: boolean;
  quality: string;
  recheck_minutes: number;
  ground_truth: string;
  scores?: { normal: number; hydrate: number; lookalike: number };
  sensors: Record<string, number | null>;
  evidence: { tool: string; summary: string; result?: unknown }[];
}
export interface Scenario {
  id: string;
  title: string;
  provenance: string;
  description: string;
  cadence_minutes: number;
  display_stride?: number;
  policy_id: string;
  policy?: Policy;
  frames: Frame[];
  sensor_labels: Record<string, string>;
  frame_count?: number;
}
