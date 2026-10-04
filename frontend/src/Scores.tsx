import { useEffect, useState } from "react";
import { ArrowRight, Check, ChevronRight, CircleAlert, Download } from "lucide-react";
import { api, number } from "./api";
import type { FleetWorkflowReport, ScoreRow } from "./fleetTypes";
import "./fleet.css";

type FleetMetrics = {
  events_detected: number;
  hydrate_events: number;
  false_alarm_episodes: number;
  false_alarm_minutes: number;
  false_alarm_minutes_per_normal_day: number;
  missed_hydrate_minutes: number;
  mean_detection_delay_minutes: number | null;
};
type FleetReport = {
  model_id: string;
  training_recordings: number;
  evaluation_recordings: number;
  training_wells: string[];
  excluded_wells: string[];
  selected_policy: {
    activation_threshold: number;
    persistence_minutes: number;
    recovery_minutes: number;
  };
  validation_baseline: FleetMetrics;
  validation_selected: FleetMetrics;
  candidates: {
    policy: { activation_threshold: number; persistence_minutes: number };
    metrics: FleetMetrics;
    accepted: boolean;
  }[];
  stages: { id: string; name: string; metrics: FleetMetrics }[];
  limits: string[];
  recordings: { well_id: string; hydrate_event: boolean }[];
  model_comparison?: {
    id: string;
    qualifies: boolean;
    selected: boolean;
    validation: {
      events_detected: number;
      hydrate_events: number;
      false_alarm_minutes: number;
      classification_recall: { lookalike: number };
    };
  }[];
};
const classifierNames: Record<string, string> = {
  base_class: "Current",
  base_record: "Equal recording weights",
  relative_record: "Relative changes",
  hybrid_record: "Combined features",
};

function useApi<T>(path: string, check?: (value: T) => void) {
  const [value, setValue] = useState<T | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let live = true;
    api<T>(path)
      .then((next) => {
        check?.(next);
        if (live) setValue(next);
      })
      .catch((e) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, [path]);
  return [value, error] as const;
}

function Empty({ text }: { text: string }) {
  return <p className="fleet-workflow-empty">{text}</p>;
}

function Ladder() {
  const [data, error] = useApi<{ rows: ScoreRow[] }>("/scores");
  const rows = data?.rows ?? [];
  // The biggest jump in events caught on the same benchmark is the step worth
  // pointing at; ties go to the later row (the improvement round, not B0 → B1).
  let step = -1;
  rows.forEach((row, i) => {
    const prev = rows[i - 1];
    const gain =
      prev && prev.total_events === row.total_events ? row.caught - prev.caught : 0;
    if (gain > 0 && (step < 0 || gain >= rows[step].caught - rows[step - 1].caught))
      step = i;
  });
  return (
    <section className="fleet-card fleet-results" aria-label="Baseline ladder">
      <div className="fleet-card-heading">
        <div>
          <div className="fleet-eyebrow">Baseline ladder</div>
          <h2>Events caught, false alarms, lead time</h2>
        </div>
        {rows[0] && (
          <span className="fleet-source">{rows[0].total_events} events</span>
        )}
      </div>
      {error ? (
        <Empty text={error} />
      ) : !data ? (
        <Empty text="Loading…" />
      ) : (
        <div className="fleet-result-table-wrap">
          <table className="fleet-result-table fleet-ladder">
            <thead>
              <tr>
                <th>System</th>
                <th>Description</th>
                <th>Caught</th>
                <th>False alarms / day</th>
                <th>Lead time (min)</th>
                <th>Notes</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row, i) => (
                <tr key={row.name} className={i === step ? "step" : ""}>
                  <td>
                    <strong>{row.name}</strong>
                  </td>
                  <td>{row.description}</td>
                  <td>
                    <div className="fleet-ladder-caught">
                      <strong>
                        {row.caught} / {row.total_events}
                      </strong>
                      {i === step && (
                        <em>+{row.caught - rows[i - 1].caught}</em>
                      )}
                      <i>
                        <b
                          style={{
                            width: `${(row.caught / (row.total_events || 1)) * 100}%`,
                          }}
                        />
                      </i>
                    </div>
                  </td>
                  <td>{number(row.false_alarms_per_day, 2)}</td>
                  <td>{number(row.mean_lead_time_min, 1)}</td>
                  <td>
                    <span className="fleet-ladder-notes" title={row.notes}>
                      {row.notes || "—"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function FleetResults() {
  const [report, error] = useApi<FleetReport>("/fleet/results", (value) => {
    if (!value.stages) throw new Error("Fleet evaluation not run yet.");
  });
  const [expanded, setExpanded] = useState(false);
  if (!report)
    return (
      <section className="fleet-results fleet-card">
        <Empty text={error || "Loading…"} />
      </section>
    );
  const policy = report.selected_policy;
  const hydrateWells = new Set(
    report.recordings.filter((r) => r.hydrate_event).map((r) => r.well_id),
  ).size;
  return (
    <section className="fleet-results fleet-card" aria-label="Fleet model holdout">
      <div className="fleet-card-heading">
        <div>
          <div className="fleet-eyebrow">Fleet model · held-out wells</div>
          <h2>Holdout results</h2>
        </div>
        <span className="fleet-source">{report.model_id}</span>
      </div>
      <div className="fleet-evaluation-split">
        <div>
          <span>01 · TRAIN</span>
          <strong>{report.training_recordings} recordings</strong>
          <small>{report.training_wells.length} wells</small>
        </div>
        <ArrowRight size={17} />
        <div>
          <span>02 · SELECT</span>
          <strong>{report.candidates.length} policies</strong>
          <small>
            {report.model_comparison?.length
              ? `${report.model_comparison.length} classifiers`
              : "Grouped validation"}
          </small>
        </div>
        <ArrowRight size={17} />
        <div>
          <span>03 · TEST</span>
          <strong>{report.evaluation_recordings} recordings</strong>
          <small>{report.excluded_wells.length} unseen wells</small>
        </div>
      </div>
      <div className="fleet-result-table-wrap">
        <table className="fleet-result-table">
          <thead>
            <tr>
              <th>Stage</th>
              <th>Caught</th>
              <th>False episodes</th>
              <th>False min</th>
              <th>Missed hydrate min</th>
              <th>False min / day</th>
              <th>Delay</th>
            </tr>
          </thead>
          <tbody>
            {report.stages.map(({ id, name, metrics: m }, index) => (
              <tr key={id}>
                <td>
                  <span>{String(index + 1).padStart(2, "0")}</span>
                  {name}
                </td>
                <td>
                  {m.events_detected} / {m.hydrate_events}
                </td>
                <td>{m.false_alarm_episodes}</td>
                <td>{m.false_alarm_minutes.toLocaleString()}</td>
                <td>{m.missed_hydrate_minutes.toLocaleString()}</td>
                <td>{number(m.false_alarm_minutes_per_normal_day, 1)}</td>
                <td>{number(m.mean_detection_delay_minutes, 2)} min</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="fleet-result-summary">
        <div>
          <h3>Policy</h3>
          <p>
            Threshold {number(policy.activation_threshold, 2)} ·{" "}
            {policy.persistence_minutes} min persistence ·{" "}
            {policy.recovery_minutes} min recovery
          </p>
        </div>
        <div>
          <h3>Validation</h3>
          <p>
            Caught {report.validation_baseline.events_detected} →{" "}
            {report.validation_selected.events_detected} · False min{" "}
            {report.validation_baseline.false_alarm_minutes.toLocaleString()} →{" "}
            {report.validation_selected.false_alarm_minutes.toLocaleString()}
          </p>
        </div>
      </div>
      <div className="fleet-result-limit">
        <CircleAlert size={14} />
        <p>
          {report.stages[0]?.metrics.hydrate_events ?? 0} hydrate events ·{" "}
          {hydrateWells || 1} held-out hydrate well{hydrateWells > 1 ? "s" : ""}{" "}
          · internal evaluation
        </p>
      </div>
      <div className="fleet-result-details">
        {!!report.model_comparison?.length && (
          <details className="fleet-model-comparison">
            <summary>Classifiers</summary>
            <div className="fleet-result-expanded">
              <div className="fleet-result-table-wrap">
                <table className="fleet-result-table">
                  <thead>
                    <tr>
                      <th>Classifier</th>
                      <th>Caught</th>
                      <th>False min</th>
                      <th>Look-alike recall</th>
                      <th>Decision</th>
                    </tr>
                  </thead>
                  <tbody>
                    {report.model_comparison.map(({ id, validation: v, selected, qualifies }) => (
                      <tr key={id}>
                        <td>{classifierNames[id] || id}</td>
                        <td>
                          {v.events_detected} / {v.hydrate_events}
                        </td>
                        <td>{v.false_alarm_minutes.toLocaleString()}</td>
                        <td>{number(v.classification_recall.lookalike * 100, 1)}%</td>
                        <td>
                          {selected
                            ? id === "base_class"
                              ? "Retained"
                              : "Selected"
                            : qualifies
                              ? "Qualifies"
                              : "Rejected"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </details>
        )}
        <button
          className="fleet-text-button"
          onClick={() => setExpanded(!expanded)}
          aria-expanded={expanded}
        >
          {expanded ? "Hide" : "Candidates & limits"}
          <ChevronRight size={13} />
        </button>
        {expanded && (
          <div className="fleet-result-expanded">
            <div className="fleet-result-table-wrap">
              <table className="fleet-result-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Threshold</th>
                    <th>Persistence</th>
                    <th>Caught</th>
                    <th>False min</th>
                    <th>Decision</th>
                  </tr>
                </thead>
                <tbody>
                  {report.candidates.map(({ policy: p, metrics: m, accepted }, index) => (
                    <tr key={index}>
                      <td>{index + 1}</td>
                      <td>{number(p.activation_threshold, 2)}</td>
                      <td>{p.persistence_minutes} min</td>
                      <td>
                        {m.events_detected} / {m.hydrate_events}
                      </td>
                      <td>{m.false_alarm_minutes.toLocaleString()}</td>
                      <td>{accepted ? "Accept" : "Keep"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <ul>
              {report.limits.map((limit) => (
                <li key={limit}>{limit}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
      <WorkflowResults />
    </section>
  );
}

function WorkflowResults() {
  const [report, error] = useApi<FleetWorkflowReport>("/fleet/workflow-results");
  const m = report?.metrics;
  const coverage = m && [
    ["Hydrate", m.hydrate_events_detected, m.hydrate_events_total],
    ["Restriction", m.restriction_events_detected, m.restriction_events_total],
    ["Scaling", m.scaling_events_detected, m.scaling_events_total],
  ];
  return (
    <section className="fleet-workflow-results" aria-label="Operator workflow">
      <div className="fleet-card-heading">
        <div>
          <div className="fleet-eyebrow">Operator workflow</div>
          <h2>Follow-up</h2>
        </div>
        <span className="fleet-source">No LLM calls</span>
      </div>
      {error ? (
        <Empty text={error} />
      ) : !report ? (
        <Empty text="Loading…" />
      ) : report.status !== "complete" || !m || !coverage ? (
        <Empty text="Not run yet." />
      ) : (
        <>
          <div className="fleet-workflow-metrics">
            <div>
              <span>Normal min flagged</span>
              <strong>
                {m.model_only_false_attention_minutes.toLocaleString()}{" "}
                <ArrowRight size={13} />{" "}
                {m.workflow_false_attention_minutes.toLocaleString()}
              </strong>
              <small>Model → workflow</small>
            </div>
            <div>
              <span>Normal min on watch</span>
              <strong>{m.workflow_false_watch_minutes.toLocaleString()}</strong>
            </div>
            <div>
              <span>Unavailable min</span>
              <strong>{m.workflow_unavailable_minutes.toLocaleString()}</strong>
            </div>
          </div>
          <p className="fleet-workflow-scope">
            {m.recordings} recordings · {m.observed_minutes.toLocaleString()}{" "}
            min · {m.normal_minutes.toLocaleString()} normal min
          </p>
          <div className="fleet-workflow-disclosures">
            <details>
              <summary>Event coverage</summary>
              <div className="fleet-workflow-detail">
                <table className="fleet-sensor-table">
                  <tbody>
                    {coverage.map(([label, hit, total]) => (
                      <tr key={label}>
                        <td>{label}</td>
                        <td>
                          {hit} / {total}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <ul>
                  {report.limits?.map((limit) => (
                    <li key={limit}>{limit}</li>
                  ))}
                </ul>
              </div>
            </details>
            <details>
              <summary>
                Scripted checks{" "}
                <span>
                  {report.summary?.passed ?? 0} / {report.summary?.total ?? 0}
                </span>
              </summary>
              <div className="fleet-workflow-detail">
                <ol className="fleet-workflow-scenarios">
                  {report.scenarios?.map((scenario) => (
                    <li key={scenario.id}>
                      <div>
                        <strong>{scenario.name}</strong>
                        <span className={scenario.status}>
                          {scenario.status === "pass" ? (
                            <Check size={12} />
                          ) : (
                            <CircleAlert size={12} />
                          )}{" "}
                          {scenario.status === "pass" ? "Pass" : "Fail"}
                        </span>
                      </div>
                      <p>{scenario.evidence}</p>
                    </li>
                  ))}
                </ol>
              </div>
            </details>
          </div>
        </>
      )}
    </section>
  );
}

export default function Scores() {
  return (
    <section className="fleet-dashboard" aria-label="Scores">
      <header className="fleet-header">
        <div>
          <div className="fleet-eyebrow">Frostline · Scores</div>
          <h1>Scores</h1>
        </div>
        <a
          className="fleet-button"
          href="/api/fleet/results"
          download="frostline-fleet-results.json"
        >
          <Download size={15} />
          Export
        </a>
      </header>
      <Ladder />
      <FleetResults />
    </section>
  );
}
