import { memo, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import {
  Activity,
  ArrowDownRight,
  ArrowRight,
  Check,
  CheckCheck,
  ChevronRight,
  CircleAlert,
  Clock3,
  Compass,
  Download,
  Gauge,
  LoaderCircle,
  MessageSquare,
  Mic,
  Pause,
  Play,
  RotateCcw,
  ShieldCheck,
  SkipForward,
  Sparkles,
  Thermometer,
  Volume2,
  VolumeX,
  X,
} from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ApiError, api, number } from "./api";
import { PressureTemperatureChart, ThresholdTracker } from "./WellCharts";
import { SensorGuide } from "./SensorGuide";
import type {
  AgentSource,
  FleetCatalog,
  FleetCheck,
  FleetCheckResult,
  FleetEvent,
  FleetFrame,
  FleetRun,
  FleetTimelineEvent,
  FleetWell,
  WellStatus,
} from "./fleetTypes";
import "./fleet.css";

const SESSION_KEY = "frostline.fleet.session.v2";
const SELECTED_WELL_KEY = "frostline.fleet.selectedWell.v2.";
const DEFAULT_WELL = "WELL-00001";
type OperatorAction =
  "acknowledge" | "observation" | "complete" | "check" | "recheck";
type CheckSubmission = {
  check_id: string;
  check_definition: string;
  result: FleetCheckResult;
  note?: string;
};
type ActionFeedback = {
  wellId: string;
  incidentId: string;
  action: OperatorAction;
  checkId?: string;
  state: "saving" | "saved" | "error";
  message: string;
};
const finished = (status?: string) =>
  ["completed", "cancelled", "failed"].includes(status || "");
const stateNames: Record<WellStatus, string> = {
  normal: "Operating normally",
  watch: "Under observation",
  attention: "Needs operator review",
  unavailable: "Awaiting sensor data",
};
const wellDescriptions: Record<string, { name: string; scenario: string }> = {
  "WELL-00001": { name: "Well 01 — Normal", scenario: "Baseline normal operations" },
  "WELL-00002": { name: "Well 02 — Restriction", scenario: "Choke restriction event" },
  "WELL-00006": { name: "Well 06 — Scaling", scenario: "Scaling in production choke" },
  "WELL-00019": { name: "Well 19 — Hydrate", scenario: "Hydrate forming in production line" },
};
const wellName = (well: FleetWell) =>
  wellDescriptions[well.id]?.name || well.name;
const elapsed = (seconds: number) =>
  `${Math.floor(seconds / 3600)
    .toString()
    .padStart(2, "0")}:${Math.floor((seconds % 3600) / 60)
    .toString()
    .padStart(2, "0")}`;
const time = (value?: string | null) =>
  value
    ? new Date(
        /[zZ]|[+-]\d\d:\d\d$/.test(value) ? value : `${value}Z`,
      ).toLocaleString("en", {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
        hour12: false,
        timeZone: "UTC",
      })
    : "Awaiting readings";
const actionMessages: Record<OperatorAction, string> = {
  acknowledge: "Acknowledged. Concern stays open.",
  observation: "Note saved.",
  check: "Finding saved.",
  complete: "Review recorded.",
  recheck: "Recheck requested.",
};
const historyNames: Record<string, string> = {
  seen: "Seen",
  checked: "Finding saved",
  recheck: "Rechecked",
};
const sensorUnit = (sensor: string) =>
  sensor.startsWith("P-")
    ? "bar"
    : sensor.startsWith("T-")
      ? "°C"
      : sensor === "ABER-CKP"
        ? "%"
        : sensor === "QGL"
          ? "m³/s"
          : "—";
function emptyWell(well: FleetCatalog["wells"][number]): FleetWell {
  return {
    ...well,
    status: "unavailable",
    source_timestamp: null,
    quality: {
      status: "unavailable",
      summary: "Replay has not started.",
      missing: [],
      invalid: [],
      unchanged: [],
    },
    last_assessed: null,
    next_check: null,
    incident: null,
    assessment: null,
    frames: [],
    investigation: "idle",
    activity: "Ready for replay",
  };
}
const fallbackWells = ["00001", "00002", "00006", "00019"].map((id) =>
  emptyWell({ id: `WELL-${id}`, name: `Well ${Number(id)}`, source_file: "" }),
);

function Sheet({
  title,
  onClose,
  children,
}: {
  title: string;
  onClose: () => void;
  children: ReactNode;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const dialog = ref.current!;
    dialog.showModal();
    return () => dialog.close();
  }, []);
  return (
    <dialog
      ref={ref}
      className="fleet-sheet"
      aria-label={title}
      onCancel={onClose}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <header>
        <h2>{title}</h2>
        <button
          className="fleet-button icon"
          aria-label="Close panel"
          onClick={onClose}
        >
          <X size={17} />
        </button>
      </header>
      <div className="fleet-sheet-body">{children}</div>
    </dialog>
  );
}

function WellIcon({ size = 27 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M11 28 15 5h3l4 23M9 28h15M12 21h9M13 15h7M14 9h5M11 6h12M17 2v6"
        stroke="currentColor"
        strokeWidth="1.65"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

const FieldMap = memo(function FieldMap({
  wells,
  selected,
  onSelect,
  run,
}: {
  wells: FleetWell[];
  selected: string;
  onSelect: (id: string) => void;
  run: FleetRun | null;
}) {
  const positions = [
    [34, 29],
    [68, 29],
    [74, 67],
    [28, 72],
  ];
  return (
    <div
      className={`fleet-map${run?.status === "running" ? " fleet-alive" : ""}`}
      aria-label="Illustrative field map with four wells"
    >
      <svg
        className="fleet-map-art"
        viewBox="0 0 800 460"
        preserveAspectRatio="none"
        aria-hidden="true"
      >
        <defs>
          <pattern
            id="fleet-grid"
            width="42"
            height="42"
            patternUnits="userSpaceOnUse"
          >
            <path
              d="M42 0H0V42"
              fill="none"
              stroke="#93b9a8"
              strokeOpacity=".1"
            />
          </pattern>
          <pattern
            id="fleet-land"
            width="11"
            height="11"
            patternUnits="userSpaceOnUse"
          >
            <circle cx="3" cy="3" r=".75" fill="#bcc7a5" opacity=".45" />
          </pattern>
          <radialGradient id="fleet-water">
            <stop stopColor="#e6f0df" />
            <stop offset="1" stopColor="#d1e6dc" />
          </radialGradient>
        </defs>
        <rect width="800" height="460" fill="url(#fleet-water)" />
        <rect width="800" height="460" fill="url(#fleet-grid)" />
        <g fill="none" stroke="#98bfae" strokeWidth="1" opacity=".28">
          <path d="M20 -20C270 20 110 167 370 211S560 389 850 392" />
          <path d="M75 -35C337 6 187 153 399 187S626 327 852 329" />
          <path d="M143 -40C367 38 274 112 459 158S681 244 850 266" />
          <path d="M236 -35C405 55 396 69 537 113S720 177 850 197" />
          <path d="M-20 83C209 173 117 256 319 293S452 426 719 499" />
          <path d="M-40 131C162 254 103 290 242 354S391 446 488 492" />
          <path d="M-10 206C82 284 82 361 177 407S232 462 271 493" />
        </g>
        <path
          d="M0 60H84C157 116 120 153 160 206S118 264 144 298S131 340 167 375S157 413 208 460H0Z"
          fill="#edf0dc"
          stroke="#c0ceac"
          strokeWidth="1.5"
        />
        <path
          d="M0 60H84C157 116 120 153 160 206S118 264 144 298S131 340 167 375S157 413 208 460H0Z"
          fill="url(#fleet-land)"
        />
        <path
          d="M5 124C92 165 54 236 96 292S52 358 127 445"
          fill="none"
          stroke="#c2cdaa"
          strokeWidth="2"
          strokeDasharray="3 8"
        />
        <g stroke="#a4c6b2" strokeWidth="6" fill="none">
          <path d="M95 365H211L344 282" />
          <path d="M344 282 272 133M344 282 544 133M344 282 592 308M344 282 224 331" />
        </g>
        <g stroke="#779f8a" strokeWidth="1.5" strokeDasharray="5 7" fill="none">
          <path d="M95 365H211L344 282" />
          <path d="M344 282 272 133M344 282 544 133M344 282 592 308M344 282 224 331" />
        </g>
        <text
          x="24"
          y="293"
          fill="#91a382"
          fontSize="10"
          letterSpacing="6"
          transform="rotate(-17 24 293)"
        >
          COAST
        </text>
        <text x="594" y="421" fill="#89aa94" fontSize="9" letterSpacing="5">
          OFFSHORE
        </text>
        <g transform="translate(78 343)">
          <rect width="29" height="31" rx="5" fill="#f9fff0" stroke="#9fbaa0" />
          <path
            d="M6 25V14l6-4v7l6-4v12M22 25V7"
            stroke="#71997d"
            strokeWidth="2"
            fill="none"
          />
        </g>
      </svg>
      <div className="fleet-map-top">
        <div className="fleet-map-label">
          <i />
          FIELD OVERVIEW
        </div>
        <div className="fleet-clock">
          <strong>
            {run
              ? `${elapsed(run.elapsed_seconds)} elapsed`
              : "Ready to monitor"}
          </strong>
          {run
            ? `${run.index} / ${run.total} steps · ${run.status}`
            : `${wells.length} well recordings`}
        </div>
      </div>
      <div className="fleet-map-compass">
        <span>N</span>
        <Compass size={19} />
      </div>
      <div className="fleet-platform">
        <svg
          width="65"
          height="63"
          viewBox="0 0 76 72"
          fill="none"
          aria-hidden="true"
        >
          <path
            d="m7 38 31-16 31 16-31 18Z"
            fill="#f8fff0"
            stroke="#6b9580"
            strokeWidth="1.5"
          />
          <path
            d="M7 38v9l31 17 31-17v-9L38 55Z"
            fill="#bad8bf"
            stroke="#6b9580"
            strokeWidth="1.5"
          />
          <path
            d="M18 54v14M39 63v9M59 54v14M31 33l7-28 8 28M28 33h21M33 25h11M35 17h7M49 25V15h12v10"
            stroke="#5d8e76"
            strokeWidth="2.2"
          />
          <path d="M36 30h5" stroke="#5d8e76" strokeWidth="2.2" />
        </svg>
        <span>Production platform</span>
      </div>
      {wells.map((well, index) => {
        const rank = run?.priority.indexOf(well.id) ?? -1;
        return (
          <button
            key={well.id}
            className={`fleet-node ${well.source_timestamp ? well.status : "pending"} ${selected === well.id ? "selected" : ""}`}
            style={{
              left: `${positions[index % 4][0]}%`,
              top: `${positions[index % 4][1]}%`,
            }}
            aria-label={`${well.name}: ${well.source_timestamp ? stateNames[well.status] : "awaiting first check"}. Select well`}
            aria-pressed={selected === well.id}
            onClick={() => onSelect(well.id)}
          >
            <span className="fleet-node-symbol">
              <WellIcon />
            </span>
            {rank >= 0 &&
              well.status !== "normal" &&
              !!well.source_timestamp && (
                <span className="fleet-rank">{rank + 1}</span>
              )}
            <span className="fleet-node-caption">
              <strong>{wellName(well)}</strong>
              <small>
                {well.source_timestamp
                  ? stateNames[well.status]
                  : "Awaiting first check"}
              </small>
            </span>
          </button>
        );
      })}
      <div className="fleet-map-legend">
        <span>
          <i className="fleet-dot" />
          Normal
        </span>
        <span>
          <i className="fleet-dot watch" />
          Watch
        </span>
        <span>
          <i className="fleet-dot attention" />
          Needs review
        </span>
        <span>
          <i className="fleet-dot unavailable" />
          Telemetry
        </span>
      </div>
    </div>
  );
});

const SensorChart = memo(function SensorChart({
  frames,
  sensor,
  title,
  unit,
  icon,
  score = false,
}: {
  frames: FleetFrame[];
  sensor: string | null;
  title: string;
  unit: string;
  icon: ReactNode;
  score?: boolean;
}) {
  const data = useMemo(
    () =>
      frames.slice(-90).map((frame) => ({
        minute: Math.round(frame.elapsed_seconds / 60),
        value: score ? frame.risk_score : sensor ? frame.sensors[sensor] : null,
      })),
    [frames, sensor, score],
  );
  const value = data.at(-1)?.value;
  const available = data.some((row) => row.value != null);
  return (
    <div className="fleet-chart">
      <div className="fleet-chart-label">
        {icon}
        {title}
      </div>
      <div className="fleet-chart-value">
        {number(value, score ? 2 : 1)}
        <small>{unit}</small>
      </div>
      <div className="fleet-chart-name">
        {score ? "Uncalibrated score" : sensor || "Unavailable"}
      </div>
      {available ? (
        <div className="fleet-chart-area">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={data}
              margin={{ top: 8, right: 7, left: 0, bottom: 0 }}
            >
              <CartesianGrid
                vertical={false}
                stroke="#edf1e8"
                strokeDasharray="2 4"
              />
              <XAxis
                dataKey="minute"
                tick={{ fontSize: 11, fill: "#5f7168" }}
                tickFormatter={(x: number) => `${x}m`}
                axisLine={false}
                tickLine={false}
                minTickGap={22}
              />
              <YAxis hide domain={score ? [0, 1] : ["auto", "auto"]} />
              <Tooltip
                content={({ active, payload, label }) =>
                  active && payload?.length ? (
                    <div className="fleet-chart-tooltip">
                      {label} min elapsed
                      <br />
                      <strong>
                        {number(payload[0].value as number, score ? 3 : 2)}{" "}
                        {unit}
                      </strong>
                    </div>
                  ) : null
                }
              />
              <Line
                type="monotone"
                dataKey="value"
                stroke={score ? "#b6a76c" : "#608e72"}
                strokeWidth={1.7}
                dot={false}
                isAnimationActive={false}
                connectNulls={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div className="fleet-chart-empty">
          {frames.length ? "No usable readings" : "Waiting"}
        </div>
      )}
    </div>
  );
});

const classLabels = {
  normal: "Normal",
  hydrate: "Hydrate",
  lookalike: "Look-alike",
} as const;

function ScoreSpark({
  frames,
  threshold,
}: {
  frames: FleetFrame[];
  threshold: number | null;
}) {
  const values = frames.slice(-60).map((frame) => frame.risk_score);
  const step = 120 / Math.max(values.length - 1, 1);
  const y = (v: number) => 26 - v * 24;
  const path = values
    .map((v, i) =>
      v == null ? null : `${i * step},${y(v).toFixed(1)}`,
    )
    .reduce<string[]>((segments, point, i) => {
      if (point == null) return segments;
      const startNew = i === 0 || values[i - 1] == null;
      return startNew
        ? [...segments, `M${point}`]
        : [...segments.slice(0, -1), `${segments.at(-1)} L${point}`];
    }, [])
    .join(" ");
  return (
    <svg
      className="fleet-model-spark"
      viewBox="0 0 120 28"
      preserveAspectRatio="none"
      aria-hidden="true"
    >
      {threshold != null && (
        <line
          x1="0"
          x2="120"
          y1={y(threshold)}
          y2={y(threshold)}
          className="threshold"
        />
      )}
      {path && <path d={path} />}
    </svg>
  );
}

const ModelPredictions = memo(function ModelPredictions({
  wells,
  selected,
  onSelect,
}: {
  wells: FleetWell[];
  selected: string;
  onSelect: (id: string) => void;
}) {
  const model = wells.find((item) => item.prediction)?.prediction;
  const threshold = model?.threshold ?? null;
  return (
    <div className="fleet-card fleet-models">
      <div className="fleet-card-heading">
        <div>
          <h2>Model predictions</h2>
          <small>
            LightGBM
            {threshold != null &&
              ` · alarm ≥ ${threshold.toFixed(2)} for ${model?.persistence_minutes} min`}
          </small>
        </div>
        {model && <span className="fleet-source">{model.model_id}</span>}
      </div>
      <div className="fleet-model-table" role="table" aria-label="Model predictions by well">
        <div className="fleet-model-row head" role="row">
          <span role="columnheader">Well</span>
          {Object.values(classLabels).map((label) => (
            <span key={label} role="columnheader">
              {label}
            </span>
          ))}
          <span role="columnheader">Hydrate · 60 min</span>
          <span role="columnheader">Call</span>
        </div>
        {wells.map((item) => {
          const scores = item.prediction?.scores;
          const top = scores
            ? (Object.keys(classLabels) as (keyof typeof classLabels)[])
                .filter((key) => scores[key] != null)
                .sort((a, b) => scores[b]! - scores[a]!)[0]
            : undefined;
          const streak = item.prediction?.alarm_streak ?? 0;
          const call = !item.prediction
            ? { label: "—", tone: "pending" }
            : item.prediction.alarm_active
              ? { label: "Hydrate alarm", tone: "attention" }
              : streak > 0
                ? {
                    label: `Rising ${streak}/${item.prediction.persistence_minutes}`,
                    tone: "watch",
                  }
                : top
                  ? { label: classLabels[top], tone: top === "normal" ? "normal" : "watch" }
                  : { label: "No data", tone: "unavailable" };
          return (
            <button
              key={item.id}
              role="row"
              className={`fleet-model-row ${selected === item.id ? "selected" : ""}`}
              aria-pressed={selected === item.id}
              onClick={() => onSelect(item.id)}
            >
              <span role="cell" className="fleet-model-well">
                {wellDescriptions[item.id]?.name.split(" — ")[0] || item.name}
              </span>
              {(Object.keys(classLabels) as (keyof typeof classLabels)[]).map(
                (key) => {
                  const value = scores?.[key];
                  return (
                    <span
                      role="cell"
                      key={key}
                      className={`fleet-model-score ${key} ${top === key ? "top" : ""}`}
                    >
                      <strong>{number(value, 2)}</strong>
                      <i>
                        <b style={{ width: `${(value ?? 0) * 100}%` }} />
                        {key === "hydrate" && threshold != null && (
                          <em style={{ left: `${threshold * 100}%` }} />
                        )}
                      </i>
                    </span>
                  );
                },
              )}
              <span role="cell">
                <ScoreSpark frames={item.frames} threshold={threshold} />
              </span>
              <span role="cell" className={`fleet-model-call ${call.tone}`}>
                <i className={`fleet-dot ${call.tone}`} />
                {call.label}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
});

function IncidentProgress({ events }: { events: FleetTimelineEvent[] }) {
  const milestones = [
    ["detected", "Detected"],
    ["seen", "Seen"],
    ["checked", "Checked"],
    ["recheck", "Recheck"],
    ["recovered", "Recovered"],
  ];
  return (
    <div className="fleet-incident-progress">
      <span>INCIDENT</span>
      <ol className="fleet-milestones">
        {milestones.map(([kind, label]) => {
          const event = [...events]
            .reverse()
            .find((item) => item.kind === kind);
          return (
            <li
              key={kind}
              className={event ? "done" : ""}
              title={
                event
                  ? `${event.summary} · ${time(event.at)}`
                  : `${label}: not recorded`
              }
            >
              <i className="fleet-milestone-dot">
                {event && <Check size={8} />}
              </i>
              <strong>{label}</strong>
              <time dateTime={event?.at}>
                {event?.at ? event.at.slice(11, 16) : "—"}
              </time>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

function OperatorCheckTask({
  task,
  busy,
  saving,
  onSave,
}: {
  task: FleetCheck;
  busy: boolean;
  saving: boolean;
  onSave: (submission: CheckSubmission) => void;
}) {
  const [result, setResult] = useState(task.status);
  const [checkNote, setCheckNote] = useState(task.note || "");
  const labels = {
    confirmed: "Confirmed",
    not_confirmed: "Not confirmed",
    unavailable: "Unable to check",
  };
  const changed =
    result !== task.status || checkNote.trim() !== (task.note || "").trim();
  return (
    <div className="fleet-check-task">
      <h4>{task.label}</h4>
      <p>{task.reason}</p>
      <div className="fleet-check-fields">
        <select
          aria-label={`Result: ${task.label}`}
          value={result}
          disabled={busy}
          onChange={(event) =>
            setResult(event.target.value as FleetCheck["status"])
          }
        >
          <option value="pending" disabled>
            Choose a finding
          </option>
          <option value="confirmed">Confirmed</option>
          <option value="not_confirmed">Not confirmed</option>
          <option value="unavailable">Unable to check</option>
        </select>
        <button
          className="fleet-button"
          disabled={busy || result === "pending" || !changed}
          onClick={() => {
            if (result !== "pending")
              onSave({
                check_id: task.id,
                check_definition: task.definition,
                result,
                ...(checkNote.trim() ? { note: checkNote.trim() } : {}),
              });
          }}
        >
          {saving ? (
            <LoaderCircle size={12} className="spin" />
          ) : (
            <Check size={12} />
          )}
          {saving ? "Saving…" : "Save"}
        </button>
      </div>
      <details className="fleet-check-note">
        <summary>Note</summary>
        <input
          aria-label={`Note: ${task.label}`}
          value={checkNote}
          disabled={busy}
          onChange={(event) => setCheckNote(event.target.value)}
          maxLength={500}
          placeholder="What did you find?"
        />
      </details>
      {task.status !== "pending" && (
        <p className="fleet-check-saved" role="status">
          <CheckCheck size={11} /> {labels[task.status]} saved
          {task.updated_at
            ? ` · ${new Date(task.updated_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`
            : ""}
        </p>
      )}
    </div>
  );
}

export default function FleetDashboard() {
  const [catalog, setCatalog] = useState<FleetCatalog | null>(null);
  const [run, setRun] = useState<FleetRun | null>(null);
  const [sessionId, setSessionId] = useState(() =>
    localStorage.getItem(SESSION_KEY),
  );
  const [selected, setSelected] = useState(() => {
    const saved = sessionId
      ? localStorage.getItem(`${SELECTED_WELL_KEY}${sessionId}`)
      : null;
    return fallbackWells.some((well) => well.id === saved)
      ? saved!
      : DEFAULT_WELL;
  });
  const [speed, setSpeed] = useState(12);
  const [useLlm, setUseLlm] = useState(false);
  const [connected, setConnected] = useState(false);
  const [busy, setBusy] = useState(false);
  const [starting, setStarting] = useState(false);
  const [actionFeedback, setActionFeedback] = useState<ActionFeedback | null>(
    null,
  );
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [dismissedRunError, setDismissedRunError] = useState("");
  const [evidenceOpen, setEvidenceOpen] = useState(false);
  const [observing, setObserving] = useState(false);
  const [note, setNote] = useState("");
  const [chartEntering, setChartEntering] = useState(false);
  const [voiceMuted, setVoiceMuted] = useState(false);
  const [voicePlaying, setVoicePlaying] = useState(false);
  const [listening, setListening] = useState(false);
  const voiceRef = useRef<HTMLAudioElement | null>(null);
  const spokenIncidents = useRef<Set<string>>(new Set());
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const recognitionRef = useRef<any>(null);
  const lastEvent = useRef(-1);
  const source = useRef<EventSource | null>(null);
  const selectedRef = useRef(selected);
  selectedRef.current = selected;
  const adopt = (next: FleetRun) => {
    lastEvent.current = next.last_event_id;
    setRun(next);
    setSpeed(next.speed);
    setUseLlm(next.use_llm);
  };
  useEffect(() => {
    let disposed = false;
    let retry: ReturnType<typeof setTimeout> | undefined;
    const load = () => {
      api<FleetCatalog>("/fleet/catalog")
        .then((value) => {
          if (disposed) return;
          setCatalog(value);
          setError("");
          // A reconnecting session owns its speed.
          if (!sessionId) setSpeed(value.default_speed);
        })
        .catch((e) => {
          if (disposed) return;
          // The API may still be starting; keep trying.
          setError(`${e.message} Retrying…`);
          retry = setTimeout(load, 3000);
        });
    };
    load();
    return () => {
      disposed = true;
      clearTimeout(retry);
    };
  }, []);
  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(""), 6000);
    return () => clearTimeout(timer);
  }, [notice]);
  useEffect(() => {
    if (sessionId && fallbackWells.some((well) => well.id === selected))
      localStorage.setItem(`${SELECTED_WELL_KEY}${sessionId}`, selected);
  }, [sessionId, selected]);
  useEffect(() => {
    setObserving(false);
    setNote("");
    setChartEntering(true);
    const timer = setTimeout(() => setChartEntering(false), 400);
    return () => clearTimeout(timer);
  }, [selected]);
  // Voice alert: auto-play TTS when a well transitions to attention
  useEffect(() => {
    if (voiceMuted || !run) return;
    for (const w of run.wells) {
      if (w.status === "attention" && w.incident && w.assessment?.summary && !spokenIncidents.current.has(w.incident.id)) {
        spokenIncidents.current.add(w.incident.id);
        const text = `Alert on ${w.name}. ${w.assessment.summary}`;
        fetch("/api/fleet/tts", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: text.slice(0, 500) }),
        })
          .then((r) => (r.ok ? r.blob() : null))
          .then((blob) => {
            if (!blob) return;
            const url = URL.createObjectURL(blob);
            const audio = new Audio(url);
            voiceRef.current = audio;
            setVoicePlaying(true);
            audio.onended = () => {
              setVoicePlaying(false);
              URL.revokeObjectURL(url);
            };
            audio.play().catch(() => setVoicePlaying(false));
          })
          .catch(() => {});
        break;
      }
    }
  }, [run?.wells, voiceMuted]);
  useEffect(() => {
    if (!sessionId) return;
    let disposed = false;
    let reopen: ReturnType<typeof setTimeout> | undefined;
    lastEvent.current = -1;
    const apply = (next: FleetRun) => {
      if (!disposed && next.last_event_id >= lastEvent.current) adopt(next);
    };
    const resetMissing = () => {
      if (disposed) return;
      source.current?.close();
      localStorage.removeItem(SESSION_KEY);
      localStorage.removeItem(`${SELECTED_WELL_KEY}${sessionId}`);
      setSessionId(null);
      setSelected(DEFAULT_WELL);
      setRun(null);
      setConnected(false);
      setNotice("Server restarted. Start a new replay.");
    };
    const open = (after: number) => {
      if (disposed) return;
      const stream = new EventSource(
        `/api/fleet/sessions/${sessionId}/events?after=${after}`,
      );
      source.current = stream;
      stream.onopen = () => {
        if (!disposed) setConnected(true);
      };
      stream.addEventListener("fleet", (message) => {
        if (disposed) return;
        try {
          const event = JSON.parse(
            (message as MessageEvent).data,
          ) as FleetEvent;
          if (event.type !== "snapshot") return;
          apply(event.payload);
          if (finished(event.payload.status)) {
            stream.close();
            setConnected(false);
          }
        } catch {
          setError("A live update could not be read.");
        }
      });
      stream.onerror = () => {
        if (disposed) return;
        setConnected(false);
        // A non-200 response closes an EventSource for good, so reopen it
        // ourselves; while it is still CONNECTING the browser retries.
        const closed = stream.readyState === EventSource.CLOSED;
        const retry = () => {
          if (disposed || !closed) return;
          stream.close();
          clearTimeout(reopen);
          reopen = setTimeout(() => open(lastEvent.current), 2000);
        };
        api<FleetRun>(`/fleet/sessions/${sessionId}`)
          .then((snapshot) => {
            apply(snapshot);
            if (!finished(snapshot.status)) retry();
          })
          .catch((e) => {
            if (e instanceof ApiError && e.status === 404) resetMissing();
            else retry();
          });
      };
    };
    api<FleetRun>(`/fleet/sessions/${sessionId}`)
      .then((snapshot) => {
        if (disposed) return;
        apply(snapshot);
        if (!finished(snapshot.status)) open(snapshot.last_event_id);
      })
      .catch((e) => {
        if (disposed) return;
        if (e instanceof ApiError && e.status === 404) resetMissing();
        else setError(e.message);
      });
    return () => {
      disposed = true;
      clearTimeout(reopen);
      source.current?.close();
      setConnected(false);
    };
  }, [sessionId]);
  const accept = (next: FleetRun) => {
    if (next.id !== run?.id || next.last_event_id >= lastEvent.current)
      adopt(next);
  };
  async function command(action: string, extra: object = {}) {
    if (!sessionId || busy) return;
    setBusy(true);
    setError("");
    try {
      accept(
        await api<FleetRun>(`/fleet/sessions/${sessionId}/control`, {
          action,
          ...extra,
        }),
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function start() {
    if (busy) return;
    setBusy(true);
    setStarting(true);
    setActionFeedback(null);
    setError("");
    try {
      // Apply the cancelled snapshot so a failed create does not leave a dead
      // run on screen that still looks paused.
      if (run && !finished(run.status))
        accept(
          await api<FleetRun>(`/fleet/sessions/${run.id}/control`, {
            action: "cancel",
          }),
        );
      source.current?.close();
      const next = await api<FleetRun>("/fleet/sessions", {
        speed,
        use_llm: useLlm,
      });
      adopt(next);
      setSelected(DEFAULT_WELL);
      setSessionId(next.id);
      localStorage.setItem(SESSION_KEY, next.id);
      setNotice("");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
      setStarting(false);
    }
  }
  async function incidentAction(
    action: OperatorAction,
    check?: CheckSubmission,
  ) {
    if (!sessionId || busy) return;
    const wellId = selected;
    const incidentId = run?.wells.find((item) => item.id === wellId)?.incident
      ?.id;
    if (!incidentId) return;
    const target = { wellId, incidentId, action, checkId: check?.check_id };
    setBusy(true);
    setError("");
    setActionFeedback({ ...target, state: "saving", message: "Saving…" });
    try {
      accept(
        await api<FleetRun>(
          `/fleet/sessions/${sessionId}/incidents/${wellId}/actions`,
          {
            action,
            incident_id: incidentId,
            ...(check ||
              (["observation", "complete"].includes(action) && note.trim()
                ? { note: note.trim() }
                : {})),
          },
        ),
      );
      setActionFeedback({
        ...target,
        state: "saved",
        message: actionMessages[action],
      });
      if (
        selectedRef.current === wellId &&
        ["observation", "complete", "acknowledge"].includes(action)
      ) {
        setObserving(false);
        setNote("");
      }
    } catch (e) {
      setActionFeedback({
        ...target,
        state: "error",
        message: `Not saved. ${(e as Error).message}`,
      });
    } finally {
      setBusy(false);
    }
  }
  const wells = useMemo(
    () =>
      run?.wells.length
        ? run.wells
        : catalog?.wells.map(emptyWell) || fallbackWells,
    [run, catalog],
  );
  const well = wells.find((item) => item.id === selected) || wells[0];
  const fleet = useMemo(() => {
    const order = run?.priority || [];
    const rank = (id: string) => {
      const index = order.indexOf(id);
      return index < 0 ? 99 : index;
    };
    const count = (test: (item: FleetWell) => unknown) =>
      wells.filter(test).length;
    return {
      queue: [...wells].sort((a, b) => rank(a.id) - rank(b.id)),
      attention: count((w) => w.frames.length && w.status === "attention"),
      unseen: count(
        (w) =>
          w.incident &&
          !w.incident.acknowledged &&
          !w.incident.completed &&
          !w.incident.condition_cleared,
      ),
      reliable: count((w) => w.frames.length && w.quality.status === "good"),
      pending: count((w) => ["queued", "running"].includes(w.investigation)),
      started: wells.some((w) => w.source_timestamp),
    };
  }, [wells, run?.priority]);
  const [pressure, temperature] = useMemo(() => {
    const has = (key: string) =>
      well.frames.some((frame) => frame.sensors[key] != null);
    return [
      ["P-TPT", "P-MON-CKP", "P-PDG"].find(has) || null,
      ["T-TPT", "T-PDG", "T-JUS-CKP"].find(has) || null,
    ];
  }, [well.frames]);
  const active = !!run && !finished(run.status);
  const running = run?.status === "running";
  const paused = run?.status === "paused";
  const preparing = run?.status === "preparing";
  const investigating =
    active && connected && well.investigation === "running";
  const incident = well.incident;
  const currentFeedback =
    actionFeedback?.wellId === well.id &&
    actionFeedback.incidentId === incident?.id
      ? actionFeedback
      : null;
  const savingAction =
    currentFeedback?.state === "saving" ? currentFeedback.action : null;
  const reviewState = !incident
    ? null
    : incident.condition_cleared
      ? "Recovered"
      : incident.completed
        ? "Review recorded"
        : incident.acknowledged
          ? "Acknowledged"
          : null;
  const nextMinutes =
    well.next_check && well.source_timestamp
      ? Math.max(
          0,
          Math.ceil(
            (Date.parse(well.next_check) - Date.parse(well.source_timestamp)) /
              60_000,
          ),
        )
      : NaN;
  const [monitorTitle, monitorDetail] = !run
    ? ["Ready", ""]
    : finished(run.status)
      ? [run.status === "failed" ? "Replay interrupted" : "Replay ended", ""]
      : preparing
        ? ["Loading recordings", ""]
        : !connected
          ? ["Reconnecting…", ""]
          : well.investigation === "running"
            ? ["Reviewing evidence", paused ? "" : well.activity]
            : well.investigation === "queued"
              ? [paused ? "Paused" : "Review queued", paused ? "Review queued." : ""]
              : paused
                ? ["Paused", ""]
                : [
                    "Monitoring",
                    Number.isFinite(nextMinutes)
                      ? `Next review in ${nextMinutes || 1} min`
                      : "",
                  ];
  const llmConfigured =
    run?.llm_configured ?? catalog?.llm_configured ?? false;
  const provider = (run?.llm_providers ?? catalog?.llm_providers ?? [])[0];
  const readyModel = run?.model_ready ?? catalog?.model_ready ?? false;
  const agentSource: AgentSource =
    well.assessment?.source || run?.agent_mode || (useLlm ? "live" : "rules");
  const sourceLabel =
    agentSource === "live"
      ? `LLM${provider ? ` · ${provider}` : ""}`
      : agentSource === "recorded"
        ? "Recorded"
        : "Model + rules";
  const runError =
    run?.error && run.error !== dismissedRunError ? run.error : "";
  const headerStatus = preparing
    ? "Loading…"
    : paused
      ? "Paused"
      : running
        ? `Monitoring ${wells.length} wells`
        : run?.status === "failed"
          ? "Replay interrupted"
          : finished(run?.status)
            ? "Replay complete"
            : "Ready";
  const wellStatus = well.source_timestamp ? well.status : "pending";
  return (
    <section
      className="fleet-dashboard"
      aria-label="Four-well operations dashboard"
    >
      <header className="fleet-header">
        <div>
          <h1>{headerStatus}</h1>
        </div>
        <div className="fleet-controls">
          <div
            className="fleet-mode-picker"
            role="group"
            aria-label="Assessment engine"
            title={llmConfigured ? undefined : "No provider key"}
          >
            {[false, true].map((value) => (
              <button
                key={String(value)}
                aria-pressed={useLlm === value}
                disabled={busy || preparing || (value && !llmConfigured)}
                title={value && !llmConfigured ? "No provider key" : undefined}
                onClick={() => {
                  if (useLlm === value) return;
                  if (active) void command("llm", { use_llm: value });
                  else setUseLlm(value);
                }}
              >
                {value ? "LLM" : "Rules"}
              </button>
            ))}
          </div>
          <select
            aria-label="Replay speed"
            value={speed}
            disabled={busy || preparing}
            onChange={(event) => {
              const value = Number(event.target.value);
              if (active) void command("speed", { speed: value });
              else setSpeed(value);
            }}
          >
            {(catalog?.speeds || [12, 30, 60, 120]).map((value) => (
              <option key={value} value={value}>
                {value}×
              </option>
            ))}
          </select>
          {!active ? (
            <button
              className="fleet-button primary"
              onClick={() => void start()}
              disabled={busy || !catalog}
            >
              {starting ? (
                <LoaderCircle size={15} className="spin" />
              ) : (
                <Play size={15} />
              )}
              Start
            </button>
          ) : (
            <button
              className="fleet-button primary"
              disabled={busy || preparing}
              onClick={() => void command(running ? "pause" : "resume")}
            >
              {preparing ? (
                <LoaderCircle size={15} className="spin" />
              ) : running ? (
                <Pause size={15} />
              ) : (
                <Play size={15} />
              )}
              {preparing ? "Preparing…" : running ? "Pause" : "Resume"}
            </button>
          )}
          <button
            className="fleet-button"
            title="Advance one minute"
            disabled={busy || !paused}
            onClick={() => void command("step")}
          >
            <SkipForward size={15} />
            +1 min
          </button>
          <button
            className={`fleet-button icon${voiceMuted ? "" : " active"}`}
            aria-label={voiceMuted ? "Unmute voice alerts" : "Mute voice alerts"}
            title={voiceMuted ? "Unmute voice alerts" : "Mute voice alerts"}
            onClick={() => {
              setVoiceMuted(!voiceMuted);
              if (!voiceMuted && voiceRef.current) {
                voiceRef.current.pause();
                setVoicePlaying(false);
              }
            }}
            style={voiceMuted ? undefined : { background: "#1a7465", color: "#fff", borderColor: "#1a7465" }}
          >
            {voiceMuted ? <VolumeX size={15} /> : <Volume2 size={15} />}
          </button>
          {run && (
            <button
              className="fleet-button icon"
              aria-label="Restart replay"
              title="Restart replay"
              disabled={busy || preparing}
              onClick={() => {
                if (
                  finished(run.status) ||
                  window.confirm("Restart? The current run will be discarded.")
                )
                  void start();
              }}
            >
              <RotateCcw size={15} />
            </button>
          )}
        </div>
      </header>
      {(error || runError) && (
        <div className="fleet-notice error" role="alert">
          <CircleAlert size={16} />
          <span>{error || runError}</span>
          <button
            aria-label="Dismiss error"
            onClick={() => {
              setError("");
              setDismissedRunError(run?.error || "");
            }}
          >
            <X size={15} />
          </button>
        </div>
      )}
      {notice && (
        <div className="fleet-notice" role="status">
          <Check size={16} />
          <span>{notice}</span>
          <button aria-label="Dismiss notice" onClick={() => setNotice("")}>
            <X size={15} />
          </button>
        </div>
      )}
      <div className="fleet-layout">
        <div className="fleet-main">
          <FieldMap
            wells={wells}
            selected={well.id}
            onSelect={setSelected}
            run={run}
          />
          <div className="fleet-card fleet-trends">
            <div className="fleet-card-heading">
              <div>
                <h2>{wellName(well)}</h2>
                <small>
                  {well.source_timestamp
                    ? `${wellDescriptions[well.id]?.scenario || "Recording replay"} · ${time(well.source_timestamp)}`
                    : well.frames.length
                      ? "Warm-up history"
                      : "Select a well"}
                </small>
              </div>
              <button
                className="fleet-text-button"
                onClick={() => setEvidenceOpen(true)}
              >
                Evidence
                <ArrowRight size={13} />
              </button>
            </div>
            <div className={`fleet-charts${chartEntering ? " fleet-entering" : ""}`}>
              <SensorChart
                frames={well.frames}
                sensor={pressure}
                title="Pressure"
                unit="bar"
                icon={<Gauge size={12} />}
              />
              <SensorChart
                frames={well.frames}
                sensor={temperature}
                title="Temperature"
                unit="°C"
                icon={<Thermometer size={12} />}
              />
              <SensorChart
                frames={well.frames}
                sensor={null}
                title="Hydrate model"
                unit="/ 1"
                icon={<Activity size={12} />}
                score
              />
            </div>
          </div>
          <PressureTemperatureChart frames={well.frames} />
          <SensorGuide sensors={catalog?.sensors ?? []} insights={well.insights} />
          <ThresholdTracker frames={well.frames} prediction={well.prediction} />
          <ModelPredictions
            wells={wells}
            selected={well.id}
            onSelect={setSelected}
          />
        </div>
        <aside
          className="fleet-rail"
          aria-label="Agent priorities and assessment"
        >
          <div className="fleet-card">
            <div className="fleet-card-heading">
              <div>
                <h2>Priority</h2>
              </div>
              <span className="fleet-source">{wells.length} wells</span>
            </div>
            <ol className="fleet-queue">
              {fleet.queue.map((item, index) => (
                <li key={item.id}>
                  <button
                    className={`fleet-queue-item ${well.id === item.id ? "selected" : ""}`}
                    aria-pressed={well.id === item.id}
                    onClick={() => setSelected(item.id)}
                  >
                    <span className="fleet-queue-rank">
                      {String(index + 1).padStart(2, "0")}
                    </span>
                    <i
                      className={`fleet-dot ${item.source_timestamp ? item.status : "pending"}`}
                    />
                    <span className="fleet-queue-copy">
                      <strong>{item.name}</strong>
                      <small>
                        {item.source_timestamp
                          ? stateNames[item.status]
                          : "Waiting"}
                        {item.incident?.acknowledged &&
                        !item.incident.completed &&
                        !item.incident.condition_cleared
                          ? " · acknowledged"
                          : ""}
                      </small>
                    </span>
                    <span className="fleet-queue-state">
                      {active &&
                      connected &&
                      item.investigation === "running" ? (
                        <LoaderCircle size={13} className="spin" />
                      ) : (
                        <ChevronRight size={13} />
                      )}
                    </span>
                  </button>
                </li>
              ))}
            </ol>
          </div>
          <div className="fleet-card">
            <div className="fleet-assessment">
              <div className="fleet-assessment-well">
                <Sparkles size={12} />
                {wellName(well)}
              </div>
              <span className={`fleet-status ${wellStatus}`}>
                <i className={`fleet-dot ${wellStatus}`} />
                {well.source_timestamp ? stateNames[well.status] : "Waiting"}
              </span>
              <h3>
                {well.assessment?.summary ||
                  (investigating ? "Investigating…" : "Waiting for readings")}
              </h3>
              {voicePlaying && selected === well.id && (
                <span className="fleet-speaker-icon">
                  <Volume2 size={13} /> Speaking
                </span>
              )}
              {well.assessment?.evidence.length ? (
                <ul className="fleet-evidence-list">
                  {well.assessment.evidence.slice(0, 2).map((item, index) => (
                    <li key={index}>
                      <ArrowDownRight size={13} />
                      {item}
                    </li>
                  ))}
                </ul>
              ) : well.frames.length && well.activity ? (
                <p>{well.activity}</p>
              ) : null}
              {!!well.timeline?.length && (
                <IncidentProgress events={well.timeline} />
              )}
              <div className="fleet-next-step">
                <p>{well.assessment?.next_step || "None"}</p>
                {incident && (
                  <div className="fleet-actions" aria-busy={!!savingAction}>
                    <button
                      className="fleet-button primary"
                      disabled={busy || !!reviewState}
                      aria-describedby={`review-help-${well.id}`}
                      onClick={() => void incidentAction("acknowledge")}
                    >
                      {savingAction === "acknowledge" ? (
                        <LoaderCircle size={13} className="spin" />
                      ) : reviewState ? (
                        <CheckCheck size={13} />
                      ) : (
                        <Check size={13} />
                      )}
                      {savingAction === "acknowledge"
                        ? "Saving…"
                        : reviewState ||
                          (currentFeedback?.state === "error" &&
                          currentFeedback.action === "acknowledge"
                            ? "Retry"
                            : "Acknowledge")}
                    </button>
                    <button
                      className={`fleet-button ${reviewState ? "primary" : ""}`}
                      disabled={busy}
                      onClick={() => setObserving(!observing)}
                    >
                      <MessageSquare size={12} />
                      Note
                    </button>
                  </div>
                )}
                {incident && reviewState && !savingAction && (
                  <div
                    id={`review-help-${well.id}`}
                    className="fleet-review-feedback saved"
                    role="status"
                    aria-live="polite"
                    aria-atomic="true"
                  >
                    <strong>
                      <CheckCheck size={13} /> {reviewState}
                    </strong>
                    {!incident.condition_cleared && (
                      <span>Open until readings recover.</span>
                    )}
                  </div>
                )}
                {currentFeedback?.state === "error" && (
                  <p className="fleet-action-error" role="alert">
                    {currentFeedback.message}
                  </p>
                )}
                {currentFeedback?.state === "saved" &&
                  ["observation", "check"].includes(currentFeedback.action) && (
                    <p className="fleet-action-saved" role="status">
                      {currentFeedback.message}
                    </p>
                  )}
                {observing && (
                  <div className="fleet-observation">
                    <label htmlFor="fleet-observation">Observation</label>
                    <textarea
                      id="fleet-observation"
                      value={note}
                      onChange={(event) => setNote(event.target.value)}
                      maxLength={500}
                      placeholder="What did you see?"
                    />
                    <div className="fleet-voice-controls">
                      <button
                        className={listening ? "recording" : ""}
                        title={listening ? "Stop listening" : "Speak to add observation"}
                        onClick={() => {
                          if (listening) {
                            // eslint-disable-next-line @typescript-eslint/no-explicit-any
                            (recognitionRef.current as any)?.stop();
                            setListening(false);
                            return;
                          }
                          // eslint-disable-next-line @typescript-eslint/no-explicit-any
                          const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
                          if (!SR) return;
                          // eslint-disable-next-line @typescript-eslint/no-explicit-any
                          const recognition: any = new SR();
                          recognitionRef.current = recognition;
                          recognition.continuous = false;
                          recognition.interimResults = false;
                          // eslint-disable-next-line @typescript-eslint/no-explicit-any
                          recognition.onresult = (event: any) => {
                            const transcript: string = event.results?.[0]?.[0]?.transcript || "";
                            setNote((prev) => (prev ? `${prev} ${transcript}` : transcript));
                          };
                          recognition.onend = () => setListening(false);
                          recognition.onerror = () => setListening(false);
                          recognition.start();
                          setListening(true);
                        }}
                      >
                        <Mic size={13} />
                        {listening ? "Listening…" : "Speak"}
                      </button>
                    </div>
                    <div className="fleet-actions">
                      <button
                        className="fleet-button primary"
                        disabled={busy || !note.trim()}
                        onClick={() => void incidentAction("observation")}
                      >
                        {savingAction === "observation" && (
                          <LoaderCircle size={12} className="spin" />
                        )}
                        {savingAction === "observation" ? "Saving…" : "Save note"}
                      </button>
                      <button
                        className="fleet-button"
                        disabled={busy || !!incident?.completed}
                        onClick={() => void incidentAction("complete")}
                      >
                        {savingAction === "complete" && (
                          <LoaderCircle size={12} className="spin" />
                        )}
                        {savingAction === "complete"
                          ? "Saving…"
                          : "Complete review"}
                      </button>
                    </div>
                  </div>
                )}
                {!!incident && !!well.assessment?.checks?.length && (
                  <details
                    className="fleet-checklist"
                    key={`${well.id}-${incident.id}`}
                  >
                    <summary>
                      Operator checks{" "}
                      <span>
                        {
                          well.assessment.checks.filter(
                            (task) => task.status !== "pending",
                          ).length
                        }{" "}
                        / {well.assessment.checks.length}
                      </span>
                    </summary>
                    <div className="fleet-checklist-body">
                      {well.assessment.checks.map((task) => (
                        <OperatorCheckTask
                          key={`${task.id}-${task.definition}-${task.updated_at || "pending"}`}
                          task={task}
                          busy={busy}
                          saving={
                            savingAction === "check" &&
                            currentFeedback?.checkId === task.id
                          }
                          onSave={(submission) =>
                            void incidentAction("check", submission)
                          }
                        />
                      ))}
                    </div>
                  </details>
                )}
                <div
                  className={`fleet-follow-up ${paused ? "paused" : ""}`}
                  aria-label="Monitoring status"
                >
                  <strong>
                    {investigating ? (
                      <LoaderCircle size={14} className="spin" />
                    ) : paused ? (
                      <Pause size={14} />
                    ) : (
                      <Clock3 size={14} />
                    )}
                    {monitorTitle}
                  </strong>
                  {monitorDetail && <p>{monitorDetail}</p>}
                  {well.followup?.last_checked_at && (
                    <div className="fleet-followup-change">
                      <p>{well.followup.change_summary}</p>
                      <small>{time(well.followup.last_checked_at)}</small>
                    </div>
                  )}
                  <div className="fleet-followup-actions">
                    {paused && (
                      <button
                        className="fleet-button"
                        disabled={busy}
                        onClick={() => void command("resume")}
                      >
                        <Play size={12} /> Resume
                      </button>
                    )}
                    {incident && (
                      <button
                        className="fleet-button"
                        disabled={
                          busy ||
                          !active ||
                          preparing ||
                          !connected ||
                          ["running", "queued"].includes(well.investigation)
                        }
                        title={
                          !active
                            ? "Start a new replay first"
                            : paused
                              ? "Runs on resume"
                              : "Review the evidence again"
                        }
                        onClick={() => void incidentAction("recheck")}
                      >
                        {savingAction === "recheck" ? (
                          <LoaderCircle size={12} className="spin" />
                        ) : (
                          <RotateCcw size={12} />
                        )}
                        {savingAction === "recheck"
                          ? "Requesting…"
                          : paused
                            ? "Queue recheck"
                            : "Recheck"}
                      </button>
                    )}
                  </div>
                  {currentFeedback?.state === "saved" &&
                    currentFeedback.action === "recheck" && (
                      <p className="fleet-action-saved" role="status">
                        {currentFeedback.message}
                      </p>
                    )}
                </div>
              </div>
              <div className="fleet-agent-status">
                <span className={`fleet-source ${agentSource}`}>
                  <span
                    className={`fleet-dot ${agentSource === "live" ? "normal" : "unavailable"}`}
                  />
                  {sourceLabel}
                </span>
                <button
                  className="fleet-text-button"
                  onClick={() => setEvidenceOpen(true)}
                >
                  Why?
                  <ArrowRight size={12} />
                </button>
              </div>
            </div>
          </div>
          <div className="fleet-mode-notice">
            {readyModel ? (
              <ShieldCheck size={13} />
            ) : (
              <LoaderCircle size={13} />
            )}
            <span>
              {readyModel
                ? "Advisory only"
                : run?.readiness_message ||
                  catalog?.readiness_message ||
                  "Loading model…"}
            </span>
          </div>
        </aside>
      </div>
      <div className="fleet-footnote">
        <span>
          <i
            className={`fleet-dot ${running && connected ? "normal" : "unavailable"}`}
          />
          {preparing
            ? run?.readiness_message
            : run
              ? `${paused ? "Paused" : finished(run.status) ? "Finished" : connected ? "Connected" : "Connecting…"}${useLlm ? ` · ${run.requests_used}/${run.request_budget} LLM calls` : ""}`
              : "Historical 3W replay"}
        </span>
        {run && (
          <a
            className="fleet-text-button"
            href={`/api/fleet/sessions/${run.id}/export`}
            download
          >
            <Download size={12} />
            Export run
          </a>
        )}
      </div>
      {evidenceOpen && (
        <EvidenceSheet
          well={well}
          elapsedSeconds={run?.elapsed_seconds || 0}
          sourceLabel={sourceLabel}
          onClose={() => setEvidenceOpen(false)}
        />
      )}
    </section>
  );
}

function EvidenceSheet({
  well,
  elapsedSeconds,
  sourceLabel,
  onClose,
}: {
  well: FleetWell;
  elapsedSeconds: number;
  sourceLabel: string;
  onClose: () => void;
}) {
  const { assessment, followup, incident, timeline } = well;
  const latest = well.frames.at(-1);
  return (
    <Sheet title={`${well.name} · Evidence`} onClose={onClose}>
      <section>
        <h3>Assessment</h3>
        <p>{assessment?.summary || "Not assessed yet."}</p>
        {assessment?.uncertainty && (
          <p style={{ marginTop: 8 }}>{assessment.uncertainty}</p>
        )}
        <ul className="fleet-evidence-list">
          {assessment?.evidence.map((item, index) => (
            <li key={index}>
              <ArrowDownRight size={13} />
              {item}
            </li>
          ))}
        </ul>
        {(assessment?.category || !!assessment?.alternatives?.length) && (
          <details className="fleet-sensor-details">
            <summary>Context</summary>
            {assessment.category && (
              <p>Hypothesis: {assessment.category.replaceAll("_", " ")}</p>
            )}
            {!!assessment.alternatives?.length && (
              <ul className="fleet-evidence-list">
                {assessment.alternatives.map((alternative, index) => (
                  <li key={index}>{alternative}</li>
                ))}
              </ul>
            )}
          </details>
        )}
      </section>
      {followup?.after && (
        <section>
          <h3>Latest check</h3>
          <p>{followup.change_summary}</p>
          <p>Trigger: {followup.trigger || "Scheduled"}</p>
          <div className="fleet-evidence-change">
            {(
              [
                ["Before", followup.before],
                ["After", followup.after],
              ] as const
            ).map(([label, snapshot]) => (
              <div key={label}>
                <span>{label}</span>
                {snapshot ? (
                  <>
                    <time dateTime={snapshot.at}>{snapshot.at}</time>
                    <strong>{stateNames[snapshot.status]}</strong>
                    <p>{snapshot.summary}</p>
                    <ul>
                      {snapshot.evidence.map((item, index) => (
                        <li key={index}>{item}</li>
                      ))}
                    </ul>
                  </>
                ) : (
                  <p>None earlier.</p>
                )}
              </div>
            ))}
          </div>
        </section>
      )}
      {!!timeline?.length && (
        <section>
          <details className="fleet-sensor-details">
            <summary>Incident history · {timeline.length}</summary>
            <ol className="fleet-history-list">
              {timeline.map((event) => (
                <li key={event.id}>
                  <strong>
                    {historyNames[event.kind] || event.kind.replaceAll("_", " ")}
                  </strong>
                  <time dateTime={event.at}>{event.at}</time>
                  <p>{event.summary}</p>
                </li>
              ))}
            </ol>
          </details>
        </section>
      )}
      <section>
        <h3>Provenance</h3>
        <dl>
          <dt>Source</dt>
          <dd>{well.source_file || "—"}</dd>
          <dt>Source time</dt>
          <dd>{well.source_timestamp || "—"}</dd>
          <dt>Replay time</dt>
          <dd>{elapsed(elapsedSeconds)}</dd>
          <dt>Sensor health</dt>
          <dd>{well.quality.summary}</dd>
          <dt>Assessed by</dt>
          <dd>{sourceLabel}</dd>
          <dt>Last assessed</dt>
          <dd>{well.last_assessed || "—"}</dd>
          <dt>Next check</dt>
          <dd>{well.next_check || "—"}</dd>
        </dl>
      </section>
      <section>
        <details className="fleet-sensor-details">
          <summary>Sensor readings</summary>
          {latest ? (
            <>
              <p>{latest.t}</p>
              <table className="fleet-sensor-table">
                <thead>
                  <tr>
                    <th>Sensor</th>
                    <th>Reading</th>
                    <th>Unit</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(latest.sensors).map(([sensor, value]) => (
                    <tr key={sensor}>
                      <td>{sensor}</td>
                      <td>
                        {value == null || !Number.isFinite(value)
                          ? "—"
                          : value.toLocaleString("en", {
                              maximumSignificantDigits: 6,
                            })}
                      </td>
                      <td>{sensorUnit(sensor)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p>QGL = gas-lift injection.</p>
            </>
          ) : (
            <p>No readings yet.</p>
          )}
        </details>
      </section>
      <section>
        <h3>Steps</h3>
        {assessment?.tools.length ? (
          <ol className="fleet-trace">
            {assessment.tools.map((tool, index) => (
              <li key={index}>
                <strong>
                  {tool.status === "running" ? (
                    <LoaderCircle size={12} className="spin" />
                  ) : tool.status === "failed" ? (
                    <CircleAlert size={12} />
                  ) : (
                    <Check size={12} />
                  )}{" "}
                  {tool.name.replaceAll("_", " ")}
                </strong>
                {tool.summary && <p>{tool.summary}</p>}
                {!!tool.sources?.length && (
                  <small>Sources: {tool.sources.join(" · ")}</small>
                )}
              </li>
            ))}
          </ol>
        ) : (
          <p>{well.activity || "None yet."}</p>
        )}
        {!!assessment?.playbook_refs?.length && (
          <p className="fleet-playbook-refs">
            <small>Playbook: {assessment.playbook_refs.join(" · ")}</small>
          </p>
        )}
      </section>
      {incident && (
        <section>
          <h3>Operator record</h3>
          <dl>
            <dt>Incident</dt>
            <dd>{incident.id}</dd>
            <dt>Seen</dt>
            <dd>{incident.acknowledged ? "Acknowledged" : "Not yet"}</dd>
            <dt>Review</dt>
            <dd>
              {incident.condition_cleared
                ? "Recovered"
                : incident.completed
                  ? "Complete"
                  : "Open"}
            </dd>
            {incident.note && (
              <>
                <dt>Observation</dt>
                <dd>{incident.note}</dd>
              </>
            )}
          </dl>
        </section>
      )}
    </Sheet>
  );
}

