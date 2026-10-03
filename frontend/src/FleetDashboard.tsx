import { useEffect, useRef, useState } from "react";
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
  Pause,
  Play,
  Radio,
  RotateCcw,
  ShieldCheck,
  SkipForward,
  Sparkles,
  Thermometer,
  Unplug,
  Waves,
  X,
  Zap,
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
import type {
  AgentSource,
  FleetCatalog,
  FleetCheck,
  FleetCheckResult,
  FleetEvent,
  FleetFrame,
  FleetMode,
  FleetRun,
  FleetTimelineEvent,
  FleetWell,
  FleetWorkflowReport,
  WellStatus,
} from "./fleetTypes";
import "./fleet.css";

const SESSION_KEY = "frostline.fleet.session.v1";
const SELECTED_WELL_KEY = "frostline.fleet.selectedWell.v1.";
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
  normal: "Normal",
  watch: "Watch",
  attention: "Needs review",
  unavailable: "Check telemetry",
};
const sourceNames: Record<AgentSource, string> = {
  live: "Live LLM assessment",
  rules: "Model + rules",
  recorded: "Recorded assessment",
};
const number = (value: number | null | undefined, digits = 1) =>
  value == null || !Number.isFinite(value) ? "—" : value.toFixed(digits);
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
class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}
async function api<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(
    `/api/fleet${path}`,
    body === undefined
      ? undefined
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new ApiError(
      typeof data.detail === "string"
        ? data.detail
        : `Could not complete the request (${response.status}).`,
      response.status,
    );
  }
  return response.json();
}
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

function FieldMap({
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
      className="fleet-map"
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
            ? `${run.index} / ${run.total} replay steps · ${run.status}`
            : "4 independent well recordings"}
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
              <strong>{well.name}</strong>
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
}

function SensorChart({
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
  const data = frames.slice(-90).map((frame) => ({
    minute: Math.round(frame.elapsed_seconds / 60),
    value: score ? frame.risk_score : sensor ? frame.sensors[sensor] : null,
  }));
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
        {score
          ? "Model score · not a probability"
          : sensor || "Channel unavailable"}
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
          {frames.length ? "No usable readings" : "Waiting for readings"}
        </div>
      )}
    </div>
  );
}

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
      <span>THIS INCIDENT · SOURCE TIME</span>
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
        <summary>Optional finding note</summary>
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
            ? ` · ${new Date(task.updated_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })} local time`
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
      : "WELL-00001";
  });
  const [speed, setSpeed] = useState(12);
  const [mode, setMode] = useState<FleetMode>("guided");
  const [connected, setConnected] = useState(false);
  const [busy, setBusy] = useState(false);
  const [starting, setStarting] = useState(false);
  const [actionFeedback, setActionFeedback] = useState<ActionFeedback | null>(
    null,
  );
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [dismissedRunError, setDismissedRunError] = useState("");
  const [sheet, setSheet] = useState<"evidence" | "fault" | null>(null);
  const [observing, setObserving] = useState(false);
  const [note, setNote] = useState("");
  const lastEvent = useRef(-1);
  const source = useRef<EventSource | null>(null);
  const selectedRef = useRef(selected);
  selectedRef.current = selected;
  useEffect(() => {
    let disposed = false;
    let retry: ReturnType<typeof setTimeout> | undefined;
    const load = () => {
      api<FleetCatalog>("/catalog")
        .then((value) => {
          if (disposed) return;
          setCatalog(value);
          setError("");
          // A reconnecting session owns its speed; a later catalog response must
          // not overwrite the value already restored from the server.
          if (!sessionId) {
            setSpeed(value.default_speed);
            setMode(value.default_mode || "guided");
          }
        })
        .catch((e) => {
          if (disposed) return;
          // The API may still be starting; keep trying instead of leaving
          // Start disabled until a manual reload.
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
    if (sessionId && fallbackWells.some((well) => well.id === selected)) {
      localStorage.setItem(`${SELECTED_WELL_KEY}${sessionId}`, selected);
    }
  }, [sessionId, selected]);
  useEffect(() => {
    setObserving(false);
    setNote("");
  }, [selected]);
  useEffect(() => {
    if (!sessionId) return;
    let disposed = false;
    lastEvent.current = -1;
    const apply = (next: FleetRun) => {
      if (disposed || next.last_event_id < lastEvent.current) return;
      lastEvent.current = next.last_event_id;
      setRun(next);
      setSpeed(next.speed);
      setMode(next.mode || "continuous");
    };
    const resetMissing = () => {
      if (disposed) return;
      source.current?.close();
      localStorage.removeItem(SESSION_KEY);
      localStorage.removeItem(`${SELECTED_WELL_KEY}${sessionId}`);
      setSessionId(null);
      setSelected("WELL-00001");
      setRun(null);
      setConnected(false);
      setNotice("The server restarted. Start a new field replay.");
    };
    let reopen: ReturnType<typeof setTimeout> | undefined;
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
            if (event.type === "snapshot") {
              apply(event.payload);
              if (finished(event.payload.status)) {
                stream.close();
                setConnected(false);
              }
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
          api<FleetRun>(`/sessions/${sessionId}`)
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
    api<FleetRun>(`/sessions/${sessionId}`)
      .then((snapshot) => {
        if (disposed) return;
        apply(snapshot);
        if (finished(snapshot.status)) return;
        open(snapshot.last_event_id);
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
    if (next.id !== run?.id || next.last_event_id >= lastEvent.current) {
      lastEvent.current = next.last_event_id;
      setRun(next);
      setSpeed(next.speed);
      setMode(next.mode || "continuous");
    }
  };
  async function command(action: string, extra: object = {}) {
    if (!sessionId || busy) return;
    setBusy(true);
    setError("");
    try {
      accept(
        await api<FleetRun>(`/sessions/${sessionId}/control`, {
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
          await api<FleetRun>(`/sessions/${run.id}/control`, {
            action: "cancel",
          }),
        );
      source.current?.close();
      const next = await api<FleetRun>("/sessions", { speed, mode });
      lastEvent.current = next.last_event_id;
      setRun(next);
      setSelected("WELL-00001");
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
    const targetWell = selected;
    const incidentId = run?.wells.find((item) => item.id === targetWell)
      ?.incident?.id;
    if (!incidentId) return;
    const target = {
      wellId: targetWell,
      incidentId,
      action,
      checkId: check?.check_id,
    };
    setBusy(true);
    setError("");
    setActionFeedback({
      ...target,
      state: "saving",
      message:
        action === "recheck"
          ? "Requesting an evidence check…"
          : "Saving your review…",
    });
    try {
      const next = await api<FleetRun>(
        `/sessions/${sessionId}/incidents/${targetWell}/actions`,
        {
          action,
          incident_id: incidentId,
          ...(check ||
            (["observation", "complete"].includes(action) && note.trim()
              ? { note: note.trim() }
              : {})),
        },
      );
      accept(next);
      const message =
        action === "recheck"
          ? "Evidence check requested. Monitoring status below shows its progress."
          : action === "check"
            ? "Finding saved. Sensor status is assessed separately."
            : action === "observation"
              ? "Observation saved to this run."
              : action === "complete"
                ? "Your review is recorded. Sensor status is assessed separately."
                : "Acknowledgment saved. The concern remains open.";
      setActionFeedback({ ...target, state: "saved", message });
      if (
        selectedRef.current === targetWell &&
        ["observation", "complete", "acknowledge"].includes(action)
      ) {
        setObserving(false);
        setNote("");
      }
    } catch (e) {
      setActionFeedback({
        ...target,
        state: "error",
        message: `${action === "recheck" ? "Could not request the check." : "Could not save your review."} ${(e as Error).message}`,
      });
    } finally {
      setBusy(false);
    }
  }
  const wells = run?.wells.length
    ? run.wells
    : catalog?.wells.map(emptyWell) || fallbackWells;
  const well = wells.find((item) => item.id === selected) || wells[0];
  const queue = [...wells].sort((a, b) => {
    const order = run?.priority || [];
    const ai = order.indexOf(a.id);
    const bi = order.indexOf(b.id);
    return (ai < 0 ? 99 : ai) - (bi < 0 ? 99 : bi);
  });
  const active = !!run && !finished(run.status);
  const running = run?.status === "running";
  const paused = run?.status === "paused";
  const preparing = run?.status === "preparing";
  const guided = mode === "guided";
  const guidePhase = run?.guide?.phase;
  const checkpoint = run?.guide?.checkpoint;
  const guideAssessing = guided && running && guidePhase === "assessing";
  const continueLabel =
    guidePhase === "overview" ? "Next important moment" : "Continue demo";
  const currentFeedback =
    actionFeedback?.wellId === well.id &&
    actionFeedback.incidentId === well.incident?.id
      ? actionFeedback
      : null;
  const savingAction =
    currentFeedback?.state === "saving" ? currentFeedback.action : null;
  const reviewSaved =
    !!well.incident &&
    (well.incident.acknowledged ||
      well.incident.completed ||
      well.incident.condition_cleared);
  const reviewTitle = well.incident?.condition_cleared
    ? "Readings recovered"
    : well.incident?.completed
      ? "Your review is recorded"
      : "Acknowledgment saved";
  const reviewDetail = well.incident?.condition_cleared
    ? "The previous incident remains in the evidence record."
    : well.incident?.completed
      ? "The concern stays open until the sensor evidence recovers."
      : "The concern is still open. After checking the evidence, add what you found.";
  const nextMinutes =
    well.next_check && well.source_timestamp
      ? Math.max(
          0,
          Math.ceil(
            (Date.parse(well.next_check) - Date.parse(well.source_timestamp)) /
              60_000,
          ),
        )
      : null;
  const monitoring = !run
    ? {
        title: "Ready to monitor",
        detail: "Start the replay to read and assess all four wells.",
      }
    : finished(run.status)
      ? {
          title:
            run.status === "failed" ? "Replay interrupted" : "Replay ended",
          detail:
            "No new readings or checks are running. Start a new replay to continue.",
        }
      : preparing
        ? {
            title: "Loading the recordings",
            detail: "Monitoring starts when the four feeds are ready.",
          }
        : !connected
          ? {
              title: "Reconnecting to the replay",
              detail:
                "Showing the last received state. Monitoring status will update after reconnection.",
            }
          : well.investigation === "running"
            ? {
                title: "Reviewing current evidence",
                detail: paused
                  ? `This check is finishing. New readings and checks wait for ${guided ? "Continue demo" : "Resume"}.`
                  : well.activity,
              }
            : paused
              ? {
                  title: guided
                    ? "Historical readings are paused"
                    : "Automatic checks are paused",
                  detail:
                    well.investigation === "queued"
                      ? guided
                        ? "A review is queued. Continue demo checks the current evidence before moving on."
                        : "A review is queued. Resume the replay to run it and read new data."
                      : guided
                        ? "Take time to read and record your findings. Continue demo advances to the next important moment."
                        : "Resume the replay to receive new readings and continue checking this well.",
                }
              : well.investigation === "queued"
                ? {
                    title: guideAssessing
                      ? "Current-reading check queued"
                      : "Assessment queued",
                    detail: guideAssessing
                      ? "Historical readings are held while the current evidence is checked. The demo pauses when the review finishes."
                      : "Sensor monitoring continues while this well waits for review.",
                  }
                : {
                    title: guideAssessing
                      ? "Current reading held for review"
                      : "Monitoring each new reading",
                    detail: guideAssessing
                      ? "The field review is finishing before the demo pauses. No new readings are advancing."
                      : nextMinutes != null && Number.isFinite(nextMinutes)
                        ? `Next automatic review in ${nextMinutes || 1} data ${nextMinutes === 1 || nextMinutes === 0 ? "minute" : "minutes"}.`
                        : "Changes in the sensor evidence trigger another review.",
                  };
  const liveConfigured =
    run?.llm_configured ?? catalog?.llm_configured ?? false;
  const llmDeferred = run?.llm_deferred ?? catalog?.llm_deferred ?? false;
  const readyModel = run?.model_ready ?? catalog?.model_ready ?? false;
  const agentSource = well.assessment?.source || run?.agent_mode || "rules";
  const attention = wells.filter(
    (item) => item.frames.length && item.status === "attention",
  ).length;
  const unseen = wells.filter(
    (item) =>
      item.incident &&
      !item.incident.acknowledged &&
      !item.incident.completed &&
      !item.incident.condition_cleared,
  ).length;
  const runError =
    run?.error && run.error !== dismissedRunError ? run.error : "";
  const reliable = wells.filter(
    (item) => item.frames.length && item.quality.status === "good",
  ).length;
  const pending = wells.filter((item) =>
    ["queued", "running"].includes(item.investigation),
  ).length;
  const hasReading = (key: string) =>
    well.frames.some((frame) => frame.sensors[key] != null);
  const pressure = ["P-TPT", "P-MON-CKP", "P-PDG"].find(hasReading) || null;
  const temperature = ["T-TPT", "T-PDG", "T-JUS-CKP"].find(hasReading) || null;
  const headerStatus = preparing
    ? "Preparing the field replay"
    : paused
      ? guided
        ? "Your field, at your pace."
        : "Historical replay paused"
      : running
        ? guided
          ? guideAssessing
            ? "Checking the evidence"
            : "Finding the next change"
          : "Monitoring four wells"
        : run?.status === "failed"
          ? "Field replay interrupted"
          : finished(run?.status)
            ? "Field replay complete"
            : "Your field, in focus.";
  const checkpointWell = wells.find((item) => item.id === checkpoint?.well_id);
  const visibleCheckpoint = ["checkpoint", "ended"].includes(guidePhase || "")
    ? checkpoint
    : null;
  const guideTitle = preparing
    ? "Preparing your walkthrough"
    : !run || guidePhase === "overview"
      ? "Explore the field first"
      : running
        ? guideAssessing
          ? "Checking current evidence"
          : "Finding the next important moment"
        : visibleCheckpoint?.title ||
          (finished(run.status)
            ? "End of the recording"
            : "Take your time to review");
  const guideSummary = preparing
    ? "Loading the four recordings and their prior history."
    : !run
      ? "Start with the overview. Then move between real changes, with time to review each one."
      : guidePhase === "overview"
        ? "Charts show prior history. Current readings have not been assessed yet. Choose Next important moment to begin."
        : running
          ? guideAssessing
            ? "The current readings are being checked. The replay pauses when the assessment is ready."
            : "Reading every intervening minute in order. The replay pauses after the next meaningful change is assessed."
          : visibleCheckpoint?.summary ||
            (finished(run.status)
              ? "No more historical readings will arrive. You can still inspect the evidence and record findings."
              : "New historical readings are paused. Continue when you are ready.");
  return (
    <section
      className="fleet-dashboard"
      aria-label="Four-well operations dashboard"
    >
      <header className="fleet-header">
        <div>
          <div className="fleet-eyebrow">OFFSHORE OPERATIONS · CASE 09</div>
          <h1>{headerStatus}</h1>
          <p className="fleet-subtitle">
            One field view. A clear next step for every well.
          </p>
        </div>
        <div className="fleet-controls">
          <div
            className="fleet-mode-picker"
            role="group"
            aria-label="Replay mode"
          >
            {(["guided", "continuous"] as FleetMode[]).map((value) => (
              <button
                key={value}
                aria-pressed={mode === value}
                disabled={busy || preparing}
                onClick={() => {
                  if (mode === value) return;
                  if (active) void command("mode", { mode: value });
                  else setMode(value);
                }}
              >
                {value === "guided" ? "Guided demo" : "Continuous replay"}
              </button>
            ))}
          </div>
          {!guided && (
            <select
              aria-label="Replay pace"
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
                  {value === 12
                    ? "Comfortable · 1 reading / 5s"
                    : value === 30
                      ? "Faster · 1 data min / 2s"
                      : value === 60
                        ? "Fast · 1 data min / 1s"
                        : "Fastest · 1 data min / 0.5s"}
                </option>
              ))}
            </select>
          )}
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
              {guided ? "Start walkthrough" : "Start field replay"}
            </button>
          ) : (
            <button
              className="fleet-button primary"
              disabled={busy || preparing}
              onClick={() =>
                void command(
                  running ? "pause" : guided ? "next_moment" : "resume",
                )
              }
            >
              {preparing ? (
                <LoaderCircle size={15} className="spin" />
              ) : running ? (
                <Pause size={15} />
              ) : (
                <Play size={15} />
              )}{" "}
              {preparing
                ? "Preparing…"
                : running
                  ? guided
                    ? "Stop scanning"
                    : "Pause"
                  : guided
                    ? continueLabel
                    : "Resume"}
            </button>
          )}
          <button
            className="fleet-button"
            title="Read one more historical minute, then pause"
            disabled={busy || !paused}
            onClick={() => void command("step")}
          >
            <SkipForward size={15} />
            Read 1 minute
          </button>
          {run && (
            <button
              className="fleet-button icon"
              aria-label="Restart field replay"
              title="Restart field replay"
              disabled={busy || preparing}
              onClick={() => {
                if (
                  finished(run.status) ||
                  window.confirm(
                    "Restart the field replay? The current run will be discarded.",
                  )
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
      {guided && (
        <section
          className={`fleet-guide-banner ${running ? "scanning" : ""}`}
          aria-label="Guided replay status"
          aria-live="polite"
        >
          <div className="fleet-guide-icon">
            {preparing || running ? (
              <LoaderCircle size={20} className="spin" />
            ) : finished(run?.status) ? (
              <CheckCheck size={20} />
            ) : (
              <Pause size={20} />
            )}
          </div>
          <div className="fleet-guide-copy">
            <div className="fleet-eyebrow">
              {!run
                ? "GUIDED HISTORICAL REPLAY"
                : preparing
                  ? "PREPARING HISTORICAL REPLAY"
                  : running
                    ? guideAssessing
                      ? "HISTORICAL REPLAY · CHECKING EVIDENCE"
                      : "HISTORICAL REPLAY · ADVANCING"
                    : finished(run.status)
                      ? "HISTORICAL REPLAY ENDED"
                      : "HISTORICAL REPLAY PAUSED"}
            </div>
            <h2>{guideTitle}</h2>
            <p>{guideSummary}</p>
            {paused && pending > 0 && (
              <small>
                A review is queued. Continue demo checks the current evidence
                first.
              </small>
            )}
            {visibleCheckpoint?.source_timestamp && (
              <small>
                {checkpointWell?.name ? `${checkpointWell.name} · ` : ""}
                {time(visibleCheckpoint.source_timestamp)} source time
              </small>
            )}
          </div>
          {visibleCheckpoint?.well_id && checkpointWell && (
            <button
              className="fleet-button"
              disabled={selected === visibleCheckpoint.well_id}
              onClick={() => setSelected(visibleCheckpoint.well_id!)}
            >
              {selected === visibleCheckpoint.well_id
                ? `Viewing ${checkpointWell.name}`
                : `View ${checkpointWell.name}`}
              <ArrowRight size={13} />
            </button>
          )}
        </section>
      )}
      <div className="fleet-kpis">
        <Kpi
          value={String(attention)}
          label="Wells needing review"
          icon={<CircleAlert size={18} />}
          warning={attention > 0}
        />
        <Kpi
          value={String(unseen)}
          label="Awaiting acknowledgment"
          icon={<MessageSquare size={18} />}
        />
        <Kpi
          value={
            wells.some((item) => item.source_timestamp)
              ? `${reliable} / ${wells.length}`
              : "—"
          }
          label="Reliable feeds"
          icon={<Radio size={18} />}
        />
        <Kpi
          value={String(pending)}
          label="Queued / active checks"
          icon={<Activity size={18} />}
        />
      </div>
      <div className="fleet-layout">
        <div className="fleet-main">
          <FieldMap
            wells={wells}
            selected={well.id}
            onSelect={setSelected}
            run={run}
          />
          <p className="fleet-map-caption">
            Illustrative layout · independent Petrobras 3W recordings aligned to
            replay time
          </p>
          <div className="fleet-card">
            <div className="fleet-card-heading">
              <div>
                <h2>
                  {well.name}{" "}
                  <span style={{ color: "#5f7168", fontWeight: 400 }}> / </span>{" "}
                  Trends
                </h2>
                <small>
                  {well.source_timestamp
                    ? `Latest reading · ${time(well.source_timestamp)} · source time`
                    : well.frames.length
                      ? "Prior history · replay readings have not started"
                      : "Select a well on the map to follow its readings."}
                </small>
              </div>
              <button
                className="fleet-text-button"
                onClick={() => setSheet("evidence")}
              >
                Evidence
                <ArrowRight size={13} />
              </button>
            </div>
            <div className="fleet-charts">
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
            <div className="fleet-trend-foot">
              <Clock3 size={11} />
              Negative minutes = prior history · gaps stay visible
            </div>
          </div>
          {run?.fault && (
            <div className="fleet-notice">
              <Zap size={15} />
              <span>
                Demo fault on{" "}
                {wells.find((item) => item.id === run.fault?.well_id)?.name ||
                  run.fault.well_id}{" "}
                · {run.fault.remaining} inputs remaining. Excluded from accuracy
                claims.
              </span>
            </div>
          )}
        </div>
        <aside
          className="fleet-rail"
          aria-label="Agent priorities and assessment"
        >
          <div className="fleet-card">
            <div className="fleet-card-heading">
              <div>
                <div className="fleet-eyebrow" style={{ fontSize: 9 }}>
                  WHERE TO LOOK FIRST
                </div>
                <h2 style={{ marginTop: 6 }}>Priority list</h2>
              </div>
              <span className="fleet-source">{wells.length} wells</span>
            </div>
            <ol className="fleet-queue">
              {queue.map((item, index) => (
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
                          : "Awaiting first check"}
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
              <div className="fleet-eyebrow">
                <Sparkles size={12} />
                {well.name} · EVIDENCE REVIEW
              </div>
              <span
                className={`fleet-status ${well.source_timestamp ? well.status : "pending"}`}
              >
                <i
                  className={`fleet-dot ${well.source_timestamp ? well.status : "pending"}`}
                />
                {well.source_timestamp
                  ? stateNames[well.status]
                  : "Awaiting first check"}
              </span>
              <h3>
                {well.assessment?.summary ||
                  (active && connected && well.investigation === "running"
                    ? "Investigating the latest readings…"
                    : "Waiting for the first readings.")}
              </h3>
              {well.assessment?.evidence.length ? (
                <ul className="fleet-evidence-list">
                  {well.assessment.evidence
                    .slice(0, 2)
                    .map((evidence, index) => (
                      <li key={index}>
                        <ArrowDownRight size={13} />
                        {evidence}
                      </li>
                    ))}
                </ul>
              ) : (
                <p>
                  {well.frames.length
                    ? well.activity || "Checking the available sensor evidence."
                    : "Start the replay. The agent will check each well and surface what needs your attention."}
                </p>
              )}
              {!!well.timeline?.length && (
                <IncidentProgress events={well.timeline} />
              )}
              <div className="fleet-next-step">
                <span>NEXT OPERATOR STEP</span>
                <p>
                  {well.assessment?.next_step ||
                    (guided && guidePhase === "overview"
                      ? "Choose Next important moment when you are ready to begin."
                      : "No action yet. Monitoring starts with the replay.")}
                </p>
                {well.incident && (
                  <div className="fleet-actions" aria-busy={!!savingAction}>
                    <button
                      className="fleet-button primary"
                      disabled={
                        busy ||
                        !well.incident ||
                        well.incident.acknowledged ||
                        well.incident.completed ||
                        well.incident.condition_cleared
                      }
                      aria-describedby={`review-help-${well.id}`}
                      onClick={() => void incidentAction("acknowledge")}
                    >
                      {savingAction === "acknowledge" ? (
                        <LoaderCircle size={13} className="spin" />
                      ) : well.incident?.acknowledged ||
                        well.incident?.completed ? (
                        <CheckCheck size={13} />
                      ) : (
                        <Check size={13} />
                      )}{" "}
                      {savingAction === "acknowledge"
                        ? "Saving acknowledgment…"
                        : well.incident?.condition_cleared
                          ? "Recovered"
                          : well.incident?.completed
                            ? "Review recorded"
                            : well.incident?.acknowledged
                              ? "Acknowledged"
                              : currentFeedback?.state === "error" &&
                                  currentFeedback.action === "acknowledge"
                                ? "Try acknowledgment again"
                                : "Acknowledge alert"}
                    </button>
                    <button
                      className={`fleet-button ${reviewSaved ? "primary" : ""}`}
                      disabled={busy || !well.incident}
                      onClick={() => setObserving(!observing)}
                    >
                      <MessageSquare size={12} />
                      Add observation
                    </button>
                  </div>
                )}
                {well.incident && (
                  <div
                    id={`review-help-${well.id}`}
                    className={`fleet-review-feedback ${reviewSaved ? "saved" : ""}`}
                    role="status"
                    aria-live="polite"
                    aria-atomic="true"
                  >
                    {savingAction === "acknowledge" ? (
                      <span>Recording that you have seen this alert…</span>
                    ) : reviewSaved ? (
                      <>
                        <strong>
                          <CheckCheck size={13} /> {reviewTitle}
                        </strong>
                        <span>{reviewDetail}</span>
                      </>
                    ) : (
                      <span>
                        Acknowledge means “I’ve seen this alert.” The sensor
                        checks run independently.
                      </span>
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
                    <label htmlFor="fleet-observation">
                      What did you observe?
                    </label>
                    <textarea
                      id="fleet-observation"
                      value={note}
                      onChange={(event) => setNote(event.target.value)}
                      maxLength={500}
                      placeholder="e.g. Telemetry checked; awaiting engineer review."
                    />
                    <div className="fleet-actions">
                      <button
                        className="fleet-button primary"
                        disabled={busy || !note.trim()}
                        onClick={() => void incidentAction("observation")}
                      >
                        {savingAction === "observation" && (
                          <LoaderCircle size={12} className="spin" />
                        )}
                        {savingAction === "observation"
                          ? "Saving observation…"
                          : "Record observation"}
                      </button>
                      <button
                        className="fleet-button"
                        disabled={busy || !!well.incident?.completed}
                        onClick={() => void incidentAction("complete")}
                      >
                        {savingAction === "complete" && (
                          <LoaderCircle size={12} className="spin" />
                        )}
                        {savingAction === "complete"
                          ? "Saving review…"
                          : "Complete review"}
                      </button>
                    </div>
                    <p className="fleet-review-help">
                      Completing your review records your check. The alert
                      follows the sensor evidence.
                    </p>
                  </div>
                )}
                {!!well.incident && !!well.assessment?.checks?.length && (
                  <details
                    className="fleet-checklist"
                    key={`${well.id}-${well.incident.id}`}
                  >
                    <summary>
                      Operator checks{" "}
                      <span>
                        {
                          well.assessment.checks.filter(
                            (task) => task.status !== "pending",
                          ).length
                        }{" "}
                        / {well.assessment.checks.length} saved
                      </span>
                    </summary>
                    <div className="fleet-checklist-body">
                      <p className="fleet-review-help">
                        Record what you verified. Findings inform the review;
                        measured conditions determine the alert.
                      </p>
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
                  aria-label="Automatic monitoring status"
                >
                  <strong>
                    {active && connected && well.investigation === "running" ? (
                      <LoaderCircle size={14} className="spin" />
                    ) : paused ? (
                      <Pause size={14} />
                    ) : (
                      <Clock3 size={14} />
                    )}
                    {monitoring.title}
                  </strong>
                  <p>{monitoring.detail}</p>
                  {well.followup?.last_checked_at && (
                    <div className="fleet-followup-change">
                      <p>{well.followup.change_summary}</p>
                      <small>
                        Last checked · {time(well.followup.last_checked_at)}{" "}
                        source time
                      </small>
                    </div>
                  )}
                  {running &&
                    connected &&
                    well.investigation !== "running" &&
                    well.investigation !== "queued" && (
                      <small>Each replay step advances one data minute.</small>
                    )}
                  <div className="fleet-followup-actions">
                    {paused && (
                      <button
                        className="fleet-button"
                        disabled={busy}
                        onClick={() =>
                          void command(guided ? "next_moment" : "resume")
                        }
                      >
                        <Play size={12} />{" "}
                        {guided ? continueLabel : "Resume automatic checks"}
                      </button>
                    )}
                    {well.incident && (
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
                            ? "Start a new replay to run another evidence check."
                            : paused
                              ? `Queues an evidence check for when you ${guided ? "continue the demo" : "resume"}.`
                              : "Request a new review of the available evidence."
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
                            : "Recheck now"}
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
                  {sourceNames[agentSource]}
                </span>
                <button
                  className="fleet-text-button"
                  onClick={() => setSheet("evidence")}
                >
                  Why?
                  <ArrowRight size={12} />
                </button>
              </div>
            </div>
          </div>
          {llmDeferred ? (
            <div className="fleet-mode-notice">
              <ShieldCheck size={13} />
              <span>
                Automatic evidence checks are active. LLM investigations are
                deferred.
              </span>
            </div>
          ) : !liveConfigured ? (
            <div className="fleet-mode-notice">
              <Unplug size={13} />
              <span>
                LLM key not configured. Sensor checks and rule-based assessments
                remain available.
              </span>
            </div>
          ) : !readyModel ? (
            <div className="fleet-mode-notice">
              <LoaderCircle size={13} />
              <span>
                {run?.readiness_message ||
                  catalog?.readiness_message ||
                  "Preparing the trained model."}
              </span>
            </div>
          ) : (
            <div className="fleet-mode-notice">
              <ShieldCheck size={13} />
              <span>Operator advice. Plant controls remain with the crew.</span>
            </div>
          )}
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
              ? `${paused ? "Historical replay paused" : finished(run.status) ? "Historical replay finished" : connected ? (guideAssessing ? "Checking current evidence" : "Replay feed connected") : "Connecting to replay"} · ${llmDeferred ? "Model + rules · LLM deferred" : `${run.requests_used} / ${run.request_budget} LLM requests`}`
              : "Historical replay · assessments computed during the run"}
        </span>
        <div className="fleet-bottom-actions">
          <button
            className="fleet-text-button"
            disabled={!active || preparing}
            onClick={() => setSheet("fault")}
          >
            <Zap size={12} />
            Try a telemetry fault
          </button>
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
      </div>
      {sheet === "evidence" && (
        <Sheet title={`${well.name} · Evidence`} onClose={() => setSheet(null)}>
          <section>
            <h3>Current assessment</h3>
            <p>
              {well.assessment?.summary ||
                "The agent has not assessed this well yet."}
            </p>
            {well.assessment?.uncertainty && (
              <p style={{ marginTop: 8 }}>{well.assessment.uncertainty}</p>
            )}
            <ul className="fleet-evidence-list">
              {well.assessment?.evidence.map((evidence, index) => (
                <li key={index}>
                  <ArrowDownRight size={13} />
                  {evidence}
                </li>
              ))}
            </ul>
            {(well.assessment?.category ||
              !!well.assessment?.alternatives?.length) && (
              <details className="fleet-sensor-details">
                <summary>Assessment context</summary>
                {well.assessment.category && (
                  <p>
                    Working category:{" "}
                    {well.assessment.category.replaceAll("_", " ")}. This is a
                    review hypothesis, not a confirmed diagnosis.
                  </p>
                )}
                {!!well.assessment.alternatives?.length && (
                  <ul className="fleet-evidence-list">
                    {well.assessment.alternatives.map((alternative, index) => (
                      <li key={index}>{alternative}</li>
                    ))}
                  </ul>
                )}
              </details>
            )}
          </section>
          {well.followup?.after && (
            <section>
              <h3>Latest evidence check</h3>
              <p>{well.followup.change_summary}</p>
              <p>
                Trigger: {well.followup.trigger || "Scheduled evidence review"}
              </p>
              <div className="fleet-evidence-change">
                {[
                  { label: "Before", snapshot: well.followup.before },
                  { label: "After", snapshot: well.followup.after },
                ].map(({ label, snapshot }) => (
                  <div key={label}>
                    <span>{label}</span>
                    {snapshot ? (
                      <>
                        <time dateTime={snapshot.at}>
                          {snapshot.at} · source time
                        </time>
                        <strong>{stateNames[snapshot.status]}</strong>
                        <p>{snapshot.summary}</p>
                        <ul>
                          {snapshot.evidence.map((item, index) => (
                            <li key={index}>{item}</li>
                          ))}
                        </ul>
                      </>
                    ) : (
                      <p>No earlier assessment in this run.</p>
                    )}
                  </div>
                ))}
              </div>
            </section>
          )}
          {!!well.timeline?.length && (
            <section>
              <details className="fleet-sensor-details">
                <summary>
                  Incident history · {well.timeline.length} recorded events
                </summary>
                <p>
                  Times show when actions or checks occurred in the source
                  replay. An acknowledgment does not trigger a sensor check.
                </p>
                <ol className="fleet-history-list">
                  {well.timeline.map((event) => (
                    <li key={event.id}>
                      <strong>
                        {event.kind === "seen"
                          ? "Seen by operator"
                          : event.kind === "checked"
                            ? "Operator finding saved"
                            : event.kind === "recheck"
                              ? "Evidence rechecked"
                              : event.kind.replaceAll("_", " ")}
                      </strong>
                      <time dateTime={event.at}>{event.at} · source time</time>
                      <p>{event.summary}</p>
                    </li>
                  ))}
                </ol>
              </details>
            </section>
          )}
          <section>
            <h3>Data & provenance</h3>
            <dl>
              <dt>Source</dt>
              <dd>{well.source_file || "Loading catalog"}</dd>
              <dt>Source timestamp</dt>
              <dd>{well.source_timestamp || "No input yet"}</dd>
              <dt>Replay time</dt>
              <dd>{elapsed(run?.elapsed_seconds || 0)} elapsed</dd>
              <dt>Sensor health</dt>
              <dd>{well.quality.summary}</dd>
              <dt>Assessment source</dt>
              <dd>{sourceNames[agentSource]}</dd>
              <dt>Last assessed</dt>
              <dd>{well.last_assessed || "Not assessed"}</dd>
              <dt>Next check</dt>
              <dd>{well.next_check || "Not scheduled"}</dd>
            </dl>
          </section>
          <section>
            <details className="fleet-sensor-details">
              <summary>All sensor readings</summary>
              {well.frames.length ? (
                <>
                  <p>Latest source reading · {well.frames.at(-1)?.t}</p>
                  <table className="fleet-sensor-table">
                    <thead>
                      <tr>
                        <th>Sensor</th>
                        <th>Reading</th>
                        <th>Unit</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(well.frames.at(-1)!.sensors).map(
                        ([sensor, value]) => {
                          const unit = sensor.startsWith("P-")
                            ? "bar"
                            : sensor.startsWith("T-")
                              ? "°C"
                              : sensor === "ABER-CKP"
                                ? "%"
                                : sensor === "QGL"
                                  ? "m³/s"
                                  : "—";
                          return (
                            <tr key={sensor}>
                              <td>{sensor}</td>
                              <td>
                                {value == null || !Number.isFinite(value)
                                  ? "Unavailable"
                                  : value.toLocaleString("en", {
                                      maximumSignificantDigits: 6,
                                    })}
                              </td>
                              <td>{unit}</td>
                            </tr>
                          );
                        },
                      )}
                    </tbody>
                  </table>
                  <p>QGL is gas-lift injection, not oil production.</p>
                </>
              ) : (
                <p>No sensor readings have arrived yet.</p>
              )}
            </details>
          </section>
          <section>
            <h3>Investigation steps</h3>
            {well.assessment?.tools.length ? (
              <ol className="fleet-trace">
                {well.assessment.tools.map((tool, index) => (
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
                    {tool.sources?.length ? (
                      <small>Sources: {tool.sources.join(" · ")}</small>
                    ) : null}
                  </li>
                ))}
              </ol>
            ) : (
              <p>
                {well.activity ||
                  "Investigation steps appear after the next assessment."}
              </p>
            )}
            {well.assessment?.playbook_refs?.length ? (
              <p className="fleet-playbook-refs">
                <small>Playbook: {well.assessment.playbook_refs.join(" · ")}</small>
              </p>
            ) : null}
          </section>
          {well.incident && (
            <section>
              <h3>Operator record</h3>
              <dl>
                <dt>Incident</dt>
                <dd>{well.incident.id}</dd>
                <dt>Seen by operator</dt>
                <dd>
                  {well.incident.acknowledged
                    ? "Acknowledged"
                    : "Awaiting acknowledgment"}
                </dd>
                <dt>Review</dt>
                <dd>
                  {well.incident.condition_cleared
                    ? "Recovered"
                    : well.incident.completed
                      ? "Complete"
                      : "Open"}
                </dd>
                {well.incident.note && (
                  <>
                    <dt>Observation</dt>
                    <dd>{well.incident.note}</dd>
                  </>
                )}
              </dl>
            </section>
          )}
          <section>
            <p>
              These recordings have independent original timelines. Map
              locations and their shared replay clock are illustrative.
              Acknowledgment records attention; it does not clear the underlying
              condition.
            </p>
          </section>
        </Sheet>
      )}
      {sheet === "fault" && (
        <Sheet
          title={`Test telemetry · ${well.name}`}
          onClose={() => setSheet(null)}
        >
          <p>
            Change the next incoming readings for this well. The agent will
            investigate the changed evidence during this run.
          </p>
          <div className="fleet-fault-list">
            <button
              className="fleet-button"
              disabled={busy}
              onClick={() => {
                void command("inject", {
                  well_id: well.id,
                  fault: "pressure_offline",
                });
                setSheet(null);
              }}
            >
              <Unplug size={19} />
              <span>
                Pressure sensor offline
                <small>
                  Remove pressure readings and test the telemetry response.
                </small>
              </span>
            </button>
            <button
              className="fleet-button"
              disabled={busy || !run?.fault}
              onClick={() => {
                void command("inject", {
                  well_id: run?.fault?.well_id ?? well.id,
                  fault: "restore",
                });
                setSheet(null);
              }}
            >
              <Waves size={19} />
              <span>
                Restore original readings
                <small>
                  Stop the injected fault and continue the historical feed.
                </small>
              </span>
            </button>
          </div>
          <p>
            Injected faults are demo interventions and excluded from accuracy
            claims.
          </p>
        </Sheet>
      )}
    </section>
  );
}

function Kpi({
  value,
  label,
  icon,
  warning = false,
}: {
  value: string;
  label: string;
  icon: ReactNode;
  warning?: boolean;
}) {
  return (
    <div className={`fleet-kpi ${warning ? "warn" : ""}`}>
      <div className="fleet-kpi-icon">{icon}</div>
      <div>
        <strong>{value}</strong>
        <span>{label}</span>
      </div>
    </div>
  );
}

type FleetMetrics = {
  events_detected: number;
  hydrate_events: number;
  false_alarm_episodes: number;
  false_alarm_minutes: number;
  false_alarm_minutes_per_normal_day: number;
  missed_hydrate_minutes: number;
  early_events: number;
  lookalikes_flagged: number;
  lookalike_recordings: number;
  illustrative_cost: number;
  mean_detection_delay_minutes: number | null;
};
type FleetReport = {
  model_id: string;
  training_recordings: number;
  evaluation_recordings: number;
  training_wells: string[];
  excluded_wells: string[];
  protocol: string;
  selected_policy: {
    activation_threshold: number;
    persistence_minutes: number;
    recovery_minutes: number;
    recovery_threshold: number;
  };
  validation_baseline: FleetMetrics;
  validation_selected: FleetMetrics;
  candidates: {
    policy: { activation_threshold: number; persistence_minutes: number };
    metrics: FleetMetrics;
    accepted: boolean;
    reason: string;
  }[];
  stages: { id: string; name: string; metrics: FleetMetrics }[];
  limits: string[];
  recordings: { well_id: string; file: string; hydrate_event: boolean }[];
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
  selected_candidate?: {
    id: string;
    feature_family: string;
    weighting: string;
  };
};

export function FleetResults() {
  const [report, setReport] = useState<FleetReport | null>(null);
  const [error, setError] = useState("");
  const [expanded, setExpanded] = useState(false);
  useEffect(() => {
    api<FleetReport>("/results")
      .then((value) => {
        if (!value.stages)
          throw new Error("The fleet evaluation has not run yet.");
        setReport(value);
      })
      .catch((e) => setError(e.message));
  }, []);
  if (error)
    return (
      <section className="fleet-results fleet-card">
        <div className="fleet-assessment">
          <h2>Real-well fleet results</h2>
          <p>{error}</p>
        </div>
      </section>
    );
  if (!report)
    return (
      <section className="fleet-results fleet-card">
        <div className="fleet-assessment">
          <p>Loading the four-well evaluation…</p>
        </div>
      </section>
    );
  const accepted = report.candidates.filter(
    (candidate) => candidate.accepted,
  ).length;
  const first = report.stages[0]?.metrics;
  const quality = report.stages.find(
    (stage) => stage.id === "quality",
  )?.metrics;
  const selectedMetrics = report.stages.find(
    (stage) => stage.id === "selected",
  )?.metrics;
  const hydrateWells = new Set(
    report.recordings
      .filter((record) => record.hydrate_event)
      .map((record) => record.well_id),
  ).size;
  return (
    <section
      className="fleet-results fleet-card"
      aria-label="Measured four-well fleet results"
    >
      <div className="fleet-card-heading">
        <div>
          <div className="fleet-eyebrow">REAL PETROBRAS 3W DATA</div>
          <h2>What changed — and what did not</h2>
          <small>
            All recordings from the four dashboard wells were excluded from
            model training.
          </small>
        </div>
        <span className="fleet-source">Measured results</span>
      </div>
      <div className="fleet-evaluation-split">
        <div>
          <span>01 · LEARN</span>
          <strong>{report.training_recordings} recordings</strong>
          <small>{report.training_wells.length} wells to train the model</small>
        </div>
        <ArrowRight size={17} />
        <div>
          <span>02 · CHOOSE</span>
          <strong>{report.candidates.length} alarm policies per model</strong>
          <small>
            {report.model_comparison?.length
              ? `${report.model_comparison.length} classifiers · `
              : ""}
            Validation folds keep wells separate
          </small>
        </div>
        <ArrowRight size={17} />
        <div>
          <span>03 · EVALUATE</span>
          <strong>{report.evaluation_recordings} recordings</strong>
          <small>
            {report.excluded_wells.length} wells held out of training
          </small>
        </div>
      </div>
      <div className="fleet-result-table-wrap">
        <table className="fleet-result-table">
          <thead>
            <tr>
              <th>Stage</th>
              <th>Events caught</th>
              <th>False alert episodes</th>
              <th>False alert minutes</th>
              <th>Missed hydrate minutes</th>
              <th>False min / normal day</th>
              <th>Mean detection delay</th>
            </tr>
          </thead>
          <tbody>
            {report.stages.map((stage, index) => (
              <tr key={stage.id}>
                <td>
                  <span>{String(index + 1).padStart(2, "0")}</span>
                  {stage.name}
                </td>
                <td>
                  {stage.metrics.events_detected} /{" "}
                  {stage.metrics.hydrate_events}
                </td>
                <td>{stage.metrics.false_alarm_episodes}</td>
                <td>{stage.metrics.false_alarm_minutes.toLocaleString()}</td>
                <td>{stage.metrics.missed_hydrate_minutes.toLocaleString()}</td>
                <td>
                  {number(stage.metrics.false_alarm_minutes_per_normal_day, 1)}
                </td>
                <td>
                  {number(stage.metrics.mean_detection_delay_minutes, 2)} min
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="fleet-result-summary">
        <div>
          <h3>
            {accepted
              ? `${accepted} alarm-policy update${accepted === 1 ? "" : "s"}`
              : "The search kept the existing policy"}
          </h3>
          <p>
            {accepted
              ? "Only validation improvements that preserved detected events could be accepted."
              : `None of the ${report.candidates.length} alarm policies improved the validation objective while preserving detected events.`}{" "}
            Active threshold{" "}
            {number(report.selected_policy.activation_threshold, 2)} ·{" "}
            {report.selected_policy.persistence_minutes}-minute persistence ·{" "}
            {report.selected_policy.recovery_minutes}-minute recovery.
          </p>
          <p style={{ marginTop: 8 }}>
            Validation: {report.validation_baseline.events_detected} →{" "}
            {report.validation_selected.events_detected} events caught;{" "}
            {report.validation_baseline.false_alarm_minutes.toLocaleString()} →{" "}
            {report.validation_selected.false_alarm_minutes.toLocaleString()}{" "}
            false alert minutes.
          </p>
        </div>
        {first && quality && (
          <div>
            <h3>A measured tradeoff</h3>
            <p>
              Sensor-quality handling changed false alert episodes from{" "}
              {first.false_alarm_episodes} to {quality.false_alarm_episodes},
              while missed hydrate minutes changed from{" "}
              {first.missed_hydrate_minutes.toLocaleString()} to{" "}
              {quality.missed_hydrate_minutes.toLocaleString()}. The overall
              result is not a uniform improvement.
            </p>
            {selectedMetrics && (
              <p style={{ marginTop: 8 }}>
                The selected policy reduced episodes to{" "}
                {selectedMetrics.false_alarm_episodes}, with{" "}
                {selectedMetrics.missed_hydrate_minutes.toLocaleString()} missed
                minutes and{" "}
                {number(selectedMetrics.mean_detection_delay_minutes, 2)}{" "}
                minutes mean detection delay.
              </p>
            )}
          </div>
        )}
      </div>
      <div className="fleet-result-limit">
        <CircleAlert size={14} />
        <p>
          {first?.hydrate_events || 0} hydrate events came from{" "}
          {hydrateWells || 1} held-out hydrate well{hydrateWells > 1 ? "s" : ""}
          . This previously explored subset is internal evaluation; it does not
          establish field reliability. These results are separate from the
          synthetic seed experiment below.
        </p>
      </div>
      <div className="fleet-result-details">
        {!!report.model_comparison?.length && (
          <details className="fleet-model-comparison">
            <summary>
              Classifier comparison ·{" "}
              {report.selected_candidate?.id === "base_class"
                ? "current classifier retained"
                : "candidate selected"}
            </summary>
            <div className="fleet-result-expanded">
              <p>
                Grouped development-validation results, selected before held-out
                prediction. An accepted alarm-policy update and a better
                classifier are separate decisions.
              </p>
              <div className="fleet-result-table-wrap">
                <table className="fleet-result-table">
                  <thead>
                    <tr>
                      <th>Classifier</th>
                      <th>Events caught</th>
                      <th>False alert minutes</th>
                      <th>Lookalike recall</th>
                      <th>Decision</th>
                    </tr>
                  </thead>
                  <tbody>
                    {report.model_comparison.map((candidate) => (
                      <tr key={candidate.id}>
                        <td>
                          {{
                            base_class: "Current classifier",
                            base_record: "Equal recording weights",
                            relative_record: "Relative changes",
                            hybrid_record: "Combined features",
                          }[candidate.id] || candidate.id}
                        </td>
                        <td>
                          {candidate.validation.events_detected} /{" "}
                          {candidate.validation.hydrate_events}
                        </td>
                        <td>
                          {candidate.validation.false_alarm_minutes.toLocaleString()}
                        </td>
                        <td>
                          {number(
                            candidate.validation.classification_recall
                              .lookalike * 100,
                            1,
                          )}
                          %
                        </td>
                        <td>
                          {candidate.selected
                            ? candidate.id === "base_class"
                              ? "Retain incumbent"
                              : "Selected"
                            : candidate.qualifies
                              ? "Qualifies"
                              : "Rejected"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p>
                {report.selected_candidate?.id === "base_class"
                  ? "No challenger passed the validation gates. The existing classifier remains active."
                  : "The selected classifier passed the validation gates. Held-out results above remain separate."}
              </p>
            </div>
          </details>
        )}
        <button
          className="fleet-text-button"
          onClick={() => setExpanded(!expanded)}
          aria-expanded={expanded}
        >
          {expanded
            ? "Hide evaluation details"
            : "Show candidates, protocol and limits"}
          <ChevronRight size={13} />
        </button>
        {expanded && (
          <div className="fleet-result-expanded">
            <p>{report.protocol}</p>
            <div className="fleet-result-table-wrap">
              <table className="fleet-result-table">
                <thead>
                  <tr>
                    <th>Candidate</th>
                    <th>Threshold</th>
                    <th>Persistence</th>
                    <th>Events caught</th>
                    <th>False alert minutes</th>
                    <th>Decision</th>
                  </tr>
                </thead>
                <tbody>
                  {report.candidates.map((candidate, index) => (
                    <tr key={index}>
                      <td>{index + 1}</td>
                      <td>
                        {number(candidate.policy.activation_threshold, 2)}
                      </td>
                      <td>{candidate.policy.persistence_minutes} min</td>
                      <td>
                        {candidate.metrics.events_detected} /{" "}
                        {candidate.metrics.hydrate_events}
                      </td>
                      <td>
                        {candidate.metrics.false_alarm_minutes.toLocaleString()}
                      </td>
                      <td>
                        {candidate.accepted ? "Accept" : "Retain incumbent"}
                      </td>
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
            <small>Frozen model: {report.model_id}</small>
          </div>
        )}
      </div>
      <WorkflowResults />
    </section>
  );
}

function WorkflowResults() {
  const [report, setReport] = useState<FleetWorkflowReport | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    api<FleetWorkflowReport>("/workflow-results")
      .then(setReport)
      .catch((e) => setError(e.message));
  }, []);
  const metrics = report?.metrics;
  return (
    <section
      className="fleet-workflow-results"
      aria-label="Operator workflow evaluation"
    >
      <div className="fleet-card-heading">
        <div>
          <div className="fleet-eyebrow">
            OPERATOR WORKFLOW · SEPARATE EVALUATION
          </div>
          <h2>Does the follow-up work?</h2>
          <small>
            Replay behavior and scripted checks. Separate from detection-model
            accuracy.
          </small>
        </div>
        <span className="fleet-source">No LLM calls</span>
      </div>
      {error ? (
        <p className="fleet-workflow-empty">
          Workflow evaluation is unavailable: {error}
        </p>
      ) : !report ? (
        <p className="fleet-workflow-empty">
          Loading the saved workflow evaluation…
        </p>
      ) : report.status !== "complete" || !metrics ? (
        <p className="fleet-workflow-empty">
          No completed workflow evaluation has been saved yet.
        </p>
      ) : (
        <>
          <div className="fleet-workflow-metrics">
            <div>
              <span>Normal minutes flagged for attention</span>
              <strong>
                {metrics.model_only_false_attention_minutes.toLocaleString()}{" "}
                <ArrowRight size={13} />{" "}
                {metrics.workflow_false_attention_minutes.toLocaleString()}
              </strong>
              <small>Model alarm → complete workflow</small>
            </div>
            <div>
              <span>Normal minutes on watch</span>
              <strong>
                {metrics.workflow_false_watch_minutes.toLocaleString()}
              </strong>
              <small>Additional operator review burden</small>
            </div>
            <div>
              <span>Minutes with unavailable evidence</span>
              <strong>
                {metrics.workflow_unavailable_minutes.toLocaleString()}
              </strong>
              <small>All replay minutes; never treated as normal</small>
            </div>
          </div>
          <p className="fleet-workflow-scope">
            {metrics.recordings} real recordings ·{" "}
            {metrics.observed_minutes.toLocaleString()} observed minutes ·{" "}
            {metrics.normal_minutes.toLocaleString()} labelled normal minutes. A
            flagged recording means watch or attention overlapped its event,
            including flags already present before the event. This does not
            establish early warning or a correct diagnosis.
          </p>
          <div className="fleet-workflow-disclosures">
            <details>
              <summary>Event coverage and limitations</summary>
              <div className="fleet-workflow-detail">
                <table className="fleet-sensor-table">
                  <thead>
                    <tr>
                      <th>Labelled recording</th>
                      <th>Flagged by workflow</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>Hydrate</td>
                      <td>
                        {metrics.hydrate_events_detected} /{" "}
                        {metrics.hydrate_events_total}
                      </td>
                    </tr>
                    <tr>
                      <td>Restriction</td>
                      <td>
                        {metrics.restriction_events_detected} /{" "}
                        {metrics.restriction_events_total}
                      </td>
                    </tr>
                    <tr>
                      <td>Scaling</td>
                      <td>
                        {metrics.scaling_events_detected} /{" "}
                        {metrics.scaling_events_total}
                      </td>
                    </tr>
                  </tbody>
                </table>
                <ul>
                  {report.limits?.map((limit) => (
                    <li key={limit}>{limit}</li>
                  ))}
                </ul>
                <small>
                  Model {report.model_id} · generated {report.generated_at}
                </small>
              </div>
            </details>
            <details>
              <summary>
                Scripted workflow checks{" "}
                <span>
                  {report.summary?.passed ?? 0} / {report.summary?.total ?? 0}{" "}
                  passed
                </span>
              </summary>
              <div className="fleet-workflow-detail">
                <p>
                  Synthetic scenarios test the software lifecycle. Passing these
                  checks is not proof of field diagnosis or operator benefit.
                </p>
                <ol className="fleet-workflow-scenarios">
                  {report.scenarios?.map((scenario) => (
                    <li key={scenario.id}>
                      <div>
                        <strong>{scenario.name}</strong>
                        <span
                          className={
                            scenario.status === "pass" ? "pass" : "fail"
                          }
                        >
                          {scenario.status === "pass" ? (
                            <Check size={12} />
                          ) : (
                            <CircleAlert size={12} />
                          )}{" "}
                          {scenario.status === "pass" ? "Pass" : "Fail"}
                        </span>
                      </div>
                      <p>{scenario.evidence}</p>
                      {scenario.metrics && (
                        <small>
                          {Object.entries(scenario.metrics)
                            .map(
                              ([key, value]) =>
                                `${key.replaceAll("_", " ")}: ${value}`,
                            )
                            .join(" · ")}
                        </small>
                      )}
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
