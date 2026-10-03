import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import {
  Activity,
  ArrowRight,
  Check,
  ChevronRight,
  CircleHelp,
  Compass,
  Download,
  FlaskConical,
  LoaderCircle,
  Pause,
  Play,
  Radio,
  RotateCcw,
  Settings2,
  ShieldCheck,
  SkipForward,
  Snowflake,
  Square,
  Unplug,
  Waves,
  X,
  Zap,
} from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Decision, Run } from "./LiveMission";
import "./overview.css";

type Props = {
  run: Run | null;
  connected: boolean;
  busy: boolean;
  error: string;
  scenario: string;
  speed: number;
  onScenario: (value: string) => void;
  onSpeed: (value: number) => void;
  onStart: () => Promise<void>;
  onCommand: (action: string, extra?: object) => Promise<void>;
  onDiagnostics: () => void;
};
type Tone = "normal" | "watch" | "alert" | "idle" | "learning";
const finished = (status?: string) =>
  ["completed", "cancelled", "failed"].includes(status || "");
const number = (value?: number | null) =>
  value == null ? "—" : value.toFixed(1);
const time = (value?: string) =>
  value
    ? new Date(value).toLocaleString("en", {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
        hour12: false,
      })
    : "Awaiting data";
const toolNames: Record<string, string> = {
  check_sensor_quality: "Sensor health",
  inspect_sensor_history: "Sensor history",
  read_recent_window: "Recent readings",
  compare_secondary_sensors: "Temperature & flow",
  test_active_policy: "Alarm rule",
  evaluate_candidate: "Candidate rule",
};

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
      className="overview-sheet"
      ref={ref}
      aria-label={title}
      onCancel={onClose}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="sheet-content">
        <header>
          <h2>{title}</h2>
          <button
            className="map-icon-button"
            aria-label="Close panel"
            onClick={onClose}
          >
            <X size={20} />
          </button>
        </header>
        {children}
      </div>
    </dialog>
  );
}

function FieldMap({
  tone,
  running,
  label,
  onInspect,
}: {
  tone: Tone;
  running: boolean;
  label: string;
  onInspect: () => void;
}) {
  return (
    <div className={`field-map tone-${tone} ${running ? "map-running" : ""}`}>
      <div className="map-caption">
        <span className="map-caption-dot" /> OFFSHORE FIELD{" "}
        <span>1 demo well</span>
      </div>
      <svg
        className="field-drawing"
        viewBox="0 0 900 520"
        role="group"
        aria-label={`Illustrative offshore layout. Demo well 01: ${label}. Select the well to inspect its evidence.`}
      >
        <defs>
          <pattern
            id="map-grid"
            width="48"
            height="48"
            patternUnits="userSpaceOnUse"
          >
            <path
              d="M 48 0 H 0 V 48"
              fill="none"
              stroke="#8dbaae"
              strokeWidth=".6"
              opacity=".25"
            />
          </pattern>
          <pattern
            id="land-grain"
            width="10"
            height="10"
            patternUnits="userSpaceOnUse"
          >
            <circle cx="2" cy="2" r=".6" fill="#9cab8f" opacity=".28" />
          </pattern>
          <radialGradient id="water-glow">
            <stop stopColor="#f1f8ef" stopOpacity=".9" />
            <stop offset="1" stopColor="#c8ded9" stopOpacity="0" />
          </radialGradient>
          <filter
            id="asset-shadow"
            x="-60%"
            y="-60%"
            width="220%"
            height="220%"
          >
            <feDropShadow
              dx="0"
              dy="8"
              stdDeviation="8"
              floodColor="#264e47"
              floodOpacity=".15"
            />
          </filter>
        </defs>
        <rect width="900" height="520" fill="#d6e7e2" />
        <rect width="900" height="520" fill="url(#map-grid)" />
        <ellipse cx="540" cy="230" rx="380" ry="270" fill="url(#water-glow)" />
        <g fill="none" stroke="#96bcb0" strokeWidth="1.1" opacity=".32">
          <path d="M-20 40 C210 50 125 210 274 293 S365 482 492 550" />
          <path d="M-20 7 C246 23 160 194 306 269 S415 465 551 550" />
          <path d="M-20-34 C297 4 188 174 349 246 S465 451 619 550" />
          <path d="M30-50 C336-3 253 153 402 221 S510 416 695 550" />
          <path d="M128-50 C400 3 314 134 461 185 S590 404 779 549" />
          <path d="M290-50 C453 3 438 112 567 144 S703 333 946 342" />
          <path d="M489-30 C524 23 562 78 693 106 S796 224 956 238" />
          <path d="M591-40 C650 31 672 38 780 65 S870 143 974 138" />
        </g>
        <path
          d="M0 0 H104 C167 61 134 105 198 150 C252 188 188 224 184 258 C180 307 239 321 199 361 C153 408 214 460 278 520 H0Z"
          fill="#e8ebdc"
          stroke="#b8cbb6"
          strokeWidth="2"
        />
        <path
          d="M0 0 H104 C167 61 134 105 198 150 C252 188 188 224 184 258 C180 307 239 321 199 361 C153 408 214 460 278 520 H0Z"
          fill="url(#land-grain)"
        />
        <path
          d="M13 80 Q122 146 112 250 T143 481"
          fill="none"
          stroke="#ccd3bc"
          strokeWidth="3"
          strokeDasharray="4 7"
        />
        <text
          x="42"
          y="239"
          className="map-region-label"
          transform="rotate(-14 42 239)"
        >
          COAST
        </text>
        <text x="661" y="410" className="map-region-label">
          OFFSHORE
        </text>
        <path
          d="M170 368 L312 368 L426 278 L628 210"
          className="pipeline-base"
        />
        <path
          d="M170 368 L312 368 L426 278 L628 210"
          className="pipeline-flow"
        />
        <text
          x="326"
          y="349"
          className="pipeline-label"
          transform="rotate(-36 326 349)"
        >
          PRODUCTION LINE
        </text>
        <g transform="translate(170 368)" filter="url(#asset-shadow)">
          <rect
            x="-21"
            y="-21"
            width="42"
            height="42"
            rx="9"
            fill="#fbfcf4"
            stroke="#97b1a1"
          />
          <path
            d="M-12 10 V-5 L-4-11 V-3 L5-8 V10 Z M7-15 H13 V10 H7Z"
            fill="#7d9a88"
          />
        </g>
        <text x="170" y="410" textAnchor="middle" className="asset-label">
          Shore terminal
        </text>
        <g transform="translate(426 278)" filter="url(#asset-shadow)">
          <path
            d="M-35-9 L0-28 L35-9 L0 10Z"
            fill="#f6f8ec"
            stroke="#719887"
            strokeWidth="1.5"
          />
          <path
            d="M-35-9 V3 L0 23 L35 3 V-9 L0 10Z"
            fill="#adccbb"
            stroke="#719887"
            strokeWidth="1.5"
          />
          <path
            d="M-22 11 V33 M0 23 V42 M22 11 V33 M-8-15 L0-53 L9-15 M-4-33 H5 M-6-24 H7 M-16-15 H17"
            fill="none"
            stroke="#507b6d"
            strokeWidth="3"
            strokeLinejoin="round"
          />
          <path
            d="M14-17 V-40 H30 M28-40 V-29"
            fill="none"
            stroke="#507b6d"
            strokeWidth="2.5"
          />
        </g>
        <text x="426" y="341" textAnchor="middle" className="asset-label">
          Production platform
        </text>
        <g
          className="well-map-node"
          role="button"
          aria-label="Inspect Demo well 01"
          tabIndex={0}
          onClick={onInspect}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              onInspect();
            }
          }}
          transform="translate(628 210)"
        >
          <circle className="well-halo outer" r="66" />
          <circle className="well-halo inner" r="46" />
          <circle className="well-base" r="27" filter="url(#asset-shadow)" />
          <path
            d="M-10 11 H10 M-7 11 L-3-11 H3 L7 11 M-5 3 H5 M-4-3 H4 M0-17 V-11 M-8-14 H8"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
            strokeLinecap="round"
          />
          <rect
            className="well-map-label"
            x="-77"
            y="53"
            width="154"
            height="53"
            rx="8"
          />
          <text x="0" y="74" textAnchor="middle" className="well-name">
            Demo well 01
          </text>
          <text x="0" y="93" textAnchor="middle" className="well-state">
            {label}
          </text>
        </g>
        <g transform="translate(840 96)" opacity=".65">
          <text x="0" y="-20" textAnchor="middle" className="asset-label">
            N
          </text>
          <path d="M0-8 L-7 12 L0 7 L7 12Z" fill="#426f61" />
        </g>
      </svg>
      <div className="map-legend">
        <span>
          <i className="normal" />
          Normal
        </span>
        <span>
          <i className="watch" />
          Watch
        </span>
        <span>
          <i className="alert" />
          Needs attention
        </span>
      </div>
      <span className="map-disclosure">
        Illustrative layout · no location data
      </span>
    </div>
  );
}

export default function WellOverview(props: Props) {
  const {
    run,
    busy,
    error,
    connected,
    scenario,
    speed,
    onScenario,
    onSpeed,
    onStart,
    onCommand,
    onDiagnostics,
  } = props;
  const [sheet, setSheet] = useState<"well" | "controls" | null>(null);
  const active = !!run && !finished(run.status);
  const executing = active && (run.status === "running" || run.stepping);
  const learning = run?.phase === "learn";
  const latest = run?.frames.at(-1);
  const previous = [...(run?.events || [])]
    .reverse()
    .find((event) => event.type === "decision" && event.phase === run?.phase);
  const decision = run?.decision || (previous?.payload as Decision | undefined);
  // Keep displayed evidence aligned with the last completed decision while
  // the next input is being investigated.
  const assessedReading = decision
    ? run?.frames.find((frame) => frame.t === decision.t) || latest
    : latest;
  const checking = !!latest && !run?.decision && !learning && active;
  const tone: Tone = learning
    ? "learning"
    : decision?.decision === "ALERT"
      ? "alert"
      : decision?.decision === "WATCH"
        ? "watch"
        : decision
          ? "normal"
          : "idle";
  const label = learning
    ? "Testing rules"
    : tone === "alert"
      ? "Needs attention"
      : tone === "watch"
        ? "Watch closely"
        : tone === "normal"
          ? "Normal"
          : checking
            ? "Checking reading"
            : "Awaiting data";
  const qualityIssue = decision?.diagnosis === "Telemetry needs attention";
  const reason = learning
    ? `${run.trials.length} of 24 candidate rules tested.`
    : qualityIssue
      ? "Sensor readings need checking."
      : tone === "alert"
        ? run?.policy.confirmation_percentile
          ? "Low pressure is confirmed by temperature or flow."
          : "Pressure has crossed the current alarm threshold."
        : tone === "watch"
          ? "Pressure is low. The alarm rule is not yet confirmed."
          : tone === "normal"
            ? "No active pressure alarm."
            : checking
              ? "Checking pressure, temperature and flow."
              : "Start monitoring to check this well.";
  const nextAction = learning
    ? "Keep a better rule, then test it."
    : qualityIssue
      ? "Check the sensor connection."
      : tone === "alert"
        ? "Ask the operator to inspect this well."
        : tone === "watch"
          ? "Recheck the next reading."
          : tone === "normal"
            ? "Continue monitoring."
            : checking
              ? "Wait for this check to finish."
              : "Waiting for the first reading.";
  const phase = learning
    ? 1
    : run?.phase === "prove" ||
        (run?.phase === "complete" && run.scenario !== "sandbox")
      ? 2
      : 0;
  const sensorRows = [
    { key: "pressure_bar", label: "Pressure", unit: "bar" },
    { key: "temp_C", label: "Temperature", unit: "°C" },
    { key: "flow_Lps", label: "Flow", unit: "L/s" },
  ];
  const chart = (run?.frames || [])
    .slice(-48)
    .map((frame) => ({ t: frame.t, pressure: frame.sensors.pressure_bar }));
  const lastTrial = run?.trials.at(-1);
  const hasFinalScore =
    !!run?.score &&
    (run.phase === "prove" ||
      (run.status === "completed" && run.scenario !== "sandbox"));

  return (
    <section className="well-overview">
      <header className="overview-heading">
        <div>
          <span className="overview-kicker">CASE 09 · OFFSHORE MONITORING</span>
          <h1>Which well needs attention?</h1>
        </div>
        <div className="overview-controls">
          <span className={`feed-state ${executing ? "on" : ""}`}>
            <i />
            {run?.stepping
              ? "Stepping"
              : executing
                ? "Running"
                : run?.status === "paused"
                  ? "Paused"
                  : run?.status === "completed"
                    ? "Complete"
                    : run?.status === "failed"
                      ? "Failed"
                      : "Ready"}
          </span>
          <button
            className="overview-primary"
            disabled={busy}
            onClick={() =>
              active ? onCommand(executing ? "pause" : "resume") : onStart()
            }
          >
            {busy ? (
              <LoaderCircle size={16} className="spin" />
            ) : executing ? (
              <Pause size={16} />
            ) : (
              <Play size={16} />
            )}
            {active
              ? executing
                ? "Pause"
                : "Resume"
              : run
                ? "New run"
                : "Start demo"}
          </button>
          <button
            className="map-icon-button"
            aria-label="Demo controls"
            title="Demo controls"
            onClick={() => setSheet("controls")}
          >
            <Settings2 size={19} />
          </button>
        </div>
      </header>
      {(error || run?.error) && (
        <div className="error" role="alert">
          <CircleHelp size={16} />
          {error || run?.error}
        </div>
      )}
      {active && !connected && (
        <div className="map-reconnecting" role="status">
          <LoaderCircle className="spin" size={14} />
          Reconnecting… readings may be out of date.
        </div>
      )}
      <div className="overview-field">
        <FieldMap
          tone={tone}
          running={!!executing && !learning && connected}
          label={label}
          onInspect={() => setSheet("well")}
        />
        <aside className={`well-summary tone-${tone}`}>
          <div className="summary-top">
            <span className="well-symbol">
              <Waves size={20} />
            </span>
            <div>
              <span>SELECTED WELL</span>
              <h2>Demo well 01</h2>
            </div>
            <Radio size={15} />
          </div>
          <div className="well-status">
            <span />
            {label}
          </div>
          <p className="well-reason">{reason}</p>
          <div className="summary-sensors">
            {sensorRows.map((sensor) => (
              <div key={sensor.key}>
                <span>{sensor.label}</span>
                <strong>
                  {number(assessedReading?.sensors[sensor.key])}
                  <small>{sensor.unit}</small>
                </strong>
              </div>
            ))}
          </div>
          <div className="well-next">
            <span>NEXT STEP</span>
            <strong>{nextAction}</strong>
            {decision && !learning && (
              <small>Next check · {time(decision.next_recheck)}</small>
            )}
          </div>
          <button className="evidence-button" onClick={() => setSheet("well")}>
            View evidence <ArrowRight size={16} />
          </button>
          <div className="well-updated">
            {checking ? (
              <>
                <LoaderCircle size={12} className="spin" />
                Checking a new reading…
              </>
            ) : (
              <>Last check · {time(decision?.t)}</>
            )}
          </div>
        </aside>
      </div>
      {latest?.injection && (
        <div className="map-injection">
          <Zap size={14} />
          Injected input: {latest.injection.replaceAll("_", " ")}
          <span>Excluded from scores</span>
        </div>
      )}
      {!latest?.injection && run?.fault && (
        <div className="map-injection">
          <Zap size={14} />
          Queued: {run.fault.kind.replaceAll("_", " ")}
          <span>
            {executing
              ? "Applies to the next reading"
              : "Resume or step to apply"}
          </span>
        </div>
      )}
      <div className="overview-bottom">
        <div className="mission-route" aria-label="Mission progress">
          {(run?.scenario || scenario) === "mission" ? (
            ["Monitor", "Improve", "Verify"].map((name, i) => (
              <span
                className={`${phase === i ? "current" : ""} ${phase > i ? "done" : ""}`}
                key={name}
              >
                <b>{phase > i ? <Check size={12} /> : i + 1}</b>
                {name}
                {i < 2 && <ChevronRight size={13} />}
              </span>
            ))
          ) : (
            <span className="current">
              <b>
                <Activity size={12} />
              </b>
              {(run?.scenario || scenario) === "sandbox"
                ? "Fault sandbox"
                : "Final test"}
            </span>
          )}
        </div>
        <span className="reading-counter">
          {learning
            ? `${run.trials.length} / 24 rules tested`
            : `${run?.tick_index || 0} / ${run?.total || (scenario === "sandbox" ? 48 : scenario === "test" ? 240 : 52)} readings`}
        </span>
        <button
          className="overview-text-button"
          onClick={() => setSheet("controls")}
        >
          <Zap size={14} />
          Try a fault
        </button>
      </div>
      {learning && (
        <div className="learning-strip">
          <FlaskConical size={19} />
          <div>
            <strong>
              {lastTrial
                ? `Candidate ${lastTrial.id}: ${lastTrial.accepted ? "kept" : "rejected"}`
                : "Testing better alarm rules"}
            </strong>
            <p>
              {lastTrial
                ? `Validation cost: ${lastTrial.metrics.cost}`
                : "Using the first 20 days only."}
            </p>
          </div>
          <progress value={run.trials.length} max={24} />
          <button className="overview-text-button" onClick={onDiagnostics}>
            Inspect tests <ChevronRight size={14} />
          </button>
        </div>
      )}
      {run?.selection && (
        <div className="map-result">
          <ShieldCheck size={19} />
          <div>
            <strong>
              {run.selection.promoted
                ? "A better rule is now active"
                : "The original rule stays active"}
            </strong>
            <span>
              Validation cost {run.selection.incumbent_cost} →{" "}
              {run.selection.selected_cost}
              {hasFinalScore
                ? ` · Final test: ${run.score!.true_positive}/${run.score!.bad_hours} bad hours flagged · ${run.score!.false_positive} false alarms`
                : ""}
            </span>
          </div>
          <button className="overview-text-button" onClick={onDiagnostics}>
            Details <ChevronRight size={14} />
          </button>
        </div>
      )}
      {!run?.selection && hasFinalScore && (
        <div className="map-result">
          <ShieldCheck size={19} />
          <div>
            <strong>
              Final test{" "}
              {run!.status === "completed" ? "complete" : "in progress"}
            </strong>
            <span>
              {run!.score!.true_positive}/{run!.score!.bad_hours} bad hours
              flagged · {run!.score!.false_positive} false alarms
            </span>
          </div>
        </div>
      )}
      <footer className="overview-note">
        <span>Historical demo · rule-based agent</span>
        <button onClick={onDiagnostics}>
          Advanced diagnostics <ChevronRight size={12} />
        </button>
      </footer>

      {sheet === "controls" && (
        <Sheet title="Demo controls" onClose={() => setSheet(null)}>
          <div className="sheet-settings">
            <label>
              Run type
              <select
                aria-label="Live mission"
                value={scenario}
                disabled={active}
                onChange={(e) => onScenario(e.target.value)}
              >
                <option value="mission">
                  Full demo · monitor, improve, verify
                </option>
                <option value="sandbox">Fault sandbox · 48 hours</option>
                <option value="test">Final test · 10 days</option>
              </select>
            </label>
            <label>
              Speed
              <select
                aria-label="Live feed speed"
                value={active ? run.speed : speed}
                onChange={(e) => onSpeed(Number(e.target.value))}
              >
                {[1, 2, 4, 8].map((n) => (
                  <option value={n} key={n}>
                    {n}×
                  </option>
                ))}
              </select>
            </label>
          </div>
          {!active && (
            <button
              className="overview-primary sheet-start"
              disabled={busy}
              onClick={async () => {
                await onStart();
                setSheet(null);
              }}
            >
              <Play size={15} />
              Start demo
            </button>
          )}
          <div className="sheet-run-buttons">
            <button
              disabled={!active || executing || busy}
              onClick={() => onCommand("step")}
            >
              <SkipForward size={16} />
              Step one reading
            </button>
            <button
              disabled={!active || busy}
              onClick={() => onCommand("cancel")}
            >
              <Square size={14} />
              Stop run
            </button>
          </div>
          {active && (
            <p className="sheet-hint">
              Stop the current run to change its type.
            </p>
          )}
          <h3>Change the next input</h3>
          <div className="fault-options">
            {[
              {
                id: "pressure_dip",
                name: "Pressure drop",
                hint: "Other sensors stay normal",
                icon: Activity,
              },
              {
                id: "pressure_offline",
                name: "Sensor outage",
                hint: "Pressure goes missing",
                icon: Unplug,
              },
              {
                id: "hydrate_pattern",
                name: "Deterioration",
                hint: "Pressure, temperature and flow fall",
                icon: Waves,
              },
              {
                id: "frozen_feed",
                name: "Frozen readings",
                hint: "The feed stops changing",
                icon: Snowflake,
              },
            ].map((fault) => (
              <button
                key={fault.id}
                disabled={!active || learning || busy}
                onClick={async () => {
                  await onCommand("inject", { fault: fault.id });
                  setSheet(null);
                }}
              >
                <fault.icon size={18} />
                <span>
                  <strong>{fault.name}</strong>
                  <small>{fault.hint}</small>
                </span>
                <ArrowRight size={14} />
              </button>
            ))}
          </div>
          <div className="queued-fault">
            <span>
              {run?.fault
                ? `${run.fault.kind.replaceAll("_", " ")} · ${run.fault.remaining} readings left`
                : "Original input active"}
            </span>
            <button
              disabled={!active || !run?.fault || busy}
              onClick={() => onCommand("inject", { fault: "restore" })}
            >
              <RotateCcw size={13} />
              Restore
            </button>
          </div>
          <p className="sheet-hint">
            Injected readings and the next five readings are excluded from
            scores.
          </p>
          <div className="sheet-links">
            <button
              onClick={() => {
                setSheet(null);
                onDiagnostics();
              }}
            >
              Advanced diagnostics <ChevronRight size={14} />
            </button>
            {run && (
              <a href={`/api/live/sessions/${run.id}/export`} download>
                <Download size={14} />
                Export run
              </a>
            )}
          </div>
        </Sheet>
      )}
      {sheet === "well" && (
        <Sheet title="Demo well 01" onClose={() => setSheet(null)}>
          <div className={`evidence-intro tone-${tone}`}>
            <span className="well-status">
              <span />
              {label}
            </span>
            <p>{reason}</p>
          </div>
          {latest?.injection && (
            <p className="sheet-injection">
              <Zap size={13} />
              Injected: {latest.injection.replaceAll("_", " ")}
            </p>
          )}
          <div className="evidence-chart-heading">
            <h3>Pressure</h3>
            <span>Last {chart.length} readings · bar</span>
          </div>
          <div className="evidence-chart">
            {chart.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={chart}
                  margin={{ left: -22, right: 12, top: 15, bottom: 0 }}
                >
                  <CartesianGrid
                    stroke="#e5ebe6"
                    vertical={false}
                    strokeDasharray="3 4"
                  />
                  <XAxis
                    dataKey="t"
                    tickFormatter={(t) =>
                      new Date(t).toLocaleTimeString("en", {
                        hour: "2-digit",
                        minute: "2-digit",
                        hour12: false,
                      })
                    }
                    tick={{ fontSize: 10 }}
                    minTickGap={40}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis
                    domain={["auto", "auto"]}
                    tick={{ fontSize: 10 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <Tooltip
                    labelFormatter={(t) => time(String(t))}
                    formatter={(v) => [number(Number(v)), "Pressure · bar"]}
                  />
                  <Line
                    type="linear"
                    dataKey="pressure"
                    stroke="#287b68"
                    strokeWidth={2}
                    dot={chart.length === 1}
                    connectNulls={false}
                    isAnimationActive={false}
                  />
                  {run && (
                    <ReferenceLine
                      y={run.policy.pressure_cutoff}
                      ifOverflow="extendDomain"
                      stroke="#bd9667"
                      strokeDasharray="4 4"
                      label={{
                        value: "Alarm threshold",
                        fontSize: 10,
                        position: "insideTopRight",
                        fill: "#9b784f",
                      }}
                    />
                  )}
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="empty-evidence">
                <Compass size={25} />
                <span>Start the demo to see readings.</span>
              </div>
            )}
          </div>
          <h3>What the agent checked</h3>
          <div className="simple-tool-list">
            {run?.tools?.length ? (
              run.tools.map((tool) => (
                <details key={tool.id}>
                  <summary>
                    {tool.status === "done" ? (
                      <Check size={15} />
                    ) : (
                      <LoaderCircle className="spin" size={15} />
                    )}
                    <span>
                      {toolNames[tool.name] || tool.name.replaceAll("_", " ")}
                    </span>
                    <ChevronRight size={13} />
                  </summary>
                  <p>{tool.result?.summary || "Checking this evidence…"}</p>
                </details>
              ))
            ) : (
              <p className="sheet-hint">Checks appear as readings arrive.</p>
            )}
          </div>
          <div className="evidence-next">
            <strong>{nextAction}</strong>
            {decision && (
              <span>Next check · {time(decision.next_recheck)}</span>
            )}
          </div>
          <p className="sheet-hint">
            Operator advice only. No equipment is controlled.
          </p>
        </Sheet>
      )}
    </section>
  );
}
