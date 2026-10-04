import { memo, useMemo, useState } from "react";
import { Gauge, SlidersHorizontal } from "lucide-react";
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
import { number } from "./api";
import type { FleetFrame, FleetPrediction } from "./fleetTypes";

const PRESSURES = ["P-TPT", "P-MON-CKP", "P-PDG"];
const TEMPERATURES = ["T-TPT", "T-JUS-CKP", "T-PDG"];
const COLORS: Record<string, string> = {
  "P-TPT": "#138578",
  "P-MON-CKP": "#5fae9a",
  "P-PDG": "#2f5f53",
  "T-TPT": "#df9362",
  "T-JUS-CKP": "#c9a14a",
  "T-PDG": "#a8644a",
};
const clock = (t: string) => t.slice(11, 16);
const axisTick = { fontSize: 11, fill: "#5f7168" };

type Row = Record<string, number | string | null>;

function Tip({
  active,
  payload,
  label,
  units,
}: {
  active?: boolean;
  payload?: { dataKey?: unknown; value?: unknown; color?: string }[];
  label?: string | number;
  units: (key: string) => string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="fleet-chart-tooltip">
      {label}
      {payload.map((item) => (
        <div key={String(item.dataKey)} style={{ color: item.color }}>
          {String(item.dataKey)}{" "}
          <strong>
            {number(item.value as number, 2)} {units(String(item.dataKey))}
          </strong>
        </div>
      ))}
    </div>
  );
}

/** Full-replay pressure (left axis) and temperature (right axis) for one well. */
export const PressureTemperatureChart = memo(function PressureTemperatureChart({
  frames,
}: {
  frames: FleetFrame[];
}) {
  const available = useMemo(
    () =>
      [...PRESSURES, ...TEMPERATURES].filter((key) =>
        frames.some((f) => f.sensors[key] != null),
      ),
    [frames],
  );
  const [hidden, setHidden] = useState<Set<string>>(
    () => new Set(["P-PDG", "T-PDG"]),
  );
  const shown = available.filter((key) => !hidden.has(key));
  const data = useMemo<Row[]>(
    () =>
      frames.map((f) => ({
        t: clock(f.t),
        ...Object.fromEntries(available.map((key) => [key, f.sensors[key]])),
      })),
    [frames, available],
  );
  const start = frames.find((f) => f.elapsed_seconds >= 0);
  const hasPressure = shown.some((key) => key.startsWith("P-"));
  const hasTemperature = shown.some((key) => key.startsWith("T-"));
  return (
    <div className="fleet-card fleet-series">
      <div className="fleet-card-heading">
        <div>
          <h2>Pressure & temperature</h2>
          <small>Source time · warm-up left of the marker</small>
        </div>
        <div className="fleet-series-chips" role="group" aria-label="Channels">
          {available.map((key) => (
            <button
              key={key}
              aria-pressed={!hidden.has(key)}
              onClick={() =>
                setHidden((prev) => {
                  const next = new Set(prev);
                  if (next.has(key)) next.delete(key);
                  else next.add(key);
                  return next;
                })
              }
            >
              <i style={{ background: COLORS[key] }} />
              {key}
            </button>
          ))}
        </div>
      </div>
      {frames.length && shown.length ? (
        <div className="fleet-series-area">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 12, right: 6, left: 6, bottom: 0 }}>
              <CartesianGrid vertical={false} stroke="#edf1e8" strokeDasharray="2 4" />
              <XAxis dataKey="t" tick={axisTick} axisLine={false} tickLine={false} minTickGap={36} />
              <YAxis
                yAxisId="p"
                hide={!hasPressure}
                domain={["auto", "auto"]}
                tick={axisTick}
                axisLine={false}
                tickLine={false}
                width={44}
                label={{ value: "bar", angle: -90, position: "insideLeft", fill: "#8d9b92", fontSize: 11 }}
              />
              <YAxis
                yAxisId="T"
                orientation="right"
                hide={!hasTemperature}
                domain={["auto", "auto"]}
                tick={axisTick}
                axisLine={false}
                tickLine={false}
                width={40}
                label={{ value: "°C", angle: 90, position: "insideRight", fill: "#8d9b92", fontSize: 11 }}
              />
              {start && (
                <ReferenceLine
                  yAxisId={hasPressure ? "p" : "T"}
                  x={clock(start.t)}
                  stroke="#9fb3a8"
                  strokeDasharray="3 3"
                  label={{ value: "replay", position: "insideTopLeft", fill: "#8d9b92", fontSize: 10 }}
                />
              )}
              <Tooltip
                content={(props) => (
                  <Tip {...(props as object)} units={(key) => (key.startsWith("P-") ? "bar" : "°C")} />
                )}
              />
              {shown.map((key) => (
                <Line
                  key={key}
                  yAxisId={key.startsWith("P-") ? "p" : "T"}
                  dataKey={key}
                  stroke={COLORS[key]}
                  strokeWidth={1.7}
                  dot={false}
                  isAnimationActive={false}
                  connectNulls={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div className="fleet-chart-empty fleet-series-area">
          {frames.length ? "No channel selected" : "Waiting for readings"}
        </div>
      )}
    </div>
  );
});

/** The decision limits applied each minute, and how they moved. */
export const ThresholdTracker = memo(function ThresholdTracker({
  frames,
  prediction,
}: {
  frames: FleetFrame[];
  prediction: FleetPrediction | null | undefined;
}) {
  const steps = useMemo(() => frames.filter((f) => f.limits), [frames]);
  const data = useMemo<Row[]>(
    () =>
      steps.map((f) => ({
        t: clock(f.t),
        score: f.risk_score,
        cutoff: f.limits!.hydrate_threshold,
        change: f.limits!.pressure_change_bar == null ? null : Math.abs(f.limits!.pressure_change_bar),
        divergence:
          f.limits!.divergence_change_bar == null ? null : Math.abs(f.limits!.divergence_change_bar),
        trigger: f.limits!.pressure_trigger_bar,
      })),
    [steps],
  );
  // Only minutes where a limit, the alarm state or a streak moved; the latest
  // minute is always shown so the log reflects the current update.
  const log = useMemo(() => {
    const rows = steps.map((f, i) => ({ f, prev: steps[i - 1]?.limits ?? null }));
    const moved = rows.filter(({ f, prev }, i) => {
      const lim = f.limits!;
      return (
        i === rows.length - 1 ||
        !prev ||
        prev.hydrate_threshold !== lim.hydrate_threshold ||
        prev.alarm_active !== lim.alarm_active ||
        prev.activation_streak !== lim.activation_streak ||
        prev.recovery_streak !== lim.recovery_streak ||
        Math.abs((prev.pressure_trigger_bar ?? 0) - (lim.pressure_trigger_bar ?? 0)) >= 0.005
      );
    });
    return moved.slice(-8).reverse();
  }, [steps]);
  const last = steps.at(-1)?.limits;
  const persistence = prediction?.persistence_minutes ?? 10;
  const recovery = prediction?.recovery_minutes ?? 5;
  const state = !last
    ? "—"
    : last.alarm_active
      ? `Alarm · clearing ${last.recovery_streak}/${recovery}`
      : last.activation_streak
        ? `Arming ${last.activation_streak}/${persistence}`
        : "Idle";
  const tone = !last ? "pending" : last.alarm_active ? "attention" : last.activation_streak ? "watch" : "normal";
  const units = (key: string) => (key === "score" || key === "cutoff" ? "" : "bar");
  return (
    <div className="fleet-card fleet-limits">
      <div className="fleet-card-heading">
        <div>
          <h2>Thresholds</h2>
          <small>Applied each minute · log shows changes</small>
        </div>
        <span className={`fleet-model-call ${tone}`}>
          <i className={`fleet-dot ${tone}`} />
          {state}
        </span>
      </div>
      <div className="fleet-charts fleet-limits-charts">
        <div className="fleet-chart">
          <div className="fleet-chart-label">
            <SlidersHorizontal size={12} />
            Hydrate cutoff
          </div>
          <div className="fleet-chart-value">
            {number(last?.hydrate_threshold, 2)}
            <small>score {number(steps.at(-1)?.risk_score, 2)}</small>
          </div>
          <div className="fleet-chart-name">
            {prediction?.threshold != null && prediction.recovery_threshold != null
              ? `${prediction.threshold.toFixed(2)} to raise · ${prediction.recovery_threshold.toFixed(2)} to clear`
              : "Score vs cutoff"}
          </div>
          <LimitChart
            data={data}
            lines={[
              { key: "score", color: "#b6a76c" },
              { key: "cutoff", color: "#bd5a43", step: true },
            ]}
            domain={[0, 1]}
            units={units}
          />
        </div>
        <div className="fleet-chart">
          <div className="fleet-chart-label">
            <Gauge size={12} />
            Pressure trigger
          </div>
          <div className="fleet-chart-value">
            {number(last?.pressure_trigger_bar, 2)}
            <small>bar · |ΔP| {number(last?.pressure_change_bar == null ? null : Math.abs(last.pressure_change_bar), 2)}</small>
          </div>
          <div className="fleet-chart-name">max(2 bar, 2% of P-TPT) · 10-min window</div>
          <LimitChart
            data={data}
            lines={[
              { key: "change", color: "#608e72" },
              { key: "divergence", color: "#6688a5" },
              { key: "trigger", color: "#bd5a43", step: true },
            ]}
            domain={[0, "auto"]}
            units={units}
          />
        </div>
      </div>
      <div className="fleet-limits-log" role="table" aria-label="Threshold updates">
        <div className="fleet-limits-row head" role="row">
          <span role="columnheader">Time</span>
          <span role="columnheader">Score</span>
          <span role="columnheader">Cutoff</span>
          <span role="columnheader">Streak</span>
          <span role="columnheader">|ΔP| 10 min</span>
          <span role="columnheader">Trigger</span>
        </div>
        {log.length ? (
          log.map(({ f, prev }) => {
            const lim = f.limits!;
            const cutoffMoved = prev && prev.hydrate_threshold !== lim.hydrate_threshold;
            const delta =
              prev?.pressure_trigger_bar != null && lim.pressure_trigger_bar != null
                ? lim.pressure_trigger_bar - prev.pressure_trigger_bar
                : null;
            return (
              <div className="fleet-limits-row" role="row" key={f.t}>
                <span role="cell">{clock(f.t)}</span>
                <span role="cell">{number(f.risk_score, 2)}</span>
                <span role="cell" className={cutoffMoved ? "moved" : ""}>
                  {cutoffMoved && `${prev!.hydrate_threshold.toFixed(2)} → `}
                  {lim.hydrate_threshold.toFixed(2)}
                </span>
                <span role="cell">
                  {lim.alarm_active
                    ? `clear ${lim.recovery_streak}/${recovery}`
                    : `${lim.activation_streak}/${persistence}`}
                </span>
                <span role="cell">
                  {number(lim.pressure_change_bar == null ? null : Math.abs(lim.pressure_change_bar), 2)}
                </span>
                <span role="cell" className={delta && Math.abs(delta) >= 0.005 ? "moved" : ""}>
                  {number(lim.pressure_trigger_bar, 2)}
                  {delta && Math.abs(delta) >= 0.005 ? (
                    <small> {delta > 0 ? "+" : ""}{delta.toFixed(2)}</small>
                  ) : null}
                </span>
              </div>
            );
          })
        ) : (
          <div className="fleet-limits-row empty">Starts with the replay</div>
        )}
      </div>
    </div>
  );
});

function LimitChart({
  data,
  lines,
  domain,
  units,
}: {
  data: Row[];
  lines: { key: string; color: string; step?: boolean }[];
  domain: [number, number | "auto"];
  units: (key: string) => string;
}) {
  if (!data.length) return <div className="fleet-chart-empty">Waiting for readings</div>;
  return (
    <div className="fleet-chart-area">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 7, left: 0, bottom: 0 }}>
          <CartesianGrid vertical={false} stroke="#edf1e8" strokeDasharray="2 4" />
          <XAxis dataKey="t" tick={axisTick} axisLine={false} tickLine={false} minTickGap={28} />
          <YAxis hide domain={domain} />
          <Tooltip content={(props) => <Tip {...(props as object)} units={units} />} />
          {lines.map((line) => (
            <Line
              key={line.key}
              type={line.step ? "stepAfter" : "monotone"}
              dataKey={line.key}
              stroke={line.color}
              strokeWidth={line.step ? 1.3 : 1.7}
              strokeDasharray={line.step ? "4 3" : undefined}
              dot={false}
              isAnimationActive={false}
              connectNulls={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
