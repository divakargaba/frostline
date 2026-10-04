import { memo, useState } from "react";
import { ArrowDownRight, ArrowUpRight, Info, Minus } from "lucide-react";
import { number } from "./api";
import type { FleetInsights, SensorChannel, SensorInfo } from "./fleetTypes";

const trendIcon = (trend: SensorChannel["trend"]) =>
  trend === "rising" ? (
    <ArrowUpRight size={12} />
  ) : trend === "falling" ? (
    <ArrowDownRight size={12} />
  ) : (
    <Minus size={12} />
  );

/** Sensor reference plus the inferences drawn from this well's latest changes. */
export const SensorGuide = memo(function SensorGuide({
  sensors,
  insights,
}: {
  sensors: SensorInfo[];
  insights: FleetInsights | null | undefined;
}) {
  const [showWhy, setShowWhy] = useState(true);
  const channels = new Map(insights?.channels.map((ch) => [ch.code, ch]));
  const inferences = insights?.inferences ?? [];
  return (
    <div className="fleet-card fleet-sensors">
      <div className="fleet-card-heading">
        <div>
          <h2>Sensors</h2>
          <small>
            Last {insights?.window_minutes ?? 10} min · observations, not a diagnosis
          </small>
        </div>
        <button
          className="fleet-text-button"
          aria-pressed={showWhy}
          onClick={() => setShowWhy(!showWhy)}
        >
          <Info size={12} />
          {showWhy ? "Hide guide" : "Show guide"}
        </button>
      </div>
      <ul className="fleet-inferences" aria-label="Inferences">
        {inferences.length ? (
          inferences.map((item) => (
            <li key={item.id} className={item.level}>
              <i className={`fleet-dot ${item.level === "watch" ? "watch" : "normal"}`} />
              <div>
                <p>
                  {item.sensors.map((code) => (
                    <code key={code}>{code}</code>
                  ))}
                  {item.text}
                </p>
                <small>{item.why}</small>
              </div>
            </li>
          ))
        ) : (
          <li className="empty">
            {insights ? "No notable pressure or temperature change." : "Starts with the replay"}
          </li>
        )}
      </ul>
      <div
        className={`fleet-sensor-guide ${showWhy ? "why" : ""}`}
        role="table"
        aria-label="Sensor codes"
      >
        <div className="fleet-sensor-row head" role="row">
          <span role="columnheader">Code</span>
          <span role="columnheader">Measures</span>
          <span role="columnheader">Now</span>
          <span role="columnheader">Δ 10 min</span>
          {showWhy && <span role="columnheader">Why it matters</span>}
        </div>
        {sensors.map((info) => {
          const ch = channels.get(info.code);
          const trend = ch?.trend ?? "unavailable";
          return (
            <div className="fleet-sensor-row" role="row" key={info.code} title={ch?.note}>
              <span role="cell">
                <code>{info.code}</code>
              </span>
              <span role="cell">
                {info.name}
                <small>{info.location}</small>
              </span>
              <span role="cell">
                {trend === "inactive" ? "inactive" : number(ch?.value, 2)}
                {ch?.value != null && trend !== "inactive" && <small>{info.unit}</small>}
              </span>
              <span role="cell" className={`fleet-sensor-trend ${ch?.notable ? trend : ""}`}>
                {trend === "unavailable" || trend === "inactive" ? (
                  "—"
                ) : (
                  <>
                    {trendIcon(trend)}
                    {ch?.change != null && `${ch.change > 0 ? "+" : ""}${number(ch.change, 2)}`}
                  </>
                )}
              </span>
              {showWhy && (
                <span role="cell" className="fleet-sensor-why">
                  {info.why}
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
});
