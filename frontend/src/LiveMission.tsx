import { useEffect, useRef, useState } from "react";
import {
  Activity,
  ArrowRight,
  Check,
  CheckCircle2,
  ChevronDown,
  CircleHelp,
  Download,
  FlaskConical,
  LoaderCircle,
  Pause,
  Play,
  Radio,
  RotateCcw,
  ShieldCheck,
  SkipForward,
  Snowflake,
  Square,
  Unplug,
  Waves,
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
import type { Metrics, Policy } from "./types";
import "./live.css";

type Status = "paused" | "running" | "completed" | "cancelled" | "failed";
type Reading = {
  t: string;
  sensors: Record<string, number | null>;
  index: number;
  injection: string | null;
  decision: string | null;
  ground_truth?: string | null;
};
type Decision = {
  t: string;
  decision: string;
  diagnosis: string;
  rationale: string;
  notify: boolean;
  quality: string;
  policy_id: string;
  next_recheck: string;
  recheck_hours: number;
  cooldown_remaining_hours: number;
};
type Score = Metrics & { injected_excluded: number; processed: number };
type Trial = {
  id: number;
  name: string;
  policy: Policy;
  metrics: Metrics;
  accepted: boolean;
  reason: string;
};
type Selection = {
  promoted: boolean;
  trial: number | null;
  incumbent_cost: number;
  selected_cost: number;
  policy_id: string;
};
type Tool = {
  id: number;
  name: string;
  args: unknown;
  result?: { summary: string };
  status: "running" | "done";
};
type Payload = Partial<Reading> & {
  phase?: string;
  status?: Status;
  stepping?: boolean;
  speed?: number;
  total?: number;
  fault?: Run["fault"];
  policy?: Policy;
  policy_id?: string;
  score?: Score | null;
  selection?: Selection;
  trial?: Trial | number;
  tool?: string;
  args?: unknown;
  result?: { summary: string };
  label?: string | null;
  tool_calls?: number;
  injected_excluded?: number;
};
type AgentEvent = {
  id: number;
  type: string;
  title: string;
  message: string;
  phase: string;
  server_time: string;
  payload: Payload;
};
type Run = {
  id: string;
  scenario: string;
  status: Status;
  stepping: boolean;
  phase: string;
  speed: number;
  tick_index: number;
  total: number;
  policy: Policy;
  policy_id: string;
  frames: Reading[];
  decision: Decision | null;
  score: Score | null;
  trials: Trial[];
  selection: Selection | null;
  events: AgentEvent[];
  last_event_id: number;
  tool_calls: number;
  injected_excluded: number;
  fault: { kind: string; remaining: number } | null;
  error: string | null;
  tools?: Tool[];
};
const ended = (status?: string) =>
  ["completed", "cancelled", "failed"].includes(status || "");
const formatTime = (value?: string) =>
  value
    ? new Date(value).toLocaleString("en", {
        month: "short",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        hour12: false,
      })
    : "Waiting for input";
const value = (n: number | null | undefined, digits = 1) =>
  n == null ? "—" : n.toFixed(digits);
const storageKey = "frostline.live.session.v1";
async function api<T>(url: string, body?: unknown): Promise<T> {
  const response = await fetch(
    url,
    body === undefined
      ? undefined
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw Object.assign(
      new Error(detail.detail || `Request failed (${response.status})`),
      { status: response.status },
    );
  }
  return response.json();
}

function applyEvent(run: Run, event: AgentEvent): Run {
  const p = event.payload;
  let next: Run = {
    ...run,
    events: [...run.events, event].slice(-120),
    last_event_id: event.id,
  };
  if (event.type === "status")
    next = {
      ...next,
      status: p.status!,
      speed: p.speed!,
      stepping: p.stepping || false,
    };
  if (event.type === "intervention") next.fault = p.fault || null;
  if (event.type === "phase") {
    next.phase = p.phase!;
    next.tools = [];
    if (p.policy)
      next = {
        ...next,
        policy: p.policy,
        policy_id: p.policy_id!,
        frames: [],
        tick_index: 0,
        total: p.total!,
        score: null,
        decision: null,
        injected_excluded: 0,
      };
  }
  if (event.type === "tick")
    next = {
      ...next,
      frames: [...next.frames, p as Reading],
      tick_index: p.index!,
      total: p.total!,
      decision: null,
      tools: [],
      fault: p.fault || null,
    };
  if (event.type === "tool_call")
    next.tools = [
      ...(next.tools || []),
      { id: event.id, name: p.tool!, args: p.args, status: "running" },
    ];
  if (event.type === "tool_result") {
    next.tool_calls += 1;
    next.tools = (next.tools || []).map((t) =>
      t.name === p.tool && t.status === "running"
        ? { ...t, status: "done", result: p.result }
        : t,
    );
  }
  if (event.type === "decision") {
    next.decision = p as unknown as Decision;
    next.frames = next.frames.map((f, i) =>
      i === next.frames.length - 1 ? { ...f, decision: p.decision! } : f,
    );
  }
  if (event.type === "outcome") {
    next.score = p.score || null;
    next.tool_calls = p.tool_calls || 0;
    next.injected_excluded = p.injected_excluded || 0;
    next.frames = next.frames.map((f, i) =>
      i === next.frames.length - 1 ? { ...f, ground_truth: p.label } : f,
    );
  }
  if (event.type === "candidate_result" && typeof p.trial === "object") {
    next.tool_calls += 1;
    next.trials = [...next.trials, p.trial];
    next.tools = (next.tools || []).map((t) => ({ ...t, status: "done" }));
  }
  if (event.type === "promotion")
    next = {
      ...next,
      policy: p.policy!,
      policy_id: p.policy_id!,
      selection: p.selection!,
    };
  if (["complete", "cancelled", "failed"].includes(event.type))
    next = {
      ...next,
      status: p.status!,
      phase: event.type === "complete" ? "complete" : next.phase,
      error: event.type === "failed" ? event.message : null,
    };
  return next;
}

function restoreTools(snapshot: Run): Run {
  const boundary = snapshot.events.reduce(
    (last, e, i) => (["tick", "proposal", "phase"].includes(e.type) ? i : last),
    -1,
  );
  const tools: Tool[] = [];
  for (const event of snapshot.events.slice(boundary + 1)) {
    const p = event.payload;
    if (event.type === "tool_call")
      tools.push({
        id: event.id,
        name: p.tool!,
        args: p.args,
        status: "running",
      });
    if (event.type === "tool_result") {
      const tool = [...tools]
        .reverse()
        .find((t) => t.name === p.tool && t.status === "running");
      if (tool) Object.assign(tool, { status: "done", result: p.result });
    }
    if (event.type === "candidate_result")
      tools.forEach((tool) => {
        tool.status = "done";
      });
  }
  return { ...snapshot, tools };
}

export default function LiveMission() {
  const [runId, setRunId] = useState<string | null>(() =>
    localStorage.getItem(storageKey),
  );
  const [run, setRun] = useState<Run | null>(null);
  const [scenario, setScenario] = useState("mission");
  const [speed, setSpeed] = useState(2);
  const [connected, setConnected] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [journalOpen, setJournalOpen] = useState(false);
  const source = useRef<EventSource | null>(null);
  const lastId = useRef(0);

  useEffect(() => {
    if (!runId) return;
    let alive = true;
    api<Run>(`/api/live/sessions/${runId}`)
      .then((snapshot) => {
        if (!alive) return;
        setRun(restoreTools(snapshot));
        setScenario(snapshot.scenario);
        setSpeed(snapshot.speed);
        lastId.current = snapshot.last_event_id;
        if (ended(snapshot.status)) return;
        const stream = new EventSource(
          `/api/live/sessions/${runId}/events?after=${snapshot.last_event_id}`,
        );
        source.current = stream;
        stream.onopen = () => {
          if (alive) {
            setConnected(true);
            setError("");
          }
        };
        stream.onerror = () => {
          if (!alive) return;
          setConnected(false);
          fetch(`/api/live/sessions/${runId}`)
            .then((response) => {
              if (alive && response.status === 404) {
                stream.close();
                setRun(null);
                setRunId(null);
                localStorage.removeItem(storageKey);
                setError(
                  "The server restarted and cleared this session. Start a new mission.",
                );
              }
            })
            .catch(() => {});
        };
        stream.addEventListener("agent", (raw) => {
          const event: AgentEvent = JSON.parse((raw as MessageEvent).data);
          if (event.id <= lastId.current || !alive) return;
          lastId.current = event.id;
          setRun((current) => (current ? applyEvent(current, event) : current));
          if (
            ["complete", "cancelled", "failed"].includes(event.type) ||
            ended(event.payload.status)
          ) {
            stream.close();
            setConnected(false);
          }
        });
      })
      .catch((e) => {
        if (alive) {
          setError(e.message);
          setRun(null);
          setRunId(null);
          localStorage.removeItem(storageKey);
        }
      });
    return () => {
      alive = false;
      source.current?.close();
      source.current = null;
      setConnected(false);
    };
  }, [runId]);

  async function start() {
    setBusy(true);
    setError("");
    try {
      if (run && !ended(run.status))
        await api(`/api/live/sessions/${run.id}/control`, { action: "cancel" });
      const next = await api<Run>("/api/live/sessions", { scenario, speed });
      setRun(next);
      setRunId(next.id);
      localStorage.setItem(storageKey, next.id);
      await api(`/api/live/sessions/${next.id}/control`, { action: "resume" });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function command(action: string, extra: object = {}) {
    if (!run) return;
    setBusy(true);
    setError("");
    try {
      await api(`/api/live/sessions/${run.id}/control`, { action, ...extra });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  const active = !!run && !ended(run.status);
  const latest = run?.frames.at(-1);
  const event = run?.events.at(-1);
  const learning = run?.phase === "learn";
  const decision = run?.decision;
  const status = run?.stepping ? "stepping" : run?.status || "ready";
  const executing = run?.status === "running" || run?.stepping;
  const events = (run?.events || []).filter(
    (e) => !["tick", "outcome", "status", "tool_result"].includes(e.type),
  );
  const currentTrial = run?.trials.at(-1);
  const phaseIndex =
    run?.phase === "learn"
      ? 1
      : ["prove", "complete"].includes(run?.phase || "")
        ? 2
        : 0;
  const chartData = (run?.frames || []).map((f) => ({
    ...f.sensors,
    t: new Date(f.t).getTime(),
    injection: f.injection,
  }));

  return (
    <section className="live-room">
      <div className="page-heading">
        <div>
          <div className="eyebrow">THE AGENT, AT WORK</div>
          <h1>Give it a well. Watch it work.</h1>
          <p>
            Incoming readings → conditional investigations → decisions →
            measured improvement.
          </p>
        </div>
        <span className={`live-connection ${connected ? "connected" : ""}`}>
          <span />
          {connected
            ? "Event stream connected"
            : ended(run?.status)
              ? "Run saved in this server session"
              : runId
                ? "Connecting to event stream…"
                : "Ready to start"}
        </span>
      </div>
      {(error || run?.error) && (
        <div className="error" role="alert">
          <CircleHelp size={17} />
          <span>{error || run?.error}</span>
        </div>
      )}
      <div className="mission-console">
        <div className="mission-title">
          <div className="mission-icon">
            <Waves size={24} />
          </div>
          <div>
            <span className="tiny-label">AUTONOMOUS MISSION</span>
            <h2>
              {learning
                ? "Testing the next idea."
                : run?.phase === "prove"
                  ? "Proving the frozen policy."
                  : run?.status === "completed"
                    ? "Evidence collected. Mission complete."
                    : "Observe. Investigate. Improve."}
            </h2>
            <p>
              Historical feed · decisions computed as inputs arrive · stateful
              deterministic controller
            </p>
          </div>
          <span className={`mission-status ${status}`}>
            <span />
            {status}
          </span>
        </div>
        <div className="mission-settings">
          <label>
            Mission
            <select
              aria-label="Live mission"
              value={scenario}
              disabled={active}
              onChange={(e) => setScenario(e.target.value)}
            >
              <option value="mission">Learn, then prove · full mission</option>
              <option value="sandbox">
                Fault-injection sandbox · 48 hours
              </option>
              <option value="test">Final test · selected policy</option>
            </select>
          </label>
          <label>
            Feed speed
            <select
              aria-label="Live feed speed"
              value={run && active ? run.speed : speed}
              onChange={(e) => {
                setSpeed(Number(e.target.value));
                if (active)
                  void command("speed", { speed: Number(e.target.value) });
              }}
            >
              {[1, 2, 4, 8].map((s) => (
                <option key={s} value={s}>
                  {s}×
                </option>
              ))}
            </select>
          </label>
          <div className="mission-buttons">
            {!active ? (
              <button
                className="button mission-start"
                disabled={busy}
                onClick={start}
              >
                {busy ? (
                  <LoaderCircle size={17} className="spin" />
                ) : (
                  <Play size={17} />
                )}{" "}
                {run ? "Start a new mission" : "Start mission"}
              </button>
            ) : (
              <>
                <button
                  className="button mission-start"
                  disabled={busy}
                  onClick={() => command(executing ? "pause" : "resume")}
                >
                  {executing ? <Pause size={16} /> : <Play size={16} />}{" "}
                  {executing ? "Pause feed" : "Resume feed"}
                </button>
                <button
                  className="mission-icon-button"
                  title="Finish this reading or process the next reading or candidate"
                  aria-label="Step one input"
                  disabled={busy || run.status !== "paused" || run.stepping}
                  onClick={() => command("step")}
                >
                  <SkipForward size={18} />
                </button>
                <button
                  className="mission-icon-button"
                  title="Stop this run"
                  aria-label="Stop mission"
                  disabled={busy}
                  onClick={() => command("cancel")}
                >
                  <Square size={15} />
                </button>
              </>
            )}
          </div>
        </div>
        <div className="mission-stages">
          {[
            {
              label: "Observe & investigate",
              sub: "A 52-hour validation window",
              icon: Radio,
            },
            {
              label: "Test & improve",
              sub: "24 candidate rules · separate validation",
              icon: FlaskConical,
            },
            {
              label: "Freeze & prove",
              sub: "240 held-out test readings",
              icon: ShieldCheck,
            },
          ].map((s, i) => (
            <div
              key={s.label}
              className={`${run && phaseIndex === i ? "current" : ""} ${run && phaseIndex > i ? "done" : ""}`}
            >
              <span>
                {run && phaseIndex > i ? (
                  <Check size={16} />
                ) : (
                  <s.icon size={16} />
                )}
              </span>
              <div>
                <strong>{s.label}</strong>
                <small>
                  {(run?.scenario || scenario) !== "mission"
                    ? i === 1
                      ? "Available in the full mission"
                      : i === 0
                        ? "Incremental sensor processing"
                        : "Evaluate after every decision"
                    : s.sub}
                </small>
              </div>
              {i < 2 && <ArrowRight size={14} />}
            </div>
          ))}
        </div>
      </div>

      <div className="live-metrics">
        <div>
          <span>Inputs processed</span>
          <strong>
            {run?.tick_index || 0}
            <small>
              {" "}
              /{" "}
              {run?.total ||
                (scenario === "sandbox" ? 48 : scenario === "test" ? 240 : 52)}
            </small>
          </strong>
          <p>
            {run?.phase === "prove"
              ? "Final test"
              : "Current observation phase"}
          </p>
        </div>
        <div>
          <span>Tools executed</span>
          <strong>{run?.tool_calls || 0}</strong>
          <p>Conditional on incoming evidence</p>
        </div>
        <div>
          <span>Bad hours detected so far</span>
          <strong>
            {run?.score?.true_positive ?? "—"}
            <small> / {run?.score?.bad_hours ?? "—"}</small>
          </strong>
          <p>Evaluated after the decision</p>
        </div>
        <div>
          <span>False-alarm hours so far</span>
          <strong>{run?.score?.false_positive ?? "—"}</strong>
          <p>
            {run?.injected_excluded || 0} intervention-context rows excluded
          </p>
        </div>
      </div>

      <div className="live-workspace">
        <div className="live-main-column">
          <section className="panel live-sensor-panel">
            <div className="panel-heading">
              <div>
                <h3>
                  {learning ? "The validation workbench" : "Incoming telemetry"}
                </h3>
                <p>
                  {learning
                    ? "Each proposal below is evaluated by the backend during this run."
                    : "The chart grows only when the server processes a new input."}
                </p>
              </div>
              <span className="tag">
                {learning
                  ? `${run.trials.length}/24 scored`
                  : latest
                    ? formatTime(latest.t)
                    : "Awaiting first input"}
              </span>
            </div>
            {learning ? (
              <div className="live-search">
                <div className="search-progress">
                  <FlaskConical size={27} />
                  <div>
                    <strong>
                      {event?.type === "proposal"
                        ? event.title
                        : currentTrial
                          ? `Candidate ${currentTrial.id} evaluated`
                          : "Measuring the incumbent"}
                    </strong>
                    <p>{event?.message}</p>
                  </div>
                  {active && executing && (
                    <LoaderCircle className="spin" size={20} />
                  )}
                </div>
                <div className="candidate-dots">
                  {Array.from({ length: 24 }, (_, i) => {
                    const t = run.trials[i];
                    return (
                      <div
                        key={i}
                        className={t ? (t.accepted ? "kept" : "rejected") : ""}
                      >
                        <span>{String(i + 1).padStart(2, "0")}</span>
                        <b>{t ? t.metrics.cost : "·"}</b>
                        <small>
                          {t ? (t.accepted ? "KEEP" : "REJECT") : "PENDING"}
                        </small>
                      </div>
                    );
                  })}
                </div>
                <p className="live-small-note">
                  Objective: 5 × missed bad hours + false-alarm hours. A
                  candidate must beat the current best cost to be kept.
                </p>
              </div>
            ) : (
              <>
                {latest?.injection && (
                  <div className="injected-reading">
                    <Zap size={13} /> Injected input:{" "}
                    {latest.injection.replaceAll("_", " ")} · excluded from
                    benchmark scores
                  </div>
                )}
                <div className="live-sensor-values">
                  {[
                    { key: "pressure_bar", label: "Pressure", unit: "bar" },
                    { key: "temp_C", label: "Temperature", unit: "°C" },
                    { key: "flow_Lps", label: "Flow", unit: "L/s" },
                  ].map((s) => (
                    <div key={s.key}>
                      <span>{s.label}</span>
                      <strong>
                        {value(latest?.sensors[s.key])}
                        <small>{s.unit}</small>
                      </strong>
                    </div>
                  ))}
                </div>
                <div className="live-chart">
                  {chartData.length ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart
                        data={chartData}
                        margin={{ top: 12, right: 16, left: -16, bottom: 0 }}
                      >
                        <CartesianGrid
                          stroke="#e9eeea"
                          strokeDasharray="3 4"
                          vertical={false}
                        />
                        <XAxis
                          type="number"
                          dataKey="t"
                          domain={["dataMin", "dataMax"]}
                          tickFormatter={(v) =>
                            new Date(v).toLocaleString("en", {
                              day: "2-digit",
                              hour: "2-digit",
                              minute: "2-digit",
                              hour12: false,
                            })
                          }
                          tick={{ fontSize: 10, fill: "#718579" }}
                          axisLine={false}
                          tickLine={false}
                          minTickGap={55}
                        />
                        <YAxis
                          domain={["auto", "auto"]}
                          tick={{ fontSize: 10, fill: "#718579" }}
                          axisLine={false}
                          tickLine={false}
                        />
                        <Tooltip
                          labelFormatter={(v) =>
                            formatTime(new Date(Number(v)).toISOString())
                          }
                          formatter={(v) => [
                            value(Number(v), 2),
                            "Pressure · bar",
                          ]}
                          contentStyle={{
                            fontSize: 11,
                            borderRadius: 8,
                            border: "1px solid #dce6dd",
                          }}
                        />
                        <Line
                          dataKey="pressure_bar"
                          type="linear"
                          stroke="#188575"
                          strokeWidth={2}
                          dot={chartData.length === 1}
                          isAnimationActive={false}
                          connectNulls={false}
                        />
                        {run && (
                          <ReferenceLine
                            y={run.policy.pressure_cutoff}
                            ifOverflow="extendDomain"
                            stroke="#cb996d"
                            strokeDasharray="5 4"
                            label={{
                              value: `Active cutoff ${run.policy.pressure_cutoff.toFixed(1)} bar`,
                              fontSize: 9,
                              fill: "#aa805b",
                              position: "insideTopRight",
                            }}
                          />
                        )}
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="live-waiting">
                      <Radio size={32} />
                      <h3>Waiting for the first reading</h3>
                      <p>
                        Start a mission. Investigations, decisions and scores
                        will appear here as the backend executes them.
                      </p>
                      <div className="waiting-wave">
                        {[12, 22, 16, 31, 42, 25, 56, 33, 20, 38, 23, 16].map(
                          (h, i) => (
                            <i key={i} style={{ height: h }} />
                          ),
                        )}
                      </div>
                    </div>
                  )}
                </div>
                <div className="feed-progress">
                  <div>
                    <span
                      style={{
                        width: `${run && run.total ? (run.tick_index / run.total) * 100 : 0}%`,
                      }}
                    />
                  </div>
                  <span>
                    {run?.stepping
                      ? "Processing one input, then pausing"
                      : run?.status === "paused"
                        ? "Feed paused — step through one input at a time"
                        : run?.status === "running"
                          ? "Processing historical hours at an accelerated demo pace"
                          : run?.status === "completed"
                            ? "All inputs in this phase processed"
                            : "No prerecorded decision timeline is used in this room"}
                  </span>
                </div>
              </>
            )}
          </section>

          <section className="panel intervention-panel">
            <div className="panel-heading">
              <div>
                <h3>Change what the agent sees</h3>
                <p>
                  Apply to the next input. Watch the tools and decision change.
                </p>
              </div>
              <Zap size={19} />
            </div>
            <div className="intervention-grid">
              {[
                {
                  id: "pressure_dip",
                  label: "Pressure dip",
                  sub: "1 sample · other sensors normal",
                  icon: Activity,
                },
                {
                  id: "pressure_offline",
                  label: "Sensor offline",
                  sub: "4 samples · pressure missing",
                  icon: Unplug,
                },
                {
                  id: "hydrate_pattern",
                  label: "Deteriorating conditions",
                  sub: "4 samples · pressure, temperature, flow",
                  icon: Waves,
                },
                {
                  id: "frozen_feed",
                  label: "Freeze the feed",
                  sub: "8 samples · unchanged telemetry",
                  icon: Snowflake,
                },
              ].map((f) => (
                <button
                  key={f.id}
                  disabled={!active || learning || busy}
                  onClick={() => command("inject", { fault: f.id })}
                >
                  <f.icon size={18} />
                  <span>
                    <strong>{f.label}</strong>
                    <small>{f.sub}</small>
                  </span>
                  <span className="inject-plus">+</span>
                </button>
              ))}
            </div>
            <div className="intervention-footer">
              <span>
                {run?.fault
                  ? `${run.fault.kind.replaceAll("_", " ")} · ${run.fault.remaining} input(s) remaining`
                  : "Original historical input active"}
              </span>
              <button
                disabled={!active || !run?.fault || busy}
                onClick={() => command("inject", { fault: "restore" })}
              >
                <RotateCcw size={12} /> Restore input
              </button>
            </div>
            <p className="live-small-note">
              Synthetic interventions are visibly tagged. Changed rows and their
              next five history-dependent rows are excluded from benchmark
              scores. Policy selection uses the original validation data.
            </p>
          </section>

          <section className="panel live-journal">
            <div className="panel-heading">
              <div>
                <h3>The event journal</h3>
                <p>
                  Server events, ordered by execution. Click an event to inspect
                  its data.
                </p>
              </div>
              {run && (
                <a
                  className="button secondary"
                  href={`/api/live/sessions/${run.id}/export`}
                  download
                >
                  <Download size={14} /> Export run
                </a>
              )}
            </div>
            {events.length ? (
              <div className="journal-list">
                {events
                  .slice(journalOpen ? -100 : -12)
                  .reverse()
                  .map((e) => (
                    <details
                      key={e.id}
                      className={`journal-event event-${e.type}`}
                    >
                      <summary>
                        <span className="journal-id">
                          {String(e.id).padStart(3, "0")}
                        </span>
                        <span className="journal-dot" />
                        <div>
                          <strong>{e.title}</strong>
                          <p>{e.message}</p>
                        </div>
                        <time>
                          {new Date(e.server_time).toLocaleTimeString("en", {
                            hour12: false,
                          })}
                        </time>
                        <ChevronDown size={12} />
                      </summary>
                      <pre>{JSON.stringify(e.payload, null, 2)}</pre>
                    </details>
                  ))}
              </div>
            ) : (
              <div className="journal-empty">
                The first event will appear when you start a mission.
              </div>
            )}
            {events.length > 12 && (
              <button
                className="journal-toggle"
                onClick={() => setJournalOpen(!journalOpen)}
              >
                {journalOpen ? "Show recent events" : "Show more events"}{" "}
                <ChevronDown size={13} />
              </button>
            )}
          </section>
        </div>

        <aside className="live-side-column">
          <section
            className={`live-current-action ${decision?.decision.toLowerCase() || ""}`}
          >
            <div className="action-heading">
              <span className="tiny-label">WHAT THE AGENT IS DOING</span>
              {active && executing ? (
                <span className="agent-pulse" />
              ) : (
                <Radio size={15} />
              )}
            </div>
            <div className="action-icon">
              {learning ? (
                <FlaskConical size={27} />
              ) : decision?.decision === "ALERT" ? (
                <Activity size={27} />
              ) : decision?.decision === "WATCH" ? (
                <CircleHelp size={27} />
              ) : (
                <ScanIcon />
              )}
            </div>
            <h2>
              {learning
                ? event?.title || "Preparing proposals"
                : decision?.diagnosis ||
                  (latest ? "Investigating this input…" : "Ready for a well")}
            </h2>
            <p>
              {learning
                ? event?.message
                : decision?.rationale ||
                  (latest
                    ? event?.message
                    : "Start the feed to see actual tool calls, evidence, decisions and follow-ups arrive.")}
            </p>
            {decision && !learning && (
              <>
                <span
                  className={`live-decision-tag ${decision.decision.toLowerCase()}`}
                >
                  {decision.decision}
                </span>
                <div className="live-followup">
                  <RefreshIcon />
                  <span>
                    Next recheck
                    <strong>{formatTime(decision.next_recheck)}</strong>
                  </span>
                </div>
                <div className="local-notification">
                  {decision.notify ? (
                    <>
                      <Radio size={14} /> Operator notification raised in this
                      demo
                    </>
                  ) : decision.decision === "ALERT" ? (
                    <>
                      <ShieldCheck size={14} /> Repeat notification suppressed
                      by cooldown
                    </>
                  ) : (
                    <>
                      <CheckCircle2 size={14} /> Monitoring continues
                    </>
                  )}
                </div>
              </>
            )}
          </section>

          <section className="panel live-tools">
            <div className="panel-heading">
              <div>
                <h3>Tools for this investigation</h3>
                <p>The controller chooses calls from current evidence.</p>
              </div>
            </div>
            {run?.tools?.length ? (
              <div className="live-tool-list">
                {run.tools.slice(-5).map((t) => (
                  <details key={t.id}>
                    <summary>
                      <span className={`tool-state ${t.status}`}>
                        {t.status === "running" ? (
                          <LoaderCircle size={13} className="spin" />
                        ) : (
                          <Check size={13} />
                        )}
                      </span>
                      <div>
                        <strong>{t.name.replaceAll("_", " ")}</strong>
                        <small>
                          {t.status === "running"
                            ? "Executing…"
                            : "Result received"}
                        </small>
                      </div>
                      <ChevronDown size={12} />
                    </summary>
                    {t.result && <p>{t.result.summary}</p>}
                    <pre>{JSON.stringify(t.result || t.args, null, 2)}</pre>
                  </details>
                ))}
              </div>
            ) : (
              <div className="tools-empty">
                <span>01</span>
                <p>
                  No tool has run for this input yet. Normal readings need fewer
                  calls; anomalies trigger a deeper investigation.
                </p>
              </div>
            )}
          </section>

          <section className="panel live-policy">
            <div className="panel-heading">
              <div>
                <h3>Active policy</h3>
                <p>
                  {run?.selection
                    ? "Promoted and frozen before the final test"
                    : run?.scenario === "mission"
                      ? "Starting incumbent: pressure P10"
                      : "Chosen from separate historical validation"}
                </p>
              </div>
              <ShieldCheck size={19} />
            </div>
            {run ? (
              <>
                <div className="active-rule">
                  <span>Pressure below</span>
                  <strong>{value(run.policy.pressure_cutoff, 2)} bar</strong>
                </div>
                <div className="active-rule">
                  <span>Second sensor</span>
                  <strong>
                    {run.policy.confirmation_percentile
                      ? `T < ${value(run.policy.temperature_cutoff)}°C or flow < ${value(run.policy.flow_cutoff)} L/s`
                      : "Not required by incumbent"}
                  </strong>
                </div>
                <div className="active-rule">
                  <span>Persistence</span>
                  <strong>{run.policy.consecutive} hourly sample(s)</strong>
                </div>
                <code>{run.policy_id}</code>
              </>
            ) : (
              <p className="policy-empty">
                The full mission starts with the P10 pressure rule, then earns
                each change through validation.
              </p>
            )}
            {run?.selection && (
              <div className="promotion-live">
                <CheckCircle2 size={18} />
                <div>
                  <strong>
                    {run.selection.promoted
                      ? `Candidate ${run.selection.trial} promoted`
                      : "Incumbent retained"}
                  </strong>
                  <p>
                    Validation cost {run.selection.incumbent_cost} →{" "}
                    {run.selection.selected_cost}
                  </p>
                </div>
              </div>
            )}
            {!!run?.trials.length && (
              <details className="candidate-review">
                <summary>
                  Review {run.trials.length} candidate tests{" "}
                  <ChevronDown size={14} />
                </summary>
                <p>
                  Costs are from validation. KEEP means it beat the best cost at
                  that point in the search.
                </p>
                {run.trials.map((trial) => (
                  <div key={trial.id}>
                    <strong>
                      #{trial.id} · {trial.name}
                    </strong>
                    <span>
                      {trial.accepted ? "KEEP" : "REJECT"} · cost{" "}
                      {trial.metrics.cost}
                    </span>
                    <p>{trial.reason}</p>
                  </div>
                ))}
              </details>
            )}
          </section>
        </aside>
      </div>
      {run?.status === "completed" && (
        <section className="mission-result" role="status">
          <CheckCircle2 size={25} />
          <div>
            <h3>Run complete. Here is the evidence.</h3>
            <p>
              {run.score?.true_positive ?? 0}/{run.score?.bad_hours ?? 0} scored
              bad hours detected · {run.score?.false_positive ?? 0} false-alarm
              hours · detection delay{" "}
              {run.score?.delay_hours == null
                ? "unavailable"
                : `${run.score.delay_hours}h`}
              .{" "}
              {run.injected_excluded
                ? "Interventions were excluded; this is not the unchanged benchmark."
                : run.scenario !== "sandbox"
                  ? "One synthetic test incident; this does not establish field reliability."
                  : "Normal-operation sandbox; no event-detection performance claim."}
            </p>
          </div>
        </section>
      )}
      <div className="footnote">
        <CircleHelp size={14} />
        <span>
          This room executes a stateful deterministic agent with conditional
          tools. No LLM is configured. Sensor data is historical; operator
          notifications are local demo events. Rechecks use sensor time.
        </span>
      </div>
    </section>
  );
}

function ScanIcon() {
  return <ShieldCheck size={27} />;
}
function RefreshIcon() {
  return <RotateCcw size={16} />;
}
