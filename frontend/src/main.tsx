import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  ArrowRight,
  ArrowUpRight,
  BookOpen,
  Check,
  CheckCircle2,
  ChevronRight,
  CircleHelp,
  Database,
  Download,
  ExternalLink,
  FlaskConical,
  Gauge,
  Layers3,
  LoaderCircle,
  Pause,
  Play,
  Radio,
  RefreshCw,
  RotateCcw,
  ScanLine,
  ShieldCheck,
  Snowflake,
  Sparkles,
  Waves,
  X,
} from "lucide-react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Frame, Report, Scenario } from "./types";
import LiveMission from "./LiveMission";
import "@fontsource-variable/dm-sans";
import "@fontsource-variable/manrope";
import "./style.css";

type Page = "live" | "monitor" | "experiments" | "real" | "method";
const nav = [
  { id: "live", label: "Live agent", icon: Activity },
  { id: "monitor", label: "Recorded replays", icon: Gauge },
  { id: "experiments", label: "Improvement lab", icon: FlaskConical },
  { id: "real", label: "Real-world validation", icon: Layers3 },
  { id: "method", label: "Research & method", icon: BookOpen },
] as const;
const pct = (x: number | null | undefined) =>
  x == null ? "—" : `${(x * 100).toFixed(1)}%`;
const num = (x: number | null | undefined, digits = 1) =>
  x == null ? "—" : x.toLocaleString("en", { maximumFractionDigits: digits });
const date = (x: string, compact = false) =>
  new Date(x).toLocaleString("en", {
    month: "short",
    day: "2-digit",
    ...(compact ? {} : { hour: "2-digit", minute: "2-digit", hour12: false }),
  });
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok)
    throw new Error(
      `Request failed (${response.status}). Check the local API and try again.`,
    );
  return response.json();
}

function App() {
  const [page, setPage] = useState<Page>("live");
  const [report, setReport] = useState<Report | null>(null);
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [selected, setSelected] = useState("seed-test");
  const [replay, setReplay] = useState<Scenario | null>(null);
  const [cursor, setCursor] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [error, setError] = useState("");
  const [running, setRunning] = useState(false);
  const [notice, setNotice] = useState("");
  const [period, setPeriod] = useState<"test" | "validation">("test");
  const [showTruth, setShowTruth] = useState(false);

  const load = () => {
    setError("");
    Promise.all([
      request<Report>("/api/research"),
      request<Scenario[]>("/api/scenarios"),
    ])
      .then(([r, s]) => {
        setReport(r);
        setScenarios(s);
      })
      .catch((e) => setError(e.message));
  };
  useEffect(load, []);
  useEffect(() => {
    const controller = new AbortController();
    setPlaying(false);
    setReplay(null);
    setError("");
    request<Scenario>(`/api/scenarios/${selected}`, {
      signal: controller.signal,
    })
      .then((s) => {
        setReplay(s);
        setCursor(
          selected === "seed-test" ? Math.min(124, s.frames.length - 1) : 0,
        );
      })
      .catch((e) => {
        if (e.name !== "AbortError") setError(e.message);
      });
    return () => controller.abort();
  }, [selected]);
  useEffect(() => {
    if (!playing || !replay) return;
    const timer = window.setInterval(
      () =>
        setCursor((c) => {
          if (c >= replay.frames.length - 1) {
            setPlaying(false);
            return c;
          }
          return c + 1;
        }),
      800 / speed,
    );
    return () => window.clearInterval(timer);
  }, [playing, replay, speed]);
  useEffect(() => {
    if (page !== "monitor") setPlaying(false);
    window.scrollTo(0, 0);
  }, [page]);
  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(""), 7000);
    return () => clearTimeout(timer);
  }, [notice]);

  async function rerun() {
    setRunning(true);
    setError("");
    try {
      setReport(
        await request<Report>("/api/experiments/run", { method: "POST" }),
      );
      setNotice(
        "Experiment reproduced. All 24 candidates evaluated on validation data; the frozen final test is unchanged.",
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setRunning(false);
    }
  }
  function download() {
    const link = document.createElement("a");
    link.href = "/api/research/export";
    link.download = "frostline-research-results.json";
    link.click();
  }
  const frame = replay?.frames[cursor];
  const auto = report?.systems.find((s) => s.id === "AUTO");
  const starter = report?.systems.find((s) => s.id === "B1");

  return (
    <div className="shell">
      <aside className="sidebar">
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            setPage("live");
          }}
        >
          <span className="brand-icon">
            <Snowflake size={24} />
          </span>
          <span>
            frostline<span className="brand-dot">.</span>
          </span>
        </a>
        <div className="workspace">
          <span className="workspace-icon">
            <Waves size={19} />
          </span>
          <div>
            Offshore intelligence<small>Case 09 · Energy systems</small>
          </div>
        </div>
        <div className="nav-heading">WORKSPACE</div>
        <nav aria-label="Main navigation">
          {nav.map((item) => (
            <button
              key={item.id}
              title={item.label}
              aria-current={page === item.id ? "page" : undefined}
              className={`nav-item ${page === item.id ? "active" : ""}`}
              onClick={() => setPage(item.id)}
            >
              <item.icon size={18} />
              {item.label}
              {page === item.id && <span className="nav-marker" />}
            </button>
          ))}
        </nav>
        <div className="sidebar-lab">
          <span className="tiny-label">BUILT TO BE QUESTIONED</span>
          <FlaskConical size={23} />
          <h3>
            Every claim,
            <br />
            backed by a run.
          </h3>
          <p>Trace the data, the decision, and the tradeoff.</p>
          <button onClick={() => setPage("method")}>
            Explore the method <ArrowUpRight size={16} />
          </button>
        </div>
        <div className="sidebar-footer">
          <span className="status-dot" />
          <div>
            Local research environment
            <small>Historical replay · advisory only</small>
          </div>
        </div>
      </aside>

      <main>
        <header className="topbar">
          <div>
            <span className="breadcrumb">Workspace</span>
            <ChevronRight size={13} />
            <strong>{nav.find((n) => n.id === page)?.label}</strong>
          </div>
          <span className="prototype">
            <span /> RESEARCH PROTOTYPE
          </span>
        </header>
        <div className="content">
          {error && (
            <div className="error" role="alert">
              <CircleHelp size={18} />
              <span>{error}</span>
              <button
                onClick={() => {
                  load();
                  request<Scenario>(`/api/scenarios/${selected}`)
                    .then((s) => {
                      setReplay(s);
                      setCursor(0);
                    })
                    .catch((e) => setError(e.message));
                }}
              >
                Retry
              </button>
            </div>
          )}
          {notice && (
            <div className="notice" role="status">
              <CheckCircle2 size={18} />
              {notice}
              <button
                aria-label="Dismiss notification"
                onClick={() => setNotice("")}
              >
                <X size={15} />
              </button>
            </div>
          )}
          {!report ? (
            <div className="loading">
              <LoaderCircle className="spin" size={26} />
              <h2>Preparing the evidence</h2>
              <p>Loading reproducible experiments and replay scenarios.</p>
            </div>
          ) : (
            <>
              <div hidden={page !== "live"}>
                <LiveMission />
              </div>
              {page !== "live" && (
                <div className="page-heading">
                  <div>
                    <div className="eyebrow">
                      {page === "monitor"
                        ? "FROM WELL DATA TO A DEFENSIBLE DECISION"
                        : page === "experiments"
                          ? "ONE CHANGE. ONE MEASURED TRADEOFF."
                          : page === "real"
                            ? "BEYOND THE SYNTHETIC SEED"
                            : "THE REASONING BEHIND THE SYSTEM"}
                    </div>
                    <h1>
                      {page === "monitor"
                        ? "See the signal. Understand the call."
                        : page === "experiments"
                          ? "Make every improvement visible."
                          : page === "real"
                            ? "Different wells. A harder test."
                            : "A clear path from evidence to action."}
                    </h1>
                    <p>
                      {page === "monitor"
                        ? "Replay a well, inspect the evidence, and follow the agent’s next move."
                        : page === "experiments"
                          ? "Compare the starter, isolated changes, and the policy chosen by validation evidence."
                          : page === "real"
                            ? "Real Petrobras recordings, with entire wells held out during model training."
                            : "What we built, why it works this way, and exactly what we can claim."}
                    </p>
                  </div>
                  <button className="button secondary" onClick={download}>
                    <Download size={16} /> Export evidence
                  </button>
                </div>
              )}

              {page === "monitor" && (
                <>
                  <div className="hero-strip">
                    <div className="hero-mark">
                      <ScanLine size={30} />
                    </div>
                    <div>
                      <div className="tiny-label">
                        FROZEN FINAL TEST · OFFICIAL SYNTHETIC SEED
                      </div>
                      <h2>
                        {auto?.test.true_positive} of {auto?.test.bad_hours} bad
                        hours caught. Every decision explained.
                      </h2>
                      <p>
                        One test incident · {auto?.test.evaluated_hours} scored
                        hours · {auto?.test.excluded_hours} missing-pressure
                        hours excluded
                      </p>
                    </div>
                    <button onClick={() => setPage("experiments")}>
                      See how we got here <ArrowRight size={17} />
                    </button>
                  </div>
                  <div className="metric-grid">
                    <Metric
                      label="Bad-hour recall"
                      value={pct(auto?.test.recall)}
                      icon={<ScanLine size={18} />}
                      detail={`Starter: ${pct(starter?.test.recall)}`}
                      badge="Final test"
                    />
                    <Metric
                      label="False-alarm hours"
                      value={String(auto?.test.false_positive)}
                      icon={<ShieldCheck size={18} />}
                      detail={`Across ${(auto?.test.evaluated_hours ?? 0) - (auto?.test.bad_hours ?? 0)} scored normal hours`}
                      badge="Measured"
                    />
                    <Metric
                      label="Detection delay"
                      value={`${num(auto?.test.delay_hours)}h`}
                      icon={<Activity size={18} />}
                      detail={`After label onset · starter ${num(starter?.test.delay_hours)}h`}
                      badge={`${num((starter?.test.delay_hours ?? 0) - (auto?.test.delay_hours ?? 0))}h sooner`}
                    />
                    <Metric
                      label="Policies evaluated"
                      value="24"
                      icon={<FlaskConical size={18} />}
                      detail="Selection uses days 11–20 only"
                      badge="Autonomous"
                    />
                  </div>
                  <div className="replay-toolbar">
                    <div className="section-title">
                      <span className="status-dot" />
                      <h2>Well replay</h2>
                      <span className="muted small">Recorded telemetry</span>
                    </div>
                    <label className="select-wrap">
                      <Database size={15} />
                      <select
                        aria-label="Replay scenario"
                        value={selected}
                        onChange={(e) => setSelected(e.target.value)}
                      >
                        {scenarios.map((s) => (
                          <option key={s.id} value={s.id}>
                            {s.title}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>
                  {!replay || !frame ? (
                    <div className="panel loading">
                      <LoaderCircle className="spin" /> Loading scenario…
                    </div>
                  ) : (
                    <>
                      <div className="scenario-meta">
                        <span
                          className={`tag ${selected.startsWith("real") ? "teal" : ""}`}
                        >
                          {replay.provenance}
                        </span>
                        <span>{replay.description}</span>
                      </div>
                      <div className="monitor-grid">
                        <div className="charts-column">
                          <section className="panel telemetry">
                            <div className="panel-heading">
                              <div>
                                <h3>Pressure & operating conditions</h3>
                                <p>
                                  Only readings available at the replay cursor
                                  are shown.
                                </p>
                              </div>
                              <span className="tag">{date(frame.t)}</span>
                            </div>
                            <SensorChart
                              replay={replay}
                              cursor={cursor}
                              sensor="pressure_bar"
                              color="#138578"
                              truth={showTruth}
                            />
                            <div className="subcharts">
                              <SensorChart
                                replay={replay}
                                cursor={cursor}
                                sensor="temp_C"
                                color="#df9362"
                                truth={false}
                                small
                              />
                              <SensorChart
                                replay={replay}
                                cursor={cursor}
                                sensor={
                                  selected.startsWith("real")
                                    ? "downstream_bar"
                                    : "flow_Lps"
                                }
                                color="#6688a5"
                                truth={false}
                                small
                              />
                            </div>
                            <div className="playback">
                              <button
                                className="icon-button"
                                aria-label="Restart replay"
                                title="Restart replay"
                                onClick={() => {
                                  setCursor(0);
                                  setPlaying(false);
                                }}
                              >
                                <RotateCcw size={17} />
                              </button>
                              <button
                                className="play-button"
                                aria-label={
                                  playing ? "Pause replay" : "Play replay"
                                }
                                onClick={() => {
                                  if (cursor === replay.frames.length - 1)
                                    setCursor(0);
                                  setPlaying(!playing);
                                }}
                              >
                                {playing ? (
                                  <Pause size={15} />
                                ) : (
                                  <Play size={15} />
                                )}
                              </button>
                              <input
                                type="range"
                                aria-label="Replay position"
                                min={0}
                                max={replay.frames.length - 1}
                                value={cursor}
                                onChange={(e) => {
                                  setCursor(Number(e.target.value));
                                  setPlaying(false);
                                }}
                              />
                              <span className="time-label">
                                {cursor + 1} / {replay.frames.length}
                              </span>
                              <select
                                aria-label="Replay speed"
                                value={speed}
                                onChange={(e) =>
                                  setSpeed(Number(e.target.value))
                                }
                              >
                                {[1, 2, 4, 8].map((s) => (
                                  <option key={s} value={s}>
                                    {s}×
                                  </option>
                                ))}
                              </select>
                            </div>
                            <div className="chart-bottom">
                              <label>
                                <input
                                  type="checkbox"
                                  checked={showTruth}
                                  onChange={(e) =>
                                    setShowTruth(e.target.checked)
                                  }
                                />{" "}
                                Show evaluator labels
                              </label>
                              <button
                                onClick={() => {
                                  const i = replay.frames.findIndex(
                                    (f) =>
                                      f.ground_truth !== "Normal" &&
                                      f.ground_truth !== "Unlabelled",
                                  );
                                  if (i >= 0) {
                                    setCursor(i);
                                    setPlaying(false);
                                    setShowTruth(true);
                                  } else
                                    setNotice(
                                      "This scenario contains no labelled event. It tests how the workflow handles normal or faulty telemetry.",
                                    );
                                }}
                              >
                                Jump to labelled event <ArrowRight size={13} />
                              </button>
                            </div>
                          </section>
                          <section className="panel loop-panel">
                            <div className="panel-heading">
                              <div>
                                <h3>A bounded autonomous loop</h3>
                                <p>
                                  Deterministic tools · labels excluded from
                                  decisions · no LLM required
                                </p>
                              </div>
                              <span className="tag teal">4 evidence steps</span>
                            </div>
                            <div className="loop-flow">
                              {[
                                {
                                  label: "Observe",
                                  desc: "Check sensor quality",
                                  icon: Radio,
                                },
                                {
                                  label: "Investigate",
                                  desc: "Compare the evidence",
                                  icon: ScanLine,
                                },
                                {
                                  label: "Decide",
                                  desc: frame.decision.toLowerCase(),
                                  icon: ShieldCheck,
                                },
                                {
                                  label: "Recheck",
                                  desc: `In ${frame.recheck_minutes} min`,
                                  icon: RefreshCw,
                                },
                              ].map((s, i) => (
                                <React.Fragment key={s.label}>
                                  <div className="loop-step">
                                    <span>
                                      <s.icon size={18} />
                                    </span>
                                    <strong>{s.label}</strong>
                                    <small>{s.desc}</small>
                                  </div>
                                  {i < 3 && (
                                    <ArrowRight
                                      className="loop-arrow"
                                      size={15}
                                    />
                                  )}
                                </React.Fragment>
                              ))}
                            </div>
                          </section>
                        </div>
                        <aside className="evidence-column">
                          <DecisionPanel
                            frame={frame}
                            policyId={replay.policy_id}
                          />
                          <section className="panel trace-panel">
                            <div className="panel-heading">
                              <div>
                                <h3>Decision trace</h3>
                                <p>Evidence at this timestamp</p>
                              </div>
                              <span className="count-badge">04</span>
                            </div>
                            <ol className="trace">
                              {frame.evidence.map((e, i) => (
                                <li key={e.tool}>
                                  <span className="trace-number">{i + 1}</span>
                                  <div>
                                    <strong>
                                      {e.tool.replaceAll("_", " ")}
                                    </strong>
                                    <p>{e.summary}</p>
                                    {e.result != null && (
                                      <details>
                                        <summary>Inspect tool result</summary>
                                        <pre>
                                          {JSON.stringify(e.result, null, 2)}
                                        </pre>
                                      </details>
                                    )}
                                  </div>
                                </li>
                              ))}
                            </ol>
                          </section>
                        </aside>
                      </div>
                      <div className="footnote">
                        <CircleHelp size={14} />
                        <span>
                          {showTruth
                            ? `Evaluator label: ${frame.ground_truth}. Labels are never inputs to the live decision function.`
                            : "Historical replay. Alerts are investigation prompts; this prototype does not recommend treatment or predict blockage time."}
                        </span>
                      </div>
                    </>
                  )}
                </>
              )}

              {page === "experiments" && (
                <>
                  <SplitTimeline report={report} />
                  <section className="panel experiment-panel">
                    <div className="panel-heading">
                      <div>
                        <h3>The improvement ladder</h3>
                        <p>
                          Bad-hour detection, without giving credit for hours
                          the system missed.
                        </p>
                      </div>
                      <div className="segmented">
                        <button
                          className={period === "test" ? "chosen" : ""}
                          onClick={() => setPeriod("test")}
                        >
                          Final test · 10 days
                        </button>
                        <button
                          className={period === "validation" ? "chosen" : ""}
                          onClick={() => setPeriod("validation")}
                        >
                          Validation · 10 days
                        </button>
                      </div>
                    </div>
                    <div className="table-scroll">
                      <table>
                        <thead>
                          <tr>
                            <th>Round / change</th>
                            <th>Bad hours caught</th>
                            <th>Recall</th>
                            <th>False hours</th>
                            <th>Delay after onset</th>
                            <th>Cost ↓</th>
                          </tr>
                        </thead>
                        <tbody>
                          {report.systems.map((s) => (
                            <tr
                              key={s.id}
                              className={s.id === "AUTO" ? "selected-row" : ""}
                            >
                              <td>
                                <div className="system-cell">
                                  <span className="system-code">{s.id}</span>
                                  <div>
                                    <strong>{s.name}</strong>
                                    <small>{s.description}</small>
                                  </div>
                                </div>
                              </td>
                              <td>
                                <b>{s[period].true_positive}</b> /{" "}
                                {s[period].bad_hours}
                              </td>
                              <td>
                                <div className="recall-cell">
                                  <div>
                                    <span
                                      style={{
                                        width: `${(s[period].recall || 0) * 100}%`,
                                      }}
                                    />
                                  </div>
                                  {pct(s[period].recall)}
                                </div>
                              </td>
                              <td>{s[period].false_positive}</td>
                              <td>
                                {s[period].delay_hours == null
                                  ? "No detection"
                                  : `${s[period].delay_hours}h`}
                              </td>
                              <td>{s[period].cost}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <div className="table-note">
                      <CircleHelp size={14} />
                      {period === "test"
                        ? "Final test contains ONE 12-hour incident. Every detecting system catches that same incident; the improvement is coverage and speed."
                        : "Validation contains ONE 30-hour incident. Candidate thresholds are fitted only on days 1–10."}
                    </div>
                  </section>
                  <div className="two-columns">
                    <section className="panel selector-panel">
                      <div className="panel-heading">
                        <div>
                          <div className="eyebrow">AUTONOMOUS IMPROVEMENT</div>
                          <h3>Search. Score. Promote. Freeze.</h3>
                        </div>
                        <Sparkles size={23} />
                      </div>
                      <p>
                        The system proposes 24 sensor rules, compares their
                        validation cost, and promotes a winner only if it beats
                        the incumbent.
                      </p>
                      <div className="formula">
                        5 <span>× missed bad hours</span> +{" "}
                        <span>false-alarm hours</span>
                      </div>
                      <div className="promotion">
                        <div>
                          <small>Incumbent validation cost</small>
                          <strong>{report.selection.incumbent_cost}</strong>
                        </div>
                        <ArrowRight size={22} />
                        <div>
                          <small>Selected validation cost</small>
                          <strong>{report.selection.validation_cost}</strong>
                        </div>
                        <span className="tag teal">
                          <Check size={13} />
                          {report.selection.promoted ? "Promoted" : "Retained"}
                        </span>
                      </div>
                      <p className="muted small">
                        {report.selection.weight_note} Final test scores never
                        choose a candidate.
                      </p>
                      <button
                        className="button primary"
                        disabled={running}
                        onClick={rerun}
                      >
                        {running ? (
                          <LoaderCircle size={16} className="spin" />
                        ) : (
                          <RefreshCw size={16} />
                        )}{" "}
                        {running
                          ? "Reproducing experiment…"
                          : "Reproduce the experiment"}
                      </button>
                    </section>
                    <section className="panel policy-panel">
                      <div className="panel-heading">
                        <div>
                          <div className="eyebrow">FROZEN FOR REPLAY</div>
                          <h3>The selected operating rule</h3>
                        </div>
                        <ShieldCheck size={23} />
                      </div>
                      <div className="rule-line">
                        <span>01</span>
                        <p>
                          Pressure below{" "}
                          <strong>
                            {num(auto?.policy?.pressure_cutoff, 2)} bar
                          </strong>
                          <small>
                            P{auto?.policy?.pressure_percentile} refitted on
                            days 1–20
                          </small>
                        </p>
                      </div>
                      <div className="rule-line">
                        <span>02</span>
                        <p>
                          Temperature below{" "}
                          <strong>
                            {num(auto?.policy?.temperature_cutoff, 2)} °C
                          </strong>
                          <br />
                          or flow below{" "}
                          <strong>
                            {num(auto?.policy?.flow_cutoff, 2)} L/s
                          </strong>
                          <small>
                            Confirmation from at least one second sensor
                          </small>
                        </p>
                      </div>
                      <div className="rule-line">
                        <span>03</span>
                        <p>
                          Evidence persists for{" "}
                          <strong>
                            {auto?.policy?.consecutive} hourly sample(s)
                          </strong>
                          <small>
                            Quality checks first; follow-up in 60 minutes
                          </small>
                        </p>
                      </div>
                      <div className="policy-fingerprint">
                        <span>POLICY FINGERPRINT</span>
                        <code>{report.policy_id}</code>
                      </div>
                    </section>
                  </div>
                  <section className="panel">
                    <div className="panel-heading">
                      <div>
                        <h3>All 24 candidates</h3>
                        <p>
                          Fitted on calibration days; ranked on validation days.
                          Full search history, including rejections.
                        </p>
                      </div>
                      <span className="tag">
                        Trial {report.selection.selected_trial} selected
                      </span>
                    </div>
                    <div className="trial-grid">
                      {report.selection.trials.map((t) => (
                        <div
                          key={t.id}
                          className={`trial ${t.selected ? "winner" : ""}`}
                        >
                          <span>
                            #{String(t.id).padStart(2, "0")}{" "}
                            {t.selected && <CheckCircle2 size={14} />}
                          </span>
                          <strong>{t.name}</strong>
                          <div>
                            <span>
                              Cost <b>{t.metrics.cost}</b>
                            </span>
                            <span>
                              {t.metrics.true_positive}/{t.metrics.bad_hours}{" "}
                              caught · {t.metrics.false_positive} false
                            </span>
                          </div>
                          <small>
                            {t.selected ? "Selected & frozen" : t.action}
                          </small>
                        </div>
                      ))}
                    </div>
                  </section>
                </>
              )}

              {page === "real" && (
                <RealValidation
                  report={report}
                  openReplay={(id) => {
                    setSelected(id);
                    setPage("monitor");
                  }}
                />
              )}
              {page === "method" && <Method report={report} />}
              <footer className="page-footer">
                <span>
                  <Snowflake size={14} /> FROSTLINE{" "}
                  <span className="muted">/</span> Case 09
                </span>
                <span>Evidence before confidence.</span>
              </footer>
            </>
          )}
        </div>
      </main>
    </div>
  );
}

function Metric({
  label,
  value,
  detail,
  icon,
  badge,
}: {
  label: string;
  value: string;
  detail: string;
  icon: React.ReactNode;
  badge: string;
}) {
  return (
    <section className="metric">
      <div>
        <span>{label}</span>
        {icon}
      </div>
      <h2>
        {value}
        <span className="metric-badge">{badge}</span>
      </h2>
      <p>{detail}</p>
    </section>
  );
}

function SensorChart({
  replay,
  cursor,
  sensor,
  color,
  truth,
  small = false,
}: {
  replay: Scenario;
  cursor: number;
  sensor: string;
  color: string;
  truth: boolean;
  small?: boolean;
}) {
  const frames = replay.frames.slice(0, cursor + 1);
  const current = frames.at(-1)!;
  const data = frames.map((f) => ({
    time: new Date(f.t).getTime(),
    value: f.sensors[sensor],
    label: f.ground_truth,
  }));
  const firstEvent = frames.find(
    (f) => f.ground_truth !== "Normal" && f.ground_truth !== "Unlabelled",
  );
  return (
    <div className={`sensor-chart ${small ? "small-chart" : ""}`}>
      <div className="sensor-label">
        <span>
          <i style={{ background: color }} />
          {replay.sensor_labels[sensor]}
        </span>
        <b>{num(current.sensors[sensor], 2)}</b>
      </div>
      <div className="chart-canvas" style={{ height: small ? 117 : 202 }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={data}
            margin={{ top: 12, right: 12, left: -21, bottom: 0 }}
          >
            <defs>
              <linearGradient id={`fill-${sensor}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={color} stopOpacity={0.17} />
                <stop offset="100%" stopColor={color} stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid
              vertical={false}
              stroke="#e9edeb"
              strokeDasharray="3 4"
            />
            <XAxis
              dataKey="time"
              type="number"
              domain={["dataMin", "dataMax"]}
              tickFormatter={(x) => date(new Date(x).toISOString(), true)}
              tick={{ fill: "#89948f", fontSize: 10 }}
              minTickGap={small ? 60 : 70}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              domain={["auto", "auto"]}
              tick={{ fill: "#89948f", fontSize: 10 }}
              tickFormatter={(v) => num(v, 0)}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              labelFormatter={(v) => date(new Date(Number(v)).toISOString())}
              formatter={(v) => [
                num(Number(v), 2),
                replay.sensor_labels[sensor],
              ]}
              contentStyle={{
                border: "1px solid #dce4df",
                borderRadius: 9,
                fontSize: 12,
              }}
            />
            <Area
              type="linear"
              dataKey="value"
              fill={`url(#fill-${sensor})`}
              stroke="none"
              isAnimationActive={false}
              connectNulls={false}
            />
            <Line
              type="linear"
              dataKey="value"
              stroke={color}
              strokeWidth={small ? 1.6 : 2}
              dot={false}
              isAnimationActive={false}
              connectNulls={false}
            />
            {sensor === "pressure_bar" && replay.policy && (
              <ReferenceLine
                y={replay.policy.pressure_cutoff}
                stroke="#dc9570"
                strokeDasharray="5 4"
                label={{
                  value: "Selected threshold",
                  position: "insideTopRight",
                  fill: "#b57855",
                  fontSize: 10,
                }}
              />
            )}
            {truth && firstEvent && (
              <ReferenceLine
                x={new Date(firstEvent.t).getTime()}
                stroke="#cb7863"
                label={{
                  value: "Label onset",
                  fill: "#b46d57",
                  fontSize: 10,
                  position: "insideBottomRight",
                }}
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function DecisionPanel({
  frame,
  policyId,
}: {
  frame: Frame;
  policyId: string;
}) {
  return (
    <section
      className={`decision-card decision-${frame.decision.toLowerCase()}`}
    >
      <div className="decision-top">
        <span className="tiny-label">AGENT DECISION</span>
        <span className="decision-badge">
          <span />
          {frame.decision}
        </span>
      </div>
      <h2>{frame.diagnosis}</h2>
      <p>{frame.brief}</p>
      {frame.scores && (
        <div className="scores">
          {Object.entries(frame.scores).map(([k, v]) => (
            <div key={k}>
              <span>{k}</span>
              <div>
                <i style={{ width: `${v * 100}%` }} />
              </div>
              <strong>{v.toFixed(3)}</strong>
            </div>
          ))}
          <small>Uncalibrated model scores</small>
        </div>
      )}
      <div className="decision-status">
        <span>
          <ShieldCheck size={14} />
          Quality: {frame.quality}
        </span>
        <span>
          <RefreshCw size={13} />
          {frame.recheck_minutes} min recheck
        </span>
      </div>
      <div className="notification-state">
        {frame.notify ? (
          <>
            <Radio size={14} /> Operator notification raised in replay
          </>
        ) : (
          <>
            <CheckCircle2 size={14} /> No new operator notification
          </>
        )}
      </div>
      <code>policy / {policyId}</code>
    </section>
  );
}

function SplitTimeline({ report }: { report: Report }) {
  return (
    <section className="panel split-panel">
      <div className="panel-heading">
        <div>
          <h3>20 days to develop. 10 days to prove it.</h3>
          <p>
            Chronological split. Candidate selection cannot see the final test.
          </p>
        </div>
        <span className="tag">720 hourly readings</span>
      </div>
      <div className="split-timeline">
        {report.split.map((s, i) => (
          <div key={s.name} className={`split-segment split-${i}`}>
            <div>
              <span>
                0{i + 1} / {s.name.toUpperCase()}
              </span>
              {i === 2 ? <ShieldCheck size={18} /> : <Database size={18} />}
            </div>
            <h3>
              {date(s.start, true)} — {date(s.end, true)}
            </h3>
            <p>
              {i === 0
                ? "Fit thresholds"
                : i === 1
                  ? "Choose among 24 policies"
                  : "Evaluate the frozen winner"}
            </p>
            <small>
              {s.rows} hours · {s.usable} with pressure
            </small>
          </div>
        ))}
      </div>
    </section>
  );
}

function RealValidation({
  report,
  openReplay,
}: {
  report: Report;
  openReplay: (id: string) => void;
}) {
  const r = report.real;
  if (r.status !== "complete")
    return (
      <div className="panel empty-state">
        <Database size={28} />
        <h2>Real-data pilot has not been run</h2>
        <p>{r.message}</p>
      </div>
    );
  return (
    <>
      <div className="research-callout">
        <Layers3 size={25} />
        <div>
          <h3>A separate model for a different physical pattern</h3>
          <p>
            In real production-line hydrate cases, upstream pressure can rise
            while downstream pressure falls. The real-data pilot learns
            multivariate patterns from 3W; the seed’s falling-pressure threshold
            is not transferred blindly.
          </p>
        </div>
      </div>
      <div className="metric-grid">
        <Metric
          label="Recordings / wells"
          value={`${r.recordings.length} / ${r.wells}`}
          icon={<Database size={18} />}
          detail={`${num(r.minutes, 0)} labelled minutes`}
          badge="Real 3W"
        />
        <Metric
          label="Hydrate minute recall"
          value={pct(r.hydrate_recall)}
          icon={<ScanLine size={18} />}
          detail="Out-of-fold classification · argmax"
          badge="Held-out wells"
        />
        <Metric
          label="Hydrate records detected"
          value={`${r.events_detected}/${r.hydrate_recordings}`}
          icon={<Activity size={18} />}
          detail={`${r.early_events} before established phase`}
          badge="3-minute alarm"
        />
        <Metric
          label="Look-alikes flagged"
          value={`${r.lookalikes_flagged}/${r.lookalike_recordings}`}
          icon={<CircleHelp size={18} />}
          detail="Any hydrate alarm during look-alike phase"
          badge="Failure metric"
        />
      </div>
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h3>Replay the real recordings</h3>
            <p>
              The first filename in each class was chosen before seeing model
              performance.
            </p>
          </div>
          <span className="tag teal">CC BY 4.0 · Petrobras</span>
        </div>
        <div className="real-scenarios">
          {[
            {
              key: "normal",
              label: "Normal operation",
              copy: "Does the model stay quiet?",
            },
            {
              key: "hydrate",
              label: "Production-line hydrate",
              copy: "Inspect the forming and established phases.",
            },
            {
              key: "scaling",
              label: "Scaling in the choke",
              copy: "Can it distinguish a look-alike?",
            },
            {
              key: "restriction",
              label: "Quick restriction",
              copy: "Challenge the alternative diagnosis.",
            },
          ].map((s) => (
            <button key={s.key} onClick={() => openReplay(`real-${s.key}`)}>
              <span>
                <Waves size={21} />
                <ArrowUpRight size={17} />
              </span>
              <strong>{s.label}</strong>
              <small>{s.copy}</small>
            </button>
          ))}
        </div>
      </section>
      <div className="two-columns">
        <section className="panel">
          <div className="panel-heading">
            <div>
              <h3>Classification confusion matrix</h3>
              <p>Rows: expert labels · columns: predicted class · minutes</p>
            </div>
          </div>
          <div className="confusion-grid">
            <span />
            {r.class_names.map((c) => (
              <strong key={c}>{c}</strong>
            ))}
            {r.confusion_matrix.map((row, i) => (
              <React.Fragment key={i}>
                <strong>{r.class_names[i]}</strong>
                {row.map((n, j) => (
                  <div key={j} className={i === j ? "diagonal" : ""}>
                    {num(n, 0)}
                  </div>
                ))}
              </React.Fragment>
            ))}
          </div>
          <div className="matrix-summary">
            <span>
              Macro F1 <b>{num(r.macro_f1, 3)}</b>
            </span>
            <span>
              False-alarm episodes / normal day{" "}
              <b>{num(r.false_alarms_per_normal_day, 2)}</b>
            </span>
          </div>
          <p className="panel-footnote">
            Classification uses argmax. Operational alarms require hydrate score
            ≥0.5 for 3 minutes; the two metrics measure different decisions.
          </p>
        </section>
        <section className="panel">
          <div className="panel-heading">
            <div>
              <h3>The train / test boundary</h3>
              <p>Every prediction comes from a model trained on other wells.</p>
            </div>
          </div>
          <div className="folds">
            {r.folds.map((f) => (
              <details key={f.fold}>
                <summary>
                  <span className="fold-number">0{f.fold}</span>
                  <div>
                    <strong>
                      {f.train_wells.length} training wells →{" "}
                      {f.test_wells.length} test wells
                    </strong>
                    <small>
                      {num(f.test_minutes, 0)} labelled test minutes
                    </small>
                  </div>
                  <span className="tag teal">{f.overlap} overlap</span>
                </summary>
                <p>
                  <b>Held out:</b> {f.test_wells.join(", ")}
                </p>
                <p>
                  <b>Training:</b> {f.train_wells.join(", ")}
                </p>
              </details>
            ))}
          </div>
          <p className="panel-footnote">{r.protocol}</p>
        </section>
      </div>
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h3>Every recording, including the misses</h3>
            <p>
              Lead time is positive before the established phase, negative after
              it. No alarm = no lead-time value.
            </p>
          </div>
        </div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Recording</th>
                <th>Class</th>
                <th>Fold</th>
                <th>Hydrate outcome</th>
                <th>Lead time</th>
                <th>False episodes</th>
              </tr>
            </thead>
            <tbody>
              {r.recordings.map((s) => (
                <tr key={s.file}>
                  <td className="mono small">
                    {s.file.replace(".parquet", "")}
                  </td>
                  <td>
                    {
                      (
                        {
                          0: "Normal",
                          6: "Restriction",
                          7: "Scaling",
                          8: "Hydrate",
                        } as Record<number, string>
                      )[s.class]
                    }
                  </td>
                  <td>{s.fold}</td>
                  <td>
                    {s.class === 8
                      ? s.detected
                        ? s.early
                          ? "Early detection"
                          : "Detected"
                        : "Missed"
                      : s.class !== 0
                        ? s.misdiagnosed
                          ? "Misdiagnosed"
                          : "No hydrate alarm"
                        : "—"}
                  </td>
                  <td>
                    {s.lead_minutes == null
                      ? "—"
                      : `${num(s.lead_minutes, 0)} min`}
                  </td>
                  <td>{s.false_alarm_episodes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      <section className="panel limitations">
        <h3>What this pilot does and does not establish</h3>
        <ul>
          {r.limits.map((s) => (
            <li key={s}>{s}</li>
          ))}
        </ul>
      </section>
    </>
  );
}

function Method({ report }: { report: Report }) {
  return (
    <>
      <div className="method-intro">
        <div className="method-number">
          09<span>CASE</span>
        </div>
        <div>
          <h2>An autonomous well-event flagging agent.</h2>
          <p>
            Frostline turns sensor readings into a traceable investigation, then
            uses historical outcomes to select a better rule. The operator can
            inspect every observation, threshold, and follow-up.
          </p>
        </div>
      </div>
      <SplitTimeline report={report} />
      <section className="panel architecture">
        <div className="panel-heading">
          <div>
            <h3>Two loops, one evidence trail</h3>
            <p>
              The operating loop sees sensor history. The improvement loop sees
              confirmed historical outcomes.
            </p>
          </div>
        </div>
        <div className="architecture-row">
          {[
            "Sensor readings",
            "Causal features",
            "Frozen rule / model",
            "Bounded investigation",
            "Operator brief + recheck",
          ].map((s, i) => (
            <React.Fragment key={s}>
              <span>{s}</span>
              {i < 4 && <ArrowRight size={17} />}
            </React.Fragment>
          ))}
        </div>
        <div className="architecture-row lower">
          {[
            "Historical labels",
            "Validation scoring",
            "24 policy candidates",
            "Promote only if better",
            "Freeze + final test",
          ].map((s, i) => (
            <React.Fragment key={s}>
              <span>{s}</span>
              {i < 4 && <ArrowRight size={17} />}
            </React.Fragment>
          ))}
        </div>
      </section>
      <div className="two-columns">
        <section className="panel method-card">
          <div className="eyebrow">WHAT MAKES IT AUTONOMOUS</div>
          <h3>The system makes bounded choices.</h3>
          <ul>
            <li>
              Proposes pressure and corroboration thresholds across 24 allowed
              combinations.
            </li>
            <li>
              Scores each proposal on separate validation history and decides
              whether to promote it.
            </li>
            <li>
              Checks incoming quality, gathers sensor evidence, and chooses
              ALERT, WATCH, or DISMISS.
            </li>
            <li>
              Schedules a recheck and suppresses repeat notifications during
              cooldown.
            </li>
          </ul>
          <p className="muted small">
            This is a deterministic autonomous workflow. No LLM calls or
            unsupervised online learning are claimed.
          </p>
        </section>
        <section className="panel method-card">
          <div className="eyebrow">THE JUDGES SHOULD SEE</div>
          <h3>A five-minute evidence story.</h3>
          <ol>
            <li>
              Start a live mission: watch new inputs trigger actual tool calls,
              decisions and scheduled rechecks.
            </li>
            <li>
              Explain the 10 / 10 / 10 split as the agent tests 24 candidates,
              rejects weaker rules and promotes an improvement.
            </li>
            <li>
              Watch the frozen policy process the final test. Compare it with
              the official starter in the improvement lab.
            </li>
            <li>
              Start the sandbox and inject a pressure dip or sensor outage.
            </li>
            <li>
              Show the real-well pilot, including misses and look-alike errors.
            </li>
          </ol>
        </section>
      </div>
      <section className="panel limitations">
        <h3>Honest boundaries</h3>
        <ul>
          {report.limits.map((l) => (
            <li key={l}>{l}</li>
          ))}
        </ul>
      </section>
      <section className="panel sources-panel">
        <div className="panel-heading">
          <div>
            <h3>Research that shaped the implementation</h3>
            <p>Primary sources and reproducible data provenance.</p>
          </div>
          <BookOpen size={21} />
        </div>
        <div className="source-links">
          {report.sources.map((s, i) => (
            <a key={s.url} href={s.url} target="_blank" rel="noreferrer">
              <span>{String(i + 1).padStart(2, "0")}</span>
              <strong>{s.title}</strong>
              <ExternalLink size={15} />
            </a>
          ))}
        </div>
        <div className="provenance">
          <p>
            <b>Seed data fingerprint</b>
            <code>{report.dataset_sha256}</code>
          </p>
          {report.real.status === "complete" && (
            <>
              <p>
                <b>3W pinned revision</b>
                <code>{report.real.revision}</code>
              </p>
              <p>
                <b>Prospective file selection</b>
                {report.real.selection}
              </p>
            </>
          )}
        </div>
      </section>
    </>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
