import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  ChevronRight,
  FlaskConical,
  MapPinned,
  Play,
  Snowflake,
  Waves,
} from "lucide-react";
import FleetDashboard from "./FleetDashboard";
import Scores from "./Scores";
import "@fontsource-variable/dm-sans";
import "@fontsource-variable/manrope";
import "./style.css";

const nav = [
  { id: "monitor", label: "Monitor", icon: MapPinned },
  { id: "scores", label: "Scores", icon: FlaskConical },
] as const;
type Page = (typeof nav)[number]["id"];
const stats = [
  ["5 → 10", "events caught (of 12)"],
  ["0", "false alarms added"],
  ["100%", "3W hydrate recall"],
];

function App() {
  const [page, setPage] = useState<Page>("monitor");
  const [showLanding, setShowLanding] = useState(true);
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [page]);

  return (
    <div className="shell">
      <aside className="sidebar">
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            setPage("monitor");
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
            Hydrate detection<small>IEEE Hackathon · Case 09</small>
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
        <div className="sidebar-footer">
          <span className="status-dot" />
          <div>
            Petrobras 3W data
            <small>4 wells · real sensors</small>
          </div>
        </div>
      </aside>

      <main>
        <header className="topbar">
          <div>
            <span className="breadcrumb">Frostline</span>
            <ChevronRight size={13} />
            <strong>{nav.find((n) => n.id === page)?.label}</strong>
          </div>
          <span className="prototype">
            <span /> DEMO
          </span>
        </header>
        {showLanding && (
          <div className="landing-overlay">
            <div className="landing-content">
              <h1 className="landing-title">
                <Snowflake size={40} className="landing-icon" />
                frostline
              </h1>
              <p className="landing-subtitle">
                Hydrate early warning for offshore wells
              </p>
              <div className="landing-results">
                {stats.map(([value, label]) => (
                  <div className="landing-stat" key={label}>
                    <span className="landing-stat-value">{value}</span>
                    <span className="landing-stat-label">{label}</span>
                  </div>
                ))}
              </div>
              <button
                className="landing-cta"
                onClick={() => setShowLanding(false)}
              >
                <Play size={20} />
                Start
              </button>
            </div>
          </div>
        )}
        <div className="content">
          {/* Monitor stays mounted so its live stream survives a page switch. */}
          <div hidden={page !== "monitor"}>
            <FleetDashboard />
          </div>
          {page === "scores" && <Scores />}
        </div>
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
