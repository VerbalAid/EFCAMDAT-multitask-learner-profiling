import { useState, useRef, useEffect } from "react";

const CEFR_WARM = [180, 110, 50];
const CEFR_COOL = [50, 100, 160];

function warmRowStyle(intensity) {
  const scaled = 0.04 + intensity * 0.22;
  return {
    backgroundColor: `rgba(${CEFR_WARM[0]}, ${CEFR_WARM[1]}, ${CEFR_WARM[2]}, ${scaled})`,
    borderColor: `rgba(${CEFR_WARM[0]}, ${CEFR_WARM[1]}, ${CEFR_WARM[2]}, ${0.12 + intensity * 0.22})`,
  };
}

function HelpMiniHeatmap() {
  const rows = [
    { rank: 1, pct: 100, text: "Example sentence here", intensity: 1, faint: false },
    { rank: 2, pct: 67, text: "Another sentence", intensity: 0.67, faint: false },
    { rank: null, pct: 12, text: "Low signal sentence", intensity: 0.12, faint: true },
  ];

  return (
    <div className="help-demo-heatmap">
      {rows.map((row) => (
        <div
          key={row.pct}
          className={`help-demo-heatmap__row${row.faint ? " help-demo-heatmap__row--faint" : ""}`}
          style={warmRowStyle(row.intensity)}
        >
          <div className="help-demo-heatmap__meta">
            {row.rank != null && <span className="help-demo-heatmap__rank">{row.rank}</span>}
            <span className="help-demo-heatmap__pct">{row.pct}%</span>
          </div>
          <span className="help-demo-heatmap__text">{row.text}</span>
        </div>
      ))}
      <p className="help-demo-heatmap__note">
        % = relative strength within each head. Not a probability.
      </p>
    </div>
  );
}

function HelpLegendRow() {
  const items = [
    {
      swatch: { backgroundColor: `rgba(${CEFR_WARM.join(",")}, 0.26)` },
      label: "Supports",
    },
    {
      swatch: { backgroundColor: `rgba(${CEFR_COOL.join(",")}, 0.15)` },
      label: "Opposes",
    },
    {
      swatch: {
        backgroundColor: `rgba(${CEFR_WARM.join(",")}, 0.36)`,
        boxShadow: "inset 0 0 0 1px rgba(180,110,50,0.2)",
      },
      label: "Key tokens",
    },
  ];

  return (
    <div className="help-legend-row">
      {items.map((item) => (
        <span key={item.label} className="help-legend-row__item">
          <span className="help-legend-row__swatch" style={item.swatch} aria-hidden="true" />
          {item.label}
        </span>
      ))}
      <span className="help-legend-row__item help-legend-row__item--ranks">
        <span className="help-legend-row__rank-demo">1</span>
        <span className="help-legend-row__rank-demo">2</span>
        <span className="help-legend-row__rank-demo">3</span>
        Rank badges
      </span>
    </div>
  );
}

function HelpSummaryExample() {
  return (
    <div className="help-demo-summary">
      <p className="help-demo-summary__text">
        The model predicted Japanese mainly because of &quot;Japanese&quot;, &quot;Japan&quot;,
        and &quot;Leite&quot; in the top sentence, which carried 46% of the signal.
      </p>
    </div>
  );
}

export default function AttributionHelpDropdown() {
  const [open, setOpen] = useState(false);
  const rootRef = useRef(null);

  useEffect(() => {
    if (!open) return;
    function onDocClick(e) {
      if (rootRef.current && !rootRef.current.contains(e.target)) {
        setOpen(false);
      }
    }
    function onKey(e) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onDocClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDocClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div className="help-dropdown" ref={rootRef}>
      <button
        type="button"
        className="help-dropdown__trigger"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-controls="help-panel"
      >
        How it works
        <span
          className={`help-dropdown__chevron${open ? " help-dropdown__chevron--open" : ""}`}
          aria-hidden="true"
        >
          ▾
        </span>
      </button>

      <div
        id="help-panel"
        className={`help-dropdown__panel-wrap${open ? " is-open" : ""}`}
        aria-hidden={!open}
      >
        <div className="help-dropdown__panel" role="region" aria-label="How it works">
          <div className="help-dropdown__grid">
            <article className="help-card">
              <h3 className="help-card__label">Attribution</h3>
              <p className="help-card__body">
                One pass per head (CEFR, L1, nationality). Sentences ranked by how much they
                pushed the classifier toward its prediction.
              </p>
              <HelpMiniHeatmap />
              <HelpLegendRow />
            </article>

            <article className="help-card">
              <h3 className="help-card__label">Summary</h3>
              <HelpSummaryExample />
              <p className="help-card__body help-card__body--tight">
                Tick <strong>Summary</strong> before explaining. One sentence restating the top
                attributed words — generated from attribution data, not an open-ended LLM call.
              </p>
              <p className="help-card__footer">
                Uses local Ollama (Mistral). Hard cap: 60 tokens, temperature 0.
              </p>
            </article>
          </div>
        </div>
      </div>
    </div>
  );
}
