import { useState, useRef, useEffect } from "react";
import "./app.css";
import SentenceHeatmap from "./SentenceHeatmap.jsx";
import AttributionHelpDropdown from "./AttributionHelpDropdown.jsx";
import { parseApiResponse, apiFetch, wakeBackend } from "./api.js";

const HEAD_LABELS = { cefr: "CEFR", l1: "L1", nat: "Nationality" };

function buildBody(text, corrected, dualMode) {
  const body = { text, dual_mode: dualMode };
  if (corrected.trim()) body.text_corrected = corrected;
  return body;
}

function headPrediction(head, result) {
  if (head === "cefr") return result.cefr;
  if (head === "l1") return result.l1;
  return result.nationality;
}

export default function App() {
  const [text, setText] = useState("");
  const [corrected, setCorrected] = useState("");
  const [dualMode, setDualMode] = useState("raw_only");
  const [loading, setLoading] = useState(false);
  const [shapLoading, setShapLoading] = useState(false);
  const [err, setErr] = useState(null);
  const [shapErr, setShapErr] = useState(null);
  const [result, setResult] = useState(null);
  const [shapResult, setShapResult] = useState(null);
  const [headCefr, setHeadCefr] = useState(true);
  const [headL1, setHeadL1] = useState(true);
  const [headNat, setHeadNat] = useState(true);
  const [summaryEnabled, setSummaryEnabled] = useState(true);
  const [ocrLoading, setOcrLoading] = useState(false);
  const [coldStartMsg, setColdStartMsg] = useState(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    wakeBackend();
  }, []);

  const onSlowRequest = () =>
    setColdStartMsg("Model loading — first request after idle may take ~30 seconds.");

  async function handleEssayUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setOcrLoading(true);
    setErr(null);
    setResult(null);
    setShapResult(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const r = await apiFetch("/upload/essay", { method: "POST", body: form }, { onSlow: onSlowRequest });
      const j = await parseApiResponse(r);
      setText(j.text);
      setCorrected("");
    } catch (x) {
      setErr(String(x.message || x));
    } finally {
      setOcrLoading(false);
      e.target.value = "";
    }
  }

  async function submitPredict(e) {
    e.preventDefault();
    setLoading(true);
    setErr(null);
    setResult(null);
    setColdStartMsg(null);
    try {
      const r = await apiFetch(
        "/predict",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(buildBody(text, corrected, dualMode)),
        },
        { onSlow: onSlowRequest },
      );
      setResult(await parseApiResponse(r));
    } catch (x) {
      setErr(String(x.message || x));
    } finally {
      setLoading(false);
    }
  }

  async function submitShap(e) {
    e.preventDefault();
    const heads = [];
    if (headCefr) heads.push("cefr");
    if (headL1) heads.push("l1");
    if (headNat) heads.push("nat");
    if (heads.length === 0) {
      setShapErr("Select at least one head (CEFR / L1 / nationality).");
      return;
    }
    setShapLoading(true);
    setShapErr(null);
    setShapResult(null);
    setColdStartMsg(null);
    try {
      const body = {
        ...buildBody(text, corrected, dualMode),
        heads,
        summarize: true,
        ollama_narrate: summaryEnabled,
      };
      const r = await apiFetch(
        "/predict/shap_explain",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
        { onSlow: onSlowRequest },
      );
      const j = await parseApiResponse(r);
      setShapResult(j);
    } catch (x) {
      setShapErr(String(x.message || x));
    } finally {
      setShapLoading(false);
    }
  }

  const isDual = dualMode === "dual";

  return (
    <div className="page">
      <header className="page__header">
        <h1 className="page__title">EF-CamDAT L2 Profiler</h1>
        <AttributionHelpDropdown />
      </header>

      <form onSubmit={(e) => e.preventDefault()}>
        <div className="section">
          <div className="section__label-row">
            <label className="section__label" htmlFor="learner-text">
              Learner text
            </label>
            <input
              ref={fileInputRef}
              id="essay-upload"
              type="file"
              className="upload-input"
              accept=".txt,.pdf,image/*"
              onChange={handleEssayUpload}
            />
            <button
              type="button"
              className="btn btn--ghost"
              disabled={ocrLoading || loading || shapLoading}
              onClick={() => fileInputRef.current?.click()}
            >
              {ocrLoading ? "Transcribing…" : "Upload essay"}
            </button>
          </div>
          <p className="upload-hint">PDF, photo scan, or .txt — OCR fills the box below</p>
          <textarea
            id="learner-text"
            className="input"
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={8}
          />
          <p className="advanced-toggle">
            {isDual ? (
              <>
                Dual mode: model sees learner text and teacher correction.{" "}
                <button
                  type="button"
                  className="link-btn"
                  onClick={() => setDualMode("raw_only")}
                >
                  Switch to raw-only
                </button>
              </>
            ) : (
              <button
                type="button"
                className="link-btn"
                onClick={() => setDualMode("dual")}
              >
                Advanced: dual mode (RAW + corrected)
              </button>
            )}
          </p>
        </div>

        {isDual && (
          <div className="section">
            <label className="section__label" htmlFor="corrected-text">
              Corrected text
            </label>
            <textarea
              id="corrected-text"
              className="input"
              value={corrected}
              onChange={(e) => setCorrected(e.target.value)}
              rows={6}
              placeholder="Teacher-corrected version of the learner text"
            />
          </div>
        )}

        <div className="actions">
          <button
            type="button"
            className="btn btn--primary"
            onClick={submitPredict}
            disabled={loading || shapLoading}
          >
            {loading ? "Predicting…" : "Predict"}
          </button>

          <span className="actions__divider" aria-hidden="true" />

          <div className="actions__group">
            <span className="actions__shap-label">Heads</span>
            <label className="check">
              <input type="checkbox" checked={headCefr} onChange={(e) => setHeadCefr(e.target.checked)} />
              CEFR
            </label>
            <label className="check">
              <input type="checkbox" checked={headL1} onChange={(e) => setHeadL1(e.target.checked)} />
              L1
            </label>
            <label className="check">
              <input type="checkbox" checked={headNat} onChange={(e) => setHeadNat(e.target.checked)} />
              Nat.
            </label>
            <label className="check">
              <input
                type="checkbox"
                checked={summaryEnabled}
                onChange={(e) => setSummaryEnabled(e.target.checked)}
              />
              Summary
            </label>
          </div>

          <button
            type="button"
            className="btn btn--secondary"
            onClick={submitShap}
            disabled={loading || shapLoading}
          >
            {shapLoading ? "Explaining…" : "Explain attributions"}
          </button>
        </div>
      </form>

      {coldStartMsg && (
        <div className="alert alert--info" role="status">
          {coldStartMsg}
        </div>
      )}

      {err && <div className="alert alert--error">{err}</div>}

      {result && (
        <div className="card card--predict">
          <h2 className="card__title">Predictions</h2>
          <div className="pred-grid">
            <div className="pred-grid__item">
              <span className="pred-grid__label">CEFR level</span>
              <span className="pill">{result.cefr}</span>
            </div>
            <div className="pred-grid__item">
              <span className="pred-grid__label">L1</span>
              <span className="pred-grid__value">{result.l1}</span>
            </div>
            <div className="pred-grid__item">
              <span className="pred-grid__label">Nationality</span>
              <span className="pred-grid__value">{result.nationality}</span>
            </div>
          </div>
          <p className="card__probs">
            <strong>Top CEFR probs</strong>:{" "}
            {result.probs_top_cefr
              ?.map((x) => `${x.label} (${(x.prob * 100).toFixed(1)}%)`)
              .join(" · ")}
          </p>
          {result.explanation && (
            <div className="card__body">
              <span className="card__body-title">Explanation</span>
              <p className="card__body-text">{result.explanation}</p>
            </div>
          )}
        </div>
      )}

      {shapErr && <div className="alert alert--error">{shapErr}</div>}

      {shapResult && shapResult.attribution_version === undefined && !shapResult.attribution_methods && (
        <div className="alert alert--error">
          Old API still running — in a terminal run:{" "}
          <code>pkill -f uvicorn; PYTHONPATH=. uvicorn api.main:app --reload --port 8000</code>
        </div>
      )}
      {shapResult && (
        <div className="card card--shap">
          <h2 className="card__title">Why this prediction</h2>
          {shapResult.head_comparison && (
            <p className="shap-head__comparison">{shapResult.head_comparison}</p>
          )}
          {Object.entries(shapResult.sentence_shap || {}).map(([head, sentences]) => (
            <div key={head} className="shap-head">
              <h3 className="shap-head__title">
                {HEAD_LABELS[head] || head}
                <span className="shap-head__pred"> → {headPrediction(head, shapResult)}</span>
              </h3>
              {shapResult.narrative?.[head] && (
                <p className="shap-head__narrative">{shapResult.narrative[head]}</p>
              )}
              {shapResult.ollama_narrative?.[head]?.trim() && (
                <p className="shap-head__summary">{shapResult.ollama_narrative[head]}</p>
              )}
              <SentenceHeatmap head={head} sentences={sentences} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
