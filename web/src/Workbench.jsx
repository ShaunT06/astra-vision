import { useEffect, useRef, useState } from "react";
import { API, IMAGES } from "./assets.js";

const DEMO = {
  id: "SV-DEMO-01",
  filename: "demo-frame.jpg",
  label: "MAIN BATTLE TANK",
  confidence: 0.974,
  detections: [{ label: "MAIN BATTLE TANK", confidence: 0.974, x: 18, y: 20, width: 61, height: 54 }],
  mode: "DEMO INFERENCE",
};

export default function Workbench() {
  const fileRef = useRef(null);
  const [api, setApi] = useState("checking");
  const [result, setResult] = useState(DEMO);
  const [preview, setPreview] = useState(IMAGES.tank);
  const [frameLabel, setFrameLabel] = useState("DEMO FRAME / 3840 × 2160");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const ctl = new AbortController();
    fetch(`${API}/`, { signal: ctl.signal })
      .then((r) => (r.ok ? setApi("ready") : setApi("offline")))
      .catch((e) => e.name !== "AbortError" && setApi("offline"));
    return () => ctl.abort();
  }, []);

  useEffect(() => () => preview?.startsWith("blob:") && URL.revokeObjectURL(preview), [preview]);

  const loadDemo = () => {
    setError("");
    setResult(DEMO);
    setPreview(IMAGES.tank);
    setFrameLabel("DEMO FRAME / 3840 × 2160");
  };

  const onFile = async (e) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setError("That file is not an image. Choose a JPG, PNG or WebP frame.");
      return;
    }
    setError("");
    setBusy(true);
    setPreview(URL.createObjectURL(file));
    setFrameLabel(`UPLOAD / ${file.name.toUpperCase().slice(0, 28)}`);
    setResult(null);
    try {
      const body = new FormData();
      body.append("file", file);
      const res = await fetch(`${API}/classify`, { method: "POST", body });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const d = data.detail || data;
        throw new Error(d.message || `The classifier rejected this frame (HTTP ${res.status}).`);
      }
      setResult(data);
      setApi("ready");
    } catch (err) {
      const offline = err instanceof TypeError;
      if (offline) setApi("offline");
      setError(
        offline
          ? "Classification API unreachable. Check that the backend is running and REACT_APP_BACKEND_URL points to it, then try again."
          : err.message
      );
    } finally {
      setBusy(false);
    }
  };

  const pct = result ? (result.confidence * 100).toFixed(1) : null;
  const isDemo = result?.mode === "DEMO INFERENCE";

  return (
    <section id="application" className="section workbench" data-testid="workbench-section">
      <div className="wrap">
        <div className="workbench__head reveal">
          <div>
            <p className="eyebrow">
              <i /> CLASSIFICATION WORKBENCH
            </p>
            <h2 className="display">
              TURN RAW IMAGERY
              <br />
              <span className="red">INTO CERTAINTY.</span>
            </h2>
            <p className="lede">
              Upload a frame and receive an object classification with a confidence score and a marked
              detection region. Each result is returned as structured JSON, ready for downstream tooling.
            </p>
          </div>
          <div
            className={`api-chip api-chip--${api}`}
            data-testid="api-status-chip"
            role="status"
          >
            <b>{api === "ready" ? "API READY" : api === "offline" ? "API OFFLINE" : "API CHECKING"}</b>
            <span className="mono">POST /api/classify</span>
          </div>
        </div>

        <div className="panel reveal" data-testid="target-analysis-card">
          <div className="panel__bar mono">
            <span data-testid="target-analysis-title">TARGET ANALYSIS</span>
            <span>NODE 07</span>
            <span className="scan-active" data-testid="scan-status">
              <i /> {busy ? "SCANNING" : "SCAN ACTIVE"}
            </span>
          </div>
          <div className="panel__body">
            <div className="frame" data-testid="analysis-frame">
              <img src={preview} alt="Imagery under analysis" data-testid="analysis-image" />
              <div className="frame__grid" aria-hidden="true" />
              {busy && <div className="frame__sweep" aria-hidden="true" />}
              {result?.detections?.map((d, i) => (
                <div
                  key={i}
                  className="bbox"
                  data-testid={`detection-box-${i}`}
                  style={{ left: `${d.x}%`, top: `${d.y}%`, width: `${d.width}%`, height: `${d.height}%` }}
                >
                  <span className="mono">
                    {d.label} {(d.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
              <span className="frame__tag mono" data-testid="frame-label">
                {frameLabel}
              </span>
            </div>

            <div className="readout" data-testid="analysis-readout">
              <p className="mono muted">PRIMARY DETECTION</p>
              {busy && <p className="readout__label" data-testid="analysis-busy">ANALYZING…</p>}
              {!busy && result && (
                <>
                  <p className="readout__label" data-testid="classification-label">
                    {result.label}
                  </p>
                  <div className="readout__conf">
                    <span className="mono muted">CONFIDENCE</span>
                    <strong data-testid="confidence-value">{pct}%</strong>
                  </div>
                  <div
                    className="meter"
                    role="progressbar"
                    aria-valuenow={Math.round(result.confidence * 100)}
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-label="Confidence"
                    data-testid="confidence-bar"
                  >
                    <span style={{ width: `${pct}%` }} />
                  </div>
                  <span className={`badge ${isDemo ? "" : "badge--live"}`} data-testid="inference-mode-badge">
                    {result.mode}
                  </span>
                  {result.uncertain && (
                    <p className="warn mono" data-testid="uncertain-warning">
                      LOW CERTAINTY. TREAT WITH CAUTION.
                    </p>
                  )}
                  {result.top3 && (
                    <ul className="top3 mono" data-testid="top3-list">
                      {result.top3.map((t) => (
                        <li key={t.label}>
                          <span>{t.label}</span>
                          <span>{(t.confidence * 100).toFixed(1)}%</span>
                        </li>
                      ))}
                    </ul>
                  )}
                  {result.explanation && <p className="explain">{result.explanation}</p>}
                  <p className="mono muted small" data-testid="result-meta">
                    {result.id}
                    {result.latency_ms != null && ` · ${result.latency_ms} ms`}
                  </p>
                </>
              )}
              {!busy && !result && !error && <p className="muted">No result yet.</p>}
              {error && (
                <p className="error" role="alert" data-testid="api-error">
                  {error}
                </p>
              )}

              <div className="actions">
                <input
                  ref={fileRef}
                  type="file"
                  accept="image/*"
                  hidden
                  onChange={onFile}
                  data-testid="image-upload-input"
                />
                <button
                  className="btn btn--red"
                  data-testid="analyze-imagery-button"
                  disabled={busy}
                  onClick={() => fileRef.current?.click()}
                >
                  Analyze imagery
                </button>
                <button className="btn btn--ghost" data-testid="load-demo-button" onClick={loadDemo}>
                  Load demo frame
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
