// Day 10: Churn risk widget.
// Form -> gateway (5000) -> FastAPI (5001) -> model -> probability + verdict.
import { useState } from "react";

const GATEWAY = "http://localhost:5000";

// 13 fields the UI doesn't surface — sent with sane defaults so the API
// still receives the full 19-feature payload.
const BASE_DEFAULTS = {
  SeniorCitizen: 0,
  gender: "Female",
  Partner: "No",
  Dependents: "No",
  PhoneService: "Yes",
  MultipleLines: "No",
  OnlineSecurity: "No",
  OnlineBackup: "No",
  DeviceProtection: "No",
  StreamingTV: "No",
  StreamingMovies: "No",
  PaperlessBilling: "Yes",
};

export default function App() {
  const [form, setForm] = useState({
    tenure: 2,
    Contract: "Month-to-month",
    MonthlyCharges: 95.5,
    TotalCharges: 190.0,
    InternetService: "Fiber optic",
    PaymentMethod: "Electronic check",
    TechSupport: "No",
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  function update(k, v) {
    setForm((f) => ({ ...f, [k]: v }));
  }

  async function submit(e) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const payload = { ...BASE_DEFAULTS, ...form };
      // numeric coercion (form inputs give strings)
      payload.tenure = Number(payload.tenure);
      payload.MonthlyCharges = Number(payload.MonthlyCharges);
      payload.TotalCharges = Number(payload.TotalCharges);

      const r = await fetch(`${GATEWAY}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const body = await r.json();
      if (!body.success) throw new Error(body.error || "Request failed");
      setResult(body.data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const verdictColor =
    result?.verdict === "high" ? "#dc2626"
    : result?.verdict === "medium" ? "#d97706"
    : "#16a34a";

  return (
    <div style={S.page}>
      <h1 style={S.h1}>Churn Risk Estimator</h1>
      <p style={S.sub}>
        Logistic Regression · ROC-AUC 0.84 · served via FastAPI (5001) behind an Express gateway (5000)
      </p>

      <form onSubmit={submit} style={S.form}>
        <Field label="Tenure (months)">
          <input
            type="number" min="0" max="100"
            value={form.tenure}
            onChange={(e) => update("tenure", e.target.value)}
            style={S.input}
          />
        </Field>

        <Field label="Contract">
          <select value={form.Contract}
                  onChange={(e) => update("Contract", e.target.value)}
                  style={S.input}>
            <option>Month-to-month</option>
            <option>One year</option>
            <option>Two year</option>
          </select>
        </Field>

        <Field label="Monthly Charges ($)">
          <input
            type="number" step="0.01" min="0"
            value={form.MonthlyCharges}
            onChange={(e) => update("MonthlyCharges", e.target.value)}
            style={S.input}
          />
        </Field>

        <Field label="Total Charges ($)">
          <input
            type="number" step="0.01" min="0"
            value={form.TotalCharges}
            onChange={(e) => update("TotalCharges", e.target.value)}
            style={S.input}
          />
        </Field>

        <Field label="Internet Service">
          <select value={form.InternetService}
                  onChange={(e) => update("InternetService", e.target.value)}
                  style={S.input}>
            <option>DSL</option>
            <option>Fiber optic</option>
            <option>No</option>
          </select>
        </Field>

        <Field label="Payment Method">
          <select value={form.PaymentMethod}
                  onChange={(e) => update("PaymentMethod", e.target.value)}
                  style={S.input}>
            <option>Electronic check</option>
            <option>Mailed check</option>
            <option>Bank transfer (automatic)</option>
            <option>Credit card (automatic)</option>
          </select>
        </Field>

        <Field label="Tech Support">
          <select value={form.TechSupport}
                  onChange={(e) => update("TechSupport", e.target.value)}
                  style={S.input}>
            <option>No</option>
            <option>Yes</option>
            <option>No internet service</option>
          </select>
        </Field>

        <button type="submit" disabled={loading} style={S.button}>
          {loading ? "Scoring…" : "Estimate churn risk"}
        </button>
      </form>

      {error && <div style={S.error}>⚠ {error}</div>}

      {result && (
        <div style={S.result}>
          <div style={{ ...S.badge, background: verdictColor }}>
            {result.verdict.toUpperCase()} RISK
          </div>

          <div style={S.probRow}>
            <span style={S.probNum}>{(result.probability * 100).toFixed(1)}%</span>
            <span style={S.probLabel}>churn probability</span>
          </div>

          <div style={S.gaugeOuter}>
            <div style={{ ...S.gaugeFill, width: `${result.probability * 100}%`, background: verdictColor }} />
          </div>

          <div style={S.thresholds}>
            low {result.thresholds.low} · medium {result.thresholds.medium} · high {result.thresholds.high}
          </div>

          <div style={S.meta}>
            model: {result.model} · v{result.version}
          </div>
        </div>
      )}
    </div>
  );
}

function Field({ label, children }) {
  return (
    <label style={S.field}>
      <span style={S.label}>{label}</span>
      {children}
    </label>
  );
}

const S = {
  page:  { maxWidth: 560, margin: "40px auto", padding: "0 20px",
           fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, sans-serif", color: "#111" },
  h1:    { fontSize: 28, marginBottom: 4 },
  sub:   { color: "#555", marginTop: 0, marginBottom: 24, fontSize: 14 },
  form:  { display: "grid", gap: 14 },
  field: { display: "grid", gap: 4 },
  label: { fontSize: 13, color: "#333", fontWeight: 600 },
  input: { padding: "8px 10px", fontSize: 14, border: "1px solid #ccc",
           borderRadius: 6, background: "#fff" },
  button:{ marginTop: 8, padding: "10px 16px", fontSize: 15, border: "none",
           borderRadius: 8, background: "#2563eb", color: "#fff", cursor: "pointer" },
  error: { marginTop: 20, padding: 12, background: "#fef2f2", color: "#991b1b",
           border: "1px solid #fecaca", borderRadius: 8, fontSize: 14 },
  result:{ marginTop: 28, padding: 20, background: "#f8fafc",
           border: "1px solid #e2e8f0", borderRadius: 12 },
  badge: { display: "inline-block", padding: "4px 12px", borderRadius: 999,
           color: "#fff", fontSize: 12, fontWeight: 700, letterSpacing: 0.5 },
  probRow: { marginTop: 14, display: "flex", alignItems: "baseline", gap: 10 },
  probNum: { fontSize: 40, fontWeight: 800, letterSpacing: -1 },
  probLabel: { color: "#555", fontSize: 14 },
  gaugeOuter: { marginTop: 12, height: 12, background: "#e5e7eb",
                borderRadius: 999, overflow: "hidden" },
  gaugeFill: { height: "100%", transition: "width 300ms ease" },
  thresholds: { marginTop: 10, fontSize: 12, color: "#666" },
  meta:  { marginTop: 6, fontSize: 12, color: "#94a3b8" },
};