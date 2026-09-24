// Day 10: Express gateway in front of the FastAPI churn service.
// Same envelope, same routes as the frontend expects — but the ML
// model lives behind this in Python. The frontend never talks to
// FastAPI directly.
const express = require("express");
const cors = require("cors");

const PORT          = process.env.PORT || 5000;
const ML_SERVICE    = process.env.ML_SERVICE || "http://localhost:5001";
const CLIENT_ORIGINS = (process.env.CLIENT_ORIGINS
  || "http://localhost:3000,http://localhost:5173,http://localhost:5174"
).split(",").map(s => s.trim());

const app = express();
app.use(cors({ origin: CLIENT_ORIGINS }));
app.use(express.json());

// --- proxy helper: forward JSON, keep the envelope, keep status codes ---
async function forward(path, init = {}) {
  const url = `${ML_SERVICE}${path}`;
  try {
    const r = await fetch(url, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init.headers || {}) },
    });
    const body = await r.json();
    return { status: r.status, body };
  } catch (e) {
    return {
      status: 502,
      body: { success: false, error: `ML service unreachable: ${e.message}` },
    };
  }
}

// --- GET /api/health : report gateway + upstream status ---
app.get("/api/health", async (_req, res) => {
  const upstream = await forward("/health");
  res.json({
    success: true,
    service: "telco-churn-gateway",
    version: "1.0",
    upstream: {
      service: upstream.body.service || null,
      model_loaded: upstream.body.model_loaded || false,
      test_roc_auc: upstream.body.test_roc_auc || null,
      reachable: upstream.status === 200,
    },
  });
});

// --- GET /api/model-card : passthrough ---
app.get("/api/model-card", async (_req, res) => {
  const r = await forward("/model-card");
  res.status(r.status).json(r.body);
});

// --- POST /api/predict : the one the frontend will call ---
app.post("/api/predict", async (req, res) => {
  const r = await forward("/predict", {
    method: "POST",
    body: JSON.stringify(req.body),
  });
  res.status(r.status).json(r.body);
});

// --- 404 fallback: envelope, not HTML ---
app.use((req, res) => {
  res.status(404).json({
    success: false,
    error: `Route not found: ${req.method} ${req.originalUrl}`,
  });
});

app.listen(PORT, () => {
  console.log("=".repeat(52));
  console.log(" TELCO CHURN GATEWAY (EXPRESS)  v1.0");
  console.log("=".repeat(52));
  console.log(` Port:   ${PORT}   URL: http://localhost:${PORT}`);
  console.log(` ML svc: ${ML_SERVICE}`);
  console.log(` CORS:   ${CLIENT_ORIGINS.join(", ")}`);
  console.log(" Endpoints:");
  console.log("   GET  /api/health      gateway + upstream status");
  console.log("   GET  /api/model-card  provenance passthrough");
  console.log("   POST /api/predict     proxy -> FastAPI /predict");
  console.log("=".repeat(52));
});