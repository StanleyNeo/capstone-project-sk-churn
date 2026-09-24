# Day 10: FastAPI service for the churn model.
# Loads model.joblib at startup, serves POST /predict with the Day 9 envelope.
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / "api" / ".env")

PORT = int(os.getenv("PORT", "5001"))
CLIENT_URLS = [u.strip() for u in os.getenv("CLIENT_URLS", "http://localhost:3000").split(",")]

MODEL_PATH = ROOT / "model.joblib"
CARD_PATH  = ROOT / "model_card.json"

MODEL = None
CARD  = None


def err(status, msg):
    return JSONResponse(status_code=status, content={"success": False, "error": msg})


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, CARD
    MODEL = joblib.load(MODEL_PATH)
    with open(CARD_PATH) as f:
        CARD = json.load(f)

    n_features = len(CARD["features_numeric"]) + len(CARD["features_categorical"])

    print("=" * 56)
    print(" TELCO CHURN API (PYTHON/FASTAPI)  v1.0")
    print("=" * 56)
    print(f" Port: {PORT}     URL: http://localhost:{PORT}")
    print(f" Model loaded ✅  ({CARD['algorithm']})")
    print(f" Test ROC-AUC:    {CARD['test_roc_auc']}")
    print(f" Trained on:      {CARD['n_train']} rows · {n_features} features")
    print(" Endpoints:")
    print("   GET  /health       service + model status")
    print("   GET  /model-card   provenance JSON")
    print("   POST /predict      {customer features} -> {probability, verdict}")
    print("=" * 56)
    yield


app = FastAPI(title="telco-churn-api", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CLIENT_URLS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(404)
async def not_found(request: Request, exc):
    return err(404, f"Route not found: {request.method} {request.url.path}")


class CustomerIn(BaseModel):
    # Numeric
    SeniorCitizen:  int   = Field(..., ge=0, le=1, examples=[0])
    tenure:         int   = Field(..., ge=0, le=100, examples=[12])
    MonthlyCharges: float = Field(..., ge=0, examples=[70.35])
    TotalCharges:   float = Field(..., ge=0, examples=[840.5])

    # Categorical (str; the pipeline's OneHotEncoder handles unknowns)
    gender:           str
    Partner:          str
    Dependents:       str
    PhoneService:     str
    MultipleLines:    str
    InternetService:  str
    OnlineSecurity:   str
    OnlineBackup:     str
    DeviceProtection: str
    TechSupport:      str
    StreamingTV:      str
    StreamingMovies:  str
    Contract:         str
    PaperlessBilling: str
    PaymentMethod:    str


def verdict_from(prob: float) -> str:
    if prob >= 0.70:
        return "high"
    if prob >= 0.40:
        return "medium"
    return "low"


@app.get("/health")
async def health():
    return {
        "success": True,
        "service": "telco-churn-api",
        "version": "1.0",
        "model_loaded": MODEL is not None,
        "test_roc_auc": CARD["test_roc_auc"] if CARD else None,
        "time": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/model-card")
async def model_card():
    return {"success": True, "data": CARD}


@app.post("/predict")
async def predict(body: CustomerIn):
    row = body.model_dump()
    X = pd.DataFrame([row])

    proba = float(MODEL.predict_proba(X)[0, 1])
    v = verdict_from(proba)

    return {
        "success": True,
        "data": {
            "probability": round(proba, 4),
            "verdict":     v,
            "thresholds":  {"low": "< 0.40", "medium": "0.40 – 0.70", "high": "≥ 0.70"},
            "model":       CARD["model_name"],
            "version":     CARD["version"],
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)