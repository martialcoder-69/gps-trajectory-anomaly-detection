from fastapi import FastAPI, HTTPException
import time
import numpy as np

from app.schema import TrajectoryRequest, PredictionResponse
from app.features import extract_features
from app.model import load_model

app = FastAPI(title="GPS Anomaly Inference Service")


@app.on_event("startup")
def startup_event():
    load_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TrajectoryRequest):
    if len(request.trajectory) < 5:
        raise HTTPException(status_code=400, detail="Trajectory too short")

    start_time = time.time()

    # Feature extraction
    features = extract_features(request.trajectory).reshape(1, -1)

    model = load_model()

    # Isolation Forest decision
    raw_score = model.decision_function(features)[0]
    anomaly_score = float(max(0, -raw_score))

    is_anomalous = model.predict(features)[0] == -1

    latency_ms = (time.time() - start_time) * 1000

    return PredictionResponse(
        anomaly_score=float(anomaly_score),
        is_anomalous=bool(is_anomalous),
        latency_ms=float(latency_ms)
    )
