# GPS Trajectory Anomaly Detection System

This project implements a production-style machine learning inference service for detecting anomalous GPS trajectories.

The system processes real-world GPS mobility logs, extracts trajectory-level features, and detects abnormal movement patterns using an Isolation Forest model.

---

## Features

- Real-world GPS data ingestion
- Robust mixed timestamp parsing
- Sliding window trajectory segmentation
- Feature engineering for mobility behavior
- Isolation Forest anomaly detection
- FastAPI inference service
- Real-time anomaly scoring
- Latency monitoring

---

## Architecture

Raw GPS Logs
    ↓
Timestamp Normalization
    ↓
Sliding Window Trajectory Segmentation
    ↓
Feature Extraction
    ↓
Isolation Forest Model
    ↓
FastAPI Inference Service

---

## Project Structure

app/
    API and inference logic

training/
    Model training pipeline

model/
    Trained anomaly detection model

---

## Features Extracted

Each trajectory is converted into numerical features:

- Mean speed
- Maximum speed
- Speed variance
- Total distance traveled
- Mean time gap
- Time gap variance

These features represent movement behavior and allow anomaly detection.

---

## API Endpoints

### Health Check

GET /health

Returns service status.

---

### Predict Anomaly

POST /predict

Example request:

{
  "device_id": "D1",
  "trajectory": [
    {"lat": 19.0147, "lon": 73.0392, "timestamp": 0},
    {"lat": 19.0148, "lon": 73.0393, "timestamp": 60},
    {"lat": 19.0149, "lon": 73.0394, "timestamp": 120}
  ]
}

Example response:

{
  "anomaly_score": 0.18,
  "is_anomalous": true,
  "latency_ms": 10.2
}

---

## Model Training

The anomaly detection model is trained using real GPS mobility data collected from a mobile application.

Data processing steps:

1. Mixed timestamp normalization
2. Sliding window trajectory segmentation
3. Feature extraction
4. Isolation Forest training

---

## Running the Project

Install dependencies:

pip install -r requirements.txt

Start the API server:

uvicorn app.main:app --reload

Open Swagger documentation:

http://127.0.0.1:8000/docs

---

## Future Improvements

- Add acceleration-based features
- Batch inference
- Model monitoring
- Real-time streaming ingestion
