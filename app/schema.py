from pydantic import BaseModel
from typing import List

class GPSPoint(BaseModel):
    lat:float
    lon:float
    timestamp:int
class TrajectoryRequest(BaseModel):
    device_id:str
    trajectory:List[GPSPoint]
class PredictionResponse(BaseModel):
    anomaly_score:float
    is_anomalous:bool
    latency_ms:float

