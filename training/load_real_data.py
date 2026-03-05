import pandas as pd
from app.schema import GPSPoint
from app.features import extract_features
import numpy as np
from datetime import datetime


def parse_mixed_timestamp(ts):
    ts = str(ts).strip()

    # Case 1: Full datetime (e.g. 23-06-2025 10:32)
    if "-" in ts:
        try:
            return datetime.strptime(ts, "%d-%m-%Y %H:%M")
        except ValueError:
            return None

    # Case 2: MM:SS(.fraction)
    if ":" in ts:
        try:
            minute, sec = ts.split(":")
            total_seconds = int(minute) * 60 + float(sec)
            return total_seconds
        except ValueError:
            return None

    return None


def load_csv(path):
    df = pd.read_csv(path)

    parsed_times = []
    base_datetime = None
    last_seconds = 0

    for ts in df["timestamp"]:
        parsed = parse_mixed_timestamp(ts)

        # If datetime, reset base
        if isinstance(parsed, datetime):
            base_datetime = parsed
            parsed_times.append(0.0)
            last_seconds = 0.0

        # If relative seconds
        elif isinstance(parsed, (int, float)):
            # Handle rollover (e.g. 25 -> 08)
            if parsed < last_seconds:
                parsed += last_seconds
            parsed_times.append(parsed)
            last_seconds = parsed

        else:
            parsed_times.append(None)

    df["time_sec"] = parsed_times
    df = df.dropna(subset=["time_sec"])

    print("Total GPS points after parsing:", len(df))

    return df


def split_into_trajectories(df, window_size=20, step_size=10):
    points = [
        GPSPoint(
            lat=row["latitude"],
            lon=row["longitude"],
            timestamp=int(row["time_sec"])
        )
        for _, row in df.iterrows()
    ]

    trajectories = []
    for i in range(0, len(points) - window_size, step_size):
        trajectories.append(points[i:i + window_size])

    return trajectories


def build_feature_matrix(trajectories):
    return np.array([extract_features(traj) for traj in trajectories])
