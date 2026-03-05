import math
import numpy as np

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))
def extract_features(trajectory):
    if len(trajectory)<2:
        return np.zeroes(6)
    distances =[]
    speeds = []
    time_gaps = []

    for i in range(1,len(trajectory)):
        p1,p2 = trajectory[i-1],trajectory[i]
        dist = haversine_distance(p1.lat,p1.lon,p2.lat,p2.lon)
        time_diff = max(p2.timestamp-p1.timestamp,1)

        speed = dist/time_diff

        distances.append(dist)
        time_gaps.append(time_diff)
        speeds.append(speed)
    return np.array([
        np.mean(speeds),
        np.max(speeds),
        np.std(speeds),
        np.sum(distances),
        np.mean(time_gaps),
        np.std(time_gaps)
    ])
