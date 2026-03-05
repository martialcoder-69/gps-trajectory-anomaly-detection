import joblib
import os

MODEL_PATH = "model/isolation_forest.pkl"
model = None


def load_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Trained model not found.")
        model = joblib.load(MODEL_PATH)
    return model
