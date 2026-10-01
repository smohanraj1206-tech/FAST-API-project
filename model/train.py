import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "student_performance.csv")
MODEL_FILE = os.path.join(os.path.dirname(__file__), "model.pkl")

FEATURE_NAMES = [
    "study_hours", "sleep_hours", "attendance", "assignment_completion",
    "previous_gpa", "screen_time", "self_study_hours",
    "revision_frequency", "test_frequency", "exercise_hours", "class_participation"
]

def train_model():
    print(f"[Training ML Model] Loading dataset from: {DATA_FILE}")
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Dataset not found at {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    X = df[FEATURE_NAMES]
    y = df["target_gpa"]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42, max_depth=12))
    ])

    pipeline.fit(X, y)

    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"[Training Complete] Saved trained Scikit-Learn model to: {MODEL_FILE}")
    return pipeline

if __name__ == "__main__":
    train_model()
