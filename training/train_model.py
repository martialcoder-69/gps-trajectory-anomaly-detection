import joblib
from sklearn.ensemble import IsolationForest
from training.load_real_data import load_csv,split_into_trajectories,build_feature_matrix


if __name__ == "__main__":
    print("Loading real GPS data...")
    df = load_csv("gps_data_final.csv")

    print("Splitting into trajectories...")
    trajectories = split_into_trajectories(df)

    print(f"Total trajectories: {len(trajectories)}")

    print("Extracting features...")
    X_train = build_feature_matrix(trajectories)

    print("Training Isolation Forest...")
    model = IsolationForest(
        n_estimators=300,
        contamination=0.03,
        random_state=42
    )
    model.fit(X_train)

    print("Saving model...")
    joblib.dump(model, "model/isolation_forest.pkl")

    print("Training complete.")
