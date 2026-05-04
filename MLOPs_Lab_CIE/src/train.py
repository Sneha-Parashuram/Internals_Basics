import os
import json
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

# CONFIG
DATA_PATH = "data/training_data.csv"
EXPERIMENT_NAME = "biomotion-injury-risk-score"

def main():
    print("=== Task 1: Biomechanics Model Training ===")

    df = pd.read_csv(DATA_PATH)

    X = df[["stride_length_cm", "ground_contact_ms", "hip_drop_degrees", "fatigue_index"]]
    y = df["injury_risk_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    results = []

    # ---------------- SVR ----------------
    with mlflow.start_run(run_name="SVR"):
        mlflow.set_tag("domain", "biomechanics")

        svr = SVR()
        svr.fit(X_train, y_train)

        preds = svr.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        mlflow.log_param("model", "SVR")
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)

        results.append({
            "name": "SVR",
            "mae": round(mae, 3),
            "rmse": round(rmse, 3)
        })

    # ---------------- Random Forest ----------------
    with mlflow.start_run(run_name="RandomForest"):
        mlflow.set_tag("domain", "biomechanics")

        rf = RandomForestRegressor(random_state=42)
        rf.fit(X_train, y_train)

        preds = rf.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        mlflow.log_param("model", "RandomForest")
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)

        results.append({
            "name": "RandomForest",
            "mae": round(mae, 3),
            "rmse": round(rmse, 3)
        })

    # ---------------- Select Best ----------------
    best_model = min(results, key=lambda x: x["rmse"])

    output = {
        "experiment_name": EXPERIMENT_NAME,
        "models": results,
        "best_model": best_model["name"],
        "best_metric_name": "rmse",
        "best_metric_value": best_model["rmse"]
    }

    os.makedirs("results", exist_ok=True)

    with open("results/step1_s1.json", "w") as f:
        json.dump(output, f, indent=4)

    print("✅ Task 1 completed")

if __name__ == "__main__":
    main()