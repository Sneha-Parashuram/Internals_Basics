import os
import json
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import joblib

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

        svr_mae = mean_absolute_error(y_test, preds)
        svr_rmse = np.sqrt(mean_squared_error(y_test, preds))

        mlflow.log_param("model", "SVR")
        mlflow.log_metric("mae", svr_mae)
        mlflow.log_metric("rmse", svr_rmse)

        results.append({
            "name": "SVR",
            "mae": round(svr_mae, 3),
            "rmse": round(svr_rmse, 3)
        })

    # ---------------- Random Forest ----------------
    with mlflow.start_run(run_name="RandomForest"):
        mlflow.set_tag("domain", "biomechanics")

        rf = RandomForestRegressor(random_state=42)
        rf.fit(X_train, y_train)

        preds = rf.predict(X_test)

        rf_mae = mean_absolute_error(y_test, preds)
        rf_rmse = np.sqrt(mean_squared_error(y_test, preds))

        mlflow.log_param("model", "RandomForest")
        mlflow.log_metric("mae", rf_mae)
        mlflow.log_metric("rmse", rf_rmse)

        results.append({
            "name": "RandomForest",
            "mae": round(rf_mae, 3),
            "rmse": round(rf_rmse, 3)
        })

    # ---------------- Select Best ----------------
    best_model_info = min(results, key=lambda x: x["rmse"])

    if best_model_info["name"] == "SVR":
        best_model = svr
    else:
        best_model = rf

    # ✅ SAVE MODEL (IMPORTANT)
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/model.pkl")

    # ---------------- Save JSON ----------------
    output = {
        "experiment_name": EXPERIMENT_NAME,
        "models": results,
        "best_model": best_model_info["name"],
        "best_metric_name": "rmse",
        "best_metric_value": best_model_info["rmse"]
    }

    os.makedirs("results", exist_ok=True)

    with open("results/step1_s1.json", "w") as f:
        json.dump(output, f, indent=4)

    print("Task 1 completed")

if __name__ == "__main__":
    main()