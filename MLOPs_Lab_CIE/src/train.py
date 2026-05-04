import os
import json
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "data/training_data.csv"
RESULT_PATH = "results/step1_s1.json"

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, r2, mape

def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop("injury_risk_score", axis=1)
    y = df["injury_risk_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_experiment("gpuforge-job-completion")

    results = []

    # -------- Ridge --------
    with mlflow.start_run(run_name="Ridge"):
        model = Ridge()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae, rmse, r2, mape = evaluate(y_test, preds)

        mlflow.log_params({"model": "Ridge"})
        mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2, "mape": mape})

        results.append({
            "name": "Ridge",
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape
        })

    # -------- Gradient Boosting --------
    with mlflow.start_run(run_name="GradientBoosting"):
        model = GradientBoostingRegressor(random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae, rmse, r2, mape = evaluate(y_test, preds)

        mlflow.log_params({"model": "GradientBoosting"})
        mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2, "mape": mape})

        results.append({
            "name": "GradientBoosting",
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape
        })

    # Select best
    best = min(results, key=lambda x: x["rmse"])

    output = {
        "experiment_name": "gpuforge-job-completion",
        "models": results,
        "best_model": best["name"],
        "best_metric_name": "rmse",
        "best_metric_value": best["rmse"]
    }

    os.makedirs("results", exist_ok=True)

    with open(RESULT_PATH, "w") as f:
        json.dump(output, f, indent=4)

    print("Step 1 completed")

if __name__ == "__main__":
    main()