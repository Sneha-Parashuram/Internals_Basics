import os
import json
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# CONFIG
DATA_PATH = "data/training_data.csv"
EXPERIMENT_NAME = "biomotion-injury-risk-score"

def main():
    print("=== Task 2: Hyperparameter Tuning ===")

    df = pd.read_csv(DATA_PATH)

    X = df[["stride_length_cm", "ground_contact_ms", "hip_drop_degrees", "fatigue_index"]]
    y = df["injury_risk_score"]

    # PARAM GRID
    param_grid = {
        "n_estimators": [50, 150, 250],
        "max_depth": [5, 10, 20],
        "min_samples_split": [2, 3, 5]
    }

    model = RandomForestRegressor(random_state=42)

    mlflow.set_experiment(EXPERIMENT_NAME)

    # PARENT RUN
    with mlflow.start_run(run_name="tuning-biomotion"):

        grid = GridSearchCV(
            model,
            param_grid,
            cv=3,
            scoring="neg_mean_squared_error",
            verbose=0
        )

        grid.fit(X, y)

        total_trials = len(grid.cv_results_["params"])

        # LOG EACH TRIAL (nested runs)
        for i, params in enumerate(grid.cv_results_["params"]):
            with mlflow.start_run(nested=True):
                mlflow.log_params(params)
                rmse = np.sqrt(-grid.cv_results_["mean_test_score"][i])
                mlflow.log_metric("rmse", rmse)

        best_params = grid.best_params_
        best_rmse = np.sqrt(-grid.best_score_)

        # Train best model
        best_model = grid.best_estimator_
        preds = best_model.predict(X)
        best_mae = mean_absolute_error(y, preds)

        # OUTPUT JSON
        output = {
            "search_type": "grid",
            "n_folds": 3,
            "total_trials": total_trials,
            "best_params": best_params,
            "best_mae": round(best_mae, 3),
            "best_cv_mae": round(best_rmse, 3),
            "parent_run_name": "tuning-biomotion"
        }

        os.makedirs("results", exist_ok=True)

        with open("results/step2_s2.json", "w") as f:
            json.dump(output, f, indent=4)

    print(" Task 2 completed")

if __name__ == "__main__":
    main()