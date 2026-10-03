import logging
import pickle
from itertools import product
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

Path("models").mkdir(exist_ok=True)
Path("logs").mkdir(exist_ok=True)
Path("mlruns").mkdir(exist_ok=True)

logging.basicConfig(
    filename="logs/etl_pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

DATA_PATH = "data/processed/customer_data_transformed.csv"
MODEL_PATH = "models/customer_churn_model_tuned.pkl"

FEATURES = [
    "age_normalized",
    "monthly_spend_normalized",
    "tenure_months_normalized",
    "support_calls_normalized",
    "contract_encoded",
    "spend_per_month",
    "high_support_usage",
]

# Small grid chosen to demonstrate systematic hyperparameter tuning.
PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, None],
}

df = pd.read_csv(DATA_PATH)
X = df[FEATURES]
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Customer_Churn_Hyperparameter_Tuning")

best_model = None
best_params = None
best_metrics = None
best_f1 = -1.0
run_summaries = []

for n_estimators, max_depth in product(
    PARAM_GRID["n_estimators"],
    PARAM_GRID["max_depth"],
):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
    )

    with mlflow.start_run() as run:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", "None" if max_depth is None else max_depth)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.30)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.set_tag("run_type", "hyperparameter_tuning")

        summary = {
            "run_id": run.info.run_id,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }
        run_summaries.append(summary)

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "random_state": 42,
            }
            best_metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }

if best_model is None:
    raise RuntimeError("Hyperparameter tuning did not produce a model.")

with open(MODEL_PATH, "wb") as model_file:
    pickle.dump(best_model, model_file)

logging.info(
    "Hyperparameter tuning completed. Best params=%s Best metrics=%s",
    best_params,
    best_metrics,
)

print("\nHYPERPARAMETER TUNING SUCCESSFUL")
print("--------------------------------")
print(f"Configurations tested: {len(run_summaries)}")

for index, result in enumerate(run_summaries, start=1):
    depth_display = "None" if result["max_depth"] is None else result["max_depth"]
    print(
        f"Run {index}: "
        f"n_estimators={result['n_estimators']}, "
        f"max_depth={depth_display}, "
        f"accuracy={result['accuracy']:.4f}, "
        f"f1={result['f1_score']:.4f}, "
        f"run_id={result['run_id']}"
    )

print("\nBEST CONFIGURATION")
print("------------------")
print(f"Parameters: {best_params}")
print(f"Metrics: {best_metrics}")
print(f"Best tuned model saved to: {MODEL_PATH}")
