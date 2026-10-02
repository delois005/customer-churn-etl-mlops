import pandas as pd
import mlflow
import mlflow.sklearn
import pickle
import logging
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

Path("models").mkdir(exist_ok=True)
Path("logs").mkdir(exist_ok=True)
Path("mlruns").mkdir(exist_ok=True)

logging.basicConfig(
    filename="logs/etl_pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load processed data
df = pd.read_csv(
    "data/processed/customer_data_transformed.csv"
)

features = [
    "age_normalized",
    "monthly_spend_normalized",
    "tenure_months_normalized",
    "support_calls_normalized",
    "contract_encoded",
    "spend_per_month",
    "high_support_usage"
]

X = df[features]
y = df["churn"]

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Model parameters
n_estimators = 100
max_depth = 5
random_state = 42

model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    random_state=random_state
)

# Configure MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Customer_Churn_ETL_MLOps")

with mlflow.start_run() as run:

    # Train model
    model.fit(X_train, y_train)

    # Generate predictions
    predictions = model.predict(X_test)

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    matrix = confusion_matrix(y_test, predictions)

    # Log model parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("test_size", 0.30)

    # Log performance metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Log trained model
    mlflow.sklearn.log_model(
        model,
        name="customer_churn_model",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

    # Save model locally for Flask deployment
    with open(
        "models/customer_churn_model.pkl", "wb"
    ) as model_file:
        pickle.dump(model, model_file)

    logging.info(
        "Model training completed. Accuracy: %.4f",
        accuracy
    )

    print("\nML MODEL TRAINING SUCCESSFUL")
    print("----------------------------")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(matrix)
    print(f"\nMLflow Run ID: {run.info.run_id}")
    print(
        "Model saved to: "
        "models/customer_churn_model.pkl"
    )
