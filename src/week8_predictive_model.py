# Week 8 Signature Assignment
# Customer Churn Predictive Analytics
# ETL-Integrated Tableau Dashboard Project

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Define project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = PROJECT_ROOT / "data/raw/customer_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "data/tableau"
MODEL_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Week 8 predictive modeling imports loaded successfully.")

# --------------------------------------------------
# Step 2: Load and validate customer churn data
# --------------------------------------------------

df = pd.read_csv(RAW_DATA)

print("\nCUSTOMER CHURN DATASET")
print("----------------------")
print(f"Total customers: {len(df)}")
print(f"Total columns: {len(df.columns)}")

# Validate required columns
required_columns = [
    "customer_id",
    "age",
    "monthly_spend",
    "tenure_months",
    "support_calls",
    "contract_type",
    "churn"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Check duplicate customer IDs
duplicate_ids = df["customer_id"].duplicated().sum()
print(f"Duplicate customer IDs: {duplicate_ids}")

if duplicate_ids > 0:
    raise ValueError("Duplicate customer IDs detected.")

# Check target variable
if df["churn"].isna().any():
    raise ValueError("Missing churn labels detected.")

if not df["churn"].isin([0, 1]).all():
    raise ValueError("Churn must contain only 0 and 1.")

print("\nMissing values before preprocessing:")
print(df.isnull().sum())

print("\nChurn distribution:")
print(df["churn"].value_counts().sort_index())

print("\nDataset loaded and validated successfully.")

# --------------------------------------------------
# Step 3: Prepare features and split the dataset
# --------------------------------------------------

numeric_features = [
    "age",
    "monthly_spend",
    "tenure_months",
    "support_calls"
]

categorical_features = [
    "contract_type"
]

feature_columns = numeric_features + categorical_features

X = df[feature_columns].copy()
y = df["churn"].astype(int)

# Split before learning any preprocessing parameters
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nTRAINING AND TESTING SPLIT")
print("--------------------------")
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")

print("\nTraining churn distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting churn distribution:")
print(y_test.value_counts().sort_index())

# --------------------------------------------------
# Step 4: Build leakage-safe preprocessing pipeline
# --------------------------------------------------

# Numerical preprocessing
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

# Categorical preprocessing
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Create the complete predictive modeling pipeline
model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)

print("\nPreprocessing and modeling pipeline configured.")

# --------------------------------------------------
# Step 5: Hyperparameter tuning with cross-validation
# --------------------------------------------------

param_grid = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [3, 5, None],
    "classifier__min_samples_leaf": [1, 2]
}

cv_strategy = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=cv_strategy,
    n_jobs=-1,
    refit=True
)

print("\nStarting Random Forest hyperparameter tuning...")

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nHYPERPARAMETER TUNING RESULTS")
print("-----------------------------")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
print(f"Configurations tested: {len(grid_search.cv_results_['params'])}")

# --------------------------------------------------
# Step 6: Evaluate the optimized predictive model
# --------------------------------------------------

# Generate predictions on the testing dataset
y_pred = best_model.predict(X_test)

# Generate churn probabilities
y_prob = best_model.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("\nOPTIMIZED MODEL EVALUATION")
print("--------------------------")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

# --------------------------------------------------
# Step 7: Export customer predictions for Tableau
# --------------------------------------------------

# Generate predictions for all customers
all_predictions = best_model.predict(X)
all_probabilities = best_model.predict_proba(X)[:, 1]

# Prepare Tableau-compatible dataset
tableau_df = df.copy()
# Clean missing values in the Tableau export
# Use medians learned from the training data only
numeric_imputer = best_model.named_steps["preprocessor"] \
    .named_transformers_["num"].named_steps["imputer"]

training_medians = dict(
    zip(numeric_features, numeric_imputer.statistics_)
)

for column in numeric_features:
    tableau_df[column] = tableau_df[column].fillna(
        training_medians[column]
    )
tableau_df["predicted_churn"] = all_predictions
tableau_df["churn_probability"] = all_probabilities

# Classify customers into risk categories
tableau_df["risk_category"] = pd.cut(
    tableau_df["churn_probability"],
    bins=[-0.01, 0.30, 0.70, 1.00],
    labels=["Low Risk", "Medium Risk", "High Risk"]
)

# Identify whether records were used for training or testing
tableau_df["data_split"] = "Training"
tableau_df.loc[X_test.index, "data_split"] = "Testing"

# Export results
output_file = OUTPUT_DIR / "customer_churn_predictions.csv"

tableau_df.to_csv(output_file, index=False)

print("\nTABLEAU PREDICTIONS EXPORTED")
print("---------------------------")
print(f"Total customers: {len(tableau_df)}")
print(f"Output file: {output_file}")

print("\nSample predictions:")
print(
    tableau_df[
        [
            "customer_id",
            "churn",
            "predicted_churn",
            "churn_probability",
            "risk_category",
            "data_split"
        ]
    ].head(10)
)

# --------------------------------------------------
# Step 8: Save the optimized predictive model
# --------------------------------------------------

model_file = MODEL_DIR / "week8_customer_churn_model.pkl"

with open(model_file, "wb") as file:
    pickle.dump(best_model, file)

print("\nOPTIMIZED MODEL SAVED")
print("---------------------")
print(f"Model location: {model_file}")

print("\nWEEK 8 PREDICTIVE MODELING PIPELINE COMPLETE")