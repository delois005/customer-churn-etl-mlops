# Customer Churn Prediction MLOps Pipeline

## Overview

This project demonstrates an end-to-end ETL and MLOps pipeline for customer churn prediction. The solution extracts customer data from CSV, a relational database, and a web API; performs preprocessing and feature engineering; trains and tunes a machine learning model; tracks experiments with MLflow; serves the model through a Flask REST API; orchestrates ETL and training with Apache Airflow; automates validation and Docker build testing with GitHub Actions; and monitors the running service with Prometheus and Grafana.

## Technologies Used

- Python
- Pandas
- Dask
- SQLAlchemy
- Requests
- Scikit-learn
- Flask
- MLflow
- Apache Airflow
- Docker
- Prometheus
- Grafana
- GitHub Actions
- SQLite

## Environment Setup

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ETL Pipeline

### Extract

The extraction phase retrieves data from multiple sources:

- CSV files containing structured customer data
- Relational database records using SQLAlchemy
- Web API data using Python Requests

The API response includes free-text fields such as `title` and `body`, providing semi-structured data that contains unstructured textual content.

CSV extraction uses explicit batch processing. The `extract_csv()` function sets `batch_size=5` and uses:

```python
pd.read_csv(file_path, chunksize=batch_size)
```

Each chunk is processed as a separate batch and logged before the batches are concatenated. This demonstrates bounded-memory batch ingestion that can scale to larger files.

### Transform

The transform stage performs:

- Missing-value handling
- Feature normalization
- Feature engineering
- Data validation
- Parallel/scalable processing with Dask

### Load

The transformed customer data is loaded into SQLite. SQLAlchemy automates database interaction and record verification.

## Machine Learning

A Random Forest customer churn classifier is trained with Scikit-learn.

Model outputs include:

- Predicted class
- Prediction probability
- Model version/artifact
- Accuracy
- Precision
- Recall
- F1 score

The baseline model is saved for Flask deployment.

## Hyperparameter Tuning

`src/tune_model.py` performs systematic Random Forest tuning across multiple combinations of:

- `n_estimators`: 50, 100, 200
- `max_depth`: 3, 5, None

Each configuration is recorded as a separate MLflow run with parameters and evaluation metrics. The configuration with the highest F1 score is selected and saved as:

```text
models/customer_churn_model_tuned.pkl
```

Run tuning with:

```bash
python src/tune_model.py
```

Open MLflow to compare the tuning runs:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## MLflow Experiment Tracking

MLflow records:

- Model parameters
- Accuracy
- Precision
- Recall
- F1 score
- Run IDs
- Experiment information

The baseline training run and the hyperparameter-tuning experiment provide reproducible model-development evidence.

## REST API

The trained model is deployed locally through Flask.

Endpoints:

- `/health`
- `/predict`
- `/metrics`

Run the API:

```bash
python app.py
```

The local Flask service is available at:

```text
http://localhost:5001
```

## Workflow Orchestration

Apache Airflow coordinates the ETL and machine learning workflow. The DAG controls the execution order of:

1. Data extraction
2. Data transformation
3. Data loading
4. Model training

The DAG is located under:

```text
airflow/dags/etl_mlops_dag.py
```

## CI/CD

GitHub Actions automates integration and deployment testing, including:

- Code validation
- Dependency installation
- ETL pipeline execution
- Model training
- Docker image build and deployment testing

The workflow is stored at:

```text
.github/workflows/ci.yml
```

## Docker Deployment

The customer churn application can run in a Docker container and expose the Flask service on port 5001. This demonstrates reproducible local container deployment. Persistent AWS/Azure deployment remains a future extension.

## Monitoring and Logging

Prometheus collects application metrics exposed by the Flask API.

Grafana dashboards display:

- API Status
- API Request Rate
- Average API Response Time

Python logging records ETL progress, batch extraction, model training, warnings, and errors.

Start Prometheus:

```bash
docker run \
  --name prometheus \
  -p 9090:9090 \
  -v "$(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml" \
  prom/prometheus
```

Start Grafana:

```bash
docker run \
  -d \
  -p 3000:3000 \
  grafana/grafana
```

## Project Structure

```text
customer-churn-etl-mlops/
├── .github/
│   └── workflows/
│       └── ci.yml
├── airflow/
│   └── dags/
│       └── etl_mlops_dag.py
├── data/
│   └── raw/
│       └── customer_data.csv
├── processed/
└── tableau/
    └── customer_churn_predictions.csv
├── src/
│   ├── extract.py
│   ├── transform.py
│   ├── load.py
│   ├── train_model.py
│   └── tune_model.py
│   └── week8_predictive_model.py
├── app.py
├── Dockerfile
├── prometheus.yml
├── requirements.txt
└── README.md
```

Generated runtime artifacts may include:

```text
data/processed/
models/
logs/
mlruns/
mlflow.db
```

## Running the Project

Run the ETL stages:

```bash
python src/extract.py
python src/transform.py
python src/load.py
```

Train the baseline model:

```bash
python src/train_model.py
```

Run hyperparameter tuning:

```bash
python src/tune_model.py
```

Run the Flask API:

```bash
python app.py
```

## Monitoring URLs

Flask API:

```text
http://localhost:5001
```

Prometheus:

```text
http://localhost:9090
```

Grafana:

```text
http://localhost:3000
```
## Week 8 Signature Assignment: ETL-Integrated Predictive Analytics and Tableau Dashboard

### Project Overview

This extension integrates customer churn predictive modeling with the existing ETL/MLOps architecture and an interactive Tableau Public dashboard. The project demonstrates data extraction, preprocessing, predictive modeling, workflow orchestration, and visualization of customer churn risk.

### Predictive Model Development

A Random Forest classifier was developed using Scikit-learn. Numerical variables were processed using median imputation and standard scaling, while categorical variables were processed using imputation and one-hot encoding.

The dataset contained 20 customer records. A stratified 70/30 split produced 14 training observations and six testing observations.

### Hyperparameter Optimization

GridSearchCV evaluated 18 Random Forest configurations using three-fold stratified cross-validation.

The optimal hyperparameters were:

- Number of trees: 50
- Maximum tree depth: 3
- Minimum samples per leaf: 1

The best cross-validation F1 score was 0.9333 (93.33%).

### Model Evaluation

The optimized model achieved the following results on six test observations:

| Metric | Result |
|---|---|
| Accuracy | 100% |
| Precision | 100% |
| Recall | 100% |
| F1 Score | 100% |
| ROC-AUC | 1.0000 |

The confusion matrix was `[[4, 0], [0, 2]]`.

These results demonstrate successful classification on the small test sample. They should not be interpreted as evidence of reliable performance on larger, unseen customer populations.

### Tableau Integration

The predictive modeling script exports customer-level predictions to:

`data/tableau/customer_churn_predictions.csv`

The exported dataset contains 20 customer records, 11 columns, zero missing values, and zero duplicate customer IDs.

The Tableau Public dashboard includes:

1. Churn Rate by Contract Type
2. Customer Risk Distribution
3. Average Churn Probability by Contract Type

An interactive Contract Type filter supports analysis of annual and monthly customers.

The dashboard is published under the title **Week 8 - Customer Churn Predictive Analytics Dashboard**.

### Airflow Automation

Apache Airflow orchestrates five sequential tasks:

`extract_data → transform_data → load_data → train_model → generate_tableau_predictions`

A local Airflow DAG test completed successfully, confirming end-to-end task execution and generation of the Tableau-compatible prediction file.

The DAG includes a daily schedule, although continuous scheduled operation has not been independently verified. Tableau Public refreshes are not automatically triggered by local CSV regeneration.

### Running the Week 8 Predictive Model

From the project root, execute:

```bash
./venv/bin/python src/week8_predictive_model.py
```

To test the complete Airflow workflow:

```bash
AIRFLOW__CORE__DAGS_FOLDER="$PWD/airflow/dags" ./venv/bin/airflow dags test customer_churn_etl_mlops_pipeline
```

### Optimization and Limitations

The workflow uses stratified cross-validation, parallel hyperparameter search, reusable preprocessing pipelines, and Apache Airflow orchestration.

The demonstration dataset contains only 20 records. Additional evaluation on larger datasets is necessary to assess generalization, runtime efficiency, and scalability.

The current implementation generates Tableau-ready CSV files locally; updating the published Tableau Public dashboard requires a separate data-source update and publication step.

## Future Improvements

- Deploy the Dockerized application to AWS or Azure.
- Replace SQLite with PostgreSQL for larger workloads.
- Use Kubernetes for container orchestration.
- Implement automated model retraining.
- Add production data-drift detection.
- Add model promotion/rollback policies.

## Author

Delois Giles
