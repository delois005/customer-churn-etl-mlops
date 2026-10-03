# Customer Churn Prediction MLOps Pipeline

## Overview

This project demonstrates the design and implementation of an end-to-end ETL and MLOps pipeline for customer churn prediction. The solution extracts customer data, performs preprocessing and feature engineering, trains a machine learning model, tracks experiments using MLflow, deploys the model through a Flask REST API, automates workflows with GitHub Actions, and monitors production performance using Prometheus and Grafana.

---

## Technologies Used

- Python
- Pandas
- Dask
- SQLAlchemy
- Scikit-learn
- Flask
- MLflow
- Docker
- Prometheus
- Grafana
- GitHub Actions
- Apache Airflow
  
## Workflow Orchestration

Apache Airflow is used to coordinate the ETL and machine learning workflow.

The Airflow DAG controls the execution order of:

1. Data extraction
2. Data transformation
3. Data loading
4. Model training

This creates a repeatable workflow and reduces the need to run individual pipeline stages manually.

---

## ETL Pipeline

### Extract

The extraction phase retrieves data from multiple sources, including:

- CSV files containing structured customer data
- Relational database records using SQLAlchemy
- Web API data using Python Requests

The API response also includes free-text fields such as `title` and `body`, providing an example of semi-structured data containing unstructured textual content.

### Transform

The pipeline performs:

- Missing value handling
- Feature normalization
- Feature engineering
- Data validation
- Parallel processing using Dask

### Load

The processed dataset is stored for machine learning model training.

---

## Machine Learning

A customer churn prediction model was developed using Scikit-learn.

Model outputs include:

- Predicted class
- Prediction probability
- Model version

Experiments are tracked with MLflow.

---

## REST API

The trained model is deployed using Flask.

Endpoints:

- `/health`
- `/predict`
- `/metrics`

---

## Monitoring

Prometheus collects application metrics.

Grafana dashboards display:

- API Status
- API Request Rate
- Average API Response Time

---

## CI/CD

Docker image build and deployment testing

- Code validation
- Dependency installation
- Pipeline execution
- Model deployment
- Docker image build and deployment testing

---
---

## Project Structure

```
customer-churn-mlops/
│
├── app.py
├── train_model.py
├── etl_pipeline.py
├── requirements.txt
├── Dockerfile
├── prometheus.yml
├── README.md
│
├── airflow/
│   └── dags/
│       └── etl_mlops_dag.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── models/
└── data/
```
---

## Future Improvements

- Deploy the application to AWS or Azure.
- Replace SQLite with PostgreSQL.
- Use Kubernetes for container orchestration.
- Implement automated model retraining.
- Add data drift detection for production monitoring.

## Running the Project

Install dependencies

```bash
pip install -r requirements.txt
```

Train the model

```bash
python train_model.py
```

Run the API

```bash
python app.py
```

Start Prometheus

```bash
docker run \
--name prometheus \
-p 9090:9090 \
-v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
prom/prometheus
```

Start Grafana

```bash
docker run \
-d \
-p 3000:3000 \
grafana/grafana
```

---

## Monitoring URLs

Flask API

```
http://localhost:5001
```

Prometheus

```
http://localhost:9090
```

Grafana

```
http://localhost:3000
```

---

## Author

Delois Giles
