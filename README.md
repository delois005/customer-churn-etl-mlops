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

---

## ETL Pipeline

### Extract

Data is extracted from structured customer datasets and prepared for processing.

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

GitHub Actions automates:

- Code validation
- Dependency installation
- Pipeline execution
- Model deployment

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
├── .github/
│   └── workflows/
│       └── ci.yml
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
