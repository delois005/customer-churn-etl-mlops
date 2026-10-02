from pathlib import Path
from datetime import datetime

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = PROJECT_ROOT / "venv/bin/python"

with DAG(
    dag_id="customer_churn_etl_mlops_pipeline",
    description="Automated ETL and ML training pipeline for customer churn",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["etl", "mlops", "customer-churn"],
) as dag:

    extract = BashOperator(
        task_id="extract_data",
        bash_command=f'cd "{PROJECT_ROOT}" && "{PYTHON}" src/extract.py',
    )

    transform = BashOperator(
        task_id="transform_data",
        bash_command=f'cd "{PROJECT_ROOT}" && "{PYTHON}" src/transform.py',
    )

    load = BashOperator(
        task_id="load_data",
        bash_command=f'cd "{PROJECT_ROOT}" && "{PYTHON}" src/load.py',
    )

    train = BashOperator(
        task_id="train_model",
        bash_command=f'cd "{PROJECT_ROOT}" && "{PYTHON}" src/train_model.py',
    )

    extract >> transform >> load >> train
