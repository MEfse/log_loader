from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

from src.loader.connect_to_db import load_data_from_db
# from src.feature_engineering import preprocess_arima

with DAG(
    dag_id='arima_dag',
    start_date=datetime(2025, 6, 18),
    schedule_interval=timedelta(hours=1),
    catchup=False,
    tags=['example'],
) as dag:

    load_data = PythonOperator(
        task_id='print_hello_task',
        python_callable=load_data_from_db,
    )
