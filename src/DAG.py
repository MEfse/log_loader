from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

from src.config.path_config import path_model, path_predict, path_logs, path_connect, path_logging
from src.loader.loader_csv_file import load_to_connect, load_data_from_db, load_recent_logs, load_predict, load_model, save_model
from src.feature_engineering import preprocess_data, split_data, get_time_series
from src.model_training import load_or_train, train_model, update_model
from src.evaluation import evaluate, get_plot_predict
from src.utils import adfuller_test, plot_predict

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'example_dag',
    default_args=default_args,
    description='DAG с функциями Python',
    schedule_interval=timedelta(hours=1),
    start_date=datetime(2025, 4, 6),
    catchup=False,
) as dag:

    task_load = PythonOperator(
        task_id='1',
        python_callable=load_data_from_db
    )

    task_preprocessing = PythonOperator(
        task_id='2',
        python_callable=preprocess_data
    )

    task_train = PythonOperator(
        task_id='3',
        python_callable=load_or_train
    )

    task_evaluate = PythonOperator(
        task_id='4',
        python_callable=evaluate
    )

    task_load >> task_preprocessing >> task_train >> task_evaluate
