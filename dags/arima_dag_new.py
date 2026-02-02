from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator


import sys
import os
from pendulum import timezone, parse

# Добавляем путь до папки 'src' в PYTHONPATH
sys.path.insert(0, '/home/user/log_loader/src')

# Импортируем необходимые модули
from extract import Run             # type: ignore
from logger_config import logger    # type: ignore

def run_execute(start_ts: str, end_ts: str, **_):
    try:
        tz = timezone("Europe/Moscow")
        start_local = parse(start_ts).in_timezone(tz)
        end_local = parse(end_ts).in_timezone(tz)
        obj = Run()
        print(start_local, end_local)      
        obj.execute(start_ts=start_local, end_ts=end_local)        
    except Exception as e:
        logger.exception("Ошибка при выполнении execute: %s", e)
        raise

# Определение DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="arima_pipeline",
    default_args=default_args,
    description="Pipeline запуска ARIMA модели",
    schedule_interval="@hourly",   
    start_date=datetime(2025, 10, 1),
    catchup=False,
    tags=["arima", "ml"])as dag:

    run_task = PythonOperator(
        task_id="run_arima_execute",
        python_callable=run_execute,
        op_kwargs={
            # Передаём границы окна как ISO-строки (UTC)
            "start_ts": "{{ data_interval_start.isoformat() }}",
            "end_ts":   "{{ data_interval_end.isoformat() }}",
        }
    )

    run_task