from src.config.path_config import path_model, path_predict, path_logs, path_connect, path_logging, path_time_series
from src.loader.connect_to_db import load_connect_csv, load_data_from_db, get_interval, merge_and_sort_logs
from src.loader.loader_csv_file import load_csv_file, load_recent_logs, save_connect_csv
from src.loader.loader_model import load_model, save_model
from src.feature_engineering import get_last_time, preprocess_arima
from src.model_training import load_or_train, train_model, update_model
from src.evaluation import evaluate, predict
from src.utils import adfuller_test, plot_predict, logger

import pandas as pd
import psycopg2

params = load_connect_csv()

try:
    with psycopg2.connect(**params) as conn:
        logger.info("Соединение установлено.")
except Exception as e:
    logger.error(f"Ошибка подключения: {e}")
