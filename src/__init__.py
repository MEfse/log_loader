# src/__init__.py

# Импортируем модули конфигурации
from src.config.path_config import path_model, path_predict, path_logs, path_inconnect, path_exconnect, \
    path_logging, path_time_series, path_tmp
from src.config.logger_config import logger

# Импортируем загрузчик
from src.loader.loader_csv_file import load_connect_csv, save_connect_csv, load_csv_file, \
    load_recent_logs, load_predict, load_time_series, save_csv_file
from src.loader.connect_to_db import load_data_from_db, get_interval, build_query, merge_and_sort_logs
from src.loader.loader_model import load_model, save_model

# Импортируем модуль для обучения моделей
from src.model_training import load_or_train, train_model, update_model

# Импортируем функции для обработки данных и построения прогноза
from src.feature_engineering import preprocess_arima
from src.utils import adfuller_test, plot_predict

from src.run_spark import spark_connect

# Объединённый публичный API
__all__ = [
    'path_model', 'path_predict', 'path_logs', 'path_inconnect', 'path_exconnect', 'path_logging', 'path_time_series', 'path_tmp',
    'logger',
    'load_csv_file', 'load_recent_logs', 'load_predict', 'load_time_series', 'save_csv_file',
    'load_connect_csv', 'load_data_from_db', 'save_connect_csv', 'load_existing_logs', 'get_interval',
    'build_query', 'merge_and_sort_logs',
    'load_model', 'save_model',
    'load_or_train', 'train_model', 'update_model',
    'preprocess_arima', 'adfuller_test', 'plot_predict',
    'spark_connect'
]
