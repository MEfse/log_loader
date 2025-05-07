# src/loader/__init__.py

# Импортируем функции из модулей загрузки данных
from src.loader.loader_csv_file import load_connect_csv, load_csv_file, load_recent_logs, load_predict, load_time_series, save_csv_file, save_connect_csv
from src.loader.connect_to_db import load_data_from_db, get_interval, build_query, merge_and_sort_logs
from src.loader.loader_model import load_model, save_model

# Публичный API для загрузчика
__all__ = [
    'load_csv_file', 'load_recent_logs', 'load_predict', 'load_time_series', 'save_csv_file',
    'load_connect_csv', 'load_data_from_db', 'save_connect_csv', 'get_interval',
    'build_query', 'merge_and_sort_logs',
    'load_model', 'save_model'
]
