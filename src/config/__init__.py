# src/config/__init__.py

# Импорт конфигурации путей и логгера
from src.config.path_config import path_model, path_predict, path_logs, path_connect, path_logging, path_time_series
from src.config.logger_config import logger

# Публичный API для конфигурации
__all__ = [
    'path_model', 'path_predict', 'path_logs', 'path_connect', 'path_logging', 'path_time_series',
    'logger'
]
