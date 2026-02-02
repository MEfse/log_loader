# src/__init__.py

# Импортируем необходимые классы и функции

# 1. Логирование
from src.logger_config import logger

# 2. Классы и функции для работы с параметрами подключения и данными
from src.extract import LoadParams, LogDataLoader, LoaderCsvFile, LoaderModel, Run
from src.forecast import TrainModel
from src.transform import Preprocessing
from src.evaluation import Metrics
from src.utils import adfuller_test, plot_predict
from src.run_spark import spark_connect

# Публичный API пакета
__all__ = [
    # Логирование
    'logger',

    # Классы и функции для работы с параметрами подключения и данными
    'LoadParams', 'LogDataLoader', 'LoaderCsvFile', 'Run', 

    # Классы для работы с моделями
    'LoaderModel', 'TrainModel',

    # Функции для обработки данных и построения прогноза
    'Preprocessing', 'adfuller_test', 'plot_predict', 'Metrics',

    # Функции для обучения моделей
    'load_or_train', 'train_model', 'update_model',

    # Функции для работы с Spark
    'spark_connect',
]
