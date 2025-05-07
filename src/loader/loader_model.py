import joblib
from typing import Any

from src.config.logger_config import logger


def load_model(path_model: str) -> Any:
    '''Загружает модель ARIMA

    Args: 
        path_model (str): Передаем путь для подключения к файлу модели models.hd5

    Returns:
        predict (DataFrame): Загружаем модель ARIMA
    '''
    try:
        # Загружаем обученную модель
        return joblib.load(path_model)
    except FileNotFoundError:
        logger.warning("Файл модели не найден.")
        raise


def save_model(model_with_order: Any, path_model: str) -> Any:
    '''Функция для сохранения модели

    Args:
        model (hd5): Модель 
        path_model (str): Передаем путь для подключения к файлу модели models.hd5

    Returns:
        model (hd5): Сохраняем модель 
    '''
    if model_with_order:
        # Сохраняем модель
        joblib.dump(model_with_order, path_model)
        logger.info("Файл модели сохранен.")
        return model_with_order
    else:
        logger.error("Модель не удалось сохранить.")
        return None
