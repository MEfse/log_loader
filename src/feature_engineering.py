import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from typing import Optional

from src.config.path_config import path_time_series
from src.loader.loader_csv_file import save_csv_file
from src.config.logger_config import logger


def preprocess_arima(data: pd.DataFrame, path: str = path_time_series):
    '''Преобразует данные в временной ряд для ARIMA

    Args: 
        data (DataFrame): Данные полученные с базы данных 

    Returns:
        time_series (DataFrame): Возвращает временной ряд
    '''
    logger.info("Старт preprocess_arima.")

    # Проверка на пустые данные или если timestamp нет в данных
    if data.empty or 'timestamp' not in data.columns:
        logger.warning("Пустой датафрейм или отсутствует колонка 'timestamp'.")
        return pd.DataFrame()

    # Делаем индекс timestamp
    data.set_index('timestamp', inplace=True)
    # Делаем ресемлинг
    time_series = data.resample('1H')['log_level'].size()
    logger.info("Ресемплинг логов.")

    # Удаляем последний час
    time_series = time_series[:-1]

    # Проверка и преобразование в DataFrame, если это Series
    if isinstance(time_series, pd.Series):
        time_series = time_series.to_frame()
        logger.info("Преобразовали Series в DataFrame.")

    # Проверка пути для сохранения
    if not os.path.exists(os.path.dirname(path)):
        logger.error(
            f"Путь {os.path.dirname(path)} не существует.")
        return pd.DataFrame()

    # Сохраняем временной ряд
    save_csv_file(time_series, path, file_desc='Временной ряд')
    return time_series


def get_last_time(data: pd.DataFrame) -> Optional[str]:
    '''Загружает последннюю дату из файла recent_logs

    Args: 
        path_logs (str): Передаем путь для подключения к файлу recent_logs
    Returns:
        last_time (datetime): Возвращаем последнее время
    '''
    # Убираем строки, где в timestamp стоит текст 'timestamp' (защита от ошибок)
    data = data[data['timestamp'] != 'timestamp']

    if data.empty:
        logger.error(f'Данных нет.')
        return None

    last_time = data['timestamp'].max()

    if pd.isna(last_time):
        logger.error(f'Не удалось извлечь дату.')
        return None

    last_time = pd.to_datetime(last_time, format='%Y-%m-%d %H:%M:%S')

    logger.info(f"Получаем последнее время: {last_time}")
    return last_time
