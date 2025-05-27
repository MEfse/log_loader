import pandas as pd
import os
from warnings import filterwarnings
from typing import Optional, Dict, List

from src.config.logger_config import logger


def load_connect_csv(path) -> Optional[Dict[str, str]]:
    '''Загружает параметры для подключения к PostgreSQL из файла connect.csv

    Args: 
        path (str): Путь до файла

    Returns:
        (Dict[str, str]): Возвращает словарь с ключами: dbname, user, password, host, port
    '''

    # Загружаем данные для файла connect.csv
    try:
        data = pd.read_csv(path)
        logger.info(f'load_connect_csv - Параметры подключения загружены.')
        return data.to_dict(orient='records')[0]

    # Обработка исключений
    except FileNotFoundError:
        logger.error(
            f'load_connect_csv - Не удалось загрузить параметры подключения. Файл не найден.')
        return None
    except pd.errors.EmptyDataError:
        logger.error(
            f'load_connect_csv - Не удалось загрузить параметры подключения. Файл пуст.')
        return None
    except Exception as e:
        logger.error(
            f'load_connect_csv - Ошибка при загрузке параметров подключения {e}')
        return None


def save_connect_csv(data, path, overwrite=False):
    '''Сохраняет параметры для подключения к PostgreSQL

    Args: 
        data (DataFrame): Передаем данные для в записи в файл
        path_connect (str): Передаем путь для подключения к файлу
        overwrite (bool): Требуется ли перезапись файла

    Returns:

    '''
    columns = ['dbname', 'user', 'password', 'host', 'port']

    # Преобразуем данные в DataFrame, если они еще не в этом формате
    if isinstance(data, (list, pd.DataFrame)):
        data = pd.DataFrame(data).T
        data.columns = columns
        logger.info(f'save_connect_csv - Данные преобразованы в DataFrame.')
    elif isinstance(data, dict):
        data = pd.DataFrame([data])
        data.columns = columns
        logger.info(f'save_connect_csv - Данные преобразованы в DataFrame.')

    # Проверка на существование файла
    if os.path.exists(path) and not overwrite:
        logger.info(
            f'save_connect_csv - Файл {path} уже существует. Установите overwrite=True для перезаписи.')
        return

    # Сохраняем файл
    try:
        data.to_csv(path, index=False, columns=columns)
        logger.info(f'save_connect_csv - Файл {path} сохранен.')
    except Exception as e:
        logger.error(
            f'save_connect_csv - Ошибка при сохранении файла {path}: {e}')


def load_csv_file(path: str, parse_dates: Optional[List[str]] = None,
                  expected_columns: Optional[List[str]] = None,
                  file_desc: str = "файл") -> pd.DataFrame:
    '''
    Универсальная функция загрузки CSV-файлов с логированием и обработкой ошибок.

    Args:
        path (str): Путь к CSV файлу
        parse_dates (list[str], optional): Какие колонки парсить как даты
        expected_columns (list[str], optional): Если указано — создаётся пустой DataFrame с этими колонками в случае ошибки
        file_desc (str): Название файла в логах для читаемости

    Returns:
        (DataFrame): Загруженный DataFrame или пустой DataFrame с нужными колонками
    '''
    try:
        data = pd.read_csv(path, parse_dates=parse_dates)
        logger.info(
            f'load_csv_file - {file_desc.capitalize()} {path} загружен.')
        return data
    except FileNotFoundError:
        logger.warning(
            f'load_csv_file - {file_desc.capitalize()} {path} не найден.')
    except pd.errors.EmptyDataError:
        logger.warning(
            f'load_csv_file - {file_desc.capitalize()} {path} пуст.')
    except Exception as e:
        logger.error(f'load_csv_file - Ошибка при загрузке {file_desc}: {e}')

    # Возврат пустого фрейма с нужными колонками (если заданы)
    if expected_columns:
        return pd.DataFrame(columns=expected_columns)
    return pd.DataFrame()


def save_csv_file(data: pd.DataFrame,
                  path: str,
                  append: bool = True,
                  log_success=True,
                  file_desc: str = "файл") -> None:
    '''
    Универсальная функция сохранения DataFrame в CSV, с добавлением новых данных.

    Args:
        data (DataFrame): Данные для сохранения
        path (str): Путь к файлу
        append (bool): Добавлять в файл. Если False — перезаписать файл.

        file_desc (str): Название файла для логирования

    Returns:

    '''
    # Обработка исключения, если data не DataFrame
    if not isinstance(data, pd.DataFrame):
        logger.error(
            f"save_csv_file - {file_desc.capitalize()} не является DataFrame.")
        return

    # Сохраняем индекс, data.index не временной
    save_index = not isinstance(data.index, pd.RangeIndex)

    try:
        # Добавляем в существующий файл
        mode = 'a' if append else 'w'
        header = not append or not os.path.exists(path)
        data.to_csv(path, mode=mode, header=header, index=save_index)

        if log_success:
            action = 'добавлен' if append else 'сохранен'
            logger.info(f"save_csv_file - {file_desc} {action} в {path}.")

    except Exception as e:
        logger.error(
            f"save_csv_file - Ошибка при сохранении {file_desc} {path}: {e}")


def load_recent_logs(path):
    '''
    Функция загрузки файла recent_logs.csv

    Args:
        path (str): Путь к CSV файлу recent_logs.csv

    Returns:
        DataFrame: Загруженный DataFrame или пустой DataFrame с нужными колонками
    '''
    return load_csv_file(path,
                         parse_dates=['timestamp'],
                         expected_columns=['timestamp',
                                           'log_level', 'log_type', 'message'],
                         file_desc="файл логов")


def load_predict(path):
    '''
    Функция загрузки файла predict_arima.csv

    Args:
        path (str): Путь к CSV файлу

    Returns:
        DataFrame: Загруженный DataFrame или пустой DataFrame с нужными колонками
    '''
    return load_csv_file(path,
                         parse_dates=['timestamp'],
                         expected_columns=['timestamp', 'prediction'],
                         file_desc="файл предсказаний")


def load_time_series(path):
    '''
    Функция загрузки файла time_series.csv

    Args:
        path (str): Путь к CSV файлу

    Returns:
        DataFrame: Загруженный DataFrame или пустой DataFrame с нужными колонками
    '''
    return load_csv_file(path,
                         parse_dates=['timestamp'],
                         expected_columns=['timestamp', 'log_level'],
                         file_desc="временной ряд")
