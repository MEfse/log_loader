import pandas as pd
import os
from warnings import filterwarnings
from typing import Optional, Dict, List

from src.config.logger_config import logger
from src.config.path_config import path_connect

filterwarnings("ignore", category=UserWarning,
               message='.*pandas only supports SQLAlchemy connectable.*')


def load_connect_csv(path: str = path_connect) -> Optional[Dict[str, str]]:
    '''Загружает параметры для подключения к PostgreSQL из файла connect.csv

    Args: 
        path (str): Путь до файла connect.csv (по умолчанию)

    Returns:
        Возвращает словарь с ключами: dbname, user, password, host, port
    '''

    # Загружаем данные для файла connect.csv
    try:
        data = pd.read_csv(path)
        logger.info(f'Файл {path} загружен.')
        return data.to_dict(orient='records')[0]

    # Обработка исключений
    except FileNotFoundError:
        logger.error(f'Файл {path} не найден.')
        return None
    except pd.errors.EmptyDataError:
        logger.error(f'Файл {path} пуст.')
        return None
    except Exception as e:
        logger.error(f'Ошибка при загрузке параметров подключения {e}')


def save_connect_csv(data, path: str = path_connect, overwrite=False):
    '''Сохраняет параметры для подключения к PostgreSQL

    Args: 
        data (DataFrame): Передаем данные для в записи в connect.csv
        path_connect (str): Передаем путь для подключения к connect.csv
        overwrite (bool): Требуется ли перезапись файла connect.csv

    Returns:

    '''
    columns = ['dbname', 'user', 'password', 'host', 'port']

    # Преобразуем данные в DataFrame, если они еще не в этом формате
    if isinstance(data, (list, pd.DataFrame)):
        data = pd.DataFrame(data).T
        data.columns = columns
        logger.info(f'Данные преобразованы в DataFrame.')
    elif isinstance(data, dict):
        data = pd.DataFrame([data])
        data.columns = columns
        logger.info(f'Данные преобразованы в DataFrame.')

    # Проверка на существование файла
    if os.path.exists(path) and not overwrite:
        logger.info(
            f'Файл {path} уже существует. Установите overwrite=True для перезаписи.')
        return

    # Сохраняем файл
    try:
        data.to_csv(path, index=False, columns=columns)
        logger.info(f'Файл {path} сохранен.')
    except Exception as e:
        logger.error(f'Ошибка при сохранении файла {path}: {e}')


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
        DataFrame: Загруженный DataFrame или пустой DataFrame с нужными колонками
    '''
    try:
        data = pd.read_csv(path, parse_dates=parse_dates)
        logger.info(f'{file_desc.capitalize()} {path} загружен.')
        return data
    except FileNotFoundError:
        logger.warning(f'{file_desc.capitalize()} {path} не найден.')
    except pd.errors.EmptyDataError:
        logger.warning(f'{file_desc.capitalize()} {path} пуст.')
    except Exception as e:
        logger.error(f'Ошибка при загрузке {file_desc}: {e}')

    # Возврат пустого фрейма с нужными колонками (если заданы)
    if expected_columns:
        return pd.DataFrame(columns=expected_columns)
    return pd.DataFrame()


def save_csv_file(data: pd.DataFrame,
                  path: str,
                  append: bool = True,
                  file_desc: str = "файл") -> None:
    '''
    Универсальная функция сохранения DataFrame в CSV, с добавлением новых данных.

    Args:
        data (DataFrame): Данные для сохранения
        path (str): Путь к файлу
        append (bool): Добавлять в файл (по умолчанию True). Если False — перезаписать файл.
        file_desc (str): Название файла для логирования

    Returns:
        None
    '''
    if not isinstance(data, pd.DataFrame):
        logger.error(f"{file_desc.capitalize()} не является DataFrame.")
        return

    save_index = not isinstance(data.index, pd.RangeIndex)

    try:
        # Добавляем в существующий файл
        if append and os.path.exists(path):
            data.to_csv(path, mode='a', header=False, index=save_index)
            logger.info(f"{file_desc.capitalize()} добавлен в {path}.")
        else:
            # Перезапись или создание файла
            data.to_csv(path, index=save_index)
            logger.info(
                f"{file_desc.capitalize()} сохранён в {path} (перезапись).")
    except Exception as e:
        logger.error(f"Ошибка при сохранении {file_desc} {path}: {e}")


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
