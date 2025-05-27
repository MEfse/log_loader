import os
import pandas as pd
import psycopg2
import csv
from tqdm import tqdm
from warnings import filterwarnings
from typing import Optional, Dict
from datetime import datetime

from src.config.path_config import path_logs, path_tmp, path_inconnect, path_exconnect
from src.loader.loader_csv_file import load_recent_logs, save_csv_file, load_connect_csv
from src.config.logger_config import logger
from src.feature_engineering import get_last_time

filterwarnings("ignore", category=UserWarning,
               message='.*pandas only supports SQLAlchemy connectable.*')


def load_data_from_db(full_load: bool = False,
                      path_logs: str = path_logs,
                      batch_size: int = 500000) -> pd.DataFrame:
    '''
    Загружает данные из PostgreSQL

    Args: 
        full_load (bool): Загружать все данные или только недостающие
        path (str): Путь до файла recent_logs.csv
        batch_size (int): Размер пакета загрузки

    Returns:
        data (DataFrame): Возвращает данные полученные из базы данных
    '''
    logger.info("load_data_from_db - Старт load_data_from_db.")

    # Загрузка параметров подключения
    DB_PARAMS = connect()

    # Загрузка recent_logs.csv
    if os.path.exists(path_logs):
        recent_logs = load_recent_logs(path_logs)
    else:
        logger.warning(
            f"load_data_from_db - Файл логов {path_logs} не найден. Загрузка всех данных.")
        full_load = True
        columns = ['timestamp', 'log_level', 'log_type', 'message']
        recent_logs = pd.DataFrame(columns=columns)
        save_csv_file(recent_logs, path_logs)

    interval_str = None

    # Получение временного интервала
    if not full_load:
        interval_str = get_interval(recent_logs)
        if interval_str is None:
            return pd.DataFrame()

    # Получение запроса
    debug = False
    query = build_query(full_load, interval_str, debug=debug)

    try:
        with psycopg2.connect(**DB_PARAMS) as conn:
            with conn.cursor(name='batched_cursor') as cursor:
                cursor.execute(query)

                # Инициализация
                first_rows = cursor.fetchmany(batch_size)
                if not first_rows:
                    logger.warning(
                        f'load_data_from_db - Запрос вернул 0 строк.')
                    return recent_logs

                columns = [desc[0] for desc in cursor.description]
                chunk_df = pd.DataFrame(first_rows, columns=columns)
                total_rows = len(chunk_df)
                save_csv_file(chunk_df, path_logs,
                              append=True, log_success=False)

                pbar = tqdm(desc="Загрузка данных из БД", unit="строк",
                            initial=total_rows, bar_format="{desc}: {elapsed} | {n:,} строк")

                while True:
                    rows = cursor.fetchmany(batch_size)
                    if not rows:
                        break
                    chunk_df = pd.DataFrame(rows, columns=columns)
                    save_csv_file(chunk_df, path_logs,
                                  append=True, log_success=False)
                    total_rows += len(chunk_df)
                    pbar.update(len(chunk_df))

                pbar.close()

        logger.info(f'load_data_from_db - Всего загружено {total_rows} строк.')

        return load_recent_logs(path_logs) if total_rows > 0 else recent_logs

    except Exception as e:
        logger.error(f"load_data_from_db - Ошибка при соединении с БД: {e}")
        return pd.DataFrame()


def get_interval(data: pd.DataFrame) -> Optional[str]:
    '''
    Получение интервала времени для запроса SQL 

    Args: 
        data (pd.DataFrame): Данные по которым считается интервал времени

    Returns:
        interval (str): Интервал недостающих данных
    '''
    # Получаем последнее время
    last_time = get_last_time(data)

    # Если последнее время пустое
    if last_time is None:
        logger.error("get_interval - Не удалось получить последнее время.")
        return None
    try:
        time_diff = datetime.now() - last_time
        interval = f"{max(1, int(time_diff.total_seconds() / 60))} minutes"
        logger.info(f"get_interval - Интервал загрузки: {interval}")
        return interval
    except Exception as e:
        logger.error(f"get_interval - Ошибка при расчёте интервала: {e}")
        return None


def build_query(full_load: bool, interval_str: str = None, debug=False) -> str:
    '''

    Args: 
        full_load (bool): Значение при котором загружать все данные или часть
        interval_str (str): Интеврал времени по которому производить загрузку данных
        debug (bool): Загружаем только часть данных для debug
    Returns:
        (str): Запрос для PostgreSQL
    '''
    if debug:
        logger.info("build_query - Debug запрос")
        return "SELECT timestamp, log_level, log_type, message FROM logs LIMIT 250000;"
    if full_load:
        logger.info("build_query - Запрос всех данных из БД.")
        return "SELECT timestamp, log_level, log_type, message FROM logs;"
    else:
        logger.info(f"build_query - Запрос данных за интервал: {interval_str}")
        return f"""SELECT timestamp, log_level, log_type, message 
                   FROM logs WHERE timestamp >= NOW() - INTERVAL '{interval_str}'"""


def connect(path_inconn: str = path_inconnect,
            path_exconn: str = path_exconnect) -> pd.DataFrame:
    '''
    Загрузка параметров подключения (внутренного или внешнего)

    Args: 
        path_inconn (str): Путь до файла internal_connection.csv
        path_exconn (str): Путь до файла external_connection.csv

    Returns:
        (pd.DataFrame): Пустой датафрейм, если подключение не удалось
        (Dict[str, str]): Параметры подключения
    '''

    try:
        DB_PARAMS = load_connect_csv(path_inconn)
        logger.info('connect - Внутреннее подключение.')
        if DB_PARAMS is None:
            return pd.DataFrame()
        else:
            return DB_PARAMS
    except:
        DB_PARAMS = load_connect_csv(path_exconn)
        logger.info('connect - Внешнее подключение.')
        if DB_PARAMS is None:
            return pd.DataFrame()
        else:
            return DB_PARAMS


def merge_and_sort_logs(old_logs: pd.DataFrame, new_logs: pd.DataFrame) -> pd.DataFrame:
    '''
    Функция для объединения старых и новых данных с сортировкой по времени

    Args: 
        old_logs (pd.DataFrame): Датафрейм со старыми данными
        new_logs (pd.DataFrame): Датафрейм с новыми данными
    Returns:
        (pd.DataFrame): Объединенный и отсортированный датафрейм
    '''
    if new_logs is None:
        merged = old_logs.copy()
    else:
        merged = pd.concat([old_logs, new_logs]).drop_duplicates(
            subset=['timestamp'])

    merged['timestamp'] = pd.to_datetime(merged['timestamp'], errors='coerce')
    return merged.sort_values(by='timestamp').reset_index(drop=True)
