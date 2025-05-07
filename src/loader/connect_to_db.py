import os
import pandas as pd
import psycopg2
import csv
from tqdm import tqdm
from warnings import filterwarnings
from typing import Optional
from datetime import datetime

from src.config.path_config import path_logs, path_tmp
from src.loader.loader_csv_file import load_recent_logs, save_csv_file, load_connect_csv
from src.config.logger_config import logger
from src.feature_engineering import get_last_time

filterwarnings("ignore", category=UserWarning,
               message='.*pandas only supports SQLAlchemy connectable.*')


def load_data_from_db(full_load: bool = False,
                      path_logs: str = path_logs,
                      path_tmp: str = path_tmp,
                      batch_size: int = 100000) -> pd.DataFrame:
    '''Загружает данные из PostgreSQL

    Args: 
        full_load (bool): Загружать все данные или только за последний час
        path (str): Путь до файла recent_logs.csv (по умолчанию)
        batch_size (int): Размер пакета загрузки

    Returns:
        data (DataFrame): Возвращает данные полученные из базы данных если соединение успешно или None если не удалось
    '''
    logger.info("Старт load_data_from_db.")

    # Загружаем параметры подключения к базе данных
    DB_PARAMS = load_connect_csv()
    if DB_PARAMS is None:
        logger.error("Не удалось загрузить параметры подключения")
        return pd.DataFrame()

    # Проверяем существует ли файл по пути path, если нет, то full_load = True
    if not os.path.exists(path_logs):
        full_load = True

    # Загружаем исторические данные из логов recent_logs.csv
    recent_logs = load_recent_logs(path_logs)

    # Если full_load = False, то получаем последнее время из recent_logs.csv
    interval_str = None
    if not full_load:
        interval_str = get_interval(recent_logs)
        if interval_str is None:
            return pd.DataFrame()

    query = build_query(full_load, interval_str)

    # Подключение к базе данных
    try:
        with psycopg2.connect(**DB_PARAMS) as conn:
            with conn.cursor(name='batched_cursor') as cursor:
                cursor.execute(query)
                total_rows = write_cursor(
                    cursor, path_tmp, batch_size=batch_size)

        new_data = pd.read_csv(path_tmp, parse_dates=[
                               'timestamp'], low_memory=False)

        # Объединяем исторические данные с новыми
        concat_data = merge_and_sort_logs(recent_logs, new_data)
        # Рассчет кол-во добавленных данных
        new_rows_added = len(concat_data) - len(recent_logs)

        if new_rows_added > 0:
            save_csv_file(
                concat_data.iloc[-new_rows_added:], path_logs, append=True)
            logger.info(
                f"Добавлено {new_rows_added} строк. Всего: {len(concat_data)}")
        else:
            logger.info("Новых данных нет.")

        return concat_data if new_rows_added > 0 else recent_logs

    # Если соединение не удалось
    except Exception as e:
        logger.error(f"Ошибка при соединении с БД: {e}")
        return pd.DataFrame()


def write_cursor(cursor, output_path: str, batch_size=100000):
    '''
    Сохраняет результат курсора PostgreSQL в CSV-файл пакетами (batch-ами).

    Args:
        cursor: psycopg2 курсор с выполненным запросом
        output_path (str): Путь к временному файлу для записи
        batch_size (int): Размер одного батча

    Returns:
        int: Общее количество записей, сохранённых в файл
    '''
    colunms_names = [desc[0] for desc in cursor.description]
    total_rows = 0

    with open(output_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(colunms_names)

        with tqdm(total=1, desc="Загрузка данных из БД", bar_format="{desc}: {elapsed}") as pbar:
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break
                writer.writerows(rows)
                total_rows += 1
                pbar.update(1)

        return total_rows


def get_interval(data: pd.DataFrame) -> Optional[str]:
    '''Получение интервала времени для SQL 

    Args: 
        data (pd.DataFrame): Данные по которым считается интервал времени

    Returns:

    '''
    # Получаем последнее время
    last_time = get_last_time(data)

    # Если последнее время пустое
    if last_time is None:
        logger.error("Не удалось получить последнее время.")
        return None
    try:
        time_diff = datetime.now() - last_time
        interval = f"{max(1, int(time_diff.total_seconds() / 60))} minutes"
        logger.info(f"Интервал загрузки: {interval}")
        return interval
    except Exception as e:
        logger.error(f"Ошибка при расчёте интервала: {e}")
        return None


def build_query(full_load: bool, interval_str: str = None) -> str:
    if full_load:
        logger.info("Запрос всех данных из БД.")
        return "SELECT timestamp, log_level, log_type, message FROM logs;"
    else:
        logger.info(f"Запрос данных за интервал: {interval_str}")
        return f"""SELECT timestamp, log_level, log_type, message 
                   FROM logs WHERE timestamp >= NOW() - INTERVAL '{interval_str}'"""


def merge_and_sort_logs(old_logs: pd.DataFrame, new_logs: pd.DataFrame) -> pd.DataFrame:
    if new_logs is None:
        merged = old_logs.copy()
    else:
        merged = pd.concat([old_logs, new_logs]).drop_duplicates(
            subset=['timestamp'])

    merged['timestamp'] = pd.to_datetime(merged['timestamp'], errors='coerce')
    return merged.sort_values(by='timestamp').reset_index(drop=True)
