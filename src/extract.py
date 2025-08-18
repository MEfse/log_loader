import os
import pandas as pd
import psycopg2
import csv
from tqdm import tqdm
from warnings import filterwarnings
from typing import Optional, Dict
from datetime import datetime

from src.config.path_config import path_logs, path_inconnect, path_exconnect
from src.loader.loader_csv_file import load_recent_logs, save_csv_file, load_connect_csv
from src.config.logger_config import logger
from src.feature_engineering import get_last_time


class LogDataLoader:
    def __init__(self):
        self.full_load = False                # Загрузка всех данных
        self.path_logs = path_logs            # Путь до файла recent_logs.csv
        self.path_inconn = path_inconnect     # Путь до файла internal_connection.csv
        self.path_exconn = path_exconnect     # Путь до файла external_connection.csv
        self.batch_size = 500000              # Размер-батча


    def get_connect(self) -> Dict[str, str] | None:
        '''
        Загрузка параметров подключения (внутреннего или внешнего).
        Пытается загрузить внутреннее подключение, при ошибке - внешнее.

        Args: 

        Returns:
            Dict[str, str] | None: Параметры подключения к БД или None, если не удалось.
        '''

        try:
            DB_PARAMS = load_connect_csv(self.path_inconn)
            if DB_PARAMS is not None:
                logger.info('1/5 | connect - Внутреннее подключение.')
            else:
                return DB_PARAMS if DB_PARAMS is not None else None
        except:
            DB_PARAMS = load_connect_csv(self.path_exconn)
            logger.info('1/5 | connect - Внешнее подключение.')
            return DB_PARAMS if DB_PARAMS is not None else None

    def load_logs(self):
        '''
        Загрузка параметров подключения

        Args: 
            path_inconn (str): Путь до файла internal_connection.csv
            path_exconn (str): Путь до файла external_connection.csv

        Returns:
            None
        '''
        if os.path.exists(self.path_logs):
            recent_logs = load_recent_logs(self.path_logs)
        else:
            logger.warning(
                f"load_data_from_db - Файл логов {path_logs} не найден. Загрузка всех данных.")
            self.full_load = True
            columns = ['timestamp', 'log_level', 'log_type', 'message']
            recent_logs = pd.DataFrame(columns=columns)
            save_csv_file(recent_logs, self.path_logs)

    def get_interval(self, data: pd.DataFrame) -> Optional[str]:
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

    def get_data(self):
        DB_PARAMS = self.get_connect()
        query = self.build_query()
        try:
            with psycopg2.connect(**DB_PARAMS) as conn:
                with conn.cursor(name='batched_cursor') as cursor:
                    cursor.execute(query)

                    # Инициализация
                    first_rows = cursor.fetchmany(self.batch_size)
                    if not first_rows:
                        logger.warning(
                            f'load_data_from_db - Запрос вернул 0 строк.')
                        return None

                    columns = [desc[0] for desc in cursor.description]
                    chunk_df = pd.DataFrame(first_rows, columns=columns)
                    total_rows = len(chunk_df)
                    save_csv_file(chunk_df, path_logs,
                                  append=True, log_success=False)

                    pbar = tqdm(desc="Загрузка данных из БД", unit="строк",
                                initial=total_rows, bar_format="{desc}: {elapsed} | {n:,} строк")

                    while True:
                        rows = cursor.fetchmany(self.batch_size)
                        if not rows:
                            break
                        chunk_df = pd.DataFrame(rows, columns=columns)
                        save_csv_file(chunk_df, path_logs,
                                      append=True, log_success=False)
                        total_rows += len(chunk_df)
                        pbar.update(len(chunk_df))

                    pbar.close()

            logger.info(
                f'load_data_from_db - Всего загружено {total_rows} строк.')

            return None

        except Exception as e:
            logger.error(
                f"load_data_from_db - Ошибка при соединении с БД: {e}")
            return None

    def merge_and_sort_logs(self,
                            old_logs: pd.DataFrame,
                            new_logs: pd.DataFrame) -> pd.DataFrame:
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

        merged['timestamp'] = pd.to_datetime(
            merged['timestamp'], errors='coerce')
        return merged.sort_values(by='timestamp').reset_index(drop=True)

    def build_query(self,
                    full_load: bool,
                    interval_str: str = None) -> str:
        '''
        Args: 
            full_load (bool): Значение при котором загружать все данные или часть
            interval_str (str): Интеврал времени по которому производить загрузку данных
        Returns:
            (str): Запрос для PostgreSQL
        '''
        if full_load:
            logger.info("build_query - Запрос всех данных из БД.")
            return "SELECT timestamp, log_level, log_type, message FROM logs;"
        else:
            logger.info(
                f"build_query - Запрос данных за интервал: {interval_str}")
            return f"""SELECT timestamp, log_level, log_type, message 
                    FROM logs WHERE timestamp >= NOW() - INTERVAL '{interval_str}'"""
