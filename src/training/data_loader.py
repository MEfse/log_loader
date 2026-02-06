import os
import pandas as pd
import psycopg2
import joblib
#from tqdm import tqdm
from datetime import datetime
from typing import Optional, List, Any
import logging

from src.common.config import settings, LoadParams
from src.common.preprocessing.pipeline import preprocess

logger = logging.getLogger(__name__)

#--------------Временный класс для запуска модели---------------------------------------------------------------------
class Run:
    def __init__(self):
        self.params = LoadParams()
        self.data_loader = DataLoader()
        self.loader_model = TrainModel()
        self.evaluation = Evaluator()
        self.DB_PARAMS = None
        self.data = None

    def execute(self, start_ts: str | None = None, end_ts: str | None = None):
        try:
            # Получаем параметры подключения
            self.DB_PARAMS = self.params.get_db_params()             
        except Exception as e:
            logger.exception("Шаг load_params провалился: %s", e)
            return

        try:
            # Загружаем и обрабатываем данные
            self.data = self.data_loader.get_data_from_db(self.DB_PARAMS, start_ts=start_ts, end_ts=end_ts)
            self.data = self.data_loader.preprocess_data(self.data)  
        except Exception as e:
            logger.exception("Шаг get_data_db провалился: %s", e)
            return

        try:
            # Обучаем модель
            self.load_model(self.data)          
        except Exception as e:
            logger.exception("Шаг load_model провалился: %s", e)
            return    

        try:
            # Предсказание
            self.predict_arima()          
        except Exception as e:
            logger.exception("Шаг predict_arima провалился: %s", e)
            return  

    def load_params(self):
        self.DB_PARAMS = self.params.get_db_params()
        return self.DB_PARAMS

    def load_model(self, data):
        self.loader_model.load_or_train(data)

    def predict_arima(self):
        self.evaluation.predict()

#--------------Загрузка параметров---------------------------------------------------------------------
class DataLoader:
    def __init__(self):
        self.loader_data = LogDataLoader()
        self.loader_csv = LoaderCsvFile()
        self.preprocessing = Preprocessing()

    def get_data_from_db(self, DB_PARAMS, start_ts, end_ts):
        """
        Загружает данные из БД с учетом временных рамок
        """
        data_list = []
        count = 0
        for data_chunk in self.loader_data.get_data(DB_PARAMS, start_ts=start_ts, end_ts=end_ts):
            count += len(data_chunk)
            logger.info(f'Получено {count} данных.') 
            self.loader_csv.save_csv_file(data_chunk, self.params.PATH_LOGS)
            data_list.append(data_chunk)
        
        data = pd.concat(data_list)
        return data

    #def preprocess_data(self, data: pd.DataFrame):
        #"""
        #Обрабатываем загруженные данные
        #"""
        #logger.info("Применяем предварительную обработку данных.")
        # Используем общий препроцессинг из src/common
        #return self.preprocessing.clean_data(data)

#--------------Загрузка параметров---------------------------------------------------------------------
class LogDataLoader:
    def __init__(self):
        self.batch_size = 1000
        self.params = LoadParams()  # Инициализируем LoadParams, чтобы получить параметры
        self.loader = LoaderCsvFile()
        self.get_query = GetInterval()

    def get_data(self, start_ts: str | None = None, end_ts: str | None = None):
        # Используем параметры из LoadParams для подключения
        DB_PARAMS = self.params.get_db_params()  # Загружаем параметры подключения

        try:
            with psycopg2.connect(**DB_PARAMS) as conn:
                with conn.cursor(name='batched_cursor') as cursor:
                    if start_ts and end_ts and os.path.exists(self.params.PATH_LOGS):
                        sql = """
                            SELECT date_trunc('hour', timestamp) AS hour,
                                COUNT(*) AS log_count
                            FROM logs
                            WHERE timestamp >= %(start)s
                            AND timestamp < %(end)s
                            GROUP BY hour
                            ORDER BY hour;
                        """
                        cursor.execute(sql, {"start": start_ts, "end": end_ts})
                    else:
                        # первый запуск без окна — можно взять всё или ограничить ретеншном
                        sql = """
                            SELECT date_trunc('hour', timestamp) AS hour,
                                COUNT(*) AS log_count
                            FROM logs
                            WHERE timestamp < %(end)s
                            GROUP BY hour
                            ORDER BY hour;
                        """
                        cursor.execute(sql, {"end": end_ts})

                    while True:
                        rows = cursor.fetchmany(self.batch_size)
                        if not rows:
                            break
                        chunk_df = pd.DataFrame(rows, columns=['hour', 'log_count'])
                        yield chunk_df

        except Exception as e:
            logger.error(f"Ошибка при соединении с БД: {e}")
            raise

#--------------Загрузка CSV файлов--------------------------------------------------------------------- 

class LoaderCsvFile:
    #def __init__(self):
        #pass
    
    def load_csv_file(self, 
                    path: str, 
                    parse_dates: Optional[List[str]] = None,
                    expected_columns: Optional[List[str]] = None) -> pd.DataFrame:
        '''
        Универсальная функция загрузки CSV-файлов с логированием и обработкой ошибок.

        Args:
            path (str): Путь к CSV файлу
            parse_dates (list[str], optional): Какие колонки парсить как даты
            expected_columns (list[str], optional): Если указано — создаётся пустой DataFrame с этими колонками в случае ошибки

        Returns:
            (DataFrame): Загруженный DataFrame или пустой DataFrame с нужными колонками
        '''
        try:
            data = pd.read_csv(path, parse_dates=parse_dates)
            logger.info(
                f'Файл {path} загружен.')
            return data
        except FileNotFoundError:
            logger.warning(
                f'Файл {path} не найден.')
            raise
        except pd.errors.EmptyDataError:
            logger.warning(
                f'Файл {path} пуст.')
            return pd.DataFrame(columns=expected_columns)
        except Exception as e:
            logger.error(f'Ошибка при загрузке {path}: {e}')
            raise

    def save_csv_file(self,
                    data: pd.DataFrame,
                    path: str) -> None:
        '''
        Универсальная функция сохранения DataFrame в CSV, с добавлением новых данных.

        Args:
            data (DataFrame): Данные для сохранения
            path (str): Путь к файлу

        Returns:

        '''
        # Обработка исключения, если data не DataFrame
        if not isinstance(data, pd.DataFrame):
            logger.error(f"Файл {path} не является DataFrame.")
            raise

        # Сохраняем индекс, data.index не временной
        save_index = not isinstance(data.index, pd.RangeIndex)

        if os.path.exists(path):
            mode = 'a' 
            header = False
        else:
            mode = 'w'
            header = True

        try:
            data.to_csv(path, mode=mode, header=header, index=save_index)

        except Exception as e:
            logger.error(f"save_csv_file - Ошибка при сохранении файла {path}: {e}")
            raise


    def load_recent_logs(self, path):
        '''
        Функция загрузки файла recent_logs.csv

        Args:
            path (str): Путь к CSV файлу recent_logs.csv

        Returns:
            DataFrame: Загруженный DataFrame или пустой DataFrame с нужными колонками
        '''
        return self.load_csv_file(path,
                            parse_dates=['hour'],
                            expected_columns=['hour', 'log_count'])


    def load_predict(self, path):
        '''
        Функция загрузки файла predict_arima.csv

        Args:
            path (str): Путь к CSV файлу

        Returns:
            DataFrame: Загруженный DataFrame или пустой DataFrame с нужными колонками
        '''
        return self.load_csv_file(path,
                            parse_dates=['hour'],
                            expected_columns=['hour', 'prediction'])
    
    def load_evaluation(self, path):
        '''
        Функция загрузки файла evaluation_arima.csv

        Args:
            path (str): Путь к CSV файлу

        Returns:
            DataFrame: Загруженный DataFrame или пустой DataFrame с нужными колонками
        '''
        return self.load_csv_file(path,
                            parse_dates=['hour'],
                            expected_columns=['hour', 'mae', 'mse'])

#--------------Загрузка модели--------------------------------------------------------------------- 
class LoaderModel:
    def __init__(self):
        self.params = LoadParams()                          

    def load_model(self, path):
        '''Загружает модель ARIMA

        Args: 
            path_model (str): Передаем путь для подключения к файлу модели models.hd5

        Returns:
            predict (DataFrame): Загружаем модель ARIMA
        '''
        try:
            # Загружаем обученную модель
            return joblib.load(path)
        except FileNotFoundError:
            logger.warning("Файл модели не найден.")
            raise


    def save_model(self, model_with_order: Any, path: str) -> Any:
        '''Функция для сохранения модели

        Args:
            model (hd5): Модель 
            path_model (str): Передаем путь для подключения к файлу модели models.hd5

        Returns:
            model (hd5): Сохраняем модель 
        '''
        if model_with_order:
            joblib.dump(model_with_order, path)
            logger.info("Файл модели сохранен.")
            return model_with_order
        else:
            logger.error("Модель не удалось сохранить.")
            raise
